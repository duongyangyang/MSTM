"""
Main evaluation entry point — A-MEM-aligned pipeline.

Pipeline: Conversation → Memory System → chunk+embed → Retrieve Top-K
→ Foundation Model answers → Evaluate with F1, BLEU-1, ROUGE-L/2, METEOR, SBERT

Usage:
    # Run Static Memory baseline on LoCoMo with GPT-4o-mini
    python run_eval.py --method static_memory --benchmark locomo --out results/

    # Run all baselines on both benchmarks with specific foundation models
    python run_eval.py --method all --benchmark all --foundation-models gpt-4o-mini,gpt-4o

    # Run MSTM with a trained checkpoint
    python run_eval.py --method mstm --checkpoint checkpoints/mstm_final --benchmark locomo

    # Full-context baseline (raw dialogue, no memory system)
    python run_eval.py --method locomo_full --benchmark locomo

    # k-sensitivity analysis (GPT-4o-mini only)
    python run_eval.py --method static_memory --k-sweep 10,20,30,40,50

    # Internal transition quality eval (unchanged)
    python run_eval.py --mode transition --method static_memory --test_data data/processed/test.jsonl

IMPORTANT: Baseline 1 (Static Memory) must be run through this first as a
smoke test before anything else is evaluated. See TODOLIST.md Phase 3, Task 7.
"""

import argparse
import importlib
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.eval.metrics import (
    CostTracker,
    compute_average_ranking,
    compute_consistency_metrics,
    compute_consistency_metrics_llm,
    compute_full_qa_metrics,
    compute_llm_judge_scores,
    compute_memory_efficiency,
    compute_per_category_metrics,
    compute_qa_metrics,
    compute_retrieval_metrics,
    compute_temporal_metrics,
    compute_transition_metrics,
    compression_ratio,
    count_tokens,
    count_tokens_tiktoken,
)
from src.eval.loaders.locomo import (
    build_full_dialogue as build_full_dialogue_locomo,
    build_memory_from_conversation as build_memory_locomo,
    extract_qa_pairs as extract_locomo_qa,
    load_locomo,
)
from src.eval.loaders.longmemeval import (
    build_full_dialogue as build_full_dialogue_lme,
    build_memory_from_conversation as build_memory_lme,
    extract_qa_pairs as extract_lme_qa,
    load_longmemeval,
)
from src.eval.retriever import MemoryRetriever


# ---------------------------------------------------------------------------
# Foundation Model Registry (matching A-MEM Table 1)
# ---------------------------------------------------------------------------

# Each entry: (model_id, base_url, needs_large_ctx)
# - base_url = None → standard OpenAI
# - base_url = "http://localhost:11434/v1" → Ollama
# - needs_large_ctx = True → set num_ctx >= 16384 for Ollama models
FOUNDATION_MODELS = {
    "gpt-4o-mini": ("gpt-4o-mini", None, False),
    "gpt-4o": ("gpt-4o", None, False),
    "qwen2.5:1.5b": ("qwen2.5:1.5b", "http://localhost:11434/v1", True),
    "qwen2.5:3b": ("qwen2.5:3b", "http://localhost:11434/v1", True),
    "llama3.2:1b": ("llama3.2:1b", "http://localhost:11434/v1", True),
    "llama3.2:3b": ("llama3.2:3b", "http://localhost:11434/v1", True),
}


def get_reader_client(model_name: str, api_key: str = None, base_url: str = None):
    """
    Get an OpenAI-compatible client for a foundation model.

    GPT models → standard OpenAI API.
    Qwen/Llama → Ollama via OpenAI-compatible endpoint (localhost:11434/v1).

    Args:
        model_name: Model identifier (e.g., "gpt-4o-mini", "qwen2.5:1.5b").
        api_key: API key override.
        base_url: Base URL override.

    Returns:
        Tuple of (model_id, client, extra_kwargs) where extra_kwargs
        includes num_ctx for Ollama models.
    """
    from openai import OpenAI

    if model_name in FOUNDATION_MODELS:
        model_id, default_base_url, needs_large_ctx = FOUNDATION_MODELS[model_name]
    else:
        model_id, default_base_url, needs_large_ctx = model_name, None, False

    resolved_base_url = base_url or default_base_url
    resolved_api_key = api_key or os.environ.get("OPENAI_API_KEY", "ollama")

    client = OpenAI(
        api_key=resolved_api_key,
        base_url=resolved_base_url,
    )

    extra_kwargs = {}
    if needs_large_ctx:
        extra_kwargs["extra_body"] = {"options": {"num_ctx": 16384}}

    return model_id, client, extra_kwargs


# ---------------------------------------------------------------------------
# Method loading
# ---------------------------------------------------------------------------

METHOD_MAP = {
    "static_memory": "src.baselines.static_memory",
    "time_decay": "src.baselines.time_decay",
    "heuristic_consolidation": "src.baselines.heuristic_consolidation",
    "llm_based": "src.baselines.llm_based",
    "mstm": "src.mstm.inference",
    "locomo_full": None,  # Special: full conversation, no memory system
}


def load_method(method_name: str, checkpoint_path: Optional[str] = None):
    """
    Load a memory transition method by name.

    Args:
        method_name: One of: static_memory, time_decay, heuristic_consolidation,
                     llm_based, mstm.
        checkpoint_path: Path to trained checkpoint (required for mstm).

    Returns:
        A callable with signature transition(M, delta_M, **kwargs) -> M_prime.
    """
    if method_name == "locomo_full":
        # Special baseline: full conversation, no memory system
        return None, None  # transition_fn = None signals "use full dialogue"

    if method_name not in METHOD_MAP:
        raise ValueError(
            f"Unknown method: {method_name}. "
            f"Available: {list(METHOD_MAP.keys())}"
        )

    module_path = METHOD_MAP[method_name]
    module = importlib.import_module(module_path)

    if method_name == "mstm":
        if checkpoint_path is None:
            raise ValueError(
                "--checkpoint is required for mstm method. "
                "Provide path to trained model checkpoint."
            )
        # Return a wrapper that uses the loaded model
        from src.mstm.inference import MSTMInference

        infer = MSTMInference(checkpoint_path)
        return infer.transition, infer
    else:
        return module.transition, None


# ---------------------------------------------------------------------------
# Evaluation on a single benchmark
# ---------------------------------------------------------------------------


# A-MEM system prompt for answer generation (same prompt for all methods)
A_MEM_SYSTEM_PROMPT = (
    "You are a helpful assistant. Based on the conversation memory provided below, "
    "answer the question accurately and concisely. "
    "If the memory does not contain enough information to answer, say "
    "\"I don't have enough information to answer this question.\" "
    "Do not make up information not present in the memory."
)


def _generate_answer(
    question: str,
    context: str,
    model_id: str = "gpt-4o-mini",
    client=None,
    extra_kwargs: dict = None,
) -> str:
    """
    Generate an answer to a question given retrieved memory context.

    Uses the A-MEM system prompt and GPT-4o-mini as the reader model.

    Args:
        question: The QA question.
        context: Retrieved memory chunks (joined as string).
        model_id: Model identifier for the API call.
        client: Pre-configured OpenAI client.
        extra_kwargs: Additional kwargs for the API call (e.g., extra_body for Ollama).

    Returns:
        Generated answer string.
    """
    if client is None:
        from openai import OpenAI
        client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", ""))

    prompt = (
        f"CONVERSATION MEMORY:\n{context}\n\n"
        f"QUESTION: {question}\n\n"
        f"ANSWER:"
    )

    kwargs = dict(
        model=model_id,
        messages=[
            {"role": "system", "content": A_MEM_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        temperature=0.0,
        max_tokens=256,
    )
    if extra_kwargs:
        kwargs.update(extra_kwargs)

    try:
        response = client.chat.completions.create(**kwargs)
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"  Warning: QA generation failed for {model_id}: {e}")
        return ""


def _compute_memory_quality(
    memory: str,
    session_summaries: List[str],
    client=None,
    model_name: str = "gpt-4o",
) -> float:
    """
    AgeMem-style Memory Quality (MQ) metric: LLM evaluates how well the
    stored memory M′ captures the relevant facts from the conversation.

    The LLM is shown the memory state and the session summaries (as a proxy
    for ground-truth facts), and scores semantic coverage on a 0–1 scale.

    Args:
        memory: The evolved memory state M′.
        session_summaries: List of session summaries (ground-truth reference).
        client: Optional OpenAI client.
        model_name: Judge model name.

    Returns:
        Memory Quality score (0.0 to 1.0).
    """
    if not session_summaries:
        return 0.0

    if client is None:
        try:
            from openai import OpenAI
            client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", ""))
        except Exception:
            return 0.0

    summaries_text = "\n".join(
        f"- {s}" for s in session_summaries[:5]  # Cap at 5 to keep prompt manageable
    )

    prompt = (
        f"Evaluate the quality of the stored memory below against the "
        f"conversation summaries (ground truth).\n\n"
        f"SESSION SUMMARIES (ground truth):\n{summaries_text}\n\n"
        f"STORED MEMORY (M′):\n{memory[:2000]}\n\n"
        f"Score the memory quality on a 0.0 to 1.0 scale:\n"
        f"- 1.0: All key facts captured accurately, no contradictions\n"
        f"- 0.7: Most key facts captured, minor omissions\n"
        f"- 0.5: About half the key facts captured\n"
        f"- 0.3: Few key facts captured, significant omissions\n"
        f"- 0.0: No relevant facts captured or contains major errors\n\n"
        f"Answer with ONLY the numeric score (e.g., 0.85)."
    )

    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=20,
        )
        text = response.choices[0].message.content.strip()
        # Parse the numeric score
        import re as _re
        match = _re.search(r"(\d+\.?\d*)", text)
        if match:
            return float(match.group(1))
        return 0.0
    except Exception:
        return 0.0


def evaluate_transition(
    transition_fn: callable,
    method_name: str,
    test_data_path: str,
    cost_tracker: Optional[CostTracker] = None,
    max_samples: Optional[int] = None,
    reader_model: str = "gpt-4o",
    **method_kwargs,
) -> Dict:
    """
    Internal transition-quality evaluation on the held-out test split.

    Loads (M, delta_M, M_prime, category) triplets from test.jsonl,
    runs each method's transition, and computes per-operation metrics
    (ROUGE-L, fact-preservation, transition-judge, compression fidelity).

    This is the direct evidence for the headline claim that generative
    rewrite excels at consolidation and abstraction.
    """
    import json
    from pathlib import Path

    test_path = Path(test_data_path)
    if not test_path.exists():
        raise FileNotFoundError(f"Test data not found: {test_path}")

    examples = []
    with open(test_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if all(k in obj for k in ["M", "delta_M", "M_prime", "category"]):
                examples.append(obj)

    if max_samples:
        examples = examples[:max_samples]

    print(f"\n{'='*60}")
    print(f"Internal Transition Eval: {method_name} on {len(examples)} test triplets")
    print(f"{'='*60}")

    predictions = []
    golds = []
    inputs = []
    categories = []
    timings = []

    for i, ex in enumerate(examples):
        if (i + 1) % 20 == 0:
            print(f"  Processing {i+1}/{len(examples)}...")

        t_start = time.time()
        predicted = transition_fn(
            ex["M"], ex["delta_M"], **method_kwargs
        )
        t_end = time.time()
        latency_ms = (t_end - t_start) * 1000

        predictions.append(predicted)
        golds.append(ex["M_prime"])
        inputs.append({"M": ex["M"], "delta_M": ex["delta_M"]})
        categories.append(ex["category"])
        timings.append(latency_ms)

        if cost_tracker:
            cost_tracker.log_inference(latency_ms)

    # Compute per-operation metrics
    # Use LLM judge only if API key is available
    client = None
    if os.environ.get("OPENAI_API_KEY"):
        from openai import OpenAI
        client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    per_op = compute_transition_metrics(
        predictions, golds, inputs, categories,
        client=client, model_name=reader_model,
    )

    avg_latency = sum(timings) / len(timings) if timings else 0.0

    results = {
        "method": method_name,
        "num_samples": len(examples),
        "per_operation": per_op,
        "avg_latency_ms": avg_latency,
        "timestamp": datetime.now().isoformat(),
    }

    print(f"\n  Per-operation transition quality for {method_name}:")
    print(f"  {'Category':<25} {'ROUGE-L':<10} {'F1':<10} {'FactPres':<10} {'T-Judge':<10} {'CompR':<10}")
    print(f"  {'-'*25} {'-'*10} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")
    for cat, metrics in sorted(per_op.items()):
        print(
            f"  {cat:<25} "
            f"{metrics['rouge_l']:<10.4f} "
            f"{metrics['token_f1']:<10.4f} "
            f"{metrics['fact_preservation']:<10.4f} "
            f"{metrics['transition_judge']:<10.4f} "
            f"{metrics['pred_compression_ratio']:<10.4f}"
        )

    return results


def _save_built_memory(
    path: str,
    memory_indexes: List[str],
    full_dialogues: List[str],
    qa_pairs: List[Dict],
    method_name: str,
    benchmark_name: str,
) -> None:
    """Save built memory states to a JSONL file for later eval."""
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w", encoding="utf-8") as f:
        for i, qa in enumerate(qa_pairs):
            row = {
                "id": i,
                "question": qa["question"],
                "ground_truth": qa.get("ground_truth") or qa.get("answer", ""),
                "category": qa.get("category", "unknown"),
                "memory": memory_indexes[i] if memory_indexes[i] else "",
                "full_dialogue": full_dialogues[i] if full_dialogues[i] else "",
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"  Saved {len(qa_pairs)} built memories to {out_path}")


def _load_built_memory(
    path: str,
    max_samples: Optional[int] = None,
) -> Tuple[List[str], List[str], List[Dict]]:
    """Load pre-built memory states from a JSONL file."""
    memory_indexes = []
    full_dialogues = []
    qa_pairs = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            memory_indexes.append(row.get("memory", ""))
            full_dialogues.append(row.get("full_dialogue", ""))
            qa_pairs.append({
                "question": row["question"],
                "ground_truth": row.get("ground_truth", ""),
                "category": row.get("category", "unknown"),
            })
            if max_samples and len(qa_pairs) >= max_samples:
                break

    print(f"  Loaded {len(qa_pairs)} pre-built memories from {path}")
    return memory_indexes, full_dialogues, qa_pairs


def evaluate_on_benchmark(
    benchmark_name: str,
    transition_fn: callable,
    method_name: str,
    cost_tracker: Optional[CostTracker] = None,
    max_samples: Optional[int] = None,
    local_data_path: Optional[str] = None,
    top_k: int = 10,
    foundation_models: Optional[List[str]] = None,
    retriever: Optional[MemoryRetriever] = None,
    save_memory_path: Optional[str] = None,
    load_memory_path: Optional[str] = None,
    **method_kwargs,
) -> List[Dict]:
    """
    A-MEM pipeline: Conversation → Memory → chunk+embed → Retrieve Top-K
    → Foundation Model answers → Evaluate with full metrics.

    For locomo_full baseline: skips memory system and retrieval, feeds
    raw full dialogue directly to the answer model.

    Args:
        benchmark_name: 'locomo' or 'longmemeval'.
        transition_fn: Memory transition function (None for locomo_full).
        method_name: Name of the method.
        cost_tracker: Optional CostTracker.
        max_samples: Max samples.
        local_data_path: Optional local data path.
        top_k: Number of chunks to retrieve (default 10).
        foundation_models: List of foundation model names (default: ["gpt-4o-mini"]).
        retriever: Optional pre-built MemoryRetriever (shared embedder).
        **method_kwargs: Additional kwargs passed to transition_fn.

    Returns:
        List of result dicts, one per (method, benchmark, model, k).
    """
    if foundation_models is None:
        foundation_models = ["gpt-4o-mini"]

    is_full_context = (method_name == "locomo_full")

    print(f"\n{'='*60}")
    print(f"Evaluating {method_name} on {benchmark_name} (k={top_k}, models={foundation_models})")
    print(f"{'='*60}")

    # Load data
    if benchmark_name == "locomo":
        dataset = load_locomo(local_path=local_data_path, max_samples=max_samples)
        qa_pairs = extract_locomo_qa(dataset)
        build_memory = build_memory_locomo
        build_full = build_full_dialogue_locomo
    elif benchmark_name == "longmemeval":
        dataset = load_longmemeval(local_path=local_data_path, max_samples=max_samples)
        qa_pairs = extract_lme_qa(dataset)
        build_memory = build_memory_lme
        build_full = build_full_dialogue_lme
    else:
        raise ValueError(f"Unknown benchmark: {benchmark_name}")

    print(f"  Loaded {len(qa_pairs)} QA pairs")

    # Build retriever (shared embedder instance)
    if retriever is None:
        retriever = MemoryRetriever()

    # Pre-process: build memory + index for each QA pair
    # For locomo_full: build full dialogue, no indexing
    if load_memory_path:
        # Load pre-built memories from file (skip building phase)
        print(f"  Loading pre-built memory from {load_memory_path}...")
        memory_indexes, full_dialogues, qa_pairs = _load_built_memory(
            load_memory_path, max_samples
        )
        token_lengths = [
            count_tokens_tiktoken(mem if mem else fd)
            for mem, fd in zip(memory_indexes, full_dialogues)
        ]
        is_full_context = any(fd is not None for fd in full_dialogues)
    else:
        print(f"  Building memory for {len(qa_pairs)} conversations...")

        conversations = []
        memory_indexes = []
        full_dialogues = []
        token_lengths = []

        for i, qa in enumerate(qa_pairs):
            if (i + 1) % 50 == 0:
                print(f"    {i+1}/{len(qa_pairs)}...")

            conversation = qa.get("conversation") or qa.get("haystack_sessions", [])

            if is_full_context:
                full_dialogue = build_full(conversation)
                full_dialogues.append(full_dialogue)
                memory_indexes.append(None)
                token_lengths.append(count_tokens_tiktoken(full_dialogue))
            else:
                evolved_memory = build_memory(conversation, method=transition_fn)
                memory_indexes.append(evolved_memory)
                full_dialogues.append(None)
                token_lengths.append(count_tokens_tiktoken(evolved_memory))

    avg_memory_tokens = sum(token_lengths) / len(token_lengths) if token_lengths else 0.0
    print(f"    Avg memory tokens: {avg_memory_tokens:.0f}")

    # Save built memory to file (for later eval on local machine)
    if save_memory_path:
        _save_built_memory(
            save_memory_path, memory_indexes, full_dialogues,
            qa_pairs, method_name, benchmark_name,
        )
        print(f"    Saved built memory to {save_memory_path}")
        return []  # Return early — answer generation runs later via --load-memory

    # Evaluate per foundation model
    all_results = []

    for model_name in foundation_models:
        print(f"\n  --- Foundation Model: {model_name} ---")

        # Get client for this model
        model_id, client, extra_kwargs = get_reader_client(model_name)

        predictions = []
        ground_truths = []
        categories = []
        retrieved_token_lengths = []
        detailed_samples = []

        for i, qa in enumerate(qa_pairs):
            if (i + 1) % 50 == 0:
                print(f"    Answering {i+1}/{len(qa_pairs)}...")

            question = qa["question"]
            ground_truth = qa.get("ground_truth") or qa.get("answer", "")
            category = qa.get("category", "unknown")

            if is_full_context:
                context = full_dialogues[i]
                ret_tokens = token_lengths[i]
            else:
                # Retrieve top-k from indexed memory
                retriever.index(memory_indexes[i])
                chunks = retriever.retrieve(question, top_k=top_k)
                context = " ".join(chunks) if chunks else memory_indexes[i]
                ret_tokens = retriever.context_token_length(model=model_name)

            # Generate answer
            prediction = _generate_answer(
                question, context,
                model_id=model_id, client=client, extra_kwargs=extra_kwargs,
            )

            predictions.append(prediction)
            ground_truths.append(ground_truth)
            categories.append(category)
            retrieved_token_lengths.append(ret_tokens)

            detailed_samples.append({
                "sample_id": i,
                "question": question,
                "ground_truth": ground_truth,
                "prediction": prediction,
                "category": category,
                "retrieved_context": context[:500],
                "retrieved_tokens": ret_tokens,
            })

        # Compute metrics
        embedder = retriever.embedder if retriever else None
        qa_metrics = compute_full_qa_metrics(predictions, ground_truths, embedder=embedder)

        # Per-category breakdown
        per_category = compute_per_category_metrics(predictions, ground_truths, categories)

        avg_ret_tokens = sum(retrieved_token_lengths) / len(retrieved_token_lengths) if retrieved_token_lengths else 0.0

        result = {
            "benchmark": benchmark_name,
            "method": method_name,
            "model": model_name,
            "top_k": top_k,
            "num_samples": len(qa_pairs),
            "qa": qa_metrics,
            "per_category": per_category,
            "avg_memory_tokens": avg_memory_tokens,
            "avg_retrieved_tokens": avg_ret_tokens,
            "timestamp": datetime.now().isoformat(),
            "detailed_samples": detailed_samples,
        }

        all_results.append(result)

        print(f"    F1: {qa_metrics['f1']:.4f} | BLEU-1: {qa_metrics['bleu_1']:.4f} | "
              f"ROUGE-L: {qa_metrics['rouge_l']:.4f} | ROUGE-2: {qa_metrics['rouge_2']:.4f} | "
              f"Ret. Tokens: {avg_ret_tokens:.0f}")

    return all_results


# ---------------------------------------------------------------------------
# Detailed report writer (human-readable Markdown)
# ---------------------------------------------------------------------------


def _write_detailed_report(
    md_path: Path,
    method: str,
    benchmark: str,
    samples: List[Dict],
    qa_metrics: Dict,
) -> None:
    """Write a human-readable Markdown report of per-sample evaluation results."""

    # Count by category
    from collections import Counter
    cat_counts = Counter(s.get("category", "unknown") for s in samples)

    lines = []
    lines.append(f"# Detailed Eval Report")
    lines.append(f"")
    lines.append(f"**Method:** `{method}` | **Benchmark:** `{benchmark}`")
    lines.append(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"**Samples:** {len(samples)}")
    lines.append(f"")
    lines.append(f"## Summary Metrics")
    lines.append(f"")
    lines.append(f"| Metric | Value |")
    lines.append(f"|---|---|")
    lines.append(f"| F1 | {qa_metrics.get('f1', 0):.4f} |")
    lines.append(f"| BLEU-1 | {qa_metrics.get('bleu_1', 0):.4f} |")
    lines.append(f"| ROUGE-L | {qa_metrics.get('rouge_l', 0):.4f} |")
    lines.append(f"| ROUGE-2 | {qa_metrics.get('rouge_2', 0):.4f} |")
    lines.append(f"| METEOR | {qa_metrics.get('meteor', 0):.4f} |")
    lines.append(f"| SBERT Sim | {qa_metrics.get('sbert_sim', 0):.4f} |")
    lines.append(f"")
    lines.append(f"## Category Distribution")
    lines.append(f"")
    for cat, count in sorted(cat_counts.items()):
        lines.append(f"- **{cat}**: {count}")
    lines.append(f"")
    lines.append(f"---")
    lines.append(f"")

    for i, sample in enumerate(samples):
        lines.append(f"## Sample {i}")
        lines.append(f"")
        lines.append(f"| Field | Value |")
        lines.append(f"|---|---|")
        lines.append(f"| **Category** | `{sample.get('category', '?')}` |")
        lines.append(f"| **Retrieved Tokens** | {sample.get('retrieved_tokens', 0)} |")
        lines.append(f"")

        # Question & Answers
        lines.append(f"### ❓ Question")
        lines.append(f"")
        lines.append(f"> {sample.get('question', '')}")
        lines.append(f"")

        lines.append(f"### 🎯 Ground Truth")
        lines.append(f"")
        lines.append(f"```")
        lines.append(f"{sample.get('ground_truth', '')}")
        lines.append(f"```")
        lines.append(f"")

        lines.append(f"### 🤖 Prediction")
        lines.append(f"")
        lines.append(f"```")
        lines.append(f"{sample.get('prediction', '')}")
        lines.append(f"```")
        lines.append(f"")

        # Memory states
        raw_mem = sample.get("raw_memory", "")
        evolved_mem = sample.get("evolved_memory", "")

        lines.append(f"### 📚 Retrieved Context")
        lines.append(f"")
        lines.append(f"```")
        lines.append(f"{sample.get('retrieved_context', '')}")
        lines.append(f"```")
        lines.append(f"")
        lines.append(f"---")
        lines.append(f"")

    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"  Detailed report saved: {md_path}")


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate memory transition methods on long-term memory benchmarks"
    )
    parser.add_argument(
        "--method",
        type=str,
        required=True,
        help=f"Method to evaluate, or 'all'. Choices: {list(METHOD_MAP.keys()) + ['all']}",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="benchmark",
        choices=["benchmark", "transition"],
        help="Evaluation mode: 'benchmark' (LoCoMo/LongMemEval QA) or "
             "'transition' (internal per-operation eval on test split)",
    )
    parser.add_argument(
        "--benchmark",
        type=str,
        default="all",
        help="Benchmark to evaluate on (benchmark mode only). "
             "Choices: locomo, longmemeval, all",
    )
    parser.add_argument(
        "--test_data",
        type=str,
        default="../../data/processed/test.jsonl",
        help="Path to test split JSONL (transition mode only)",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to trained MSTM checkpoint (required for --method mstm)",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="../../results/raw_outputs/",
        help="Output directory for results",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Max samples to evaluate (for quick smoke tests)",
    )
    parser.add_argument(
        "--reader-model",
        type=str,
        default="gpt-4o",
        help="Model used for answer generation (default: gpt-4o)",
    )
    parser.add_argument(
        "--local-data",
        type=str,
        default=None,
        help="Path to local benchmark data (skips HuggingFace download)",
    )
    parser.add_argument(
        "--age-threshold",
        type=float,
        default=2.0,
        help="Age threshold for time_decay baseline (years)",
    )
    parser.add_argument(
        "--similarity-threshold",
        type=float,
        default=0.85,
        help="Similarity threshold for heuristic_consolidation baseline",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Number of chunks to retrieve for QA (default: 10)",
    )
    parser.add_argument(
        "--k-sweep",
        type=str,
        default=None,
        help="Comma-separated k values for sensitivity analysis (e.g., 10,20,30,40,50). "
             "GPT-4o-mini ONLY per A-MEM paper.",
    )
    parser.add_argument(
        "--foundation-models",
        type=str,
        default="gpt-4o-mini",
        help="Comma-separated foundation model names for answer generation "
             "(default: gpt-4o-mini). Available: " + ", ".join(FOUNDATION_MODELS.keys()),
    )
    parser.add_argument(
        "--save-memory",
        type=str,
        default=None,
        help="Save built memory states to JSONL file (skip answer generation). "
             "Use with --load-memory on another machine.",
    )
    parser.add_argument(
        "--load-memory",
        type=str,
        default=None,
        help="Load pre-built memory states from JSONL file (skip memory building). "
             "Use with --save-memory from cloud GPU.",
    )

    args = parser.parse_args()

    # Determine methods to evaluate
    if args.method == "all":
        methods = [m for m in METHOD_MAP.keys() if m != "mstm"]
        if args.checkpoint:
            methods.append("mstm")
    else:
        methods = [args.method]

    # Parse foundation models and k-values
    foundation_models = [m.strip() for m in args.foundation_models.split(",")]
    k_values = [args.top_k]
    if args.k_sweep:
        k_values = [int(k.strip()) for k in args.k_sweep.split(",")]

    # Shared retriever (one embedder instance for all methods)
    retriever = MemoryRetriever()

    # Setup output directory
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Run evaluations
    cost_tracker = CostTracker()
    all_results = []

    # ── Transition mode ────────────────────────────────────────────────
    if args.mode == "transition":
        for method_name in methods:
            print(f"\n{'#'*60}")
            print(f"# Transition Eval: {method_name}")
            print(f"{'#'*60}")

            try:
                transition_fn, extra = load_method(method_name, args.checkpoint)
            except Exception as e:
                print(f"  ERROR loading method '{method_name}': {e}")
                continue

            method_kwargs = {}
            if method_name == "time_decay":
                method_kwargs["age_threshold"] = args.age_threshold
            elif method_name == "heuristic_consolidation":
                method_kwargs["similarity_threshold"] = args.similarity_threshold

            try:
                results = evaluate_transition(
                    transition_fn=transition_fn,
                    method_name=method_name,
                    test_data_path=args.test_data,
                    cost_tracker=cost_tracker,
                    max_samples=args.max_samples,
                    reader_model=foundation_models[0],  # Use first model as judge
                    **method_kwargs,
                )
                all_results.append(results)
            except Exception as e:
                print(f"  ERROR in transition eval: {e}")
                import traceback
                traceback.print_exc()
                continue

        # Save transition results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = out_dir / f"transition_eval_{timestamp}.json"
        with open(results_file, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "results": all_results,
                    "cost_summary": cost_tracker.summary(),
                    "config": {
                        "methods": methods,
                        "test_data": args.test_data,
                        "max_samples": args.max_samples,
                        "timestamp": timestamp,
                    },
                },
                f,
                indent=2,
                ensure_ascii=False,
            )

        # Print per-operation summary
        for r in all_results:
            print(f"\n  {r['method']} per-operation breakdown:")
            print(f"  {'Category':<25} {'ROUGE-L':<10} {'F1':<10} {'FactPres':<10} {'T-Judge':<10} {'CompR':<10}")
            print(f"  {'-'*25} {'-'*10} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")
            for cat, m in sorted(r["per_operation"].items()):
                print(
                    f"  {cat:<25} "
                    f"{m['rouge_l']:<10.4f} "
                    f"{m['token_f1']:<10.4f} "
                    f"{m['fact_preservation']:<10.4f} "
                    f"{m['transition_judge']:<10.4f} "
                    f"{m['pred_compression_ratio']:<10.4f}"
                )

        print(f"\nSummary: {results_file}")
        return

    # ── Benchmark mode (A-MEM pipeline) ─────────────────────────────────

    # Determine benchmarks
    if args.benchmark == "all":
        benchmarks = ["locomo", "longmemeval"]
    else:
        benchmarks = [args.benchmark]

    for method_name in methods:
        print(f"\n{'#'*60}")
        print(f"# Method: {method_name}")
        print(f"{'#'*60}")

        try:
            transition_fn, extra = load_method(method_name, args.checkpoint)
        except Exception as e:
            print(f"  ERROR loading method '{method_name}': {e}")
            print(f"  Skipping...")
            continue

        method_kwargs = {}
        if method_name == "time_decay":
            method_kwargs["age_threshold"] = args.age_threshold
        elif method_name == "heuristic_consolidation":
            method_kwargs["similarity_threshold"] = args.similarity_threshold

        for benchmark_name in benchmarks:
            for k in k_values:
                try:
                    results = evaluate_on_benchmark(
                        benchmark_name=benchmark_name,
                        transition_fn=transition_fn,
                        method_name=method_name,
                        cost_tracker=cost_tracker,
                        max_samples=args.max_samples,
                        local_data_path=args.local_data,
                        top_k=k,
                        foundation_models=foundation_models,
                        retriever=retriever,
                        save_memory_path=args.save_memory,
                        load_memory_path=args.load_memory,
                        **method_kwargs,
                    )
                    if results:  # Empty list when --save-memory is used
                        all_results.extend(results)
                except Exception as e:
                    print(f"  ERROR on {benchmark_name} (k={k}): {e}")
                    import traceback
                    traceback.print_exc()
                    continue

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 1. Save summary JSON (without detailed_samples)
    summary_results = []
    for r in all_results:
        summary = {k: v for k, v in r.items() if k != "detailed_samples"}
        summary_results.append(summary)

    results_file = out_dir / f"eval_results_{timestamp}.json"
    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(
            {
                "results": summary_results,
                "cost_summary": cost_tracker.summary(),
                "config": {
                    "methods": methods,
                    "benchmarks": benchmarks,
                    "foundation_models": foundation_models,
                    "top_k": k_values,
                    "max_samples": args.max_samples,
                    "timestamp": timestamp,
                },
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    # 2. Save detailed per-sample results
    details_dir = out_dir / "details"
    details_dir.mkdir(parents=True, exist_ok=True)

    for r in all_results:
        method = r["method"]
        benchmark = r["benchmark"]
        model = r.get("model", "unknown")
        k = r.get("top_k", 0)
        samples = r.get("detailed_samples", [])

        if not samples:
            continue

        # Save as JSONL
        details_file = details_dir / f"{method}_{model}_k{k}_{benchmark}_{timestamp}.jsonl"
        with open(details_file, "w", encoding="utf-8") as f:
            for sample in samples:
                json.dump(sample, f, ensure_ascii=False)
                f.write("\n")

        # Save as Markdown
        md_file = details_dir / f"{method}_{model}_k{k}_{benchmark}_{timestamp}.md"
        _write_detailed_report(md_file, method, f"{benchmark}/{model}/k={k}", samples, r["qa"])

    print(f"\n{'='*60}")
    print(f"Evaluation complete!")
    print(f"  Summary: {results_file}")
    print(f"  Details: {details_dir}/")
    print(f"{'='*60}")

    # Print summary table (A-MEM style: one row per method × model)
    header = f"{'Method':<25} {'Model':<16} {'Benchmark':<14} {'k':<4} {'F1':<8} {'BLEU-1':<8} {'ROUGE-L':<8} {'ROUGE-2':<8} {'MemTok':<8} {'RetTok':<8}"
    print(f"\n{header}")
    print("-" * len(header))
    for r in all_results:
        print(
            f"{r['method']:<25} "
            f"{r.get('model', '?'):<16} "
            f"{r['benchmark']:<14} "
            f"{r.get('top_k', '?'):<4} "
            f"{r['qa']['f1']:<8.4f} "
            f"{r['qa']['bleu_1']:<8.4f} "
            f"{r['qa']['rouge_l']:<8.4f} "
            f"{r['qa']['rouge_2']:<8.4f} "
            f"{r.get('avg_memory_tokens', 0):<8.0f} "
            f"{r.get('avg_retrieved_tokens', 0):<8.0f}"
        )


if __name__ == "__main__":
    main()
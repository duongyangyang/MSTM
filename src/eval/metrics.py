"""
Evaluation metrics for memory state transition.

Implements all metrics specified in PROPOSAL.md Section 8 and TABLE.md:
- QA: Exact Match (EM), F1 Score
- Retrieval: Recall@K, MRR (Mean Reciprocal Rank)
- Memory efficiency: record count, token count, compression ratio
- Consistency: contradiction rate, redundancy rate
- Temporal: temporal QA accuracy, update-sensitive accuracy
- Cost: GPU-hours, examples seen, inference latency

Usage:
    from src.eval.metrics import compute_qa_metrics, compute_retrieval_metrics, ...
"""

import re
import string
import time
from collections import Counter
from typing import Dict, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# QA Metrics
# ---------------------------------------------------------------------------


def normalize_text(text: str) -> str:
    """Normalize text for comparison: lowercase, strip, normalize whitespace."""
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    return text


def exact_match(prediction: str, ground_truth: str) -> float:
    """
    Exact Match: 1.0 if prediction matches ground truth exactly after
    normalization, 0.0 otherwise.
    """
    return 1.0 if normalize_text(prediction) == normalize_text(ground_truth) else 0.0


def token_f1(prediction: str, ground_truth: str) -> float:
    """
    Token-level F1 score: harmonic mean of token precision and recall.

    Tokens are split on whitespace after removing punctuation-only tokens.
    """
    pred_tokens = _tokenize(prediction)
    gt_tokens = _tokenize(ground_truth)

    if not pred_tokens and not gt_tokens:
        return 1.0
    if not pred_tokens or not gt_tokens:
        return 0.0

    pred_counter = Counter(pred_tokens)
    gt_counter = Counter(gt_tokens)

    common = pred_counter & gt_counter
    num_common = sum(common.values())

    if num_common == 0:
        return 0.0

    precision = num_common / len(pred_tokens)
    recall = num_common / len(gt_tokens)

    if precision + recall == 0:
        return 0.0

    return 2 * precision * recall / (precision + recall)


def _tokenize(text: str) -> List[str]:
    """Tokenize text into words, filtering out pure punctuation tokens."""
    text = normalize_text(text)
    # Remove punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))
    return [t for t in text.split() if t.strip()]


def compute_qa_metrics(
    predictions: List[str],
    ground_truths: List[str],
    llm_judge_scores: Optional[List[float]] = None,
) -> Dict[str, float]:
    """
    Compute QA metrics across a batch of predictions.

    Args:
        predictions: List of predicted answers.
        ground_truths: List of ground truth answers.
        llm_judge_scores: Optional pre-computed LLM judge scores (0.0-1.0).
                         If provided, included as 'llm_judge_accuracy'.

    Returns:
        Dict with keys: em (Exact Match), f1 (F1 Score), count (num samples),
        and optionally llm_judge_accuracy.
    """
    assert len(predictions) == len(ground_truths), (
        f"Length mismatch: {len(predictions)} preds vs {len(ground_truths)} gts"
    )

    em_scores = []
    f1_scores = []

    for pred, gt in zip(predictions, ground_truths):
        em_scores.append(exact_match(pred, gt))
        f1_scores.append(token_f1(pred, gt))

    n = len(predictions)
    result = {
        "em": sum(em_scores) / n if n > 0 else 0.0,
        "f1": sum(f1_scores) / n if n > 0 else 0.0,
        "count": n,
    }

    if llm_judge_scores is not None:
        result["llm_judge_accuracy"] = (
            sum(llm_judge_scores) / len(llm_judge_scores)
            if llm_judge_scores else 0.0
        )

    return result


def llm_judge_score(
    prediction: str,
    ground_truth: str,
    question: str = "",
    client=None,
    model_name: str = "gpt-4o",
) -> float:
    """
    Use an LLM judge to score whether a prediction is semantically correct.

    This is the standard evaluation approach in memory benchmarks (LoCoMo,
    LongMemEval) because EM is too strict for free-form QA answers.

    Args:
        prediction: The model's predicted answer.
        ground_truth: The correct answer.
        question: The original question (provides context for judging).
        client: Optional pre-configured OpenAI client.
        model_name: LLM to use as judge.

    Returns:
        Score: 1.0 (correct), 0.5 (partially correct), 0.0 (incorrect).
    """
    if client is None:
        try:
            from openai import OpenAI
            client = OpenAI()
        except Exception:
            # Fallback: return F1 as a rough proxy if no API available
            return token_f1(prediction, ground_truth)

    judge_prompt = (
        f'Question: {question}\n\n'
        f'Correct answer: {ground_truth}\n\n'
        f'Predicted answer: {prediction}\n\n'
        f'Is the predicted answer correct? Answer with ONLY one word: '
        f'"correct" (fully matches the meaning), '
        f'"partial" (partially correct but missing details or has minor errors), '
        f'or "incorrect" (wrong or contradictory).'
    )

    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": judge_prompt}],
            temperature=0.0,
            max_tokens=10,
        )
        verdict = response.choices[0].message.content.strip().lower()

        if "correct" in verdict and "partial" not in verdict and "incorrect" not in verdict:
            return 1.0
        elif "partial" in verdict:
            return 0.5
        else:
            return 0.0
    except Exception:
        # Fallback on API error
        return token_f1(prediction, ground_truth)


def compute_llm_judge_scores(
    predictions: List[str],
    ground_truths: List[str],
    questions: List[str] = None,
    client=None,
    model_name: str = "gpt-4o",
) -> List[float]:
    """
    Compute LLM judge scores for a batch.

    Args:
        predictions: List of predicted answers.
        ground_truths: List of ground truth answers.
        questions: List of original questions (for context).
        client: Optional OpenAI client.
        model_name: Judge model name.

    Returns:
        List of scores (1.0, 0.5, or 0.0).
    """
    if questions is None:
        questions = [""] * len(predictions)

    assert len(predictions) == len(ground_truths) == len(questions)

    scores = []
    for pred, gt, q in zip(predictions, ground_truths, questions):
        score = llm_judge_score(pred, gt, q, client=client, model_name=model_name)
        scores.append(score)

    return scores


# ---------------------------------------------------------------------------
# Retrieval Metrics
# ---------------------------------------------------------------------------


def recall_at_k(
    retrieved_ids: List[str],
    relevant_ids: List[str],
    k: int,
) -> float:
    """
    Recall@K: proportion of relevant items retrieved in the top K results.

    Args:
        retrieved_ids: Ranked list of retrieved item IDs (ordered by relevance).
        relevant_ids: Set of relevant item IDs.
        k: Number of top results to consider.

    Returns:
        Recall@K score (0.0 to 1.0).
    """
    if not relevant_ids:
        return 1.0  # No relevant items → perfect recall vacuously

    top_k = set(retrieved_ids[:k])
    relevant_set = set(relevant_ids)
    return len(top_k & relevant_set) / len(relevant_set)


def mean_reciprocal_rank(
    retrieved_ids: List[str],
    relevant_ids: List[str],
) -> float:
    """
    MRR: Mean Reciprocal Rank — 1 / rank of the first relevant item.

    Args:
        retrieved_ids: Ranked list of retrieved item IDs.
        relevant_ids: Set of relevant item IDs.

    Returns:
        MRR score (0.0 to 1.0).
    """
    if not relevant_ids:
        return 1.0

    relevant_set = set(relevant_ids)
    for i, rid in enumerate(retrieved_ids, start=1):
        if rid in relevant_set:
            return 1.0 / i
    return 0.0


def compute_retrieval_metrics(
    all_retrieved: List[List[str]],
    all_relevant: List[List[str]],
    k_values: List[int] = [1, 5, 10],
) -> Dict[str, float]:
    """
    Compute retrieval metrics across a batch.

    Args:
        all_retrieved: List of ranked retrieval results.
        all_relevant: List of relevant item ID sets.
        k_values: K values for Recall@K.

    Returns:
        Dict with recall@k and mrr scores.
    """
    assert len(all_retrieved) == len(all_relevant)

    metrics = {}
    for k in k_values:
        scores = [
            recall_at_k(ret, rel, k)
            for ret, rel in zip(all_retrieved, all_relevant)
        ]
        metrics[f"recall@{k}"] = sum(scores) / len(scores) if scores else 0.0

    mrr_scores = [
        mean_reciprocal_rank(ret, rel)
        for ret, rel in zip(all_retrieved, all_relevant)
    ]
    metrics["mrr"] = sum(mrr_scores) / len(mrr_scores) if mrr_scores else 0.0
    metrics["count"] = len(all_retrieved)

    return metrics


# ---------------------------------------------------------------------------
# Memory Efficiency Metrics
# ---------------------------------------------------------------------------


def count_records(memory: str) -> int:
    """
    Count the number of memory records (bullet points or lines).
    """
    lines = [l.strip() for l in memory.split("\n") if l.strip()]
    return len(lines)


def count_tokens(memory: str) -> int:
    """
    Estimate token count using whitespace splitting.
    For exact token counts, use the tokenizer from the model.
    """
    return len(memory.split())


def compression_ratio(
    original_memory: str,
    transformed_memory: str,
) -> float:
    """
    Compression ratio: |M′| / |M|.
    < 1.0 means the memory was compressed (fewer tokens).
    > 1.0 means the memory grew.
    """
    orig_tokens = count_tokens(original_memory)
    if orig_tokens == 0:
        return 1.0
    return count_tokens(transformed_memory) / orig_tokens


def compute_memory_efficiency(
    original_memories: List[str],
    transformed_memories: List[str],
) -> Dict[str, float]:
    """
    Compute memory efficiency metrics across a batch.

    Args:
        original_memories: List of original memory states (before transition).
        transformed_memories: List of evolved memory states (after transition).

    Returns:
        Dict with avg_records, avg_tokens, compression_ratio, etc.
    """
    assert len(original_memories) == len(transformed_memories)

    orig_records = [count_records(m) for m in original_memories]
    trans_records = [count_records(m) for m in transformed_memories]
    orig_tokens_list = [count_tokens(m) for m in original_memories]
    trans_tokens_list = [count_tokens(m) for m in transformed_memories]

    ratios = [
        compression_ratio(o, t)
        for o, t in zip(original_memories, transformed_memories)
    ]

    n = len(original_memories)
    return {
        "avg_original_records": sum(orig_records) / n if n > 0 else 0.0,
        "avg_transformed_records": sum(trans_records) / n if n > 0 else 0.0,
        "avg_original_tokens": sum(orig_tokens_list) / n if n > 0 else 0.0,
        "avg_transformed_tokens": sum(trans_tokens_list) / n if n > 0 else 0.0,
        "compression_ratio": sum(ratios) / n if n > 0 else 1.0,
        "count": n,
    }


# ---------------------------------------------------------------------------
# Consistency Metrics
# ---------------------------------------------------------------------------


def _detect_contradictions(memory: str) -> int:
    """
    Heuristic contradiction detection: look for patterns suggesting
    conflicting facts (e.g., "previously" + current state, "moved from").
    This is a rough heuristic — for precise evaluation, an LLM judge
    is preferred.
    """
    contradiction_markers = [
        # Temporal contradictions
        (r"\b(previously|formerly|used to)\b.*\b(now|currently|is)\b", 0.5),
        # Location contradictions
        (r"\blives in\b.*\blives in\b", 0.3),
        # Status contradictions
        (r"\b(is|works as)\b.*\b(was|worked as)\b", 0.2),
    ]

    lines = [l.strip().lower() for l in memory.split("\n") if l.strip()]
    contradiction_score = 0

    for pattern, weight in contradiction_markers:
        for line in lines:
            if re.search(pattern, line):
                contradiction_score += weight

    return int(contradiction_score)


def _detect_redundancy(memory: str) -> int:
    """
    Heuristic redundancy detection: look for repeated information across
    multiple records.
    """
    lines = [l.strip().lower() for l in memory.split("\n") if l.strip()]
    if len(lines) <= 1:
        return 0

    # Simple token overlap between line pairs
    redundant_pairs = 0
    for i in range(len(lines)):
        tokens_i = set(_tokenize(lines[i]))
        if len(tokens_i) < 3:
            continue
        for j in range(i + 1, len(lines)):
            tokens_j = set(_tokenize(lines[j]))
            if len(tokens_j) < 3:
                continue
            overlap = len(tokens_i & tokens_j) / min(len(tokens_i), len(tokens_j))
            if overlap > 0.7:
                redundant_pairs += 1

    return redundant_pairs


def compute_consistency_metrics(
    memories: List[str],
) -> Dict[str, float]:
    """
    Compute consistency metrics across a batch of memory states.

    Args:
        memories: List of memory state strings.

    Returns:
        Dict with contradiction_rate and redundancy_rate.
    """
    contradiction_counts = [_detect_contradictions(m) for m in memories]
    redundancy_counts = [_detect_redundancy(m) for m in memories]

    n = len(memories)
    total_records = sum(count_records(m) for m in memories)

    return {
        "contradiction_rate": (
            sum(contradiction_counts) / total_records if total_records > 0 else 0.0
        ),
        "redundancy_rate": (
            sum(redundancy_counts) / total_records if total_records > 0 else 0.0
        ),
        "total_contradictions": sum(contradiction_counts),
        "total_redundancies": sum(redundancy_counts),
        "count": n,
    }


# ---------------------------------------------------------------------------
# Temporal Reasoning Metrics
# ---------------------------------------------------------------------------


def compute_temporal_metrics(
    predictions: List[str],
    ground_truths: List[str],
    categories: List[str],
) -> Dict[str, float]:
    """
    Compute temporal reasoning metrics, broken down by question category.

    Args:
        predictions: List of predicted answers.
        ground_truths: List of ground truth answers.
        categories: List of question categories (must include 'temporal'
                   and 'update-sensitive' for meaningful breakdown).

    Returns:
        Dict with overall accuracy, temporal_accuracy, update_sensitive_accuracy.
    """
    assert len(predictions) == len(ground_truths) == len(categories)

    temporal_preds = []
    temporal_gts = []
    update_preds = []
    update_gts = []
    all_f1 = []

    for pred, gt, cat in zip(predictions, ground_truths, categories):
        f1 = token_f1(pred, gt)
        all_f1.append(f1)

        cat_lower = cat.lower().replace("_", "-").replace(" ", "-")
        if "temporal" in cat_lower:
            temporal_preds.append(pred)
            temporal_gts.append(gt)
        if "update" in cat_lower:
            update_preds.append(pred)
            update_gts.append(gt)

    n = len(predictions)
    result = {
        "overall_f1": sum(all_f1) / n if n > 0 else 0.0,
        "count": n,
    }

    if temporal_preds:
        temporal_metrics = compute_qa_metrics(temporal_preds, temporal_gts)
        result["temporal_f1"] = temporal_metrics["f1"]
        result["temporal_count"] = temporal_metrics["count"]
    else:
        result["temporal_f1"] = 0.0
        result["temporal_count"] = 0

    if update_preds:
        update_metrics = compute_qa_metrics(update_preds, update_gts)
        result["update_sensitive_f1"] = update_metrics["f1"]
        result["update_sensitive_count"] = update_metrics["count"]
    else:
        result["update_sensitive_f1"] = 0.0
        result["update_sensitive_count"] = 0

    return result


# ---------------------------------------------------------------------------
# ROUGE-L Metric
# ---------------------------------------------------------------------------


def rouge_l(prediction: str, ground_truth: str) -> float:
    """
    ROUGE-L F-measure: longest common subsequence (LCS) based F1.

    Implements the standard ROUGE-L formulation without external dependencies.
    LCS recall = |LCS| / |ground_truth_tokens|
    LCS precision = |LCS| / |prediction_tokens|
    F1 = 2 * P * R / (P + R)

    Args:
        prediction: Predicted text.
        ground_truth: Reference text.

    Returns:
        ROUGE-L F1 score (0.0 to 1.0).
    """
    pred_tokens = _tokenize(prediction)
    gt_tokens = _tokenize(ground_truth)

    if not pred_tokens and not gt_tokens:
        return 1.0
    if not pred_tokens or not gt_tokens:
        return 0.0

    lcs_len = _lcs_length(pred_tokens, gt_tokens)
    if lcs_len == 0:
        return 0.0

    precision = lcs_len / len(pred_tokens)
    recall = lcs_len / len(gt_tokens)
    return 2 * precision * recall / (precision + recall)


def _lcs_length(a: list, b: list) -> int:
    """Compute length of longest common subsequence using DP."""
    m, n = len(a), len(b)
    # Use 1D DP for memory efficiency
    prev = [0] * (n + 1)
    for i in range(1, m + 1):
        curr = [0] * (n + 1)
        for j in range(1, n + 1):
            if a[i - 1] == b[j - 1]:
                curr[j] = prev[j - 1] + 1
            else:
                curr[j] = max(prev[j], curr[j - 1])
        prev = curr
    return prev[n]


# ---------------------------------------------------------------------------
# A-MEM-aligned QA Metrics
# ---------------------------------------------------------------------------

# NLTK bootstrap for METEOR (first use downloads wordnet data)
_nltk_ready = False


def _ensure_nltk():
    """Lazy-download NLTK data needed for METEOR."""
    global _nltk_ready
    if _nltk_ready:
        return
    try:
        import nltk
        nltk.download("wordnet", quiet=True)
        nltk.download("omw-1.4", quiet=True)
        _nltk_ready = True
    except Exception:
        pass  # METEOR will fail gracefully later


def bleu_1(prediction: str, ground_truth: str) -> float:
    """
    BLEU-1 score: unigram precision with brevity penalty.

    Args:
        prediction: Predicted text.
        ground_truth: Reference text.

    Returns:
        BLEU-1 score (0.0 to 1.0).
    """
    pred_tokens = _tokenize(prediction)
    gt_tokens = _tokenize(ground_truth)

    if not pred_tokens and not gt_tokens:
        return 1.0
    if not pred_tokens or not gt_tokens:
        return 0.0

    pred_counter = Counter(pred_tokens)
    gt_counter = Counter(gt_tokens)

    common = pred_counter & gt_counter
    num_common = sum(common.values())

    if num_common == 0:
        return 0.0

    precision = num_common / len(pred_tokens)

    # Brevity penalty
    bp = 1.0 if len(pred_tokens) >= len(gt_tokens) else np.exp(1 - len(gt_tokens) / len(pred_tokens))

    return bp * precision


def rouge_n(prediction: str, ground_truth: str, n: int = 2) -> float:
    """
    ROUGE-N F-measure: n-gram overlap F1.

    Args:
        prediction: Predicted text.
        ground_truth: Reference text.
        n: N-gram size (1 for unigram, 2 for bigram, etc.).

    Returns:
        ROUGE-N F1 score (0.0 to 1.0).
    """
    pred_tokens = _tokenize(prediction)
    gt_tokens = _tokenize(ground_truth)

    if not pred_tokens and not gt_tokens:
        return 1.0
    if not pred_tokens or not gt_tokens:
        return 0.0

    pred_ngrams = Counter(_ngrams(pred_tokens, n))
    gt_ngrams = Counter(_ngrams(gt_tokens, n))

    common = pred_ngrams & gt_ngrams
    num_common = sum(common.values())

    if num_common == 0:
        return 0.0

    precision = num_common / max(len(pred_ngrams), 1)
    recall = num_common / max(len(gt_ngrams), 1)

    return 2 * precision * recall / (precision + recall)


def _ngrams(tokens: List[str], n: int) -> List[Tuple[str, ...]]:
    """Generate n-grams from a token list."""
    return [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]


def meteor(prediction: str, ground_truth: str) -> float:
    """
    METEOR score via NLTK.

    Requires NLTK wordnet data (auto-downloaded on first use).

    Args:
        prediction: Predicted text.
        ground_truth: Reference text.

    Returns:
        METEOR score (0.0 to 1.0).
    """
    _ensure_nltk()
    try:
        from nltk.translate.meteor_score import meteor_score
        return meteor_score([ground_truth.split()], prediction.split())
    except Exception:
        # Fallback to F1 if METEOR fails
        return token_f1(prediction, ground_truth)


def sbert_similarity(
    prediction: str,
    ground_truth: str,
    embedder: object = None,
) -> float:
    """
    SBERT cosine similarity between prediction and ground truth.

    Args:
        prediction: Predicted text.
        ground_truth: Reference text.
        embedder: Optional pre-loaded SentenceTransformer instance.
                 If None, loads all-MiniLM-L6-v2 (slow for batch use).

    Returns:
        Cosine similarity (0.0 to 1.0).
    """
    if not prediction.strip() and not ground_truth.strip():
        return 1.0
    if not prediction.strip() or not ground_truth.strip():
        return 0.0

    if embedder is None:
        from sentence_transformers import SentenceTransformer
        embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    embeddings = embedder.encode(
        [prediction, ground_truth],
        convert_to_numpy=True,
        show_progress_bar=False,
    )

    # Cosine similarity
    a, b = embeddings[0], embeddings[1]
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))


def count_tokens_tiktoken(text: str, model: str = "gpt-4o-mini") -> int:
    """
    Accurate token count using tiktoken (matching A-MEM Token Length metric).

    Args:
        text: The text to tokenize.
        model: Model name for tiktoken encoding.

    Returns:
        Number of tokens.
    """
    try:
        import tiktoken
    except ImportError:
        return len(text.split())  # fallback

    try:
        enc = tiktoken.encoding_for_model(model)
    except KeyError:
        enc = tiktoken.get_encoding("cl100k_base")

    return len(enc.encode(text))


def compute_full_qa_metrics(
    predictions: List[str],
    ground_truths: List[str],
    embedder: object = None,
) -> Dict[str, float]:
    """
    Compute all A-MEM-aligned QA metrics across a batch.

    Metrics: F1, BLEU-1, ROUGE-L, ROUGE-2, METEOR, SBERT Similarity.

    Args:
        predictions: List of predicted answers.
        ground_truths: List of ground truth answers.
        embedder: Optional pre-loaded SentenceTransformer (shared with retriever).

    Returns:
        Dict with keys: f1, bleu_1, rouge_l, rouge_2, meteor, sbert_sim, count.
    """
    assert len(predictions) == len(ground_truths), (
        f"Length mismatch: {len(predictions)} preds vs {len(ground_truths)} gts"
    )

    f1_scores, bleu1_scores, rouge_l_scores, rouge_2_scores = [], [], [], []
    meteor_scores, sbert_scores = [], []

    for pred, gt in zip(predictions, ground_truths):
        f1_scores.append(token_f1(pred, gt))
        bleu1_scores.append(bleu_1(pred, gt))
        rouge_l_scores.append(rouge_l(pred, gt))
        rouge_2_scores.append(rouge_n(pred, gt, n=2))
        meteor_scores.append(meteor(pred, gt))
        sbert_scores.append(sbert_similarity(pred, gt, embedder=embedder))

    n = len(predictions)
    return {
        "f1": sum(f1_scores) / n,
        "bleu_1": sum(bleu1_scores) / n,
        "rouge_l": sum(rouge_l_scores) / n,
        "rouge_2": sum(rouge_2_scores) / n,
        "meteor": sum(meteor_scores) / n,
        "sbert_sim": sum(sbert_scores) / n,
        "count": n,
    }


def compute_average_ranking(
    per_category_scores: Dict[str, Dict[str, Dict[str, float]]],
    metric: str = "f1",
) -> Dict[str, float]:
    """
    A-MEM Average Ranking: per-method mean rank across LoCoMo categories.

    For each category, methods are ranked by the given metric.
    A method's average ranking = mean of its within-category ranks.
    Lower is better (1.0 = best, ranks #1 in every category).

    Ties are handled with average ranking (competition ranking skews).

    Args:
        per_category_scores: Nested dict: {category: {method: {metric: value}}}.
        metric: Which metric to rank by (default: "f1").

    Returns:
        Dict mapping method → average_ranking (lower is better).
    """
    # Collect per-category rankings
    method_ranks: Dict[str, List[float]] = {}

    for cat, method_scores in per_category_scores.items():
        # Sort methods by metric (higher is better for F1/BLEU)
        sorted_methods = sorted(
            method_scores.items(),
            key=lambda x: x[1].get(metric, 0.0),
            reverse=True,
        )

        # Assign ranks with tie handling (average)
        rank = 1
        i = 0
        while i < len(sorted_methods):
            # Find all methods with the same score
            j = i
            while j < len(sorted_methods) and _scores_equal(
                sorted_methods[j][1].get(metric, 0.0),
                sorted_methods[i][1].get(metric, 0.0),
            ):
                j += 1

            # Average rank for tied group
            avg_rank = (rank + rank + (j - i - 1)) / 2.0
            for k in range(i, j):
                method_name = sorted_methods[k][0]
                if method_name not in method_ranks:
                    method_ranks[method_name] = []
                method_ranks[method_name].append(avg_rank)

            rank += (j - i)
            i = j

    # Average across categories
    return {
        method: sum(ranks) / len(ranks)
        for method, ranks in method_ranks.items()
    }


def _scores_equal(a: float, b: float, eps: float = 1e-6) -> bool:
    """Check if two scores are equal within epsilon."""
    return abs(a - b) < eps


# ---------------------------------------------------------------------------
# Per-Category Metric Aggregation
# ---------------------------------------------------------------------------


def compute_per_category_metrics(
    predictions: List[str],
    ground_truths: List[str],
    categories: List[str],
    llm_judge_scores: Optional[List[float]] = None,
) -> Dict[str, Dict[str, float]]:
    """
    Compute QA metrics grouped by question/memory-operation category.

    Args:
        predictions: List of predicted answers.
        ground_truths: List of ground truth answers.
        categories: List of category labels (same length).
        llm_judge_scores: Optional pre-computed LLM judge scores.

    Returns:
        Dict mapping category → {f1, em, judge, rouge_l, count}.
    """
    assert len(predictions) == len(ground_truths) == len(categories)

    # Group by category
    by_cat: Dict[str, list] = {}
    for i, cat in enumerate(categories):
        by_cat.setdefault(cat, []).append(i)

    result = {}
    for cat, indices in by_cat.items():
        cat_preds = [predictions[i] for i in indices]
        cat_gts = [ground_truths[i] for i in indices]
        cat_judge = (
            [llm_judge_scores[i] for i in indices]
            if llm_judge_scores is not None
            else None
        )

        qa = compute_qa_metrics(cat_preds, cat_gts, llm_judge_scores=cat_judge)
        rl_scores = [rouge_l(p, g) for p, g in zip(cat_preds, cat_gts)]
        qa["rouge_l"] = sum(rl_scores) / len(rl_scores) if rl_scores else 0.0
        result[cat] = qa

    return result


# ---------------------------------------------------------------------------
# Internal Transition-Quality Metrics (for test-split per-operation eval)
# ---------------------------------------------------------------------------


def fact_preservation_rate(
    pred_memory: str,
    gold_memory: str,
    client=None,
    model_name: str = "gpt-4o",
) -> float:
    """
    LLM judge: what fraction of facts in the gold M′ are present in the
    predicted M′? Used for internal transition-quality evaluation.

    The LLM extracts atomic facts from the gold memory and checks each one
    against the predicted memory, returning a ratio (0.0 to 1.0).

    Args:
        pred_memory: The model's predicted M′.
        gold_memory: The gold M′ from the dataset.
        client: Optional OpenAI client.
        model_name: Judge model name.

    Returns:
        Fact preservation ratio (0.0 to 1.0).
    """
    if client is None:
        try:
            from openai import OpenAI
            client = OpenAI()
        except Exception:
            return token_f1(pred_memory, gold_memory)

    prompt = (
        f"GOLD MEMORY (reference):\n{gold_memory}\n\n"
        f"PREDICTED MEMORY:\n{pred_memory}\n\n"
        f"List each atomic fact from the GOLD MEMORY. For each fact, indicate "
        f"whether it is preserved in the PREDICTED MEMORY (YES/NO/PARTIAL). "
        f"Then give the final preservation ratio as: RATIO: X/N\n"
        f"where X = number of preserved facts, N = total facts.\n"
        f"Only output the final line: RATIO: X/N"
    )

    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=512,
        )
        text = response.choices[0].message.content.strip()
        # Parse "RATIO: X/N"
        import re as _re
        match = _re.search(r"RATIO:\s*(\d+)\s*/\s*(\d+)", text)
        if match:
            num, den = int(match.group(1)), int(match.group(2))
            return num / den if den > 0 else 0.0
        return 0.0
    except Exception:
        return token_f1(pred_memory, gold_memory)


def transition_judge_score(
    pred_memory: str,
    gold_memory: str,
    current_memory: str = "",
    delta_memory: str = "",
    client=None,
    model_name: str = "gpt-4o",
) -> float:
    """
    LLM judge for transition quality: how well does the predicted M′ match
    the gold M′ semantically?

    Returns 1.0 (correct/semantically equivalent), 0.5 (partial — captures
    some but not all key changes), or 0.0 (incorrect — misses key changes
    or introduces errors).

    Args:
        pred_memory: The model's predicted M′.
        gold_memory: The gold M′ from the dataset.
        current_memory: The original M (for context).
        delta_memory: The new information ΔM (for context).
        client: Optional OpenAI client.
        model_name: Judge model name.

    Returns:
        1.0, 0.5, or 0.0.
    """
    if client is None:
        try:
            from openai import OpenAI
            client = OpenAI()
        except Exception:
            return token_f1(pred_memory, gold_memory)

    # Build context lines only if provided
    context_lines = []
    if current_memory:
        context_lines.append(f"Current memory (M):\n{current_memory}")
    if delta_memory:
        context_lines.append(f"New information (ΔM):\n{delta_memory}")
    context = "\n\n".join(context_lines)

    judge_prompt = (
        f"{context}\n\n"
        f"GOLD UPDATED MEMORY (M′):\n{gold_memory}\n\n"
        f"PREDICTED UPDATED MEMORY (M′):\n{pred_memory}\n\n"
        f"Does the predicted M′ capture the same semantic updates as the gold M′? "
        f"Answer with ONLY one word: "
        f"\"correct\" (semantically equivalent — same key facts, updates, and structure), "
        f"\"partial\" (captures some but not all key changes, or has minor errors), "
        f"or \"incorrect\" (misses key changes, introduces wrong facts, or is contradictory)."
    )

    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": judge_prompt}],
            temperature=0.0,
            max_tokens=10,
        )
        verdict = response.choices[0].message.content.strip().lower()

        if "correct" in verdict and "partial" not in verdict and "incorrect" not in verdict:
            return 1.0
        elif "partial" in verdict:
            return 0.5
        else:
            return 0.0
    except Exception:
        return token_f1(pred_memory, gold_memory)


def compute_transition_metrics(
    predictions: List[str],
    golds: List[str],
    inputs: List[Dict[str, str]],
    categories: List[str],
    client=None,
    model_name: str = "gpt-4o",
) -> Dict[str, Dict[str, float]]:
    """
    Compute comprehensive transition-quality metrics, grouped by operation
    category. This is the core per-operation eval for the headline claim.

    Metrics computed per category:
    - rouge_l: ROUGE-L between predicted and gold M′
    - token_f1: token-level F1
    - fact_preservation: LLM-judge fraction of gold facts preserved
    - transition_judge: LLM-judge semantic match
    - compression fidelity: predicted compression ratio vs gold ratio

    Args:
        predictions: List of predicted M′ strings.
        golds: List of gold M′ strings.
        inputs: List of dicts with keys 'M' and 'delta_M' (the original input).
        categories: List of operation category labels.
        client: Optional OpenAI client.
        model_name: Judge model name.

    Returns:
        Dict mapping category → metrics dict.
    """
    assert len(predictions) == len(golds) == len(inputs) == len(categories)

    # Group indices by category
    by_cat: Dict[str, list] = {}
    for i, cat in enumerate(categories):
        by_cat.setdefault(cat, []).append(i)

    result = {}
    for cat, indices in by_cat.items():
        cat_preds = [predictions[i] for i in indices]
        cat_golds = [golds[i] for i in indices]
        cat_inputs = [inputs[i] for i in indices]

        n = len(cat_preds)

        # ROUGE-L
        rl_scores = [rouge_l(p, g) for p, g in zip(cat_preds, cat_golds)]
        mean_rl = sum(rl_scores) / n if n > 0 else 0.0

        # Token F1
        f1_scores = [token_f1(p, g) for p, g in zip(cat_preds, cat_golds)]
        mean_f1 = sum(f1_scores) / n if n > 0 else 0.0

        # Fact preservation (LLM judge — expensive, sample if needed)
        fp_scores = []
        for p, g in zip(cat_preds, cat_golds):
            fp_scores.append(fact_preservation_rate(p, g, client, model_name))
        mean_fp = sum(fp_scores) / n if n > 0 else 0.0

        # Transition judge (LLM)
        tj_scores = []
        for p, g, inp in zip(cat_preds, cat_golds, cat_inputs):
            tj_scores.append(
                transition_judge_score(
                    p, g,
                    current_memory=inp.get("M", ""),
                    delta_memory=inp.get("delta_M", ""),
                    client=client,
                    model_name=model_name,
                )
            )
        mean_tj = sum(tj_scores) / n if n > 0 else 0.0

        # Compression fidelity
        gold_ratios = []
        pred_ratios = []
        for p, g, inp in zip(cat_preds, cat_golds, cat_inputs):
            orig_tokens = count_tokens(inp.get("M", "")) + count_tokens(inp.get("delta_M", ""))
            if orig_tokens > 0:
                gold_ratios.append(count_tokens(g) / orig_tokens)
                pred_ratios.append(count_tokens(p) / orig_tokens)
        mean_gold_ratio = sum(gold_ratios) / len(gold_ratios) if gold_ratios else 0.0
        mean_pred_ratio = sum(pred_ratios) / len(pred_ratios) if pred_ratios else 0.0

        result[cat] = {
            "rouge_l": mean_rl,
            "token_f1": mean_f1,
            "fact_preservation": mean_fp,
            "transition_judge": mean_tj,
            "gold_compression_ratio": mean_gold_ratio,
            "pred_compression_ratio": mean_pred_ratio,
            "count": n,
        }

    return result


# ---------------------------------------------------------------------------
# LLM-Based Consistency Metrics
# ---------------------------------------------------------------------------


def compute_consistency_metrics_llm(
    memories: List[str],
    client=None,
    model_name: str = "gpt-4o",
    sample_size: Optional[int] = None,
) -> Dict[str, float]:
    """
    LLM-based consistency evaluation — more accurate than the heuristic
    regex-based _detect_contradictions/_detect_redundancy.

    Args:
        memories: List of memory state strings.
        client: Optional OpenAI client.
        model_name: Judge model name.
        sample_size: Optional limit on number of memories to evaluate
                    (LLM calls are expensive; sample for large batches).

    Returns:
        Dict with contradiction_rate, redundancy_rate, total_contradictions,
        total_redundancies, count (sampled count).
    """
    if client is None:
        try:
            from openai import OpenAI
            client = OpenAI()
        except Exception:
            # Fallback to heuristic
            return compute_consistency_metrics(memories)

    if sample_size and sample_size < len(memories):
        import random
        memories = random.Random(42).sample(memories, sample_size)

    contradiction_counts = []
    redundancy_counts = []

    for memory in memories:
        prompt = (
            f"Analyze this memory state for contradictions and redundancy:\n\n"
            f"{memory}\n\n"
            f"1. Contradictions: count how many pairs of facts contradict each other "
            f"(e.g., \"lives in NYC\" and \"lives in Chicago\" without temporal resolution). "
            f"Output: CONTRADICTIONS: N\n"
            f"2. Redundancy: count how many facts are duplicated or express the same "
            f"information in different words. "
            f"Output: REDUNDANCIES: N\n\n"
            f"Only output the two lines above, nothing else."
        )

        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=100,
            )
            text = response.choices[0].message.content.strip()

            import re as _re
            c_match = _re.search(r"CONTRADICTIONS:\s*(\d+)", text)
            r_match = _re.search(r"REDUNDANCIES:\s*(\d+)", text)

            contradiction_counts.append(int(c_match.group(1)) if c_match else 0)
            redundancy_counts.append(int(r_match.group(1)) if r_match else 0)
        except Exception:
            contradiction_counts.append(0)
            redundancy_counts.append(0)

    n = len(memories)
    total_records = sum(count_records(m) for m in memories)

    return {
        "contradiction_rate": (
            sum(contradiction_counts) / total_records if total_records > 0 else 0.0
        ),
        "redundancy_rate": (
            sum(redundancy_counts) / total_records if total_records > 0 else 0.0
        ),
        "total_contradictions": sum(contradiction_counts),
        "total_redundancies": sum(redundancy_counts),
        "count": n,
        "method": "llm",
    }


class CostTracker:
    """
    Track training and inference costs for RQ4 / TABLE.md Table 2.

    Usage:
        tracker = CostTracker()
        tracker.start_training()
        # ... train ...
        tracker.end_training(num_examples=5000)
        tracker.log_inference(latency_ms=42.0)
        print(tracker.summary())
    """

    def __init__(self):
        self.training_start = None
        self.training_end = None
        self.gpu_hours = None
        self.num_training_examples = 0
        self.inference_latencies = []

    def start_training(self):
        """Record training start time."""
        self.training_start = time.time()

    def end_training(self, num_examples: int, gpu_hours: Optional[float] = None):
        """
        Record training end time and cost.

        Args:
            num_examples: Number of training examples seen.
            gpu_hours: Optional manual GPU-hours (if known from scheduler).
                       If not provided, estimated from wall-clock time.
        """
        self.training_end = time.time()
        self.num_training_examples = num_examples

        if gpu_hours is not None:
            self.gpu_hours = gpu_hours
        else:
            # Rough estimate: assume 1 GPU for the wall-clock duration
            wall_hours = (self.training_end - self.training_start) / 3600
            self.gpu_hours = wall_hours

    def log_inference(self, latency_ms: float):
        """Log a single inference latency measurement."""
        self.inference_latencies.append(latency_ms)

    def summary(self) -> Dict[str, float]:
        """Return cost summary."""
        avg_latency = (
            sum(self.inference_latencies) / len(self.inference_latencies)
            if self.inference_latencies
            else 0.0
        )

        return {
            "gpu_hours": self.gpu_hours or 0.0,
            "num_training_examples": self.num_training_examples,
            "avg_inference_latency_ms": avg_latency,
            "num_inference_samples": len(self.inference_latencies),
        }
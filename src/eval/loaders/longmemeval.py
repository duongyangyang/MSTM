"""
LongMemEval benchmark loader.

Loads the LongMemEval benchmark for evaluating long-term memory in chat assistants.

Dataset: xiaowu0162/longmemeval-cleaned on HuggingFace
Paper: "LongMemEval: Benchmarking Chat Assistants on Long-Term Interactive Memory"
       (ICLR 2025)

Each entry contains:
- question_id: unique question identifier
- question: the question text
- answer: ground truth answer
- answer_session_ids: list of session IDs containing evidence
- haystack_sessions: list of all conversation sessions (each session is a list
  of turns with {role, content})

Usage:
    from src.eval.loaders.longmemeval import load_longmemeval
    dataset = load_longmemeval()
"""

import json
from pathlib import Path
from typing import Dict, List, Optional


def _load_from_huggingface() -> List[Dict]:
    """
    Load LongMemEval from HuggingFace datasets library.

    Uses xiaowu0162/longmemeval-cleaned which contains the cleaned
    LongMemEval-S dataset (500 questions, ~48 sessions each).
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError(
            "datasets library is required. Install with: pip install datasets"
        )

    # LongMemEval is stored as a single JSON file on HuggingFace
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(
        repo_id="xiaowu0162/longmemeval-cleaned",
        filename="longmemeval_s_cleaned.json",
        repo_type="dataset",
    )

    return _load_from_local(path)


def _load_from_local(path: str) -> List[Dict]:
    """Load LongMemEval from a local JSON file."""
    local_path = Path(path)
    if not local_path.exists():
        raise FileNotFoundError(f"LongMemEval data not found at: {local_path}")

    with open(local_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, list):
        return data
    elif isinstance(data, dict) and "data" in data:
        return data["data"]
    else:
        raise ValueError(
            f"Unexpected JSON structure in {local_path}. "
            f"Expected a list or dict with 'data' key."
        )


def load_longmemeval(
    local_path: Optional[str] = None,
    max_samples: Optional[int] = None,
) -> List[Dict]:
    """
    Load LongMemEval benchmark data.

    Args:
        local_path: Optional path to local JSON file. If provided, loads from
                    local instead of HuggingFace.
        max_samples: Optional limit on number of samples to load.

    Returns:
        List of dicts, each containing:
        - question_id: str — unique question identifier
        - question: str — the question text
        - answer: str — the ground truth answer
        - answer_session_ids: List[str] — session IDs containing evidence
        - haystack_sessions: List[List[Dict]] — all conversation sessions,
          each session is a list of turns with {role, content}
        - category: str (optional) — question category if available
    """
    if local_path:
        data = _load_from_local(local_path)
    else:
        data = _load_from_huggingface()

    if max_samples:
        data = data[:max_samples]

    return data


def extract_qa_pairs(dataset: List[Dict]) -> List[Dict]:
    """
    Extract (question, answer, evidence) pairs from LongMemEval data.

    Returns a list of dicts with keys: question_id, question, answer,
    answer_session_ids, haystack_sessions.
    """
    qa_pairs = []
    for item in dataset:
        qa_pairs.append({
            "question_id": item.get("question_id", ""),
            "question": item.get("question", ""),
            "answer": item.get("answer", ""),
            "answer_session_ids": item.get("answer_session_ids", []),
            "haystack_sessions": item.get("haystack_sessions", []),
            "category": item.get("category", "unknown"),
        })
    return qa_pairs


def build_memory_from_conversation(
    conversations: List[List[Dict]],
    method: callable = None,
    session_ids: Optional[List[str]] = None,
) -> str:
    """
    Build a memory state from conversation sessions by applying a memory
    transition method session-by-session.

    This simulates the process of an agent accumulating memory over time.

    Args:
        conversations: List of sessions, each session is a list of turns
                       with {role, content}.
        method: Memory transition function with signature
                transition(M, delta_M) -> M_prime.
                If None, uses pass-through (Static Memory).
        session_ids: Optional list of session IDs (for logging).

    Returns:
        Final memory state as a string.
    """
    if method is None:
        from src.baselines.static_memory import transition as static_transition
        method = static_transition

    memory = ""
    for i, session in enumerate(conversations):
        # Extract user messages from this session as delta_M
        user_messages = []
        for turn in session:
            if turn.get("role") == "user":
                user_messages.append(turn["content"])
        if not user_messages:
            continue

        sid = session_ids[i] if session_ids else f"session_{i}"
        delta = f"[{sid}]\n" + "\n".join(user_messages)
        memory = method(memory, delta)

    return memory


def build_memory_batched(
    conversations_list: List[List[List[Dict]]],
    method,
    batch_size: int = 16,
    verbose: bool = True,
) -> List[str]:
    """
    Build memory states for multiple conversations in parallel by batching
    transitions at each session step across conversations.

    Args:
        conversations_list: List of conversations, each conversation is a
                           list of sessions, each session is a list of turns
                           with {role, content}.
        method: An MSTMInference instance (must have transition_batch method)
                or a callable with transition(M, delta_M) -> M_prime.
        batch_size: Max samples per forward pass.
        verbose: Print progress.

    Returns:
        List of final memory state strings, one per conversation.
    """
    n_conversations = len(conversations_list)

    # Extract user messages per session per conversation
    conv_deltas = []
    for sessions in conversations_list:
        deltas = []
        for session in sessions:
            user_messages = [
                turn["content"]
                for turn in session
                if turn.get("role") == "user"
            ]
            if user_messages:
                deltas.append("\n".join(user_messages))
        conv_deltas.append(deltas)

    max_sessions = max(len(d) for d in conv_deltas) if conv_deltas else 0
    memories = ["" for _ in range(n_conversations)]
    has_batch = hasattr(method, "transition_batch")

    from tqdm import tqdm
    pbar = tqdm(total=max_sessions, desc="  Building memory", unit="sess") if verbose else None

    for step in range(max_sessions):
        active_indices = []
        active_memories = []
        active_deltas = []

        for conv_idx in range(n_conversations):
            if step < len(conv_deltas[conv_idx]):
                active_indices.append(conv_idx)
                active_memories.append(memories[conv_idx])
                active_deltas.append(conv_deltas[conv_idx][step])

        if not active_indices:
            if pbar:
                pbar.update(1)
            continue

        if pbar:
            pbar.set_postfix({"conv": len(active_indices)})

        if has_batch:
            new_memories = method.transition_batch(
                active_memories,
                active_deltas,
                batch_size=batch_size,
            )
        else:
            new_memories = [
                method(m, d)
                for m, d in zip(active_memories, active_deltas)
            ]

        for i, conv_idx in enumerate(active_indices):
            memories[conv_idx] = new_memories[i]

        if pbar:
            pbar.update(1)

    if pbar:
        pbar.close()

    return memories


def build_full_dialogue(conversations: List[List[Dict]]) -> str:
    """
    Build the full conversation dialogue as a flat string (all turns,
    user + assistant, interleaved). Used for the full-context baseline
    (A-MEM's "LoCoMo" baseline — no memory system, feeds raw dialogue
    to the answer model).

    Args:
        conversations: List of sessions, each session is a list of turns
                       with {role, content}.

    Returns:
        Full dialogue text.
    """
    lines = []
    for session in conversations:
        for turn in session:
            role = turn.get("role", "unknown")
            content = turn.get("content", "")
            if content.strip():
                lines.append(f"[{role}]: {content}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Schema reference for LongMemEval (xiaowu0162/longmemeval-cleaned):
#
# Each entry:
# {
#   "question_id": str,
#   "question": str,
#   "answer": str,
#   "answer_session_ids": [str],
#   "haystack_sessions": [
#     [
#       {"role": "user"|"assistant", "content": str, "has_answer": bool|null}
#     ]
#   ],
#   "haystack_session_ids": [str],
#   "category": str (optional)
# }
#
# Note: LongMemEval-S has ~500 questions with ~48 sessions each (~115K tokens).
# ---------------------------------------------------------------------------
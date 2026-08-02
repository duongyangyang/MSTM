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
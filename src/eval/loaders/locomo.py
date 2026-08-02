"""
LoCoMo benchmark loader.

Loads the LoCoMo (Long Conversation Memory) benchmark for evaluating
long-term conversational memory.

Dataset: Percena/locomo-mc10 on HuggingFace
Paper: "LoCoMo: Long-Context Conversation Memory Benchmark"

Each entry contains:
- haystack_sessions: list of conversation sessions, each with turns
- questions with ground truth answers
- categories: single-hop, multi-hop, temporal, open-domain, adversarial

Usage:
    from src.eval.loaders.locomo import load_locomo
    dataset = load_locomo()
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple


def _load_from_huggingface(split: str = "test") -> List[Dict]:
    """
    Load LoCoMo from HuggingFace.

    The Percena/locomo-mc10 dataset has a complex nested schema that the
    datasets library sometimes fails to cast. We download the raw JSON file
    directly via huggingface_hub and parse it manually for reliability.
    """
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        raise ImportError(
            "huggingface_hub is required. Install with: pip install huggingface_hub"
        )

    # Download the transformed mc10 JSONL file directly
    path = hf_hub_download(
        repo_id="Percena/locomo-mc10",
        filename="transformed/locomo_mc10_with_name.json",
        repo_type="dataset",
    )

    # The file is JSONL format (one JSON object per line)
    with open(path, "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f if line.strip()]

    return data


def _load_from_local(path: str) -> List[Dict]:
    """Load LoCoMo from a local JSON/JSONL file."""
    local_path = Path(path)
    if not local_path.exists():
        raise FileNotFoundError(f"LoCoMo data not found at: {local_path}")

    with open(local_path, "r", encoding="utf-8") as f:
        if local_path.suffix == ".jsonl":
            return [json.loads(line) for line in f if line.strip()]
        else:
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


def load_locomo(
    split: str = "test",
    local_path: Optional[str] = None,
    max_samples: Optional[int] = None,
) -> List[Dict]:
    """
    Load LoCoMo benchmark data.

    Args:
        split: Dataset split to load ('train', 'test', 'validation').
        local_path: Optional path to local JSON/JSONL file. If provided,
                    loads from local instead of HuggingFace.
        max_samples: Optional limit on number of samples to load.

    Returns:
        List of dicts, each containing:
        - question: str — the question text
        - ground_truth: str — the correct answer
        - category: str — question category (single-hop, multi-hop, temporal,
                      open-domain, adversarial)
        - haystack_sessions: List[List[Dict]] — conversation sessions,
          each session is a list of turns with {role, content}
        - num_sessions: int — number of sessions
        - haystack_session_summaries: List[str] — session summaries
        - haystack_session_datetimes: List[str] — session timestamps
    """
    if local_path:
        data = _load_from_local(local_path)
    else:
        data = _load_from_huggingface(split)

    if max_samples:
        data = data[:max_samples]

    return data


def extract_qa_pairs(dataset: List[Dict]) -> List[Dict]:
    """
    Extract (question, answer, category) pairs from LoCoMo data.

    Handles both the mc10 format (Percena/locomo-mc10) and the original
    LoCoMo format (snap-research/locomo).

    Returns a list of dicts with keys: question, ground_truth, category,
    session_ids, conversation_context.
    """
    qa_pairs = []
    for item in dataset:
        # mc10 format: question, answer, question_type, haystack_sessions, choices
        question = item.get("question", "")
        # In mc10 format, 'answer' is the correct answer string
        ground_truth = item.get("answer", item.get("ground_truth", ""))
        # mc10 uses 'question_type', original uses 'category'
        category = item.get("question_type", item.get("category", "unknown"))
        # mc10 uses 'haystack_sessions', original uses 'conversation' or 'sessions'
        conversation = (
            item.get("haystack_sessions")
            or item.get("conversation")
            or item.get("sessions")
            or []
        )
        session_summaries = item.get("haystack_session_summaries", [])
        session_datetimes = item.get("haystack_session_datetimes", [])
        num_sessions = item.get("num_sessions") or len(conversation)

        qa_pairs.append({
            "question": question,
            "ground_truth": ground_truth,
            "category": category,
            "conversation": conversation,
            "session_summaries": session_summaries,
            "session_datetimes": session_datetimes,
            "num_sessions": num_sessions,
        })
    return qa_pairs


def build_memory_from_conversation(
    conversations: List[List[Dict]],
    method: callable = None,
) -> str:
    """
    Build a memory state from conversation sessions by applying a memory
    transition method session-by-session.

    This simulates the process of an agent accumulating memory over time
    as it converses with the user across multiple sessions.

    Args:
        conversations: List of sessions, each session is a list of turns
                       with {role, content}.
        method: Memory transition function with signature
                transition(M, delta_M) -> M_prime.
                If None, uses pass-through (Static Memory).

    Returns:
        Final memory state as a string.
    """
    if method is None:
        from src.baselines.static_memory import transition as static_transition
        method = static_transition

    memory = ""
    for session in conversations:
        # Extract user messages from this session as delta_M
        user_messages = [
            turn["content"]
            for turn in session
            if turn.get("role") == "user"
        ]
        if not user_messages:
            continue

        delta = "\n".join(user_messages)
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

    This is the key optimization for GPU utilization: instead of calling
    transition() once per session per conversation (GPU mostly idle with
    a small model), we batch all conversations' session-i transitions
    into a single forward pass.

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
    # conv_deltas[conv_idx] = [delta_session_0, delta_session_1, ...]
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

    # Find max number of sessions across all conversations
    max_sessions = max(len(d) for d in conv_deltas) if conv_deltas else 0

    # Initialize memories as empty strings
    memories = ["" for _ in range(n_conversations)]

    has_batch = hasattr(method, "transition_batch")

    for step in range(max_sessions):
        # Collect conversations that have this session
        active_indices = []
        active_memories = []
        active_deltas = []

        for conv_idx in range(n_conversations):
            if step < len(conv_deltas[conv_idx]):
                active_indices.append(conv_idx)
                active_memories.append(memories[conv_idx])
                active_deltas.append(conv_deltas[conv_idx][step])

        if not active_indices:
            continue

        if verbose and (step == 0 or step == max_sessions - 1 or (step + 1) % 10 == 0):
            print(f"  Session {step+1}/{max_sessions}: "
                  f"processing {len(active_indices)} conversations in batch")

        if has_batch:
            # Batched generation — all active conversations in one pass
            new_memories = method.transition_batch(
                active_memories,
                active_deltas,
                batch_size=batch_size,
            )
        else:
            # Fallback: sequential (for non-MSTM methods)
            new_memories = [
                method(m, d)
                for m, d in zip(active_memories, active_deltas)
            ]

        # Update memories
        for i, conv_idx in enumerate(active_indices):
            memories[conv_idx] = new_memories[i]

    return memories


def build_full_dialogue(conversations: List[List[Dict]]) -> str:
    """
    Build the full conversation dialogue as a flat string (all turns,
    user + assistant, interleaved). Used for the LoCoMo full-context
    baseline (A-MEM's "LoCoMo" baseline — no memory system, feeds raw
    dialogue to the answer model).

    Args:
        conversations: List of sessions, each session is a list of turns
                       with {role, content}.

    Returns:
        Full dialogue text.
    """
    lines = []
    for session_idx, session in enumerate(conversations):
        for turn in session:
            role = turn.get("role", "unknown")
            content = turn.get("content", "")
            if content.strip():
                lines.append(f"[{role}]: {content}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Schema reference for the original LoCoMo dataset (snap-research/locomo):
#
# Each conversation:
# {
#   "conversation_id": str,
#   "persona": {...},
#   "event_graph": {...},
#   "sessions": [
#     {
#       "session_id": str,
#       "datetime": str,
#       "summary": str,
#       "dialogue": [
#         {"speaker": "user"|"assistant", "text": str, "image": str|null}
#       ]
#     }
#   ],
#   "qa_pairs": [
#     {
#       "question": str,
#       "answer": str,
#       "category": "single-hop"|"multi-hop"|"temporal"|"open-domain"|"adversarial",
#       "evidence_sessions": [str]
#     }
#   ]
# }
# ---------------------------------------------------------------------------
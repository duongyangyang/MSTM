"""
Baseline 1 — Static Memory.

Pure pass-through: stores memory records without modification.
All new information is appended; nothing is ever updated, merged, or deleted.
Used as the pipeline smoke test — see TODOLIST.md Phase 3, Task 7.
"""

from typing import List


def transition(
    M: str,
    delta_M: str,
    **kwargs,
) -> str:
    """
    No-op transition: new information is simply appended, nothing is rewritten.

    Args:
        M: Current memory state as a string.
        delta_M: New information to incorporate.
        **kwargs: Ignored (for interface compatibility).

    Returns:
        M_prime: The evolved memory state (M + delta_M concatenated).
    """
    if not M.strip():
        return delta_M.strip()
    if not delta_M.strip():
        return M.strip()

    # Simple concatenation: append new information with a separator
    separator = "\n" if M.endswith("\n") else "\n"
    return M.rstrip() + separator + delta_M.strip()


def transition_batch(
    memories: List[str],
    deltas: List[str],
    **kwargs,
) -> List[str]:
    """
    Batch version of transition for efficiency.

    Args:
        memories: List of current memory states.
        deltas: List of new information (same length as memories).
        **kwargs: Ignored.

    Returns:
        List of evolved memory states.
    """
    assert len(memories) == len(deltas), (
        f"Length mismatch: {len(memories)} memories vs {len(deltas)} deltas"
    )
    return [transition(m, d) for m, d in zip(memories, deltas)]


# ---------------------------------------------------------------------------
# For the eval harness: a method is callable as method.transition(M, delta_M)
# and returns M_prime. This file is importable as a module.
# ---------------------------------------------------------------------------
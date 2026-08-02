"""
Baseline 3 — Heuristic Consolidation.

Merges memory records using an embedding-similarity threshold.
Records that are too similar are merged into a single consolidated record.
See TODOLIST.md Phase 3, Task 8.

Uses sentence-transformers for embedding generation.
If sentence-transformers is not available, falls back to a simple
TF-IDF + cosine similarity approach via sklearn.
"""

import re
from typing import List, Optional, Tuple


def _get_embedder():
    """Try to get a sentence-transformers model; fall back to sklearn TF-IDF."""
    try:
        from sentence_transformers import SentenceTransformer

        # Use a small, fast model suitable for local use
        model = SentenceTransformer("all-MiniLM-L6-v2")
        return "sbert", model
    except ImportError:
        pass

    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity

        # TF-IDF fallback: we wrap it in a callable interface matching SentenceTransformer
        class TfidfEmbedder:
            def __init__(self):
                self.vectorizer = TfidfVectorizer(stop_words="english")

            def encode(self, sentences, **kwargs):
                """Return TF-IDF matrix as dense array."""
                return self.vectorizer.fit_transform(sentences).toarray()

        return "tfidf", TfidfEmbedder()
    except ImportError:
        raise ImportError(
            "Either sentence-transformers or scikit-learn is required. "
            "Install with: pip install sentence-transformers  OR  pip install scikit-learn"
        )


# Global embedder instance (lazy-loaded)
_EMBEDDER = None
_EMBEDDER_TYPE = None


def _load_embedder():
    global _EMBEDDER, _EMBEDDER_TYPE
    if _EMBEDDER is None:
        _EMBEDDER_TYPE, _EMBEDDER = _get_embedder()
    return _EMBEDDER_TYPE, _EMBEDDER


def _parse_records(M: str) -> List[str]:
    """Parse memory string into individual records (bullet points)."""
    records = []
    for line in M.split("\n"):
        stripped = line.strip()
        if stripped:
            # Strip common bullet markers for cleaner comparison
            cleaned = re.sub(r"^[-*•]\s*", "", stripped)
            records.append(cleaned)
    return records


def _merge_records(r1: str, r2: str) -> str:
    """
    Merge two similar records into one.
    Strategy: keep the longer/more detailed record as the base.
    """
    # If one is clearly a superset of the other, keep the longer one
    if r1.lower() in r2.lower():
        return r2
    if r2.lower() in r1.lower():
        return r1

    # Otherwise, keep the longer one (assumed more detailed)
    return r1 if len(r1) >= len(r2) else r2


def transition(
    M: str,
    delta_M: str,
    similarity_threshold: float = 0.85,
    **kwargs,
) -> str:
    """
    Consolidate similar memory records, then append new information.

    Args:
        M: Current memory state as a string.
        delta_M: New information to incorporate.
        similarity_threshold: Cosine similarity threshold above which records
                             are considered duplicates (0.0–1.0). Default 0.85.
        **kwargs: Additional kwargs (ignored).

    Returns:
        M_prime: The evolved memory state with similar records merged.
    """
    embedder_type, embedder = _load_embedder()

    # Parse existing records
    records = _parse_records(M)

    if len(records) <= 1:
        # Nothing to merge
        result = M.rstrip()
        if delta_M.strip():
            result += "\n" + delta_M.strip()
        return result

    # Compute embeddings
    embeddings = embedder.encode(records)

    # Find and merge similar pairs
    merged_indices = set()
    final_records = []

    for i in range(len(records)):
        if i in merged_indices:
            continue

        current_record = records[i]
        current_emb = embeddings[i]

        for j in range(i + 1, len(records)):
            if j in merged_indices:
                continue

            # Compute cosine similarity
            sim = float(
                (current_emb @ embeddings[j].T)
                / (max(float(current_emb @ current_emb.T) ** 0.5, 1e-8)
                   * max(float(embeddings[j] @ embeddings[j].T) ** 0.5, 1e-8))
            )

            if sim >= similarity_threshold:
                # Merge records
                current_record = _merge_records(current_record, records[j])
                merged_indices.add(j)

        final_records.append(current_record)

    # Reconstruct memory with bullet points
    result = "\n".join(f"- {r}" for r in final_records)

    # Append new information
    if delta_M.strip():
        result += "\n" + delta_M.strip()

    return result


def transition_batch(
    memories: List[str],
    deltas: List[str],
    similarity_threshold: float = 0.85,
    **kwargs,
) -> List[str]:
    """Batch version of transition."""
    assert len(memories) == len(deltas)
    return [
        transition(m, d, similarity_threshold=similarity_threshold)
        for m, d in zip(memories, deltas)
    ]
"""
Memory Retriever — A-MEM-style retrieval for the evaluation pipeline.

Chunks memory text into retrievable units, embeds them with all-MiniLM-L6-v2,
and retrieves top-k relevant chunks for a given query.

This is the retrieval component required by the A-MEM evaluation pipeline:
Conversation → Memory System → chunk+embed → Retrieve Top-K → LLM Answer → Eval

Usage:
    from src.eval.retriever import MemoryRetriever

    retriever = MemoryRetriever()
    retriever.index(memory_text)
    chunks = retriever.retrieve("What is the user's job?", top_k=10)
    num_tokens = retriever.context_token_length()  # for efficiency metric
"""

from __future__ import annotations

import re
from typing import List, Optional, Tuple

import numpy as np


class MemoryRetriever:
    """
    A-MEM Style Memory Retrieval using all-MiniLM-L6-v2 embeddings.

    Splits memory into bullet-point chunks, embeds them, and retrieves
    the top-k most relevant chunks for a query via cosine similarity.

    Attributes:
        embed_model: The sentence-transformers model name.
        embedder: Loaded SentenceTransformer instance (lazy-loaded on first use).
        chunks: List of text chunks from the indexed memory.
        embeddings: numpy array of chunk embeddings (shape: [n_chunks, 384]).
        retrieved_context: The last retrieved context (joined chunks string).
    """

    def __init__(
        self,
        embed_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        embedder: object = None,  # Allow sharing an existing embedder
    ):
        """
        Initialize the MemoryRetriever.

        Args:
            embed_model: Name of the sentence-transformers model.
            embedder: Optional pre-loaded SentenceTransformer instance.
                     If provided, embed_model is ignored and the shared
                     instance is used (avoids loading two copies).
        """
        self.embed_model_name = embed_model
        self._embedder = embedder  # lazy-loaded if None
        self.chunks: List[str] = []
        self.embeddings: Optional[np.ndarray] = None
        self._retrieved_chunks: List[str] = []

    @property
    def embedder(self):
        """Lazy-load the embedding model (shared instance if provided)."""
        if self._embedder is None:
            from sentence_transformers import SentenceTransformer
            self._embedder = SentenceTransformer(self.embed_model_name)
        return self._embedder

    # ------------------------------------------------------------------
    # Indexing
    # ------------------------------------------------------------------

    def index(self, memory_text: str) -> None:
        """
        Chunk and embed the memory text.

        Chunking strategy (A-MEM analog):
        1. Split by newlines → each bullet/paragraph is a candidate chunk.
        2. Filter empty/whitespace-only chunks.
        3. Guard: sub-split any chunk >128 words on sentence boundaries
           (periods, exclamation, question marks followed by space).

        Args:
            memory_text: The full memory state string (M′).
        """
        self.chunks = self._chunk_text(memory_text)
        if not self.chunks:
            self.embeddings = np.empty((0, 384))
            return

        self.embeddings = self.embedder.encode(
            self.chunks,
            convert_to_numpy=True,
            show_progress_bar=False,
        )

    def _chunk_text(self, text: str) -> List[str]:
        """Split text into retrievable chunks."""
        if not text or not text.strip():
            return []

        # Split by newlines (bullet points / paragraphs)
        raw_chunks = [line.strip() for line in text.split("\n") if line.strip()]

        # Guard: sub-split oversized chunks on sentence boundaries
        MAX_WORDS = 128
        final_chunks = []
        for chunk in raw_chunks:
            words = chunk.split()
            if len(words) <= MAX_WORDS:
                final_chunks.append(chunk)
            else:
                # Split on sentence boundaries (. ! ? followed by space)
                sentences = re.split(r"(?<=[.!?])\s+", chunk)
                current = ""
                for sent in sentences:
                    if len(current.split()) + len(sent.split()) <= MAX_WORDS:
                        current = (current + " " + sent).strip()
                    else:
                        if current:
                            final_chunks.append(current)
                        current = sent
                if current:
                    final_chunks.append(current)

        return final_chunks

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def retrieve(self, query: str, top_k: int = 10) -> List[str]:
        """
        Retrieve the top-k most relevant chunks for a query.

        Uses cosine similarity between the query embedding and chunk embeddings.

        Args:
            query: The question text.
            top_k: Number of chunks to retrieve.

        Returns:
            List of text chunks, ordered by decreasing relevance.
        """
        if not self.chunks or self.embeddings is None or len(self.embeddings) == 0:
            return []

        # Embed the query
        query_embedding = self.embedder.encode(
            [query],
            convert_to_numpy=True,
            show_progress_bar=False,
        )  # shape: (1, 384)

        # Cosine similarity
        similarities = self._cosine_similarity(query_embedding, self.embeddings)

        # Top-k
        k = min(top_k, len(self.chunks))
        top_indices = np.argsort(similarities)[-k:][::-1]

        self._retrieved_chunks = [self.chunks[i] for i in top_indices]
        return self._retrieved_chunks

    @staticmethod
    def _cosine_similarity(
        query_emb: np.ndarray,
        chunk_embs: np.ndarray,
    ) -> np.ndarray:
        """Compute cosine similarity between query and chunk embeddings."""
        query_norm = query_emb / (np.linalg.norm(query_emb, axis=1, keepdims=True) + 1e-8)
        chunk_norms = chunk_embs / (np.linalg.norm(chunk_embs, axis=1, keepdims=True) + 1e-8)
        return (query_norm @ chunk_norms.T).flatten()

    # ------------------------------------------------------------------
    # Efficiency
    # ------------------------------------------------------------------

    def context_token_length(self, model: str = "gpt-4o-mini") -> int:
        """
        Compute the token length of the retrieved context.

        Uses tiktoken for accurate token counting (matching A-MEM's
        Token Length metric definition: full-prompt tokens).

        Args:
            model: Model name for tiktoken encoding (default: gpt-4o-mini).

        Returns:
            Number of tokens in the concatenated retrieved chunks.
        """
        try:
            import tiktoken
        except ImportError:
            # Fallback: whitespace estimate
            context = " ".join(self._retrieved_chunks)
            return len(context.split())

        try:
            enc = tiktoken.encoding_for_model(model)
        except KeyError:
            enc = tiktoken.get_encoding("cl100k_base")

        context = " ".join(self._retrieved_chunks)
        return len(enc.encode(context))

    # ------------------------------------------------------------------
    # Info
    # ------------------------------------------------------------------

    @property
    def chunk_count(self) -> int:
        """Number of chunks in the current index."""
        return len(self.chunks)

    @property
    def retrieved_context(self) -> str:
        """The last retrieved context as a single string (joined chunks)."""
        return " ".join(self._retrieved_chunks)
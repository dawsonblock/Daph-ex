from __future__ import annotations

import hashlib
import math
from typing import Callable, Sequence

from hrm_adaptive_memory.memory.chunking import Chunk
from .lexical import tokenize


class HashingEmbedder:
    """Dependency-free deterministic baseline, not a scientific dense encoder."""
    def __init__(self, dimensions: int = 256):
        self.dimensions = dimensions

    def __call__(self, text: str) -> list[float]:
        values = [0.0] * self.dimensions
        for token in tokenize(text):
            digest = hashlib.blake2b(token.encode(), digest_size=8).digest()
            index = int.from_bytes(digest[:4], "big") % self.dimensions
            sign = 1.0 if digest[4] & 1 else -1.0
            values[index] += sign
        norm = math.sqrt(sum(value * value for value in values)) or 1.0
        return [value / norm for value in values]


def cosine(left: Sequence[float], right: Sequence[float]) -> float:
    return sum(a * b for a, b in zip(left, right))


class DenseRetriever:
    def __init__(self, chunks: Sequence[Chunk], embedder: Callable[[str], Sequence[float]] | None = None):
        self.chunks = list(chunks); self.embedder = embedder or HashingEmbedder()
        self.embeddings = [list(self.embedder(chunk.content)) for chunk in self.chunks]

    def search(self, query: str, top_k: int = 30) -> list[tuple[Chunk, float]]:
        query_embedding = self.embedder(query)
        scored = [(chunk, cosine(query_embedding, embedding)) for chunk, embedding in zip(self.chunks, self.embeddings)]
        return sorted(scored, key=lambda item: (-item[1], item[0].chunk_id))[:top_k]

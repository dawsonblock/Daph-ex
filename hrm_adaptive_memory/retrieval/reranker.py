from __future__ import annotations

from typing import Protocol, Sequence

from hrm_adaptive_memory.memory.chunking import Chunk
from .lexical import tokenize


class Reranker(Protocol):
    def score(self, query: str, candidate: Chunk) -> float: ...


class LexicalOverlapReranker:
    """Cheap baseline control; replace with a cross-encoder for real runs."""
    def score(self, query: str, candidate: Chunk) -> float:
        query_terms, candidate_terms = set(tokenize(query)), set(tokenize(candidate.content))
        return len(query_terms & candidate_terms) / max(1, len(query_terms))


def rerank(query: str, chunks: Sequence[Chunk], scorer: Reranker, top_k: int) -> list[tuple[Chunk, float]]:
    scored = [(chunk, float(scorer.score(query, chunk))) for chunk in chunks]
    return sorted(scored, key=lambda item: (-item[1], item[0].chunk_id))[:top_k]

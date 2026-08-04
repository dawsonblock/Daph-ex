from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from hrm_adaptive_memory.memory.chunking import Chunk
from .dense import DenseRetriever
from .lexical import BM25Retriever
from .reranker import Reranker, rerank


@dataclass(frozen=True)
class RetrievalCandidate:
    chunk: Chunk
    rrf_score: float
    reranker_score: float | None = None
    dense_rank: int | None = None
    lexical_rank: int | None = None


class HybridRetriever:
    def __init__(self, chunks: Sequence[Chunk], *, dense: DenseRetriever | None = None,
                 lexical: BM25Retriever | None = None, rrf_k: int = 60):
        self.chunks = list(chunks); self.dense = dense or DenseRetriever(self.chunks)
        self.lexical = lexical or BM25Retriever(self.chunks); self.rrf_k = rrf_k

    def search(self, query: str, *, candidate_k: int = 30, reranker: Reranker | None = None,
               final_k: int = 10) -> list[RetrievalCandidate]:
        dense = self.dense.search(query, candidate_k); lexical = self.lexical.search(query, candidate_k)
        rows: dict[str, dict[str, object]] = {}
        for source, results in (("dense", dense), ("lexical", lexical)):
            for rank, (chunk, _score) in enumerate(results, 1):
                row = rows.setdefault(chunk.chunk_id, {"chunk": chunk, "score": 0.0})
                row["score"] = float(row["score"]) + 1.0 / (self.rrf_k + rank)
                row[f"{source}_rank"] = rank
        fused = sorted(rows.values(), key=lambda row: (-float(row["score"]), row["chunk"].chunk_id))
        if reranker is not None:
            reranked = rerank(query, [row["chunk"] for row in fused], reranker, final_k)
            by_id = {chunk.chunk_id: score for chunk, score in reranked}
            fused = [row for row in fused if row["chunk"].chunk_id in by_id]
            fused.sort(key=lambda row: (-by_id[row["chunk"].chunk_id], -float(row["score"])))
        else:
            fused = fused[:final_k]
            by_id = {}
        return [RetrievalCandidate(
            chunk=row["chunk"], rrf_score=float(row["score"]),
            reranker_score=by_id.get(row["chunk"].chunk_id),
            dense_rank=row.get("dense_rank"), lexical_rank=row.get("lexical_rank"),
        ) for row in fused]

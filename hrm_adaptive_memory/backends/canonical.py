"""Canonical Gate B retrieval arms behind the common async backend contract.

Arms:
  bm25            lexical baseline
  hash            dependency-free hashing-vector negative control
  dense           revision-pinned transformer encoder
  hybrid_score    normalized score fusion (BM25 + dense)
  hybrid_rrf      reciprocal rank fusion (BM25 + dense)
  hybrid_rerank   RRF candidates re-scored by lexical overlap

External engines (TurboVec, RuVector) remain adapters and are qualified only
at Stage 8, against these baselines.
"""

from __future__ import annotations

import hashlib
import json
import time
from enum import Enum
from typing import Any, Sequence

from ..contracts import (
    BackendCapabilities,
    BackendHealth,
    BackendState,
    EvidenceFilter,
    IndexReceipt,
    IndexRecord,
    RetrievalReceipt,
    RetrievalResult,
    RetrievedEvidence,
    sha256_text,
)
from ..memory.chunking import Chunk
from ..retrieval.dense import HashingEmbedder, cosine
from ..retrieval.embedding import EmbeddingSpec, PinnedTransformerEmbedder
from ..retrieval.lexical import BM25Retriever
from ..retrieval.reranker import LexicalOverlapReranker


class CanonicalRetrievalMode(str, Enum):
    BM25 = "bm25"
    HASH = "hash"
    DENSE = "dense"
    HYBRID_SCORE = "hybrid_score"
    HYBRID_RRF = "hybrid_rrf"
    HYBRID_RERANK = "hybrid_rerank"
    DENSE_BGE = "dense_bge"


_DENSE_MODES = {
    CanonicalRetrievalMode.DENSE,
    CanonicalRetrievalMode.DENSE_BGE,
    CanonicalRetrievalMode.HYBRID_SCORE,
    CanonicalRetrievalMode.HYBRID_RRF,
    CanonicalRetrievalMode.HYBRID_RERANK,
}
_SPARSE_MODES = {
    CanonicalRetrievalMode.BM25,
    CanonicalRetrievalMode.HYBRID_SCORE,
    CanonicalRetrievalMode.HYBRID_RRF,
    CanonicalRetrievalMode.HYBRID_RERANK,
}


def _min_max(values: dict[str, float]) -> dict[str, float]:
    """Scale scores into [0, 1]; an all-equal set maps to 0 (no signal)."""

    if not values:
        return {}
    low, high = min(values.values()), max(values.values())
    if high - low < 1e-12:
        return {key: 0.0 for key in values}
    return {key: (value - low) / (high - low) for key, value in values.items()}


class CanonicalRetrievalBackend:
    """One implementation, six pinned arms, identical receipts."""

    backend_version = "canonical-v1"

    def __init__(
        self, mode: CanonicalRetrievalMode, records: Sequence[IndexRecord] = (), *,
        embedder: Any = None, embedding_spec: EmbeddingSpec | None = None,
        rrf_k: int = 60, candidate_k: int = 50, dense_weight: float = 0.5,
    ):
        self.mode = CanonicalRetrievalMode(mode)
        self.backend_id = f"canonical:{self.mode.value}"
        self.rrf_k = rrf_k
        self.candidate_k = candidate_k
        self.dense_weight = dense_weight
        self.embedding_spec = embedding_spec
        self._embedder = embedder
        if self.mode in _DENSE_MODES and self._embedder is None:
            from ..retrieval.embedding import BGE_SMALL
            default = (EmbeddingSpec(**BGE_SMALL)
                       if self.mode == CanonicalRetrievalMode.DENSE_BGE else EmbeddingSpec())
            self._embedder = PinnedTransformerEmbedder(embedding_spec or default)
            self.embedding_spec = self._embedder.spec
        elif self.mode == CanonicalRetrievalMode.HASH and self._embedder is None:
            self._embedder = HashingEmbedder()
        self._records: dict[str, IndexRecord] = {}
        self._chunks: list[Chunk] = []
        self._by_id: dict[str, Chunk] = {}
        self._lexical: BM25Retriever | None = None
        self._vectors: dict[str, Sequence[float]] = {}
        self._reranker = LexicalOverlapReranker()
        if records:
            self._install(records)

    # ---- indexing -------------------------------------------------------

    def _install(self, records: Sequence[IndexRecord]) -> None:
        for record in records:
            previous = self._records.get(record.evidence_id)
            if previous is not None and previous != record:
                raise ValueError(f"Evidence ID {record.evidence_id!r} is immutable")
            self._records[record.evidence_id] = record
        self._chunks = [
            Chunk(
                chunk_id=row.evidence_id, source_id=row.source_id,
                source_type=row.source_type,
                title=str(row.metadata.get("title", row.source_id)),
                section=str(row.metadata.get("section", "")),
                content=row.content, token_count=row.token_count, metadata=row.metadata,
            )
            for row in sorted(self._records.values(), key=lambda value: value.evidence_id)
        ]
        self._by_id = {chunk.chunk_id: chunk for chunk in self._chunks}
        if self.mode in _SPARSE_MODES:
            self._lexical = BM25Retriever(self._chunks)
        if self.mode in _DENSE_MODES:
            contents = [chunk.content for chunk in self._chunks]
            vectors = self._embedder.embed_documents(contents)
            self._vectors = {
                chunk.chunk_id: vector for chunk, vector in zip(self._chunks, vectors)
            }
        elif self.mode == CanonicalRetrievalMode.HASH:
            self._vectors = {
                chunk.chunk_id: self._embedder(chunk.content) for chunk in self._chunks
            }

    async def index(self, records: Sequence[IndexRecord]) -> IndexReceipt:
        self._install(records)
        canonical = json.dumps([
            (row.evidence_id, row.source_id, sha256_text(row.content))
            for row in sorted(records, key=lambda value: value.evidence_id)
        ], separators=(",", ":"))
        return IndexReceipt(
            backend_id=self.backend_id,
            indexed_ids=tuple(sorted(row.evidence_id for row in records)),
            source_digest=hashlib.sha256(canonical.encode()).hexdigest(),
        )

    async def health(self) -> BackendHealth:
        return BackendHealth(self.backend_id, BackendState.HEALTHY, self.backend_version)

    async def capabilities(self) -> BackendCapabilities:
        return BackendCapabilities(
            dense=self.mode in _DENSE_MODES or self.mode == CanonicalRetrievalMode.HASH,
            sparse=self.mode in _SPARSE_MODES,
            batch_indexing=True,
        )

    def config_digest(self) -> str:
        payload = {
            "backend_id": self.backend_id,
            "backend_version": self.backend_version,
            "rrf_k": self.rrf_k,
            "candidate_k": self.candidate_k,
            "dense_weight": self.dense_weight,
            "embedding": None if self.embedding_spec is None else self.embedding_spec.digest(),
        }
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()

    # ---- scoring --------------------------------------------------------

    def _dense_scores(self, query: str) -> dict[str, float]:
        if self.mode == CanonicalRetrievalMode.HASH:
            query_vector = self._embedder(query)
        else:
            query_vector = self._embedder.embed_query(query)
        return {
            chunk_id: cosine(query_vector, vector)
            for chunk_id, vector in self._vectors.items()
        }

    def _lexical_scores(self, query: str) -> dict[str, float]:
        assert self._lexical is not None
        return {
            chunk.chunk_id: score
            for chunk, score in self._lexical.search(query, len(self._chunks))
        }

    @staticmethod
    def _ranked(scores: dict[str, float], limit: int) -> list[tuple[str, float]]:
        return sorted(scores.items(), key=lambda item: (-item[1], item[0]))[:limit]

    def _search(self, query: str, k: int) -> list[tuple[str, dict[str, float | None]]]:
        if self.mode == CanonicalRetrievalMode.BM25:
            return [
                (chunk_id, {"lexical": score, "dense": None, "fusion": None})
                for chunk_id, score in self._ranked(self._lexical_scores(query), k)
            ]
        if self.mode in (CanonicalRetrievalMode.DENSE, CanonicalRetrievalMode.DENSE_BGE,
                         CanonicalRetrievalMode.HASH):
            return [
                (chunk_id, {"dense": score, "lexical": None, "fusion": None})
                for chunk_id, score in self._ranked(self._dense_scores(query), k)
            ]

        lexical = self._lexical_scores(query)
        dense = self._dense_scores(query)
        lexical_top = dict(self._ranked(lexical, self.candidate_k))
        dense_top = dict(self._ranked(dense, self.candidate_k))
        pool = set(lexical_top) | set(dense_top)

        if self.mode == CanonicalRetrievalMode.HYBRID_SCORE:
            norm_lexical = _min_max({key: lexical.get(key, 0.0) for key in pool})
            norm_dense = _min_max({key: dense.get(key, 0.0) for key in pool})
            fused = {
                key: self.dense_weight * norm_dense[key]
                + (1.0 - self.dense_weight) * norm_lexical[key]
                for key in pool
            }
        else:
            lexical_rank = {
                key: rank for rank, (key, _) in
                enumerate(self._ranked(lexical_top, self.candidate_k), 1)
            }
            dense_rank = {
                key: rank for rank, (key, _) in
                enumerate(self._ranked(dense_top, self.candidate_k), 1)
            }
            fused = {
                key: sum(
                    1.0 / (self.rrf_k + ranks[key])
                    for ranks in (lexical_rank, dense_rank) if key in ranks
                )
                for key in pool
            }

        ordered = self._ranked(fused, self.candidate_k)
        if self.mode == CanonicalRetrievalMode.HYBRID_RERANK:
            scored = [
                (chunk_id, float(self._reranker.score(query, self._by_id[chunk_id])))
                for chunk_id, _ in ordered
            ]
            fused_by_id = dict(ordered)
            scored.sort(key=lambda item: (-item[1], -fused_by_id[item[0]], item[0]))
            return [
                (chunk_id, {
                    "lexical": lexical.get(chunk_id), "dense": dense.get(chunk_id),
                    "fusion": fused_by_id[chunk_id], "reranker": score,
                })
                for chunk_id, score in scored[:k]
            ]
        return [
            (chunk_id, {
                "lexical": lexical.get(chunk_id), "dense": dense.get(chunk_id),
                "fusion": score,
            })
            for chunk_id, score in ordered[:k]
        ]

    async def search(
        self, query: str, *, k: int, filters: EvidenceFilter | None = None,
    ) -> RetrievalResult:
        if k < 1:
            raise ValueError("k must be positive")
        if filters is not None and any(
            (filters.source_types, filters.tags, filters.valid_at, filters.metadata)
        ):
            raise RuntimeError("Canonical Gate B arms do not support filters")
        started = time.perf_counter()
        hits = self._search(query, k)
        rows = [
            RetrievedEvidence.from_index(
                self._records[chunk_id], backend_id=self.backend_id, rank=rank,
                dense_score=scores.get("dense"), lexical_score=scores.get("lexical"),
                reranker_score=scores.get("reranker", scores.get("fusion")),
            )
            for rank, (chunk_id, scores) in enumerate(hits, 1)
        ]
        receipt = RetrievalReceipt(
            backend_id=self.backend_id,
            backend_version=self.backend_version,
            query_sha256=sha256_text(query),
            requested_k=k,
            returned_ids=tuple(row.evidence_id for row in rows),
            latency_ms=(time.perf_counter() - started) * 1000,
            capabilities=await self.capabilities(),
            filters=filters,
        )
        return RetrievalResult(tuple(rows), receipt)

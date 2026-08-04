"""Dependency-light BM25/hash retrieval controls behind the common contract."""

from __future__ import annotations

import hashlib
import json
import time
from enum import Enum
from typing import Sequence

from ..contracts import (
    BackendCapabilities,
    BackendHealth,
    BackendState,
    EvidenceFilter,
    IndexReceipt,
    IndexRecord,
    RetrievedEvidence,
    RetrievalReceipt,
    RetrievalResult,
    sha256_text,
)
from ..memory.chunking import Chunk
from ..retrieval.dense import DenseRetriever, HashingEmbedder
from ..retrieval.hybrid import HybridRetriever
from ..retrieval.lexical import BM25Retriever


class LocalRetrievalMode(str, Enum):
    BM25 = "bm25"
    HASH = "hash"
    HYBRID = "hybrid"


class LocalControlBackend:
    """Primitive retrieval arm; never represented as production semantic search."""

    backend_version = "control-v1"

    def __init__(self, mode: LocalRetrievalMode, records: Sequence[IndexRecord] = ()):
        self.mode = LocalRetrievalMode(mode)
        self.backend_id = f"control:{self.mode.value}"
        self._records: dict[str, IndexRecord] = {}
        self._chunks: list[Chunk] = []
        self._dense: DenseRetriever | None = None
        self._lexical: BM25Retriever | None = None
        self._hybrid: HybridRetriever | None = None
        self._install(records)

    def _install(self, records: Sequence[IndexRecord]) -> None:
        for record in records:
            old = self._records.get(record.evidence_id)
            if old is not None and old != record:
                raise ValueError(f"Evidence ID {record.evidence_id!r} is immutable")
            self._records[record.evidence_id] = record
        self._chunks = [
            Chunk(
                chunk_id=row.evidence_id,
                source_id=row.source_id,
                source_type=row.source_type,
                title=str(row.metadata.get("title", row.source_id)),
                section=str(row.metadata.get("section", "")),
                content=row.content,
                token_count=row.token_count,
                metadata=row.metadata,
            )
            for row in sorted(self._records.values(), key=lambda value: value.evidence_id)
        ]
        self._dense = DenseRetriever(self._chunks, HashingEmbedder())
        self._lexical = BM25Retriever(self._chunks)
        self._hybrid = HybridRetriever(self._chunks, dense=self._dense, lexical=self._lexical)

    async def health(self) -> BackendHealth:
        return BackendHealth(self.backend_id, BackendState.HEALTHY, self.backend_version)

    async def capabilities(self) -> BackendCapabilities:
        return BackendCapabilities(
            dense=self.mode in {LocalRetrievalMode.HASH, LocalRetrievalMode.HYBRID},
            sparse=self.mode in {LocalRetrievalMode.BM25, LocalRetrievalMode.HYBRID},
            batch_indexing=True,
        )

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

    async def search(
        self, query: str, *, k: int, filters: EvidenceFilter | None = None,
    ) -> RetrievalResult:
        if k < 1:
            raise ValueError("k must be positive")
        if filters is not None and any((filters.source_types, filters.tags, filters.valid_at, filters.metadata)):
            raise RuntimeError("Primitive controls do not support filters")
        started = time.perf_counter()
        rows: list[RetrievedEvidence] = []
        if self.mode == LocalRetrievalMode.BM25:
            assert self._lexical is not None
            for rank, (chunk, score) in enumerate(self._lexical.search(query, k), 1):
                rows.append(RetrievedEvidence.from_index(
                    self._records[chunk.chunk_id], backend_id=self.backend_id,
                    rank=rank, lexical_score=score,
                ))
        elif self.mode == LocalRetrievalMode.HASH:
            assert self._dense is not None
            for rank, (chunk, score) in enumerate(self._dense.search(query, k), 1):
                rows.append(RetrievedEvidence.from_index(
                    self._records[chunk.chunk_id], backend_id=self.backend_id,
                    rank=rank, dense_score=score,
                ))
        else:
            assert self._hybrid is not None
            for rank, row in enumerate(self._hybrid.search(query, final_k=k), 1):
                rows.append(RetrievedEvidence.from_index(
                    self._records[row.chunk.chunk_id], backend_id=self.backend_id,
                    rank=rank,
                    dense_score=None if row.dense_rank is None else 1.0 / row.dense_rank,
                    lexical_score=None if row.lexical_rank is None else 1.0 / row.lexical_rank,
                    reranker_score=row.rrf_score,
                ))
        capabilities = await self.capabilities()
        receipt = RetrievalReceipt(
            backend_id=self.backend_id,
            backend_version=self.backend_version,
            query_sha256=sha256_text(query),
            requested_k=k,
            returned_ids=tuple(row.evidence_id for row in rows),
            latency_ms=(time.perf_counter() - started) * 1000,
            capabilities=capabilities,
            filters=filters,
        )
        return RetrievalResult(tuple(rows), receipt)

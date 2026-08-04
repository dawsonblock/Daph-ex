"""Fail-closed client for the future pinned RuVector bridge."""

from __future__ import annotations

import asyncio
import json
import time
import urllib.error
import urllib.request
from dataclasses import asdict
from typing import Any, Mapping, Sequence

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
from .config import SidecarEndpoint


class RuVectorBackend:
    """Use only after Gate A; the client never falls back to local retrieval."""

    def __init__(self, endpoint: SidecarEndpoint):
        self.endpoint = endpoint
        self.backend_id = endpoint.backend_id

    def _request(self, method: str, path: str, payload: Mapping[str, Any] | None = None) -> Any:
        data = None if payload is None else json.dumps(payload).encode()
        request = urllib.request.Request(
            self.endpoint.base_url.rstrip("/") + path,
            data=data,
            method=method,
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(request, timeout=self.endpoint.timeout_seconds) as response:
                return json.loads(response.read().decode())
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"RuVector sidecar unavailable: {exc}") from exc

    async def health(self) -> BackendHealth:
        row = await asyncio.to_thread(self._request, "GET", "/health")
        version = str(row.get("version", ""))
        if version != self.endpoint.pinned_version:
            raise RuntimeError("RuVector sidecar version does not match source lock")
        return BackendHealth(self.backend_id, BackendState.HEALTHY, version)

    async def capabilities(self) -> BackendCapabilities:
        row = await asyncio.to_thread(self._request, "GET", "/v1/capabilities")
        return BackendCapabilities(**{key: bool(value) for key, value in row.items() if key in BackendCapabilities.__dataclass_fields__})

    async def index(self, records: Sequence[IndexRecord]) -> IndexReceipt:
        await self.health()
        row = await asyncio.to_thread(
            self._request, "POST", "/v1/index",
            {"records": [asdict(record) for record in records]},
        )
        returned = tuple(str(value) for value in row.get("indexed_ids", ()))
        expected = tuple(sorted(record.evidence_id for record in records))
        if tuple(sorted(returned)) != expected:
            raise RuntimeError("RuVector index receipt does not match requested IDs")
        canonical = json.dumps([
            (record.evidence_id, record.source_id, sha256_text(record.content))
            for record in sorted(records, key=lambda value: value.evidence_id)
        ], separators=(",", ":"))
        expected_digest = sha256_text(canonical)
        if str(row.get("source_digest", "")) != expected_digest:
            raise RuntimeError("RuVector index source digest does not match requested records")
        return IndexReceipt(self.backend_id, returned, expected_digest)

    async def search(
        self, query: str, *, k: int, filters: EvidenceFilter | None = None,
    ) -> RetrievalResult:
        health = await self.health()
        capabilities = await self.capabilities()
        if filters is not None and any((filters.source_types, filters.tags, filters.valid_at, filters.metadata)):
            capabilities.require("filtering")
        started = time.perf_counter()
        row = await asyncio.to_thread(
            self._request, "POST", "/v1/search",
            {"query": query, "k": k, "filters": None if filters is None else asdict(filters)},
        )
        evidence = tuple(RetrievedEvidence(**item) for item in row.get("evidence", ()))
        return RetrievalResult(evidence, RetrievalReceipt(
            backend_id=self.backend_id,
            backend_version=health.version,
            query_sha256=sha256_text(query),
            requested_k=k,
            returned_ids=tuple(item.evidence_id for item in evidence),
            latency_ms=(time.perf_counter() - started) * 1000,
            capabilities=capabilities,
            filters=filters,
        ))

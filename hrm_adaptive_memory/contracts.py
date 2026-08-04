"""Stable control-plane records and asynchronous backend contracts."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping, Protocol, Sequence, runtime_checkable

from .controller.actions import Action, ActionOutcome
from .memory.schema import MemoryRecord, MemoryStatus


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


class BackendState(str, Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNAVAILABLE = "unavailable"


@dataclass(frozen=True)
class BackendCapabilities:
    dense: bool = False
    sparse: bool = False
    maxsim: bool = False
    filtering: bool = False
    batch_indexing: bool = False
    graph_traversal: bool = False

    def require(self, *names: str) -> None:
        missing = [name for name in names if not bool(getattr(self, name, False))]
        if missing:
            raise RuntimeError(f"Backend lacks required capabilities: {', '.join(missing)}")


@dataclass(frozen=True)
class BackendHealth:
    backend_id: str
    state: BackendState
    version: str
    detail: str = ""


@dataclass(frozen=True)
class EvidenceFilter:
    source_types: tuple[str, ...] = ()
    tags: tuple[str, ...] = ()
    valid_at: str | None = None
    metadata: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class IndexRecord:
    evidence_id: str
    source_id: str
    content: str
    token_count: int
    source_type: str = "source"
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.evidence_id or not self.source_id or not self.content.strip():
            raise ValueError("Index records require evidence_id, source_id, and content")
        if self.token_count < 1:
            raise ValueError("token_count must be positive")


@dataclass(frozen=True)
class RetrievedEvidence:
    evidence_id: str
    source_id: str
    content: str
    content_sha256: str
    token_count: int
    backend_id: str
    dense_score: float | None = None
    lexical_score: float | None = None
    reranker_score: float | None = None
    rank: int = 0
    valid_from: str | None = None
    valid_until: str | None = None
    provenance: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.content_sha256 != sha256_text(self.content):
            raise ValueError("Retrieved evidence content digest mismatch")
        if self.token_count < 1 or self.rank < 0:
            raise ValueError("Retrieved evidence has invalid token_count or rank")

    @classmethod
    def from_index(
        cls, record: IndexRecord, *, backend_id: str, rank: int,
        dense_score: float | None = None, lexical_score: float | None = None,
        reranker_score: float | None = None,
    ) -> "RetrievedEvidence":
        return cls(
            evidence_id=record.evidence_id,
            source_id=record.source_id,
            content=record.content,
            content_sha256=sha256_text(record.content),
            token_count=record.token_count,
            backend_id=backend_id,
            dense_score=dense_score,
            lexical_score=lexical_score,
            reranker_score=reranker_score,
            rank=rank,
            provenance=dict(record.metadata),
        )


@dataclass(frozen=True)
class RetrievalReceipt:
    backend_id: str
    backend_version: str
    query_sha256: str
    requested_k: int
    returned_ids: tuple[str, ...]
    latency_ms: float
    capabilities: BackendCapabilities
    filters: EvidenceFilter | None = None


@dataclass(frozen=True)
class RetrievalResult:
    evidence: tuple[RetrievedEvidence, ...]
    receipt: RetrievalReceipt


@dataclass(frozen=True)
class IndexReceipt:
    backend_id: str
    indexed_ids: tuple[str, ...]
    source_digest: str


@dataclass(frozen=True)
class GraphFact:
    fact_id: str
    statement: str
    source_ids: tuple[str, ...]
    status: MemoryStatus = MemoryStatus.CANDIDATE
    valid_from: str | None = None
    valid_until: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TopicDocument:
    topic_id: str
    markdown: str
    supporting_source_ids: tuple[str, ...]
    content_sha256: str
    version: str

    def __post_init__(self) -> None:
        if self.content_sha256 != sha256_text(self.markdown):
            raise ValueError("Topic document digest mismatch")


@dataclass(frozen=True)
class DerivationReceipt:
    provider: str
    model_revision: str
    prompt_sha256: str
    source_sha256: tuple[str, ...]
    output_sha256: str
    verifier: str
    verified: bool

    @property
    def cache_key(self) -> str:
        payload = "\0".join((
            self.provider, self.model_revision, self.prompt_sha256,
            *self.source_sha256,
        ))
        return sha256_text(payload)


@dataclass(frozen=True)
class HRMStateSnapshot:
    z_h: Any
    z_l: Any
    high_cycle: int
    low_cycle: int
    token_state: Any
    prefix_state: Any
    rng_state: Any
    model_digest: str


@runtime_checkable
class RetrievalBackend(Protocol):
    backend_id: str

    async def health(self) -> BackendHealth: ...
    async def capabilities(self) -> BackendCapabilities: ...
    async def index(self, records: Sequence[IndexRecord]) -> IndexReceipt: ...
    async def search(
        self, query: str, *, k: int, filters: EvidenceFilter | None = None,
    ) -> RetrievalResult: ...


@runtime_checkable
class MemoryBackend(Protocol):
    async def write(self, record: MemoryRecord) -> None: ...
    async def read(self, memory_id: str) -> MemoryRecord | None: ...
    async def transition(self, memory_id: str, target: MemoryStatus) -> MemoryRecord: ...


@runtime_checkable
class GraphMemoryBackend(Protocol):
    async def add_candidate(self, fact: GraphFact) -> GraphFact: ...
    async def promote(self, fact_id: str) -> GraphFact: ...
    async def supersede(self, prior_fact_id: str, replacement: GraphFact) -> GraphFact: ...
    async def search_entities(self, query: str, *, k: int) -> Sequence[GraphFact]: ...
    async def traverse(self, entity_id: str, *, depth: int) -> Sequence[GraphFact]: ...


@runtime_checkable
class ConsolidationBackend(Protocol):
    async def consolidate(
        self, *, topic_id: str, records: Sequence[MemoryRecord],
    ) -> tuple[TopicDocument, DerivationReceipt]: ...


@runtime_checkable
class HRMRuntime(Protocol):
    async def initialize(self, tokens: Any, prefix_state: Any) -> HRMStateSnapshot: ...
    async def run_low(self, state: HRMStateSnapshot) -> HRMStateSnapshot: ...
    async def run_high(self, state: HRMStateSnapshot) -> HRMStateSnapshot: ...
    async def decode(self, state: HRMStateSnapshot) -> Any: ...
    async def snapshot(self, state: HRMStateSnapshot) -> HRMStateSnapshot: ...
    async def restore(self, snapshot: HRMStateSnapshot) -> HRMStateSnapshot: ...


@runtime_checkable
class ActionExecutor(Protocol):
    async def execute(self, state: Any, action: Action) -> ActionOutcome: ...

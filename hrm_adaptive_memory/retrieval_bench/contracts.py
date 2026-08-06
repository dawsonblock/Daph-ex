"""Internal retrieval contracts. BEIR plugs into these; DAPH does not depend on it.

Ownership boundary (deliberate): BEIR owns standard IR dataset/evaluation
plumbing, FlagEmbedding owns embedding and reranker inference, and DAPH owns
V4, the proof graph, oracle separation, custom proof metrics, receipts, and
claim control. If BEIR disappeared tomorrow another adapter must be writable
without touching the research protocol — so nothing here imports it, and the
export emits BEIR *format* rather than BEIR *objects*.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol, Sequence


@dataclass(frozen=True)
class RetrievalQuery:
    task_id: str
    text: str
    split: str
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RetrievalDocument:
    evidence_id: str
    text: str
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RankedCandidate:
    evidence_id: str
    score: float
    rank: int
    backend: str


class RetrievalBackend(Protocol):
    backend_id: str

    def index(self, documents: Sequence[RetrievalDocument]) -> None: ...
    def retrieve(self, query: RetrievalQuery, *, k: int) -> list[RankedCandidate]: ...
    def manifest(self) -> dict[str, Any]: ...


class OracleLeakError(RuntimeError):
    """Raised when evaluator-only truth reaches a runtime component."""


# Keys that identify evaluator-only truth. A runtime component that sees any of
# these has been handed the answer key, which would make Gate C2 unfalsifiable.
ORACLE_KEYS = frozenset({
    "_oracle_metadata", "proof_edges", "proof_path_ids", "required_evidence_ids",
    "oracle_evidence_ids", "bridge_ids", "answer_record_ids", "latent_subject",
    "latent_bridge", "answer_node", "qrels",
})


def assert_runtime_clean(payload: Any, *, where: str) -> None:
    """Fail closed if evaluator-only truth appears in a runtime payload."""

    def walk(node: Any, path: str) -> None:
        if isinstance(node, Mapping):
            for key, value in node.items():
                if str(key) in ORACLE_KEYS:
                    raise OracleLeakError(
                        f"{where}: evaluator-only key {key!r} reached runtime at {path}")
                walk(value, f"{path}.{key}")
        elif isinstance(node, (list, tuple)):
            for index, value in enumerate(node):
                walk(value, f"{path}[{index}]")

    walk(payload, where)

"""
RFSN integration hooks for DAPH / ExFusion v3.

Lightweight, dependency-free adapters that emit structured evidence events
(effort scores, routing decisions, early-exits, merge results) into an
agentic memory / immutable vault interface.

Designed to plug into RFSN-style stacks:
  - Immutable evidence vault
  - Bi-temporal knowledge graph
  - Salience / decay
  - Contradiction detection

No hard dependency on a specific RFSN runtime — implement the protocol
or use the in-memory reference sink.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field, asdict
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Protocol,
    runtime_checkable,
)

import torch
from torch import Tensor


# ---------------------------------------------------------------------------
# Event schema
# ---------------------------------------------------------------------------

@dataclass
class ExFusionEvent:
    """Single structured evidence record."""

    event_id: str
    event_type: str                 # effort | routing | early_exit | merge | forward
    timestamp: float                # unix seconds (valid time)
    transaction_time: float         # when recorded (bi-temporal)
    layer_index: Optional[int] = None
    sequence_id: Optional[str] = None
    payload: Dict[str, Any] = field(default_factory=dict)
    salience: float = 1.0           # initial salience for decay systems
    tags: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _new_id() -> str:
    return uuid.uuid4().hex[:16]


def _now() -> float:
    return time.time()


def _tensor_summary(t: Optional[Tensor], max_items: int = 8) -> Any:
    """Detach + CPU + small summary suitable for vault storage."""
    if t is None:
        return None
    if not isinstance(t, Tensor):
        return t
    x = t.detach().float().cpu()
    if x.numel() == 0:
        return {"shape": list(x.shape), "empty": True}
    flat = x.reshape(-1)
    summary: Dict[str, Any] = {
        "shape": list(x.shape),
        "mean": float(flat.mean()),
        "std": float(flat.std()) if flat.numel() > 1 else 0.0,
        "min": float(flat.min()),
        "max": float(flat.max()),
    }
    if flat.numel() <= max_items:
        summary["values"] = flat.tolist()
    return summary


# ---------------------------------------------------------------------------
# Protocol — implement this to connect a real RFSN vault / KG
# ---------------------------------------------------------------------------

@runtime_checkable
class RFSNSink(Protocol):
    """Minimal interface a memory/vault backend must satisfy."""

    def emit(self, event: ExFusionEvent) -> None:
        """Append an immutable evidence event."""
        ...

    def query(
        self,
        event_type: Optional[str] = None,
        layer_index: Optional[int] = None,
        since: Optional[float] = None,
        limit: int = 100,
    ) -> List[ExFusionEvent]:
        """Retrieve recent events (for diagnostics / bi-temporal reads)."""
        ...


# ---------------------------------------------------------------------------
# In-memory reference sink (useful for tests & local prototyping)
# ---------------------------------------------------------------------------

class InMemoryRFSNSink:
    """Append-only list with simple filtering. Not durable."""

    def __init__(self, max_events: int = 10_000) -> None:
        self.max_events = max_events
        self._events: List[ExFusionEvent] = []

    def emit(self, event: ExFusionEvent) -> None:
        self._events.append(event)
        if len(self._events) > self.max_events:
            self._events = self._events[-self.max_events :]

    def query(
        self,
        event_type: Optional[str] = None,
        layer_index: Optional[int] = None,
        since: Optional[float] = None,
        limit: int = 100,
    ) -> List[ExFusionEvent]:
        out: List[ExFusionEvent] = []
        for e in reversed(self._events):
            if event_type is not None and e.event_type != event_type:
                continue
            if layer_index is not None and e.layer_index != layer_index:
                continue
            if since is not None and e.timestamp < since:
                continue
            out.append(e)
            if len(out) >= limit:
                break
        return list(reversed(out))

    def clear(self) -> None:
        self._events.clear()

    def __len__(self) -> int:
        return len(self._events)


# ---------------------------------------------------------------------------
# Emitter — call from HybridBlock / Model
# ---------------------------------------------------------------------------

class ExFusionEmitter:
    """
    Thin façade that turns model meta dicts into typed ExFusionEvents
    and pushes them to a sink.
    """

    def __init__(
        self,
        sink: Optional[RFSNSink] = None,
        sequence_id: Optional[str] = None,
        enabled: bool = True,
        default_salience: float = 1.0,
    ) -> None:
        self.sink = sink if sink is not None else InMemoryRFSNSink()
        self.sequence_id = sequence_id or _new_id()
        self.enabled = enabled
        self.default_salience = default_salience

    def _emit(
        self,
        event_type: str,
        payload: Dict[str, Any],
        layer_index: Optional[int] = None,
        salience: Optional[float] = None,
        tags: Optional[List[str]] = None,
    ) -> Optional[ExFusionEvent]:
        if not self.enabled:
            return None
        now = _now()
        event = ExFusionEvent(
            event_id=_new_id(),
            event_type=event_type,
            timestamp=now,
            transaction_time=now,
            layer_index=layer_index,
            sequence_id=self.sequence_id,
            payload=payload,
            salience=salience if salience is not None else self.default_salience,
            tags=tags or [],
        )
        self.sink.emit(event)
        return event

    def emit_effort(
        self,
        effort_info: Dict[str, Any],
        layer_index: Optional[int] = None,
    ) -> Optional[ExFusionEvent]:
        payload = {
            "effort_score": _tensor_summary(effort_info.get("effort_score")),
            "expected_cost": _tensor_summary(effort_info.get("expected_cost")),
            "effort_probs": _tensor_summary(effort_info.get("effort_probs")),
        }
        return self._emit("effort", payload, layer_index=layer_index, tags=["compute"])

    def emit_early_exit(
        self,
        meta: Dict[str, Any],
        layer_index: Optional[int] = None,
    ) -> Optional[ExFusionEvent]:
        payload = {
            "early_exited": bool(meta.get("early_exited", False)),
            "early_exited_partial": bool(meta.get("early_exited_partial", False)),
            "early_exit_mask": _tensor_summary(meta.get("early_exit_mask")),
        }
        sal = 0.6 if payload["early_exited"] else 0.9
        return self._emit(
            "early_exit", payload, layer_index=layer_index, salience=sal, tags=["compute", "exit"]
        )

    def emit_routing(
        self,
        router_logits: Optional[Tensor],
        layer_index: Optional[int] = None,
    ) -> Optional[ExFusionEvent]:
        payload = {"router_logits": _tensor_summary(router_logits)}
        return self._emit("routing", payload, layer_index=layer_index, tags=["moe"])

    def emit_forward_summary(
        self,
        num_layers: int,
        effort_scores: List[Any],
        extra: Optional[Dict[str, Any]] = None,
    ) -> Optional[ExFusionEvent]:
        payload = {
            "num_layers": num_layers,
            "effort_scores": [_tensor_summary(s) for s in effort_scores],
        }
        if extra:
            payload.update(extra)
        return self._emit("forward", payload, tags=["trace"])

    def emit_merge(
        self,
        merge_kind: str,
        num_experts: int,
        extra: Optional[Dict[str, Any]] = None,
    ) -> Optional[ExFusionEvent]:
        payload = {"merge_kind": merge_kind, "num_experts": num_experts}
        if extra:
            payload.update(extra)
        return self._emit(
            "merge", payload, salience=1.2, tags=["merge", "weights"]
        )


# ---------------------------------------------------------------------------
# Optional model wrapper that auto-emits on forward
# ---------------------------------------------------------------------------

def attach_emitter_to_model(model: Any, emitter: ExFusionEmitter) -> Any:
    """
    Monkey-patch / wrap a DAPHHybridModelV3 so each forward emits
    effort + early-exit + routing events. Returns the same model instance.
    """
    if getattr(model, "_exfusion_emitter", None) is not None:
        model._exfusion_emitter = emitter
        return model

    original_forward = model.forward

    def forward_with_emit(*args: Any, **kwargs: Any) -> Any:
        out = original_forward(*args, **kwargs)
        if not isinstance(out, dict):
            return out
        scores = out.get("effort_scores") or []
        for i, score in enumerate(scores):
            # score may be a tensor; synthesize a minimal effort_info
            info = {"effort_score": score}
            emitter.emit_effort(info, layer_index=i)
        emitter.emit_forward_summary(
            num_layers=len(scores),
            effort_scores=scores,
            extra={"logits_shape": list(out["logits"].shape) if "logits" in out else None},
        )
        return out

    model.forward = forward_with_emit  # type: ignore[method-assign]
    model._exfusion_emitter = emitter
    return model


# ---------------------------------------------------------------------------
# Immutable vault adapter (bi-temporal + salience decay)
# ---------------------------------------------------------------------------

@dataclass
class VaultRecord:
    """Immutable evidence row stored in the vault."""

    event: ExFusionEvent
    # Bi-temporal: valid_from = event.timestamp, recorded_at = event.transaction_time
    # Soft-delete / supersession via valid_to (None = still current)
    valid_to: Optional[float] = None
    salience: float = 1.0
    content_hash: str = ""

    @property
    def is_current(self) -> bool:
        return self.valid_to is None


class ImmutableVaultSink:
    """
    Append-only vault with:
      - bi-temporal validity (valid_from / valid_to / recorded_at)
      - content hashing for integrity
      - salience scores with optional exponential decay
      - simple as-of and history queries

    This is a local, in-process reference implementation intended to match
    the *interface* of a production RFSN immutable evidence vault so you
    can swap in a disk / SQLite / content-addressed backend later.
    """

    def __init__(
        self,
        max_records: int = 50_000,
        salience_half_life_s: Optional[float] = None,
    ) -> None:
        self.max_records = max_records
        self.salience_half_life_s = salience_half_life_s
        self._records: List[VaultRecord] = []
        self._by_id: Dict[str, VaultRecord] = {}

    # --- RFSNSink protocol -------------------------------------------------

    def emit(self, event: ExFusionEvent) -> None:
        import hashlib
        import json

        payload_str = json.dumps(event.payload, sort_keys=True, default=str)
        h = hashlib.sha256(
            f"{event.event_id}|{event.event_type}|{event.timestamp}|{payload_str}".encode()
        ).hexdigest()[:32]

        rec = VaultRecord(
            event=event,
            valid_to=None,
            salience=event.salience,
            content_hash=h,
        )
        self._records.append(rec)
        self._by_id[event.event_id] = rec
        if len(self._records) > self.max_records:
            # Drop oldest *current* records first (still append-only log; just
            # trim the in-memory index for the prototype).
            overflow = len(self._records) - self.max_records
            self._records = self._records[overflow:]
            self._by_id = {r.event.event_id: r for r in self._records}

    def query(
        self,
        event_type: Optional[str] = None,
        layer_index: Optional[int] = None,
        since: Optional[float] = None,
        limit: int = 100,
    ) -> List[ExFusionEvent]:
        return [
            r.event
            for r in self.as_of(at=None, event_type=event_type, layer_index=layer_index, since=since, limit=limit)
        ]

    # --- Vault-specific API ------------------------------------------------

    def __len__(self) -> int:
        return len(self._records)

    def get(self, event_id: str) -> Optional[VaultRecord]:
        return self._by_id.get(event_id)

    def supersede(self, event_id: str, at: Optional[float] = None) -> bool:
        """Mark a record as no longer valid after `at` (soft delete)."""
        rec = self._by_id.get(event_id)
        if rec is None or rec.valid_to is not None:
            return False
        rec.valid_to = at if at is not None else _now()
        return True

    def decay_salience(self, now: Optional[float] = None) -> None:
        """Apply exponential salience decay based on half-life."""
        if self.salience_half_life_s is None or self.salience_half_life_s <= 0:
            return
        now = now if now is not None else _now()
        import math
        ln2 = math.log(2)
        for rec in self._records:
            if not rec.is_current:
                continue
            age = max(0.0, now - rec.event.timestamp)
            rec.salience = rec.event.salience * math.exp(-ln2 * age / self.salience_half_life_s)

    def as_of(
        self,
        at: Optional[float] = None,
        event_type: Optional[str] = None,
        layer_index: Optional[int] = None,
        since: Optional[float] = None,
        min_salience: float = 0.0,
        limit: int = 100,
    ) -> List[VaultRecord]:
        """
        Bi-temporal read: records that were valid at time `at`
        (default = now), optionally filtered.
        """
        at = at if at is not None else _now()
        out: List[VaultRecord] = []
        for rec in reversed(self._records):
            ev = rec.event
            # valid_from <= at < valid_to (or valid_to is None)
            if ev.timestamp > at:
                continue
            if rec.valid_to is not None and rec.valid_to <= at:
                continue
            if event_type is not None and ev.event_type != event_type:
                continue
            if layer_index is not None and ev.layer_index != layer_index:
                continue
            if since is not None and ev.timestamp < since:
                continue
            if rec.salience < min_salience:
                continue
            out.append(rec)
            if len(out) >= limit:
                break
        return list(reversed(out))

    def history(self, event_id: str) -> List[VaultRecord]:
        """All versions / supersessions for an id (prototype: single row)."""
        rec = self._by_id.get(event_id)
        return [rec] if rec is not None else []

    def verify_integrity(self) -> bool:
        """Recompute content hashes; return True if all match."""
        import hashlib
        import json

        for rec in self._records:
            ev = rec.event
            payload_str = json.dumps(ev.payload, sort_keys=True, default=str)
            h = hashlib.sha256(
                f"{ev.event_id}|{ev.event_type}|{ev.timestamp}|{payload_str}".encode()
            ).hexdigest()[:32]
            if h != rec.content_hash:
                return False
        return True

    def clear(self) -> None:
        self._records.clear()
        self._by_id.clear()


# ---------------------------------------------------------------------------
# Honest naming alias
# ---------------------------------------------------------------------------

# ImmutableVaultSink is an in-memory prototype — not cryptographically immutable.
# Prefer PrototypeEvidenceVault or AppendOnlyJSONLVault for durable logs.
PrototypeEvidenceVault = ImmutableVaultSink


class AppendOnlyJSONLVault:
    """
    Tamper-evident append-only event log.

    Each line is a JSON event with:
      event fields + content_hash + parent_hash + experiment_id

    H_i = SHA256(H_{i-1} || canonical(event_i))

    Does not delete records. Capacity management = rotate segment files.
    """

    def __init__(
        self,
        path: str,
        experiment_id: str = "default",
    ) -> None:
        import os
        self.path = path
        self.experiment_id = experiment_id
        self._parent_hash = "0" * 64
        self._count = 0
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        # resume parent hash from last line if file exists
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    import json
                    rec = json.loads(line)
                    self._parent_hash = rec.get("content_hash", self._parent_hash)
                    self._count += 1

    def emit(self, event: ExFusionEvent) -> None:
        import hashlib
        import json

        payload = event.to_dict()
        payload["experiment_id"] = self.experiment_id
        payload["parent_hash"] = self._parent_hash
        canonical = json.dumps(payload, sort_keys=True, default=str)
        h = hashlib.sha256(
            (self._parent_hash + canonical).encode("utf-8")
        ).hexdigest()
        payload["content_hash"] = h
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload, default=str) + "\n")
        self._parent_hash = h
        self._count += 1

    def query(
        self,
        event_type: Optional[str] = None,
        layer_index: Optional[int] = None,
        since: Optional[float] = None,
        limit: int = 100,
    ) -> List[ExFusionEvent]:
        import json
        import os

        if not os.path.exists(self.path):
            return []
        events: List[ExFusionEvent] = []
        with open(self.path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                if event_type is not None and rec.get("event_type") != event_type:
                    continue
                if layer_index is not None and rec.get("layer_index") != layer_index:
                    continue
                if since is not None and rec.get("timestamp", 0) < since:
                    continue
                events.append(
                    ExFusionEvent(
                        event_id=rec.get("event_id", ""),
                        event_type=rec.get("event_type", ""),
                        timestamp=rec.get("timestamp", 0.0),
                        transaction_time=rec.get("transaction_time", 0.0),
                        layer_index=rec.get("layer_index"),
                        sequence_id=rec.get("sequence_id"),
                        payload=rec.get("payload") or {},
                        salience=rec.get("salience", 1.0),
                        tags=rec.get("tags") or [],
                    )
                )
        return events[-limit:]

    def verify_chain(self) -> bool:
        import hashlib
        import json
        import os

        if not os.path.exists(self.path):
            return True
        parent = "0" * 64
        with open(self.path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                stored = rec.get("content_hash")
                rec_no_hash = {k: v for k, v in rec.items() if k != "content_hash"}
                # parent_hash already in rec
                canonical = json.dumps(rec_no_hash, sort_keys=True, default=str)
                h = hashlib.sha256((parent + canonical).encode("utf-8")).hexdigest()
                if h != stored:
                    return False
                parent = stored
        return True

    def __len__(self) -> int:
        return self._count

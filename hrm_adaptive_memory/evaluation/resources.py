"""Honest compute and resource accounting.

Every cost is attributed to the phase that incurred it, and every metric this
process cannot actually observe is reported as ``None`` rather than guessed.
In particular: Python allocator statistics are never reported as physical
model memory, and framework allocator peaks are only emitted for devices that
expose them (CUDA). MPS has no peak-stats API, so its allocator reading is
labelled as a current-allocation sample, not a peak.
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from typing import Any, Iterator

PHASES = (
    "retrieval",
    "prompt_construction",
    "model",
    "verification",
    "calculation",
)


@dataclass
class CallCounters:
    retrieval_calls: int = 0
    model_calls: int = 0
    verifier_calls: int = 0
    calculator_calls: int = 0
    hrm_steps: int = 0

    def merge(self, other: "CallCounters") -> None:
        self.retrieval_calls += other.retrieval_calls
        self.model_calls += other.model_calls
        self.verifier_calls += other.verifier_calls
        self.calculator_calls += other.calculator_calls
        self.hrm_steps += other.hrm_steps


@dataclass
class TokenCounters:
    prompt_tokens: int = 0
    evidence_tokens: int = 0
    completion_tokens: int = 0


@dataclass
class MemorySample:
    """Memory readings, each explicitly labelled by what it actually measures."""

    process_rss_bytes: int | None = None
    framework_allocator_peak_bytes: int | None = None
    framework_allocator_current_bytes: int | None = None
    framework_allocator_kind: str | None = None
    model_parameter_bytes: int | None = None
    python_allocator_peak_bytes: int | None = None

    @classmethod
    def sample(cls, *, model: Any = None, include_python_allocator: bool = False) -> "MemorySample":
        row = cls()
        try:
            import resource
            usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            # Linux reports kilobytes, macOS reports bytes.
            import sys
            row.process_rss_bytes = int(usage) if sys.platform == "darwin" else int(usage) * 1024
        except (ImportError, OSError):
            row.process_rss_bytes = None
        try:
            import torch
            if torch.cuda.is_available():
                row.framework_allocator_kind = "cuda"
                row.framework_allocator_peak_bytes = int(torch.cuda.max_memory_allocated())
                row.framework_allocator_current_bytes = int(torch.cuda.memory_allocated())
            elif torch.backends.mps.is_available():
                # MPS exposes no peak-stats API; a current sample is the only
                # honest reading, so the peak stays None.
                row.framework_allocator_kind = "mps"
                row.framework_allocator_current_bytes = int(torch.mps.current_allocated_memory())
            if model is not None:
                row.model_parameter_bytes = int(sum(
                    parameter.numel() * parameter.element_size()
                    for parameter in model.parameters()
                ))
        except (ImportError, AttributeError, RuntimeError):
            pass
        if include_python_allocator:
            import tracemalloc
            if tracemalloc.is_tracing():
                row.python_allocator_peak_bytes = int(tracemalloc.get_traced_memory()[1])
        return row


@dataclass
class ResourceLedger:
    """Phase-attributed latency plus call, token, and memory accounting."""

    phase_latency_ms: dict[str, float] = field(
        default_factory=lambda: {phase: 0.0 for phase in PHASES}
    )
    total_wall_ms: float = 0.0
    calls: CallCounters = field(default_factory=CallCounters)
    tokens: TokenCounters = field(default_factory=TokenCounters)
    memory: MemorySample = field(default_factory=MemorySample)
    _started: float | None = field(default=None, repr=False)

    @contextmanager
    def phase(self, name: str) -> Iterator[None]:
        if name not in self.phase_latency_ms:
            raise ValueError(f"Unknown resource phase {name!r}; declare it in PHASES")
        started = time.perf_counter()
        try:
            yield
        finally:
            self.phase_latency_ms[name] += (time.perf_counter() - started) * 1000

    @contextmanager
    def wall_clock(self) -> Iterator[None]:
        started = time.perf_counter()
        try:
            yield
        finally:
            self.total_wall_ms += (time.perf_counter() - started) * 1000

    @property
    def unattributed_ms(self) -> float | None:
        """Wall time not attributed to any declared phase."""

        if not self.total_wall_ms:
            return None
        return round(self.total_wall_ms - sum(self.phase_latency_ms.values()), 6)

    def to_dict(self) -> dict[str, Any]:
        return {
            "latency_ms": {
                **{f"{phase}_ms": round(value, 3) for phase, value in self.phase_latency_ms.items()},
                "total_wall_ms": round(self.total_wall_ms, 3),
                "unattributed_ms": self.unattributed_ms,
            },
            "calls": asdict(self.calls),
            "tokens": asdict(self.tokens),
            "memory": asdict(self.memory),
        }

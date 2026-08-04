"""Cycle-level state tracing for the native HrmText model modules."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable


@dataclass(frozen=True)
class RecurrentStateTrace:
    state_type: str
    high_cycle: int
    low_cycle: int | None
    last_token: list[float]
    mean_pool: list[float]
    rms: float


def _tensor_from_output(output: Any) -> Any:
    if hasattr(output, "last_hidden_state"):
        return output.last_hidden_state
    if isinstance(output, (tuple, list)):
        return output[0]
    return output


class HRMRecurrentTracer:
    """Register hooks on `L_module` and `H_module` without forking Transformers.

    Native HRM reuses each stack, so hook call order maps deterministically to
    L1..L3,H1,L4..L6,H2 for the released 2x3 schedule.
    """

    def __init__(self, model: Any, *, projector: Callable[[Any], Any] | None = None):
        core = getattr(model, "model", model)
        if not hasattr(core, "L_module") or not hasattr(core, "H_module"):
            raise TypeError("Expected native HrmText model with L_module and H_module")
        self.core = core
        self.projector = projector
        self.traces: list[RecurrentStateTrace] = []
        self._handles: list[Any] = []
        self._low_calls = 0
        self._high_calls = 0

    def _summary(self, output: Any, state_type: str) -> RecurrentStateTrace:
        import torch
        hidden = _tensor_from_output(output).detach().float()
        if self.projector is not None:
            hidden = self.projector(hidden)
        high_cycle = self._high_calls if state_type == "L" else self._high_calls + 1
        low_cycle = (self._low_calls % int(self.core.config.L_cycles)) + 1 if state_type == "L" else None
        trace = RecurrentStateTrace(
            state_type=state_type,
            high_cycle=high_cycle,
            low_cycle=low_cycle,
            last_token=hidden[:, -1].mean(dim=0).cpu().tolist(),
            mean_pool=hidden.mean(dim=(0, 1)).cpu().tolist(),
            rms=float(torch.sqrt(hidden.square().mean()).cpu()),
        )
        if state_type == "L":
            self._low_calls += 1
        else:
            self._high_calls += 1
        return trace

    def __enter__(self) -> "HRMRecurrentTracer":
        self.traces.clear(); self._low_calls = 0; self._high_calls = 0
        self._handles = [
            self.core.L_module.register_forward_hook(lambda _m, _i, out: self.traces.append(self._summary(out, "L"))),
            self.core.H_module.register_forward_hook(lambda _m, _i, out: self.traces.append(self._summary(out, "H"))),
        ]
        return self

    def __exit__(self, *_args: Any) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

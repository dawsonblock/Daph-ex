"""Selective State Space Model (SSM) with pluggable scan backends."""

from __future__ import annotations

import os
import warnings
from typing import Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


def _selective_scan_impl(
    xin: Tensor,
    b_matrix: Tensor,
    c_matrix: Tensor,
    dt: Tensor,
    a_matrix: Tensor,
    d_skip: Tensor,
    h_init: Tensor,
) -> Tuple[Tensor, Tensor]:
    """Minimal selective scan using FP32 recurrent state arithmetic."""
    input_dtype = xin.dtype
    xin_f = xin.float()
    b_f = b_matrix.float()
    c_f = c_matrix.float()
    dt_f = dt.float()
    a_f = a_matrix.float()
    d_f = d_skip.float()
    state = h_init.float()

    outputs: List[Tensor] = []
    for token_index in range(xin.shape[1]):
        dt_t = dt_f[:, token_index].unsqueeze(-1)
        transition = torch.exp(dt_t * a_f.unsqueeze(0))
        input_term = (
            dt_t
            * b_f[:, token_index].unsqueeze(1)
            * xin_f[:, token_index].unsqueeze(-1)
        )
        state = transition * state + input_term
        projected = (state * c_f[:, token_index].unsqueeze(1)).sum(dim=-1)
        outputs.append(projected)

    y = torch.stack(outputs, dim=1)
    y = y + d_f * xin_f
    return y.to(input_dtype), state


_SCAN_BACKENDS: Dict[str, Callable[..., Tuple[Tensor, Tensor]]] = {}
_COMPILED_FALLBACK: Optional[Callable[..., Tuple[Tensor, Tensor]]] = None


def register_scan_backend(
    name: str,
    fn: Callable[..., Tuple[Tensor, Tensor]],
) -> None:
    """Register an optimized selective-scan backend (e.g. mamba_ssm / Triton).

    Select it at runtime with DAPH_SCAN_BACKEND=<name>.
    The callable must match _selective_scan_impl's signature.
    """
    _SCAN_BACKENDS[name] = fn


def dispatch_selective_scan(
    xin: Tensor,
    b_matrix: Tensor,
    c_matrix: Tensor,
    dt: Tensor,
    a_matrix: Tensor,
    d_skip: Tensor,
    h_init: Tensor,
) -> Tuple[Tensor, Tensor]:
    """Route the scan to a registered backend or fall back to compiled/eager FP32."""
    global _COMPILED_FALLBACK
    backend = os.environ.get("DAPH_SCAN_BACKEND")
    if backend:
        if backend not in _SCAN_BACKENDS:
            raise ValueError(
                f"DAPH_SCAN_BACKEND='{backend}' not registered; "
                f"available: {sorted(_SCAN_BACKENDS)}"
            )
        return _SCAN_BACKENDS[backend](
            xin, b_matrix, c_matrix, dt, a_matrix, d_skip, h_init
        )
    if _COMPILED_FALLBACK is None:
        _COMPILED_FALLBACK = _maybe_compiled_scan()
    return _COMPILED_FALLBACK(
        xin, b_matrix, c_matrix, dt, a_matrix, d_skip, h_init
    )


def _maybe_compiled_scan() -> Callable[..., Tuple[Tensor, Tensor]]:
    if os.environ.get("DAPH_USE_COMPILE", "0") != "1":
        return _selective_scan_impl
    try:
        return torch.compile(_selective_scan_impl, dynamic=True, fullgraph=False)
    except Exception as exc:  # pragma: no cover
        warnings.warn(f"torch.compile unavailable ({exc}); using eager scan")
        return _selective_scan_impl


class SelectiveSSM(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        state_size: int = 16,
        dt_min: float = 0.001,
        dt_max: float = 0.1,
    ) -> None:
        super().__init__()
        if hidden_size <= 0 or state_size <= 0:
            raise ValueError("hidden_size and state_size must be positive")
        if not (0 < dt_min <= dt_max):
            raise ValueError("Require 0 < dt_min <= dt_max")

        self.hidden_size = hidden_size
        self.state_size = state_size
        self.in_proj = nn.Linear(hidden_size, hidden_size * 2, bias=False)
        self.x_proj = nn.Linear(hidden_size, state_size * 2 + hidden_size, bias=False)
        self.dt_proj = nn.Linear(hidden_size, hidden_size, bias=True)
        self.A_log = nn.Parameter(
            torch.log(torch.arange(1, state_size + 1, dtype=torch.float32))
            .unsqueeze(0)
            .expand(hidden_size, -1)
            .clone()
        )
        self.D = nn.Parameter(torch.ones(hidden_size))
        self.out_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.dt_min = dt_min
        self.dt_max = dt_max
        self._scan = dispatch_selective_scan

    def _validate_state(self, state: Tensor, batch_size: int) -> None:
        expected = (batch_size, self.hidden_size, self.state_size)
        if tuple(state.shape) != expected:
            raise ValueError(
                f"Invalid SSM state shape {tuple(state.shape)}; expected {expected}"
            )

    def forward(
        self,
        x: Tensor,
        state: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
        bypass_decay: float = 0.0,
    ) -> Tuple[Tensor, Tensor]:
        if x.dim() != 3:
            raise ValueError(f"x must have shape (B, L, H); got {tuple(x.shape)}")
        batch_size, seq_len, hidden_size = x.shape
        if hidden_size != self.hidden_size:
            raise ValueError(
                f"Input hidden size {hidden_size} != configured {self.hidden_size}"
            )

        xz = self.in_proj(x)
        xin, gate = xz.chunk(2, dim=-1)
        bcdt = self.x_proj(xin)
        b_matrix, c_matrix, dt_raw = torch.split(
            bcdt,
            [self.state_size, self.state_size, self.hidden_size],
            dim=-1,
        )
        dt = F.softplus(self.dt_proj(xin) + dt_raw).clamp(self.dt_min, self.dt_max)

        if mask is not None:
            if mask.shape != (batch_size, seq_len):
                raise ValueError(
                    f"SSM mask shape {tuple(mask.shape)}; expected {(batch_size, seq_len)}"
                )
            mask_f = mask.unsqueeze(-1).to(dt.dtype)
            if bypass_decay > 0.0:
                # Opt-in gamma-decay on bypassed steps
                dt = dt * mask_f + bypass_decay * (1.0 - mask_f)
            else:
                dt = dt * mask_f

        a_matrix = -torch.exp(self.A_log.float())
        if state is None:
            h_init = torch.zeros(
                batch_size,
                self.hidden_size,
                self.state_size,
                dtype=torch.float32,
                device=x.device,
            )
        else:
            self._validate_state(state, batch_size)
            h_init = state.to(device=x.device, dtype=torch.float32)

        y, new_state = self._scan(
            xin, b_matrix, c_matrix, dt, a_matrix, self.D, h_init
        )
        output = self.out_proj(y * F.silu(gate))
        return output, new_state

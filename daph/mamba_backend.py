"""
Optional mamba-ssm selective-scan backend for SelectiveSSM.

Install (GPU):
  pip install mamba-ssm causal-conv1d --no-build-isolation
  # if CUDA extension missing:
  # MAMBA_KEEP_CUDA_BUILD=TRUE pip install mamba-ssm --no-build-isolation

Enable:
  export DAPH_SCAN_BACKEND=mamba_ssm
  # or call register_mamba_ssm_backend() at startup

Falls back gracefully if mamba_ssm is not installed.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
from torch import Tensor

from .ssm import register_scan_backend


def _mamba_selective_scan(
    xin: Tensor,
    b_matrix: Tensor,
    c_matrix: Tensor,
    dt: Tensor,
    a_matrix: Tensor,
    d_skip: Tensor,
    h_init: Tensor,
) -> Tuple[Tensor, Tensor]:
    """
    Bridge SelectiveSSM layout → mamba_ssm.selective_scan_fn.

    SelectiveSSM convention (this repo):
      xin:      (B, L, H)
      b_matrix: (B, L, N)
      c_matrix: (B, L, N)
      dt:       (B, L, H)
      a_matrix: (H, N)   negative continuous A (already -exp(A_log))
      d_skip:   (H,)
      h_init:   (B, H, N)

    mamba_ssm selective_scan_fn:
      u, delta: (B, D, L)
      A:        (D, N)
      B, C:     (B, N, L)  or (B, G, N, L)
      D:        (D,)
      return_last_state → last_state (B, D, N)
    """
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn

    # (B, L, H) → (B, H, L)
    u = xin.transpose(1, 2).contiguous()
    delta = dt.transpose(1, 2).contiguous()
    # (B, L, N) → (B, N, L)
    B = b_matrix.transpose(1, 2).contiguous()
    C = c_matrix.transpose(1, 2).contiguous()
    A = a_matrix.contiguous()  # (H, N) — already negative
    D = d_skip.contiguous()

    # Seed initial state if provided (mamba_ssm uses zeros unless we inject)
    # selective_scan_fn does not take h_init directly; we run with zeros and
    # only use return_last_state for AR. For non-zero h_init, fall back is safer.
    if h_init is not None and float(h_init.abs().sum()) > 0:
        # Manual one-step unrolling from h_init is complex; use reference path
        # by raising and letting caller fall back — or do a pure pytorch prefix.
        # For training (h_init zeros) this branch is unused.
        from .ssm import _selective_scan_impl
        return _selective_scan_impl(xin, b_matrix, c_matrix, dt, a_matrix, d_skip, h_init)

    y, last_state = selective_scan_fn(
        u,
        delta,
        A,
        B,
        C,
        D=D.float() if D is not None else None,
        z=None,
        delta_bias=None,
        delta_softplus=False,  # dt already softplus'd in SelectiveSSM
        return_last_state=True,
    )
    # y: (B, H, L) → (B, L, H)
    y = y.transpose(1, 2).contiguous().to(xin.dtype)
    # last_state: (B, H, N)
    return y, last_state.to(h_init.dtype if h_init is not None else y.dtype)


def register_mamba_ssm_backend(*, name: str = "mamba_ssm") -> bool:
    """
    Register mamba-ssm scan if available. Returns True if registered.
    """
    try:
        import mamba_ssm  # noqa: F401
        from mamba_ssm.ops.selective_scan_interface import selective_scan_fn  # noqa: F401
    except ImportError:
        return False
    register_scan_backend(name, _mamba_selective_scan)
    return True


def try_enable_mamba_backend(env_default: bool = True) -> Optional[str]:
    """
    Register backend and optionally set DAPH_SCAN_BACKEND if unset.
    Returns backend name if enabled, else None.
    """
    import os

    ok = register_mamba_ssm_backend()
    if not ok:
        return None
    if env_default and not os.environ.get("DAPH_SCAN_BACKEND"):
        os.environ["DAPH_SCAN_BACKEND"] = "mamba_ssm"
    return "mamba_ssm"


# Auto-register on import if package present (does not force env)
register_mamba_ssm_backend()

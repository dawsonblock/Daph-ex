"""Input-dependent full-rank channel gates (Kimi K3 style)."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from .norms import RMSNorm


class ChannelGate(nn.Module):
    """
    y = W_o [ σ(W_g x) ⊙ Norm(o) ]

    identity_init(): near-pass-through of branch_output (for imported backbone).
    zero_out_init(): near-zero contribution (for new ExFusion branches).
    """

    def __init__(self, hidden_size: int, use_norm: bool = True, norm_type: str = "rms") -> None:
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, hidden_size, bias=True)
        self.out_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.use_norm = use_norm
        if use_norm:
            self.norm = RMSNorm(hidden_size) if norm_type == "rms" else nn.LayerNorm(hidden_size)
        else:
            self.norm = nn.Identity()

    def identity_init(self) -> None:
        """Initialize so y ≈ branch_output (sigmoid≈1, W_o≈I, norm scale≈1)."""
        nn.init.zeros_(self.gate_proj.weight)
        nn.init.constant_(self.gate_proj.bias, 4.0)  # sigmoid(4) ≈ 0.98
        nn.init.eye_(self.out_proj.weight)
        if hasattr(self.norm, "weight"):
            nn.init.ones_(self.norm.weight)
        if hasattr(self.norm, "bias") and self.norm.bias is not None:
            nn.init.zeros_(self.norm.bias)

    def zero_out_init(self) -> None:
        """Initialize so y ≈ 0 (suppress branch until trained)."""
        nn.init.zeros_(self.out_proj.weight)
        if self.gate_proj.bias is not None:
            nn.init.zeros_(self.gate_proj.bias)

    def forward(self, x: Tensor, branch_output: Tensor) -> Tensor:
        gate = torch.sigmoid(self.gate_proj(x))
        gated = gate * self.norm(branch_output)
        return self.out_proj(gated)

"""Normalization layers — LayerNorm and RMSNorm (Qwen/LLaMA style)."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization (no mean centering, no bias)."""

    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        # x: (..., H)
        variance = x.pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return self.weight * x


def make_norm(hidden_size: int, norm_type: str = "layer", eps: float = 1e-6) -> nn.Module:
    if norm_type in ("rms", "rmsnorm"):
        return RMSNorm(hidden_size, eps=eps)
    return nn.LayerNorm(hidden_size, eps=eps)

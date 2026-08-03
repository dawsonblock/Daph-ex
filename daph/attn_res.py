"""
Block Attention Residuals (AttnRes) — depth mixing.

Inspired by Kimi K3 Attention Residuals:
  - Selective retrieval from prior block representations
  - Learned pseudo-queries + RMSNorm keys
  - Softmax over depth dimension
  - Allows non-destructive access to earlier computational stages

This is complementary to the residual stream: instead of forcing every
layer to carry all previous information, later blocks can *query*
earlier states when needed.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class RMSNorm(nn.Module):
    """Lightweight RMSNorm used for key normalization in AttnRes."""

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        # x: (..., D)
        variance = x.pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return x * self.weight


class BlockAttnRes(nn.Module):
    """
    Block-level Attention Residual.

    Maintains a bank of previous block outputs and lets the current
    representation selectively retrieve from them via a small attention
    over the depth axis.

    Args:
        hidden_size: model dimension
        max_blocks: maximum number of historical blocks to keep
        num_heads: number of depth-attention heads (usually small)
    """

    def __init__(
        self,
        hidden_size: int,
        max_blocks: int = 8,
        num_heads: int = 4,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if hidden_size % num_heads != 0:
            raise ValueError("hidden_size must be divisible by num_heads")
        self.hidden_size = hidden_size
        self.max_blocks = max_blocks
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        # Pseudo-query projection from current state
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        # Key / value projections applied to historical block states
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)

        self.key_norm = RMSNorm(self.head_dim)
        self.out_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.dropout = nn.Dropout(dropout)

        # Learnable temperature for depth softmax (stability)
        self.logit_scale = nn.Parameter(torch.tensor(1.0))

    def forward(
        self,
        current: Tensor,
        history: List[Tensor],
    ) -> Tuple[Tensor, Tensor]:
        """
        Args:
            current: (B, L, H) — representation after the main block ops
            history: list of previous block outputs, each (B, L, H)
                     (most recent last). Empty list → identity.

        Returns:
            retrieved: (B, L, H) — depth-mixed contribution
            attn_weights: (B, num_heads, L, T) — optional diagnostics
        """
        if not history:
            # No history yet → zero contribution (caller decides residual)
            return torch.zeros_like(current), torch.empty(0)

        # Stack history: (T, B, L, H) → (B, T, L, H)
        hist = torch.stack(history[-self.max_blocks :], dim=0)  # (T, B, L, H)
        T = hist.shape[0]
        B, L, H = current.shape
        hist = hist.permute(1, 0, 2, 3)  # (B, T, L, H)

        # Project
        q = self.q_proj(current)                    # (B, L, H)
        k = self.k_proj(hist)                       # (B, T, L, H)
        v = self.v_proj(hist)                       # (B, T, L, H)

        # Multi-head reshape
        def split_heads(x: Tensor, extra: int = 0) -> Tensor:
            # x: (B, ..., L, H) → (B, num_heads, ..., L, head_dim)
            shape = list(x.shape)
            shape[-1] = self.num_heads
            shape.append(self.head_dim)
            x = x.view(*shape)
            # move heads right after batch
            dims = list(range(len(shape)))
            # for (B, T, L, heads, d) we want (B, heads, T, L, d)
            if extra == 1:  # history case
                return x.permute(0, 3, 1, 2, 4)
            # current: (B, L, heads, d) → (B, heads, L, d)
            return x.permute(0, 2, 1, 3)

        q = split_heads(q)                          # (B, heads, L, d)
        k = split_heads(k, extra=1)                 # (B, heads, T, L, d)
        v = split_heads(v, extra=1)                 # (B, heads, T, L, d)

        # RMSNorm keys (per-head)
        k = self.key_norm(k)

        # Depth attention: for each position, attend over the T historical blocks
        # scores: (B, heads, L, T)
        # q: (B, h, L, d)  k: (B, h, T, L, d)
        # We want similarity between current position and the same position
        # across depth → contract over head_dim after aligning L.
        scores = torch.einsum("bhld,bhtld->bhlt", q, k) * self.logit_scale
        scores = scores / (self.head_dim ** 0.5)
        attn = F.softmax(scores, dim=-1)            # (B, heads, L, T)
        attn = self.dropout(attn)

        # Weighted sum of values over depth
        # attn: (B, h, L, T)  v: (B, h, T, L, d) → (B, h, L, d)
        out = torch.einsum("bhlt,bhtld->bhld", attn, v)
        out = out.permute(0, 2, 1, 3).contiguous().view(B, L, H)
        out = self.out_proj(out)

        return out, attn


class AttnResBank(nn.Module):
    """
    Convenience wrapper that owns the history list and applies BlockAttnRes.

    Typical usage inside a multi-layer model:

        bank = AttnResBank(config)
        ...
        for block in layers:
            x, meta = block(x, ...)
            x = x + bank(x)          # or gated residual
            bank.push(x)
    """

    def __init__(
        self,
        hidden_size: int,
        max_blocks: int = 8,
        num_heads: int = 4,
        dropout: float = 0.0,
        gate: bool = True,
        detach_history: bool = True,
    ) -> None:
        super().__init__()
        self.attn_res = BlockAttnRes(
            hidden_size=hidden_size,
            max_blocks=max_blocks,
            num_heads=num_heads,
            dropout=dropout,
        )
        self.gate = None
        if gate:
            from .gates import ChannelGate
            self.gate = ChannelGate(hidden_size)
        self.detach_history = detach_history
        self._history: List[Tensor] = []

    def reset(self) -> None:
        self._history = []

    def push(self, block_out: Tensor) -> None:
        """Store a block output (call after residual has been applied)."""
        if self.detach_history:
            self._history.append(block_out.detach())
        else:
            self._history.append(block_out)

    def forward(self, current: Tensor) -> Tensor:
        retrieved, _ = self.attn_res(current, self._history)
        if self.gate is not None:
            return self.gate(current, retrieved)
        return retrieved

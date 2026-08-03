"""
Kimi Delta Attention (KDA) — recurrent alternative to SelectiveSSM.

v3.1.1: full streaming state = (S, conv_k_hist, conv_v_hist).
Short causal conv history is part of the cache so incremental ≡ full-sequence.
"""

from __future__ import annotations

from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


def lower_bound_decay(raw: Tensor, g_min: float = -5.0) -> Tensor:
    """Map unconstrained logits → forget gate in (sigmoid(g_min), 1]."""
    return torch.sigmoid(g_min + F.softplus(raw))


# State: (matrix S, conv_k history, conv_v history)
KDAState = Tuple[Tensor, Optional[Tensor], Optional[Tensor]]


class KimiDeltaAttention(nn.Module):
    """
    Single KDA layer with complete streaming state.

    next_state = (S, conv_k_hist, conv_v_hist)
    so AR incremental matches full-sequence execution.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int = 4,
        head_dim: Optional[int] = None,
        g_min: float = -5.0,
        expand_k: int = 1,
        expand_v: int = 1,
        dropout: float = 0.0,
        conv_kernel: int = 3,
    ) -> None:
        super().__init__()
        if head_dim is None:
            if hidden_size % num_heads != 0:
                raise ValueError("hidden_size must be divisible by num_heads")
            head_dim = hidden_size // num_heads
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.g_min = g_min
        self.k_dim = head_dim * expand_k
        self.v_dim = head_dim * expand_v
        self.conv_kernel = conv_kernel

        self.q_proj = nn.Linear(hidden_size, num_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, num_heads * self.k_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, num_heads * self.v_dim, bias=False)
        self.g_proj = nn.Linear(hidden_size, num_heads * self.v_dim, bias=True)
        self.o_proj_gate = nn.Linear(hidden_size, num_heads * self.v_dim, bias=True)
        self.out_proj = nn.Linear(num_heads * self.v_dim, hidden_size, bias=False)

        self.norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)

        self.use_short_conv = conv_kernel > 1
        if self.use_short_conv:
            self.conv_k = nn.Conv1d(
                num_heads * self.k_dim,
                num_heads * self.k_dim,
                kernel_size=conv_kernel,
                padding=0,
                groups=num_heads * self.k_dim,
                bias=False,
            )
            self.conv_v = nn.Conv1d(
                num_heads * self.v_dim,
                num_heads * self.v_dim,
                kernel_size=conv_kernel,
                padding=0,
                groups=num_heads * self.v_dim,
                bias=False,
            )
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        for m in (self.q_proj, self.k_proj, self.v_proj, self.out_proj):
            nn.init.xavier_uniform_(m.weight)
        nn.init.zeros_(self.g_proj.weight)
        nn.init.constant_(self.g_proj.bias, 1.0)
        nn.init.zeros_(self.o_proj_gate.weight)
        nn.init.constant_(self.o_proj_gate.bias, 0.0)

    def _shape(self, x: Tensor, dim: int) -> Tensor:
        B, L, _ = x.shape
        return x.view(B, L, self.num_heads, dim).transpose(1, 2)

    def _unpack_state(
        self, state: Optional[Union[Tensor, KDAState]], B: int, device, dtype
    ) -> KDAState:
        if state is None:
            S = torch.zeros(
                B, self.num_heads, self.v_dim, self.k_dim, device=device, dtype=dtype
            )
            return S, None, None
        if isinstance(state, tuple):
            return state[0], state[1], state[2]
        return state, None, None

    def _causal_conv(
        self,
        proj: Tensor,
        conv: nn.Conv1d,
        hist: Optional[Tensor],
        mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        """
        Causal depthwise conv with explicit left history.
        Mask-aware: padded tokens do NOT advance convolution history.
        proj: (B, L, C); mask: (B, L) with 1=valid.
        """
        B, L, C = proj.shape
        k = self.conv_kernel
        if hist is None:
            hist = torch.zeros(B, C, k - 1, device=proj.device, dtype=proj.dtype)
        if mask is None:
            mask = torch.ones(B, L, device=proj.device, dtype=proj.dtype)
        else:
            mask = mask.to(device=proj.device, dtype=proj.dtype)

        # Sequential update so padding is a true no-op on hist
        outs = []
        h = hist
        for t in range(L):
            xt = proj[:, t, :].unsqueeze(-1)  # (B, C, 1)
            window = torch.cat([h, xt], dim=2)  # (B, C, k)
            yt = conv(window)  # (B, C, 1)
            outs.append(yt.squeeze(-1))
            m = mask[:, t].view(B, 1, 1)
            # only shift hist when valid
            h = torch.where(m > 0.5, window[:, :, 1:].contiguous(), h)
        y = torch.stack(outs, dim=1)  # (B, L, C)
        return y, h

    def forward(
        self,
        x: Tensor,
        state: Optional[Union[Tensor, KDAState]] = None,
        mask: Optional[Tensor] = None,
        bypass_decay: float = 0.0,
    ) -> Tuple[Tensor, KDAState]:
        if x.dim() != 3:
            raise ValueError(f"KDA expects (B, L, H); got {tuple(x.shape)}")
        B, L, H = x.shape
        x_n = self.norm(x)

        if mask is None:
            mask = torch.ones(B, L, device=x.device, dtype=x.dtype)
        else:
            mask = mask.to(dtype=x.dtype)

        q = self._shape(self.q_proj(x_n), self.head_dim)
        k_raw = self.k_proj(x_n)
        v_raw = self.v_proj(x_n)

        S, hist_k, hist_v = self._unpack_state(state, B, x.device, x.dtype)

        if self.use_short_conv:
            k_raw, hist_k = self._causal_conv(k_raw, self.conv_k, hist_k, mask=mask)
            v_raw, hist_v = self._causal_conv(v_raw, self.conv_v, hist_v, mask=mask)

        k = self._shape(k_raw, self.k_dim)
        v = self._shape(v_raw, self.v_dim)
        k = F.normalize(k, dim=-1)
        q = F.normalize(q, dim=-1)

        g_raw = self._shape(self.g_proj(x_n), self.v_dim)
        g = lower_bound_decay(g_raw, self.g_min)
        o_gate = torch.sigmoid(self._shape(self.o_proj_gate(x_n), self.v_dim))

        outputs = []
        s = S
        for t in range(L):
            m_t = mask[:, t].view(B, 1, 1, 1)
            g_t = g[:, :, t, :].unsqueeze(-1)
            k_t = k[:, :, t, :].unsqueeze(-2)
            v_t = v[:, :, t, :].unsqueeze(-1)
            q_t = q[:, :, t, :]

            write = v_t * k_t
            s_new = g_t * s + (1.0 - g_t) * write
            if bypass_decay == 0.0:
                s = torch.where(m_t > 0.5, s_new, s)
            else:
                s = m_t * s_new + (1.0 - m_t) * (bypass_decay * s)

            if self.k_dim == self.head_dim:
                q_for_read = q_t.unsqueeze(-1)
            else:
                q_for_read = F.adaptive_avg_pool1d(
                    q_t.unsqueeze(1), self.k_dim
                ).squeeze(1).unsqueeze(-1)

            y_t = (s @ q_for_read).squeeze(-1)
            y_t = y_t * o_gate[:, :, t, :]
            outputs.append(y_t)

        y = torch.stack(outputs, dim=2)
        y = y.transpose(1, 2).contiguous().view(B, L, self.num_heads * self.v_dim)
        y = self.out_proj(y)
        y = self.dropout(y)

        return y, (s, hist_k, hist_v)


KDA = KimiDeltaAttention

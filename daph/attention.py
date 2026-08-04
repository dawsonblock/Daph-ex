"""
Causal self-attention with KV cache, optional GQA and RoPE (Qwen/LLaMA compatible).
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


def rotate_half(x: Tensor) -> Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q: Tensor, k: Tensor, cos: Tensor, sin: Tensor) -> Tuple[Tensor, Tensor]:
    # q,k: (B, H, L, D); cos/sin: (1, 1, L, D) or (L, D)
    while cos.dim() < q.dim():
        cos = cos.unsqueeze(0)
        sin = sin.unsqueeze(0)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, max_position: int = 8192, base: float = 10000.0) -> None:
        super().__init__()
        self.dim = dim
        self.max_position = max_position
        self.base = base
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._build_cache(max_position)

    def _build_cache(self, seq_len: int) -> None:
        t = torch.arange(seq_len, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :], persistent=False)
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :], persistent=False)

    def forward(self, seq_len: int, position_offset: int = 0) -> Tuple[Tensor, Tensor]:
        end = position_offset + seq_len
        if end > self.cos_cached.shape[2]:
            self._build_cache(end + 64)
        cos = self.cos_cached[:, :, position_offset:end, :]
        sin = self.sin_cached[:, :, position_offset:end, :]
        return cos, sin


class CausalSelfAttention(nn.Module):
    """
    Multi-head or grouped-query causal attention with optional RoPE and KV cache.

    When num_key_value_heads < num_heads, GQA is used (Qwen-style).
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_key_value_heads: Optional[int] = None,
        dropout: float = 0.0,
        bias: bool = False,
        out_bias: Optional[bool] = None,
        use_rope: bool = False,
        rope_theta: float = 10000.0,
        max_position: int = 8192,
    ) -> None:
        super().__init__()
        if hidden_size % num_heads != 0:
            raise ValueError(
                f"hidden_size ({hidden_size}) must be divisible by num_heads ({num_heads})"
            )
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_key_value_heads = num_key_value_heads or num_heads
        if self.num_heads % self.num_key_value_heads != 0:
            raise ValueError("num_heads must be divisible by num_key_value_heads")
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.head_dim = hidden_size // num_heads
        self.use_rope = use_rope

        output_bias = bias if out_bias is None else out_bias
        self.q_proj = nn.Linear(hidden_size, self.num_heads * self.head_dim, bias=bias)
        self.k_proj = nn.Linear(hidden_size, self.num_key_value_heads * self.head_dim, bias=bias)
        self.v_proj = nn.Linear(hidden_size, self.num_key_value_heads * self.head_dim, bias=bias)
        self.out_proj = nn.Linear(self.num_heads * self.head_dim, hidden_size, bias=output_bias)
        self.dropout = nn.Dropout(dropout)
        self.attn_dropout = nn.Dropout(dropout)

        if use_rope:
            # RoPE applied to full head_dim (even); if odd, last dim unrotated in practice
            rope_dim = self.head_dim if self.head_dim % 2 == 0 else self.head_dim - 1
            self.rotary = RotaryEmbedding(rope_dim, max_position=max_position, base=rope_theta)
            self.rope_dim = rope_dim
        else:
            self.rotary = None
            self.rope_dim = 0

    def _shape_q(self, x: Tensor) -> Tensor:
        B, L, _ = x.shape
        return x.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)

    def _shape_kv(self, x: Tensor) -> Tensor:
        B, L, _ = x.shape
        return x.view(B, L, self.num_key_value_heads, self.head_dim).transpose(1, 2)

    def _repeat_kv(self, x: Tensor) -> Tensor:
        if self.num_key_value_groups == 1:
            return x
        # (B, n_kv, L, D) → (B, n_heads, L, D)
        return (
            x[:, :, None, :, :]
            .expand(-1, -1, self.num_key_value_groups, -1, -1)
            .reshape(x.shape[0], self.num_heads, x.shape[2], self.head_dim)
        )

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Optional[Tensor] = None,
        past_k: Optional[Tensor] = None,
        past_v: Optional[Tensor] = None,
        use_cache: bool = False,
        position_offset: int = 0,
    ) -> Tuple[Tensor, Optional[Tensor], Optional[Tensor]]:
        B, L, _ = hidden_states.shape
        q = self._shape_q(self.q_proj(hidden_states))
        k = self._shape_kv(self.k_proj(hidden_states))
        v = self._shape_kv(self.v_proj(hidden_states))

        if self.use_rope and self.rotary is not None:
            # RoPE only on first rope_dim dims if head_dim odd
            if self.rope_dim == self.head_dim:
                cos, sin = self.rotary(L, position_offset=position_offset)
                # expand cos/sin to full sequence for k if past exists after cat
                q, k = apply_rotary_pos_emb(q, k, cos, sin)
            else:
                cos, sin = self.rotary(L, position_offset=position_offset)
                q_r, k_r = apply_rotary_pos_emb(
                    q[..., : self.rope_dim], k[..., : self.rope_dim], cos, sin
                )
                q = torch.cat([q_r, q[..., self.rope_dim :]], dim=-1)
                k = torch.cat([k_r, k[..., self.rope_dim :]], dim=-1)

        if past_k is not None:
            k = torch.cat([past_k, k], dim=2)
            v = torch.cat([past_v, v], dim=2)

        present_k = k if use_cache else None
        present_v = v if use_cache else None

        k_rep = self._repeat_kv(k)
        v_rep = self._repeat_kv(v)

        scale = self.head_dim ** -0.5
        attn = torch.matmul(q, k_rep.transpose(-2, -1)) * scale

        Lq = q.shape[2]
        Lk = k_rep.shape[2]
        q_pos = torch.arange(Lk - Lq, Lk, device=attn.device)
        k_pos = torch.arange(Lk, device=attn.device)
        causal = k_pos.unsqueeze(0) > q_pos.unsqueeze(1)
        attn = attn.masked_fill(causal.view(1, 1, Lq, Lk), float("-inf"))

        if attention_mask is not None:
            if attention_mask.shape[-1] == Lk:
                key_pad = ~attention_mask.bool()
                attn = attn.masked_fill(key_pad[:, None, None, :], float("-inf"))
            elif attention_mask.shape[-1] == L and past_k is None:
                key_pad = ~attention_mask.bool()
                attn = attn.masked_fill(key_pad[:, None, None, :], float("-inf"))

        attn = F.softmax(attn, dim=-1)
        attn = self.attn_dropout(attn)
        out = torch.matmul(attn, v_rep)
        out = out.transpose(1, 2).contiguous().view(B, L, self.num_heads * self.head_dim)
        out = self.out_proj(out)
        out = self.dropout(out)
        return out, present_k, present_v

"""
Strict Qwen-compatible decoder block for Phase-0 retention.

Implements exact pre-norm residual order:

  residual = x
  x = input_rmsnorm(x)
  x = rope_gqa_attn(x)
  x = residual + x
  residual = x
  x = post_attn_rmsnorm(x)
  x = shared_swiglu(x)
  x = residual + x

No recurrent path, no ChannelGate, no routed MoE, no extra final norm.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor

from .attention import CausalSelfAttention
from .latent_moe import SharedSwiGLU
from .norms import RMSNorm


class QwenCompatBlock(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_key_value_heads: int,
        intermediate_size: int,
        *,
        rope_theta: float = 10000.0,
        max_position: int = 8192,
        rms_eps: float = 1e-6,
        dropout: float = 0.0,
        attention_bias: bool = False,
        attention_output_bias: Optional[bool] = None,
    ) -> None:
        super().__init__()
        self.input_layernorm = RMSNorm(hidden_size, eps=rms_eps)
        self.self_attn = CausalSelfAttention(
            hidden_size,
            num_heads,
            num_key_value_heads=num_key_value_heads,
            dropout=dropout,
            bias=attention_bias,
            out_bias=attention_output_bias,
            use_rope=True,
            rope_theta=rope_theta,
            max_position=max_position,
        )
        self.post_attention_layernorm = RMSNorm(hidden_size, eps=rms_eps)
        self.mlp = SharedSwiGLU(hidden_size, intermediate_size, dropout=dropout)

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Optional[Tensor] = None,
        past_k: Optional[Tensor] = None,
        past_v: Optional[Tensor] = None,
        use_cache: bool = False,
        position_offset: int = 0,
    ) -> Tuple[Tensor, Optional[Tensor], Optional[Tensor]]:
        residual = hidden_states
        x = self.input_layernorm(hidden_states)
        attn_out, pk, pv = self.self_attn(
            x,
            attention_mask=attention_mask,
            past_k=past_k,
            past_v=past_v,
            use_cache=use_cache,
            position_offset=position_offset,
        )
        hidden_states = residual + attn_out

        residual = hidden_states
        x = self.post_attention_layernorm(hidden_states)
        mlp_out = self.mlp(x)
        hidden_states = residual + mlp_out
        return hidden_states, pk, pv


class QwenCompatModel(nn.Module):
    """Stack of QwenCompatBlocks + embed + lm_head for retention diagnostics."""

    def __init__(
        self,
        vocab_size: int,
        hidden_size: int,
        num_layers: int,
        num_heads: int,
        num_key_value_heads: int,
        intermediate_size: int,
        *,
        rope_theta: float = 10000.0,
        max_position: int = 8192,
        rms_eps: float = 1e-6,
        tie_word_embeddings: bool = True,
        attention_bias: bool = False,
        attention_output_bias: Optional[bool] = None,
    ) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden_size)
        self.layers = nn.ModuleList(
            [
                QwenCompatBlock(
                    hidden_size, num_heads, num_key_value_heads, intermediate_size,
                    rope_theta=rope_theta, max_position=max_position, rms_eps=rms_eps,
                    attention_bias=attention_bias,
                    attention_output_bias=attention_output_bias,
                )
                for _ in range(num_layers)
            ]
        )
        self.norm = RMSNorm(hidden_size, eps=rms_eps)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)
        if tie_word_embeddings:
            self.lm_head.weight = self.embed.weight

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Optional[Tensor] = None,
    ) -> Tensor:
        h = self.embed(input_ids)
        for layer in self.layers:
            h, _, _ = layer(h, attention_mask=attention_mask)
        h = self.norm(h)
        return self.lm_head(h)

"""
LatentMoE — Kimi-K3-inspired sparse experts in a reduced latent space.

z = W_down(x)
u = Σ_{i ∈ TopK} p_i * E_i(z)
y = W_up( RMSNorm(u) ) + shared experts

Enhancements (K3):
  - SiTU-GLU: bounded GLU with separate soft-cap betas on gate / up branches
  - Quantile Balancing (QB): load balance via score-margin quantiles + expert bias
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


def situ_glu(gate: Tensor, up: Tensor, beta_gate: float = 1.5, beta_up: float = 1.5) -> Tensor:
    """
    SiTU-GLU: soft-capped GLU.

    gate_capped = beta_g * tanh(gate / beta_g)
    up_capped   = beta_u * tanh(up   / beta_u)
    return silu(gate_capped) * up_capped
    """
    if beta_gate > 0:
        gate = beta_gate * torch.tanh(gate / beta_gate)
    if beta_up > 0:
        up = beta_up * torch.tanh(up / beta_up)
    return F.silu(gate) * up


class ExpertFFN(nn.Module):
    """Expert operating in latent dimension. Supports SwiGLU or SiTU-GLU."""

    def __init__(
        self,
        latent_size: int,
        expansion: float = 2.0,
        activation: str = "swiglu",
        beta_gate: float = 1.5,
        beta_up: float = 1.5,
    ) -> None:
        super().__init__()
        hidden = int(latent_size * expansion)
        self.w1 = nn.Linear(latent_size, hidden, bias=False)  # gate
        self.w2 = nn.Linear(latent_size, hidden, bias=False)  # up
        self.w3 = nn.Linear(hidden, latent_size, bias=False)
        self.activation = activation
        self.beta_gate = beta_gate
        self.beta_up = beta_up

    def forward(self, x: Tensor) -> Tensor:
        gate = self.w1(x)
        up = self.w2(x)
        if self.activation == "situ":
            h = situ_glu(gate, up, self.beta_gate, self.beta_up)
        else:
            h = F.silu(gate) * up
        return self.w3(h)



class SharedSwiGLU(nn.Module):
    """Full-width SwiGLU FFN matching Qwen/LLaMA: down(silu(gate(x)) * up(x))."""

    def __init__(self, hidden_size: int, intermediate_size: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        return self.dropout(self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x)))


class LatentMoE(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        latent_size: int,
        num_routed_experts: int = 16,
        num_shared_experts: int = 2,
        top_k: int = 2,
        dropout: float = 0.0,
        activation: str = "swiglu",       # "swiglu" | "situ"
        beta_gate: float = 1.5,
        beta_up: float = 1.5,
        use_quantile_balancing: bool = True,
        qb_quantile: float = 0.75,
        qb_bias_lr: float = 0.01,
        intermediate_size: Optional[int] = None,
        shared_ffn: str = "swiglu",  # "swiglu" | "gelu_mlp"
    ) -> None:
        super().__init__()
        if latent_size > hidden_size:
            raise ValueError("latent_size should be ≤ hidden_size")
        if activation not in ("swiglu", "situ"):
            raise ValueError(f"activation must be 'swiglu' or 'situ'; got {activation}")

        self.hidden_size = hidden_size
        self.latent_size = latent_size
        self.num_routed = num_routed_experts
        self.num_shared = num_shared_experts
        self.top_k = top_k
        self.activation = activation
        self.use_qb = use_quantile_balancing
        self.qb_quantile = qb_quantile
        self.qb_bias_lr = qb_bias_lr

        # Down / up projections (latent routed path)
        self.down = nn.Linear(hidden_size, latent_size, bias=False)
        self.up = nn.Linear(latent_size, hidden_size, bias=False)

        inter = intermediate_size if intermediate_size is not None else hidden_size * 2
        self.intermediate_size = inter
        self.shared_ffn = shared_ffn
        if shared_ffn == "swiglu":
            self.shared = nn.ModuleList(
                [SharedSwiGLU(hidden_size, inter, dropout=dropout) for _ in range(num_shared_experts)]
            )
        else:
            self.shared = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Linear(hidden_size, inter, bias=False),
                        nn.GELU(),
                        nn.Linear(inter, hidden_size, bias=False),
                        nn.Dropout(dropout),
                    )
                    for _ in range(num_shared_experts)
                ]
            )

        # Routed experts in latent space
        self.routed = nn.ModuleList(
            [
                ExpertFFN(
                    latent_size,
                    activation=activation,
                    beta_gate=beta_gate,
                    beta_up=beta_up,
                )
                for _ in range(num_routed_experts)
            ]
        )

        # Router (on latent)
        self.router = nn.Linear(latent_size, num_routed_experts, bias=False)

        # Load-balancing bias: controller-managed buffer (NOT a gradient Parameter).
        # Manual integral feedback updates this buffer; no dual ownership with optimizer.
        self.register_buffer("expert_bias", torch.zeros(num_routed_experts), persistent=True)
        self.use_load_balance = use_quantile_balancing  # kept for API compat; see load_balance_step

        # Critical: RMSNorm before up-projection (K3 finding)
        self.post_norm = (
            nn.RMSNorm(latent_size) if hasattr(nn, "RMSNorm") else nn.LayerNorm(latent_size)
        )

        self.dropout = nn.Dropout(dropout)

    def _apply_load_balance_bias(
        self,
        router_logits: Tensor,
    ) -> Tensor:
        """
        Apply expert bias and (during training) update bias via quantile of
        score margins so that load stays balanced across experts.

        Score margin for expert e ≈ max_score − score_e (or simply the logit).
        We push bias so that the qb_quantile of each expert's scores approaches
        a common target.
        """
        # Add bias to logits for routing decisions
        biased = router_logits + self.expert_bias  # (B, L, E)

        if self.training and self.use_load_balance:
            with torch.no_grad():
                probs = F.softmax(biased, dim=-1)
                _, top_idx = torch.topk(probs, self.top_k, dim=-1)
                load = torch.zeros_like(probs)
                load.scatter_(-1, top_idx, 1.0)
                load_frac = load.mean(dim=(0, 1))  # (E,)
                target = 1.0 / self.num_routed
                # Integral load feedback: b <- b + η (L_target - L_e)
                delta = self.qb_bias_lr * (target - load_frac)
                self.expert_bias.add_(delta)

        return biased

    def forward(
        self,
        x: Tensor,
        return_router_logits: bool = False,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """
        x : (B, L, H)
        returns (output, router_logits or None)
        """
        # Shared path (full width)
        shared_out = torch.zeros_like(x)
        for expert in self.shared:
            shared_out = shared_out + expert(x)
        if self.num_shared > 0:
            shared_out = shared_out / self.num_shared

        # Latent path
        z = self.down(x)  # (B, L, latent)
        router_logits = self.router(z)  # (B, L, E)

        # Quantile Balancing bias
        if self.use_load_balance:
            routing_logits = self._apply_load_balance_bias(router_logits)
        else:
            routing_logits = router_logits

        routing_weights = F.softmax(routing_logits, dim=-1)

        # Top-k selection
        top_weights, top_indices = torch.topk(routing_weights, self.top_k, dim=-1)
        top_weights = top_weights / top_weights.sum(dim=-1, keepdim=True).clamp_min(1e-9)

        # ---- Genuine sparse dispatch (gather → expert → scatter) ----
        B, L, D = z.shape
        latent_flat = z.reshape(B * L, D)  # (N, D)
        top_idx_flat = top_indices.reshape(B * L, self.top_k)  # (N, K)
        top_w_flat = top_weights.reshape(B * L, self.top_k)  # (N, K)

        latent_out_flat = torch.zeros_like(latent_flat)
        expert_call_count = 0
        expert_token_evaluations = 0
        tokens_per_expert = []

        for e in range(self.num_routed):
            # tokens that selected expert e in any of their top-k slots
            # Build list of (token_index, slot_weight) for this expert
            sel_mask = top_idx_flat == e  # (N, K)
            if not sel_mask.any():
                tokens_per_expert.append(0)
                continue
            # For each token, sum weights if expert appears multiple times (rare)
            token_weight = (sel_mask.float() * top_w_flat).sum(dim=-1)  # (N,)
            token_ids = token_weight.nonzero(as_tuple=False).squeeze(-1)
            if token_ids.numel() == 0:
                tokens_per_expert.append(0)
                continue
            tokens_per_expert.append(int(token_ids.numel()))
            z_e = latent_flat.index_select(0, token_ids)  # (Ne, D)
            y_e = self.routed[e](z_e)  # only selected tokens
            w_e = token_weight.index_select(0, token_ids).unsqueeze(-1)  # (Ne, 1)
            latent_out_flat.index_add_(0, token_ids, w_e * y_e)
            expert_call_count += 1
            expert_token_evaluations += int(token_ids.numel())

        latent_out = latent_out_flat.view(B, L, D)

        # K3 critical fix: normalize before up-projection
        latent_out = self.post_norm(latent_out)
        routed_out = self.up(latent_out)
        routed_out = self.dropout(routed_out)

        output = shared_out + routed_out

        # Telemetry attached as attributes for callers (optional)
        self._last_telemetry = {
            "expert_call_count": expert_call_count,
            "expert_token_evaluations": expert_token_evaluations,
            "tokens_per_expert": tokens_per_expert,
            "top_k": self.top_k,
            "num_tokens": B * L,
        }

        if return_router_logits:
            return output, router_logits
        return output, None

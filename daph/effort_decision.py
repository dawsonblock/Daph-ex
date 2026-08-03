"""Structured effort decisions and compute accounting for v3.1.2."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch
from torch import Tensor


@dataclass
class EffortDecision:
    """Per-sample effort decision from the controller."""

    probabilities: Tensor  # (B, num_levels)
    levels: Tensor  # (B,) int64
    confidence: Tensor  # (B,) max prob
    entropy: Tensor  # (B,)
    source_layer: int = 0
    source_position: str = "post_probe"
    margin: Optional[Tensor] = None  # (B,) top1 - top2
    hidden_anchor: Optional[Tensor] = None  # (B, H) optional

    def to_dict(self) -> Dict:
        return {
            "levels": self.levels.tolist(),
            "confidence": self.confidence.tolist(),
            "entropy": self.entropy.tolist(),
            "source_layer": self.source_layer,
            "source_position": self.source_position,
            "margin": self.margin.tolist() if self.margin is not None else None,
        }


@dataclass
class ComputeStats:
    """Logical compute counters (prefer over estimated FLOPs for verification)."""

    effort_levels: Optional[Tensor] = None  # (B,)
    recurrent_token_evals: int = 0
    attention_token_evals: int = 0
    expert_token_evals: int = 0
    shared_expert_token_evals: int = 0
    latent_refine_token_evals: int = 0
    recurrent_iterations: int = 0
    latent_iterations: int = 0
    estimated_flops: float = 0.0
    # per-sample normalized compute relative to E3 (optional)
    per_sample_compute: Optional[Tensor] = None

    def to_dict(self) -> Dict:
        return {
            "effort_levels": self.effort_levels.tolist() if self.effort_levels is not None else None,
            "recurrent_token_evals": self.recurrent_token_evals,
            "attention_token_evals": self.attention_token_evals,
            "expert_token_evals": self.expert_token_evals,
            "shared_expert_token_evals": self.shared_expert_token_evals,
            "latent_refine_token_evals": self.latent_refine_token_evals,
            "recurrent_iterations": self.recurrent_iterations,
            "latent_iterations": self.latent_iterations,
            "estimated_flops": self.estimated_flops,
            "per_sample_compute": (
                self.per_sample_compute.tolist()
                if self.per_sample_compute is not None
                else None
            ),
        }

    def merge_from(self, other: "ComputeStats") -> None:
        self.recurrent_token_evals += other.recurrent_token_evals
        self.attention_token_evals += other.attention_token_evals
        self.expert_token_evals += other.expert_token_evals
        self.shared_expert_token_evals += other.shared_expert_token_evals
        self.latent_refine_token_evals += other.latent_refine_token_evals
        self.recurrent_iterations += other.recurrent_iterations
        self.latent_iterations += other.latent_iterations
        self.estimated_flops += other.estimated_flops


def decide_from_probs(
    probs: Tensor,
    source_layer: int = 0,
    source_position: str = "post_probe",
    hidden_anchor: Optional[Tensor] = None,
) -> EffortDecision:
    """Build EffortDecision from (B, E) probability tensor."""
    if probs.dim() != 2:
        raise ValueError(f"probs must be (B, E); got {tuple(probs.shape)}")
    levels = probs.argmax(dim=-1)
    conf, _ = probs.max(dim=-1)
    # entropy
    logp = torch.log(probs.clamp_min(1e-12))
    entropy = -(probs * logp).sum(dim=-1)
    # margin
    top2 = torch.topk(probs, k=min(2, probs.size(-1)), dim=-1).values
    margin = top2[:, 0] - top2[:, 1] if top2.size(-1) > 1 else conf
    return EffortDecision(
        probabilities=probs,
        levels=levels,
        confidence=conf,
        entropy=entropy,
        source_layer=source_layer,
        source_position=source_position,
        margin=margin,
        hidden_anchor=hidden_anchor,
    )


# Nominal relative compute weights for E0..E3 (for per-sample accounting)
NOMINAL_COMPUTE = {
    0: 0.30,  # recurrent only
    1: 0.48,  # + MoE top-1
    2: 0.72,  # + attention + full MoE
    3: 1.00,  # + latent refine
}


def nominal_compute_for_levels(levels: Tensor) -> Tensor:
    """Map effort levels → nominal relative compute in [0.3, 1.0]."""
    out = torch.zeros(levels.shape[0], dtype=torch.float32, device=levels.device)
    for e, c in NOMINAL_COMPUTE.items():
        out = torch.where(levels == e, torch.full_like(out, c), out)
    return out


def estimate_compute(
    *,
    batch_size: int,
    seq_len: int,
    hidden_size: int,
    num_layers: int,
    num_recurrent_per_block: int,
    effort_level: int,
    top_k: int = 2,
    num_shared_experts: int = 0,
    latent_steps: int = 0,
    latent_size: int = 0,
    force_skip_attention: bool = False,
    force_skip_moe: bool = False,
    recurrent_type: str = "ssm",
    past_seq_len: int = 0,
) -> "ComputeStats":
    """
    Single source of truth for logical counters + rough FLOP proxy.

    Invariant: same (config, B, L, effort_level) → identical ComputeStats
    whether called from fixed or adaptive paths.
    """
    B, L, H = batch_size, seq_len, hidden_size
    skip_attn = force_skip_attention or effort_level <= 1
    skip_moe = force_skip_moe or effort_level <= 0
    if effort_level <= 0:
        skip_attn = True
        skip_moe = True
        top_k_eff = 0
        lat = 0
    elif effort_level == 1:
        skip_attn = True
        skip_moe = False
        top_k_eff = 1
        lat = 0
    elif effort_level == 2:
        skip_attn = False
        skip_moe = False
        top_k_eff = top_k
        lat = 0
    else:
        skip_attn = False
        skip_moe = False
        top_k_eff = top_k
        lat = max(latent_steps, 1)

    rec = B * L * num_layers * num_recurrent_per_block
    attn = 0 if skip_attn else B * L * num_layers
    exp = 0 if skip_moe else B * L * num_layers * top_k_eff
    shared = 0 if skip_moe else B * L * num_layers * max(num_shared_experts, 0)
    lat_tok = B * L * lat * num_layers

    # Rough FLOP proxy — operator-aware enough for relative comparisons
    # Recurrent: KDA slightly heavier than SSM (extra gates/conv)
    rec_mult = 3.0 if recurrent_type == "kda" else 2.0
    # Attention score term uses full key length (past + new)
    key_len = max(past_seq_len + L, L)
    flops = (
        rec * H * H * rec_mult
        + attn * (H * H * 4 + key_len * H)  # proj + score over keys
        + exp * (max(latent_size, H // 2) * H * 4)  # expert MLP proxy
        + shared * (max(latent_size, H // 2) * H * 4)
        + lat_tok * H * H * 2
    )

    levels = None  # caller sets
    stats = ComputeStats(
        recurrent_token_evals=rec,
        attention_token_evals=attn,
        expert_token_evals=exp,
        shared_expert_token_evals=shared,
        latent_refine_token_evals=lat_tok,
        recurrent_iterations=num_layers * num_recurrent_per_block,
        latent_iterations=lat * num_layers,
        estimated_flops=float(flops),
    )
    return stats

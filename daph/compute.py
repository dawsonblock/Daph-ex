"""Deterministic compute accounting for the canonical QwenExFusion path."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional


@dataclass
class EffortComputeReceipt:
    effort_mode: str
    executed_layer_count: int
    skipped_layer_count: int
    attention_calls: int
    ffn_calls: int
    recurrent_steps: int
    latent_steps: int
    routed_expert_calls: int
    token_count: int
    raw_compute_units: float = 0.0
    normalized_compute_cost: float = 0.0
    wall_clock_latency_ms: Optional[float] = None
    peak_memory_bytes: Optional[int] = None
    depth_fraction: float = 1.0
    refinement_insertion_layer: Optional[int] = None
    refinement_region_start: Optional[int] = None
    refinement_region_end: Optional[int] = None
    middle_refinement_steps: int = 0
    repeated_pretrained_layer_calls: int = 0
    middle_refiner_calls: int = 0
    selected_profile_digest: Optional[str] = None
    e3_variant: Optional[str] = None

    @property
    def estimated_compute(self) -> float:
        return self.raw_compute_units

    def to_dict(self) -> Dict[str, Any]:
        out = asdict(self)
        out["estimated_compute"] = self.estimated_compute
        # This is a deterministic operator-family proxy, not measured device
        # FLOPs.  Keep the name precise so reports do not overclaim accuracy.
        out["estimated_compute_units"] = self.raw_compute_units
        out["layers_executed"] = self.executed_layer_count
        out["layers_skipped"] = self.skipped_layer_count
        return out


def estimate_compute(
    receipt: EffortComputeReceipt,
    *,
    hidden_size: int,
    intermediate_size: int,
    num_heads: int,
    num_key_value_heads: int,
    sequence_length: int,
    batch_size: int = 1,
    num_routed_experts: int = 0,
) -> float:
    """Internally consistent multiply-add proxy; deliberately hardware-neutral."""
    h = hidden_size
    t = batch_size * sequence_length
    head_dim = h // num_heads
    kv = num_key_value_heads * head_dim
    # Q/K/V/O projections plus causal score and value products.
    attention = t * (h * h + 2 * h * kv + h * h) + batch_size * sequence_length**2 * h * 2
    # SwiGLU gate/up/down projections.
    ffn = t * (3 * h * intermediate_size)
    recurrent = t * (3 * h * h + 2 * h * max(1, head_dim))
    latent = t * (4 * h * h)
    router = t * h * max(1, num_routed_experts)
    expert = t * (3 * h * intermediate_size)
    return float(
        receipt.attention_calls * attention
        + receipt.ffn_calls * ffn
        + receipt.recurrent_steps * recurrent
        + receipt.latent_steps * latent
        + receipt.routed_expert_calls * expert
        + (receipt.routed_expert_calls > 0) * router
    )

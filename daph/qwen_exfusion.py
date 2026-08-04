"""
QwenExFusion — preserved Qwen backbone + zero-scaled ExFusion augmentations.

Invariant (Gate 0B):
  With all augmentation scales at 0 and effort fixed_E2,
  logits(QwenExFusion) ≈ logits(QwenCompat).

Augmentations never rewrite the imported Qwen residual graph.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Mapping, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch import Tensor

from .qwen_compat import QwenCompatBlock, QwenCompatModel
from .ssm import SelectiveSSM
from .kda import KimiDeltaAttention
from .latent_moe import LatentMoE
from .latent_refine import LatentRefineBlock
from .attn_res import BlockAttnRes
from .norms import RMSNorm
from .compute import EffortComputeReceipt, estimate_compute
from .effort import EffortController
from .effort_decision import EffortDecision, decide_from_probs
from .e3_architecture import E3RefinementConfig, E3RegionSelection, resolve_e3_region


@dataclass(frozen=True)
class ExFusionParameterProvenance:
    imported_parameter_names: Tuple[str, ...]
    new_parameter_names: Tuple[str, ...]
    augmentation_parameter_names: Tuple[str, ...]
    scale_parameter_names: Tuple[str, ...]
    continuation_parameter_names: Tuple[str, ...] = ()
    e3_refinement_parameter_names: Tuple[str, ...] = ()
    e3_scale_parameter_names: Tuple[str, ...] = ()
    e3_middle_layer_parameter_names: Tuple[str, ...] = ()

    @staticmethod
    def _digest(names: Tuple[str, ...]) -> str:
        return hashlib.sha256("\n".join(names).encode()).hexdigest()

    def to_dict(self) -> Dict[str, Any]:
        out = asdict(self)
        for key, value in list(out.items()):
            out[key] = list(value)
            out[f"{key}_digest"] = self._digest(tuple(value))
        return out


@dataclass(frozen=True)
class TrainingInitReceipt:
    gate0b_passed: bool
    epsilon: float
    changed_scale_names: Tuple[str, ...]
    backbone_unchanged: bool


@dataclass
class EffortProbeResult:
    probe_hidden: Tensor
    probe_layer: int
    executed_layers: int
    partial_hidden_state: Tensor
    compute_receipt: EffortComputeReceipt
    decision: EffortDecision
    policy_logits: Optional[Tensor] = None

    def __iter__(self):
        # Backward-compatible unpacking used by the existing collector.
        yield self.probe_hidden
        yield None
        yield self.decision


class QwenExFusionBlock(nn.Module):
    """
    base = exact QwenCompatBlock path
    out  = base
         + rec_scale    * recurrent(x)
         + moe_scale    * routed_moe(base)
         + attnres_scale * attn_res(...)
         + (E3 only) latent_scale * refine(out)
    """

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
        attention_bias: bool = False,
        attention_output_bias: Optional[bool] = None,
        recurrent_type: str = "ssm",
        state_size: int = 16,
        latent_size: Optional[int] = None,
        num_routed_experts: int = 4,
        top_k: int = 2,
        use_attn_res: bool = False,
        dropout: float = 0.0,
        latent_scale_limit: float = 0.01,
    ) -> None:
        super().__init__()
        H = hidden_size
        latent_size = latent_size or max(8, H // 2)

        self.base = QwenCompatBlock(
            H, num_heads, num_key_value_heads, intermediate_size,
            rope_theta=rope_theta, max_position=max_position,
            rms_eps=rms_eps, attention_bias=attention_bias,
            attention_output_bias=attention_output_bias,
        )

        # Recurrent augmentation (SSM or KDA)
        self.recurrent_type = recurrent_type
        if recurrent_type == "kda":
            self.recurrent = KimiDeltaAttention(H, num_heads=max(1, H // 16), dropout=dropout)
        else:
            self.recurrent = SelectiveSSM(H, state_size=state_size)

        # Routed latent MoE only (shared FFN lives in base.mlp)
        self.routed_moe = LatentMoE(
            hidden_size=H,
            latent_size=latent_size,
            num_routed_experts=num_routed_experts,
            num_shared_experts=0,  # preserved SwiGLU is in base
            top_k=top_k,
            dropout=dropout,
            activation="swiglu",
            use_quantile_balancing=False,
            intermediate_size=intermediate_size,
            shared_ffn="swiglu",
        )

        self.use_attn_res = use_attn_res
        self.attn_res = BlockAttnRes(hidden_size=H) if use_attn_res else None
        self.latent_refine = LatentRefineBlock(H, expansion=2.0, dropout=dropout)
        if latent_scale_limit <= 0:
            raise ValueError("latent_scale_limit must be positive")
        self.latent_scale_limit = float(latent_scale_limit)

        # Exact-zero residual scales (scalar)
        self.rec_scale = nn.Parameter(torch.zeros(()))
        self.moe_scale = nn.Parameter(torch.zeros(()))
        self.attn_res_scale = nn.Parameter(torch.zeros(()))
        self.latent_scale = nn.Parameter(torch.zeros(()))

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Optional[Tensor] = None,
        past_k: Optional[Tensor] = None,
        past_v: Optional[Tensor] = None,
        use_cache: bool = False,
        position_offset: int = 0,
        *,
        use_recurrent: bool = True,
        use_routed_moe: bool = True,
        use_attn_res: bool = True,
        latent_steps: int = 0,
        recurrent_state: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Optional[Tensor], Optional[Tensor], Optional[Tensor]]:
        # Preserved Qwen path
        base_out, pk, pv = self.base(
            hidden_states,
            attention_mask=attention_mask,
            past_k=past_k,
            past_v=past_v,
            use_cache=use_cache,
            position_offset=position_offset,
        )
        out = base_out
        new_rec_state = None

        if use_recurrent:
            if self.recurrent_type == "kda":
                rec_out, new_rec_state = self.recurrent(hidden_states, state=recurrent_state)
            else:
                rec_out, new_rec_state = self.recurrent(hidden_states, state=recurrent_state)
            out = out + self.rec_scale * rec_out

        if use_routed_moe:
            moe_out, _ = self.routed_moe(base_out)
            out = out + self.moe_scale * moe_out

        if use_attn_res and self.attn_res is not None:
            ar, _ = self.attn_res(out, history=[])
            out = out + self.attn_res_scale * ar

        if latent_steps > 0:
            ref, _ = self.latent_refine(out, num_steps=int(latent_steps))
            # Refinement returns a representation containing its input.  Scale only
            # its learned delta so latent_scale=0 is a clean identity operation.
            # Bound the effective residual to prevent a raw scale update from
            # overwhelming the preserved pretrained representation.
            effective_scale = self.latent_scale_limit * torch.tanh(
                self.latent_scale / self.latent_scale_limit
            )
            out = out + effective_scale * (ref - out)

        return out, pk, pv, new_rec_state


class CheapContinuation(nn.Module):
    """Optional shallow-exit residual MLP, far cheaper than a skipped Qwen block."""

    def __init__(self, hidden_size: int, bottleneck_size: Optional[int] = None) -> None:
        super().__init__()
        inner = bottleneck_size or max(8, hidden_size // 4)
        self.norm = RMSNorm(hidden_size)
        self.up = nn.Linear(hidden_size, inner, bias=False)
        self.down = nn.Linear(inner, hidden_size, bias=False)
        nn.init.zeros_(self.down.weight)

    def forward(self, hidden: Tensor) -> Tensor:
        return hidden + self.down(torch.nn.functional.silu(self.up(self.norm(hidden))))


class QwenExFusionModel(nn.Module):
    """
    Stack of QwenExFusionBlocks + embed + lm_head.

    Effort modes (initial semantics):
      E2: base only (all scales effectively zero at init) — Qwen-equivalent
      E3: full base + configured bounded middle/final refinement experiment
      E1: configurable intermediate-depth exit
      E0: configurable shallow exit
    """

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
        recurrent_type: str = "ssm",
        state_size: int = 16,
        latent_size: Optional[int] = None,
        num_routed_experts: int = 4,
        top_k: int = 2,
        use_attn_res: bool = False,
        dropout: float = 0.0,
        default_e3_steps: int = 2,
        e0_depth_fraction: float = 0.50,
        e1_depth_fraction: float = 0.75,
        e2_depth_fraction: float = 1.00,
        e3_depth_fraction: float = 1.00,
        e0_layer_count: Optional[int] = None,
        e1_layer_count: Optional[int] = None,
        use_shallow_continuation: bool = False,
        continuation_bottleneck_size: Optional[int] = None,
        latent_scale_limit: float = 0.01,
        effort_controller_hidden_size: int = 128,
        enable_effort_controller: bool = True,
        effort_probe_layer: Optional[int] = None,
        effort_probe_fraction: float = 0.125,
        e3_config: Optional[E3RefinementConfig] = None,
    ) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden_size)
        self.layers = nn.ModuleList(
            [
                QwenExFusionBlock(
                    hidden_size, num_heads, num_key_value_heads, intermediate_size,
                    rope_theta=rope_theta, max_position=max_position, rms_eps=rms_eps,
                    attention_bias=attention_bias, recurrent_type=recurrent_type,
                    attention_output_bias=attention_output_bias,
                    state_size=state_size, latent_size=latent_size,
                    num_routed_experts=num_routed_experts, top_k=top_k,
                    use_attn_res=use_attn_res, dropout=dropout,
                    latent_scale_limit=latent_scale_limit,
                )
                for _ in range(num_layers)
            ]
        )
        self.norm = RMSNorm(hidden_size, eps=rms_eps)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)
        if tie_word_embeddings:
            self.lm_head.weight = self.embed.weight
        self.default_e3_steps = default_e3_steps
        self.e3_config = e3_config or E3RefinementConfig(e3_refine_steps=default_e3_steps)
        self.e3_config.validate(num_layers)
        self.e3_region: E3RegionSelection = resolve_e3_region(self.e3_config, num_layers)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_heads = num_heads
        self.num_key_value_heads = num_key_value_heads
        self.num_routed_experts = num_routed_experts
        self.top_k = top_k
        self.recurrent_type = recurrent_type
        self.state_size = state_size
        self.latent_size = latent_size or max(8, hidden_size // 2)
        self.rope_theta = rope_theta
        self.max_position = max_position
        self.rms_eps = rms_eps
        self.attention_bias = attention_bias
        self.attention_output_bias = attention_bias if attention_output_bias is None else attention_output_bias
        self.tie_word_embeddings = tie_word_embeddings
        self.continuation_bottleneck_size = continuation_bottleneck_size
        self.use_shallow_continuation = use_shallow_continuation
        self.latent_scale_limit = float(latent_scale_limit)
        self.effort_controller_hidden_size = int(effort_controller_hidden_size)
        self.enable_effort_controller = bool(enable_effort_controller)
        requested_probe = (
            int(effort_probe_layer) + 1
            if effort_probe_layer is not None
            else max(1, int(math.ceil(num_layers * effort_probe_fraction)))
        )
        self.effort_probe_fraction = float(effort_probe_fraction)
        self.effort_probe_layer_count = requested_probe
        self.effort_controller = (
            EffortController(hidden_size, num_levels=4, hidden_router=self.effort_controller_hidden_size)
            if self.enable_effort_controller else None
        )
        self.e0_continuation = CheapContinuation(hidden_size, continuation_bottleneck_size) if use_shallow_continuation else None
        self.e1_continuation = CheapContinuation(hidden_size, continuation_bottleneck_size) if use_shallow_continuation else None
        self.depth_fractions = (e0_depth_fraction, e1_depth_fraction, e2_depth_fraction, e3_depth_fraction)
        self.layer_count_overrides = (e0_layer_count, e1_layer_count, None, None)
        self.parameter_provenance: Optional[ExFusionParameterProvenance] = None
        self._verified_effort_policy = False
        self._effort_policy_artifact_digest: Optional[str] = None
        self._validate_depths()

    def _layer_count(self, effort: int) -> int:
        override = self.layer_count_overrides[effort]
        if override is not None:
            return max(1, min(len(self.layers), int(override)))
        return max(1, min(len(self.layers), int(math.ceil(len(self.layers) * self.depth_fractions[effort]))))

    def _validate_depths(self) -> None:
        counts = [self._layer_count(i) for i in range(4)]
        # Tiny one/two-layer fixtures remain constructible for parity tests; a
        # strict three-level depth hierarchy is mathematically impossible there.
        if len(self.layers) >= 3 and (counts[0] >= counts[1] or counts[1] >= counts[2]):
            raise ValueError(
                "Effort depths must satisfy E0 < E1 < E2. "
                f"Resolved layer counts are {counts}; default fractions require at least four layers, or use explicit overrides."
            )
        if counts[2] != len(self.layers) or counts[3] != len(self.layers):
            raise ValueError("E2 and E3 must execute the complete pretrained backbone")
        if not 1 <= self.effort_probe_layer_count <= counts[0]:
            raise ValueError(
                "Probe depth must satisfy 1 <= probe_layers <= E0 layers; "
                f"got probe={self.effort_probe_layer_count}, E0={counts[0]}"
            )

    def _effort_flags(self, effort_mode: str) -> Dict[str, object]:
        # At init all scales are 0, so flags only matter after training.
        if effort_mode in ("fixed_2", "e2", "2"):
            return dict(use_recurrent=False, use_routed_moe=False, use_attn_res=False, latent_steps=0)
        if effort_mode in ("fixed_3", "e3", "3"):
            steps = (
                self.e3_config.e3_refine_steps
                if self.e3_config.e3_refinement_mode in {
                    "final_refine", "middle_recurrent", "profiled_middle_recurrent"
                }
                else 0
            )
            return dict(use_recurrent=False, use_routed_moe=False, use_attn_res=False,
                        latent_steps=steps)
        if effort_mode in ("fixed_1", "e1", "1"):
            return dict(use_recurrent=False, use_routed_moe=False, use_attn_res=False, latent_steps=0)
        if effort_mode in ("fixed_0", "e0", "0"):
            return dict(use_recurrent=False, use_routed_moe=False, use_attn_res=False, latent_steps=0)
        # default E2
        return dict(use_recurrent=False, use_routed_moe=False, use_attn_res=False, latent_steps=0)

    @staticmethod
    def _effort_index(effort_mode: str) -> int:
        aliases = {"fixed_0": 0, "e0": 0, "0": 0, "fixed_1": 1, "e1": 1, "1": 1,
                   "fixed_2": 2, "e2": 2, "2": 2, "disabled": 2, "full": 2,
                   "fixed_3": 3, "e3": 3, "3": 3}
        key = str(effort_mode).lower()
        if key not in aliases:
            raise ValueError(f"Unknown effort mode {effort_mode!r}. Use fixed_0..fixed_3 or adaptive.")
        return aliases[key]

    def install_effort_policy(
        self, artifact: Any, state_dict: Dict[str, Tensor], *,
        base_model_digest: str, research_override: bool = False,
    ) -> None:
        if self.effort_controller is None:
            raise RuntimeError("This model was created without an effort controller")
        from .policy_trainer import install_effort_policy
        install_effort_policy(
            self.effort_controller, artifact, state_dict,
            base_model_digest=base_model_digest,
            require_verified_fit=not research_override,
        )
        verified = getattr(artifact, "training_status", None) == "VERIFIED_FIT"
        if not verified and not research_override:
            raise ValueError("Adaptive installation requires a VERIFIED_FIT policy artifact")
        self._verified_effort_policy = verified
        self._effort_policy_artifact_digest = getattr(artifact, "state_dict_digest", None)

    def remove_effort_policy(self) -> None:
        self._verified_effort_policy = False
        self._effort_policy_artifact_digest = None

    def has_verified_effort_policy(self) -> bool:
        return bool(self._verified_effort_policy and self.effort_controller is not None)

    def compute_effort_probe(
        self, input_ids: Tensor, attention_mask: Optional[Tensor] = None,
    ) -> EffortProbeResult:
        """Run the shared early Qwen prefix and return continuable internal state.

        The returned hidden state is exactly the configured early-prefix state reused by
        adaptive execution. It is an internal model representation, not a
        pooled token embedding or an extra unaccounted computation.
        """
        if not self.layers:
            raise RuntimeError("QwenExFusion requires at least one layer for its effort probe")
        hidden = self.embed(input_ids) if input_ids.dim() == 2 else input_ids
        if hidden.dim() != 3:
            raise ValueError("compute_effort_probe expects token ids (B,L) or hidden states (B,L,H)")
        probe_h = hidden
        for layer_index in range(self.effort_probe_layer_count):
            probe_h, _, _, _ = self.layers[layer_index](
                probe_h, attention_mask=attention_mask, use_recurrent=False,
                use_routed_moe=False, use_attn_res=False, latent_steps=0,
            )
        anchor = EffortController.pool_last_valid(probe_h, attention_mask)
        if self.effort_controller is None or not self.has_verified_effort_policy():
            probs = torch.zeros(anchor.size(0), 4, device=anchor.device, dtype=anchor.dtype)
            probs[:, 2] = 1.0
            policy_logits = None
        else:
            policy_output = self.effort_controller(anchor)
            probs = policy_output["effort_probs"]
            policy_logits = policy_output["effort_logits"]
        decision = decide_from_probs(
            probs, source_layer=self.effort_probe_layer_count - 1,
            source_position="post_qwen_probe", hidden_anchor=anchor.detach(),
        )
        receipt = self._compute_receipt(
            effort_mode="probe", layer_count=self.effort_probe_layer_count,
            batch_size=hidden.shape[0], sequence_length=hidden.shape[1],
            flags=self._effort_flags("fixed_2"),
        )
        return EffortProbeResult(
            probe_hidden=probe_h,
            probe_layer=self.effort_probe_layer_count - 1,
            executed_layers=self.effort_probe_layer_count,
            partial_hidden_state=probe_h,
            compute_receipt=receipt,
            decision=decision,
            policy_logits=policy_logits,
        )

    def _run_layers(
        self, hidden: Tensor, attention_mask: Optional[Tensor], effort: int,
        *, start_layer: int = 0, layer_count: Optional[int] = None,
        e3_refinement_steps_override: Optional[int] = None,
    ) -> Tensor:
        """Execute a fixed arm, optionally reusing a completed shared probe."""
        count = self._layer_count(effort) if layer_count is None else layer_count
        flags = self._effort_flags(f"fixed_{effort}")
        if e3_refinement_steps_override is not None:
            flags = {**flags, "latent_steps": int(e3_refinement_steps_override)}
        mode = self.e3_config.e3_refinement_mode
        insertion_layer = self.e3_region.insertion_layer
        for layer_index in range(start_layer, count):
            latent_steps = 0
            if effort == 3:
                if mode == "final_refine" and layer_index == count - 1:
                    latent_steps = int(flags["latent_steps"])
                elif mode in {"middle_recurrent", "profiled_middle_recurrent"} and layer_index == insertion_layer:
                    latent_steps = int(flags["latent_steps"])
            hidden, _, _, _ = self.layers[layer_index](
                hidden, attention_mask=attention_mask,
                use_recurrent=bool(flags["use_recurrent"]),
                use_routed_moe=bool(flags["use_routed_moe"]),
                use_attn_res=bool(flags["use_attn_res"]), latent_steps=latent_steps,
            )
            if effort == 3 and mode == "middle_repeat" and layer_index == self.e3_region.region_end:
                reuse_layers = self.e3_config.e3_reuse_layers or list(self.e3_region.selected_layers)
                repeated_hidden = hidden
                for _ in range(self.e3_config.e3_repeat_count):
                    for reuse_index in reuse_layers:
                        repeated_hidden, _, _, _ = self.layers[reuse_index](
                            repeated_hidden, attention_mask=attention_mask, use_recurrent=False,
                            use_routed_moe=False, use_attn_res=False, latent_steps=0,
                        )
                # The repeated modules share the exact pretrained parameters.
                # A bounded zero-initialized gate preserves E3==E2 at Gate 0B.
                gate_block = self.layers[self.e3_region.insertion_layer]
                scale = gate_block.latent_scale_limit * torch.tanh(
                    gate_block.latent_scale / gate_block.latent_scale_limit
                )
                hidden = hidden + scale * (repeated_hidden - hidden)
        if effort == 0 and self.e0_continuation is not None:
            hidden = self.e0_continuation(hidden)
        elif effort == 1 and self.e1_continuation is not None:
            hidden = self.e1_continuation(hidden)
        return hidden

    def _adaptive_receipt_stats(
        self, levels: Tensor, *, batch_size: int, sequence_length: int,
    ) -> Tuple[List[EffortComputeReceipt], Dict[str, Any]]:
        """Aggregate exact per-sample receipts for a partitioned adaptive batch."""
        per_sample = [
            self._compute_receipt(
                effort_mode=f"fixed_{int(level)}", layer_count=self._layer_count(int(level)),
                batch_size=1, sequence_length=sequence_length,
                flags=self._effort_flags(f"fixed_{int(level)}"),
            ) for level in levels.tolist()
        ]
        raw = [r.raw_compute_units for r in per_sample]
        normalized = [r.normalized_compute_cost for r in per_sample]
        stats: Dict[str, Any] = {
            "effort_mode": "adaptive", "effort_levels": levels.tolist(),
            "chosen_effort_levels": levels.tolist(),
            "chosen_effort_level": int(levels[0]) if bool(torch.all(levels == levels[0])) else levels.tolist(),
            "per_sample_compute": normalized, "per_sample_raw_compute_units": raw,
            "raw_compute_units": float(sum(raw)), "estimated_compute_units": float(sum(raw)),
            "normalized_compute_cost": float(sum(normalized) / max(batch_size, 1)),
            "executed_layer_count": int(sum(r.executed_layer_count for r in per_sample)),
            "skipped_layer_count": int(sum(r.skipped_layer_count for r in per_sample)),
            "attention_calls": int(sum(r.attention_calls for r in per_sample)),
            "ffn_calls": int(sum(r.ffn_calls for r in per_sample)),
            "recurrent_steps": int(sum(r.recurrent_steps for r in per_sample)),
            "latent_steps": int(sum(r.latent_steps for r in per_sample)),
            "routed_expert_calls": int(sum(r.routed_expert_calls for r in per_sample)),
            "token_count": batch_size * sequence_length,
            "probe_layer_count": self.effort_probe_layer_count,
            "probe_compute_included": True,
            "e3_variants": [r.e3_variant for r in per_sample],
            "middle_refinement_steps": int(sum(r.middle_refinement_steps for r in per_sample)),
            "repeated_pretrained_layer_calls": int(sum(r.repeated_pretrained_layer_calls for r in per_sample)),
        }
        return per_sample, stats

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Optional[Tensor] = None,
        effort_mode: str = "fixed_2",
        *,
        max_layers: Optional[int] = None,
        return_compute_receipt: bool = False,
        return_hidden_state: bool = False,
        effort_levels_override: Optional[Tensor] = None,
        e3_refinement_steps_override: Optional[int] = None,
        allow_unverified_policy: bool = False,
        precomputed_probe: Optional[EffortProbeResult] = None,
    ) -> Union[Tensor, Dict[str, Any]]:
        mode = str(effort_mode).lower()
        if e3_refinement_steps_override is not None:
            if mode not in {"fixed_3", "e3", "3"} or effort_levels_override is not None:
                raise ValueError("E3 step override is a fixed-E3 research interface")
            if input_ids.size(0) != 1:
                raise ValueError("Per-example E3 step override currently requires batch_size=1")
            if not 1 <= int(e3_refinement_steps_override) <= self.e3_config.e3_max_refine_steps:
                raise ValueError("E3 step override is outside the configured refinement range")
        if mode == "adaptive" or effort_levels_override is not None:
            if max_layers is not None:
                raise ValueError("max_layers is incompatible with adaptive effort dispatch")
            if effort_levels_override is None and not self.has_verified_effort_policy() and not allow_unverified_policy:
                raise RuntimeError(
                    "Adaptive execution requires an installed VERIFIED_FIT effort policy; "
                    "use an explicit effort override for architecture research"
                )
            if self.effort_controller is None and effort_levels_override is None:
                raise RuntimeError("This checkpoint has no effort controller")
            h0 = self.embed(input_ids)
            probe_result = self.compute_effort_probe(h0, attention_mask)
            probe_h, decision = probe_result.partial_hidden_state, probe_result.decision
            policy_logits = probe_result.policy_logits
            if allow_unverified_policy and not self.has_verified_effort_policy() and self.effort_controller is not None:
                policy_output = self.effort_controller(decision.hidden_anchor)
                probs = policy_output["effort_probs"]
                policy_logits = policy_output["effort_logits"]
                decision = decide_from_probs(
                    probs, source_layer=probe_result.probe_layer,
                    source_position="post_qwen_probe_research_override",
                    hidden_anchor=decision.hidden_anchor,
                )
            levels = decision.levels
            if effort_levels_override is not None:
                levels = torch.as_tensor(effort_levels_override, device=input_ids.device, dtype=torch.long)
                if levels.dim() != 1 or levels.numel() != input_ids.size(0):
                    raise ValueError("effort_levels_override must have shape (batch_size,)")
                if bool(torch.any((levels < 0) | (levels > 3))):
                    raise ValueError("effort_levels_override values must be in [0, 3]")
                probs = torch.zeros(input_ids.size(0), 4, device=input_ids.device, dtype=probe_h.dtype)
                probs.scatter_(1, levels.unsqueeze(1), 1.0)
                decision = decide_from_probs(
                    probs, source_layer=0, source_position="override_post_qwen_block_0",
                    hidden_anchor=decision.hidden_anchor,
                )
            # Partition active samples; the shared prefix is never re-run.
            h = torch.empty_like(probe_h)
            for level in levels.unique(sorted=True).tolist():
                indices = (levels == int(level)).nonzero(as_tuple=False).squeeze(1)
                sub_hidden = probe_h.index_select(0, indices)
                sub_mask = attention_mask.index_select(0, indices) if attention_mask is not None else None
                h.index_copy_(
                    0, indices,
                    self._run_layers(
                        sub_hidden, sub_mask, int(level),
                        start_layer=self.effort_probe_layer_count,
                    ),
                )
            h = self.norm(h)
            logits = self.lm_head(h)
            receipts, stats = self._adaptive_receipt_stats(
                levels, batch_size=input_ids.shape[0], sequence_length=input_ids.shape[1]
            )
            if return_compute_receipt or return_hidden_state:
                result: Dict[str, Any] = {
                    "logits": logits, "effort_decision": decision.to_dict(), "compute_stats": stats,
                    "chosen_effort": int(levels[0]) if len(levels) == 1 else levels.tolist(),
                    "policy_logits": policy_logits,
                    "probe_hidden": decision.hidden_anchor,
                }
                if return_compute_receipt:
                    result["compute_receipt"] = receipts[0] if len(receipts) == 1 else receipts
                if return_hidden_state:
                    result["hidden_state"] = h
                return result
            return logits

        effort = self._effort_index(effort_mode)
        flags = self._effort_flags(effort_mode)
        layer_count = self._layer_count(effort) if max_layers is None else max(1, min(len(self.layers), max_layers))
        if effort in (2, 3) and layer_count != len(self.layers):
            raise ValueError("E2/E3 cannot use a partial-depth override")
        h = self.embed(input_ids) if precomputed_probe is None else precomputed_probe.partial_hidden_state
        start_layer = 0
        if precomputed_probe is not None:
            if precomputed_probe.partial_hidden_state.shape[:2] != input_ids.shape:
                raise ValueError("precomputed_probe does not match input batch/sequence shape")
            if precomputed_probe.executed_layers != self.effort_probe_layer_count:
                raise ValueError("precomputed_probe depth does not match this model")
            if effort == 3:
                mode = self.e3_config.e3_refinement_mode
                earliest_extra = (
                    self.e3_region.region_end if mode == "middle_repeat"
                    else self.e3_region.insertion_layer
                )
                if mode not in {"none", "final_refine"} and earliest_extra < self.effort_probe_layer_count:
                    raise ValueError("E3 extra computation occurs inside the probe prefix and cannot be reused")
            start_layer = precomputed_probe.executed_layers
        h = self._run_layers(
            h, attention_mask, effort, start_layer=start_layer, layer_count=layer_count,
            e3_refinement_steps_override=e3_refinement_steps_override,
        )
        h = self.norm(h)
        logits = self.lm_head(h)
        receipt = self._compute_receipt(
            effort_mode=f"fixed_{effort}", layer_count=layer_count,
            batch_size=input_ids.shape[0], sequence_length=input_ids.shape[1],
            flags=({**flags, "latent_steps": int(e3_refinement_steps_override)} if e3_refinement_steps_override is not None else flags),
        )
        if return_compute_receipt or return_hidden_state:
            result: Dict[str, Any] = {"logits": logits}
            if return_compute_receipt:
                result.update(compute_receipt=receipt, compute_stats=receipt.to_dict())
                result["compute_stats"]["research_step_override"] = e3_refinement_steps_override is not None
            if return_hidden_state:
                result["hidden_state"] = h
            return result
        return logits

    def _compute_receipt(
        self, *, effort_mode: str, layer_count: int, batch_size: int,
        sequence_length: int, flags: Mapping[str, object],
    ) -> EffortComputeReceipt:
        is_e3 = effort_mode == "fixed_3"
        e3_mode = self.e3_config.e3_refinement_mode if is_e3 else None
        repeated_calls = 0
        if is_e3 and e3_mode == "middle_repeat":
            reuse_layers = self.e3_config.e3_reuse_layers or list(self.e3_region.selected_layers)
            repeated_calls = len(reuse_layers) * self.e3_config.e3_repeat_count
        middle_steps = int(flags["latent_steps"]) if is_e3 and e3_mode in {
            "middle_recurrent", "profiled_middle_recurrent"
        } else 0
        rec = EffortComputeReceipt(
            effort_mode=effort_mode,
            executed_layer_count=layer_count,
            skipped_layer_count=len(self.layers) - layer_count,
            attention_calls=layer_count + repeated_calls,
            ffn_calls=layer_count + repeated_calls,
            recurrent_steps=(layer_count if flags["use_recurrent"] else 0)
            + (1 if self.use_shallow_continuation and effort_mode in ("fixed_0", "fixed_1") else 0),
            latent_steps=int(flags["latent_steps"]),
            routed_expert_calls=layer_count * self.layers[0].routed_moe.top_k if flags["use_routed_moe"] else 0,
            token_count=batch_size * sequence_length,
            depth_fraction=layer_count / len(self.layers),
            refinement_insertion_layer=(self.e3_region.insertion_layer if is_e3 and e3_mode != "none" else None),
            refinement_region_start=(self.e3_region.region_start if is_e3 and e3_mode != "none" else None),
            refinement_region_end=(self.e3_region.region_end if is_e3 and e3_mode != "none" else None),
            middle_refinement_steps=middle_steps,
            repeated_pretrained_layer_calls=repeated_calls,
            middle_refiner_calls=(1 if middle_steps > 0 else 0),
            selected_profile_digest=(self.e3_region.source_profile_digest if is_e3 else None),
            e3_variant=e3_mode,
        )
        rec.raw_compute_units = estimate_compute(
            rec, hidden_size=self.hidden_size, intermediate_size=self.intermediate_size,
            num_heads=self.num_heads, num_key_value_heads=self.num_key_value_heads,
            sequence_length=sequence_length, batch_size=batch_size,
            num_routed_experts=self.num_routed_experts,
        )
        e2 = EffortComputeReceipt("fixed_2", len(self.layers), 0, len(self.layers), len(self.layers), 0, 0, 0, batch_size * sequence_length)
        e2_cost = estimate_compute(
            e2, hidden_size=self.hidden_size, intermediate_size=self.intermediate_size,
            num_heads=self.num_heads, num_key_value_heads=self.num_key_value_heads,
            sequence_length=sequence_length, batch_size=batch_size,
            num_routed_experts=self.num_routed_experts,
        )
        rec.normalized_compute_cost = rec.raw_compute_units / e2_cost
        return rec

    def compute_receipt(self, input_ids: Tensor, effort_mode: str) -> EffortComputeReceipt:
        effort = self._effort_index(effort_mode)
        return self._compute_receipt(
            effort_mode=f"fixed_{effort}", layer_count=self._layer_count(effort),
            batch_size=input_ids.shape[0], sequence_length=input_ids.shape[1],
            flags=self._effort_flags(effort_mode),
        )

    @torch.no_grad()
    def generate(
        self,
        input_ids: Tensor,
        attention_mask: Optional[Tensor] = None,
        *,
        max_new_tokens: int = 16,
        effort_mode: str = "fixed_2",
        eos_token_id: Optional[int] = None,
        tokenizer: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """Minimal greedy generation with receipt accumulation for counterfactuals."""
        ids = input_ids
        mask = attention_mask if attention_mask is not None else torch.ones_like(ids)
        total_raw = 0.0
        total_normalized = 0.0
        last_logits: Optional[Tensor] = None
        last_receipt: Optional[EffortComputeReceipt] = None
        last_stats: Optional[Dict[str, Any]] = None
        generated = 0
        for _ in range(max_new_tokens):
            out = self(ids, attention_mask=mask, effort_mode=effort_mode, return_compute_receipt=True)
            last_logits = out["logits"]
            raw_receipt = out["compute_receipt"]
            last_receipt = raw_receipt if isinstance(raw_receipt, EffortComputeReceipt) else None
            last_stats = dict(out["compute_stats"])
            total_raw += float(last_stats["raw_compute_units"])
            total_normalized += float(last_stats["normalized_compute_cost"])
            token = last_logits[:, -1].argmax(dim=-1, keepdim=True)
            ids = torch.cat((ids, token), dim=1)
            mask = torch.cat((mask, torch.ones_like(token)), dim=1)
            generated += 1
            if eos_token_id is not None and bool(torch.all(token == eos_token_id)):
                break
        if last_receipt is None:
            if last_stats is None:
                last_receipt = self.compute_receipt(ids, effort_mode)
                last_stats = last_receipt.to_dict()
                last_logits = self(ids, attention_mask=mask, effort_mode=effort_mode)
        stats = dict(last_stats or last_receipt.to_dict())
        stats["raw_compute_units"] = total_raw
        stats["estimated_compute_units"] = total_raw
        stats["normalized_compute_cost"] = total_normalized
        stats["compute_normalization"] = "sum_of_per_decode_step_e2_equivalents"
        stats["generated_tokens"] = generated
        result: Dict[str, Any] = {"sequences": ids, "logits": last_logits, "compute_stats": stats}
        if tokenizer is not None:
            result["generated_text"] = tokenizer.batch_decode(ids, skip_special_tokens=True)
        return result

    def zero_augmentation_scales(self) -> None:
        with torch.no_grad():
            for layer in self.layers:
                layer.rec_scale.zero_()
                layer.moe_scale.zero_()
                layer.attn_res_scale.zero_()
                layer.latent_scale.zero_()


def augment_qwen_compat_model(
    compat: QwenCompatModel,
    *,
    recurrent_type: str = "ssm",
    state_size: int = 16,
    latent_size: Optional[int] = None,
    num_routed_experts: int = 4,
    top_k: int = 2,
    use_attn_res: bool = False,
    dropout: float = 0.0,
    default_e3_steps: int = 2,
    e0_depth_fraction: float = 0.50,
    e1_depth_fraction: float = 0.75,
    e0_layer_count: Optional[int] = None,
    e1_layer_count: Optional[int] = None,
    use_shallow_continuation: bool = False,
    continuation_bottleneck_size: Optional[int] = None,
    latent_scale_limit: float = 0.01,
    effort_controller_hidden_size: int = 128,
    effort_probe_layer: Optional[int] = None,
    effort_probe_fraction: float = 0.125,
    e3_config: Optional[E3RefinementConfig] = None,
) -> QwenExFusionModel:
    """
    Convert a loaded QwenCompatModel into QwenExFusionModel.

    Copies backbone weights exactly; initializes augmentation scales to 0.
    """
    # Infer dims from compat
    H = compat.embed.embedding_dim
    V = compat.embed.num_embeddings
    L = len(compat.layers)
    # head info from first layer
    blk0 = compat.layers[0]
    n_heads = blk0.self_attn.num_heads
    n_kv = blk0.self_attn.num_key_value_heads
    inter = blk0.mlp.gate_proj.out_features
    rope_theta = getattr(blk0.self_attn.rotary, "base", 10000.0) if blk0.self_attn.rotary else 10000.0
    max_pos = getattr(blk0.self_attn.rotary, "max_position", 8192) if blk0.self_attn.rotary else 8192
    rms_eps = blk0.input_layernorm.eps
    attn_bias = blk0.self_attn.q_proj.bias is not None
    attn_out_bias = blk0.self_attn.out_proj.bias is not None

    model = QwenExFusionModel(
        vocab_size=V, hidden_size=H, num_layers=L, num_heads=n_heads,
        num_key_value_heads=n_kv, intermediate_size=inter,
        rope_theta=rope_theta, max_position=max_pos, rms_eps=rms_eps,
        tie_word_embeddings=compat.lm_head.weight.data_ptr() == compat.embed.weight.data_ptr(),
        attention_bias=attn_bias, recurrent_type=recurrent_type,
        attention_output_bias=attn_out_bias,
        state_size=state_size, latent_size=latent_size,
        num_routed_experts=num_routed_experts, top_k=top_k,
        use_attn_res=use_attn_res, dropout=dropout,
        default_e3_steps=default_e3_steps,
        e0_depth_fraction=e0_depth_fraction, e1_depth_fraction=e1_depth_fraction,
        e0_layer_count=e0_layer_count, e1_layer_count=e1_layer_count,
        use_shallow_continuation=use_shallow_continuation,
        continuation_bottleneck_size=continuation_bottleneck_size,
        latent_scale_limit=latent_scale_limit,
        effort_controller_hidden_size=effort_controller_hidden_size,
        effort_probe_layer=effort_probe_layer,
        effort_probe_fraction=effort_probe_fraction,
        e3_config=e3_config,
    )

    # Copy embed / norm / lm_head
    model.embed.weight.data.copy_(compat.embed.weight.data)
    model.norm.weight.data.copy_(compat.norm.weight.data)
    if model.lm_head.weight.data_ptr() != model.embed.weight.data_ptr():
        model.lm_head.weight.data.copy_(compat.lm_head.weight.data)

    # Copy each base block
    for i in range(L):
        model.layers[i].base.load_state_dict(compat.layers[i].state_dict())

    model.zero_augmentation_scales()
    all_names = {n for n, _ in model.named_parameters()}
    imported = {"embed.weight", "norm.weight", "lm_head.weight"}
    imported.update(
        n for n in all_names if any(n.startswith(f"layers.{i}.base.") for i in range(L))
    )
    imported &= all_names
    scales = {n for n in all_names if n.endswith("_scale")}
    augmentation = all_names - imported
    continuation = {n for n in all_names if "_continuation." in n}
    final_layer = len(model.layers) - 1
    refinement_layer = (
        final_layer
        if model.e3_config.e3_refinement_mode == "final_refine"
        else model.e3_region.insertion_layer
    )
    # Keep the refinement LayerNorm at identity for the first E3 study. It is
    # not part of the learned transformation, and this avoids an observed MPS
    # LayerNorm-scale backward instability while fc1/fc2 and residual scale
    # remain fully trainable.
    e3_refinement = {
        n for n in all_names
        if n.startswith(f"layers.{refinement_layer}.latent_refine.")
        and ".latent_refine.norm." not in n
    }
    e3_scales = {f"layers.{refinement_layer}.latent_scale"} & all_names
    e3_middle = {
        n for n in imported
        if any(n.startswith(f"layers.{layer}.base.") for layer in model.e3_region.selected_layers)
    }
    model.parameter_provenance = ExFusionParameterProvenance(
        imported_parameter_names=tuple(sorted(imported)),
        new_parameter_names=tuple(sorted(all_names - imported)),
        augmentation_parameter_names=tuple(sorted(augmentation)),
        scale_parameter_names=tuple(sorted(scales)),
        continuation_parameter_names=tuple(sorted(continuation)),
        e3_refinement_parameter_names=tuple(sorted(e3_refinement)),
        e3_scale_parameter_names=tuple(sorted(e3_scales)),
        e3_middle_layer_parameter_names=tuple(sorted(e3_middle)),
    )
    return model


def prepare_exfusion_for_training(
    model: QwenExFusionModel,
    *,
    gate0b_passed: bool,
    epsilon: float = 1e-4,
    enabled_scales: Optional[Tuple[str, ...]] = None,
) -> TrainingInitReceipt:
    """Explicitly transition exact-zero Gate-0B scales to trainable epsilon."""
    if not gate0b_passed:
        raise RuntimeError("Gate 0B must pass before preparing ExFusion for training")
    if epsilon < 0:
        raise ValueError("epsilon must be non-negative")
    before = {n: p.detach().clone() for n, p in model.named_parameters() if ".base." in n or n in {"embed.weight", "norm.weight", "lm_head.weight"}}
    changed: List[str] = []
    active_names = set(model.parameter_provenance.e3_scale_parameter_names) if model.parameter_provenance else {
        f"layers.{model.e3_region.insertion_layer}.latent_scale"
    }
    with torch.no_grad():
        for name, p in model.named_parameters():
            enabled = (
                name in active_names
                if enabled_scales is None
                else any(name.endswith(s) for s in enabled_scales)
            )
            if enabled and float(p.detach()) == 0.0:
                p.fill_(epsilon)
                changed.append(name)
    unchanged = all(torch.equal(before[n], dict(model.named_parameters())[n].detach()) for n in before)
    return TrainingInitReceipt(gate0b_passed, epsilon, tuple(sorted(changed)), unchanged)


def load_qwen_exfusion_checkpoint(path: str, *, map_location: str = "cpu") -> QwenExFusionModel:
    """Load a canonical checkpoint written by ``save_adapted_checkpoint``."""
    payload = torch.load(path, map_location=map_location, weights_only=False)
    cfg = payload["model_config"]
    if cfg.get("architecture") != "QwenExFusionModel":
        raise ValueError(f"Not a canonical QwenExFusion checkpoint: {cfg.get('architecture')}")
    fractions = cfg.get("depth_fractions") or (0.5, 0.75, 1.0, 1.0)
    overrides = cfg.get("layer_count_overrides") or (None, None, None, None)
    state_dict = payload.get("state_dict", payload.get("model"))
    # Artifacts written before adaptive runtime lack controller tensors and
    # remain loadable for fixed E0–E3 use.
    has_controller = any(key.startswith("effort_controller.") for key in state_dict)
    e3_payload = cfg.get("e3_config")
    e3_config = E3RefinementConfig(**e3_payload) if e3_payload else E3RefinementConfig(
        e3_refinement_mode="final_refine",
        e3_refine_steps=int(cfg.get("default_e3_steps") or 2),
    )
    model = QwenExFusionModel(
        vocab_size=int(cfg["vocab_size"]), hidden_size=int(cfg["hidden_size"]),
        num_layers=int(cfg["num_layers"]), num_heads=int(cfg["num_heads"]),
        num_key_value_heads=int(cfg["num_key_value_heads"]),
        intermediate_size=int(cfg["intermediate_size"]),
        num_routed_experts=int(cfg.get("num_routed_experts") or 4),
        top_k=int(cfg.get("top_k") or 2), recurrent_type=str(cfg.get("recurrent_type") or "ssm"),
        state_size=int(cfg.get("state_size") or 16), latent_size=cfg.get("latent_size"),
        rope_theta=float(cfg.get("rope_theta") or 10000.0),
        max_position=int(cfg.get("max_position") or 8192),
        rms_eps=float(cfg.get("rms_eps") or 1e-6),
        attention_bias=bool(cfg.get("attention_bias", False)),
        attention_output_bias=bool(cfg.get("attention_output_bias", False)),
        tie_word_embeddings=bool(cfg.get("tie_word_embeddings", True)),
        e0_depth_fraction=float(fractions[0]), e1_depth_fraction=float(fractions[1]),
        e0_layer_count=overrides[0], e1_layer_count=overrides[1],
        default_e3_steps=int(cfg.get("default_e3_steps") or 2),
        use_shallow_continuation=bool(cfg.get("use_shallow_continuation", False)),
        continuation_bottleneck_size=cfg.get("continuation_bottleneck_size"),
        latent_scale_limit=float(cfg.get("latent_scale_limit") or 0.01),
        effort_controller_hidden_size=int(cfg.get("effort_controller_hidden_size") or 128),
        enable_effort_controller=bool(cfg.get("enable_effort_controller", has_controller)) and has_controller,
        effort_probe_layer=(int(cfg["effort_probe_layer_count"]) - 1 if cfg.get("effort_probe_layer_count") else None),
        effort_probe_fraction=float(cfg.get("effort_probe_fraction") or 0.125),
        e3_config=e3_config,
    )
    model.load_state_dict(state_dict)
    policy_metadata = cfg.get("effort_policy")
    if policy_metadata is not None:
        if policy_metadata.get("status") != "VERIFIED_FIT":
            raise ValueError(f"Unsupported effort-policy checkpoint status: {policy_metadata.get('status')!r}")
        if model.effort_controller is None:
            raise ValueError("Verified effort-policy metadata requires controller weights")
        from .policy_trainer import _state_dict_digest
        expected_digest = policy_metadata.get("state_dict_digest")
        actual_digest = _state_dict_digest(model.effort_controller.state_dict())
        if not expected_digest or actual_digest != expected_digest:
            raise ValueError(
                "Verified effort-policy controller digest mismatch: "
                f"expected={expected_digest!r} actual={actual_digest!r}"
            )
        model._verified_effort_policy = True
        model._effort_policy_artifact_digest = actual_digest
    p = payload.get("parameter_provenance")
    if p:
        model.parameter_provenance = ExFusionParameterProvenance(
            imported_parameter_names=tuple(p["imported_parameter_names"]),
            new_parameter_names=tuple(p["new_parameter_names"]),
            augmentation_parameter_names=tuple(p["augmentation_parameter_names"]),
            scale_parameter_names=tuple(p["scale_parameter_names"]),
            continuation_parameter_names=tuple(p.get("continuation_parameter_names", ())),
            e3_refinement_parameter_names=tuple(p.get("e3_refinement_parameter_names", ())),
            e3_scale_parameter_names=tuple(p.get("e3_scale_parameter_names", ())),
            e3_middle_layer_parameter_names=tuple(p.get("e3_middle_layer_parameter_names", ())),
        )
    return model


@torch.no_grad()
def gate0b_exact_parity(
    compat: QwenCompatModel,
    exfusion: QwenExFusionModel,
    input_ids: Tensor,
    attention_mask: Optional[Tensor] = None,
    *,
    rtol: float = 1e-5,
    atol: float = 1e-6,
) -> Dict[str, float]:
    """
    Gate 0B: with zero scales and fixed_E2, ExFusion logits must match Compat.
    """
    exfusion.zero_augmentation_scales()
    exfusion.eval()
    compat.eval()
    logits_c = compat(input_ids, attention_mask=attention_mask)
    logits_e = exfusion(input_ids, attention_mask=attention_mask, effort_mode="fixed_2")
    diff = (logits_c.float() - logits_e.float()).abs()
    flat_c = logits_c.float().reshape(-1, logits_c.size(-1))
    flat_e = logits_e.float().reshape(-1, logits_e.size(-1))
    log_e = torch.nn.functional.log_softmax(flat_e, dim=-1)
    prob_c = torch.nn.functional.softmax(flat_c, dim=-1)
    ce_c = torch.nn.functional.cross_entropy(logits_c[:, :-1].float().reshape(-1, logits_c.size(-1)), input_ids[:, 1:].reshape(-1))
    ce_e = torch.nn.functional.cross_entropy(logits_e[:, :-1].float().reshape(-1, logits_e.size(-1)), input_ids[:, 1:].reshape(-1))
    backbone_identical = True
    for name, p in compat.named_parameters():
        target_name = name if not name.startswith("layers.") else name.replace(".input_layernorm", ".base.input_layernorm").replace(".self_attn", ".base.self_attn").replace(".post_attention_layernorm", ".base.post_attention_layernorm").replace(".mlp", ".base.mlp")
        q = dict(exfusion.named_parameters()).get(target_name)
        if q is None or not torch.equal(p.detach().cpu(), q.detach().cpu()):
            backbone_identical = False
            break
    metrics = {
        "ce_compat": float(ce_c.item()),
        "ce_exfusion": float(ce_e.item()),
        "ce_difference": float((ce_e - ce_c).item()),
        "logit_mae": float(diff.mean().item()),
        "logit_max_abs": float(diff.max().item()),
        "cosine_similarity": float(torch.nn.functional.cosine_similarity(flat_c, flat_e, dim=-1).mean().item()),
        "top1_agreement": float((flat_c.argmax(-1) == flat_e.argmax(-1)).float().mean().item()),
        "kl_divergence": float(torch.nn.functional.kl_div(log_e, prob_c, reduction="batchmean").item()),
        "source_backbone_parameter_identity": backbone_identical,
        "passed": bool(torch.allclose(logits_c.float(), logits_e.float(), rtol=rtol, atol=atol)),
    }
    metrics["decision"] = "PASS_EXACT" if metrics["passed"] and backbone_identical else "FAIL"
    return metrics

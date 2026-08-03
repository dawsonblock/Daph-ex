"""
HybridBlock — structured complementary operators (v3.1).

Default: recurrent mixing (SelectiveSSM or KDA)
Periodic: global attention (optional, cacheable)
LatentMoE specialization (sparse dispatch)
E3: shared-weight latent refinement
Effort-controlled path selection
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor

from .attention import CausalSelfAttention
from .norms import make_norm
from .config import DAPHConfigV3
from .effort import EffortController, early_exit_mask_from_effort
from .gates import ChannelGate
from .kda import KimiDeltaAttention
from .latent_moe import LatentMoE
from .latent_refine import LatentRefineBlock
from .ssm import SelectiveSSM


class HybridBlock(nn.Module):
    def __init__(self, config: DAPHConfigV3) -> None:
        super().__init__()
        self.config = config
        H = config.hidden_size

        self.recurrent_type = getattr(config, "recurrent_type", "ssm")
        if self.recurrent_type == "kda":
            n_heads = max(1, getattr(config, "kda_num_heads", min(4, config.num_attention_heads)))
            self.recurrent_layers = nn.ModuleList(
                [
                    KimiDeltaAttention(
                        H,
                        num_heads=n_heads,
                        g_min=getattr(config, "kda_g_min", -5.0),
                        dropout=config.dropout,
                    )
                    for _ in range(config.num_recurrent_per_block)
                ]
            )
        else:
            self.recurrent_layers = nn.ModuleList(
                [SelectiveSSM(H, config.state_size) for _ in range(config.num_recurrent_per_block)]
            )
            for layer in self.recurrent_layers:
                layer.bypass_decay = getattr(config, "ssm_bypass_decay", 0.0)

        self.use_global_attention = bool(getattr(config, "use_global_attention", True))
        if self.use_global_attention:
            n_kv = getattr(config, "num_key_value_heads", None) or config.num_attention_heads
            self.attn = CausalSelfAttention(
                H,
                config.num_attention_heads,
                num_key_value_heads=n_kv,
                dropout=config.dropout,
                bias=getattr(config, "attention_bias", False),
                use_rope=getattr(config, "use_rope", False),
                rope_theta=getattr(config, "rope_theta", 10000.0),
                max_position=getattr(config, "max_position_embeddings", 8192),
            )
            self.attn_norm = make_norm(
                H,
                getattr(config, "norm_type", "layer"),
                eps=getattr(config, "rms_norm_eps", 1e-6),
            )
        else:
            self.attn = None
            self.attn_norm = None

        self.moe = LatentMoE(
            hidden_size=H,
            latent_size=config.latent_size,
            num_routed_experts=config.num_routed_experts,
            num_shared_experts=config.num_shared_experts,
            top_k=config.top_k_experts,
            dropout=config.moe_dropout,
            activation=getattr(config, "moe_activation", "swiglu"),
            beta_gate=getattr(config, "moe_beta_gate", 1.5),
            beta_up=getattr(config, "moe_beta_up", 1.5),
            use_quantile_balancing=getattr(
                config, "use_load_balancing", getattr(config, "use_quantile_balancing", True)
            ),
            qb_quantile=getattr(config, "qb_quantile", 0.75),
            qb_bias_lr=getattr(config, "qb_bias_lr", 0.01),
            intermediate_size=getattr(config, "intermediate_size", None),
            shared_ffn=getattr(config, "shared_ffn", "swiglu"),
        )

        self.use_gates = config.enable_channel_gates
        if self.use_gates:
            self.gate_rec = ChannelGate(H)
            self.gate_attn = ChannelGate(H) if self.use_global_attention else None
            self.gate_moe = ChannelGate(H)

        self.effort = EffortController(H, num_levels=config.effort_levels)
        self.latent_refine = LatentRefineBlock(
            H,
            expansion=2.0,
            dropout=config.dropout,
            workspace_slots=getattr(config, "workspace_slots", 0),
        )
        self.default_e3_steps = getattr(config, "default_e3_steps", 2)
        self.max_latent_steps = getattr(config, "max_latent_steps", 4)
        self.final_norm = make_norm(H, getattr(config, "norm_type", "layer"), eps=getattr(config, "rms_norm_eps", 1e-6))
        self.early_exit_threshold = getattr(config, "early_exit_threshold", 0.55)


    def run_recurrent_only(
        self,
        hidden_states: Tensor,
        attention_mask: Optional[Tensor] = None,
        recurrent_states: Optional[List[Optional[Tensor]]] = None,
        use_cache: bool = False,
    ) -> Tuple[Tensor, List[Tensor], Dict[str, Any]]:
        """
        Stage 1 of a HybridBlock: recurrent stack only.
        Returns (probe_hidden, new_rec_states, meta).
        """
        B, L, H = hidden_states.shape
        residual = hidden_states
        if attention_mask is None:
            ssm_mask = torch.ones(B, L, device=hidden_states.device, dtype=hidden_states.dtype)
            attn_mask = None
        else:
            # If full-history mask is longer than current tokens (decode),
            # SSM sees only the new-token slice; attention gets full mask.
            m = attention_mask.to(hidden_states.dtype)
            if m.shape[-1] > L:
                ssm_mask = m[:, -L:]
                attn_mask = m  # full key validity for cached attention
            else:
                ssm_mask = m
                attn_mask = m
            if self.config.mask_convention == "pytorch":
                ssm_mask = 1.0 - ssm_mask

        rec_out = hidden_states
        new_rec_states: List[Tensor] = []
        for i, layer in enumerate(self.recurrent_layers):
            state_in = None
            if recurrent_states is not None and i < len(recurrent_states):
                state_in = recurrent_states[i]
            rec_out, state_out = layer(
                rec_out,
                state=state_in,
                mask=ssm_mask,
                bypass_decay=getattr(layer, "bypass_decay", 0.0),
            )
            new_rec_states.append(state_out)

        if self.use_gates:
            rec_contrib = self.gate_rec(residual, rec_out)
        else:
            rec_contrib = rec_out
        probe_hidden = residual + rec_contrib
        meta: Dict[str, Any] = {"recurrent_only": True}
        if use_cache:
            meta["recurrent_states"] = new_rec_states
        return probe_hidden, new_rec_states, meta

    def continue_after_recurrent(
        self,
        probe_hidden: Tensor,
        attention_mask: Optional[Tensor] = None,
        past_attention_k: Optional[Tensor] = None,
        past_attention_v: Optional[Tensor] = None,
        recurrent_states_out: Optional[List[Optional[Tensor]]] = None,
        use_cache: bool = False,
        force_skip_attention: bool = False,
        force_skip_moe: bool = False,
        moe_top_k_override: Optional[int] = None,
        force_latent_steps: int = 0,
        emitter: Optional[Any] = None,
        layer_index: Optional[int] = None,
    ) -> Tuple[Tensor, Dict[str, Any]]:
        """
        Stage 2: attention / MoE / latent refine from already-computed probe_hidden.
        Does NOT re-run the recurrent stack.
        """
        B, L, H = probe_hidden.shape
        hidden_states = probe_hidden
        residual = probe_hidden  # residual for partial exit is probe itself
        meta: Dict[str, Any] = {}
        rec_contrib = torch.zeros_like(probe_hidden)  # already applied

        if attention_mask is not None and attention_mask.shape[-1] > L:
            attn_mask = attention_mask
            ssm_slice = attention_mask[:, -L:]
        else:
            attn_mask = attention_mask
            ssm_slice = attention_mask

        # Effort telemetry (using consistent anchor feature)
        from .effort import EffortController
        anchor = EffortController.pool_last_valid(probe_hidden, ssm_slice)
        effort_info = self.effort(anchor)  # (B, H) path
        meta.update(effort_info)

        # Attention
        attention_executed = False
        present_k = present_v = None
        if (
            self.use_global_attention
            and self.attn is not None
            and not force_skip_attention
        ):
            _am = locals().get("attn_mask", attention_mask)
            attn_out, present_k, present_v = self.attn(
                hidden_states,
                attention_mask=_am,
                past_k=past_attention_k,
                past_v=past_attention_v,
                use_cache=use_cache,
            )
            attn_out = self.attn_norm(attn_out + hidden_states)
            if self.use_gates and self.gate_attn is not None:
                attn_contrib = self.gate_attn(hidden_states, attn_out)
            else:
                attn_contrib = attn_out - hidden_states
            hidden_states = hidden_states + attn_contrib
            attention_executed = True
        meta["attention_executed"] = attention_executed
        if use_cache:
            meta["attention_k"] = present_k
            meta["attention_v"] = present_v

        # MoE
        if not force_skip_moe:
            orig_top_k = self.moe.top_k
            if moe_top_k_override is not None:
                self.moe.top_k = int(moe_top_k_override)
            moe_out, router_logits = self.moe(hidden_states, return_router_logits=True)
            if moe_top_k_override is not None:
                self.moe.top_k = orig_top_k
            if self.use_gates:
                moe_contrib = self.gate_moe(hidden_states, moe_out)
            else:
                moe_contrib = moe_out
            hidden_states = hidden_states + moe_contrib
            if router_logits is not None:
                meta["moe_router_logits"] = router_logits
            tel = getattr(self.moe, "_last_telemetry", None)
            if tel is not None:
                meta["moe_telemetry"] = tel
            meta["moe_executed"] = True
        else:
            meta["moe_executed"] = False

        # Latent refine
        latent_steps = int(force_latent_steps)
        if latent_steps > 0:
            latent_steps = min(latent_steps, self.max_latent_steps)
            hidden_states, _ = self.latent_refine(hidden_states, num_steps=latent_steps)
        meta["latent_steps"] = latent_steps

        output = self.final_norm(hidden_states)
        if use_cache and recurrent_states_out is not None:
            meta["recurrent_states"] = recurrent_states_out
        meta["early_exited"] = False
        return output, meta

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Optional[Tensor] = None,
        recurrent_states: Optional[List[Optional[Tensor]]] = None,
        past_attention_k: Optional[Tensor] = None,
        past_attention_v: Optional[Tensor] = None,
        use_cache: bool = False,
        enable_early_exit: bool = False,
        early_exit_threshold: Optional[float] = None,
        emitter: Optional[Any] = None,
        layer_index: Optional[int] = None,
        force_skip_attention: bool = False,
        force_skip_moe: bool = False,
        moe_top_k_override: Optional[int] = None,
        force_latent_steps: int = 0,
    ) -> Tuple[Tensor, Dict[str, Any]]:
        if hidden_states.dim() != 3:
            raise ValueError(f"hidden_states must be (B, L, H); got {tuple(hidden_states.shape)}")
        B, L, H = hidden_states.shape
        residual = hidden_states
        meta: Dict[str, Any] = {}

        if attention_mask is None:
            ssm_mask = torch.ones(B, L, device=hidden_states.device, dtype=hidden_states.dtype)
            attn_mask = None
        else:
            # If full-history mask is longer than current tokens (decode),
            # SSM sees only the new-token slice; attention gets full mask.
            m = attention_mask.to(hidden_states.dtype)
            if m.shape[-1] > L:
                ssm_mask = m[:, -L:]
                attn_mask = m  # full key validity for cached attention
            else:
                ssm_mask = m
                attn_mask = m
            if self.config.mask_convention == "pytorch":
                ssm_mask = 1.0 - ssm_mask

        # 1. Recurrent path
        rec_out = hidden_states
        new_rec_states: List[Tensor] = []
        for i, layer in enumerate(self.recurrent_layers):
            state_in = None
            if recurrent_states is not None and i < len(recurrent_states):
                state_in = recurrent_states[i]
            rec_out, state_out = layer(
                rec_out,
                state=state_in,
                mask=ssm_mask,
                bypass_decay=getattr(layer, "bypass_decay", 0.0),
            )
            new_rec_states.append(state_out)

        if self.use_gates:
            rec_contrib = self.gate_rec(residual, rec_out)
        else:
            rec_contrib = rec_out
        hidden_states = residual + rec_contrib

        # 2. Effort
        effort_info = self.effort(hidden_states)
        meta.update(effort_info)
        if emitter is not None:
            emitter.emit_effort(effort_info, layer_index=layer_index)

        thresh = (
            early_exit_threshold
            if early_exit_threshold is not None
            else self.early_exit_threshold
        )
        exit_mask = None
        if enable_early_exit:
            exit_mask = early_exit_mask_from_effort(
                effort_info, threshold=thresh, max_exit_level=0
            )
            meta["early_exit_mask"] = exit_mask
            if exit_mask.all():
                meta["latent_steps"] = 0
                meta["attention_executed"] = False
                meta["moe_executed"] = False
                output = self.final_norm(hidden_states)
                if use_cache:
                    meta["recurrent_states"] = new_rec_states
                    meta["attention_k"] = None
                    meta["attention_v"] = None
                meta["early_exited"] = True
                if emitter is not None:
                    emitter.emit_early_exit(meta, layer_index=layer_index)
                return output, meta

        # 3. Global attention
        attention_executed = False
        present_k = present_v = None
        if (
            self.use_global_attention
            and self.attn is not None
            and not force_skip_attention
        ):
            _am = locals().get("attn_mask", attention_mask)
            attn_out, present_k, present_v = self.attn(
                hidden_states,
                attention_mask=_am,
                past_k=past_attention_k,
                past_v=past_attention_v,
                use_cache=use_cache,
            )
            attn_out = self.attn_norm(attn_out + hidden_states)
            if self.use_gates and self.gate_attn is not None:
                attn_contrib = self.gate_attn(hidden_states, attn_out)
            else:
                attn_contrib = attn_out - hidden_states
            hidden_states = hidden_states + attn_contrib
            attention_executed = True
        meta["attention_executed"] = attention_executed
        if use_cache:
            meta["attention_k"] = present_k
            meta["attention_v"] = present_v

        # 4. LatentMoE
        if not force_skip_moe:
            orig_top_k = self.moe.top_k
            if moe_top_k_override is not None:
                self.moe.top_k = int(moe_top_k_override)
            moe_out, router_logits = self.moe(hidden_states, return_router_logits=True)
            if moe_top_k_override is not None:
                self.moe.top_k = orig_top_k
            if self.use_gates:
                moe_contrib = self.gate_moe(hidden_states, moe_out)
            else:
                moe_contrib = moe_out
            hidden_states = hidden_states + moe_contrib
            if router_logits is not None:
                meta["moe_router_logits"] = router_logits
                if emitter is not None:
                    emitter.emit_routing(router_logits, layer_index=layer_index)
            tel = getattr(self.moe, "_last_telemetry", None)
            if tel is not None:
                meta["moe_telemetry"] = tel
            meta["moe_executed"] = True
        else:
            meta["moe_executed"] = False

        # Partial early-exit blend (full-batch expensive path already run)
        if exit_mask is not None and not exit_mask.all():
            keep_full = (~exit_mask).float().view(B, 1, 1)
            cheap = residual + rec_contrib
            hidden_states = keep_full * hidden_states + (1.0 - keep_full) * cheap
            meta["early_exited_partial"] = True

        # 5. E3 latent refinement
        latent_steps = int(force_latent_steps)
        if latent_steps > 0:
            latent_steps = min(latent_steps, self.max_latent_steps)
            hidden_states, _ = self.latent_refine(hidden_states, num_steps=latent_steps)
        meta["latent_steps"] = latent_steps

        output = self.final_norm(hidden_states)
        if use_cache:
            meta["recurrent_states"] = new_rec_states
        meta["early_exited"] = False
        return output, meta

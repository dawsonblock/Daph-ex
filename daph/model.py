"""
DAPHHybridModelV3 — multi-layer structured hybrid model.

v3.1.2: per-sample adaptive effort + active batch partitioning.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch import Tensor

from .attn_res import AttnResBank
from .config import DAPHConfigV3
from .effort_decision import (
    ComputeStats,
    EffortDecision,
    decide_from_probs,
    nominal_compute_for_levels,
    estimate_compute,
)
from .norms import make_norm
from .hybrid_block import HybridBlock


@dataclass
class LayerCache:
    recurrent_states: Optional[List[Optional[Tensor]]] = None
    attention_k: Optional[Tensor] = None
    attention_v: Optional[Tensor] = None


@dataclass
class ModelCache:
    layers: List[LayerCache] = field(default_factory=list)
    sequence_length: int = 0
    effort_levels: Optional[Tensor] = None  # (B,) persisted for decode
    attention_mask: Optional[Tensor] = None  # (B, L_total) full key validity

    @classmethod
    def empty(cls, num_layers: int) -> "ModelCache":
        return cls(layers=[LayerCache() for _ in range(num_layers)], sequence_length=0)

    def index_select(self, indices: Tensor) -> "ModelCache":
        """Gather cache rows for active subset."""
        new = ModelCache.empty(len(self.layers))
        new.sequence_length = self.sequence_length
        if self.effort_levels is not None:
            new.effort_levels = self.effort_levels.index_select(0, indices)
        if self.attention_mask is not None:
            new.attention_mask = self.attention_mask.index_select(0, indices)
        for i, lc in enumerate(self.layers):
            rec = None
            if lc.recurrent_states is not None:
                rec = []
                for st in lc.recurrent_states:
                    if st is None:
                        rec.append(None)
                    elif isinstance(st, tuple):
                        # KDA state (S, hist_k, hist_v)
                        rec.append(
                            tuple(
                                t.index_select(0, indices) if t is not None else None
                                for t in st
                            )
                        )
                    else:
                        rec.append(st.index_select(0, indices))
            ak = lc.attention_k.index_select(0, indices) if lc.attention_k is not None else None
            av = lc.attention_v.index_select(0, indices) if lc.attention_v is not None else None
            new.layers[i] = LayerCache(recurrent_states=rec, attention_k=ak, attention_v=av)
        return new


class DAPHHybridModelV3(nn.Module):
    def __init__(self, config: DAPHConfigV3) -> None:
        super().__init__()
        self.config = config
        self.embed = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            [HybridBlock(config) for _ in range(config.num_layers)]
        )
        self.use_attn_res = config.use_attn_res
        if self.use_attn_res:
            n_heads = getattr(config, "attn_res_num_heads", None)
            if n_heads is None:
                n_heads = max(1, config.num_attention_heads // 4)
            if config.hidden_size % n_heads != 0:
                raise ValueError(
                    f"hidden_size ({config.hidden_size}) must be divisible by "
                    f"attn_res_num_heads ({n_heads})"
                )
            self.attn_res_bank = AttnResBank(
                hidden_size=config.hidden_size,
                max_blocks=config.num_attn_res_blocks,
                num_heads=n_heads,
                dropout=config.dropout,
                gate=True,
                detach_history=getattr(config, "attn_res_detach_history", True),
            )
        else:
            self.attn_res_bank = None
        self.final_norm = make_norm(config.hidden_size, getattr(config, "norm_type", "layer"), eps=getattr(config, "rms_norm_eps", 1e-6))
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        if config.tie_word_embeddings:
            self.lm_head.weight = self.embed.weight
        self._init_weights()

    def _init_weights(self) -> None:
        from .kda import KimiDeltaAttention

        for module in self.modules():
            if isinstance(module, KimiDeltaAttention):
                continue
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
        for module in self.modules():
            if isinstance(module, KimiDeltaAttention):
                module._reset_parameters()

    # ------------------------------------------------------------------
    # Effort helpers
    # ------------------------------------------------------------------
    def _flags_for_level(self, level: int) -> Dict[str, Any]:
        force_skip_attn = False
        force_skip_moe = False
        moe_top_k = None
        force_latent_steps = 0
        if level <= 0:
            force_skip_attn = True
            force_skip_moe = True
        elif level == 1:
            force_skip_attn = True
            moe_top_k = 1
        elif level == 2:
            pass
        else:  # >= 3
            force_latent_steps = getattr(self.config, "default_e3_steps", 2)
        return {
            "force_skip_attention": force_skip_attn,
            "force_skip_moe": force_skip_moe,
            "moe_top_k_override": moe_top_k,
            "force_latent_steps": force_latent_steps,
        }

    def compute_effort_probe(
        self,
        hidden: Tensor,
        attention_mask: Optional[Tensor] = None,
        past_recurrent_states: Optional[List[Optional[Tensor]]] = None,
        use_cache: bool = False,
    ) -> Tuple[Tensor, List, EffortDecision]:
        """
        Cheap mandatory probe = layer-0 recurrent only (reused as E0 work).
        Returns (probe_hidden, new_rec_states, decision).
        Policy feature = last valid token (same as stored hidden_anchor).
        """
        from .effort import EffortController
        layer0 = self.layers[0]
        probe_h, new_rec, meta = layer0.run_recurrent_only(
            hidden,
            attention_mask=attention_mask,
            recurrent_states=past_recurrent_states,
            use_cache=use_cache,
        )
        anchor = EffortController.pool_last_valid(probe_h, attention_mask)
        effort_info = layer0.effort(anchor)  # (B, H) — no mean-pool mismatch
        probs = effort_info["effort_probs"]
        if probs.dim() == 3:
            probs = probs.mean(dim=1)
        decision = decide_from_probs(
            probs,
            source_layer=0,
            source_position="post_probe",
            hidden_anchor=anchor.detach(),
        )
        return probe_h, new_rec, decision

    def _run_layers(
        self,
        hidden: Tensor,
        attention_mask: Optional[Tensor],
        past_cache: Optional[ModelCache],
        use_cache: bool,
        emitter: Optional[Any],
        flags: Dict[str, Any],
        start_layer: int = 0,
        reuse_layer0_hidden: Optional[Tensor] = None,
    ) -> Tuple[Tensor, List[Dict[str, Any]], Optional[ModelCache]]:
        all_meta: List[Dict[str, Any]] = []
        next_cache = ModelCache.empty(len(self.layers)) if use_cache else None
        if self.attn_res_bank is not None:
            self.attn_res_bank.reset()

        for i, layer in enumerate(self.layers):
            if i == 0 and reuse_layer0_hidden is not None and start_layer == 0:
                # Continue after probe: skip re-running full layer0 recurrent+effort
                # by applying only expensive branches of layer0 with probe as input
                hidden, meta = layer(
                    reuse_layer0_hidden if flags.get("force_skip_attention") and flags.get("force_skip_moe")
                    else (reuse_layer0_hidden if False else hidden),
                    attention_mask=attention_mask,
                    recurrent_states=None,
                    past_attention_k=None,
                    past_attention_v=None,
                    use_cache=use_cache,
                    emitter=emitter,
                    layer_index=i,
                    **flags,
                )
                # Simpler: always run full layer from current hidden for correctness;
                # probe is only for decision. Avoid double-count complexity for v3.1.2.
            layer_past_rec = None
            past_k = past_v = None
            if past_cache is not None and i < len(past_cache.layers):
                layer_past_rec = past_cache.layers[i].recurrent_states
                past_k = past_cache.layers[i].attention_k
                past_v = past_cache.layers[i].attention_v

            hidden, meta = layer(
                hidden,
                attention_mask=attention_mask,
                recurrent_states=layer_past_rec,
                past_attention_k=past_k,
                past_attention_v=past_v,
                use_cache=use_cache,
                emitter=emitter,
                layer_index=i,
                **flags,
            )
            all_meta.append(meta)
            if use_cache and next_cache is not None:
                next_cache.layers[i] = LayerCache(
                    recurrent_states=meta.get("recurrent_states"),
                    attention_k=meta.get("attention_k"),
                    attention_v=meta.get("attention_v"),
                )
            if self.attn_res_bank is not None:
                depth_contrib = self.attn_res_bank(hidden)
                hidden = hidden + depth_contrib
                self.attn_res_bank.push(hidden)

        if next_cache is not None:
            next_cache.sequence_length = (
                (past_cache.sequence_length if past_cache else 0) + hidden.size(1)
            )
        return hidden, all_meta, next_cache

    def _forward_from_probe(
        self,
        probe_h: Tensor,
        new_rec0: list,
        attention_mask: Optional[Tensor],
        past_cache: Optional[ModelCache],
        use_cache: bool,
        emitter: Optional[Any],
        decision: EffortDecision,
    ) -> Tuple[Tensor, List[Dict[str, Any]], Optional[ModelCache], ComputeStats]:
        """Continue from layer-0 recurrent probe without re-running recurrence."""
        B, L, H = probe_h.shape
        levels = decision.levels
        out_hidden = torch.zeros_like(probe_h)
        total_stats = ComputeStats(effort_levels=levels.clone())
        total_stats.per_sample_compute = nominal_compute_for_levels(levels)
        total_stats.recurrent_token_evals += B * L * self.config.num_recurrent_per_block
        total_stats.recurrent_iterations += self.config.num_recurrent_per_block

        last_meta: List[Dict[str, Any]] = []
        next_cache = ModelCache.empty(len(self.layers)) if use_cache else None
        if next_cache is not None:
            next_cache.effort_levels = levels.clone()
            next_cache.sequence_length = (past_cache.sequence_length if past_cache else 0) + L

        for level in levels.unique().tolist():
            level = int(level)
            indices = (levels == level).nonzero(as_tuple=False).squeeze(-1)
            if indices.numel() == 0:
                continue
            flags = self._flags_for_level(level)
            sub_probe = probe_h.index_select(0, indices)
            sub_mask = attention_mask.index_select(0, indices) if attention_mask is not None else None
            sub_rec0 = None
            if new_rec0 is not None:
                sub_rec0 = []
                for st in new_rec0:
                    if st is None:
                        sub_rec0.append(None)
                    elif isinstance(st, tuple):
                        sub_rec0.append(tuple(t.index_select(0, indices) if t is not None else None for t in st))
                    else:
                        sub_rec0.append(st.index_select(0, indices))
            past_k = past_v = None
            if past_cache is not None and past_cache.layers[0].attention_k is not None:
                past_k = past_cache.layers[0].attention_k.index_select(0, indices)
                past_v = past_cache.layers[0].attention_v.index_select(0, indices)

            h0, meta0 = self.layers[0].continue_after_recurrent(
                sub_probe,
                attention_mask=sub_mask,
                past_attention_k=past_k,
                past_attention_v=past_v,
                recurrent_states_out=sub_rec0,
                use_cache=use_cache,
                emitter=emitter,
                layer_index=0,
                **flags,
            )
            metas = [meta0]
            if use_cache and next_cache is not None:
                self._scatter_layer_cache(next_cache, 0, indices, B, meta0)

            h = h0
            for i in range(1, len(self.layers)):
                layer_past_rec = layer_past_k = layer_past_v = None
                if past_cache is not None and i < len(past_cache.layers):
                    sub_past = past_cache.index_select(indices)
                    layer_past_rec = sub_past.layers[i].recurrent_states
                    layer_past_k = sub_past.layers[i].attention_k
                    layer_past_v = sub_past.layers[i].attention_v
                h, meta = self.layers[i](
                    h,
                    attention_mask=sub_mask,
                    recurrent_states=layer_past_rec,
                    past_attention_k=layer_past_k,
                    past_attention_v=layer_past_v,
                    use_cache=use_cache,
                    emitter=emitter,
                    layer_index=i,
                    **flags,
                )
                metas.append(meta)
                if use_cache and next_cache is not None:
                    self._scatter_layer_cache(next_cache, i, indices, B, meta)

            out_hidden.index_copy_(0, indices, h)
            nb = indices.numel()
            total_stats.attention_token_evals += 0 if flags["force_skip_attention"] else nb * L * self.config.num_layers
            topk = flags["moe_top_k_override"] or self.config.top_k_experts
            total_stats.expert_token_evals += 0 if flags["force_skip_moe"] else nb * L * self.config.num_layers * topk
            if not flags["force_skip_moe"]:
                n_shared = getattr(self.config, "num_shared_experts", 0)
                total_stats.shared_expert_token_evals += nb * L * self.config.num_layers * max(n_shared, 0)
            total_stats.latent_refine_token_evals += nb * L * flags["force_latent_steps"] * self.config.num_layers
            total_stats.latent_iterations += flags["force_latent_steps"] * self.config.num_layers
            total_stats.recurrent_token_evals += nb * L * max(0, self.config.num_layers - 1) * self.config.num_recurrent_per_block
            total_stats.recurrent_iterations += max(0, self.config.num_layers - 1) * self.config.num_recurrent_per_block
            last_meta = metas

        # Rebuild aggregate from per-level estimate_compute for consistency with fixed path
        agg = ComputeStats(effort_levels=levels.clone())
        for level in levels.unique().tolist():
            level = int(level)
            nb = int((levels == level).sum().item())
            if nb == 0:
                continue
            flags = self._flags_for_level(level)
            s = estimate_compute(
                batch_size=nb,
                seq_len=L,
                hidden_size=H,
                num_layers=self.config.num_layers,
                num_recurrent_per_block=self.config.num_recurrent_per_block,
                effort_level=level,
                top_k=self.config.top_k_experts,
                num_shared_experts=getattr(self.config, "num_shared_experts", 0),
                latent_steps=flags["force_latent_steps"],
                latent_size=self.config.latent_size,
                force_skip_attention=flags["force_skip_attention"],
                force_skip_moe=flags["force_skip_moe"],
                recurrent_type=getattr(self.config, "recurrent_type", "ssm"),
                past_seq_len=int(past_cache.sequence_length) if past_cache is not None else 0,
            )
            agg.merge_from(s)
        agg.effort_levels = levels.clone()
        agg.per_sample_compute = nominal_compute_for_levels(levels)
        return out_hidden, last_meta, next_cache, agg

    def _scatter_layer_cache(self, next_cache: ModelCache, layer_i: int, indices: Tensor, B: int, meta: Dict) -> None:
        src_rec = meta.get("recurrent_states")
        src_k = meta.get("attention_k")
        src_v = meta.get("attention_v")
        dst = next_cache.layers[layer_i]
        if src_rec is not None:
            if dst.recurrent_states is None:
                dst_rec = []
                for st in src_rec:
                    if st is None:
                        dst_rec.append(None)
                    elif isinstance(st, tuple):
                        dst_rec.append(tuple(
                            torch.zeros(B, *tt.shape[1:], device=tt.device, dtype=tt.dtype) if tt is not None else None
                            for tt in st
                        ))
                    else:
                        dst_rec.append(torch.zeros(B, *st.shape[1:], device=st.device, dtype=st.dtype))
                dst.recurrent_states = dst_rec
            for j, st in enumerate(src_rec):
                if st is None:
                    continue
                if isinstance(st, tuple):
                    for k, tt in enumerate(st):
                        if tt is not None:
                            dst.recurrent_states[j][k].index_copy_(0, indices, tt)
                else:
                    dst.recurrent_states[j].index_copy_(0, indices, st)
        if src_k is not None:
            if dst.attention_k is None:
                dst.attention_k = torch.zeros(B, *src_k.shape[1:], device=src_k.device, dtype=src_k.dtype)
                dst.attention_v = torch.zeros(B, *src_v.shape[1:], device=src_v.device, dtype=src_v.dtype)
            dst.attention_k.index_copy_(0, indices, src_k)
            dst.attention_v.index_copy_(0, indices, src_v)


    def _forward_uniform(
        self,
        hidden: Tensor,
        attention_mask: Optional[Tensor],
        past_cache: Optional[ModelCache],
        use_cache: bool,
        emitter: Optional[Any],
        level: int,
    ) -> Tuple[Tensor, List[Dict[str, Any]], Optional[ModelCache], ComputeStats]:
        flags = self._flags_for_level(level)
        hidden, all_meta, next_cache = self._run_layers(
            hidden, attention_mask, past_cache, use_cache, emitter, flags
        )
        B, L, H = hidden.shape
        past_len = int(past_cache.sequence_length) if past_cache is not None else 0
        stats = estimate_compute(
            batch_size=B,
            seq_len=L,
            hidden_size=H,
            num_layers=self.config.num_layers,
            num_recurrent_per_block=self.config.num_recurrent_per_block,
            effort_level=level,
            top_k=self.config.top_k_experts,
            num_shared_experts=getattr(self.config, "num_shared_experts", 0),
            latent_steps=flags["force_latent_steps"],
            latent_size=self.config.latent_size,
            force_skip_attention=flags["force_skip_attention"],
            force_skip_moe=flags["force_skip_moe"],
            recurrent_type=getattr(self.config, "recurrent_type", "ssm"),
            past_seq_len=past_len,
        )
        stats.effort_levels = torch.full((B,), level, dtype=torch.long, device=hidden.device)
        stats.per_sample_compute = nominal_compute_for_levels(stats.effort_levels)
        return hidden, all_meta, next_cache, stats

    def _forward_mixed_adaptive(
        self,
        hidden: Tensor,
        attention_mask: Optional[Tensor],
        past_cache: Optional[ModelCache],
        use_cache: bool,
        emitter: Optional[Any],
        decision: EffortDecision,
    ) -> Tuple[Tensor, List[Dict[str, Any]], Optional[ModelCache], ComputeStats]:
        """Partition batch by effort level and execute only required paths."""
        B, L, H = hidden.shape
        levels = decision.levels  # (B,)
        device = hidden.device
        out_hidden = torch.zeros_like(hidden)
        total_stats = ComputeStats(effort_levels=levels.clone())
        total_stats.per_sample_compute = nominal_compute_for_levels(levels)
        # Use last group metas for telemetry shape (not perfect but informative)
        last_meta: List[Dict[str, Any]] = []
        next_cache = ModelCache.empty(len(self.layers)) if use_cache else None
        if next_cache is not None:
            next_cache.effort_levels = levels.clone()
            next_cache.sequence_length = (past_cache.sequence_length if past_cache else 0) + L

        unique_levels = levels.unique().tolist()
        for level in unique_levels:
            level = int(level)
            indices = (levels == level).nonzero(as_tuple=False).squeeze(-1)
            if indices.numel() == 0:
                continue
            sub_h = hidden.index_select(0, indices)
            sub_mask = (
                attention_mask.index_select(0, indices) if attention_mask is not None else None
            )
            sub_past = past_cache.index_select(indices) if past_cache is not None else None
            sub_out, metas, sub_cache, sub_stats = self._forward_uniform(
                sub_h, sub_mask, sub_past, use_cache, emitter, level
            )
            out_hidden.index_copy_(0, indices, sub_out)
            total_stats.merge_from(sub_stats)
            last_meta = metas
            if use_cache and next_cache is not None and sub_cache is not None:
                for li in range(len(self.layers)):
                    # scatter layer caches — only fill selected rows
                    src_lc = sub_cache.layers[li]
                    dst_lc = next_cache.layers[li]
                    if src_lc.recurrent_states is not None:
                        if dst_lc.recurrent_states is None:
                            # init full-batch placeholders from first subset shapes
                            dst_rec = []
                            for st in src_lc.recurrent_states:
                                if st is None:
                                    dst_rec.append(None)
                                elif isinstance(st, tuple):
                                    dst_rec.append(
                                        tuple(
                                            torch.zeros(
                                                B, *t.shape[1:], device=t.device, dtype=t.dtype
                                            )
                                            if t is not None
                                            else None
                                            for t in st
                                        )
                                    )
                                else:
                                    dst_rec.append(
                                        torch.zeros(B, *st.shape[1:], device=st.device, dtype=st.dtype)
                                    )
                            dst_lc.recurrent_states = dst_rec
                        for j, st in enumerate(src_lc.recurrent_states):
                            if st is None:
                                continue
                            if isinstance(st, tuple):
                                for k, t in enumerate(st):
                                    if t is not None:
                                        dst_lc.recurrent_states[j][k].index_copy_(0, indices, t)
                            else:
                                dst_lc.recurrent_states[j].index_copy_(0, indices, st)
                    if src_lc.attention_k is not None:
                        if dst_lc.attention_k is None:
                            dst_lc.attention_k = torch.zeros(
                                B,
                                *src_lc.attention_k.shape[1:],
                                device=src_lc.attention_k.device,
                                dtype=src_lc.attention_k.dtype,
                            )
                            dst_lc.attention_v = torch.zeros(
                                B,
                                *src_lc.attention_v.shape[1:],
                                device=src_lc.attention_v.device,
                                dtype=src_lc.attention_v.dtype,
                            )
                        dst_lc.attention_k.index_copy_(0, indices, src_lc.attention_k)
                        dst_lc.attention_v.index_copy_(0, indices, src_lc.attention_v)

        return out_hidden, last_meta, next_cache, total_stats

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Optional[Tensor] = None,
        past_recurrent_states: Optional[List[List[Optional[Tensor]]]] = None,
        cache: Optional[ModelCache] = None,
        use_cache: bool = False,
        return_dict: bool = True,
        emitter: Optional[Any] = None,
        effort_mode: str = "disabled",
        effort_levels_override: Optional[Tensor] = None,
    ) -> Union[Tensor, Dict[str, Any]]:
        if input_ids.dim() != 2:
            raise ValueError(f"input_ids must be (B, L); got {tuple(input_ids.shape)}")

        B, L = input_ids.shape
        hidden = self.embed(input_ids)

        # Resolve cache
        if cache is not None:
            past_cache = cache
        elif past_recurrent_states is not None:
            past_cache = ModelCache.empty(len(self.layers))
            for i, rs in enumerate(past_recurrent_states):
                if i < len(past_cache.layers):
                    past_cache.layers[i].recurrent_states = rs
        else:
            past_cache = None

        decision: Optional[EffortDecision] = None
        stats: Optional[ComputeStats] = None
        all_meta: List[Dict[str, Any]] = []
        next_cache: Optional[ModelCache] = None

        # Persist effort from cache during decode
        if past_cache is not None and past_cache.effort_levels is not None and effort_mode == "adaptive":
            levels = past_cache.effort_levels
            if levels.numel() == B:
                # uniform if all same, else mixed
                if int(levels.min()) == int(levels.max()):
                    hidden, all_meta, next_cache, stats = self._forward_uniform(
                        hidden, attention_mask, past_cache, use_cache, emitter, int(levels[0])
                    )
                else:
                    decision = EffortDecision(
                        probabilities=torch.zeros(B, self.config.effort_levels, device=hidden.device),
                        levels=levels,
                        confidence=torch.ones(B, device=hidden.device),
                        entropy=torch.zeros(B, device=hidden.device),
                    )
                    hidden, all_meta, next_cache, stats = self._forward_mixed_adaptive(
                        hidden, attention_mask, past_cache, use_cache, emitter, decision
                    )
            else:
                raise ValueError("cache.effort_levels batch size mismatch")
        elif effort_levels_override is not None:
            levels = effort_levels_override.to(device=hidden.device, dtype=torch.long)
            if levels.numel() != B:
                raise ValueError("effort_levels_override must have shape (B,)")
            decision = EffortDecision(
                probabilities=torch.zeros(B, self.config.effort_levels, device=hidden.device),
                levels=levels,
                confidence=torch.ones(B, device=hidden.device),
                entropy=torch.zeros(B, device=hidden.device),
                source_position="override",
            )
            if int(levels.min()) == int(levels.max()):
                hidden, all_meta, next_cache, stats = self._forward_uniform(
                    hidden, attention_mask, past_cache, use_cache, emitter, int(levels[0])
                )
            else:
                hidden, all_meta, next_cache, stats = self._forward_mixed_adaptive(
                    hidden, attention_mask, past_cache, use_cache, emitter, decision
                )
        elif effort_mode == "adaptive":
            # Probe = layer-0 recurrent (reused, not discarded)
            past_rec0 = None
            if past_cache is not None and len(past_cache.layers) > 0:
                past_rec0 = past_cache.layers[0].recurrent_states
            probe_h, new_rec0, decision = self.compute_effort_probe(
                hidden, attention_mask, past_recurrent_states=past_rec0, use_cache=use_cache
            )
            levels = decision.levels
            # Continue from probe: partition by level without re-running layer-0 recurrent
            hidden, all_meta, next_cache, stats = self._forward_from_probe(
                probe_h,
                new_rec0,
                attention_mask,
                past_cache,
                use_cache,
                emitter,
                decision,
            )
            if next_cache is not None:
                next_cache.effort_levels = levels.clone()
        elif effort_mode.startswith("fixed_"):
            level = int(effort_mode.split("_")[1])
            hidden, all_meta, next_cache, stats = self._forward_uniform(
                hidden, attention_mask, past_cache, use_cache, emitter, level
            )
        else:
            # disabled / full
            hidden, all_meta, next_cache, stats = self._forward_uniform(
                hidden, attention_mask, past_cache, use_cache, emitter, 2
            )

        hidden = self.final_norm(hidden)
        logits = self.lm_head(hidden)

        if not return_dict:
            return logits

        out: Dict[str, Any] = {
            "logits": logits,
            "effort_scores": [m.get("effort_score") for m in all_meta],
            "compute_stats": stats.to_dict() if stats is not None else {},
            "effort_decision": decision.to_dict() if decision is not None else None,
        }
        # convenience mirrors
        if stats is not None and stats.effort_levels is not None:
            out["compute_stats"]["attention_executed"] = [
                not (int(stats.effort_levels[0]) <= 1) for _ in all_meta
            ] if stats.effort_levels.numel() else []
            # better: per-layer from meta
            out["compute_stats"]["attention_executed"] = [
                m.get("attention_executed") for m in all_meta
            ]
            out["compute_stats"]["moe_executed"] = [m.get("moe_executed") for m in all_meta]
            out["compute_stats"]["latent_steps"] = [m.get("latent_steps", 0) for m in all_meta]
            out["compute_stats"]["effort_mode"] = effort_mode
            if decision is not None:
                out["compute_stats"]["chosen_effort_levels"] = decision.levels.tolist()
                out["compute_stats"]["chosen_effort_level"] = (
                    int(decision.levels[0]) if decision.levels.numel() == 1
                    or int(decision.levels.min()) == int(decision.levels.max())
                    else decision.levels.tolist()
                )
            elif stats.effort_levels is not None:
                out["compute_stats"]["chosen_effort_level"] = int(stats.effort_levels[0])

        if use_cache:
            out["cache"] = next_cache
            if next_cache is not None:
                out["past_recurrent_states"] = [
                    lc.recurrent_states for lc in next_cache.layers
                ]
        if emitter is not None:
            emitter.emit_forward_summary(
                num_layers=len(self.layers),
                effort_scores=out["effort_scores"],
                extra={"logits_shape": list(logits.shape)},
            )
        return out



    @torch.no_grad()
    def generate(
        self,
        input_ids: Tensor,
        attention_mask: Optional[Tensor] = None,
        max_new_tokens: int = 16,
        effort_mode: str = "fixed_2",
        effort_levels_override: Optional[Tensor] = None,
        eos_token_id: Optional[int] = None,
        pad_token_id: Optional[int] = None,
        tokenizer: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """
        Greedy autoregressive generation with effort-aware caching.

        Padding-aware: next token is chosen from the last *valid* position,
        and full attention_mask is persisted in the cache for decode.
        """
        if input_ids.dim() != 2:
            raise ValueError(f"input_ids must be (B, L); got {tuple(input_ids.shape)}")
        self.eval()
        B = input_ids.size(0)
        sequences = input_ids.clone()
        mask = attention_mask
        if mask is None:
            mask = torch.ones_like(sequences)
        else:
            mask = mask.to(device=sequences.device)

        # Prefill
        out = self.forward(
            sequences,
            attention_mask=mask,
            use_cache=True,
            effort_mode=effort_mode,
            effort_levels_override=effort_levels_override,
        )
        cache = out["cache"]
        if cache is not None:
            cache.attention_mask = mask.clone()
        stats_acc = out.get("compute_stats") or {}
        generated = []
        finished = torch.zeros(B, dtype=torch.bool, device=sequences.device)

        def _last_valid_logits(logits: Tensor, m: Tensor) -> Tensor:
            # logits (B, L, V), m (B, L) — gather last valid position per row
            lengths = m.long().sum(dim=1).clamp(min=1) - 1  # (B,)
            idx = lengths.view(B, 1, 1).expand(-1, 1, logits.size(-1))
            return logits.gather(1, idx).squeeze(1)  # (B, V)

        for step in range(max_new_tokens):
            if step == 0:
                next_logits = _last_valid_logits(out["logits"], mask)
            else:
                # decode step: single new token is always last position of L_new=1
                next_logits = out["logits"][:, -1, :]

            next_id = next_logits.argmax(dim=-1, keepdim=True)
            if eos_token_id is not None and pad_token_id is not None:
                next_id = torch.where(
                    finished.view(B, 1),
                    torch.full_like(next_id, pad_token_id),
                    next_id,
                )
            generated.append(next_id)
            sequences = torch.cat([sequences, next_id], dim=1)
            ones = torch.ones(B, 1, device=sequences.device, dtype=mask.dtype)
            # full mask for next step includes history
            mask = torch.cat([mask, ones], dim=1)
            if cache is not None:
                cache.attention_mask = mask.clone()

            if eos_token_id is not None:
                finished = finished | (next_id.squeeze(-1) == eos_token_id)
                if bool(finished.all()):
                    break

            # pass FULL mask so cached keys from pad positions stay masked
            out = self.forward(
                next_id,
                attention_mask=mask,  # full sequence mask
                cache=cache,
                use_cache=True,
                effort_mode=effort_mode,
                effort_levels_override=effort_levels_override,
            )
            cache = out.get("cache", cache)
            if cache is not None:
                cache.attention_mask = mask.clone()
            step_stats = out.get("compute_stats") or {}
            # merge counters
            for k, v in step_stats.items():
                if isinstance(v, (int, float)) and k in stats_acc and isinstance(stats_acc.get(k), (int, float)):
                    stats_acc[k] = stats_acc[k] + v
                elif k not in stats_acc:
                    stats_acc[k] = v

        gen_ids = torch.cat(generated, dim=1) if generated else sequences.new_zeros(B, 0)
        stats_acc["num_decode_steps"] = len(generated)

        result: Dict[str, Any] = {
            "sequences": sequences,
            "generated_ids": gen_ids,
            "compute_stats": stats_acc,
            "effort_mode": effort_mode,
        }
        if tokenizer is not None and hasattr(tokenizer, "decode"):
            texts = []
            for b in range(B):
                texts.append(tokenizer.decode(gen_ids[b].tolist(), skip_special_tokens=True))
            result["generated_text"] = texts if B > 1 else texts[0]
        return result


    def get_input_embeddings(self) -> nn.Embedding:
        return self.embed

    def set_input_embeddings(self, value: nn.Embedding) -> None:
        self.embed = value
        if self.config.tie_word_embeddings:
            self.lm_head.weight = self.embed.weight

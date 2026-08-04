"""
Pretrained weight import for ExFusion.

Goal: map a dense causal LM (Qwen/LLaMA-style) into the E2-compatible path
so adaptation trains new modules (SSM/KDA, MoE router, latent refine, gates)
around a competent language backbone.

Optional dependency: transformers (for HF download). Local .pt works without it.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
from torch import Tensor

from .config import DAPHConfigV3
from .model import DAPHHybridModelV3


@dataclass
class PretrainedImportReport:
    source_model: str
    source_revision: str
    source_digest: str
    matched_parameters: int          # exact shape match, full copy
    transformed_parameters: int      # vocab-resize ONLY for embed/lm_head
    source_seeded_parameters: int     # truncated/seeded from source (not safe retention)
    partial_block_parameters: int     # dangerous overlapping submatrix copy
    newly_initialized_parameters: int
    skipped_parameters: int
    coverage_percent: float          # (matched + vocab-transformed) / total
    exact_coverage_percent: float = 0.0
    notes: List[str] = field(default_factory=list)
    matched_keys: List[str] = field(default_factory=list)
    transformed_keys: List[str] = field(default_factory=list)
    source_seeded_keys: List[str] = field(default_factory=list)
    partial_block_keys: List[str] = field(default_factory=list)
    new_keys: List[str] = field(default_factory=list)
    skipped_keys: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def save(self, path: str) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2))


def _tensor_digest(t: Tensor) -> str:
    x = t.detach().cpu().contiguous()
    h = hashlib.sha256()
    h.update(str(tuple(x.shape)).encode())
    h.update(str(x.dtype).encode())
    h.update(bytes(x.reshape(-1).view(torch.uint8).numpy()))
    return h.hexdigest()[:16]


def _state_digest(sd: Dict[str, Tensor]) -> str:
    h = hashlib.sha256()
    for k in sorted(sd.keys()):
        h.update(k.encode())
        h.update(_tensor_digest(sd[k]).encode())
    return h.hexdigest()[:32]


def research_config(
    *,
    hidden_size: int = 256,
    num_layers: int = 6,
    num_heads: int = 8,
    vocab_size: int = 32000,
    num_experts: int = 8,
    top_k: int = 2,
) -> DAPHConfigV3:
    """~20–40M param research config — do not train the 466M default from scratch."""
    return DAPHConfigV3(
        hidden_size=hidden_size,
        latent_size=max(64, hidden_size // 2),
        num_layers=num_layers,
        num_attention_heads=num_heads,
        state_size=max(8, hidden_size // 16),
        num_recurrent_per_block=1,
        num_routed_experts=num_experts,
        top_k_experts=top_k,
        num_shared_experts=1,
        vocab_size=vocab_size,
        use_attn_res=False,
        dropout=0.0,
        use_quantile_balancing=False,
        use_load_balancing=True,
        default_e3_steps=2,
        tie_word_embeddings=True,
        use_rope=False,  # set True when matching Qwen
        norm_type="layer",  # set "rms" when matching Qwen
        shared_ffn="swiglu",
    )


def zero_init_new_modules(model: nn.Module) -> List[str]:
    """
    Branch-specific init for retention:
      - gate_attn / gate_moe → identity (preserve imported backbone)
      - gate_rec, latent_refine, effort, router → near-zero (new ExFusion)
    """
    from .gates import ChannelGate

    touched: List[str] = []
    for name, module in model.named_modules():
        if isinstance(module, ChannelGate):
            if name.endswith("gate_attn") or name.endswith("gate_moe"):
                module.identity_init()
                touched.append(name + ":identity")
            else:
                # gate_rec and any other gates
                module.zero_out_init()
                touched.append(name + ":zero")
            continue
        lname = name.lower()
        if any(s in lname for s in ("effort", "latent_refine", "attn_res", "moe.router", "moe.expert_bias")):
            for pname, p in module.named_parameters(recurse=False):
                if p.dim() >= 2:
                    nn.init.normal_(p, mean=0.0, std=1e-3)
                else:
                    nn.init.zeros_(p)
                touched.append(f"{name}.{pname}")
    return sorted(set(touched))


def freeze_pretrained_keys(
    model: DAPHHybridModelV3,
    matched_keys: Sequence[str],
) -> int:
    """Freeze parameters that were imported from the source model."""
    n = 0
    matched = set(matched_keys)
    for name, p in model.named_parameters():
        if name in matched:
            p.requires_grad = False
            n += 1
    return n


VOCAB_RESIZABLE_KEYS = frozenset({"embed.weight", "lm_head.weight"})


def _copy_compatible(
    dst: Tensor,
    src: Tensor,
    key: str,
    report: PretrainedImportReport,
    *,
    allow_partial_block: bool = False,
    allow_source_seed: bool = False,
) -> bool:
    """
    Copy policy (strict by default):
      1. exact shape → matched
      2. embed/lm_head only, same H, different V → vocab-transformed (safe)
      3. other same-column truncation → source_seeded (NOT safe coverage) if allowed
      4. partial block → only if allow_partial_block
      else skip
    """
    if dst.shape == src.shape:
        dst.data.copy_(src.to(dtype=dst.dtype))
        report.matched_parameters += int(dst.numel())
        report.matched_keys.append(key)
        return True
    # Safe vocab resize — ONLY for embedding / lm_head
    if (
        key in VOCAB_RESIZABLE_KEYS
        and dst.dim() == 2
        and src.dim() == 2
        and dst.shape[1] == src.shape[1]
    ):
        rows = min(dst.shape[0], src.shape[0])
        dst.data.zero_()
        dst.data[:rows].copy_(src[:rows].to(dtype=dst.dtype))
        report.transformed_parameters += int(dst.numel())
        report.transformed_keys.append(key)
        report.notes.append(
            f"vocab-resize {key}: src={tuple(src.shape)}→dst={tuple(dst.shape)}"
        )
        return True
    # Source-seeded truncation (e.g. routed experts from FFN) — not safe retention
    if (
        allow_source_seed
        and dst.dim() == 2
        and src.dim() == 2
        and dst.shape[1] == src.shape[1]
    ):
        rows = min(dst.shape[0], src.shape[0])
        dst.data.zero_()
        dst.data[:rows].copy_(src[:rows].to(dtype=dst.dtype))
        report.source_seeded_parameters += int(dst.numel())
        report.source_seeded_keys.append(key)
        report.notes.append(
            f"source-seeded {key}: src={tuple(src.shape)}→dst={tuple(dst.shape)}"
        )
        return True
    if allow_partial_block and dst.dim() == 2 and src.dim() == 2:
        r, c = min(dst.shape[0], src.shape[0]), min(dst.shape[1], src.shape[1])
        dst.data.zero_()
        dst.data[:r, :c].copy_(src[:r, :c].to(dtype=dst.dtype))
        report.partial_block_parameters += int(dst.numel())
        report.partial_block_keys.append(key)
        report.notes.append(
            f"UNSAFE partial-block {key}: src={tuple(src.shape)}→dst={tuple(dst.shape)}"
        )
        return True
    report.skipped_parameters += int(dst.numel())
    report.skipped_keys.append(key)
    report.notes.append(
        f"shape mismatch skipped {key}: src={tuple(src.shape)} dst={tuple(dst.shape)}"
    )
    return False


def build_qwen_key_map(
    exfusion_sd: Dict[str, Tensor],
    qwen_sd: Dict[str, Tensor],
) -> Dict[str, str]:
    """
    source_key → model_key for Qwen2 / LLaMA-style HF state dicts.

    Maps:
      embed, lm_head, final norm
      per-layer attn q/k/v/o and input_layernorm
      per-layer MLP gate/up/down → shared expert (+ optional routed copies)
    """
    mapping: Dict[str, str] = {}

    embed_aliases = [
        ("model.embed_tokens.weight", "embed.weight"),
        ("embed_tokens.weight", "embed.weight"),
        ("transformer.wte.weight", "embed.weight"),
    ]
    for sk, mk in embed_aliases:
        if sk in qwen_sd and mk in exfusion_sd:
            mapping[sk] = mk

    lm_aliases = [
        ("lm_head.weight", "lm_head.weight"),
    ]
    for sk, mk in lm_aliases:
        if sk in qwen_sd and mk in exfusion_sd:
            mapping[sk] = mk

    # final norm
    for sk in ("model.norm.weight", "model.norm.bias", "transformer.ln_f.weight"):
        if sk in qwen_sd:
            mk = "final_norm.weight" if sk.endswith("weight") else "final_norm.bias"
            if mk in exfusion_sd:
                mapping[sk] = mk

    # Discover layer count from both sides
    def _layers(sd: Dict[str, Tensor], pattern: str) -> int:
        ids = set()
        for k in sd:
            m = re.search(pattern, k)
            if m:
                ids.add(int(m.group(1)))
        return (max(ids) + 1) if ids else 0

    n_src = _layers(qwen_sd, r"layers\.(\d+)\.")
    n_dst = _layers(exfusion_sd, r"layers\.(\d+)\.")
    n = min(n_src, n_dst)

    # Qwen2 naming
    for i in range(n):
        pairs = [
            (f"model.layers.{i}.self_attn.q_proj.weight", f"layers.{i}.attn.q_proj.weight"),
            (f"model.layers.{i}.self_attn.k_proj.weight", f"layers.{i}.attn.k_proj.weight"),
            (f"model.layers.{i}.self_attn.v_proj.weight", f"layers.{i}.attn.v_proj.weight"),
            (f"model.layers.{i}.self_attn.o_proj.weight", f"layers.{i}.attn.out_proj.weight"),
            (f"model.layers.{i}.self_attn.q_proj.bias", f"layers.{i}.attn.q_proj.bias"),
            (f"model.layers.{i}.self_attn.k_proj.bias", f"layers.{i}.attn.k_proj.bias"),
            (f"model.layers.{i}.self_attn.v_proj.bias", f"layers.{i}.attn.v_proj.bias"),
            (f"model.layers.{i}.self_attn.o_proj.bias", f"layers.{i}.attn.out_proj.bias"),

            (f"model.layers.{i}.input_layernorm.weight", f"layers.{i}.attn_norm.weight"),
            (f"model.layers.{i}.post_attention_layernorm.weight", f"layers.{i}.final_norm.weight"),
            # MLP → SharedSwiGLU (exact three-matrix map)
            (f"model.layers.{i}.mlp.gate_proj.weight", f"layers.{i}.moe.shared.0.gate_proj.weight"),
            (f"model.layers.{i}.mlp.up_proj.weight", f"layers.{i}.moe.shared.0.up_proj.weight"),
            (f"model.layers.{i}.mlp.down_proj.weight", f"layers.{i}.moe.shared.0.down_proj.weight"),
        ]
        for sk, mk in pairs:
            if sk in qwen_sd and mk in exfusion_sd:
                mapping[sk] = mk

        # Optional: seed routed experts from same FFN (small noise applied later)
        for e in range(8):
            mk_w1 = f"layers.{i}.moe.routed.{e}.w1.weight"
            mk_w3 = f"layers.{i}.moe.routed.{e}.w3.weight"
            sk_gate = f"model.layers.{i}.mlp.gate_proj.weight"
            sk_up = f"model.layers.{i}.mlp.up_proj.weight"
            sk_down = f"model.layers.{i}.mlp.down_proj.weight"
            # only map if shapes will be handled by block-copy
            if sk_gate in qwen_sd and mk_w1 in exfusion_sd:
                mapping[f"{sk_gate}#routed{e}_w1"] = mk_w1  # virtual key handled below
            if sk_up in qwen_sd and mk_w3 in exfusion_sd:
                mapping[f"{sk_up}#routed{e}_w3"] = mk_w3

    return mapping


def import_state_dict(
    model: nn.Module,
    source_sd: Dict[str, Tensor],
    *,
    source_name: str = "external",
    source_revision: str = "unknown",
    key_map: Optional[Dict[str, str]] = None,
    zero_init_new: bool = True,
    seed_routed_from_ffn: bool = True,
    allow_partial_block: bool = False,
    allow_source_seed: bool = False,
) -> PretrainedImportReport:
    report = PretrainedImportReport(
        source_model=source_name,
        source_revision=source_revision,
        source_digest=_state_digest(source_sd),
        matched_parameters=0,
        transformed_parameters=0,
        source_seeded_parameters=0,
        partial_block_parameters=0,
        newly_initialized_parameters=0,
        skipped_parameters=0,
        coverage_percent=0.0,
        exact_coverage_percent=0.0,
    )
    model_sd = model.state_dict()
    if key_map is None:
        key_map = build_qwen_key_map(model_sd, source_sd)

    # Resolve virtual routed keys: "src#routed{e}_w1" → actual src tensor
    resolved: Dict[str, Tuple[str, Tensor]] = {}  # model_key → (label, tensor)
    for sk, mk in key_map.items():
        if "#" in sk:
            base, _tag = sk.split("#", 1)
            if base in source_sd and mk in model_sd:
                resolved[mk] = (sk, source_sd[base])
        elif sk in source_sd and mk in model_sd:
            resolved[mk] = (sk, source_sd[sk])

    # Direct name matches not in key_map
    for mk in model_sd:
        if mk in resolved:
            continue
        if mk in source_sd:
            resolved[mk] = (mk, source_sd[mk])

    for mk, dst in model_sd.items():
        if mk not in resolved:
            report.newly_initialized_parameters += int(dst.numel())
            report.new_keys.append(mk)
            continue
        label, src = resolved[mk]
        _copy_compatible(dst, src, mk, report, allow_partial_block=allow_partial_block, allow_source_seed=allow_source_seed)

    model.load_state_dict(model_sd, strict=False)

    if seed_routed_from_ffn:
        # tiny noise so experts don't start identical
        with torch.no_grad():
            for name, p in model.named_parameters():
                if ".moe.routed." in name and name in report.transformed_keys + report.matched_keys:
                    p.add_(torch.randn_like(p) * 1e-3)

    if zero_init_new:
        touched = zero_init_new_modules(model)
        report.notes.append(f"zero_init_new_modules: {len(touched)}")

    total = sum(p.numel() for p in model.parameters())
    # coverage excludes unsafe partial blocks
    safe = report.matched_parameters + report.transformed_parameters
    report.coverage_percent = min(100.0, 100.0 * safe / max(total, 1))
    report.exact_coverage_percent = min(100.0, 100.0 * report.matched_parameters / max(total, 1))
    report.notes.append(
        f"coverage: exact={report.exact_coverage_percent:.1f}% "
        f"safe(exact+vocab)={report.coverage_percent:.1f}% "
        f"source_seeded={100.0 * report.source_seeded_parameters / max(total,1):.1f}% "
        f"partial_block={100.0 * report.partial_block_parameters / max(total,1):.1f}% "
        f"new={100.0 * report.newly_initialized_parameters / max(total,1):.1f}%"
    )
    return report


def try_load_hf_causal_lm(
    model_id: str,
    *,
    revision: str = "main",
    torch_dtype: Optional[torch.dtype] = torch.float32,
) -> Tuple[Dict[str, Tensor], Dict[str, Any]]:
    try:
        from transformers import AutoConfig, AutoModelForCausalLM  # type: ignore
    except ImportError as e:
        raise ImportError(
            "pip install transformers  (or pass a local state_dict via import_state_dict)"
        ) from e
    cfg = AutoConfig.from_pretrained(model_id, revision=revision)
    m = AutoModelForCausalLM.from_pretrained(
        model_id, revision=revision, torch_dtype=torch_dtype
    )
    sd = {k: v.detach().cpu() for k, v in m.state_dict().items()}
    meta = {
        "model_id": model_id,
        "revision": revision,
        "architectures": getattr(cfg, "architectures", None),
        "hidden_size": getattr(cfg, "hidden_size", None),
        "num_hidden_layers": getattr(cfg, "num_hidden_layers", None),
        "vocab_size": getattr(cfg, "vocab_size", None),
        "num_attention_heads": getattr(cfg, "num_attention_heads", None),
    }
    return sd, meta



def build_qwen_compat_key_map(
    compat_sd: Dict[str, Tensor],
    qwen_sd: Dict[str, Tensor],
) -> Dict[str, str]:
    """
    source_key → QwenCompatModel destination key.
    Exact 1:1 backbone map for retention.
    """
    mapping: Dict[str, str] = {}
    for sk, mk in [
        ("model.embed_tokens.weight", "embed.weight"),
        ("embed_tokens.weight", "embed.weight"),
        ("lm_head.weight", "lm_head.weight"),
        ("model.norm.weight", "norm.weight"),
    ]:
        if sk in qwen_sd and mk in compat_sd:
            mapping[sk] = mk

    def _n_layers(sd: Dict[str, Tensor], pat: str) -> int:
        import re
        ids = set()
        for k in sd:
            m = re.search(pat, k)
            if m:
                ids.add(int(m.group(1)))
        return (max(ids) + 1) if ids else 0

    n = min(_n_layers(qwen_sd, r"layers\.(\d+)\."), _n_layers(compat_sd, r"layers\.(\d+)\."))
    for i in range(n):
        pairs = [
            (f"model.layers.{i}.input_layernorm.weight", f"layers.{i}.input_layernorm.weight"),
            (f"model.layers.{i}.post_attention_layernorm.weight", f"layers.{i}.post_attention_layernorm.weight"),
            (f"model.layers.{i}.self_attn.q_proj.weight", f"layers.{i}.self_attn.q_proj.weight"),
            (f"model.layers.{i}.self_attn.k_proj.weight", f"layers.{i}.self_attn.k_proj.weight"),
            (f"model.layers.{i}.self_attn.v_proj.weight", f"layers.{i}.self_attn.v_proj.weight"),
            (f"model.layers.{i}.self_attn.o_proj.weight", f"layers.{i}.self_attn.out_proj.weight"),
            (f"model.layers.{i}.self_attn.q_proj.bias", f"layers.{i}.self_attn.q_proj.bias"),
            (f"model.layers.{i}.self_attn.k_proj.bias", f"layers.{i}.self_attn.k_proj.bias"),
            (f"model.layers.{i}.self_attn.v_proj.bias", f"layers.{i}.self_attn.v_proj.bias"),
            (f"model.layers.{i}.self_attn.o_proj.bias", f"layers.{i}.self_attn.out_proj.bias"),
            (f"model.layers.{i}.mlp.gate_proj.weight", f"layers.{i}.mlp.gate_proj.weight"),
            (f"model.layers.{i}.mlp.up_proj.weight", f"layers.{i}.mlp.up_proj.weight"),
            (f"model.layers.{i}.mlp.down_proj.weight", f"layers.{i}.mlp.down_proj.weight"),
        ]
        for sk, mk in pairs:
            if sk in qwen_sd and mk in compat_sd:
                mapping[sk] = mk
    return mapping



def source_parameter_coverage(
    source_sd: Dict[str, Tensor],
    key_map: Dict[str, str],
) -> Dict[str, Any]:
    """Fraction of source tensors/params that were mapped."""
    mapped_keys = [sk for sk in key_map if sk in source_sd and "#" not in sk]
    mapped_params = sum(int(source_sd[k].numel()) for k in mapped_keys)
    total_params = sum(int(v.numel()) for v in source_sd.values())
    unmapped = sorted(set(source_sd.keys()) - set(mapped_keys))
    critical_substrings = (
        "self_attn", "mlp.", "layernorm", "embed", "lm_head", "norm.weight",
        "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj",
    )
    unmapped_critical = [
        k for k in unmapped
        if any(s in k for s in critical_substrings)
    ]
    return {
        "source_tensors_total": len(source_sd),
        "source_tensors_mapped": len(mapped_keys),
        "source_param_coverage_percent": 100.0 * mapped_params / max(total_params, 1),
        "unmapped_source_key_count": len(unmapped),
        "unmapped_critical_keys": unmapped_critical,
        "unmapped_critical_count": len(unmapped_critical),
        "unmapped_source_keys_preview": unmapped[:50],
    }

def import_into_qwen_compat(
    model: "QwenCompatModel",
    source_sd: Dict[str, Tensor],
    *,
    source_name: str = "qwen",
    source_revision: str = "unknown",
) -> PretrainedImportReport:
    """Import Qwen weights into QwenCompatModel with exact name map."""
    key_map = build_qwen_compat_key_map(model.state_dict(), source_sd)
    return import_state_dict(
        model,  # type: ignore[arg-type]
        source_sd,
        source_name=source_name,
        source_revision=source_revision,
        key_map=key_map,
        zero_init_new=False,  # no ExFusion extras to zero
        allow_partial_block=False,
        allow_source_seed=False,
    )


def load_pretrained_into_exfusion(
    model: DAPHHybridModelV3,
    *,
    checkpoint: Optional[str] = None,
    hf_model_id: Optional[str] = None,
    hf_revision: str = "main",
    key_map: Optional[Dict[str, str]] = None,
    zero_init_new: bool = True,
) -> PretrainedImportReport:
    if hf_model_id is not None:
        sd, meta = try_load_hf_causal_lm(hf_model_id, revision=hf_revision)
        return import_state_dict(
            model,
            sd,
            source_name=hf_model_id,
            source_revision=str(meta.get("revision", hf_revision)),
            key_map=key_map,
            zero_init_new=zero_init_new,
        )
    if checkpoint is None:
        raise ValueError("Provide checkpoint= or hf_model_id=")
    obj = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if isinstance(obj, dict) and "state_dict" in obj:
        sd = obj["state_dict"]
        name = obj.get("source_model", checkpoint)
        rev = str(obj.get("revision", "local"))
    elif isinstance(obj, dict):
        sd = obj
        name, rev = checkpoint, "local"
    else:
        raise ValueError("Unsupported checkpoint format")
    return import_state_dict(
        model, sd, source_name=str(name), source_revision=rev,
        key_map=key_map, zero_init_new=zero_init_new,
    )


def save_adapted_checkpoint(
    model: nn.Module,
    path: str,
    *,
    report: Optional[PretrainedImportReport] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    cfg = getattr(model, "config", None)
    if cfg is not None and hasattr(cfg, "__dataclass_fields__"):
        model_config = {k: getattr(cfg, k) for k in cfg.__dataclass_fields__}
    else:
        model_config = {
            "architecture": type(model).__name__,
            "vocab_size": getattr(model, "vocab_size", None),
            "hidden_size": getattr(model, "hidden_size", None),
            "num_layers": len(getattr(model, "layers", [])),
            "intermediate_size": getattr(model, "intermediate_size", None),
            "num_heads": getattr(model, "num_heads", None),
            "num_key_value_heads": getattr(model, "num_key_value_heads", None),
            "num_routed_experts": getattr(model, "num_routed_experts", None),
            "top_k": getattr(model, "top_k", None),
            "recurrent_type": getattr(model, "recurrent_type", None),
            "state_size": getattr(model, "state_size", None),
            "latent_size": getattr(model, "latent_size", None),
            "rope_theta": getattr(model, "rope_theta", None),
            "max_position": getattr(model, "max_position", None),
            "rms_eps": getattr(model, "rms_eps", None),
            "attention_bias": getattr(model, "attention_bias", None),
            "attention_output_bias": getattr(model, "attention_output_bias", None),
            "tie_word_embeddings": getattr(model, "tie_word_embeddings", None),
            "depth_fractions": getattr(model, "depth_fractions", None),
            "layer_count_overrides": getattr(model, "layer_count_overrides", None),
            "default_e3_steps": getattr(model, "default_e3_steps", None),
            "use_shallow_continuation": getattr(model, "use_shallow_continuation", False),
            "continuation_bottleneck_size": getattr(model, "continuation_bottleneck_size", None),
            "latent_scale_limit": getattr(model, "latent_scale_limit", None),
            "effort_probe_layer_count": getattr(model, "effort_probe_layer_count", None),
            "effort_controller_hidden_size": getattr(model, "effort_controller_hidden_size", None),
            "enable_effort_controller": getattr(model, "enable_effort_controller", False),
            "effort_probe_fraction": getattr(model, "effort_probe_fraction", None),
            "e3_config": (
                model.e3_config.to_dict() if getattr(model, "e3_config", None) is not None else None
            ),
            "e3_region": (
                model.e3_region.to_dict() if getattr(model, "e3_region", None) is not None else None
            ),
        }
    if getattr(model, "_verified_effort_policy", False):
        controller = getattr(model, "effort_controller", None)
        expected_digest = getattr(model, "_effort_policy_artifact_digest", None)
        if controller is None or not expected_digest:
            raise RuntimeError("Verified effort-policy metadata is incomplete")
        from .policy_trainer import _state_dict_digest
        controller_digest = _state_dict_digest(controller.state_dict())
        if controller_digest != expected_digest:
            raise RuntimeError(
                "Effort-controller weights changed after verified policy installation; "
                "re-verify and reinstall the policy before saving"
            )
        model_config["effort_policy"] = {
            "status": "VERIFIED_FIT",
            "state_dict_digest": controller_digest,
        }
    payload: Dict[str, Any] = {"state_dict": model.state_dict(), "model_config": model_config}
    provenance = getattr(model, "parameter_provenance", None)
    if provenance is not None:
        payload["parameter_provenance"] = provenance.to_dict()
    if report is not None:
        payload["import_report"] = report.to_dict()
    if extra:
        payload.update(extra)
    torch.save(payload, path)

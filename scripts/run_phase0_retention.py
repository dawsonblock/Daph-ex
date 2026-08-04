#!/usr/bin/env python3
"""
Phase 0 retention gates (executable).

Phase 0A — Qwen parity:
  HF Qwen  vs  QwenCompatModel
  Expect near-numerical equivalence (strict threshold).

Phase 0B — exact canonical conversion:
  QwenCompatModel  vs  QwenExFusionModel E2 (zero augmentation scales)
  Requires near-numerical identity.

Usage:
  python scripts/run_phase0_retention.py --synthetic --output runs/phase0
  python scripts/run_phase0_retention.py \\
    --hf-model Qwen/Qwen2.5-0.5B-Instruct \\
    --hf-revision <commit-sha> \\
    --data val.jsonl \\
    --output runs/phase0_qwen \\
    --phase both
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from daph.config import DAPHConfigV3
from daph.qwen_compat import QwenCompatModel
from daph.qwen_exfusion import augment_qwen_compat_model, gate0b_exact_parity
from daph.pretrained import (
    import_into_qwen_compat,
    save_adapted_checkpoint,
)


def _ppl(loss: float) -> float:
    try:
        return float(math.exp(loss))
    except OverflowError:
        return float("inf")


def _file_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def decision_from_delta(delta: float, strong: float, weak: float) -> str:
    if delta <= strong:
        return "PASS"
    if delta <= weak:
        return "CONDITIONAL"
    return "FAIL"



@torch.no_grad()
def logit_parity_metrics(
    source: torch.nn.Module,
    compat: torch.nn.Module,
    batches: list,
    *,
    max_batches: int = 4,
) -> dict:
    """Compare HF source vs compat logits on non-padding positions only."""
    source.eval()
    compat.eval()
    mae_vals, max_vals, cos_vals, kl_vals = [], [], [], []
    top1_agree = 0
    top1_total = 0
    for bi, batch in enumerate(batches):
        if bi >= max_batches:
            break
        if len(batch) != 3:
            continue
        ids, labels, mask = batch
        out_s = source(input_ids=ids, attention_mask=mask)
        logits_s = out_s.logits
        try:
            out_c = compat(ids, attention_mask=mask)
        except TypeError:
            out_c = compat(ids)
        logits_c = out_c if isinstance(out_c, torch.Tensor) else (
            out_c["logits"] if isinstance(out_c, dict) else out_c.logits
        )
        L = min(logits_s.shape[1], logits_c.shape[1], mask.shape[1])
        ls = logits_s[:, :L, :].float()
        lc = logits_c[:, :L, :].float()
        # valid positions: attention_mask == 1
        valid = mask[:, :L].bool()
        if valid.sum() == 0:
            continue
        # gather valid token logits: (N_valid, V)
        flat_s = ls[valid]
        flat_c = lc[valid]
        diff = (flat_s - flat_c).abs()
        mae_vals.append(diff.mean().item())
        max_vals.append(diff.max().item())
        cos_vals.append(F.cosine_similarity(flat_s, flat_c, dim=-1).mean().item())
        top1_agree += int((flat_s.argmax(-1) == flat_c.argmax(-1)).sum().item())
        top1_total += flat_s.shape[0]
        log_pc = F.log_softmax(flat_c, dim=-1)
        ps = F.softmax(flat_s, dim=-1)
        kl_vals.append(F.kl_div(log_pc, ps, reduction="batchmean").item())
    return {
        "logit_mae": sum(mae_vals) / max(len(mae_vals), 1),
        "logit_max_abs": max(max_vals) if max_vals else float("inf"),
        "logit_cosine": sum(cos_vals) / max(len(cos_vals), 1),
        "top1_agreement": top1_agree / max(top1_total, 1),
        "kl_source_compat": sum(kl_vals) / max(len(kl_vals), 1),
        "n_valid_positions": top1_total,
    }

@torch.no_grad()
def eval_lm_loss(
    model: torch.nn.Module,
    batches: list,
    *,
    effort_mode: str = "fixed_2",
    is_hf: bool = False,
) -> Tuple[float, int]:
    """Manual next-token CE; returns (mean_loss, n_tokens)."""
    model.eval()
    total_loss, ntok = 0.0, 0
    for batch in batches:
        if len(batch) == 3:
            ids, labels, mask = batch
        else:
            x, y = batch
            if is_hf:
                out = model(input_ids=x)
                logits = out.logits if hasattr(out, "logits") else out
            else:
                try:
                    out = model(x, effort_mode=effort_mode)
                except TypeError:
                    out = model(x)
                logits = out["logits"] if isinstance(out, dict) else (
                    out.logits if hasattr(out, "logits") else out
                )
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1), reduction="sum")
            total_loss += float(loss.item())
            ntok += int(y.numel())
            continue

        if is_hf:
            out = model(input_ids=ids, attention_mask=mask)
            logits = out.logits
        else:
            try:
                out = model(ids, attention_mask=mask, effort_mode=effort_mode)
            except TypeError:
                out = model(ids, attention_mask=mask)
            logits = out["logits"] if isinstance(out, dict) else (
                out.logits if hasattr(out, "logits") else out
            )
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()
        loss = F.cross_entropy(
            shift_logits.reshape(-1, shift_logits.size(-1)),
            shift_labels.reshape(-1),
            ignore_index=-100,
            reduction="sum",
        )
        valid = int((shift_labels != -100).sum().item())
        total_loss += float(loss.item())
        ntok += max(valid, 0)
    return total_loss / max(ntok, 1), ntok


def make_synthetic_batches(vocab: int, n: int = 4, B: int = 2, L: int = 32, device="cpu"):
    batches = []
    for _ in range(n):
        ids = torch.randint(0, vocab, (B, L), device=device)
        mask = torch.ones_like(ids)
        labels = ids.clone()
        batches.append((ids, labels, mask))
    return batches


def write_config_artifact(output: Path, payload: Dict[str, Any]) -> None:
    (output / "phase0_config.json").write_text(json.dumps(payload, indent=2, default=str))


def write_dataset_manifest(output: Path, payload: Dict[str, Any]) -> None:
    (output / "phase0_dataset_manifest.json").write_text(json.dumps(payload, indent=2, default=str))


def env_meta() -> Dict[str, Any]:
    meta = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
    }
    try:
        import transformers
        meta["transformers"] = transformers.__version__
    except ImportError:
        meta["transformers"] = None
    return meta


def build_compat_from_hf(cfg_hf) -> QwenCompatModel:
    H = int(cfg_hf.hidden_size)
    L = int(cfg_hf.num_hidden_layers)
    n_q = int(cfg_hf.num_attention_heads)
    n_kv = int(getattr(cfg_hf, "num_key_value_heads", n_q) or n_q)
    V = int(cfg_hf.vocab_size)
    inter = int(getattr(cfg_hf, "intermediate_size", H * 2))
    rope_params = getattr(cfg_hf, "rope_parameters", None) or {}
    rope = float(getattr(cfg_hf, "rope_theta", None) or rope_params.get("rope_theta", 10000.0))
    eps = float(getattr(cfg_hf, "rms_norm_eps", 1e-6) or 1e-6)
    max_pos = int(getattr(cfg_hf, "max_position_embeddings", 8192) or 8192)
    tie = bool(getattr(cfg_hf, "tie_word_embeddings", True))
    model_type = str(getattr(cfg_hf, "model_type", ""))
    # Qwen2/Qwen2.5 use Q/K/V bias and a bias-free O projection.
    attn_bias = True if model_type == "qwen2" else bool(getattr(cfg_hf, "attention_bias", False))
    attn_out_bias = False if model_type == "qwen2" else attn_bias
    unsupported = []
    sw = getattr(cfg_hf, "sliding_window", None)
    if sw not in (None, False, 0):
        unsupported.append(f"sliding_window={sw}")
    rs = getattr(cfg_hf, "rope_scaling", None)
    if rs not in (None, {}, False) and rs.get("rope_type", "default") != "default":
        unsupported.append(f"rope_scaling={rs}")
    if unsupported:
        raise RuntimeError(
            f"Phase 0A hard-reject unsupported source features: {unsupported}. "
            "Strict parity cannot be claimed."
        )
    return QwenCompatModel(
        vocab_size=V, hidden_size=H, num_layers=L, num_heads=n_q,
        num_key_value_heads=n_kv, intermediate_size=inter,
        rope_theta=rope, max_position=max_pos, rms_eps=eps,
        tie_word_embeddings=tie, attention_bias=attn_bias,
        attention_output_bias=attn_out_bias,
    )


def _model_digest(model: torch.nn.Module) -> str:
    h = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        h.update(name.encode())
        h.update(value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def _gate0b_report(compat: QwenCompatModel, exfusion, batches: list) -> Dict[str, Any]:
    ids, _, mask = batches[0]
    parity = gate0b_exact_parity(compat, exfusion, ids, mask)
    scales = {n: float(p.detach()) for n, p in exfusion.named_parameters() if n.endswith("_scale")}
    provenance = exfusion.parameter_provenance.to_dict() if exfusion.parameter_provenance else {}
    return {
        "source_compat_digest": _model_digest(compat),
        "exfusion_digest": _model_digest(exfusion),
        "config_digest": hashlib.sha256(json.dumps({"layers": len(exfusion.layers), "hidden": exfusion.hidden_size}, sort_keys=True).encode()).hexdigest(),
        "parameter_counts": {
            "compat": sum(p.numel() for p in compat.parameters()),
            "exfusion": sum(p.numel() for p in exfusion.parameters()),
            "new_augmentation": sum(p.numel() for n, p in exfusion.named_parameters() if n in set(provenance.get("new_parameter_names", []))),
        },
        "parity_metrics": parity,
        "augmentation_scales": scales,
        "parameter_provenance": provenance,
        "result": parity["decision"],
    }


def run_synthetic(
    output: Path, *, shallow_continuation: bool = False,
    num_routed_experts: int = 2, top_k: int = 1,
    latent_size: Optional[int] = None, save_checkpoint: bool = True,
) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    device = "cpu"
    H, L, V, I, n_q, n_kv = 64, 2, 128, 128, 4, 2
    compat = QwenCompatModel(
        V, H, L, n_q, n_kv, I, rope_theta=10000.0, rms_eps=1e-6, tie_word_embeddings=True
    ).to(device)
    src = {
        "model.embed_tokens.weight": torch.randn(V, H),
        "lm_head.weight": torch.randn(V, H),
        "model.norm.weight": torch.ones(H),
    }
    for i in range(L):
        src[f"model.layers.{i}.input_layernorm.weight"] = torch.ones(H)
        src[f"model.layers.{i}.post_attention_layernorm.weight"] = torch.ones(H)
        src[f"model.layers.{i}.self_attn.q_proj.weight"] = torch.randn(H, H)
        src[f"model.layers.{i}.self_attn.k_proj.weight"] = torch.randn(n_kv * (H // n_q), H)
        src[f"model.layers.{i}.self_attn.v_proj.weight"] = torch.randn(n_kv * (H // n_q), H)
        src[f"model.layers.{i}.self_attn.o_proj.weight"] = torch.randn(H, H)
        src[f"model.layers.{i}.mlp.gate_proj.weight"] = torch.randn(I, H)
        src[f"model.layers.{i}.mlp.up_proj.weight"] = torch.randn(I, H)
        src[f"model.layers.{i}.mlp.down_proj.weight"] = torch.randn(H, I)

    report = import_into_qwen_compat(compat, src, source_name="synthetic-qwen")
    report.save(str(output / "phase0_import_report.json"))
    batches = make_synthetic_batches(V, n=4, device=device)
    loss_c, ntok = eval_lm_loss(compat, batches)

    exfusion = augment_qwen_compat_model(
        compat, num_routed_experts=num_routed_experts, top_k=top_k,
        latent_size=latent_size,
        use_shallow_continuation=shallow_continuation,
    ).to(device)
    loss_h, _ = eval_lm_loss(exfusion, batches, effort_mode="fixed_2")
    gate_report = _gate0b_report(compat, exfusion, batches)
    (output / "phase0b_gate_report.json").write_text(json.dumps(gate_report, indent=2))
    if save_checkpoint:
        save_adapted_checkpoint(exfusion, str(output / "qwen_exfusion_gate0b.pt"), extra={"gate0b_report": gate_report})

    metrics = {
        "source_mode": "synthetic",
        "phase0a": {
            "compat_loss": loss_c,
            "compat_ppl": _ppl(loss_c),
            "exact_coverage_percent": report.exact_coverage_percent,
            "matched_keys": len(report.matched_keys),
            "decision": "SYNTHETIC_ONLY",
        },
        "phase0b": {
            "hybrid_e2_loss": loss_h,
            "hybrid_e2_ppl": _ppl(loss_h),
            "decision": gate_report["result"],
            "parity_metrics": gate_report["parity_metrics"],
        },
        "n_tokens": ntok,
    }
    (output / "phase0_metrics.json").write_text(json.dumps(metrics, indent=2))
    write_config_artifact(output, {
        "mode": "synthetic",
        "dims": {"H": H, "L": L, "V": V, "I": I, "n_q": n_q, "n_kv": n_kv},
        "environment": env_meta(),
        "thresholds": {
            "phase0a_relative_ce": 0.01,
            "phase0b_delta_ppl_strong": 0.15,
            "phase0b_delta_ppl_weak": 0.40,
        },
    })
    write_dataset_manifest(output, {
        "type": "synthetic_random_tokens",
        "n_batches": 4,
        "batch_size": 2,
        "seq_len": 32,
        "n_tokens_scored": ntok,
    })
    decision = f"""# Phase 0 Decision (SYNTHETIC)

## Phase 0A — QwenCompatModel import
- exact_coverage: {report.exact_coverage_percent:.2f}%
- matched_keys: {len(report.matched_keys)}
- compat_loss: {loss_c:.4f}
- **Decision: SYNTHETIC_ONLY**

## Phase 0B — QwenExFusion E2 exact conversion
- exfusion_e2_loss: {loss_h:.4f}
- **Decision: {gate_report['result']}**

Use --hf-model + --hf-revision for real gates.
"""
    (output / "phase0_decision.md").write_text(decision)
    print(decision)
    return metrics


def run_hf(
    model_id: str,
    revision: str,
    data_path: Optional[str],
    output: Path,
    *,
    phase: str,
    strong_0a: float,
    strong_0b: float,
    weak_0b: float,
    max_batches: int,
    seq_len: int,
    shallow_continuation: bool = False,
    num_routed_experts: int = 2,
    top_k: int = 1,
    latent_size: Optional[int] = None,
    save_checkpoint: bool = True,
) -> dict:
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    output.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tok = AutoTokenizer.from_pretrained(model_id, revision=revision, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    cfg_hf = AutoConfig.from_pretrained(model_id, revision=revision, trust_remote_code=True)
    source = AutoModelForCausalLM.from_pretrained(
        model_id, revision=revision, trust_remote_code=True, torch_dtype=torch.float32
    ).to(device)
    source.eval()

    # Data
    data_manifest: Dict[str, Any] = {"path": data_path, "tokenizer": model_id, "revision": revision}
    if data_path:
        data_manifest["sha256"] = _file_sha256(data_path)
        texts = []
        with open(data_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                texts.append(obj.get("text") or obj.get("input") or "")
        data_manifest["n_records"] = len(texts)
        batches = []
        for i in range(0, min(len(texts), max_batches * 2), 2):
            chunk = texts[i : i + 2]
            if not chunk:
                break
            enc = tok(chunk, padding="max_length", truncation=True, max_length=seq_len, return_tensors="pt")
            ids = enc["input_ids"].to(device)
            mask = enc["attention_mask"].to(device)
            labels = ids.clone()
            labels[mask == 0] = -100
            batches.append((ids, labels, mask))
    else:
        V = int(cfg_hf.vocab_size)
        batches = make_synthetic_batches(V, n=max_batches, B=2, L=seq_len, device=device)
        data_manifest["type"] = "synthetic_vocab_tokens"

    loss_src, ntok = eval_lm_loss(source, batches, is_hf=True)
    ppl_src = _ppl(loss_src)
    data_manifest["n_tokens_scored"] = ntok
    data_manifest["seq_len"] = seq_len
    data_manifest["max_batches"] = max_batches
    write_dataset_manifest(output, data_manifest)

    metrics: Dict[str, Any] = {
        "source_mode": "hf",
        "hf_model": model_id,
        "hf_revision": revision,
        "loss_source": loss_src,
        "PPL_source": ppl_src,
        "n_tokens": ntok,
    }

    # ---- Phase 0A ----
    if phase in ("0a", "both", "a"):
        compat = build_compat_from_hf(cfg_hf).to(device)
        sd = {k: v.detach().cpu() for k, v in source.state_dict().items()}
        report = import_into_qwen_compat(compat, sd, source_name=model_id, source_revision=revision)
        report.save(str(output / "phase0a_import_report.json"))
        loss_c, _ = eval_lm_loss(compat, batches)
        ppl_c = _ppl(loss_c)
        rel_ce = abs(loss_c - loss_src) / max(loss_src, 1e-8)
        delta_ppl = (ppl_c - ppl_src) / max(ppl_src, 1e-8)
        parity = logit_parity_metrics(source, compat, batches)
        from daph.pretrained import build_qwen_compat_key_map, source_parameter_coverage
        km = build_qwen_compat_key_map(compat.state_dict(), sd)
        src_cov = source_parameter_coverage(sd, km)
        unmapped_critical = src_cov["unmapped_critical_keys"]
        # Functional retention gate (not float-noise exactness)
        functional_pass = (
            rel_ce <= strong_0a
            and parity["logit_mae"] <= 0.05
            and parity["logit_max_abs"] <= 2.0
            and parity["logit_cosine"] >= 0.98
            and parity["top1_agreement"] >= 0.95
            and parity["kl_source_compat"] <= 0.05
            and src_cov["source_param_coverage_percent"] >= 95.0
            and src_cov["unmapped_critical_count"] == 0
            and report.partial_block_parameters == 0
        )
        # Near-exact parity (stricter)
        exact_pass = (
            functional_pass
            and parity["logit_mae"] <= 1e-3
            and parity["logit_max_abs"] <= 0.05
            and parity["top1_agreement"] >= 0.999
            and parity["kl_source_compat"] <= 1e-4
            and src_cov["source_param_coverage_percent"] >= 99.5
        )
        if exact_pass:
            dec_a = "PASS_EXACT"
        elif functional_pass:
            dec_a = "PASS"
        elif rel_ce <= strong_0a * 5 and parity["top1_agreement"] >= 0.80:
            dec_a = "CONDITIONAL"
        else:
            dec_a = "FAIL"
        metrics["phase0a"] = {
            "loss_compat": loss_c,
            "PPL_compat": ppl_c,
            "rel_ce": rel_ce,
            "delta_ppl": delta_ppl,
            "exact_coverage_percent": report.exact_coverage_percent,
            "matched_keys": len(report.matched_keys),
            "skipped_keys": len(report.skipped_keys),
            "source_coverage": src_cov,
            "logit_parity": parity,
            "unmapped_critical_count": src_cov["unmapped_critical_count"],
            "decision": dec_a,
            "threshold_rel_ce": strong_0a,
        }
    else:
        compat = None
        report = None
        dec_a = "SKIPPED"

    # ---- Phase 0B ----
    if phase in ("0b", "both", "b"):
        if compat is None:
            compat = build_compat_from_hf(cfg_hf).to(device)
            sd = {k: v.detach().cpu() for k, v in source.state_dict().items()}
            report = import_into_qwen_compat(compat, sd, source_name=model_id, source_revision=revision)
        exfusion = augment_qwen_compat_model(
            compat, use_shallow_continuation=shallow_continuation,
            num_routed_experts=num_routed_experts, top_k=top_k,
            latent_size=latent_size,
        ).to(device)
        loss_h, _ = eval_lm_loss(exfusion, batches, effort_mode="fixed_2")
        ppl_h = _ppl(loss_h)
        base_ppl = _ppl(eval_lm_loss(compat, batches)[0])
        delta_ppl_b = (ppl_h - base_ppl) / max(base_ppl, 1e-8)
        gate_report = _gate0b_report(compat, exfusion, batches)
        (output / "phase0b_gate_report.json").write_text(json.dumps(gate_report, indent=2))
        if save_checkpoint:
            save_adapted_checkpoint(exfusion, str(output / "qwen_exfusion_gate0b.pt"), extra={"gate0b_report": gate_report})
        dec_b = gate_report["result"]
        metrics["phase0b"] = {
            "loss_exfusion_e2": loss_h,
            "PPL_exfusion_e2": ppl_h,
            "delta_ppl_vs_base": delta_ppl_b,
            "base": "compat",
            "parity_metrics": gate_report["parity_metrics"],
            "decision": dec_b,
        }
    else:
        dec_b = "SKIPPED"

    write_config_artifact(output, {
        "hf_model": model_id,
        "hf_revision": revision,
        "phase": phase,
        "seq_len": seq_len,
        "max_batches": max_batches,
        "thresholds": {
            "phase0a_rel_ce": strong_0a,
            "phase0b_delta_ppl_strong": strong_0b,
            "phase0b_delta_ppl_weak": weak_0b,
        },
        "source_config": {
            "hidden_size": getattr(cfg_hf, "hidden_size", None),
            "num_hidden_layers": getattr(cfg_hf, "num_hidden_layers", None),
            "num_attention_heads": getattr(cfg_hf, "num_attention_heads", None),
            "num_key_value_heads": getattr(cfg_hf, "num_key_value_heads", None),
            "intermediate_size": getattr(cfg_hf, "intermediate_size", None),
            "vocab_size": getattr(cfg_hf, "vocab_size", None),
            "rope_theta": getattr(cfg_hf, "rope_theta", None),
            "rms_norm_eps": getattr(cfg_hf, "rms_norm_eps", None),
        },
        "tokenizer_pad_token_id": tok.pad_token_id,
        "environment": env_meta(),
    })
    (output / "phase0_metrics.json").write_text(json.dumps(metrics, indent=2))

    lines = [
        f"# Phase 0 Decision",
        f"",
        f"- Source: `{model_id}` @ `{revision}`",
        f"- loss_source: {loss_src:.6f}  PPL_source: {ppl_src:.4f}",
        f"- tokens scored: {ntok}",
        f"",
    ]
    if "phase0a" in metrics:
        a = metrics["phase0a"]
        lines += [
            f"## Phase 0A — HF Qwen vs QwenCompatModel",
            f"- loss_compat: {a['loss_compat']:.6f}  PPL: {a['PPL_compat']:.4f}",
            f"- rel_ce: {a['rel_ce']:.6f}  (threshold ≤ {strong_0a})",
            f"- exact_coverage: {a['exact_coverage_percent']:.2f}%",
            f"- **Decision: {a['decision']}**",
            f"",
        ]
    if "phase0b" in metrics:
        b = metrics["phase0b"]
        lines += [
            f"## Phase 0B — QwenExFusion E2 exact parity",
            f"- loss_exfusion_e2: {b['loss_exfusion_e2']:.6f}  PPL: {b['PPL_exfusion_e2']:.4f}",
            f"- ΔPPL vs {b['base']}: {b['delta_ppl_vs_base']:.4f}",
            f"- **Decision: {b['decision']}**",
            f"",
        ]
    lines.append("Phase 0A must PASS (near-exact) before trusting Phase 0B adaptation.")
    decision = "\n".join(lines) + "\n"
    (output / "phase0_decision.md").write_text(decision)
    print(decision)
    return metrics


def main():
    ap = argparse.ArgumentParser(description="Phase 0A/0B retention gates")
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--hf-model", type=str, default="")
    ap.add_argument("--hf-revision", type=str, default="")
    ap.add_argument("--data", type=str, default="")
    ap.add_argument("--output", type=str, default="runs/phase0")
    ap.add_argument("--phase", type=str, default="both", choices=["0a", "0b", "both", "a", "b"])
    ap.add_argument("--strong-0a", type=float, default=0.01, help="Phase 0A max relative CE")
    ap.add_argument("--strong-0b", type=float, default=0.15, help="Phase 0B strong ΔPPL")
    ap.add_argument("--weak-0b", type=float, default=0.40, help="Phase 0B conditional ΔPPL")
    ap.add_argument("--max-batches", type=int, default=8)
    ap.add_argument("--seq-len", type=int, default=64)
    ap.add_argument("--shallow-continuation", action="store_true", help="include zero-residual E0/E1 continuation modules for Stage 1")
    ap.add_argument("--num-routed-experts", type=int, default=2)
    ap.add_argument("--top-k", type=int, default=1)
    ap.add_argument("--latent-size", type=int, default=0, help="0 selects the model default")
    ap.add_argument("--no-save-checkpoint", action="store_true", help="write reports without the potentially large ExFusion checkpoint")
    args = ap.parse_args()
    out = Path(args.output)

    if args.synthetic or not args.hf_model:
        run_synthetic(
            out, shallow_continuation=args.shallow_continuation,
            num_routed_experts=args.num_routed_experts, top_k=args.top_k,
            latent_size=args.latent_size or None,
            save_checkpoint=not args.no_save_checkpoint,
        )
        return
    if not args.hf_revision or args.hf_revision == "main":
        print("ERROR: --hf-revision must be an immutable commit SHA (not 'main').", file=sys.stderr)
        sys.exit(2)
    run_hf(
        args.hf_model, args.hf_revision, args.data or None, out,
        phase=args.phase, strong_0a=args.strong_0a, strong_0b=args.strong_0b,
        weak_0b=args.weak_0b, max_batches=args.max_batches, seq_len=args.seq_len,
        shallow_continuation=args.shallow_continuation,
        num_routed_experts=args.num_routed_experts, top_k=args.top_k,
        latent_size=args.latent_size or None,
        save_checkpoint=not args.no_save_checkpoint,
    )


if __name__ == "__main__":
    main()

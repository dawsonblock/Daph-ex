#!/usr/bin/env python3
"""Memory-bounded real QwenExFusion adaptation smoke experiment."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import platform
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from daph.pretrained import import_into_qwen_compat
from daph.qwen_exfusion import (
    augment_qwen_compat_model,
    gate0b_exact_parity,
    prepare_exfusion_for_training,
)
from daph.train_real import (
    RealTrainConfig,
    TextBatcher,
    TrainingStageConfig,
    eval_per_effort,
    load_jsonl_texts,
    train_adapt,
)
from scripts.run_phase0_retention import build_compat_from_hf, logit_parity_metrics


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


@torch.no_grad()
def _latency(model, ids, mask, repeats: int = 2) -> Dict[str, float]:
    result: Dict[str, float] = {}
    model.eval()
    for effort in range(4):
        mode = f"fixed_{effort}"
        model(ids, attention_mask=mask, effort_mode=mode)
        _synchronize(ids.device)
        samples = []
        for _ in range(repeats):
            start = time.perf_counter()
            model(ids, attention_mask=mask, effort_mode=mode)
            _synchronize(ids.device)
            samples.append((time.perf_counter() - start) * 1000.0)
        result[mode] = sum(samples) / len(samples)
    return result


def _quality_report(before: Dict[str, Any], after: Dict[str, Any]) -> Dict[str, Any]:
    report: Dict[str, Any] = {}
    for effort in range(4):
        mode = f"fixed_{effort}"
        b, a = before[mode], after[mode]
        report[mode] = {
            "ce_before": b["ce"],
            "ce_after": a["ce"],
            "ce_delta": a["ce"] - b["ce"],
            "perplexity_before": b["perplexity"],
            "perplexity_after": a["perplexity"],
            "quality_before": float(torch.exp(torch.tensor(-b["ce"])).item()),
            "quality_after": float(torch.exp(torch.tensor(-a["ce"])).item()),
            "normalized_compute": a["estimated_normalized_compute"],
            "executed_layers": a["average_executed_layers"],
        }
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--revision", required=True)
    parser.add_argument("--train", required=True)
    parser.add_argument("--validation", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--steps", type=int, default=12)
    parser.add_argument("--seq-len", type=int, default=32)
    parser.add_argument("--eval-batches", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--latent-size", type=int, default=64)
    parser.add_argument("--epsilon", type=float, default=1e-5)
    parser.add_argument("--lr-new", type=float, default=1e-5)
    parser.add_argument("--lr-scales", type=float, default=1e-4)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    if args.device == "auto":
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.revision)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    validation_texts = load_jsonl_texts(args.validation)
    enc = tokenizer(
        validation_texts[:2], padding="max_length", truncation=True,
        max_length=args.seq_len, return_tensors="pt",
    )
    parity_ids = enc["input_ids"]
    parity_mask = enc["attention_mask"]
    parity_labels = parity_ids.clone()
    parity_labels[parity_mask == 0] = -100
    parity_batches = [(parity_ids, parity_labels, parity_mask)]

    cfg_hf = AutoConfig.from_pretrained(args.model, revision=args.revision)
    source = AutoModelForCausalLM.from_pretrained(
        args.model, revision=args.revision, dtype=torch.float32
    ).eval()
    compat = build_compat_from_hf(cfg_hf)
    source_state = {name: value.detach().cpu() for name, value in source.state_dict().items()}
    import_report = import_into_qwen_compat(
        compat, source_state, source_name=args.model, source_revision=args.revision
    )
    phase0a = logit_parity_metrics(source, compat, parity_batches, max_batches=1)
    del source_state, source
    gc.collect()

    model = augment_qwen_compat_model(
        compat,
        num_routed_experts=1,
        top_k=1,
        latent_size=args.latent_size,
        use_shallow_continuation=True,
        default_e3_steps=1,
    )
    phase0b = gate0b_exact_parity(compat, model, parity_ids, parity_mask)
    if phase0b["decision"] != "PASS_EXACT":
        raise RuntimeError(f"Gate 0B failed: {phase0b}")
    training_init = prepare_exfusion_for_training(
        model, gate0b_passed=True, epsilon=args.epsilon
    )
    del compat
    gc.collect()
    model = model.to(device)

    train_texts = load_jsonl_texts(args.train)
    before_batcher = TextBatcher(
        validation_texts, tokenizer=tokenizer, seq_len=args.seq_len,
        batch_size=1, device=device, seed=args.seed + 1,
    )
    before = eval_per_effort(
        model, before_batcher, n_batches=args.eval_batches, detailed=True
    )

    stage = TrainingStageConfig(
        name="stage1_real_smoke",
        steps=args.steps,
        train_parameter_groups=("continuation", "augmentation", "scales"),
        freeze_parameter_groups=("imported",),
        lr_new=args.lr_new,
        lr_scales=args.lr_scales,
        effort_sampling=(0.40, 0.40, 0.0, 0.20),
        distill_e0=True,
        distill_e1=True,
        train_e3=True,
        teacher_mode="fixed_2",
    )
    train_cfg = RealTrainConfig(
        steps=args.steps,
        batch_size=1,
        seq_len=args.seq_len,
        grad_accum=1,
        warmup_steps=max(1, min(3, args.steps)),
        log_every=1,
        eval_every=max(args.steps + 1, 100),
        seed=args.seed,
        device=str(device),
        effort_mode="sample",
        effort_schedule=("fixed_0", "fixed_1", "fixed_3", "fixed_0", "fixed_1"),
        data_path=args.train,
        tokenizer_name=args.model,
        tokenizer_revision=args.revision,
        output_dir=str(output / "trainer"),
        distillation_temperature=2.0,
        beta_e0=1.0,
        beta_e1=1.0,
        stages=(stage,),
        save_periodic_checkpoints=False,
        save_final_checkpoint=False,
        save_model_artifact=False,
    )
    train_result = train_adapt(model, train_cfg)

    after_batcher = TextBatcher(
        validation_texts, tokenizer=tokenizer, seq_len=args.seq_len,
        batch_size=1, device=device, seed=args.seed + 1,
    )
    after = eval_per_effort(
        model, after_batcher, n_batches=args.eval_batches, detailed=True
    )
    bench_enc = tokenizer(
        validation_texts[:1], padding="max_length", truncation=True,
        max_length=args.seq_len, return_tensors="pt",
    )
    bench_ids = bench_enc["input_ids"].to(device)
    bench_mask = bench_enc["attention_mask"].to(device)
    latency = _latency(model, bench_ids, bench_mask)
    quality = _quality_report(before, after)
    costs = [quality[f"fixed_{e}"]["normalized_compute"] for e in range(4)]
    compute_ordered = all(costs[i] < costs[i + 1] for i in range(3))
    e2_retained = abs(quality["fixed_2"]["ce_delta"]) <= 1e-7

    report = {
        "experiment": "qwen2.5-0.5b-wikitext2-memory-bounded-smoke",
        "model": {"id": args.model, "revision": args.revision},
        "dataset": {
            "id": "Salesforce/wikitext",
            "revision": "b08601e04326c79dfdd32d625aee71d232d685c3",
            "config": "wikitext-2-raw-v1",
            "train_records": len(train_texts),
            "validation_records": len(validation_texts),
            "train_sha256": _sha256(args.train),
            "validation_sha256": _sha256(args.validation),
        },
        "environment": {
            "platform": platform.platform(),
            "torch": torch.__version__,
            "device": str(device),
        },
        "phase0a": {
            "source_parameter_coverage_percent": import_report.exact_coverage_percent,
            "parity": phase0a,
        },
        "phase0b": phase0b,
        "training_init": asdict(training_init),
        "training": {
            "config": asdict(train_cfg),
            "effort_hist": train_result["effort_hist"],
            "examples_by_effort": train_result["examples_by_effort"],
            "tokens_by_effort": train_result["tokens_by_effort"],
            "optimizer_steps_completed": train_result["optimizer_steps_completed"],
            "wall_time_s": train_result["wall_time_s"],
            "history_tail": train_result["history_tail"],
        },
        "per_effort": quality,
        "latency_ms": latency,
        "qualification": {
            "physical_compute_ordered": compute_ordered,
            "e2_retained": e2_retained,
            "e0_ce_regret_vs_e2": quality["fixed_0"]["ce_after"] - quality["fixed_2"]["ce_after"],
            "e1_ce_regret_vs_e2": quality["fixed_1"]["ce_after"] - quality["fixed_2"]["ce_after"],
            "e3_ce_delta_vs_e2": quality["fixed_3"]["ce_after"] - quality["fixed_2"]["ce_after"],
            "e3_improved_on_validation": quality["fixed_3"]["ce_after"] < quality["fixed_2"]["ce_after"],
        },
        "limitations": [
            "Small smoke subset and very short training schedule; not a capability claim.",
            "Validation CE is based on a deterministic subset, not the full WikiText-2 validation split.",
            "Latency is local MPS wall time and is not portable across hardware.",
            "No adaptive policy is trained unless effort modes and the oracle later qualify.",
        ],
    }
    (output / "experiment_report.json").write_text(json.dumps(report, indent=2, default=str))
    print(json.dumps(report["qualification"], indent=2))


if __name__ == "__main__":
    main()

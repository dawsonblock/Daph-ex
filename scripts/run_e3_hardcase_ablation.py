#!/usr/bin/env python3
"""Gated E3 refinement ablation on verified hard tasks.

E2 remains frozen. Only final latent refinement and its residual scale train
after Gate 0B. Variants select on a selection split; held-out evaluation runs
once for the selected number of refinement steps.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import platform
import re
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Sequence

import torch
import torch.nn.functional as F
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from daph.e3_metrics import e3_pair_metrics
from daph.pretrained import import_into_qwen_compat
from daph.qwen_exfusion import augment_qwen_compat_model, gate0b_exact_parity, prepare_exfusion_for_training
from daph.train_real import RealTrainConfig, TrainingStageConfig, train_adapt
from scripts.run_phase0_retention import build_compat_from_hf


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_tasks(path: Path) -> List[Dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if not rows:
        raise ValueError(f"No tasks in {path}")
    for index, row in enumerate(rows):
        if not row.get("prompt") or row.get("expected") is None:
            raise ValueError(f"Task {index} in {path} must contain prompt and expected")
        row.setdefault("task_id", f"{path.stem}-{index}")
        row.setdefault("difficulty_bucket", "hard")
        row.setdefault("task_family", "unspecified")
    return rows


def _write_training_text(tasks: Sequence[Dict[str, Any]], path: Path) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for task in tasks:
            handle.write(json.dumps({"text": f"{task['prompt']}{task['expected']}"}) + "\n")


def _numeric_correct(text: str, expected: Any) -> bool:
    values = re.findall(r"[-+]?\d+(?:\.\d+)?", text)
    if not values:
        return False
    try:
        return float(values[-1]) == float(expected)
    except (TypeError, ValueError):
        return text.strip() == str(expected).strip()


@torch.no_grad()
def _e2_logits(model: Any, tokenizer: Any, prompt: str, device: torch.device) -> torch.Tensor:
    ids = tokenizer(prompt, add_special_tokens=False, return_tensors="pt")["input_ids"].to(device)
    return model(ids, attention_mask=torch.ones_like(ids), effort_mode="fixed_2").detach().cpu()


@torch.no_grad()
def _task_eval(model: Any, tokenizer: Any, tasks: Sequence[Dict[str, Any]], *, device: torch.device, max_new_tokens: int) -> Dict[str, Any]:
    pairs: List[Dict[str, Any]] = []
    e2_losses, e3_losses, deltas, costs = [], [], [], []
    model.eval()
    for task in tasks:
        prompt, expected = str(task["prompt"]), str(task["expected"])
        prompt_ids = tokenizer(prompt, add_special_tokens=False, return_tensors="pt")["input_ids"].to(device)
        answer_ids = tokenizer(expected, add_special_tokens=False, return_tensors="pt")["input_ids"].to(device)
        ids, mask = torch.cat((prompt_ids, answer_ids), dim=1), None
        mask = torch.ones_like(ids)
        labels = torch.full_like(ids, -100)
        labels[:, prompt_ids.size(1):] = answer_ids
        e2 = model(ids, attention_mask=mask, effort_mode="fixed_2", return_compute_receipt=True, return_hidden_state=True)
        e3 = model(ids, attention_mask=mask, effort_mode="fixed_3", return_compute_receipt=True, return_hidden_state=True)
        for out, store in ((e2, e2_losses), (e3, e3_losses)):
            logits, target = out["logits"][:, :-1].contiguous(), labels[:, 1:].contiguous()
            store.append(float(F.cross_entropy(logits.reshape(-1, logits.size(-1)), target.reshape(-1), ignore_index=-100).item()))
        deltas.append(float((e3["hidden_state"] - e2["hidden_state"]).norm(dim=-1).mean().item()))
        costs.append(float(e3["compute_stats"]["normalized_compute_cost"] - e2["compute_stats"]["normalized_compute_cost"]))
        generated, correct = {}, {}
        for mode in ("fixed_2", "fixed_3"):
            generated_out = model.generate(prompt_ids, attention_mask=torch.ones_like(prompt_ids), effort_mode=mode, max_new_tokens=max_new_tokens)
            completion = tokenizer.decode(generated_out["sequences"][0, prompt_ids.size(1):], skip_special_tokens=True)
            generated[mode], correct[mode] = completion, _numeric_correct(completion, task["expected"])
        pairs.append({
            "task_id": task["task_id"], "task_family": task["task_family"], "difficulty_bucket": task["difficulty_bucket"],
            "e2_correct": correct["fixed_2"], "e3_correct": correct["fixed_3"],
            "e2_completion": generated["fixed_2"], "e3_completion": generated["fixed_3"],
        })
    report = e3_pair_metrics(pairs)
    report.update({
        "e2_completion_ce": sum(e2_losses) / len(e2_losses), "e3_completion_ce": sum(e3_losses) / len(e3_losses),
        "e3_ce_delta_vs_e2": (sum(e3_losses) - sum(e2_losses)) / len(e2_losses),
        "mean_hidden_delta_l2": sum(deltas) / len(deltas), "e3_compute_overhead": sum(costs) / len(costs),
        "task_outcomes": pairs,
    })
    return report


def _parse_counts(value: str) -> List[int]:
    counts = sorted(set(int(part) for part in value.split(",") if part.strip()))
    if not counts or any(count < 1 for count in counts):
        raise ValueError("--latent-step-counts must be comma-separated positive integers")
    return counts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--revision", required=True)
    parser.add_argument("--hard-train", required=True)
    parser.add_argument("--selection", required=True)
    parser.add_argument("--test", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--latent-step-counts", default="1,2,4")
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--e3-scale", type=float, default=1e-3)
    parser.add_argument("--lr-refinement", type=float, default=1e-4)
    parser.add_argument("--lr-scale", type=float, default=1e-5)
    parser.add_argument("--seq-len", type=int, default=96)
    parser.add_argument("--max-new-tokens", type=int, default=12)
    parser.add_argument("--latent-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=20260803)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    counts, output = _parse_counts(args.latent_step_counts), Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    train_tasks, selection_tasks, test_tasks = _load_tasks(Path(args.hard_train)), _load_tasks(Path(args.selection)), _load_tasks(Path(args.test))
    train_jsonl = output / "hard_train_causal_lm.jsonl"
    _write_training_text(train_tasks, train_jsonl)
    device = torch.device("mps" if args.device == "auto" and torch.backends.mps.is_available() else ("cpu" if args.device == "auto" else args.device))
    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.revision)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    source = AutoModelForCausalLM.from_pretrained(args.model, revision=args.revision, dtype=torch.float32).eval()
    compat = build_compat_from_hf(AutoConfig.from_pretrained(args.model, revision=args.revision))
    source_state = {name: value.detach().cpu() for name, value in source.state_dict().items()}
    import_report = import_into_qwen_compat(compat, source_state, source_name=args.model, source_revision=args.revision)
    del source_state, source
    torch.manual_seed(args.seed)
    model = augment_qwen_compat_model(compat, num_routed_experts=1, top_k=1, latent_size=args.latent_size, default_e3_steps=counts[0])
    parity_ids = tokenizer(str(selection_tasks[0]["prompt"]), return_tensors="pt")["input_ids"]
    phase0b = gate0b_exact_parity(compat, model, parity_ids)
    if phase0b["decision"] != "PASS_EXACT":
        raise RuntimeError(f"Gate 0B failed: {phase0b}")
    init = prepare_exfusion_for_training(model, gate0b_passed=True, epsilon=args.e3_scale)
    model, compat = model.to(device), None
    initial_refinement, initial_scale = copy.deepcopy(model.layers[-1].latent_refine.state_dict()), model.layers[-1].latent_scale.detach().clone()
    e2_anchor = _e2_logits(model, tokenizer, str(selection_tasks[0]["prompt"]), device)
    variants, candidate_states = {}, {}
    for latent_steps in counts:
        model.layers[-1].latent_refine.load_state_dict(initial_refinement)
        with torch.no_grad():
            model.layers[-1].latent_scale.copy_(initial_scale)
        model.default_e3_steps = latent_steps
        stage = TrainingStageConfig(
            name=f"e3_hardcase_{latent_steps}_steps", steps=args.steps,
            train_parameter_groups=("e3_refinement", "e3_scale"), freeze_parameter_groups=(),
            lr_new=args.lr_refinement, lr_scales=args.lr_scale, effort_sampling=(0.0, 0.0, 0.0, 1.0),
            distill_e0=False, distill_e1=False, train_e3=True,
        )
        cfg = RealTrainConfig(
            steps=args.steps, batch_size=1, seq_len=args.seq_len, grad_accum=1, warmup_steps=min(10, args.steps),
            log_every=max(1, args.steps // 10), eval_every=args.steps + 1, seed=args.seed, device=str(device),
            effort_schedule=("fixed_3",), data_path=str(train_jsonl), tokenizer_name=args.model, tokenizer_revision=args.revision,
            output_dir=str(output / f"latent_{latent_steps}"), stages=(stage,), save_periodic_checkpoints=False,
            save_final_checkpoint=False, save_model_artifact=False,
        )
        result, selection = train_adapt(model, cfg), _task_eval(model, tokenizer, selection_tasks, device=device, max_new_tokens=args.max_new_tokens)
        e2_unchanged = torch.equal(_e2_logits(model, tokenizer, str(selection_tasks[0]["prompt"]), device), e2_anchor)
        if not e2_unchanged:
            raise RuntimeError("E2 anchor output changed during an E3-only phase")
        variants[str(latent_steps)] = {
            "latent_steps": latent_steps, "training": {"config": asdict(cfg), "optimizer_steps": result["optimizer_steps_completed"], "history_tail": result["history_tail"], "trained_parameters": result["stage_parameter_membership"]["trained"]},
            "selection": selection, "raw_latent_scale": float(model.layers[-1].latent_scale.detach().cpu()),
            "effective_latent_scale": float((model.latent_scale_limit * torch.tanh(model.layers[-1].latent_scale / model.latent_scale_limit)).detach().cpu()), "e2_anchor_unchanged": e2_unchanged,
        }
        candidate_states[latent_steps] = {"refinement": copy.deepcopy(model.layers[-1].latent_refine.state_dict()), "scale": model.layers[-1].latent_scale.detach().clone()}
    winner = max(counts, key=lambda step: (variants[str(step)]["selection"]["net_rescue_rate"], variants[str(step)]["selection"]["e3_accuracy"], -variants[str(step)]["selection"]["e3_completion_ce"]))
    model.layers[-1].latent_refine.load_state_dict(candidate_states[winner]["refinement"])
    with torch.no_grad():
        model.layers[-1].latent_scale.copy_(candidate_states[winner]["scale"])
    model.default_e3_steps = winner
    heldout = _task_eval(model, tokenizer, test_tasks, device=device, max_new_tokens=args.max_new_tokens)
    qualified = heldout["net_rescue_rate"] > 0.0 and heldout["e3_accuracy"] > heldout["e2_accuracy"]
    report = {
        "experiment": "frozen-e2-hardcase-e3-latent-step-ablation", "model": {"id": args.model, "revision": args.revision, "source_exact_coverage_percent": import_report.exact_coverage_percent},
        "environment": {"torch": torch.__version__, "platform": platform.platform(), "device": str(device)},
        "datasets": {key: {"path": path, "sha256": _sha256(Path(path)), "tasks": len(tasks)} for key, path, tasks in (("train", args.hard_train, train_tasks), ("selection", args.selection, selection_tasks), ("test", args.test, test_tasks))},
        "phase0b": phase0b, "post_gate0b_initialization": asdict(init), "variants": variants, "selected_latent_steps": winner, "heldout": heldout,
        "qualification": {"e2_frozen": all(value["e2_anchor_unchanged"] for value in variants.values()), "positive_heldout_net_rescue": heldout["net_rescue_rate"] > 0.0, "heldout_e3_accuracy_above_e2": heldout["e3_accuracy"] > heldout["e2_accuracy"], "qualified": qualified},
        "limitations": ["This is a targeted task-loss ablation, not a general-language capability claim.", "Causal-LM training currently supervises prompt tokens as well as answer tokens.", "Do not train or claim an effort router unless this result replicates on independent hard-task families."],
    }
    (output / "e3_hardcase_ablation_report.json").write_text(json.dumps(report, indent=2, default=str))
    print(json.dumps(report["qualification"], indent=2))


if __name__ == "__main__":
    main()

# Canonical pipeline commands

These examples assume commands run from the project directory.

## Phase 0A and exact Gate 0B

```bash
python scripts/run_phase0_retention.py \
  --hf-model Qwen/Qwen2.5-0.5B-Instruct \
  --hf-revision <immutable-commit-sha> \
  --data data/retention.jsonl \
  --output runs/phase0 \
  --phase both \
  --shallow-continuation
```

This writes `phase0b_gate_report.json` and `qwen_exfusion_gate0b.pt`.

## Staged multi-effort adaptation

```bash
python - <<'PY'
import json
from daph import (
    RealTrainConfig, TrainingStageConfig, load_qwen_exfusion_checkpoint,
    prepare_exfusion_for_training, train_adapt,
)

model = load_qwen_exfusion_checkpoint("runs/phase0/qwen_exfusion_gate0b.pt")
gate = json.load(open("runs/phase0/phase0b_gate_report.json"))
prepare_exfusion_for_training(model, gate0b_passed=gate["result"] == "PASS_EXACT", epsilon=1e-4)
stages = (
    TrainingStageConfig(
        name="stage1_new_modules", steps=1000,
        train_parameter_groups=("continuation", "augmentation", "scales"),
        freeze_parameter_groups=("imported",),
        effort_sampling=(0.30, 0.30, 0.0, 0.40),
    ),
    TrainingStageConfig(
        name="stage2_low_lr_backbone", steps=500,
        train_parameter_groups=("imported", "new", "scales"),
        freeze_parameter_groups=(), lr_pretrained=1e-6, lr_new=1e-4,
        effort_sampling=(0.30, 0.30, 0.20, 0.20),
    ),
)
train_adapt(model, RealTrainConfig(
    steps=1500, data_path="data/adapt.jsonl", val_path="data/validation.jsonl",
    output_dir="runs/adapt", stages=stages, grad_accum=8,
    retention_kl_threshold=0.05, device="cuda",
))
PY
```

## Per-effort evaluation

```bash
python - <<'PY'
from daph import eval_per_effort, load_qwen_exfusion_checkpoint
from daph.train_real import TextBatcher, load_jsonl_texts, try_load_tokenizer
import torch

model = load_qwen_exfusion_checkpoint("runs/adapt/model_final.pt")
batcher = TextBatcher(load_jsonl_texts("data/validation.jsonl"), tokenizer=None,
                      seq_len=256, batch_size=4, device=torch.device("cpu"))
print(eval_per_effort(model, batcher, n_batches=10, detailed=True))
PY
```

## Counterfactual collection

```bash
python - <<'PY'
import json, torch
from daph import CounterfactualCollector, load_qwen_exfusion_checkpoint

model = load_qwen_exfusion_checkpoint("runs/adapt/model_final.pt").eval()
tasks = [json.loads(line) for line in open("data/tasks-tokenized.jsonl")]
for task in tasks:
    task["input_ids"] = torch.tensor(task["input_ids"])
with CounterfactualCollector(model, lambda_cost=0.15) as collector:
    collector.collect_many(tasks, out_path="runs/counterfactuals.jsonl")
PY
```

## Effort and oracle qualification

```bash
python - <<'PY'
import json
from daph import EffortCounterfactual, oracle_analysis, qualify_effort_hierarchy

records = [EffortCounterfactual(**json.loads(line)) for line in open("runs/counterfactuals.jsonl")]
effort_report = qualify_effort_hierarchy(records)
assert effort_report["qualified"], effort_report
oracle_report = oracle_analysis(records)
assert oracle_report["oracle_gap_lcb95"] > 0, oracle_report
print(effort_report)
print(oracle_report)
PY
```

## Hidden policy training

```bash
python - <<'PY'
import json
from daph import EffortController, EffortCounterfactual, EffortPolicyTrainer, PolicyTrainingConfig

records = [EffortCounterfactual(**json.loads(line)) for line in open("runs/counterfactuals.jsonl")]
controller = EffortController(hidden_size=len(records[0].probe_hidden), num_levels=4)
trainer = EffortPolicyTrainer(controller, PolicyTrainingConfig(epochs=20, batch_size=32))
metrics, receipt = trainer.fit(records, mode="hidden")
print(metrics)
print(receipt.to_dict())
PY
```

Policy claims must still be evaluated against best-fixed, prompt-sham, effort-frequency random, raw-compute-matched random, oracle, IID test, and leave-family-out OOD controls.

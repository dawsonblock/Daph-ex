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

## Memory-bounded real-model smoke

After reconstructing the deterministic WikiText-2 JSONL slices described in `QUALITY_CORRECTION_REPORT.md`:

```bash
python scripts/run_small_real_adaptation.py \
  --model Qwen/Qwen2.5-0.5B \
  --revision 060db6499f32faf8b98477b0a26969ef7d8b9987 \
  --train runs/wikitext2-smoke/train.jsonl \
  --validation runs/wikitext2-smoke/validation.jsonl \
  --output runs/qwen2.5-0.5b-wikitext2-final-selected \
  --exit-steps 100 --e3-steps 100 \
  --e0-layers 18 --e1-layers 22 \
  --seq-len 32 --eval-batches 16
```

The harness pins the tokenizer revision, trains and validation-selects E0/E1/E3 separately, freezes imported E2 weights, rejects non-finite loss/gradients, skips large checkpoint serialization, and writes `experiment_report.json`.

## Layer-contribution profile

The profiler CLI consumes tokenized JSONL rows containing `input_ids`. Its bundled objective is supervised CE and is labeled accordingly.

```bash
# C. sparse profile (recommended first real scan)
python scripts/profile_layer_contribution.py \
  --checkpoint runs/phase0/qwen_exfusion_gate0b.pt \
  --train data/profile-train-tokenized.jsonl \
  --validation data/profile-validation-tokenized.jsonl \
  --profile-mode sparse --steps 100 \
  --output artifacts/layer_profile/sparse

# D. full profile (only after cost review)
python scripts/profile_layer_contribution.py \
  --checkpoint runs/phase0/qwen_exfusion_gate0b.pt \
  --train data/profile-train-tokenized.jsonl \
  --validation data/profile-validation-tokenized.jsonl \
  --profile-mode full --steps 100 \
  --output artifacts/layer_profile/full

# E. direct single-layer reproduction-style run
python scripts/profile_layer_contribution.py \
  --checkpoint runs/phase0/qwen_exfusion_gate0b.pt \
  --train data/profile-train-tokenized.jsonl \
  --validation data/profile-validation-tokenized.jsonl \
  --layers 12 --steps 100 \
  --output artifacts/layer_profile/layer12
```

For verified reward/GRPO, call `LayerContributionProfiler.run()` with a `LayerAdaptationObjective(kind="verified_reward", verified_reward=True)` and an external adaptation callback. The repository intentionally does not describe the CE CLI as RLVR.

## Frozen-E2 hard-case E3 ablation

```bash
python scripts/make_e3_hardcase_arithmetic.py --output runs/e3-hardcase-data
python scripts/run_e3_hardcase_ablation.py \
  --model Qwen/Qwen2.5-0.5B \
  --revision 060db6499f32faf8b98477b0a26969ef7d8b9987 \
  --hard-train runs/e3-hardcase-data/train.jsonl \
  --selection runs/e3-hardcase-data/selection.jsonl \
  --test runs/e3-hardcase-data/test.jsonl \
  --output runs/e3-hardcase-ablation \
  --latent-step-counts 1,2,4 --steps 200 --e3-scale 1e-3
```

This keeps E2 frozen, trains only the configured E3 refiner and scale after Gate 0B, and evaluates exact numeric E2/E3 outcomes. New checkpoints default to middle refinement; historical checkpoints retain final refinement. The report includes rescues, regressions, net rescue rate, completion CE, hidden-state delta magnitude, and receipt-backed compute overhead. The supplied arithmetic generator is only a reproducible smoke curriculum; qualify a policy only after independent hard-task-family replication.

## E3 architecture, dose, and location contracts

```bash
python - <<'PY'
from daph import canonical_variant_matrix, dose_response_variants, location_ablation_variants
print("F/J profile-vs-heuristic matrix:", canonical_variant_matrix())
print("I dose response:", dose_response_variants((0, 1, 2, 4, 8)))
print("J location ablation:", location_ablation_variants(steps=2))
PY
```

Use `run_variant_study(variants, evaluate_callback, output_dir)` to emit one compatible JSON/CSV schema. The callback must hold data, steps, optimizer, seed, and metric definition fixed. `configure_e3_training()` implements E3-A and controlled E3-B parameter opening; `E3HardCaseMiner` writes the hard-case mining manifest for G/H. Pass the tokenizer to `E3HardCaseMiner(..., tokenizer=tokenizer)` when using the built-in text verifiers. Mining fails closed on `UNVERIFIABLE`, execution-error, or timeout statuses so those examples cannot be mislabeled as hard E2 failures.

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
effort_report = qualify_effort_hierarchy(records)
oracle_report = oracle_analysis(records)
trainer.authorize_policy_training(effort_report, oracle_report)
metrics, receipt = trainer.fit(records, mode="hidden")
print(metrics)
print(receipt.to_dict())
artifact, policy_state = trainer.build_artifact(
    base_model_digest=records[0].model_digest,
    train_records=records,
    metrics=metrics,
)

# Only after the fixed-arm and oracle gates pass, install the verified state
# into the canonical model and let its shared early-Qwen probe dispatch
# physical E0–E3 execution.
from daph import load_qwen_exfusion_checkpoint
model = load_qwen_exfusion_checkpoint("runs/adapt/model_final.pt")
model.install_effort_policy(
    artifact, policy_state,
    base_model_digest=records[0].model_digest,
)
batch_ids = torch.tensor([[1, 2, 3]])
out = model(batch_ids, effort_mode="adaptive", return_compute_receipt=True)
print(out["effort_decision"], out["compute_stats"])
PY
```

Policy claims must still be evaluated against best-fixed, prompt-sham, effort-frequency random, raw-compute-matched random, oracle, IID test, and leave-family-out OOD controls.

## A–O workflow map

| Stage | Command/API | Mandatory stop gate |
|---|---|---|
| A | `run_phase0_retention.py --phase 0a` | source import/parity |
| B | `run_phase0_retention.py --phase both` | exact Gate 0B |
| C | `profile_layer_contribution.py --profile-mode sparse` | partial result labeled partial |
| D | same with `--profile-mode full` | cost review first |
| E | same with `--layers K` | single imported layer only |
| F | `E3RefinementConfig(e3_region_selection="middle_heuristic")` | compare final control |
| G | `E3HardCaseMiner.mine()/save()` | verifier required |
| H | `configure_e3_training()` + external verified loss | stop if regressions dominate |
| I | `dose_response_variants()` + `run_variant_study()` | no monotonicity assumption |
| J | `location_ablation_variants()` | equal capacity/budget |
| K | `qualify_effort_hierarchy()` | arms must qualify |
| L | `CounterfactualCollector.collect_many()` | frozen model/digests |
| M | `oracle_analysis()` | LCB95 gap > 0 and non-collapse |
| N | `authorize_policy_training()` then `fit()` | both K and M pass |
| O | IID + leave-family-out evaluation and sham/random controls | untouched test |

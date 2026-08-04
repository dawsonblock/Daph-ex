# Phase 0 Experimental Plan — Two-Stage Retention

## Phase 0A — Qwen implementation parity

**Compare:** HuggingFace Qwen vs `QwenCompatModel`  
**Import:** exact key map, 100% backbone coverage target  
**Threshold:** relative CE ≤ 1% (default) → PASS  
**Meaning:** Our modules reproduce Qwen.

## Phase 0B — exact QwenExFusion conversion

**Compare:** `QwenCompatModel` vs `QwenExFusionModel` fixed E2  
**Conversion:** `augment_qwen_compat_model()` with exact copied backbone and zero augmentation scales  
**Threshold:** near-numerical parity; binary `PASS_EXACT` or `FAIL`  
**Meaning:** The canonical adaptive model preserves the complete pretrained E2 anchor exactly.

## Artifacts (all required)

- `phase0_import_report.json` / `phase0a_*` / `phase0b_*`
- `phase0_metrics.json`
- `phase0_decision.md`
- `phase0_config.json`
- `phase0_dataset_manifest.json`
- `phase0b_gate_report.json`
- `qwen_exfusion_gate0b.pt`

## Runner

```bash
python scripts/run_phase0_retention.py --synthetic --output runs/phase0
python scripts/run_phase0_retention.py \
  --hf-model Qwen/Qwen2.5-0.5B-Instruct \
  --hf-revision <sha> \
  --data val.jsonl \
  --phase both \
  --output runs/phase0_qwen
```

Do **not** adapt effort modes until 0A and exact 0B pass. Do **not** train the policy until the physical effort hierarchy and oracle gate also qualify.

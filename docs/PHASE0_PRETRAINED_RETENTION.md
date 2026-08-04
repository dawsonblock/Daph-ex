# Phase 0 — Pretrained Retention Gate

**Do not train the effort policy until this gate passes.**

## Goal

Measure whether a dimension-matched Qwen (or compatible) checkpoint can be
imported into ExFusion **without destroying most of its capability**.

## Procedure

1. Choose source: e.g. `Qwen2.5-0.5B-Instruct` (or smaller).
2. Build ExFusion config with **matching** `hidden_size`, `num_layers`,
   `num_attention_heads`, `vocab_size` (use source tokenizer).
3. Disable AttnRes, keep `allow_partial_block=False`.
4. Import:
   ```python
   report = load_pretrained_into_exfusion(model, hf_model_id=..., zero_init_new=True)
   ```
5. Record report fields:
   - `exact_coverage_percent`
   - `coverage_percent` (exact + vocab-safe)
   - `partial_block_parameters` (must be 0 unless explicitly opted in)
   - `newly_initialized_parameters`
6. Evaluate **before any training**:
   - Source model validation loss / accuracy
   - ExFusion **E2** validation loss / accuracy (same tokenizer, same data)
   - Optionally E0/E1/E3 (expect worse)

## Gate

Proceed only if E2 retains a **substantial** fraction of source quality
(e.g. modest PPL increase, not collapse). Exact threshold is an experiment
choice; document it before measuring.

If E2 collapses: repair identity init / mapping; do **not** freeze bad weights
and hope Stage-1 recovers.

## After gate passes

Stage 1: freeze exact-matched imported keys; train new modules.
Stage 2: multi-effort `sample` curriculum.
Then oracle → policy.

## Out of scope for Phase 0

- Effort policy training
- Model merging (DARE/TIES/Fisher)
- AttnRes
- GPU kernel optimization claims

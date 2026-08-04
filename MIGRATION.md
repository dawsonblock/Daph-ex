# Migration to QwenExFusion v3.2 middle-layer E3

`DAPHHybridModelV3` has not been removed. Existing checkpoints, APIs, tests, and research experiments using that architecture continue to work. It is now classified as the legacy experimental hybrid path.

New pretrained experiments must use this sequence:

1. Import the immutable Hugging Face Qwen checkpoint into `QwenCompatModel` and pass Gate 0A.
2. Call `augment_qwen_compat_model(compat, ...)`.
3. Run exact Gate 0B with `effort_mode="fixed_2"`; archive `phase0b_gate_report.json`.
4. Call `prepare_exfusion_for_training(..., gate0b_passed=True)` if selected augmentation modules need a nonzero training epsilon.
5. Profile adaptation contribution on the exact checkpoint, or predeclare the 40%–60% heuristic as a partial-evidence baseline.
6. Train E0/E1 with the frozen E2 teacher. Train E3 through `configure_e3_training()` with a verified-task objective and only a moderate regression guard.
7. Freeze the qualified model before counterfactual collection. Run the oracle gate before policy training.

Important API differences:

- `QwenExFusionModel.forward(..., return_compute_receipt=True)` returns `logits`, `compute_receipt`, and a serializable `compute_stats` dictionary. The default remains a logits tensor for compatibility.
- E0/E1 are partial-depth exits, not post-backbone branches.
- E2 never enables new branches by default.
- E3 defaults to a bounded middle-layer recurrent delta. `final_refine` remains an explicit control and `middle_repeat` is experimental.
- Fractional layer regions are zero-based and include `floor(start × L)` through `ceil(end × L) - 1`.
- `compute_effort_probe()` returns a structured, continuable internal-Qwen state. Raw embeddings are reserved for the prompt-sham control.
- Adaptive inference now raises unless a verified policy is installed (or an explicit research effort override is supplied).
- `EffortPolicyTrainer.fit()` raises until `authorize_policy_training()` receives passing arm and oracle reports.
- Exact imported/new/augmentation/scale parameter names come from `model.parameter_provenance`; canonical optimizer grouping does not infer them from substrings.
- AttnRes is disabled in the canonical path until cross-layer history is wired.

Legacy `load_pretrained_into_exfusion()` and `DAPHHybridModelV3` utilities remain for reproduction of earlier experiments; they are not the canonical Phase 0B route.

Old QwenExFusion checkpoints without `e3_config` load with `final_refine`, preserving their historical graph. New checkpoints serialize the E3 configuration, selected region, profile digest, and probe depth.

## HRM adaptive-memory namespace in 3.6

The canonical HRM research package is now `hrm_adaptive_memory`:

```python
from hrm_adaptive_memory.experiments.context_study import ContextStudyRunner
```

Imports through `hrm_memory` remain deprecated compatibility aliases for the
3.6 release and emit `DeprecationWarning`. They are scheduled for removal in
3.7. Migrate imports now; do not create new modules under the legacy namespace.

This rename does not enable RuVector, Graphiti, consolidation, adaptive
recurrence, or executive-policy training. Those remain fail-closed behind the
paired B0/B1/B2/B3 Gate A study and their later scientific gates.

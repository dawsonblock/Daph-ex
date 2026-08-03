# Migration to canonical QwenExFusion

`DAPHHybridModelV3` has not been removed. Existing checkpoints, APIs, tests, and research experiments using that architecture continue to work. It is now classified as the legacy experimental hybrid path.

New pretrained experiments must use this sequence:

1. Import the immutable Hugging Face Qwen checkpoint into `QwenCompatModel` and pass Gate 0A.
2. Call `augment_qwen_compat_model(compat, ...)`.
3. Run exact Gate 0B with `effort_mode="fixed_2"`; archive `phase0b_gate_report.json`.
4. Call `prepare_exfusion_for_training(..., gate0b_passed=True)` if selected augmentation modules need a nonzero training epsilon.
5. Train E0/E1 with the frozen E2 teacher and E3 primarily with task loss.
6. Freeze the qualified model before counterfactual collection and policy training.

Important API differences:

- `QwenExFusionModel.forward(..., return_compute_receipt=True)` returns `logits`, `compute_receipt`, and a serializable `compute_stats` dictionary. The default remains a logits tensor for compatibility.
- E0/E1 are partial-depth exits, not post-backbone branches.
- E2 never enables new branches by default.
- E3 is the only canonical mode that enables recurrent/MoE/latent additions.
- Exact imported/new/augmentation/scale parameter names come from `model.parameter_provenance`; canonical optimizer grouping does not infer them from substrings.
- AttnRes is disabled in the canonical path until cross-layer history is wired.

Legacy `load_pretrained_into_exfusion()` and `DAPHHybridModelV3` utilities remain for reproduction of earlier experiments; they are not the canonical Phase 0B route.

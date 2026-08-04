# Real qualification preflight v3.4.1

The pinned Qwen2.5-0.5B run reached the first real protocol gate and stopped correctly.

- 3,600 verified multi-family tasks generated.
- 500 frozen as untouched natural held-out.
- 3,100 evaluated under E2 for calibrated-sensitivity capacity.
- Required: at least five family-balanced mixed-success bands for 500 train, 200 selection, and 500 calibrated-test tasks.
- Observed: only three families had sufficient E2 successes and failures.
- Decision: `NOT_RUN_CALIBRATION_CAPACITY_FAILED`.

No E3 training, placement comparison, or router training was performed. The complete capacity table and compressed raw E2 outcomes are in [`evidence/e3_qualification_v341_preflight`](../evidence/e3_qualification_v341_preflight).

A local one-step/24-task real-model timing smoke completed in 23.86 seconds and peaked at approximately 8.4 GB. Because the full protocol repeats E2/E3 generation for three doses, three placements, three seeds, and two 500-task tests, the current uncached runner is estimated at roughly 18–30 hours on the available 16 GB Apple MPS host, plus profiling.

The preferred execution target is a Hugging Face `t4-small` job (16 GB VRAM), billed at $0.40/hour as of this preflight ([official Jobs pricing](https://huggingface.co/docs/hub/jobs-pricing)). A remote job must persist every completed arm to a Hub dataset repository and use a timeout with checkpoint/resume margin. Paid cloud submission requires explicit budget approval.

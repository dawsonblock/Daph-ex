# E3 experiment protocol v3.4.0

## Pre-registration

Use [`configs/e3_qualification_v340.json`](../configs/e3_qualification_v340.json). Record model/revision, source and tokenizer digests, training and evaluation seeds, placement/dose arms, split IDs, lambda values, bootstrap group, sample size, promotion thresholds, and expected claim level before training.

Tiers are configurable. As of v3.4.1, the executable paths enforce their task, group, and independent-seed minimums before a result is promotable. `SMOKE` validates mechanics, `PILOT` estimates effects, `QUALIFICATION` requires at least 500 held-out tasks and three seeds, and `FINAL` requires an exact predeclared size. Profile tiers separately distinguish `PROFILE_SMOKE`, `PROFILE_PILOT`, and `PROFILE_FULL`. See the superseding [`E3_EXPERIMENT_PROTOCOL_V341.md`](E3_EXPERIMENT_PROTOCOL_V341.md).

## Data

Maintain two immutable tests:

1. Calibrated sensitivity: selected only from E2 outcomes to create a mixed-success band. It may expose rescues and regressions efficiently.
2. Natural held-out: selected from generator state before E2/E3 evaluation. It cannot inspect either arm's outcomes.

Use exact verifiers and include family, template, difficulty, generator version, verifier version, and generation seed. Initial families include carry addition, subtraction, multiplication, comparison, modular and multi-step arithmetic, symbolic substitution, pattern continuation, and code-output prediction.

## Training

Freeze E2. Train the refiner and E3 scale with answer-token-only CE and a weak KL regression guard. Log task loss, unweighted/weighted guard, total loss, scale, hidden delta, gradient norms, examples, tokens, and optimizer steps. Mine `HARD_FAILURE`, `HARD_UNCERTAIN`, and `EASY_CORRECT_REGRESSION_GUARD` in declared proportions and save realized proportions/source IDs.

Answer-only CE is teacher-forced supervised training. It is not RLVR. Sequence-level verified training must use a real external callback; the GRPO adapter deliberately raises until a real implementation is installed.

## Arms

Compare final, heuristic middle, and stable-profile middle placement at matched doses `1, 2, 4` (optionally `8` after predeclaration), three training seeds, identical optimizer/data budgets, and both tests. Do not assume higher dose or profiled placement wins.

## Statistics

Bootstrap template or family groups, not independent prompt variants. Report mean, SE, CI95, LCB95, and per-seed/per-family/per-difficulty results for both quality and utility. Report rescues, regressions, net rescue, CE, hidden delta, receipt compute, and latency.

## Profile promotion

Across seeds report Spearman rank correlation, top-k overlap, middle-region stability, best-layer stability, best-contiguous-region stability, and layer contribution mean/std. An unstable profile cannot promote a profiled placement; heuristic middle remains canonical.

## Frontier and router gate

After a non-E2 arm qualifies, freeze the model and collect actual E0-E3 receipts per task. Mark arms `PARETO`, `DOMINATED`, `UNQUALIFIED`, or `ANCHOR`. Then test the oracle-minus-best-fixed utility gap with grouped bootstrap. Policy training remains false unless a non-E2 arm qualifies and the oracle lower bound is positive.

## Claim language

Generated reports use only `ENGINEERING_PASS`, `MECHANISM_SIGNAL`, `PILOT_EVIDENCE`, `STATISTICALLY_QUALIFIED`, or `FINAL_EVIDENCE`. One changed answer among 24 tasks cannot be called statistically qualified.

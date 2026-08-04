# E3 v3.4.1 qualification preflight

The real pinned Qwen2.5-0.5B E2 calibration pass evaluated 3,100 candidates while keeping a separate 500-task natural held-out set untouched. The preflight failed closed before E3 training because no subset of at least five families could supply the predeclared 500/200/500 splits at approximately 50% E2 accuracy within every family.

Only integer comparison, modular arithmetic, and pattern continuation were naturally mixed at the required scale. Addition, multiplication, and subtraction were overwhelmingly E2-correct; code output, multi-step arithmetic, and symbolic substitution were overwhelmingly E2-wrong.

This is not an E3 capability result. It is evidence that the first qualification candidate distribution cannot identify rescue and regression behavior across five balanced families. The gate was not weakened, no E3 outcomes were inspected, and E3 training did not start.

The raw E2 outcomes are retained as `e2_outcomes.jsonl.gz`. The next run must expand or predeclare better difficulty ladders until at least five families satisfy the calibration-capacity gate. The untouched natural test must continue to retain all nine families.

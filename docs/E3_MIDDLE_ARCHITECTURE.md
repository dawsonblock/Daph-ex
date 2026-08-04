# E3 middle-layer architecture report

## Canonical graph

E2 executes every imported Qwen layer exactly once, followed by the preserved final norm and LM head. E3 executes that full path and injects additional bounded computation after the selected internal insertion layer:

`h <- h + alpha * (R_k(h) - h)`

where `alpha` is zero at Gate 0B and bounded with `tanh`. The remaining imported layers decode the refined state. The default heuristic selects zero-based layers `floor(0.40L)` through `ceil(0.60L)-1` and inserts at the central selected layer.

Available modes are `none`, `final_refine`, `middle_recurrent`, `middle_repeat`, and `profiled_middle_recurrent`. Repeated pretrained layers reuse the same parameter objects and place their aggregate delta behind the same zero-initialized bounded gate. Thus even this experimental variant preserves E3=E2 at Gate 0B while receipts still count the attempted extra operations.

## Training

`configure_e3_training()` implements E3-A (refiner and scale only) and the controlled E3-B opening of explicitly permitted selected imported layers. Gate 0B uses exact zero; the training transition uses the serialized epsilon. `e3_verified_objective()` makes the external differentiable verified-task loss primary and adds only a configured E2 KL regression guard. E0/E1 distillation remains separate.

## Evidence status

Unit tests establish graph placement, identity at zero scale, parameter sharing, gradients, and compute accounting. They do not establish task improvement. The earlier final-state real-model run remains the only bundled real checkpoint evidence and was scientifically negative/insufficient. A sparse exact-checkpoint profile and controlled V1–V5 hard-task study are still required.

Promotion requires separate positive grouped-bootstrap lower bounds for verified quality (E3-Q) and receipt-priced utility (E3-U), more rescues than regressions, cross-seed replication, and a natural-test pass. If profile rankings are unstable, the profiled placement cannot be promoted; the heuristic middle remains canonical. A qualified arm still requires a positive oracle-opportunity gate before router training.

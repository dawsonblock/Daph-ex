# Standalone marginal-utility controller

## Scope

`daph_metareasoner` is a deliberately small Stage 1 research system around one frozen base model. It does not add a latent workspace, retrieval system, specialist network, vector communication, or jointly trained backbone.

At each state it considers only:

`STOP`, `THINK`, `VERIFY`, and `DECOMPOSE`.

The controller predicts gross verified-quality change. Runtime subtracts action cost and an uncertainty penalty exactly once:

`VOC(s,a) = predicted_quality_gain(s,a) - action_cost(a) - beta * uncertainty(s,a)`

It stops when every available continuation has non-positive conservative VOC.

Experience records keep gross quality change, cost, and net utility as separate fields. This prevents cost from being embedded in the label and then accidentally subtracted again by the policy.

## Implemented gates

The workflow fails closed in this order:

1. Counterfactual branches must start from the same immutable state and carry model, environment, task, dataset, and execution digests.
2. The oracle must beat the best fixed action by more than `0.02`, with a positive paired bootstrap lower bound.
3. A hidden-state continuation probe must exceed AUROC `0.65` and beat a cheap-feature probe by more than `0.03`.
4. Only then may hidden and cheap-sham action-value ensembles train.
5. The hidden policy must have a positive held-out utility lower bound against best-fixed, cheap sham, selected heuristic, family lookup, and action-frequency-matched random controls.
6. Oracle capture must exceed 25%, and the same comparison gates must pass OOD.
7. Only a `VERIFIED_FIT` artifact may execute on-path without an explicit research override.

The runtime executes only the chosen action. It has no access to counterfactual alternatives and enforces step, token, latency, cost, repeated-action, state-recurrence, unchanged-answer, and alternating-cycle limits.

## Hidden and cheap features

The frozen Hugging Face adapter captures pooled hidden states at 25%, 50%, 75%, and 100% of model depth, plus the final-token state. Runtime features include answer entropy/log probability, confidence, token count, step, budget, hidden-state change, answer change, confidence slope, and repetition count. The controller's shared representation has a separate calibrated-current-correctness head; model confidence is retained as a feature but is not treated as correctness.

The cheap sham sees prompt length/shape and runtime confidence statistics but no hidden representation. Additional baselines cover fixed action, confidence threshold, entropy threshold, answer stability, prompt length, family lookup, and action-frequency-matched random selection.

## Commands

Create the predeclared split sizes:

```bash
python scripts/make_voc_tasks.py --output runs/voc/tasks
```

Collect each split with the same immutable checkpoint:

```bash
python scripts/collect_voc_experience.py \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --revision 7ae557604adf67be50417f59c2c2f167def9a775 \
  --tasks runs/voc/tasks/experience.jsonl \
  --split experience \
  --output runs/voc/experience.jsonl
```

Run the mandatory oracle gate:

```bash
python scripts/analyze_voc_oracle.py \
  --experience runs/voc/experience.jsonl \
  --output runs/voc/oracle.json
```

Training refuses to proceed unless the oracle and probe gates pass:

```bash
python scripts/train_voc_controller.py \
  --train runs/voc/experience.jsonl \
  --validation runs/voc/validation.jsonl \
  --test runs/voc/test.jsonl \
  --ood runs/voc/ood.jsonl \
  --output runs/voc/controller
```

On-path execution accepts verified artifacts by default:

```bash
python scripts/run_voc_on_path.py \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --revision 7ae557604adf67be50417f59c2c2f167def9a775 \
  --controller runs/voc/controller/hidden_controller.pt \
  --tasks runs/voc/tasks/test.jsonl --split test \
  --output runs/voc/on_path_test.json
```

The final proof runs complete learned, sham, fixed, heuristic, and immediate-stop policies independently. Every policy performs its own real model execution; paired utility confidence bounds are calculated per task:

```bash
python scripts/run_voc_policy_suite.py \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --revision 7ae557604adf67be50417f59c2c2f167def9a775 \
  --hidden-controller runs/voc/controller/hidden_controller.pt \
  --sham-controller runs/voc/controller/sham_controller.pt \
  --tasks runs/voc/tasks/test.jsonl --split test \
  --fixed-action STOP \
  --output runs/voc/on_path_policy_suite.json
```

The counterfactual oracle remains an upper bound and is never exposed to the on-path policy.

The full predeclared configuration is [`configs/metareasoner_stage1.json`](../configs/metareasoner_stage1.json).

## First real-model smoke

An engineering smoke used the pinned Qwen2.5-0.5B-Instruct checkpoint, 16 experience tasks, and eight tasks in each validation, immutable test, and OOD split. Every initial state branched independently to all four actions.

| Split | States | Best fixed | Best fixed marginal utility | Oracle utility | Oracle gain over fixed | Oracle gain LCB | Gate |
|---|---:|---|---:|---:|---:|---:|---|
| experience | 16 | THINK | 0.1050 | 0.1225 | 0.0175 | 0.0150 | fail |
| validation | 8 | STOP | 0.0000 | 0.0000 | 0.0000 | 0.0000 | fail |
| test | 8 | STOP | 0.0000 | 0.0000 | 0.0000 | 0.0000 | fail |
| OOD | 8 | THINK | 0.1050 | 0.1225 | 0.0175 | 0.0125 | fail |

The experience oracle used THINK for two states and STOP for 14. Validation and test selected STOP for every state. The predeclared minimum oracle gain was `0.02`, so controller training was blocked before the action-value ensemble.

This is not a negative result about learned metareasoning: the smoke is far too small and the action prompts are primitive. It is a successful negative test of the research protocol. The system did not manufacture a router result when the sampled action space lacked enough conditional value.

The next allowed experiment is a better-powered experience generation—not added architecture. It should broaden verified task difficulty and include more genuine correct-to-wrong and wrong-to-correct continuations. If two or three properly powered generations still fail the oracle, hidden-signal, or on-path policy gates, the program stop rule applies.

Compressed raw state/action records, tasks, split manifests, execution receipts, oracle reports, the blocked training receipt, and SHA-256 hashes are bundled under [`evidence/voc_stage1_smoke_v1`](../evidence/voc_stage1_smoke_v1/manifest.json).

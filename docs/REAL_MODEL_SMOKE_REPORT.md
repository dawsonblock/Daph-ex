# Qwen2.5-0.5B real-model smoke report

> This report preserves the original failing smoke result. The failure analysis, corrected architecture, and replacement measurements are in [`QUALITY_CORRECTION_REPORT.md`](QUALITY_CORRECTION_REPORT.md).

## Decision

**Engineering validation: PASS. Quality qualification: FAIL. Policy training: BLOCKED by design.**

The canonical path was exercised with a real pretrained checkpoint and real text:

`HF Qwen → QwenCompat → Gate 0A → QwenExFusion → exact Gate 0B → E0/E1/E3 adaptation → per-effort evaluation`

The experiment proved exact E2 retention and a physical E0 < E1 < E2 < E3 hierarchy. Ten memory-bounded updates slightly improved E0 and E1, but did not make either shallow exit competitive with E2, and E3 degraded instead of improving. These modes must not be routed by a learned policy yet.

## Immutable inputs

| Item | Value |
|---|---|
| Model | `Qwen/Qwen2.5-0.5B` |
| Model revision | `060db6499f32faf8b98477b0a26969ef7d8b9987` |
| Dataset | `Salesforce/wikitext`, `wikitext-2-raw-v1` |
| Dataset revision | `b08601e04326c79dfdd32d625aee71d232d685c3` |
| Train slice | first 48 non-empty train rows with at least 160 characters |
| Validation slice | first 16 non-empty validation rows with at least 160 characters |
| Train SHA-256 | `098281a29883b8436de10847f3e702b1a02dc93644046d2d6209924bb98de7cf` |
| Validation SHA-256 | `e5da5b281ca3e5cedab84d7a49eb37223a90097b50e405f36b073f211c0e1044` |

The dataset text is not committed. The immutable source revision, deterministic selection rule, row counts, and hashes are sufficient to reconstruct and verify the slice.

## Hardware and run configuration

- Apple M2 Pro (`Mac14,10`), 16 GiB unified memory, macOS 26.2.
- PyTorch 2.10.0, MPS execution.
- Sequence length 32, batch size 1, 10 optimizer steps, gradient clipping 1.0.
- One routed expert, top-1 routing, latent size 64, one E3 latent step.
- Imported E2 parameters frozen.
- New-module LR `1e-5`; scale LR `1e-4`; augmentation epsilon `1e-5` after Gate 0B.
- Deterministic effort cycle: E0, E1, E3, E0, E1. Realized exposure: E0=4, E1=4, E3=2.
- Checkpoint serialization disabled for this smoke run to stay within local disk limits.
- Training loop wall time was 2.12 seconds after model construction/evaluation. End-to-end command time was 24.38 seconds.
- Measured maximum resident set was 7,567,933,440 bytes; macOS reported an 8,414,174,840-byte peak footprint.

An initial aggressive-LR attempt (`5e-4` for new modules, `1e-3` for scales) became non-finite on its second E3 update. The trainer now fails immediately on a non-finite loss or gradient norm. The published measurements use the stable configuration above.

## Retention gates

Gate 0A imported 100% of source parameters. HF-to-compat parity measured logit MAE `6.14e-6`, maximum absolute error `6.29e-5`, top-1 agreement `1.0`, and near-zero KL.

Gate 0B returned `PASS_EXACT`: CE difference `0.0`, logit MAE/max error `0.0`, top-1 agreement `1.0`, and exact source-backbone parameter identity.

## Per-effort result

| Mode | Layers | Normalized compute | CE before | CE after | CE change | Local latency |
|---|---:|---:|---:|---:|---:|---:|
| E0 | 12 | 0.5070 | 14.2846 | 14.2355 | -0.0491 | 22.75 ms |
| E1 | 18 | 0.7570 | 10.2233 | 10.2044 | -0.0189 | 27.22 ms |
| E2 | 24 | 1.0000 | 2.5122 | 2.5122 | 0.0000 | 31.32 ms |
| E3 | 24 + augmentation | 2.2567 | 2.6431 | 3.4515 | +0.8085 | 140.05 ms |

Latency is the mean of two warmed local MPS measurements and is descriptive, not a portable benchmark. Deterministic compute receipts—not latency—are the hard ordering gate.

## Interpretation

- E0 and E1 are genuinely cheaper. The implementation now answers the original architectural defect.
- E2 is a trustworthy capability anchor because its path and outputs remained unchanged through adaptation.
- E0/E1 are currently weak early exits. Their small CE improvements show the training path works, but ten examples are far too little to close the large regret against E2.
- E3 is currently dominated by E2: it costs 2.26× as much and performs worse. Its very large pre-clipping gradient norms indicate that its random augmentations need a longer, more controlled warm-up.
- A router would learn over unqualified choices, so counterfactual/oracle/policy work stops here.

## Detailed next workflow

1. **Reproduce the anchor.** Rebuild the two deterministic WikiText slices, verify both hashes, load the pinned model/tokenizer revision, and require Gate 0A plus `PASS_EXACT` Gate 0B.
2. **Train the shallow exits first.** Freeze imported parameters; train only E0/E1 continuation parameters with E2 distillation for enough tokens to observe a held-out curve. Keep E3 disabled during this warm-up.
3. **Warm E3 separately.** Start from tiny scales, train E3 primarily with task CE, record pre-clip gradient norms, and reject every non-finite run. Compare E3 against the untouched E2 anchor on a substantially larger validation set.
4. **Run joint adaptation only after stability.** Use explicit deterministic or logged sampling, retain the frozen/low-LR E2 anchor, and save examples, tokens, and optimizer steps by effort.
5. **Qualify each mode.** Require physical compute ordering, bounded E0/E1 regret at their compute savings, zero/bounded E2 retention drift, and positive E3 quality delta. Remove or disable dominated modes.
6. **Freeze the qualified base model.** Only then collect verified E0–E3 counterfactuals with receipt-backed compute and immutable task manifests.
7. **Apply the oracle gate.** Continue only if `LCB95(U_oracle - U_best_fixed) > 0` and oracle choices have meaningful diversity.
8. **Train and evaluate the hidden policy.** Compare it with best-fixed, prompt sham, effort-frequency random, compute-matched random, and oracle on untouched IID and family-held-out OOD splits.

## Reproduction command

```bash
python scripts/run_small_real_adaptation.py \
  --model Qwen/Qwen2.5-0.5B \
  --revision 060db6499f32faf8b98477b0a26969ef7d8b9987 \
  --train runs/wikitext2-smoke/train.jsonl \
  --validation runs/wikitext2-smoke/validation.jsonl \
  --output runs/qwen2.5-0.5b-wikitext2-stable \
  --steps 10 --seq-len 32 --eval-batches 2
```

The command intentionally does not claim that a ten-step smoke test is a trained model. It is a reproducible systems and qualification probe.

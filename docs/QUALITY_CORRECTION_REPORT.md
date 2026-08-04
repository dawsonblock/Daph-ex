# Real-model quality correction report

## Outcome

The failure mechanisms from the first Qwen2.5-0.5B smoke run were isolated and corrected.

**Engineering correction: PASS. Capability/router qualification: still pending.**

The corrected implementation preserves exact E2, keeps physical compute ordered, improves both shallow exits on the 16-record validation slice, and prevents E3 from degrading the anchor. E3 now records a small positive CE delta, but the improvement is too small and the validation set too limited to support a capability claim.

## Root causes

1. **E3 enabled too much random computation.** The old graph activated recurrent, routed-MoE, and latent modules in all 24 layers. Isolating the branches at scale `1e-5` showed that the recurrent branch alone moved CE from `2.77594` to `2.99719`; MoE and latent-only paths were near-neutral.
2. **E3 scale updates were unbounded.** An aggressive scale LR grew the raw latent scale to about `0.094`, produced multi-million pre-clipping gradient norms, and degraded validation CE.
3. **E0/E1 had severe representation mismatch.** Directly decoding intermediate states gave CE `14.28` at 12 layers, `10.03` at 18, `5.11` at 22, and `3.69` at 23 versus E2 at `2.78` in the depth-isolation probe.
4. **The first smoke schedule was far too short.** E0 and E1 received only four optimizer steps each at LR `1e-5`; their continuation adapters were effectively untrained.
5. **There was no candidate rollback.** A noisy short adaptation could make an exit worse and still be evaluated as the final model.

## Corrections

- Canonical E3 is now the full exact Qwen path plus one final-layer latent refinement. Experimental recurrent/MoE modules remain in the source tree but are not enabled by canonical E3.
- E3 receipts count the one refinement actually executed; normalized compute fell from `2.2567×` to `1.00894×` E2.
- Only `layers.23.latent_scale` is activated after Gate 0B for the 24-layer model.
- The effective latent residual is bounded with `0.01 * tanh(raw_scale / 0.01)`. Zero remains an exact no-op.
- E0/E1 support padding-aware hidden-state MSE in addition to causal CE and temperature-scaled E2 KL.
- The real-model harness uses measured 18/22-layer exits, trains E0 and E1 separately for 100 steps each, and gives E3 a separate 100-step task-loss phase.
- Each exit candidate is retained only if its validation CE improves. E3 refinement is retained only if it beats the E2 validation anchor.
- Non-finite loss and gradient norms remain hard failures; all optimizer steps retain gradient clipping.

## Final pinned run

Inputs remain pinned to:

- `Qwen/Qwen2.5-0.5B` revision `060db6499f32faf8b98477b0a26969ef7d8b9987`.
- `Salesforce/wikitext` revision `b08601e04326c79dfdd32d625aee71d232d685c3`, config `wikitext-2-raw-v1`.
- 48 deterministic training records and all 16 deterministic smoke-validation records.
- Sequence length 32, batch size 1, Apple M2 Pro MPS.

Gate 0A retained 100% parameter coverage. Gate 0B returned `PASS_EXACT` with zero CE/logit difference and exact backbone identity.

| Mode | Layers | Normalized compute | CE before | CE after | Change | Candidate |
|---|---:|---:|---:|---:|---:|---|
| E0 | 18 | 0.75702 | 10.15022 | 8.68212 | -1.46810 | accepted |
| E1 | 22 | 0.92369 | 5.85181 | 5.18359 | -0.66823 | accepted |
| E2 | 24 | 1.00000 | 3.34208 | 3.34208 | 0.00000 | exact anchor |
| E3 | 24 + 1 refinement | 1.00894 | 3.34210 | 3.34201 | -0.00010 | accepted |

Deterministic compute proves `E0 < E1 < E2 < E3`. Local latency was E0 `24.42 ms`, E1 `28.06 ms`, E2 `29.83 ms`, and E3 `29.14 ms`; the E2/E3 inversion is timer noise at this scale, which is why latency is not the hard ordering gate.

The command used about 6.66 GB maximum resident memory and an 8.41 GB peak macOS footprint. The complete local suite passes 97 tests.

## What remains

- E0 and E1 are improved but still have CE regret of `5.34` and `1.84` versus E2. They are not yet demonstrated as useful choices under a real task verifier.
- E3 improves CE by only `0.000076` versus E2. This is directionally correct but not statistically persuasive.
- Candidate selection and final reporting currently share the small smoke-validation slice. A serious experiment needs distinct selection and untouched qualification sets.
- No router should be trained until a larger corpus establishes bounded E0/E1 regret, a repeatable positive E3 delta, and a positive oracle LCB.

## Next qualification workflow

1. Build immutable train, selection, retention, and untouched qualification manifests.
2. Train E0/E1 continuations to convergence curves, testing multiple depth/compute points without touching E2.
3. Train final-layer E3 refinement with fixed or tightly bounded scales and select only on the selection split.
4. Re-run exact E2 retention and physical-compute gates.
5. Evaluate all four modes on untouched tasks with real correctness verifiers.
6. Remove any dominated mode, then freeze the base checkpoint.
7. Collect counterfactual utility and require `LCB95(U_oracle - U_best_fixed) > 0`.
8. Only after that gate, train the hidden router and compare it with fixed, sham, frequency-random, and compute-matched controls.

## Reproduction

```bash
python scripts/run_small_real_adaptation.py \
  --model Qwen/Qwen2.5-0.5B \
  --revision 060db6499f32faf8b98477b0a26969ef7d8b9987 \
  --train runs/wikitext2-smoke/train.jsonl \
  --validation runs/wikitext2-smoke/validation.jsonl \
  --output runs/qwen2.5-0.5b-wikitext2-final-selected \
  --exit-steps 100 --e3-steps 100 \
  --e0-layers 18 --e1-layers 22 \
  --seq-len 32 --eval-batches 16
```

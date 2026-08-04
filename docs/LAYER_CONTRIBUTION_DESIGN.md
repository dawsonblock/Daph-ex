# Layer-contribution design

## Research basis and boundary

The attached paper, *Is One Layer Enough? Training a Single Transformer Layer Can Match Full-Parameter RL Training* (arXiv:2607.01232v2), reports non-uniform single-layer RL adaptation gains across Qwen3 and Qwen2.5 models and math, code, and agentic tasks. Its strongest regions tend to occur around 40%–60% depth. It defines contribution as:

`C(k) = (S_k - S_base) / (S_full - S_base)`.

This is a structural prior only. It does not prove that DAPH’s exact checkpoint has the same ranking, nor that recursively applying a high-contribution layer improves inference.

## Implementation

`daph/layer_contribution.py` evaluates the base model, obtains or measures a full-training reference, restores the exact base state, freezes every parameter except one imported transformer block, adapts that block, evaluates it, and records signed contribution. Negative values are retained.

Profiles can be:

- `full`: every layer; report label `FULL_PROFILE`.
- `sparse`: 0%, 20%, 30%, 40%, 45%, 50%, 55%, 60%, 70%, 80%, and 100%, mapped to legal unique indices.
- `middle_only`: every layer in the 30%–70% interval.
- explicit: a predeclared index list.

Partial scans are always labeled `PARTIAL_PROFILE` and never presented as an exact global ranking. Reports include ranking, best available contiguous region, quartile means, 40%–60% mean, depth correlation, checkpoint/config/profile digests, and CSV/JSONL plot data.

`LayerAdaptationObjective` distinguishes supervised CE, explicitly verified reward, and an external callback. The bundled CLI implements CE only and labels it `negative_causal_ce`; it does not relabel CE as RLVR. GRPO/other RLVR implementations plug in through the callback contract after their own dependency and reproducibility qualification.

## Stop rule

If a sufficiently powered profile shows no meaningful layer variation, stop profile-guided E3 promotion. If a partial scan is flat, densify only if its uncertainty and budget justify doing so; do not manufacture a middle-layer conclusion.

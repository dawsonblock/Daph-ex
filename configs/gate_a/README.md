# Gate A frozen configuration

`gate_a_v2_frozen.json` is a byte-identical copy of
`configs/gate_a_qualification.json` (sha256
`07f7a4b34f9c6bf43666da5256c40ca9dca430f51e305326bfe68722d106bdde`, the exact
digest pinned in `evidence/gate_a/protocol_manifest_v2.json`). It exists so the
canonical frozen protocol has a stable, discoverable home; the original file is
kept untouched because historical manifests reference its path.

## Why `prompt_condition: direct` (not the older `synth,cot` default)

The run script's historical default was `synth,cot`. Condition selection for
Gate A was performed on development data only (25-task smoke corpus, seed 1101
— never qualification data):

| condition | B3−B0 | mean latency | mean completion tokens | output shape |
|---|---|---|---|---|
| `direct` | **1.00** | 1.1 s | 7.4 | terse `NNN<|box_end|>` |
| `cot` | 0.80 | 9.6 s | 44.1 | rambling self-questioning |
| `synth,cot` | 0.80 | 9.1 s | 52.7 | verbose synthetic-CoT |

`direct` dominated every axis, so it was frozen before the pilot and enforced
during qualification via `--frozen-config` (the run aborts on drift).
Comparison receipts: `evidence/hrm_smoke_v2_{direct,cot,synthcot}/`.

Historical Gate A evidence is immutable; nothing in this directory rewrites it.

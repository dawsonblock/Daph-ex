# DAPH / ExFusion v3 Design Brief

## Pretrained path (canonical)

`QwenCompatModel` preserves the imported Qwen residual graph. `augment_qwen_compat_model()` copies those parameters exactly into the `.base` submodules of `QwenExFusionModel`; exact names and digests are retained as optimizer/checkpoint provenance. Gate 0B evaluates E2 only and accepts `PASS_EXACT` or `FAIL`.

E0 and E1 physically stop after deterministic layer prefixes. Skipped blocks are not called. Their shallow states use the shared final RMSNorm and LM head, forming the baseline that is trained against E2 with task CE plus temperature-scaled KL. E2 executes all base blocks with every augmentation disabled. E3 executes the full base and then enabled additions; refinement is applied as `base + scale × (refined − base)`.

Compute receipts count executed/skipped blocks, attention and FFN calls, recurrent operations, active routed experts, latent iterations, tokens, raw compute units, and E2-normalized cost. Latency and peak memory are optional benchmark fields, never the sole unit-test gate.

The canonical order is `E0 < E1 < E2 < E3`. Tiny one/two-layer unit fixtures can test parity but cannot qualify a three-depth hierarchy.

## Philosophy

Complementary operators instead of exclusive macro-routing:

| Dimension        | Operator                          | Role                              |
|------------------|-----------------------------------|-----------------------------------|
| Sequence mixing  | SelectiveSSM *or* KDA + periodic global attention | continuous state + global retrieval |
| Width mixing     | LatentMoE (latent + RMSNorm + SiTU/QB) | specialization                    |
| Depth mixing     | BlockAttnRes / AttnResBank        | selective access to earlier blocks|
| Compute budget   | EffortController + cost loss      | how much extra work is justified  |
| Memory / evidence| RFSN hooks + ImmutableVaultSink   | bi-temporal evidence, salience    |

## Implemented

1. LatentMoE + SiTU-GLU + Quantile Balancing
2. ChannelGate
3. EffortController + cost-aware loss + early-exit
4. HybridBlock (SSM or KDA + attn + MoE + gates + exit)
5. SelectiveSSM + KDA
6. BlockAttnRes / AttnResBank
7. DAPHHybridModelV3
8. DARE → TIES → Fisher merge
9. RFSN hooks (Emitter, InMemory sink, Protocol)
10. **ImmutableVaultSink** — append-only, bi-temporal as-of, supersede, salience decay, content-hash integrity
11. **Benchmarks** — HybridBlock / Model latency & throughput (SSM vs KDA)
12. pyproject.toml packaging

## RFSN vault usage

```python
from daph import ImmutableVaultSink, ExFusionEmitter, DAPHHybridModelV3

vault = ImmutableVaultSink(salience_half_life_s=3600)
emitter = ExFusionEmitter(sink=vault, sequence_id="run-1")
out = model(input_ids, emitter=emitter)

vault.verify_integrity()
current = vault.as_of()                    # bi-temporal current slice
vault.supersede(current[0].event.event_id) # soft-delete
vault.decay_salience()
```

## Benchmarks

```python
from daph import run_quick_suite, print_suite
print_suite(run_quick_suite("cpu"))
```

## Still open

- Disk/SQLite content-addressed vault backend
- Chunk-parallel KDA kernels
- Expanded CI / larger-scale benchmarks

# HRM External Memory + Adaptive Compute

> Superseded for implementation by
> [`HRM_ADAPTIVE_MEMORY_CONTROL_PLANE.md`](HRM_ADAPTIVE_MEMORY_CONTROL_PLANE.md).
> This document preserves the original v3.5 foundation protocol.

## Scope and claim discipline

This branch adds a standalone research path around the native
`sapientinc/HRM-Text-1B` checkpoint. It does not modify GGUF files and does not
claim that external memory, iterative retrieval, variable recurrence, or a
learned controller has improved the model.

The primary hypothesis is:

> A 4096-token recurrent HRM can outperform its native-context baseline on
> long-context tasks by selectively retrieving and repeatedly reasoning over
> external evidence.

The later controller hypothesis is:

`a* = argmax_a E[delta utility(a) | h_t] - cost(a)`

with actions `ANSWER`, `RETRIEVE`, `VERIFY`, `CONTINUE`, and `STOP`. Learned
control remains blocked until oracle-context, iterative-RAG, counterfactual
opportunity, sham-control, and OOD gates pass.

## Pinned native model

The canonical research checkpoint is:

- model: `sapientinc/HRM-Text-1B`
- revision: `9f082d68b8cd0ebc56e33f1c88c45609174c272c`
- native class: `HrmTextForCausalLM`
- minimum Transformers version: `5.9.0`
- hidden size: `1536`
- layers per H/L stack: `16`
- recurrence: `H_cycles=2`, `L_cycles=3`
- context: `4096`
- PrefixLM: enabled

Inference must pass `token_type_ids=1` over prompt positions. Omitting this
falls back to causal attention and does not match the checkpoint's training
distribution. Direct NLP tasks use the `direct` condition; reasoning tasks use
the composite `synth,cot` condition. The checkpoint is pre-alignment, not a
finished chat assistant.

Primary references:

- <https://huggingface.co/sapientinc/HRM-Text-1B>
- <https://github.com/sapientinc/HRM-Text>
- <https://huggingface.co/docs/transformers/main/model_doc/hrm_text>

## Implemented now

### Stage A — untouched HRM baseline

`hrm_adaptive_memory.hrm.HRMAdapter` loads the pinned native checkpoint lazily, verifies
its configuration, renders official condition prefixes, and creates the
PrefixLM mask. The baseline command records prompt/completion tokens, latency,
task family, difficulty, model revision, and raw output.

```bash
pip install -e '.[hrm]'
python scripts/run_hrm_baseline.py \
  --tasks tasks.jsonl \
  --condition synth,cot \
  --baseline-condition B0_NO_CONTEXT \
  --output runs/hrm-baseline/results.jsonl
```

The HRM prompt condition (`direct`, `cot`, or `synth,cot`) and the experimental
context condition (`B0` through `B3`) are separate receipt fields. Run the same
immutable task IDs once per B0/B1/B2/B3 condition, changing only the supplied
context. Every task requires an `expected` field; the baseline command records
normalized exact match and a dataset digest beside the raw generations.

### Stage B — external-memory substrate

The package implements:

- immutable source, semantic, and episodic JSONL stores;
- current/superseded/contradicted/uncertain memory status;
- structural prose, Python code, experiment, and conversation chunking;
- exact provenance on every chunk;
- independent BM25 and dense interfaces;
- deterministic hashing dense retrieval as a dependency-free baseline only;
- Reciprocal Rank Fusion;
- reranker protocol and cheap lexical sham/baseline;
- Recall@K, MRR, nDCG, and all-required-evidence recall;
- MMR-like token-budgeted evidence selection;
- a structured PrefixLM working-memory packet.

```bash
python scripts/build_hrm_memory.py docs/*.md daph/*.py \
  --output runs/hrm-memory/source_chunks.jsonl
python scripts/run_hrm_memory_smoke.py --output runs/hrm-memory-smoke
```

For a scientific run, replace `HashingEmbedder` and
`LexicalOverlapReranker` with declared, revision-pinned dense and cross-encoder
models. The cheap implementations are explicit baselines, not evidence that
semantic retrieval works.

### Stage C — mandatory oracle-context gate

Every task must be evaluated under paired conditions:

| Condition | Context |
|---|---|
| B0 | none |
| B1 | random |
| B2 | naive retrieval |
| B3 | manually verified oracle evidence |

```bash
python scripts/qualify_hrm_oracle_context.py \
  --results runs/hrm-oracle/paired_results.jsonl \
  --minimum-tasks 100 \
  --output runs/hrm-oracle/gate.json
```

If B3 does not beat B0 by the predeclared margin, structured RAG is not the next
bottleneck. The next step is retrieval-conditioned adaptation or evidence-use
diagnosis. Controller training remains false in either case.

## Research hooks, not qualified systems

`HRMRecurrentTracer` observes reused native `L_module` and `H_module` calls and
stores compact last-token/mean/RMS summaries. It does not retain complete
sequence tensors. The expected released trace order is:

`L1, L2, L3, H1, L4, L5, L6, H2`.

The recurrence ablation declares `H1L3`, pretrained `H2L3`, `H3L3`, and
`H4L3`. Arbitrarily changing the checkpoint schedule is not claimed safe;
extra-cycle arms require measurement and likely variable-depth continuation
training.

The counterfactual collector deep-copies a reachable decision state, executes
each available action independently, and records actual utility relative to
STOP/ANSWER. `UtilityController` refuses adaptive execution without
`VERIFIED_FIT`, except for an explicit research override.

## Gates and build order

1. A: untouched native baseline.
2. B: hybrid retrieval plumbing.
3. C: oracle-context diagnostic. Stop if HRM cannot use perfect evidence.
4. D: structured one-shot RAG.
5. E: retrieval-conditioned SFT if required.
6. F: iterative retrieval. Require utility above one-shot RAG.
7. G: isolated counterfactual action execution.
8. H: learned two-action ANSWER/RETRIEVE controller. Require wins over sham and heuristic controls.
9. I: recurrence ablations and variable-depth training.
10. J: unified value-of-computation controller.
11. K: textual-plus-latent memory slots only after the textual ceiling is known.

## Evidence contract

Every real run should record:

- source-tree commit and dirty status;
- model ID and immutable revision;
- tokenizer revision;
- task IDs and immutable split digests;
- condition prefix and `token_type_ids` policy;
- source/chunk/retrieval/reranker versions;
- retrieved IDs, scores, and packed evidence order;
- oracle evidence IDs;
- answer, verifier result, utility, tokens, cycles, latency, and peak memory;
- recurrent trace schema/version when enabled;
- action executor versions for counterfactual runs.

Do not train a controller from task-family or human difficulty labels. Labels
must come from executed state/action outcomes. Do not call a smoke report a
capability result, and do not call a fixed recurrence ablation adaptive compute.

# HRM Adaptive Memory Control Plane

## Implemented boundary

Version 3.6.1 implements the control plane and the first decisive experiment. It
does not merge or vendor any uploaded repository. DAPH owns HRM execution,
context construction, verification, experimental receipts, counterfactual
utility, and future executive decisions.

The operational memory architecture is deliberately limited to three layers:

1. immutable sources and DAPH episode receipts;
2. BM25 controls plus optional TurboVec/RuVector retrieval and a derived Graphiti temporal graph, after their gates;
3. off-path Infini-style topic consolidation recorded as Markdown/Git history.

Source, episodic, semantic, procedural, and consolidated memory remain logical
types, not five physical databases. AgentDB concepts are reserved for a later
procedural arm. PixelRAG remains a late source-modality adapter. LongMemEval is
external evaluation, not the mechanism-discovery dataset.

The current implementation stops at Gate A. It includes:

- canonical `hrm_adaptive_memory` imports and one-release `hrm_memory` aliases;
- asynchronous backend contracts and immutable receipts;
- legal memory lifecycle transitions and provider-neutral cached derivations;
- BM25, hash-vector, and hybrid controls behind one retrieval contract;
- a fail-closed, loopback-only RuVector client and a disabled TurboVec source lock;
- audited source hashes in `third_party/sources.lock.json`;
- a native HRM B0/B1/B2/B3 runner;
- grouped, tier-enforced Gate A qualification.

All external runtimes are disabled in the source lock. A passing Gate A report
is necessary but not sufficient to activate RuVector: a later integration
change must also add a tested immutable runtime image digest. There is no
silent fallback.

## Oracle dataset contract

The task JSONL is independent experimental truth:

```json
{
  "task_id": "task-001",
  "question": "Which configuration produced result R7?",
  "answer": "config-v4",
  "required_evidence_ids": ["doc-4#result-r7"],
  "oracle_evidence_ids": ["doc-4#result-r7"],
  "family": "single-hop",
  "template_id": "result-config-v1",
  "source_cluster_id": "run-notes-2026-08",
  "split": "test",
  "verifier": "exact"
}
```

The immutable evidence JSONL contains `evidence_id`, `source_id`, `content`,
and optional `source_type` and `metadata`. The retrieval backend never creates
or edits the task file.

The runner constructs:

- B0 with no external evidence;
- B1 with deterministic irrelevant evidence, excluding oracle/required IDs and
  exactly matching B3's evidence-token count;
- optional B1b with answer-free, token-matched lexical hard distractors;
- B2 from the selected retrieval backend;
- B3 from the independently declared oracle IDs.

It aborts if the context budget is exceeded, B1 cannot be matched, evidence is
missing, task IDs repeat, or B0 and B3 prompt hashes are equal. The model never
sees an experiment-arm label, evidence ID, or source ID; those are receipt
metadata only. It sees neutral evidence slots and the evidence content.

Two different studies remain separate:

- `CAPABILITY_USE` asks every arm to answer, isolating whether oracle evidence
  improves verified task quality. This is the only mode that can pass Gate A.
- `EVIDENCE_GROUNDED` permits `INSUFFICIENT_EVIDENCE`; it evaluates grounded
  abstention and is explicitly non-promotable.

Every run writes the final prompt and prompt hash in its per-task JSONL, plus
separate environment, backend-capability, model, and dataset manifests. Gate A
writes the grouped bootstrap statistics and machine-readable promotion decision
as a separate immutable report.

## Commands

Install the native HRM dependency only for real execution:

```bash
python -m pip install -e '.[dev,hrm]'
```

Run a non-promotable 25-task smoke:

```bash
python scripts/build_hrm_controlled_gate_a_dataset.py \
  --tasks-per-family 5 \
  --output /tmp/hrm_context_smoke_data

python scripts/run_hrm_context_study.py \
  --tasks /tmp/hrm_context_smoke_data/oracle_tasks.jsonl \
  --evidence /tmp/hrm_context_smoke_data/evidence.jsonl \
  --tier SMOKE \
  --retriever bm25 \
  --output evidence/hrm_context_smoke_v1
```

Build the committed controlled corpus from its deterministic generator when an
independent rebuild is needed. It tests synthetic post-checkpoint facts and
multi-document integration; it is not a natural-memory benchmark.

```bash
python scripts/build_hrm_controlled_gate_a_dataset.py \
  --output /tmp/hrm-controlled-gate-a-rebuild
```

Run the predeclared qualification with at least 500 tasks and five independent
template groups:

```bash
python scripts/run_hrm_context_study.py \
  --tasks data/hrm/controlled_gate_a_v1/oracle_tasks.jsonl \
  --evidence data/hrm/controlled_gate_a_v1/evidence.jsonl \
  --tier QUALIFICATION \
  --retriever bm25 \
  --include-hard-distractor \
  --output evidence/hrm_context_qualification_v1

python scripts/qualify_hrm_context_gate_a.py \
  --results evidence/hrm_context_qualification_v1/per_task_results.jsonl \
  --tasks data/hrm/controlled_gate_a_v1/oracle_tasks.jsonl \
  --tier QUALIFICATION \
  --require-hard-distractor \
  --output evidence/hrm_context_qualification_v1/gate_a.json
```

Gate A passes only when all four primary conditions are paired (and B1b is
present for every task when selected),
the study has at least 500 tasks and five groups in each of `template_id`,
`family`, and `source_cluster_id`, the mean B3−B0 quality gain is at least
0.05, and every grouped-bootstrap 95% lower bound is positive. The most
conservative clustering result is recorded as the primary qualification view.

TurboVec is recorded as an audited, disabled compressed-dense backend. After
Gate A it may be compared as a low-cost dense arm underneath `RetrievalBackend`,
especially where a graph/time/provenance layer supplies an allowlist. It does
not replace RuVector or add a second memory system.

## Locked next stages

If Gate A fails, the next work is retrieval-conditioned evidence-use training;
do not add memory infrastructure. If it passes, the next branch may:

1. build and pin TurboVec and RuVector integration artifacts;
2. compare BM25/hash controls with TurboVec dense and RuVector retrieval;
3. require receipt-backed Gate B and Gate C success;
4. only then add Graphiti temporal memory;
5. compare one-pass and deterministic two-pass retrieval;
6. instrument resumable H/L recurrence and measure recurrence opportunity;
7. collect ANSWER/RETRIEVE counterfactuals before training an executive.

Infini-style consolidation, procedural priors, PixelRAG, and unified adaptive
compute remain later work. No action is enabled merely because its interface
exists.

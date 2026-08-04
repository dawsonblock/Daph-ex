# Controlled Gate A corpus v1

This committed dataset is a deterministic controlled benchmark for the first
HRM oracle-context question: can the pinned model use supplied evidence that it
could not have memorized during pretraining?

It contains 500 numeric-verification tasks across five families:

- single-hop synthetic facts;
- two-hop registry composition;
- numeric derivation from two records;
- temporal supersession;
- distractor-heavy allocation records.

`oracle_tasks.jsonl` contains the independent required/oracle evidence IDs,
template IDs, family labels, and source-cluster IDs. `evidence.jsonl` contains
immutable controlled source records. `dataset_manifest.json` contains the seed
and canonical content hashes. Rebuild with
`scripts/build_hrm_controlled_gate_a_dataset.py` into a new directory and
compare the manifest hashes.

This is a controlled synthetic evidence-use benchmark only. It cannot support a
claim about natural long-term memory, open-domain retrieval, or LongMemEval.

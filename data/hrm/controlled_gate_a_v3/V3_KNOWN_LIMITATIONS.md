# controlled_gate_a_v3 — known limitations

Four defects were found in this corpus **after** its qualification and OOD
splits were evaluated. Per the standing rule, the affected evaluations are
**voided as quantitative measurements** rather than adjusted, and the corpus
and its receipts are retained unmodified as historical evidence.

Each defect below was verified against the committed corpus, not merely
asserted.

## 1. Source-style holdout is violated at the evidence level

`_build_chain` selected the second hop's style from the **global** style tuple:

```python
second_style = SOURCE_STYLES[(SOURCE_STYLES.index(style) + 1) % len(SOURCE_STYLES)]
```

and the temporal builders hard-coded `change_log`. Neither respects the split's
allowed style set. Measured evidence-record styles:

| split | styles present in evidence |
|---|---|
| qualification | formal_registry 325, change_log 525, key_value_log 372, technical_note 450, **message 78**, **table_text 216** |
| OOD | table_text 325, message 336, **change_log 250**, **formal_registry 72** |

Overlap is `{table_text, message, change_log, formal_registry}` — the holdout is
violated in **both** directions.

**Why the test missed it.** `test_ood_split_holds_out_whole_styles_and_naming_regimes`
compared `task["metadata"]["source_style"]`, which records only the *first*
record's style. It asserted a property of task metadata while claiming a
property of the corpus. A test that reads labels instead of content can pass
while the thing it names is false.

## 2. The alias regime does not produce aliases

```python
return Entity(latent, regime, surface, f"{noun} {role[0].upper()}{role[1]}")
```

For `Nimbus assembly` this yields the query surface `Nimbus As` — a
two-character truncation, not an alias. The regime tests prefix-truncation
robustness, which is not a capability anyone wants.

## 3. Description-regime tasks are not answerable from evidence

**0 of 120** OOD description tasks have the question's subject appearing
anywhere in their own evidence. A question asks about *"the auxiliary unit
listed in the intake record"* while its evidence says *"Beacon controller"*,
with **no record linking the two**. Solving these requires recovering a hidden
generator mapping, which is not an information-retrieval problem.

**48% of the OOD split (120 of 250) is therefore unanswerable by any
retriever.** The OOD `oracle_evidence` arm still scores 0.764 because it
bypasses retrieval and hands over the required record directly — which is why
the 0.080 → 0.764 gap looked so dramatic.

## 4. The `oracle_bridge` arm is not an oracle

It re-derives the bridge by running the same `extract_entities` under test, so
it fails identically wherever extraction fails. On OOD it fired zero follow-ups
and scored identically to the deterministic arm.

## What survives and what does not

**Survives** — verifiable from code and corpus text alone, independent of task
solvability:

- `ENTITY_PATTERN` requires a hyphen-digit suffix and matches **0 of 250** OOD
  questions and **0 of 160** `natural_name` questions inside qualification.
- Consequently **zero follow-ups fired on all 250 OOD tasks**.
- The mechanism performs lexical identifier chaining, not bridge inference.

**Survives** — the qualification decomposition. Qualification contains no
description regime, and question subjects appear in their own evidence for
~81% of tasks across all three of its regimes (the remainder is an artifact of
the audit's subject-extraction regex, not the corpus). Defect 1 affects the
*holdout claim*, not qualification's internal validity.

**Voided** — as quantitative measurements:

- Every OOD number in `evidence/gate_c/v3_ood/`.
- The claim that qualification and OOD share no source style.
- The alias regime as an alias-resolution test.
- The OOD bridge/retrieval split.

## Consequence

`controlled_gate_a_v3` is retained as historical evidence for the qualitative
finding only. `controlled_gate_a_v4` supersedes it, with split-local style
selection enforced at the evidence-record level, genuine aliases carrying
explicit alias evidence, inferable description identities, evaluator-only proof
graphs, and an oracle ladder built from generator metadata rather than from the
extractor under test.

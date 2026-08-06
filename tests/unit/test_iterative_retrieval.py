"""Sprint 2: evidence state, sufficiency, bounded two-pass retrieval, calculator."""

from __future__ import annotations

import asyncio

import pytest

from hrm_adaptive_memory.actions.calculate import (
    UnsafeExpression,
    calculate_from_evidence,
    safe_eval,
)
from hrm_adaptive_memory.backends import CanonicalRetrievalBackend, CanonicalRetrievalMode
from hrm_adaptive_memory.contracts import IndexRecord
from hrm_adaptive_memory.evidence.packing import select_evidence
from hrm_adaptive_memory.evidence.state import EvidenceRecordView, build_evidence_state
from hrm_adaptive_memory.evidence.sufficiency import SufficiencyVerdict, assess
from hrm_adaptive_memory.retrieval.iterative import TwoPassRetriever


def run(value):
    return asyncio.run(value)


def view(evidence_id: str, content: str, rank: int) -> EvidenceRecordView:
    from hrm_adaptive_memory.evidence.state import extract_entities, extract_numbers
    return EvidenceRecordView(
        evidence_id=evidence_id, source_id=f"src-{evidence_id}", content=content,
        token_count=len(content.split()), rank=rank,
        entities=extract_entities(content), numbers=extract_numbers(content),
    )


HOP1 = "The deployment record for Trial-000-867 lists Adapter-78103 as its adapter."
HOP2 = "The classification registry maps Adapter-78103 to category code 840."
QUESTION = "Which category code applies to Trial-000-867?"


# ---- evidence state -------------------------------------------------------

def test_first_pass_names_the_unresolved_bridge_entity():
    state = build_evidence_state(question=QUESTION, records=[view("a", HOP1, 1)])
    assert state.required_entities == ("Trial-000-867",)
    assert "Adapter-78103" in state.observed_entities
    assert state.bridge_entities == ("Adapter-78103",)
    assert state.missing_entities == ()
    assert state.entity_coverage == 1.0

    report = assess(state)
    assert report.verdict == SufficiencyVerdict.MISSING_BRIDGE
    assert report.followup_terms == ("Adapter-78103",)
    assert report.needs_followup


def test_resolved_bridge_becomes_sufficient():
    state = build_evidence_state(
        question=QUESTION, records=[view("a", HOP1, 1), view("b", HOP2, 2)],
    )
    assert state.bridge_entities == ()
    report = assess(state)
    assert report.verdict == SufficiencyVerdict.SUFFICIENT
    assert not report.needs_followup


def test_missing_subject_is_distinct_from_missing_bridge():
    state = build_evidence_state(
        question=QUESTION, records=[view("z", "An unrelated note about Station-004-226.", 1)],
    )
    report = assess(state)
    assert report.verdict == SufficiencyVerdict.MISSING_SUBJECT
    assert report.followup_terms == ("Trial-000-867",)


def test_empty_evidence_is_reported_explicitly():
    report = assess(build_evidence_state(question=QUESTION, records=[]))
    assert report.verdict == SufficiencyVerdict.EMPTY


def test_conflicting_bindings_are_surfaced_not_silently_resolved():
    state = build_evidence_state(question="What setting does Service-1-2 use?", records=[
        view("cur", "Release 9 supersedes earlier settings: Service-1-2 now uses setting 875.", 1),
        view("old", "Release 8 recorded setting 205 for Service-1-2; superseded.", 2),
    ])
    assert state.contradictions
    assert assess(state).verdict == SufficiencyVerdict.CONFLICTING


def test_sufficiency_report_serializes_with_its_gap():
    row = assess(build_evidence_state(question=QUESTION, records=[view("a", HOP1, 1)])).to_dict()
    assert row["verdict"] == "MISSING_BRIDGE"
    assert row["needs_followup"] is True
    assert row["missing_information"]


# ---- packing / near-duplicate suppression ---------------------------------

def test_entity_anchoring_drops_confusable_lookalikes():
    """The Gate B failure mode: same-template records with different entities."""

    records = [
        view("keep-1", HOP1, 1),
        view("keep-2", HOP2, 2),
        view("noise-1", "The classification registry maps Adapter-11111 to category code 111.", 3),
        view("noise-2", "The classification registry maps Adapter-22222 to category code 222.", 4),
    ]
    selected, receipt = select_evidence(
        records, anchor_entities=("Trial-000-867", "Adapter-78103"), token_budget=500,
    )
    assert [row.evidence_id for row in selected] == ["keep-1", "keep-2"]
    assert set(receipt.dropped_unanchored_ids) == {"noise-1", "noise-2"}


def test_selection_respects_token_budget_and_reports_drops():
    records = [view(f"r{i}", f"Record {i} about Trial-000-867 with padding words here.", i)
               for i in range(1, 6)]
    selected, receipt = select_evidence(
        records, anchor_entities=("Trial-000-867",), token_budget=18, lambda_redundancy=0.0,
    )
    assert receipt.selected_tokens <= 18
    assert receipt.dropped_over_budget_ids
    assert len(selected) < len(records)


def test_duplicate_records_are_suppressed():
    records = [view("a", HOP1, 1), view("a", HOP1, 2), view("b", HOP2, 3)]
    selected, receipt = select_evidence(records, anchor_entities=("Trial-000-867", "Adapter-78103"))
    assert [row.evidence_id for row in selected] == ["a", "b"]
    assert "a" in receipt.dropped_redundant_ids


def test_anchoring_can_be_disabled_for_ablation():
    records = [view("keep", HOP1, 1), view("other", "Note about Station-1-2 only.", 2)]
    selected, _ = select_evidence(
        records, anchor_entities=("Trial-000-867",), enforce_anchoring=False,
    )
    assert len(selected) == 2, "ablation arm must be able to keep unanchored records"


# ---- bounded two-pass retrieval -------------------------------------------

def corpus_records():
    rows = [
        IndexRecord(evidence_id="hop-1", source_id="s1", content=HOP1, token_count=12),
        IndexRecord(evidence_id="hop-2", source_id="s2", content=HOP2, token_count=10),
    ]
    for index in range(8):
        rows.append(IndexRecord(
            evidence_id=f"decoy-{index}", source_id=f"d{index}", token_count=10,
            content=f"The classification registry maps Adapter-9000{index} to category code 10{index}.",
        ))
    return rows


def test_two_pass_retrieval_recovers_the_second_hop():
    backend = CanonicalRetrievalBackend(CanonicalRetrievalMode.BM25, corpus_records())
    one_pass = run(TwoPassRetriever(backend, k=5, max_passes=1).retrieve(QUESTION))
    assert "hop-2" not in one_pass.receipt.selected_ids

    two_pass = run(TwoPassRetriever(backend, k=5, followup_k=5).retrieve(QUESTION))
    assert two_pass.receipt.passes == 2
    assert two_pass.receipt.followup_query == "Adapter-78103"
    assert set(two_pass.receipt.selected_ids) >= {"hop-1", "hop-2"}
    assert two_pass.report.verdict == SufficiencyVerdict.SUFFICIENT


def test_two_pass_does_not_fire_when_evidence_is_already_sufficient():
    backend = CanonicalRetrievalBackend(CanonicalRetrievalMode.BM25, corpus_records())
    result = run(TwoPassRetriever(backend, k=5).retrieve("Adapter-78103 category code"))
    assert result.receipt.passes == 1
    assert result.receipt.followup_query is None
    assert result.receipt.retrieval_calls == 1


def test_retrieval_depth_is_bounded_at_two():
    backend = CanonicalRetrievalBackend(CanonicalRetrievalMode.BM25, corpus_records())
    result = run(TwoPassRetriever(backend, k=5).retrieve(QUESTION))
    assert result.receipt.passes <= 2
    assert result.receipt.retrieval_calls <= 2
    with pytest.raises(ValueError, match="at most two passes"):
        TwoPassRetriever(backend, max_passes=3)


def test_merge_deduplicates_and_reranks_across_passes():
    backend = CanonicalRetrievalBackend(CanonicalRetrievalMode.BM25, corpus_records())
    result = run(TwoPassRetriever(backend, k=5, followup_k=5).retrieve(QUESTION))
    merged = result.receipt.merged_ids
    assert len(merged) == len(set(merged)), "merge must deduplicate"
    assert result.receipt.selection is not None


# ---- safe calculator ------------------------------------------------------

@pytest.mark.parametrize("expression,expected", [
    ("2 + 3", 5.0), ("(32) * (7)", 224.0), ("10 / 4", 2.5),
    ("7 % 3", 1.0), ("2 ** 8", 256.0), ("-(4) + 10", 6.0),
])
def test_safe_eval_computes_permitted_arithmetic(expression, expected):
    assert safe_eval(expression) == expected


@pytest.mark.parametrize("expression", [
    "__import__('os').system('ls')",
    "open('/etc/passwd').read()",
    "(1).__class__",
    "[x for x in range(10)]",
    "lambda: 1",
    "2 ** 999999",
    "1 / 0",
    "True + 1",
    "x + 1",
])
def test_safe_eval_rejects_everything_outside_the_arithmetic_subset(expression):
    with pytest.raises(UnsafeExpression):
        safe_eval(expression)


def test_calculation_from_evidence_produces_a_verified_receipt():
    receipt = calculate_from_evidence([
        {"evidence_id": "u", "content": "The base ledger records 32 units for Plan-000-965."},
        {"evidence_id": "m", "content": "The operating rule for Plan-000-965 multiplies its units by 7."},
    ])
    assert receipt is not None
    assert receipt.result == "224"
    assert receipt.operation == "*"
    assert receipt.operands == ("32", "7")
    assert receipt.source_evidence_ids == ("u", "m")
    assert receipt.verified


def test_calculation_returns_none_rather_than_guessing():
    assert calculate_from_evidence([
        {"evidence_id": "a", "content": "No operation is stated here, only 42."},
    ]) is None
    assert calculate_from_evidence([
        {"evidence_id": "a", "content": "It multiplies things but states no numbers."},
    ]) is None
    # Ambiguous operand count is refused rather than silently truncated.
    assert calculate_from_evidence([
        {"evidence_id": "a", "content": "Multiplies 2 and 3 and 4 and 5."},
    ]) is None


def test_composer_is_byte_identical_to_the_gate_a_composer():
    """Sprint 2 arms must stay comparable to frozen Gate A/Gate B numbers."""

    import asyncio as _asyncio

    from hrm_adaptive_memory.evidence.packing import compose_evidence_prompt
    from hrm_adaptive_memory.experiments.context_study import (
        ContextConstructor, ContextStudyConfig, EvidenceCorpus, ExperimentTier,
        OracleTask, StudyCondition,
    )

    records = corpus_records()
    task = OracleTask(
        task_id="t", question=QUESTION, answer="840",
        required_evidence_ids=("hop-1", "hop-2"), oracle_evidence_ids=("hop-1", "hop-2"),
        family="two_hop", template_id="tpl", source_cluster_id="cl", split="test",
        verifier="numeric",
    )
    constructor = ContextConstructor(
        EvidenceCorpus(records),
        CanonicalRetrievalBackend(CanonicalRetrievalMode.BM25, records),
        config=ContextStudyConfig(tier=ExperimentTier.SMOKE),
    )
    gate_a = _asyncio.run(constructor.construct(task, StudyCondition.B3_ORACLE_EVIDENCE))
    mine = compose_evidence_prompt(QUESTION, [HOP1, HOP2])
    assert mine == gate_a.prompt

    empty_a = _asyncio.run(constructor.construct(task, StudyCondition.B0_NO_CONTEXT))
    assert compose_evidence_prompt(QUESTION, []) == empty_a.prompt


def test_second_hop_survives_packing_when_pass_one_already_found_it():
    """Regression: the packer silently dropped an already-retrieved second hop.

    The record that resolves a bridge names the bridge, not the question's
    subject, so anchoring on question entities alone discarded it — costing
    9 of 500 tasks in the first Sprint 2 run while reporting the failure as
    "bridge not detected".
    """

    records = corpus_records()
    backend = CanonicalRetrievalBackend(CanonicalRetrievalMode.BM25, records)
    # k large enough that pass one already returns both hops.
    result = run(TwoPassRetriever(backend, k=10, followup_k=10).retrieve(QUESTION))
    assert set(result.receipt.selected_ids) >= {"hop-1", "hop-2"}, (
        "second hop was retrieved but dropped during selection"
    )
    assert result.report.verdict == SufficiencyVerdict.SUFFICIENT


def test_linked_entities_are_exposed_for_anchoring():
    state = build_evidence_state(
        question=QUESTION, records=[view("a", HOP1, 1), view("b", HOP2, 2)],
    )
    assert "Adapter-78103" in state.linked_entities
    assert state.bridge_entities == (), "a resolved link is no longer a bridge"
    # A resolved link must still anchor selection.
    selected, receipt = select_evidence(
        [view("a", HOP1, 1), view("b", HOP2, 2)],
        anchor_entities=tuple(set(state.required_entities) | set(state.linked_entities)),
    )
    assert {row.evidence_id for row in selected} == {"a", "b"}
    assert receipt.dropped_unanchored_ids == ()


def test_retrieval_and_selection_never_read_gold_labels():
    """The follow-up mechanism must be derivable at inference time.

    A reformulator that consulted required_evidence_ids or the gold answer
    would make Gate C unfalsifiable, so the whole retrieval/selection path is
    kept structurally label-blind.
    """

    import inspect

    from hrm_adaptive_memory.evidence import packing, state, sufficiency
    from hrm_adaptive_memory.retrieval import iterative

    forbidden = ("required_evidence_ids", "oracle_evidence_ids", "gold_answer")
    for module in (iterative, state, sufficiency, packing):
        source = inspect.getsource(module)
        for name in forbidden:
            assert name not in source, f"{module.__name__} reads gold label {name!r}"


def test_followup_query_is_a_bridge_entity_not_an_answer():
    backend = CanonicalRetrievalBackend(CanonicalRetrievalMode.BM25, corpus_records())
    result = run(TwoPassRetriever(backend, k=5, followup_k=5).retrieve(QUESTION))
    assert result.receipt.followup_query == "Adapter-78103"
    assert "840" not in (result.receipt.followup_query or ""), "answer leaked into the query"

"""Adversarial contracts for the structural (chain) selectors.

The central fixture is the exact failure that invalidated the relation-keyword
arm: three wrong-entity records that all state the asked-for relation, versus
one connected bridge record that states a DIFFERENT relation but closes the path
from the question subject to the answer. A document scorer prefers the three
decoys. A structural selector must not.
"""
from __future__ import annotations

import pytest

from hrm_adaptive_memory.retrieval_bench.selectors.chain import (
    build_task_graph,
    derive_entity_types,
    enumerate_chains,
    extract_mentions,
    parse_target_relation,
    s2a_entity_connectivity,
    s2b_chain_completion,
    s2c_chain_plus_relation,
)

STRUCTURAL_ARMS = (s2a_entity_connectivity, s2b_chain_completion, s2c_chain_plus_relation)

QUESTION = "Which custody band is held by SCD-24?"

# The real chain: SCD-24 -> Shasta cable drum -> Denali guide rail -> CHI2-IVORY
CHAIN = {
    "identity": "SCD-24 abbreviates Shasta cable drum.",
    "bridge": "Dispatch: Shasta cable drum :: catalogued asset :: Denali guide rail",
    "answer": "Transfer record | Denali guide rail | custody band | CHI2-IVORY",
}
# Decoys that state the asked-for relation for unrelated entities.
DECOYS = {
    "decoy1": "Ledger: custody band(Whitney tension block) = PSI2-CEDAR",
    "decoy2": "Under survey, PHI2-SABLE was recorded as the custody band of Makalu hoist frame.",
    "decoy3": "Field memo: Denali brake shoe -> custody band -> ALPHA3-DUNE.",
    "decoy4": "From the field: Ojos anchor cleat's custody band came back as cleared.",
    "decoy5": '{"unit": "Kazbek brake shoe", "custody band": "audited"}',
}


def _pool(order):
    return [{"document_id": key, "rank": i} for i, key in enumerate(order, 1)]


def _texts():
    return {**CHAIN, **DECOYS}


def _decoys_first():
    """Worst case: every decoy outranks the real chain in the pool."""
    return list(DECOYS) + list(CHAIN)


def test_parse_target_relation_reads_the_question_not_metadata():
    assert parse_target_relation(QUESTION) == "custody band"
    assert parse_target_relation("What operating district does Denali guide rail carry?") == \
        "operating district"
    assert parse_target_relation("Completely unparseable sentence.") is None


def test_entity_types_are_derived_from_the_pool_not_hardcoded():
    types = derive_entity_types(list(_texts().values()))
    # Shared across multiple capitalized heads -> a real entity type.
    assert "cable drum" in types or "guide rail" in types
    # Prose openers must not become entity types.
    assert "survey lists" not in types


def test_prose_opener_is_not_extracted_as_an_entity():
    corpus = ["The survey lists governing routine for Jaya tension block as Nanda brake shoe.",
              "The survey lists mounted gauge for Tambora guide rail as Ushba brake shoe.",
              # A second head for 'tension block' so the type is derivable at all.
              "Under survey, Ushba tension block was logged."]
    types = derive_entity_types(corpus)
    mentions = extract_mentions(corpus[0], types)
    # The contract: no prose opener becomes an entity, in either position.
    assert not any(m.startswith(("the ", "under ", "from ")) for m in mentions), mentions
    assert "survey lists" not in types
    assert "jaya tension block" in mentions


def test_identifier_surface_is_extracted_from_the_question():
    graph = build_task_graph(_pool(_decoys_first()), QUESTION, _texts())
    assert "scd 24" in graph.question_entities


def test_connected_bridge_beats_unrelated_relation_match():
    """The headline contract: structure must outrank relation keyword decoys."""
    for arm in STRUCTURAL_ARMS:
        chosen = arm(_pool(_decoys_first()), budget=3, question=QUESTION, texts=_texts())
        assert "bridge" in chosen, f"{arm.__name__} dropped the connected bridge: {chosen}"


def test_identity_edge_is_not_dropped_for_low_query_similarity():
    """'SCD-24 abbreviates Shasta cable drum.' shares almost nothing with the question."""
    for arm in STRUCTURAL_ARMS:
        chosen = arm(_pool(_decoys_first()), budget=3, question=QUESTION, texts=_texts())
        assert "identity" in chosen, f"{arm.__name__} dropped the identity record: {chosen}"


def test_wrong_entity_relation_matches_do_not_dominate_packet():
    for arm in STRUCTURAL_ARMS:
        chosen = arm(_pool(_decoys_first()), budget=3, question=QUESTION, texts=_texts())
        decoys = [c for c in chosen if c in DECOYS]
        assert len(decoys) <= 1, f"{arm.__name__} packed decoys {decoys}: {chosen}"


def test_selector_preserves_subject_to_answer_connectivity():
    """At the true chain length the whole path must survive."""
    for arm in STRUCTURAL_ARMS:
        chosen = set(arm(_pool(_decoys_first()), budget=3, question=QUESTION, texts=_texts()))
        assert {"identity", "bridge", "answer"} <= chosen, f"{arm.__name__}: {chosen}"


def test_chain_scoring_uses_runtime_text_only():
    """Passing oracle keys must not change any selection."""
    pool = _pool(_decoys_first())
    poisoned = [{**row, "_oracle_metadata": {"answer_node": "x"},
                 "required_evidence_ids": ["answer"]} for row in pool]
    for arm in STRUCTURAL_ARMS:
        clean = arm(pool, budget=3, question=QUESTION, texts=_texts())
        with_oracle = arm(poisoned, budget=3, question=QUESTION, texts=_texts())
        assert clean == with_oracle, f"{arm.__name__} is sensitive to oracle fields"


def test_chain_selector_has_no_oracle_metadata_access():
    """The graph builder must ignore evaluator-only fields even if handed them."""
    graph = build_task_graph(_pool(_decoys_first()), QUESTION, _texts())
    serialized = repr(graph)
    for forbidden in ("answer_node", "proof_edges", "latent_subject", "_oracle_metadata"):
        assert forbidden not in serialized


def test_enumerate_chains_finds_the_full_three_hop_path():
    graph = build_task_graph(_pool(_decoys_first()), QUESTION, _texts())
    chains = enumerate_chains(graph)
    assert any({"identity", "bridge", "answer"} <= set(c.records) for c in chains), \
        "the true path was never enumerated"


def test_budget_is_respected_and_records_are_unique():
    for arm in STRUCTURAL_ARMS:
        for budget in (1, 2, 3, 5, 8):
            chosen = arm(_pool(_decoys_first()), budget=budget, question=QUESTION,
                         texts=_texts())
            assert len(chosen) == min(budget, len(_texts()))
            assert len(set(chosen)) == len(chosen)


def test_pool_order_does_not_change_the_structural_verdict():
    """A structural selector must not depend on the decoys' rank positions."""
    orders = [_decoys_first(), list(CHAIN) + list(DECOYS),
              ["decoy1", "identity", "decoy2", "bridge", "decoy3", "answer", "decoy4", "decoy5"]]
    for arm in STRUCTURAL_ARMS:
        for order in orders:
            chosen = set(arm(_pool(order), budget=3, question=QUESTION, texts=_texts()))
            assert {"identity", "bridge", "answer"} <= chosen, \
                f"{arm.__name__} failed on order {order}: {chosen}"


@pytest.mark.parametrize("missing", ["identity", "bridge", "answer"])
def test_selector_degrades_gracefully_when_a_link_is_absent(missing):
    """An incomplete pool must still yield a budget-sized packet, not an error."""
    texts = {k: v for k, v in _texts().items() if k != missing}
    order = [k for k in _decoys_first() if k != missing]
    for arm in STRUCTURAL_ARMS:
        chosen = arm(_pool(order), budget=3, question=QUESTION, texts=texts)
        assert len(chosen) == 3

"""ConnectedProofRetention: reachability over proof edges the packet retains.

CPR asks whether the selected packet still supports a connected path from the
question-side node to the answer node. It is evaluator-only.

On the descv4 corpora the proof graph is a simple directed path and
required_evidence_ids covers every record on it, so CPR coincides with CSR
there. These tests therefore also pin the cases where the two DIVERGE -- a
branching proof with a redundant path -- so the metric is known to be real
reachability rather than a renamed CSR.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location(
    "run_selector_ladder_mod", ROOT / "scripts/run_selector_ladder.py")
runner = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(runner)

cpr = runner.connected_proof_retained


def _path_meta():
    """surface:X -> #s -> #b -> #v, one record per hop."""
    return {
        "answer_node": "#v",
        "proof_edges": [
            {"source": "surface:X", "target": "#s", "relation": "refers_to",
             "record_id": "identity"},
            {"source": "#s", "target": "#b", "relation": "catalogued asset",
             "record_id": "link"},
            {"source": "#b", "target": "#v", "relation": "custody band",
             "record_id": "value"},
        ],
    }


POOL = {"identity", "link", "value", "alt_link", "alt_value", "noise"}


def test_full_path_selected_is_retained():
    assert cpr(_path_meta(), {"identity", "link", "value"}, POOL) is True


def test_dropping_the_bridge_breaks_the_path():
    """The exact failure mode the structural selectors exist to prevent."""
    assert cpr(_path_meta(), {"identity", "value"}, POOL) is False


def test_dropping_the_identity_record_breaks_the_path():
    assert cpr(_path_meta(), {"link", "value"}, POOL) is False


def test_dropping_the_answer_record_breaks_the_path():
    assert cpr(_path_meta(), {"identity", "link"}, POOL) is False


def test_extra_distractors_do_not_break_a_complete_path():
    assert cpr(_path_meta(), {"identity", "link", "value", "noise"}, POOL) is True


def test_returns_none_when_the_path_was_never_retrievable():
    """A record missing from the POOL is a retrieval failure, not a selection one."""
    assert cpr(_path_meta(), {"identity", "link"}, {"identity", "link"}) is None


def _branching_meta():
    """Two independent routes from #s to #v; either alone suffices."""
    return {
        "answer_node": "#v",
        "proof_edges": [
            {"source": "#s", "target": "#b1", "relation": "r1", "record_id": "link"},
            {"source": "#b1", "target": "#v", "relation": "r2", "record_id": "value"},
            {"source": "#s", "target": "#b2", "relation": "r3", "record_id": "alt_link"},
            {"source": "#b2", "target": "#v", "relation": "r4", "record_id": "alt_value"},
        ],
    }


def test_cpr_diverges_from_csr_when_a_redundant_path_exists():
    """CSR would demand every record; CPR accepts either complete route."""
    meta = _branching_meta()
    required_all = {"link", "value", "alt_link", "alt_value"}
    only_first_route = {"link", "value"}
    # CSR-style completeness fails, because two records are absent...
    assert not required_all <= only_first_route
    # ...but the proof is still connected, which is what CPR measures.
    assert cpr(meta, only_first_route, POOL) is True


def test_half_of_each_route_is_not_connected():
    """Mixing incomplete routes retains records but proves nothing."""
    assert cpr(_branching_meta(), {"link", "alt_value"}, POOL) is False


def test_empty_selection_is_never_connected():
    assert cpr(_path_meta(), set(), POOL) is False


def test_single_hop_task_needs_only_its_answer_record():
    meta = {"answer_node": "#v",
            "proof_edges": [{"source": "#s", "target": "#v", "relation": "assigned band",
                             "record_id": "fact"}]}
    assert cpr(meta, {"fact"}, {"fact", "noise"}) is True
    assert cpr(meta, {"noise"}, {"fact", "noise"}) is False

"""Release integrity: version, README, changelog, and gate state must agree.

3.6.1 shipped Gate A/Gate B science under stale metadata — pyproject said
3.6.1, `daph.__version__` said 3.4.1, and the README asserted Gate A had never
been run. These tests make that class of drift a build failure.
"""

from __future__ import annotations

import json
import re
import tomllib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def _pyproject_version() -> str:
    return tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]["version"]


def _status() -> dict:
    return json.loads((ROOT / "RESEARCH_STATUS.json").read_text())


def test_every_declared_version_agrees():
    import daph
    import hrm_adaptive_memory

    version = _pyproject_version()
    assert hrm_adaptive_memory.__version__ == version, "canonical package version drifted"
    assert daph.__version__ == version, "legacy package version drifted"
    assert _status()["version"] == version, "RESEARCH_STATUS.json version drifted"
    assert (ROOT / "README.md").read_text().splitlines()[0].endswith(f"v{version}")
    assert f"## {version}" in (ROOT / "CHANGELOG.md").read_text(), "changelog entry missing"


def test_research_status_declares_every_gate():
    gates = _status()["gates"]
    expected = {
        "gate_a0_controlled_evidence_use", "gate_a1_structural_generalization",
        "gate_b_single_pass_retrieval", "gate_c0_controlled_iterative_retrieval",
        "gate_c1_structural_generalization_iterative_retrieval",
        "gate_c2_semantic_information_gap_retrieval",
        "gate_c2_r_retrieval_coverage",
        "gate_c2_i_identity_resolution",
        "gate_c2_s_evidence_selection",
        "gate_c2c_chain_completion",
        "gate_c2c_v2_chain_validation",
        "gate_c2c_v3_chain_validation",
        "gate_c2c_chain_completion_qualified",
        "gate_c3_surface_identity_resolution",
        "gate_n1_natural_external_memory",
        "gate_d_conditional_retrieval_opportunity", "gate_e_learned_retrieval_control",
        "gate_f_recurrence_opportunity", "gate_g_adaptive_recurrence",
        "gate_h_verification_control", "gate_i_unified_executive",
        "gate_j_persistent_memory", "gate_k_consolidation", "gate_l_latent_memory",
    }
    assert set(gates) == expected
    # A gate may also fail for a reason that is about the benchmark rather
    # than the mechanism; that distinction must survive in the status file.
    permitted_prefixes = ("PASS", "FAIL", "PENDING", "IN_PROGRESS", "BLOCKED",
                          "MECHANISM_SUCCESS")
    for name, value in gates.items():
        assert value.startswith(permitted_prefixes), f"{name} has unknown status {value!r}"


def test_readme_does_not_claim_gate_a_is_unrun():
    readme = (ROOT / "README.md").read_text()
    assert "Gate A has not been run" not in readme
    if _status()["gates"]["gate_a0_controlled_evidence_use"] == "PASS":
        assert "Gate A0 — PASSED" in readme


def test_downstream_capabilities_stay_blocked_until_their_gate_passes():
    status = _status()
    gates, capabilities = status["gates"], status["capabilities"]
    coupling = {
        "adaptive_retrieval": "gate_e_learned_retrieval_control",
        "adaptive_recurrence": "gate_g_adaptive_recurrence",
        "executive_training": "gate_i_unified_executive",
        "graphiti_temporal_memory": "gate_j_persistent_memory",
    }
    for capability, gate in coupling.items():
        if gates[gate] != "PASS":
            assert capabilities[capability] == "BLOCKED", (
                f"{capability} must stay BLOCKED while {gate} is {gates[gate]}"
            )


def test_qualified_claims_state_their_scope_limits():
    """A qualified claim must say what it is *not* evidence for."""

    for name, claim in _status()["qualified_claims"].items():
        assert claim["claim"].strip(), f"{name} has no claim text"
        assert claim["not_a_claim_about"], f"{name} does not bound its scope"
        for forbidden in ("general long-term memory", "open-domain RAG"):
            assert forbidden not in claim["claim"], (
                f"{name} overstates scope: {forbidden!r} appears in the claim itself"
            )


def test_gate_a_claim_is_scoped_to_the_controlled_benchmark():
    claim = _status()["qualified_claims"]["gate_a0"]
    assert "controlled synthetic" in claim["claim"].lower()
    assert claim["mean_b3_minus_b0"] == pytest.approx(0.998, abs=1e-3)
    assert claim["grouped_bootstrap_lcb95"] > 0


def test_gate_b_claim_names_the_specific_dense_stack_it_tested():
    claim = _status()["qualified_claims"]["gate_b"]
    assert "MiniLM" in " ".join(claim["not_a_claim_about"])
    assert "dense retrieval being inferior in general" in claim["not_a_claim_about"]


def test_referenced_evidence_paths_exist():
    status = _status()
    for name, claim in status["qualified_claims"].items():
        report = ROOT / claim["report"]
        assert report.exists(), f"{name} references a missing report: {claim['report']}"


def test_benchmark_lineage_records_superseded_corpora():
    lineage = _status()["benchmark_lineage"]
    assert lineage["controlled_gate_a_v1"].startswith("SUPERSEDED")
    assert lineage["controlled_gate_a_v2"].startswith("CANONICAL")
    for name, note in lineage.items():
        built = (ROOT / "data" / "hrm" / name).exists()
        assert built or note.startswith(("PLANNED", "REQUIRED")), (
            f"{name} is neither built nor marked as future work: {note!r}"
        )

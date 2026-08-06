"""Scientific-integrity guards for the V4→BEIR adapter.

These are not convenience tests. If evaluator-only proof truth reaches a
retriever, selector, or prompt, Gate C2 becomes unfalsifiable and every
downstream number is worthless.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from hrm_adaptive_memory.retrieval_bench.contracts import (
    ORACLE_KEYS, OracleLeakError, assert_runtime_clean)

ROOT = Path(__file__).resolve().parents[2]
BEIR = ROOT / "data" / "hrm" / "controlled_gate_a_v4_beir"
SPLITS = ("development", "qualification", "ood")


def rows(split, name):
    return [json.loads(l) for l in (BEIR / split / name).read_text().splitlines() if l.strip()]


@pytest.mark.parametrize("split", SPLITS)
def test_corpus_and_queries_carry_no_oracle_keys(split):
    for row in rows(split, "corpus.jsonl") + rows(split, "queries.jsonl"):
        assert_runtime_clean(row, where=f"{split} runtime input")
        assert set(row) & ORACLE_KEYS == set()


@pytest.mark.parametrize("split", SPLITS)
def test_corpus_documents_expose_only_id_title_text(split):
    for row in rows(split, "corpus.jsonl"):
        assert set(row) == {"_id", "title", "text"}


@pytest.mark.parametrize("split", SPLITS)
def test_queries_expose_only_id_and_text(split):
    for row in rows(split, "queries.jsonl"):
        assert set(row) == {"_id", "text"}


@pytest.mark.parametrize("split", SPLITS)
def test_corpus_text_never_contains_latent_proof_identifiers(split):
    for row in rows(split, "corpus.jsonl"):
        assert "#subject" not in row["text"] and "#bridge" not in row["text"]
        assert "proof_edges" not in row["text"] and "latent_" not in row["text"]


@pytest.mark.parametrize("split", SPLITS)
def test_qrels_match_required_evidence_exactly(split):
    lines = (BEIR / split / "qrels" / "test.tsv").read_text().splitlines()[1:]
    from_qrels = {}
    for line in lines:
        query_id, doc_id, score = line.split("\t")
        assert score == "1"
        from_qrels.setdefault(query_id, set()).add(doc_id)
    source = ROOT / "data" / "hrm" / "controlled_gate_a_v4" / split / "oracle_tasks.jsonl"
    for task in (json.loads(l) for l in source.read_text().splitlines() if l.strip()):
        assert from_qrels[task["task_id"]] == set(task["required_evidence_ids"])


def test_oracle_ground_truth_never_enters_retriever():
    """The guard itself must actually fire."""

    proof = rows("qualification", "proof_ground_truth.jsonl")[0]
    with pytest.raises(OracleLeakError, match="reached runtime"):
        assert_runtime_clean(proof, where="retriever input")
    with pytest.raises(OracleLeakError):
        assert_runtime_clean({"docs": [{"_oracle_metadata": {}}]}, where="nested")
    with pytest.raises(OracleLeakError):
        assert_runtime_clean({"qrels": {"a": 1}}, where="qrels as input")


def test_proof_truth_lives_outside_the_beir_triple():
    for split in SPLITS:
        assert (BEIR / split / "proof_ground_truth.jsonl").exists()
        corpus_ids = {r["_id"] for r in rows(split, "corpus.jsonl")}
        for row in rows(split, "proof_ground_truth.jsonl"):
            # Proof labels point at corpus documents but are not part of the corpus.
            for value in row["required_evidence_ids"]:
                assert value in corpus_ids
            assert row["task_id"] not in corpus_ids


def test_export_is_deterministic_and_hash_stable():
    manifest = json.loads((BEIR / "EXPORT_MANIFEST.json").read_text())
    import hashlib
    for split, block in manifest["splits"].items():
        digest = hashlib.sha256((BEIR / split / "corpus.jsonl").read_bytes()).hexdigest()
        assert digest == block["corpus_sha256"]


def test_adapter_did_not_change_the_benchmark():
    """Phase 6 gate: BM25 and dense must reproduce the pre-adapter references."""

    for backend, split, reference in (("bm25", "qualification", 0.350),
                                      ("bm25", "ood", 0.048),
                                      ("dense", "qualification", 0.364),
                                      ("dense", "ood", 0.072)):
        path = ROOT / "evidence" / "gate_c2" / "retrieval" / f"{backend}_{split}" / "manifest.json"
        if not path.exists():
            pytest.skip(f"{backend}/{split} not yet run")
        manifest = json.loads(path.read_text())
        assert manifest["reproduces_reference"] is True
        assert abs(manifest["measured_complete_set@50"] - reference) <= 0.001

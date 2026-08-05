from __future__ import annotations

import asyncio
import hashlib
import importlib
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

import hrm_adaptive_memory.memory.schema as canonical_schema
from hrm_adaptive_memory.backends import (
    LocalControlBackend,
    LocalRetrievalMode,
    RuVectorBackend,
    SidecarEndpoint,
)
from hrm_adaptive_memory.contracts import (
    BackendCapabilities,
    EvidenceFilter,
    IndexRecord,
)
from hrm_adaptive_memory.derivation import CachedDerivationStore, derive_cached
from hrm_adaptive_memory.evaluation import GateAConfig, qualify_gate_a
from hrm_adaptive_memory.experiments.context_study import (
    ContextConstructor,
    ContextStudyConfig,
    ContextStudyRunner,
    EvaluationMode,
    EvidenceCorpus,
    ExperimentTier,
    ModelOutput,
    OracleTask,
    StudyCondition,
)
from hrm_adaptive_memory.experiments.controlled_dataset import build_controlled_gate_a_corpus
from hrm_adaptive_memory.memory import MemoryLifecycle, MemoryRecord, MemoryStatus, MemoryType
from hrm_adaptive_memory.source_lock import SourceLock


def run(value):
    return asyncio.run(value)


def test_legacy_namespace_is_one_release_alias():
    with pytest.warns(DeprecationWarning, match="deprecated"):
        legacy_schema = importlib.import_module("hrm_memory.memory.schema")
    assert legacy_schema is canonical_schema
    assert legacy_schema.MemoryRecord is canonical_schema.MemoryRecord


def test_capability_requirements_fail_closed():
    capabilities = BackendCapabilities(dense=True)
    capabilities.require("dense")
    with pytest.raises(RuntimeError, match="sparse"):
        capabilities.require("dense", "sparse")


def test_memory_lifecycle_blocks_source_and_illegal_promotion():
    source = MemoryRecord("source", MemoryType.SOURCE, "immutable", "source")
    with pytest.raises(ValueError, match="Immutable source"):
        MemoryLifecycle.transition(source, MemoryStatus.VERIFIED)
    with pytest.raises(ValueError, match="must remain RAW"):
        MemoryRecord(
            "bad-source", MemoryType.SOURCE, "not raw", "source",
            status=MemoryStatus.VERIFIED,
        )
    candidate = MemoryLifecycle.new_generated_candidate(
        memory_id="fact", memory_type=MemoryType.SEMANTIC, content="A fact", source_id="source",
    )
    verified = MemoryLifecycle.transition(candidate, MemoryStatus.VERIFIED)
    promoted = MemoryLifecycle.transition(verified, MemoryStatus.PROMOTED)
    assert promoted.status == MemoryStatus.PROMOTED
    with pytest.raises(ValueError, match="Illegal"):
        MemoryLifecycle.transition(candidate, MemoryStatus.PROMOTED)


def test_legacy_memory_statuses_migrate_without_remaining_canonical_states():
    base = {
        "memory_id": "m", "memory_type": "semantic", "content": "fact", "source_id": "s",
    }
    assert MemoryRecord.from_dict({**base, "status": "current"}).status == MemoryStatus.PROMOTED
    assert MemoryRecord.from_dict({**base, "status": "uncertain"}).status == MemoryStatus.CANDIDATE
    assert MemoryRecord.from_dict({**base, "status": "contradicted"}).status == MemoryStatus.REJECTED


class FakeProvider:
    provider_id = "fake"
    model_revision = "fake-v1"

    def __init__(self):
        self.calls = 0

    async def derive(self, prompt, sources):
        self.calls += 1
        return f"{prompt}:{'|'.join(sources)}"


def test_provider_neutral_derivations_are_cached_and_integrity_checked(tmp_path):
    provider = FakeProvider()
    cache = CachedDerivationStore(tmp_path)
    first = run(derive_cached(provider, cache, prompt="p", sources=["s"], verifier="exact", verified=True))
    second = run(derive_cached(provider, cache, prompt="p", sources=["s"], verifier="exact", verified=True))
    assert first == second
    assert provider.calls == 1
    path = tmp_path / f"{first.receipt.cache_key}.json"
    row = json.loads(path.read_text()); row["output"] = "tampered"
    path.write_text(json.dumps(row))
    with pytest.raises(RuntimeError, match="integrity"):
        cache.get(first.receipt.cache_key)


def test_sidecar_endpoint_rejects_non_loopback_and_unpinned_versions():
    with pytest.raises(ValueError, match="loopback"):
        SidecarEndpoint("remote", "https://example.com", pinned_version="1")
    with pytest.raises(ValueError, match="pinned"):
        SidecarEndpoint("local", "http://127.0.0.1:8080")
    assert SidecarEndpoint("local", "http://localhost:8080", pinned_version="1").backend_id == "local"


def records(count=30):
    return [IndexRecord(
        evidence_id=f"e{index}", source_id=f"s{index}",
        content=f"fact number {index}", token_count=3,
        metadata={"section": f"section-{index}"},
    ) for index in range(count)]


def tasks(count=24):
    return [OracleTask(
        task_id=f"t{index}", question=f"question {index}", answer="42",
        required_evidence_ids=(f"e{index}",), oracle_evidence_ids=(f"e{index}",),
        family=f"family-{index % 2}", template_id=f"template-{index % 2}",
        source_cluster_id=f"source-cluster-{index % 2}", split="test",
    ) for index in range(count)]


def test_local_control_backend_is_async_receipt_backed_and_rejects_filters():
    backend = LocalControlBackend(LocalRetrievalMode.BM25, records())
    result = run(backend.search("fact number 7", k=3))
    assert result.evidence[0].evidence_id == "e7"
    assert result.receipt.returned_ids[0] == "e7"
    with pytest.raises(RuntimeError, match="filters"):
        run(backend.search("fact", k=1, filters=EvidenceFilter(tags=("x",))))


class FakeRuVectorHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, _format, *_args):
        return

    def _send(self, payload):
        data = json.dumps(payload).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path == "/health":
            self._send({"version": "0.1.2"})
        elif self.path == "/v1/capabilities":
            self._send({"dense": True, "sparse": False, "maxsim": False, "filtering": False})
        else:
            self.send_error(404)

    def do_POST(self):
        size = int(self.headers.get("Content-Length", "0"))
        payload = json.loads(self.rfile.read(size))
        if self.path == "/v1/index":
            source = json.dumps([
                (row["evidence_id"], row["source_id"], hashlib.sha256(row["content"].encode()).hexdigest())
                for row in sorted(payload["records"], key=lambda value: value["evidence_id"])
            ], separators=(",", ":"))
            self._send({
                "indexed_ids": [row["evidence_id"] for row in payload["records"]],
                "source_digest": hashlib.sha256(source.encode()).hexdigest(),
            })
        elif self.path == "/v1/search":
            content = "bridge evidence"
            self._send({"evidence": [{
                "evidence_id": "bridge-1",
                "source_id": "source-1",
                "content": content,
                "content_sha256": hashlib.sha256(content.encode()).hexdigest(),
                "token_count": 2,
                "backend_id": "ruvector-test",
                "dense_score": 0.9,
                "rank": 1,
            }]})
        else:
            self.send_error(404)


def test_ruvector_bridge_contract_uses_loopback_receipts_and_no_filter_fallback():
    server = ThreadingHTTPServer(("127.0.0.1", 0), FakeRuVectorHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        backend = RuVectorBackend(SidecarEndpoint(
            "ruvector-test", f"http://127.0.0.1:{server.server_port}",
            pinned_version="0.1.2",
        ))
        indexed = run(backend.index(records(2)))
        assert indexed.indexed_ids == ("e0", "e1")
        result = run(backend.search("bridge", k=1))
        assert result.receipt.returned_ids == ("bridge-1",)
        assert result.receipt.capabilities.maxsim is False
        with pytest.raises(RuntimeError, match="filtering"):
            run(backend.search("bridge", k=1, filters=EvidenceFilter(tags=("x",))))
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


class FakeExecutor:
    model_id = "fake-engineering-only"
    model_revision = "fake-v1"
    scientific_eligible = False

    async def generate(self, prompt):
        return ModelOutput("42", len(prompt.split()), 1, 1.0)


def test_context_runner_constructs_true_paired_arms_and_matched_b1():
    source = records()
    backend = LocalControlBackend(LocalRetrievalMode.BM25, source)
    receipts = run(ContextStudyRunner(
        corpus=EvidenceCorpus(source), retrieval_backend=backend, executor=FakeExecutor(),
        config=ContextStudyConfig(tier=ExperimentTier.SMOKE),
    ).run(tasks()))
    assert len(receipts) == 96
    by_task = {}
    for receipt in receipts:
        by_task.setdefault(receipt.task_id, {})[receipt.condition] = receipt
    for task in tasks():
        arms = by_task[task.task_id]
        assert set(arms) == set(ContextStudyConfig(tier=ExperimentTier.SMOKE).conditions())
        assert arms[StudyCondition.B0_NO_CONTEXT].final_prompt_sha256 != arms[StudyCondition.B3_ORACLE_EVIDENCE].final_prompt_sha256
        assert arms[StudyCondition.B3_ORACLE_EVIDENCE].final_prompt
        assert "[CONTEXT CONDITION]" not in arms[StudyCondition.B3_ORACLE_EVIDENCE].final_prompt
        assert StudyCondition.B3_ORACLE_EVIDENCE.value not in arms[StudyCondition.B3_ORACLE_EVIDENCE].final_prompt
        assert "evidence_id=" not in arms[StudyCondition.B3_ORACLE_EVIDENCE].final_prompt
        assert "source_id=" not in arms[StudyCondition.B3_ORACLE_EVIDENCE].final_prompt
        assert arms[StudyCondition.B3_ORACLE_EVIDENCE].evaluation_mode == EvaluationMode.CAPABILITY_USE
        assert arms[StudyCondition.B1_RANDOM_CONTEXT].evidence_tokens == arms[StudyCondition.B3_ORACLE_EVIDENCE].evidence_tokens
        origins = {value.split("#b1:", 1)[0] for value in arms[StudyCondition.B1_RANDOM_CONTEXT].evidence_ids}
        assert not origins & set(task.oracle_evidence_ids)
        assert arms[StudyCondition.B3_ORACLE_EVIDENCE].evidence_ids == task.oracle_evidence_ids


def test_fake_context_results_cannot_qualify():
    source = records()
    task_rows = tasks()
    receipts = run(ContextStudyRunner(
        corpus=EvidenceCorpus(source),
        retrieval_backend=LocalControlBackend(LocalRetrievalMode.BM25, source),
        executor=FakeExecutor(),
        config=ContextStudyConfig(tier=ExperimentTier.SMOKE),
    ).run(task_rows))
    with pytest.raises(ValueError, match="fake"):
        qualify_gate_a([row.to_dict() for row in receipts], task_rows, GateAConfig(
            tier=ExperimentTier.SMOKE, bootstrap_samples=20,
        ))


def leak_corpus_records(*, clean_count: int = 12):
    rows = [IndexRecord(
        evidence_id="oracle-1", source_id="source-oracle",
        content="The registry assigns the project the access code 58342.", token_count=10,
    )]
    for index in range(5):
        rows.append(IndexRecord(
            evidence_id=f"leak-{index}", source_id=f"source-leak-{index}",
            content=f"An unrelated audit row {index} happens to mention the value 58342 in passing.",
            token_count=14,
        ))
    for index in range(clean_count):
        rows.append(IndexRecord(
            evidence_id=f"clean-{index}", source_id=f"source-clean-{index}",
            content=f"Harmless filler note {index} about an unrelated maintenance topic entirely.",
            token_count=11,
        ))
    return rows


def leak_task():
    return OracleTask(
        task_id="leak-task", question="What access code is assigned to the project?",
        answer="58342", required_evidence_ids=("oracle-1",), oracle_evidence_ids=("oracle-1",),
        family="single_hop", template_id="single_hop-template-0",
        source_cluster_id="single_hop-source-cluster-0", split="test", verifier="numeric",
    )


def make_constructor(source, seed=42):
    return ContextConstructor(
        EvidenceCorpus(source),
        LocalControlBackend(LocalRetrievalMode.BM25, source),
        config=ContextStudyConfig(tier=ExperimentTier.SMOKE, seed=seed),
    )


def test_b1_excludes_required_oracle_and_answer_leaking_chunks():
    source = leak_corpus_records()
    constructor = make_constructor(source)
    task = leak_task()
    b1 = run(constructor.construct(task, StudyCondition.B1_RANDOM_CONTEXT))
    b3 = run(constructor.construct(task, StudyCondition.B3_ORACLE_EVIDENCE))
    origins = {value.split("#b1:", 1)[0] for value in (row.evidence_id for row in b1.evidence)}
    assert not origins & set(task.required_evidence_ids)
    assert not origins & set(task.oracle_evidence_ids)
    assert all(not origin.startswith("leak-") for origin in origins)
    assert task.answer not in b1.prompt
    assert b1.evidence_tokens == b3.evidence_tokens


def test_b1_selection_is_deterministic_per_seed():
    task = leak_task()
    first = run(make_constructor(leak_corpus_records(), seed=7).construct(task, StudyCondition.B1_RANDOM_CONTEXT))
    second = run(make_constructor(leak_corpus_records(), seed=7).construct(task, StudyCondition.B1_RANDOM_CONTEXT))
    assert [row.evidence_id for row in first.evidence] == [row.evidence_id for row in second.evidence]
    assert first.prompt_sha256 == second.prompt_sha256


class CharTokenCodec:
    """Subword-like codec: truncation can split a number mid-way, as BPE can."""

    def count(self, text):
        return len(text)

    def truncate(self, text, tokens):
        return text[:tokens]


def test_b1_truncation_cannot_create_the_answer_token():
    # Oracle content is 9 chars, so B1 truncates candidates to 9 chars.
    # Truncating "abcdef 741..." at 9 chars would yield "abcdef 74" — the
    # exact gold answer created at the cut. The constructor must skip it.
    source = [
        IndexRecord(evidence_id="oracle-1", source_id="s-oracle", content="paycode74", token_count=9),
        IndexRecord(evidence_id="trap-1", source_id="s-trap", content="abcdef 741zz", token_count=12),
        IndexRecord(evidence_id="clean-1", source_id="s-clean", content="ghijkl mn op", token_count=12),
    ]
    task = OracleTask(
        task_id="trunc-task", question="What is the paycode for Zone Q?", answer="74",
        required_evidence_ids=("oracle-1",), oracle_evidence_ids=("oracle-1",),
        family="single_hop", template_id="single_hop-template-0",
        source_cluster_id="single_hop-source-cluster-0", split="test", verifier="numeric",
    )
    constructor = ContextConstructor(
        EvidenceCorpus(source),
        LocalControlBackend(LocalRetrievalMode.BM25, source),
        config=ContextStudyConfig(tier=ExperimentTier.SMOKE, seed=42),
        token_codec=CharTokenCodec(),
    )
    b1 = run(constructor.construct(task, StudyCondition.B1_RANDOM_CONTEXT))
    b3 = run(constructor.construct(task, StudyCondition.B3_ORACLE_EVIDENCE))
    assert b1.evidence_tokens == b3.evidence_tokens
    assert "74" not in " ".join(
        part for part in b1.prompt.split() if part not in task.question.split()
    )
    origins = {row.evidence_id.split("#b1:", 1)[0] for row in b1.evidence}
    assert origins == {"clean-1"}
    # With only the trap candidate left, construction must fail closed.
    trapped = ContextConstructor(
        EvidenceCorpus(source[:2]),
        LocalControlBackend(LocalRetrievalMode.BM25, source[:2]),
        config=ContextStudyConfig(tier=ExperimentTier.SMOKE, seed=42),
        token_codec=CharTokenCodec(),
    )
    with pytest.raises(ValueError, match="Insufficient"):
        run(trapped.construct(task, StudyCondition.B1_RANDOM_CONTEXT))


def test_b1_fails_closed_instead_of_shortening_when_only_leaking_chunks_remain():
    source = leak_corpus_records(clean_count=0)
    constructor = make_constructor(source)
    with pytest.raises(ValueError, match="Insufficient"):
        run(constructor.construct(leak_task(), StudyCondition.B1_RANDOM_CONTEXT))


def test_optional_hard_distractor_is_answer_free_and_token_matched():
    source = records()
    receipts = run(ContextStudyRunner(
        corpus=EvidenceCorpus(source),
        retrieval_backend=LocalControlBackend(LocalRetrievalMode.BM25, source),
        executor=FakeExecutor(),
        config=ContextStudyConfig(tier=ExperimentTier.SMOKE, include_hard_distractor=True),
    ).run(tasks()))
    by_task = {}
    for receipt in receipts:
        by_task.setdefault(receipt.task_id, {})[receipt.condition] = receipt
    for task in tasks():
        b1b = by_task[task.task_id][StudyCondition.B1_HARD_DISTRACTOR]
        b3 = by_task[task.task_id][StudyCondition.B3_ORACLE_EVIDENCE]
        assert b1b.evidence_tokens == b3.evidence_tokens
        assert "42" not in b1b.final_prompt
        assert "#b1b:" not in b1b.final_prompt
        origins = {value.split("#b1b:", 1)[0] for value in b1b.evidence_ids}
        assert not origins & set(task.oracle_evidence_ids)


def gate_fixture(count: int):
    task_rows = [OracleTask(
        task_id=f"q{index}", question=f"q {index}", answer="a",
        required_evidence_ids=(f"oracle-{index}",),
        oracle_evidence_ids=(f"oracle-{index}",),
        family=f"family-{index % 5}", template_id=f"template-{index % 5}",
        source_cluster_id=f"source-cluster-{index % 5}", split="test",
    ) for index in range(count)]
    rows = []
    for task in task_rows:
        for condition, quality, evidence in (
            (StudyCondition.B0_NO_CONTEXT, 0.0, ()),
            (StudyCondition.B1_RANDOM_CONTEXT, 0.0, (f"irrelevant-{task.task_id}#b1:1",)),
            (StudyCondition.B2_NAIVE_RETRIEVAL, 0.0, ()),
            (StudyCondition.B3_ORACLE_EVIDENCE, 1.0, task.oracle_evidence_ids),
        ):
            rows.append({
                "task_id": task.task_id, "condition": condition.value,
                "family": task.family, "template_id": task.template_id,
                "source_cluster_id": task.source_cluster_id, "split": "test",
                "evaluation_mode": EvaluationMode.CAPABILITY_USE.value,
                "evidence_ids": evidence, "source_ids": evidence, "retrieval_scores": (),
                "final_prompt": f"prompt {task.task_id} evidence",
                "final_prompt_sha256": f"{task.task_id}-{condition.value}",
                "prompt_tokens": 10, "evidence_tokens": 1 if evidence else 0,
                "completion_tokens": 1, "output": "a" if quality else "wrong",
                "verified_quality": quality, "verified_utility": quality,
                "exact_match": bool(quality), "latency_ms": 1, "peak_memory_bytes": 0,
                "model_id": "hrm", "model_revision": "pinned", "corpus_digest": "corpus",
                "retrieval_backend_id": "control:bm25" if condition == StudyCondition.B2_NAIVE_RETRIEVAL else None,
                "retrieval_receipt": None, "scientific_eligible": True,
            })
    return task_rows, rows


def test_gate_a_cannot_promote_a_low_n_result():
    task_rows, rows = gate_fixture(2)
    report = qualify_gate_a(rows, task_rows, GateAConfig(
        tier=ExperimentTier.QUALIFICATION, bootstrap_samples=50,
    ))
    assert report["status"] == "INSUFFICIENT_POWER"
    assert report["retrieval_expansion_allowed"] is False


def test_gate_a_qualification_is_paired_grouped_and_fail_closed():
    task_rows, rows = gate_fixture(500)
    report = qualify_gate_a(rows, task_rows, GateAConfig(
        tier=ExperimentTier.QUALIFICATION, bootstrap_samples=100,
    ))
    assert report["status"] == "PASS_HRM_CAN_USE_ORACLE_EVIDENCE"
    assert report["quality_bootstrap_by_group"]["template_id"]["group_count"] == 5
    assert report["quality_bootstrap_by_group"]["family"]["group_count"] == 5
    assert report["quality_bootstrap_by_group"]["source_cluster_id"]["group_count"] == 5
    assert report["retrieval_expansion_allowed"] is True
    assert report["graphiti_integration_allowed"] is False
    assert report["controller_training_allowed"] is False


def test_evidence_grounded_study_is_reported_but_cannot_promote_retrieval():
    task_rows, rows = gate_fixture(500)
    for row in rows:
        row["evaluation_mode"] = EvaluationMode.EVIDENCE_GROUNDED.value
    report = qualify_gate_a(rows, task_rows, GateAConfig(
        tier=ExperimentTier.QUALIFICATION,
        evaluation_mode=EvaluationMode.EVIDENCE_GROUNDED,
        bootstrap_samples=100,
    ))
    assert report["status"] == "EVIDENCE_GROUNDED_NONPROMOTABLE"
    assert report["retrieval_expansion_allowed"] is False
    assert report["next_stage"] == "GROUNDING_DIAGNOSTIC_ONLY"


def test_gate_a_rejects_model_visible_arm_labels_and_requires_full_prompt_receipts():
    task_rows, rows = gate_fixture(2)
    rows[0]["final_prompt"] = "[CONTEXT CONDITION] B0_NO_CONTEXT"
    with pytest.raises(ValueError, match="model-visible"):
        qualify_gate_a(rows, task_rows, GateAConfig(
            tier=ExperimentTier.SMOKE, bootstrap_samples=20,
        ))
    rows[0]["final_prompt"] = "[E1] evidence_id=required-record"
    with pytest.raises(ValueError, match="Control/provenance"):
        qualify_gate_a(rows, task_rows, GateAConfig(
            tier=ExperimentTier.SMOKE, bootstrap_samples=20,
        ))
    rows[0]["final_prompt"] = None
    with pytest.raises(ValueError, match="final prompt"):
        qualify_gate_a(rows, task_rows, GateAConfig(
            tier=ExperimentTier.SMOKE, bootstrap_samples=20,
        ))


def test_source_lock_keeps_every_external_runtime_disabled():
    payload = json.loads(Path("third_party/sources.lock.json").read_text())
    lock = SourceLock(payload)
    assert lock.bundle["committed"] is False
    assert len(lock.sources) == 9
    assert lock.sources["TurboVec"].runtime_enabled is False
    assert lock.sources["TurboVec"].archive_sha256 == (
        "3eba4445ed152392ad572a3c85f903f80091d95e52908521e72ec22d043ee341"
    )
    with pytest.raises(RuntimeError, match="gate-blocked"):
        lock.require_runtime("RuVector")
    with pytest.raises(RuntimeError, match="gate-blocked"):
        lock.require_runtime("TurboVec")


def test_controlled_gate_a_corpus_is_reproducible_and_has_independent_evidence_ids():
    first = build_controlled_gate_a_corpus(seed=7, tasks_per_family=5)
    second = build_controlled_gate_a_corpus(seed=7, tasks_per_family=5)
    assert first.manifest == second.manifest
    assert first.manifest["task_count"] == 25
    assert len({row["task_id"] for row in first.tasks}) == 25
    evidence_ids = {row["evidence_id"] for row in first.evidence}
    source_clusters = {row["source_cluster_id"] for row in first.tasks}
    assert len(source_clusters) >= 5
    for row in first.tasks:
        task = OracleTask.from_dict(row)
        assert set(task.required_evidence_ids).issubset(evidence_ids)
        assert task.verifier == "numeric"
    ContextStudyConfig(tier=ExperimentTier.SMOKE).validate(
        [OracleTask.from_dict(row) for row in first.tasks]
    )


def test_controlled_generator_never_leaks_answer_into_question():
    corpus = build_controlled_gate_a_corpus(seed=3601, tasks_per_family=100)
    for row in corpus.tasks:
        question_terms = tuple(row["question"].lower().replace("-", " ").split())
        answer = row["answer"].lower()
        assert answer not in question_terms, row["task_id"]


@pytest.mark.parametrize("dataset", ["controlled_gate_a_v1", "controlled_gate_a_v2"])
def test_committed_controlled_corpus_has_the_predeclared_qualification_shape(dataset):
    root = Path("data/hrm") / dataset
    manifest = json.loads((root / "dataset_manifest.json").read_text())
    task_rows = [json.loads(line) for line in (root / "oracle_tasks.jsonl").read_text().splitlines()]
    evidence_rows = [json.loads(line) for line in (root / "evidence.jsonl").read_text().splitlines()]
    tasks = [OracleTask.from_dict(row) for row in task_rows]
    evidence = EvidenceCorpus([
        IndexRecord(
            evidence_id=row["evidence_id"], source_id=row["source_id"],
            content=row["content"], token_count=max(1, len(row["content"].split())),
            source_type=row["source_type"], metadata=row["metadata"],
        )
        for row in evidence_rows
    ])
    assert manifest["claim_strength"] == "CONTROLLED_SYNTHETIC_BENCHMARK_ONLY"
    assert manifest["natural_memory_claim_allowed"] is False
    assert len(tasks) == manifest["task_count"] == 500
    assert len(evidence_rows) == manifest["evidence_count"] == 1200
    ContextStudyConfig(
        tier=ExperimentTier.QUALIFICATION, include_hard_distractor=True,
    ).validate(tasks)
    for task in tasks:
        evidence.validate_task(task)

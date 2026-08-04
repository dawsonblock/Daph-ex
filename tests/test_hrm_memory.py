from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from hrm_adaptive_memory.baseline.evaluator import BaselineCondition, BaselineResult, OracleContextGate
from hrm_adaptive_memory.context.packer import ContextBudget, EvidenceItem, EvidencePacker
from hrm_adaptive_memory.controller.actions import Action, ActionOutcome
from hrm_adaptive_memory.controller.policy import UtilityController
from hrm_adaptive_memory.execution.counterfactual import CounterfactualCollector, DecisionState
from hrm_adaptive_memory.execution.oracle import oracle_opportunity
from hrm_adaptive_memory.hrm.model import HRMAdapter, HRMModelSpec, PromptCondition
from hrm_adaptive_memory.hrm.recurrent_hooks import HRMRecurrentTracer
from hrm_adaptive_memory.hrm.variable_recurrence import recurrence_arms
from hrm_adaptive_memory.memory.chunking import Chunk, StructuralChunker
from hrm_adaptive_memory.memory.contradiction import ContradictionLedger
from hrm_adaptive_memory.memory.schema import MemoryRecord, MemoryStatus, MemoryType
from hrm_adaptive_memory.memory.stores import SemanticMemoryStore, SourceMemoryStore
from hrm_adaptive_memory.retrieval.dense import DenseRetriever, HashingEmbedder
from hrm_adaptive_memory.retrieval.evaluator import evaluate_retrieval
from hrm_adaptive_memory.retrieval.hybrid import HybridRetriever, RetrievalCandidate
from hrm_adaptive_memory.retrieval.lexical import BM25Retriever
from hrm_adaptive_memory.retrieval.reranker import LexicalOverlapReranker


def chunk(identifier: str, content: str, tokens: int = 20, source: str = "source") -> Chunk:
    return Chunk(identifier, source, "prose", source, "section", content, tokens)


class FakeTokenizer:
    def __call__(self, text, return_tensors=None):
        return {"input_ids": torch.tensor([[index + 1 for index, _ in enumerate(text.split())]])}

    def decode(self, values, skip_special_tokens=False):
        return " ".join(str(int(value)) for value in values)


class FakeHRM:
    def __init__(self):
        self.config = SimpleNamespace(
            model_type="hrm_text", architectures=["HrmTextForCausalLM"],
            hidden_size=1536, num_layers_per_stack=16, H_cycles=2, L_cycles=3,
            max_position_embeddings=4096, prefix_lm=True,
        )
        self.device = torch.device("cpu")
        self.last_inputs = None

    def generate(self, **kwargs):
        self.last_inputs = kwargs
        return torch.cat((kwargs["input_ids"], torch.tensor([[99, 100]])), dim=1)


def test_hrm_adapter_pins_native_shape_and_prefixlm_mask():
    model = FakeHRM(); adapter = HRMAdapter(model, FakeTokenizer())
    result = adapter.generate("solve task", condition=PromptCondition.SYNTH_COT, max_new_tokens=2)
    assert result["prompt"].startswith("<|im_start|><|quad_end|><|object_ref_end|>")
    assert torch.equal(model.last_inputs["token_type_ids"], torch.ones_like(model.last_inputs["input_ids"]))
    assert result["completion_tokens"] == 2
    assert adapter.spec.revision == "9f082d68b8cd0ebc56e33f1c88c45609174c272c"


def test_hrm_adapter_rejects_wrong_checkpoint_shape():
    model = FakeHRM(); model.config.hidden_size = 1024
    with pytest.raises(ValueError, match="config mismatch"):
        HRMAdapter(model, FakeTokenizer())


class FakeStack(nn.Module):
    def forward(self, hidden):
        return hidden + 1


class FakeCore(nn.Module):
    def __init__(self):
        super().__init__(); self.L_module = FakeStack(); self.H_module = FakeStack()
        self.config = SimpleNamespace(L_cycles=3)


def test_recurrent_tracer_maps_reused_stack_calls_to_cycles():
    core = FakeCore(); hidden = torch.zeros(1, 2, 4)
    with HRMRecurrentTracer(core) as tracer:
        for _ in range(3): hidden = core.L_module(hidden)
        hidden = core.H_module(hidden)
        for _ in range(3): hidden = core.L_module(hidden)
        hidden = core.H_module(hidden)
    assert [(row.state_type, row.high_cycle, row.low_cycle) for row in tracer.traces] == [
        ("L", 0, 1), ("L", 0, 2), ("L", 0, 3), ("H", 1, None),
        ("L", 1, 1), ("L", 1, 2), ("L", 1, 3), ("H", 2, None),
    ]
    assert len(tracer.traces[0].mean_pool) == 4


def test_recurrence_ablation_marks_only_released_schedule_as_pretrained():
    arms = recurrence_arms()
    assert [arm.name for arm in arms] == ["H1L3", "H2L3", "H3L3", "H4L3"]
    assert [arm.name for arm in arms if arm.is_pretrained_schedule] == ["H2L3"]
    assert arms[-1].stack_invocations == 16


def test_source_memory_is_append_only_and_typed(tmp_path):
    store = SourceMemoryStore(tmp_path / "source.jsonl")
    record = MemoryRecord("source-1", MemoryType.SOURCE, "immutable document", "doc")
    store.append(record); store.append(record)
    assert list(store) == [record]
    with pytest.raises(ValueError, match="immutable"):
        store.append(MemoryRecord("source-1", MemoryType.SOURCE, "changed", "doc"))
    with pytest.raises(ValueError, match="Expected source"):
        store.append(MemoryRecord("sem-1", MemoryType.SEMANTIC, "claim", "doc"))


def test_semantic_lineage_requires_explicit_supersedes(tmp_path):
    store = SemanticMemoryStore(tmp_path / "semantic.jsonl")
    old = MemoryRecord("old", MemoryType.SEMANTIC, "X is 5", "doc")
    new = MemoryRecord("new", MemoryType.SEMANTIC, "X is 7", "doc", supersedes="old")
    store.append(old); store.append(new)
    edge = ContradictionLedger().supersede(old, new)
    assert (edge.prior_id, edge.current_id, edge.relation) == ("old", "new", MemoryStatus.SUPERSEDED.value)


def test_structural_prose_chunking_preserves_sections():
    chunks = StructuralChunker(target_tokens=20, overlap_tokens=4, token_counter=lambda value: len(value.split())).prose(
        "# Alpha\nOne two three.\n\nFour five six.\n# Beta\nSeven eight.", source_id="doc", title="Title",
    )
    assert {row.section for row in chunks} == {"Alpha", "Beta"}
    assert all(row.source_id == "doc" and row.chunk_id.startswith("chunk_") for row in chunks)


def test_python_chunking_does_not_split_functions():
    text = "x = 1\n\ndef alpha(a):\n    return a + 1\n\nclass Beta:\n    def method(self):\n        return 2\n"
    chunks = StructuralChunker(target_tokens=4, overlap_tokens=1).code(text, source_id="code.py")
    by_section = {row.section: row.content for row in chunks}
    assert "return a + 1" in by_section["alpha"]
    assert "def method" in by_section["Beta"]


def test_bm25_catches_exact_identifiers():
    chunks = [chunk("a", "Gate A A7 failure in phase_10.py"), chunk("b", "general architecture notes")]
    result = BM25Retriever(chunks).search("A7 phase_10.py", top_k=1)
    assert result[0][0].chunk_id == "a"


def test_hash_dense_and_hybrid_rrf_are_deterministic():
    chunks = [chunk("a", "adaptive compute controller"), chunk("b", "source memory evidence")]
    embedder = HashingEmbedder(32)
    assert embedder("same text") == embedder("same text")
    hybrid = HybridRetriever(chunks, dense=DenseRetriever(chunks, embedder), lexical=BM25Retriever(chunks))
    first = hybrid.search("adaptive compute", final_k=2)
    second = hybrid.search("adaptive compute", final_k=2)
    assert first == second
    assert first[0].chunk.chunk_id == "a"


def test_reranker_reorders_fused_candidates():
    chunks = [chunk("a", "weakly related"), chunk("b", "exact HRM recurrence cycles")]
    results = HybridRetriever(chunks).search(
        "HRM recurrence cycles", reranker=LexicalOverlapReranker(), final_k=1,
    )
    assert results[0].chunk.chunk_id == "b"
    assert results[0].reranker_score == 1.0


def test_retrieval_metrics_include_all_required_evidence():
    report = evaluate_retrieval(["a", "b", "c"], {"a", "c"}, k=2, required_evidence_ids={"a", "c"})
    assert report.recall_at_k == 0.5
    assert report.mrr == 1.0
    assert report.oracle_evidence_recall == 0.0


def test_context_budget_rejects_overallocation():
    with pytest.raises(ValueError, match="exceed"):
        ContextBudget(total=100, task=30, evidence=40, state=30, generation=20)


def test_evidence_packer_respects_budget_and_penalizes_duplicates():
    budget = ContextBudget(total=120, task=20, evidence=45, state=10, generation=20)
    candidates = [
        EvidenceItem(chunk("a", "same repeated evidence", 20), 1.0),
        EvidenceItem(chunk("b", "same repeated evidence", 20), 0.99),
        EvidenceItem(chunk("c", "different supporting fact", 20), 0.90),
    ]
    packet = EvidencePacker(budget=budget, token_counter=lambda value: max(1, len(value.split()) // 2), redundancy_weight=0.5).pack(
        objective="Solve", current_state="Waiting", candidates=candidates, unresolved=("missing?",),
    )
    assert packet.evidence_tokens <= 45
    assert "a" in packet.selected_chunk_ids and "c" in packet.selected_chunk_ids
    assert "[OBJECTIVE]" in packet.rendered and "[UNRESOLVED INFORMATION]" in packet.rendered
    assert packet.provenance["a"] == "source"


def baseline(task, condition, quality):
    return BaselineResult(task, condition, quality, quality, quality == 1.0)


def test_oracle_context_gate_requires_paired_ids():
    rows = [baseline("a", BaselineCondition.NO_CONTEXT, 0), baseline("b", BaselineCondition.ORACLE_EVIDENCE, 1)]
    with pytest.raises(ValueError, match="paired"):
        OracleContextGate().evaluate(rows)


def test_oracle_context_gate_rejects_duplicate_condition_rows():
    rows = [
        baseline("a", BaselineCondition.NO_CONTEXT, 0),
        baseline("a", BaselineCondition.NO_CONTEXT, 0),
        baseline("a", BaselineCondition.ORACLE_EVIDENCE, 1),
    ]
    with pytest.raises(ValueError, match="Duplicate task IDs"):
        OracleContextGate(minimum_paired_tasks=1).evaluate(rows)


def test_oracle_context_gate_separates_retrieval_from_model_use():
    rows = []
    for task in ("a", "b"):
        rows.append(baseline(task, BaselineCondition.NO_CONTEXT, 0.25))
        rows.append(baseline(task, BaselineCondition.NAIVE_RETRIEVAL, 0.35))
        rows.append(baseline(task, BaselineCondition.ORACLE_EVIDENCE, 0.75))
    report = OracleContextGate(minimum_oracle_quality_gain=0.2).evaluate(rows)
    assert report["passed"] is True
    assert report["oracle_quality_gain"] == 0.5
    assert report["controller_training_allowed"] is False


def test_counterfactual_collector_isolates_state_and_uses_executed_utility():
    state = DecisionState("task", 0, (0.1, 0.2), "draft", metadata={"calls": []})
    def execute(action, quality, cost):
        def inner(candidate):
            candidate.metadata["calls"].append(action.value)
            return ActionOutcome(action, quality, cost)
        return inner
    collector = CounterfactualCollector({
        Action.STOP: execute(Action.STOP, 0.0, 0.0),
        Action.RETRIEVE: execute(Action.RETRIEVE, 1.0, 0.2),
    })
    records = collector.collect(state)
    assert state.metadata["calls"] == []
    retrieve = next(row for row in records if row.action == Action.RETRIEVE)
    assert retrieve.utility == pytest.approx(0.8)
    assert retrieve.reference_action == Action.STOP
    assert retrieve.delta_utility_vs_reference == pytest.approx(0.8)


def test_counterfactual_records_all_action_cost_dimensions():
    collector = CounterfactualCollector({
        Action.STOP: lambda _state: ActionOutcome(Action.STOP, 0.0),
        Action.RETRIEVE: lambda _state: ActionOutcome(
            Action.RETRIEVE,
            1.0,
            compute_cost=0.1,
            latency_cost=0.2,
            token_cost=0.3,
            retrieval_cost=0.4,
            verification_cost=0.5,
        ),
    }, lambda_compute=1.0, lambda_latency=2.0, lambda_tokens=3.0,
       lambda_retrieval=4.0, lambda_verification=5.0)
    record = next(row for row in collector.collect(DecisionState("task", 0, (0.0,), ""))
                  if row.action == Action.RETRIEVE)
    assert record.gross_quality == 1.0
    assert record.quality == 1.0
    assert record.retrieval_cost == 0.4
    assert record.compute_cost == 0.1
    assert record.latency_cost == 0.2
    assert record.token_cost == 0.3
    assert record.verification_cost == 0.5
    assert record.utility == pytest.approx(-4.5)


def test_counterfactual_reference_must_be_in_requested_actions():
    collector = CounterfactualCollector({
        Action.STOP: lambda _state: ActionOutcome(Action.STOP, 0.0),
        Action.RETRIEVE: lambda _state: ActionOutcome(Action.RETRIEVE, 1.0),
    })
    with pytest.raises(ValueError, match="reference"):
        collector.collect(DecisionState("task", 0, (0.0,), ""), actions=(Action.RETRIEVE,))


def test_oracle_opportunity_blocks_controller_when_fixed_action_dominates():
    collector = CounterfactualCollector({
        Action.STOP: lambda _state: ActionOutcome(Action.STOP, 0.5),
        Action.RETRIEVE: lambda _state: ActionOutcome(Action.RETRIEVE, 0.4),
    })
    records = sum((collector.collect(DecisionState(task, 0, (0,), "")) for task in ("a", "b")), [])
    report = oracle_opportunity(records)
    assert report["passed"] is False
    assert report["controller_training_allowed"] is False


def test_utility_controller_fails_closed_then_stops_on_nonpositive_voc():
    controller = UtilityController()
    with pytest.raises(RuntimeError, match="VERIFIED_FIT"):
        controller.decide({Action.RETRIEVE: 1.0})
    decision = controller.decide({Action.ANSWER: 0.0, Action.RETRIEVE: 0.1}, {Action.RETRIEVE: 0.2}, research_override=True)
    assert decision.action == Action.ANSWER
    assert decision.stopped is True


def test_engineering_smoke_command(tmp_path):
    output = tmp_path / "smoke"
    subprocess.run([sys.executable, "scripts/run_hrm_memory_smoke.py", "--output", str(output)], check=True)
    report = json.loads((output / "report.json").read_text())
    assert report["status"] == "ENGINEERING_SMOKE_ONLY"
    assert report["controller_training_allowed"] is False

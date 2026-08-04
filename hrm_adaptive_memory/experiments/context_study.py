"""Canonical paired B0/B1/B2/B3 context construction and execution."""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Mapping, Protocol, Sequence

from ..context.packer import ContextBudget
from ..contracts import (
    IndexRecord,
    RetrievedEvidence,
    RetrievalBackend,
    RetrievalReceipt,
    sha256_text,
)


class StudyCondition(str, Enum):
    B0_NO_CONTEXT = "B0_NO_CONTEXT"
    B1_RANDOM_CONTEXT = "B1_RANDOM_CONTEXT"
    B2_NAIVE_RETRIEVAL = "B2_NAIVE_RETRIEVAL"
    B3_ORACLE_EVIDENCE = "B3_ORACLE_EVIDENCE"


class ExperimentTier(str, Enum):
    SMOKE = "SMOKE"
    PILOT = "PILOT"
    QUALIFICATION = "QUALIFICATION"


TIER_REQUIREMENTS = {
    ExperimentTier.SMOKE: {"tasks": 24, "groups": 2, "promotable": False},
    ExperimentTier.PILOT: {"tasks": 100, "groups": 5, "promotable": False},
    ExperimentTier.QUALIFICATION: {"tasks": 500, "groups": 5, "promotable": True},
}


@dataclass(frozen=True)
class OracleTask:
    task_id: str
    question: str
    answer: str
    required_evidence_ids: tuple[str, ...]
    oracle_evidence_ids: tuple[str, ...]
    family: str
    template_id: str
    split: str
    verifier: str = "exact"
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not all((self.task_id, self.question.strip(), self.answer.strip(), self.family, self.template_id, self.split)):
            raise ValueError("Oracle tasks require identifiers, content, family, template, and split")
        if not self.required_evidence_ids or not self.oracle_evidence_ids:
            raise ValueError("Oracle tasks require independently labeled evidence")
        if not set(self.required_evidence_ids).issubset(self.oracle_evidence_ids):
            raise ValueError("Oracle evidence must contain every required evidence ID")

    @classmethod
    def from_dict(cls, row: Mapping[str, Any]) -> "OracleTask":
        return cls(**{
            **dict(row),
            "required_evidence_ids": tuple(row["required_evidence_ids"]),
            "oracle_evidence_ids": tuple(row["oracle_evidence_ids"]),
        })


class EvidenceCorpus:
    def __init__(self, records: Sequence[IndexRecord]):
        self.records: dict[str, IndexRecord] = {}
        for record in records:
            previous = self.records.get(record.evidence_id)
            if previous is not None and previous != record:
                raise ValueError(f"Duplicate evidence ID with different content: {record.evidence_id}")
            self.records[record.evidence_id] = record

    def validate_task(self, task: OracleTask) -> None:
        missing = sorted(set(task.oracle_evidence_ids) - self.records.keys())
        if missing:
            raise ValueError(f"Task {task.task_id} references missing oracle evidence: {missing}")

    def digest(self) -> str:
        payload = json.dumps([
            (row.evidence_id, row.source_id, sha256_text(row.content), row.token_count)
            for row in sorted(self.records.values(), key=lambda value: value.evidence_id)
        ], separators=(",", ":"))
        return sha256_text(payload)


class TokenCodec(Protocol):
    def count(self, text: str) -> int: ...
    def truncate(self, text: str, tokens: int) -> str: ...


class WhitespaceTokenCodec:
    def count(self, text: str) -> int:
        return len(text.split())

    def truncate(self, text: str, tokens: int) -> str:
        return " ".join(text.split()[:tokens])


@dataclass(frozen=True)
class ModelOutput:
    text: str
    prompt_tokens: int
    completion_tokens: int
    latency_ms: float
    peak_memory_bytes: int = 0


class StudyModelExecutor(Protocol):
    model_id: str
    model_revision: str
    scientific_eligible: bool

    async def generate(self, prompt: str) -> ModelOutput: ...


@dataclass(frozen=True)
class ContextStudyConfig:
    tier: ExperimentTier = ExperimentTier.SMOKE
    retrieval_k: int = 10
    seed: int = 42
    budget: ContextBudget = field(default_factory=ContextBudget)
    lambda_evidence_tokens: float = 0.0

    def validate(self, tasks: Sequence[OracleTask]) -> None:
        requirement = TIER_REQUIREMENTS[self.tier]
        unique = {task.task_id for task in tasks}
        groups = {task.template_id for task in tasks}
        if len(unique) != len(tasks):
            raise ValueError("Context-study task IDs must be unique")
        if len(unique) < int(requirement["tasks"]):
            raise ValueError(f"{self.tier.value} requires at least {requirement['tasks']} tasks")
        if len(groups) < int(requirement["groups"]):
            raise ValueError(f"{self.tier.value} requires at least {requirement['groups']} groups")
        if self.retrieval_k < 1 or self.lambda_evidence_tokens < 0:
            raise ValueError("Invalid retrieval or utility configuration")


@dataclass(frozen=True)
class ConstructedContext:
    condition: StudyCondition
    prompt: str
    prompt_sha256: str
    evidence: tuple[RetrievedEvidence, ...]
    evidence_tokens: int
    retrieval_receipt: RetrievalReceipt | None


@dataclass(frozen=True)
class ContextStudyReceipt:
    task_id: str
    condition: StudyCondition
    family: str
    template_id: str
    split: str
    evidence_ids: tuple[str, ...]
    source_ids: tuple[str, ...]
    retrieval_scores: tuple[Mapping[str, float | None], ...]
    final_prompt: str
    final_prompt_sha256: str
    prompt_tokens: int
    evidence_tokens: int
    completion_tokens: int
    output: str
    verified_quality: float
    verified_utility: float
    exact_match: bool
    latency_ms: float
    peak_memory_bytes: int
    model_id: str
    model_revision: str
    corpus_digest: str
    retrieval_backend_id: str | None
    retrieval_receipt: Mapping[str, Any] | None
    scientific_eligible: bool

    def to_dict(self) -> dict[str, Any]:
        row = asdict(self)
        row["condition"] = self.condition.value
        return row


def _normalize(value: str) -> str:
    value = re.sub(r"<\|[^>]+\|>", " ", value.lower().strip())
    return " ".join(value.split())


def verify_answer(task: OracleTask, output: str) -> tuple[float, bool]:
    if task.verifier == "exact":
        passed = _normalize(output) == _normalize(task.answer)
    elif task.verifier == "numeric":
        numbers = re.findall(r"[-+]?\d+(?:\.\d+)?", output)
        passed = bool(numbers) and math.isclose(float(numbers[-1]), float(task.answer), rel_tol=1e-9, abs_tol=1e-9)
    else:
        raise ValueError(f"Unsupported verifier: {task.verifier}")
    return float(passed), passed


class ContextConstructor:
    def __init__(
        self, corpus: EvidenceCorpus, retrieval_backend: RetrievalBackend,
        *, config: ContextStudyConfig, token_codec: TokenCodec | None = None,
    ):
        self.corpus = corpus
        self.retrieval_backend = retrieval_backend
        self.config = config
        self.token_codec = token_codec or WhitespaceTokenCodec()

    @staticmethod
    def _score(row: RetrievedEvidence) -> Mapping[str, float | None]:
        return {
            "dense": row.dense_score,
            "lexical": row.lexical_score,
            "reranker": row.reranker_score,
        }

    def _direct(self, record: IndexRecord, backend_id: str, rank: int) -> RetrievedEvidence:
        return RetrievedEvidence.from_index(record, backend_id=backend_id, rank=rank)

    def _matched_irrelevant(self, task: OracleTask) -> tuple[RetrievedEvidence, ...]:
        target = sum(self.token_codec.count(self.corpus.records[value].content) for value in task.oracle_evidence_ids)
        excluded = set(task.required_evidence_ids) | set(task.oracle_evidence_ids)
        candidates = [row for key, row in self.corpus.records.items() if key not in excluded]
        candidates.sort(key=lambda row: hashlib.sha256(
            f"{self.config.seed}\0{task.task_id}\0{row.evidence_id}".encode()
        ).hexdigest())
        selected: list[RetrievedEvidence] = []
        remaining = target
        for record in candidates:
            if remaining <= 0:
                break
            take = min(remaining, self.token_codec.count(record.content))
            content = self.token_codec.truncate(record.content, take)
            if not content:
                continue
            derived = IndexRecord(
                evidence_id=f"{record.evidence_id}#b1:{take}",
                source_id=record.source_id,
                content=content,
                token_count=self.token_codec.count(content),
                source_type=record.source_type,
                metadata={**record.metadata, "matched_irrelevant_from": record.evidence_id},
            )
            selected.append(self._direct(derived, "matched-irrelevant", len(selected) + 1))
            remaining -= derived.token_count
        if remaining != 0:
            raise ValueError(f"Insufficient irrelevant evidence to match B3 for task {task.task_id}")
        return tuple(selected)

    def _compose(self, task: OracleTask, condition: StudyCondition, evidence: Sequence[RetrievedEvidence]) -> str:
        parts = ["[OBJECTIVE]", task.question, "[CONTEXT CONDITION]", condition.value, "[EVIDENCE]"]
        if not evidence:
            parts.append("[NO EXTERNAL EVIDENCE]")
        for index, row in enumerate(evidence, 1):
            parts.extend([
                f"[E{index}] evidence_id={row.evidence_id} source_id={row.source_id}",
                row.content,
            ])
        parts.extend([
            "[RESPONSE REQUIREMENT]",
            "Return only the answer. If the supplied evidence is insufficient, return INSUFFICIENT_EVIDENCE.",
        ])
        return "\n".join(parts)

    async def construct(self, task: OracleTask, condition: StudyCondition) -> ConstructedContext:
        self.corpus.validate_task(task)
        retrieval_receipt = None
        if condition == StudyCondition.B0_NO_CONTEXT:
            evidence: tuple[RetrievedEvidence, ...] = ()
        elif condition == StudyCondition.B1_RANDOM_CONTEXT:
            evidence = self._matched_irrelevant(task)
        elif condition == StudyCondition.B2_NAIVE_RETRIEVAL:
            result = await self.retrieval_backend.search(task.question, k=self.config.retrieval_k)
            evidence, retrieval_receipt = result.evidence, result.receipt
        elif condition == StudyCondition.B3_ORACLE_EVIDENCE:
            evidence = tuple(
                self._direct(self.corpus.records[value], "oracle", rank)
                for rank, value in enumerate(task.oracle_evidence_ids, 1)
            )
        else:  # pragma: no cover - exhaustive enum guard
            raise ValueError(condition)
        evidence_tokens = sum(self.token_codec.count(row.content) for row in evidence)
        if evidence_tokens > self.config.budget.evidence:
            raise ValueError(f"Evidence exceeds configured budget for {task.task_id}/{condition.value}")
        prompt = self._compose(task, condition, evidence)
        if self.token_codec.count(task.question) > self.config.budget.task:
            raise ValueError("Task exceeds configured task budget")
        if self.token_codec.count(prompt) > self.config.budget.total - self.config.budget.generation:
            raise ValueError("Constructed prompt exceeds working-context budget")
        return ConstructedContext(
            condition=condition,
            prompt=prompt,
            prompt_sha256=sha256_text(prompt),
            evidence=tuple(evidence),
            evidence_tokens=evidence_tokens,
            retrieval_receipt=retrieval_receipt,
        )


class ContextStudyRunner:
    def __init__(
        self, *, corpus: EvidenceCorpus, retrieval_backend: RetrievalBackend,
        executor: StudyModelExecutor, config: ContextStudyConfig,
        token_codec: TokenCodec | None = None,
    ):
        self.corpus = corpus
        self.executor = executor
        self.config = config
        self.constructor = ContextConstructor(
            corpus, retrieval_backend, config=config, token_codec=token_codec,
        )

    async def run(self, tasks: Sequence[OracleTask]) -> list[ContextStudyReceipt]:
        self.config.validate(tasks)
        receipts: list[ContextStudyReceipt] = []
        for task in tasks:
            contexts = {
                condition: await self.constructor.construct(task, condition)
                for condition in StudyCondition
            }
            if contexts[StudyCondition.B0_NO_CONTEXT].prompt_sha256 == contexts[StudyCondition.B3_ORACLE_EVIDENCE].prompt_sha256:
                raise RuntimeError("B0/B3 prompt digests must differ when B3 has evidence")
            for condition in StudyCondition:
                context = contexts[condition]
                output = await self.executor.generate(context.prompt)
                quality, exact = verify_answer(task, output.text)
                utility = quality - self.config.lambda_evidence_tokens * context.evidence_tokens
                retrieval = context.retrieval_receipt
                receipts.append(ContextStudyReceipt(
                    task_id=task.task_id,
                    condition=condition,
                    family=task.family,
                    template_id=task.template_id,
                    split=task.split,
                    evidence_ids=tuple(row.evidence_id for row in context.evidence),
                    source_ids=tuple(row.source_id for row in context.evidence),
                    retrieval_scores=tuple(ContextConstructor._score(row) for row in context.evidence),
                    final_prompt=context.prompt,
                    final_prompt_sha256=context.prompt_sha256,
                    prompt_tokens=output.prompt_tokens,
                    evidence_tokens=context.evidence_tokens,
                    completion_tokens=output.completion_tokens,
                    output=output.text,
                    verified_quality=quality,
                    verified_utility=utility,
                    exact_match=exact,
                    latency_ms=output.latency_ms,
                    peak_memory_bytes=output.peak_memory_bytes,
                    model_id=self.executor.model_id,
                    model_revision=self.executor.model_revision,
                    corpus_digest=self.corpus.digest(),
                    retrieval_backend_id=None if retrieval is None else retrieval.backend_id,
                    retrieval_receipt=None if retrieval is None else asdict(retrieval),
                    scientific_eligible=bool(self.executor.scientific_eligible),
                ))
        return receipts

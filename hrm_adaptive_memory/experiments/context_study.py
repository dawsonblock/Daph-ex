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
    B1_HARD_DISTRACTOR = "B1B_HARD_DISTRACTOR"
    B2_NAIVE_RETRIEVAL = "B2_NAIVE_RETRIEVAL"
    B3_ORACLE_EVIDENCE = "B3_ORACLE_EVIDENCE"


PRIMARY_STUDY_CONDITIONS = (
    StudyCondition.B0_NO_CONTEXT,
    StudyCondition.B1_RANDOM_CONTEXT,
    StudyCondition.B2_NAIVE_RETRIEVAL,
    StudyCondition.B3_ORACLE_EVIDENCE,
)


class EvaluationMode(str, Enum):
    """Keep capability use and evidence-grounded abstention as separate claims."""

    CAPABILITY_USE = "CAPABILITY_USE"
    EVIDENCE_GROUNDED = "EVIDENCE_GROUNDED"


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
    source_cluster_id: str
    split: str
    verifier: str = "exact"
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not all((
            self.task_id, self.question.strip(), self.answer.strip(), self.family,
            self.template_id, self.source_cluster_id, self.split,
        )):
            raise ValueError(
                "Oracle tasks require identifiers, content, family, template, source cluster, and split"
            )
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
    evaluation_mode: EvaluationMode = EvaluationMode.CAPABILITY_USE
    include_hard_distractor: bool = False

    def conditions(self) -> tuple[StudyCondition, ...]:
        if self.include_hard_distractor:
            return (
                StudyCondition.B0_NO_CONTEXT,
                StudyCondition.B1_RANDOM_CONTEXT,
                StudyCondition.B1_HARD_DISTRACTOR,
                StudyCondition.B2_NAIVE_RETRIEVAL,
                StudyCondition.B3_ORACLE_EVIDENCE,
            )
        return PRIMARY_STUDY_CONDITIONS

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
        if not isinstance(self.evaluation_mode, EvaluationMode):
            raise ValueError("Context study requires an explicit EvaluationMode")


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
    source_cluster_id: str
    split: str
    evaluation_mode: EvaluationMode
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
        row["evaluation_mode"] = self.evaluation_mode.value
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
    elif task.verifier == "canonical":
        # Non-numeric answers (symbolic labels, enums, booleans, entity names,
        # small JSON fields). Mirrors the numeric verifier's semantics: the
        # answer must be the last candidate the output commits to, so an output
        # that lists several candidates does not pass on the strength of one.
        answer_terms = tuple(re.findall(r"\w+", _normalize(task.answer)))
        output_terms = tuple(re.findall(r"\w+", _normalize(output)))
        width = len(answer_terms)
        starts = [
            index for index in range(len(output_terms) - width + 1)
            if output_terms[index:index + width] == answer_terms
        ] if width else []
        passed = bool(starts) and starts[-1] + width == len(output_terms)
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

    def _direct(
        self, record: IndexRecord, backend_id: str, rank: int,
        *, lexical_score: float | None = None,
    ) -> RetrievedEvidence:
        return RetrievedEvidence.from_index(
            record, backend_id=backend_id, rank=rank, lexical_score=lexical_score,
        )

    def _take_matched_evidence(
        self, task: OracleTask, candidates: Sequence[tuple[IndexRecord, float | None]], *,
        backend_id: str, suffix: str,
    ) -> tuple[RetrievedEvidence, ...]:
        """Truncate a deterministic candidate sequence to B3's exact token budget."""

        target = sum(
            self.token_codec.count(self.corpus.records[value].content)
            for value in task.oracle_evidence_ids
        )
        selected: list[RetrievedEvidence] = []
        remaining = target
        for record, score in candidates:
            if remaining <= 0:
                break
            take = min(remaining, self.token_codec.count(record.content))
            content = self.token_codec.truncate(record.content, take)
            if not content:
                continue
            derived = IndexRecord(
                evidence_id=f"{record.evidence_id}#{suffix}:{take}",
                source_id=record.source_id,
                content=content,
                token_count=self.token_codec.count(content),
                source_type=record.source_type,
                metadata={**record.metadata, "matched_control_from": record.evidence_id},
            )
            # Subword truncation can cut a longer number at the boundary and
            # thereby create the gold-answer token; re-check the derived text.
            if self._leaks_answer(task, derived):
                continue
            selected.append(self._direct(
                derived, backend_id, len(selected) + 1, lexical_score=score,
            ))
            remaining -= derived.token_count
        if remaining != 0:
            raise ValueError(
                f"Insufficient non-oracle evidence to token-match B3 for task {task.task_id}"
            )
        return tuple(selected)

    @staticmethod
    def _leaks_answer(task: OracleTask, record: IndexRecord) -> bool:
        """True when the record contains the normalized gold-answer token sequence."""

        answer_terms = tuple(re.findall(r"\w+", _normalize(task.answer)))
        content_terms = tuple(re.findall(r"\w+", _normalize(record.content)))
        width = len(answer_terms)
        return bool(width) and any(
            content_terms[index:index + width] == answer_terms
            for index in range(len(content_terms) - width + 1)
        )

    def _matched_irrelevant(self, task: OracleTask) -> tuple[RetrievedEvidence, ...]:
        excluded = set(task.required_evidence_ids) | set(task.oracle_evidence_ids)
        candidates = [
            row for key, row in self.corpus.records.items()
            if key not in excluded and not self._leaks_answer(task, row)
        ]
        candidates.sort(key=lambda row: hashlib.sha256(
            f"{self.config.seed}\0{task.task_id}\0{row.evidence_id}".encode()
        ).hexdigest())
        return self._take_matched_evidence(
            task, [(row, None) for row in candidates], backend_id="matched-random", suffix="b1",
        )

    def _hard_distractors(self, task: OracleTask) -> tuple[RetrievedEvidence, ...]:
        """Select answer-free lexical distractors without exposing an arm label to HRM."""

        excluded = set(task.required_evidence_ids) | set(task.oracle_evidence_ids)
        query_terms = set(re.findall(r"\w+", _normalize(task.question)))
        oracle_source_types = {
            self.corpus.records[value].source_type for value in task.oracle_evidence_ids
        }

        def candidate_score(record: IndexRecord) -> tuple[int, int, str]:
            content_terms = set(re.findall(r"\w+", _normalize(record.content)))
            overlap = len(query_terms & content_terms)
            source_match = int(record.source_type in oracle_source_types)
            tie_break = hashlib.sha256(
                f"{self.config.seed}\0{task.task_id}\0{record.evidence_id}".encode()
            ).hexdigest()
            return source_match, overlap, tie_break

        candidates = [
            row for key, row in self.corpus.records.items()
            if key not in excluded and not self._leaks_answer(task, row)
        ]
        candidates.sort(
            key=lambda row: (-candidate_score(row)[0], -candidate_score(row)[1], candidate_score(row)[2])
        )
        return self._take_matched_evidence(
            task,
            [(row, float(candidate_score(row)[1])) for row in candidates],
            backend_id="hard-distractor",
            suffix="b1b",
        )

    def _response_requirement(self) -> str:
        if self.config.evaluation_mode == EvaluationMode.CAPABILITY_USE:
            return "Return only the answer. Use supplied evidence when helpful, but answer the task regardless."
        return (
            "Return only the answer when the supplied evidence supports it; "
            "otherwise return INSUFFICIENT_EVIDENCE."
        )

    def _compose(self, task: OracleTask, evidence: Sequence[RetrievedEvidence]) -> str:
        parts = ["[OBJECTIVE]", task.question, "[EVIDENCE]"]
        if not evidence:
            parts.append("[NO EXTERNAL EVIDENCE]")
        for index, row in enumerate(evidence, 1):
            parts.extend([
                f"[E{index}]",
                row.content,
            ])
        parts.extend([
            "[RESPONSE REQUIREMENT]",
            self._response_requirement(),
        ])
        return "\n".join(parts)

    async def construct(self, task: OracleTask, condition: StudyCondition) -> ConstructedContext:
        self.corpus.validate_task(task)
        retrieval_receipt = None
        if condition == StudyCondition.B0_NO_CONTEXT:
            evidence: tuple[RetrievedEvidence, ...] = ()
        elif condition == StudyCondition.B1_RANDOM_CONTEXT:
            evidence = self._matched_irrelevant(task)
        elif condition == StudyCondition.B1_HARD_DISTRACTOR:
            evidence = self._hard_distractors(task)
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
        prompt = self._compose(task, evidence)
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
            receipts.extend(await self.run_task(task))
        return receipts

    async def run_task(self, task: OracleTask) -> list[ContextStudyReceipt]:
        """Execute every configured arm for one task.

        Public so long qualification runs can persist receipts incrementally
        and resume without duplicating completed tasks.
        """

        receipts: list[ContextStudyReceipt] = []
        contexts = {
            condition: await self.constructor.construct(task, condition)
            for condition in self.config.conditions()
        }
        if contexts[StudyCondition.B0_NO_CONTEXT].prompt_sha256 == contexts[StudyCondition.B3_ORACLE_EVIDENCE].prompt_sha256:
            raise RuntimeError("B0/B3 prompt digests must differ when B3 has evidence")
        for condition in self.config.conditions():
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
                source_cluster_id=task.source_cluster_id,
                split=task.split,
                evaluation_mode=self.config.evaluation_mode,
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

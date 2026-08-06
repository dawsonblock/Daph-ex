"""Bounded two-pass retrieval.

Maximum depth is 2 by construction — there is no free-running agent loop here.
A follow-up happens only when the structured evidence state names a specific
unresolved entity, and the follow-up query is derived from that name.

Stage 3 explicitly requires deterministic/template reformulation first; an LLM
reformulator may be *compared* against this later, never substituted silently.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from typing import Any, Protocol, Sequence

from ..evidence.packing import SelectionReceipt, select_evidence
from ..evidence.state import EvidenceRecordView, EvidenceState, build_evidence_state
from ..evidence.sufficiency import SufficiencyReport, SufficiencyVerdict, assess


@dataclass(frozen=True)
class FollowupQuery:
    text: str
    reason: str
    terms: tuple[str, ...]


class QueryReformulator(Protocol):
    reformulator_id: str

    def formulate_followup(
        self, question: str, state: EvidenceState, report: SufficiencyReport,
    ) -> FollowupQuery | None: ...


class DeterministicReformulator:
    """Query on the unresolved entity the evidence itself introduced.

    For a two-hop question the first pass binds the subject to a linking
    entity; querying that entity alone is both the minimal and the most
    selective way to retrieve the second hop.
    """

    reformulator_id = "deterministic-bridge-v1"

    def formulate_followup(
        self, question: str, state: EvidenceState, report: SufficiencyReport,
    ) -> FollowupQuery | None:
        if not report.needs_followup or not report.followup_terms:
            return None
        terms = tuple(report.followup_terms)
        return FollowupQuery(
            text=" ".join(terms),
            reason=report.verdict.value,
            terms=terms,
        )


@dataclass
class IterativeRetrievalReceipt:
    question: str
    reformulator_id: str
    passes: int
    first_pass_ids: tuple[str, ...]
    followup_query: str | None
    followup_reason: str | None
    second_pass_ids: tuple[str, ...]
    merged_ids: tuple[str, ...]
    selected_ids: tuple[str, ...]
    first_state: dict[str, Any]
    first_report: dict[str, Any]
    final_state: dict[str, Any]
    final_report: dict[str, Any]
    selection: dict[str, Any] | None
    retrieval_calls: int
    latency_ms: float
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class IterativeResult:
    records: tuple[EvidenceRecordView, ...]
    state: EvidenceState
    report: SufficiencyReport
    receipt: IterativeRetrievalReceipt


class TwoPassRetriever:
    """One optional follow-up, then merge, deduplicate, and select."""

    def __init__(
        self, backend: Any, *, reformulator: QueryReformulator | None = None,
        k: int = 10, followup_k: int = 10, token_budget: int = 2200,
        max_records: int | None = None, lambda_redundancy: float = 0.5,
        enforce_anchoring: bool = True, max_passes: int = 2,
    ):
        if max_passes not in (1, 2):
            raise ValueError("Bounded iterative retrieval permits at most two passes")
        self.backend = backend
        self.reformulator = reformulator or DeterministicReformulator()
        self.k = k
        self.followup_k = followup_k
        self.token_budget = token_budget
        self.max_records = max_records
        self.lambda_redundancy = lambda_redundancy
        self.enforce_anchoring = enforce_anchoring
        self.max_passes = max_passes

    @staticmethod
    def _merge(
        first: Sequence[EvidenceRecordView], second: Sequence[EvidenceRecordView],
    ) -> tuple[EvidenceRecordView, ...]:
        """Interleave passes by rank so neither pass is systematically buried."""

        merged: dict[str, EvidenceRecordView] = {}
        for row in first:
            merged.setdefault(row.evidence_id, row)
        for row in second:
            merged.setdefault(row.evidence_id, row)
        ordered = sorted(merged.values(), key=lambda row: (row.rank, row.evidence_id))
        return tuple(
            EvidenceRecordView(**{**asdict(row), "rank": index})
            for index, row in enumerate(ordered, 1)
        )

    async def retrieve(self, question: str, *, select: bool = True) -> IterativeResult:
        started = time.perf_counter()
        first = await self.backend.search(question, k=self.k)
        first_views = tuple(EvidenceRecordView.from_retrieved(row) for row in first.evidence)
        first_state = build_evidence_state(question=question, records=first_views)
        first_report = assess(first_state)

        followup = None
        second_views: tuple[EvidenceRecordView, ...] = ()
        calls = 1
        if self.max_passes == 2:
            followup = self.reformulator.formulate_followup(question, first_state, first_report)
        if followup is not None:
            second = await self.backend.search(followup.text, k=self.followup_k)
            second_views = tuple(EvidenceRecordView.from_retrieved(row) for row in second.evidence)
            calls += 1

        merged = self._merge(first_views, second_views) if second_views else first_views
        final_state = build_evidence_state(question=question, records=merged)
        final_report = assess(final_state)

        selection_receipt: SelectionReceipt | None = None
        selected = merged
        if select:
            # Anchor on the question's entities *and* on entities the evidence
            # links to them. The record that resolves a link names the link,
            # not the question's subject, so anchoring on question entities
            # alone silently discards the second hop even when pass one
            # already retrieved it.
            anchors = set(final_state.required_entities) | set(final_state.linked_entities)
            if followup is not None:
                anchors |= set(followup.terms)
            selected, selection_receipt = select_evidence(
                merged, anchor_entities=tuple(anchors),
                token_budget=self.token_budget, max_records=self.max_records,
                lambda_redundancy=self.lambda_redundancy,
                enforce_anchoring=self.enforce_anchoring,
            )
            final_state = build_evidence_state(question=question, records=selected)
            final_report = assess(final_state)

        receipt = IterativeRetrievalReceipt(
            question=question,
            reformulator_id=self.reformulator.reformulator_id,
            passes=calls,
            first_pass_ids=tuple(row.evidence_id for row in first_views),
            followup_query=None if followup is None else followup.text,
            followup_reason=None if followup is None else followup.reason,
            second_pass_ids=tuple(row.evidence_id for row in second_views),
            merged_ids=tuple(row.evidence_id for row in merged),
            selected_ids=tuple(row.evidence_id for row in selected),
            first_state=first_state.to_dict(),
            first_report=first_report.to_dict(),
            final_state=final_state.to_dict(),
            final_report=final_report.to_dict(),
            selection=None if selection_receipt is None else selection_receipt.to_dict(),
            retrieval_calls=calls,
            latency_ms=(time.perf_counter() - started) * 1000,
        )
        return IterativeResult(selected, final_state, final_report, receipt)

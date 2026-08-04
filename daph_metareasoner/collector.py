"""Counterfactual experience generation from isolated copies of one state."""

from __future__ import annotations

import json
import hashlib
import gzip
from collections import deque
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable, Iterable, List, Mapping, Protocol, Sequence

from .schema import (
    ALL_ACTIONS,
    Action,
    BranchResult,
    ExperienceRecord,
    ReasoningState,
    Task,
    canonical_digest,
    records_digest,
)
from .utility import UtilityConfig


class ReasoningAdapter(Protocol):
    model_digest: str
    environment_digest: str

    def initial_state(self, task: Task, *, budget: float) -> ReasoningState: ...

    def execute(self, task: Task, state: ReasoningState, action: Action) -> BranchResult: ...


Verifier = Callable[[Mapping[str, Any], Mapping[str, Any]], tuple[float, str]]


@dataclass(frozen=True)
class CollectionConfig:
    max_depth: int = 1
    max_states_per_task: int = 16
    initial_budget: float = 1.0
    actions: Sequence[Action] = ALL_ACTIONS

    def validate(self) -> None:
        if self.max_depth < 0:
            raise ValueError("max_depth must be non-negative")
        if self.max_states_per_task < 1:
            raise ValueError("max_states_per_task must be positive")
        if Action.STOP not in self.actions:
            raise ValueError("Counterfactual collection must include STOP")


class CounterfactualExperienceCollector:
    def __init__(
        self,
        adapter: ReasoningAdapter,
        verifier: Verifier,
        utility: UtilityConfig = UtilityConfig(),
        config: CollectionConfig = CollectionConfig(),
    ) -> None:
        config.validate()
        self.adapter = adapter
        self.verifier = verifier
        self.utility = utility
        self.config = config

    @staticmethod
    def _task_payload(task: Task) -> Mapping[str, Any]:
        return {"expected": task.expected, **dict(task.metadata)}

    def _verify(self, answer: str, task: Task) -> tuple[float, str]:
        return self.verifier({"generated_text": answer}, self._task_payload(task))

    def collect_task(self, task: Task) -> List[ExperienceRecord]:
        initial = self.adapter.initial_state(task, budget=self.config.initial_budget)
        pending = deque([initial])
        visited: set[str] = set()
        records: List[ExperienceRecord] = []
        dataset_digest = task.digest()
        while pending and len(visited) < self.config.max_states_per_task:
            state = pending.popleft()
            if state.state_id in visited:
                continue
            visited.add(state.state_id)
            before_raw, before_status = self._verify(state.answer, task)
            quality_before = self.utility.quality(before_raw)
            for action in self.config.actions:
                # The state is frozen; every branch receives the exact same object.
                branch = self.adapter.execute(task, state, action)
                if branch.state_before_id != state.state_id:
                    raise RuntimeError("Adapter violated isolated-branch state identity")
                after_raw, after_status = self._verify(branch.next_state.answer, task)
                quality_after = self.utility.quality(after_raw)
                delta_quality, cost, delta_utility = self.utility.voc(
                    quality_before=quality_before,
                    quality_after=quality_after,
                    action=action,
                    receipt=branch.receipt,
                )
                records.append(ExperienceRecord(
                    task=task,
                    state=state,
                    action=action.value,
                    next_state_id=branch.next_state.state_id,
                    answer_before=state.answer,
                    answer_after=branch.next_state.answer,
                    verifier_status_before=before_status,
                    verifier_status_after=after_status,
                    quality_before=quality_before,
                    quality_after=quality_after,
                    delta_quality=delta_quality,
                    action_cost=cost,
                    delta_utility=delta_utility,
                    receipt=branch.receipt,
                    model_digest=self.adapter.model_digest,
                    environment_digest=self.adapter.environment_digest,
                    dataset_digest=dataset_digest,
                ))
                if (
                    action is not Action.STOP
                    and state.step < self.config.max_depth
                    and branch.next_state.budget_remaining > 0.0
                ):
                    pending.append(branch.next_state)
        if pending:
            raise RuntimeError(
                f"State expansion for {task.task_id} exceeded max_states_per_task="
                f"{self.config.max_states_per_task}; refusing a partial counterfactual table"
            )
        return records

    def collect_many(self, tasks: Iterable[Task]) -> List[ExperienceRecord]:
        task_rows = list(tasks)
        dataset_digest = canonical_digest(
            [task.digest() for task in sorted(task_rows, key=lambda row: row.task_id)]
        )
        records: List[ExperienceRecord] = []
        for task in task_rows:
            records.extend(self.collect_task(task))
        return [replace(record, dataset_digest=dataset_digest) for record in records]

    @staticmethod
    def save(records: Sequence[ExperienceRecord], path: str | Path) -> Mapping[str, Any]:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record.to_dict(), sort_keys=True) + "\n")
        receipt = {
            "records": len(records),
            "states": len({row.state.state_id for row in records}),
            "tasks": len({row.task.task_id for row in records}),
            "records_digest": records_digest(list(records)),
            "file_digest": hashlib.sha256(destination.read_bytes()).hexdigest(),
        }
        (destination.with_suffix(destination.suffix + ".receipt.json")).write_text(
            json.dumps(receipt, indent=2)
        )
        return receipt


def load_records(path: str | Path) -> List[ExperienceRecord]:
    source = Path(path)
    text = (
        gzip.open(source, "rt", encoding="utf-8").read()
        if source.suffix == ".gz" else source.read_text()
    )
    return [
        ExperienceRecord.from_dict(json.loads(line))
        for line in text.splitlines()
        if line.strip()
    ]

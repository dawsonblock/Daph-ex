"""Canonical HRM recurrent-state contract and commit ledger.

Adapted from the Hierarchos engineering discipline (versioned state schema,
signature-validated resume, fail-closed layout rejection, and exact
selected-state commitment) without any of its RWKV-specific machinery.

The fundamental invariant is:

    state producing accepted output == state committed for later computation

No alternate candidate state may accidentally propagate.  This module is
behavior-neutral for Gate A: nothing in the qualified pipeline imports it yet.
It becomes load-bearing when counterfactual actions (Gate C) and recurrence
control (Stage 10+) need real state branching.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Any, Mapping

HRM_STATE_SCHEMA_VERSION = "hrm-state-v1"

_SIGNATURE_KEYS = ("hidden_size", "high_layers", "low_layers", "schema")


class ActionType(str, Enum):
    ANSWER = "ANSWER"
    RETRIEVE = "RETRIEVE"
    RETRIEVE_FOLLOWUP = "RETRIEVE_FOLLOWUP"
    THINK = "THINK"
    CALCULATE = "CALCULATE"
    VERIFY = "VERIFY"
    ABSTAIN = "ABSTAIN"
    STOP = "STOP"


def _tensor_digest(value: Any) -> str:
    """Deterministic digest of a tensor-like object (or None)."""

    if value is None:
        return "none"
    try:
        import torch
        if torch.is_tensor(value):
            data = value.detach().to("cpu").contiguous()
            payload = bytes(data.numpy().tobytes())
            return hashlib.sha256(
                f"{tuple(data.shape)}|{data.dtype}".encode() + payload
            ).hexdigest()
    except ImportError:  # pragma: no cover - torch is a base dependency
        pass
    return hashlib.sha256(repr(value).encode()).hexdigest()


@dataclass(frozen=True)
class HRMState:
    """One immutable recurrent-state snapshot.

    ``high_state``/``low_state`` are the H/L module states; ``workspace_state``
    is the optional packed-context state.  Frozen so that "mutation" is always
    an explicit new object — accidental in-place propagation is impossible.
    """

    high_state: Any
    low_state: Any
    workspace_state: Any | None
    step_index: int
    reasoning_depth: int
    last_action: ActionType | None
    halted: bool
    state_schema_version: str = HRM_STATE_SCHEMA_VERSION
    signature: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.state_schema_version != HRM_STATE_SCHEMA_VERSION:
            raise ValueError(
                f"Incompatible HRM state schema {self.state_schema_version!r}; "
                f"expected {HRM_STATE_SCHEMA_VERSION!r}. No silent migration."
            )
        if self.step_index < 0 or self.reasoning_depth < 0:
            raise ValueError("HRM state counters cannot be negative")

    def state_hash(self) -> str:
        payload = "|".join((
            _tensor_digest(self.high_state),
            _tensor_digest(self.low_state),
            _tensor_digest(self.workspace_state),
            str(self.step_index),
            str(self.reasoning_depth),
            "-" if self.last_action is None else self.last_action.value,
            str(self.halted),
            self.state_schema_version,
            repr(sorted(self.signature.items())),
        ))
        return hashlib.sha256(payload.encode()).hexdigest()

    def advanced(self, *, action: ActionType, high_state: Any = None,
                 low_state: Any = None, workspace_state: Any = None,
                 halted: bool = False) -> "HRMState":
        """Produce the successor state for an executed action."""

        return replace(
            self,
            high_state=self.high_state if high_state is None else high_state,
            low_state=self.low_state if low_state is None else low_state,
            workspace_state=self.workspace_state if workspace_state is None else workspace_state,
            step_index=self.step_index + 1,
            reasoning_depth=self.reasoning_depth + (1 if action == ActionType.THINK else 0),
            last_action=action,
            halted=halted,
        )

    def validate_signature(self, expected: Mapping[str, Any]) -> None:
        """Fail closed when resuming into a different model/layout."""

        mismatches = {
            key: (self.signature.get(key), expected.get(key))
            for key in _SIGNATURE_KEYS
            if key in expected and self.signature.get(key) != expected.get(key)
        }
        if mismatches:
            raise ValueError(f"HRM state signature mismatch: {mismatches}")


@dataclass(frozen=True)
class StepResult:
    """Output of one recurrent execution step."""

    output: Any
    next_state: HRMState
    diagnostics: Mapping[str, Any] = field(default_factory=dict)


class StateCommitLedger:
    """Enforces that exactly the selected candidate state is committed.

    Counterfactual execution creates several candidate successor states from
    one parent.  The ledger is the single authority on which state the rest of
    the system may continue from; committing a state that was never selected,
    or selecting a state from a different parent, is an error, never a warning.
    """

    def __init__(self, initial: HRMState):
        self._committed = initial
        self._candidates: dict[str, HRMState] = {}
        self._selected: str | None = None
        self.history: list[tuple[str, str]] = [("initial", initial.state_hash())]

    @property
    def committed(self) -> HRMState:
        return self._committed

    def propose(self, action: ActionType, candidate: StepResult) -> str:
        parent_hash = self._committed.state_hash()
        if candidate.next_state.step_index <= self._committed.step_index:
            raise ValueError("Candidate state does not advance the committed state")
        key = f"{action.value}:{candidate.next_state.state_hash()}"
        self._candidates[key] = candidate.next_state
        self.history.append((f"propose:{action.value}", parent_hash))
        return key

    def select(self, key: str) -> None:
        if key not in self._candidates:
            raise ValueError(f"Cannot select unknown candidate {key!r}")
        self._selected = key

    def commit_selected(self) -> HRMState:
        if self._selected is None:
            raise ValueError("No candidate selected; refusing to commit")
        state = self._candidates[self._selected]
        self._committed = state
        self.history.append((f"commit:{self._selected}", state.state_hash()))
        self._candidates.clear()
        self._selected = None
        return state

    def commit(self, state: HRMState) -> HRMState:
        """Direct commit path: valid only for the currently selected candidate."""

        if self._selected is None or self._candidates.get(self._selected) is not state:
            raise ValueError(
                "Refusing to commit a state that is not the selected candidate"
            )
        return self.commit_selected()

    def discard_candidates(self) -> None:
        """Reject all outstanding candidates; committed state is untouched."""

        self._candidates.clear()
        self._selected = None

    def stop(self) -> HRMState:
        """STOP performs no additional operation and must not mutate state."""

        self.discard_candidates()
        stopped = replace(self._committed, halted=True, last_action=ActionType.STOP)
        self.history.append(("stop", self._committed.state_hash()))
        # Halting is a terminal marker, not a computation: everything except
        # the halt flag and terminal action must be byte-identical.
        assert stopped.high_state is self._committed.high_state
        assert stopped.low_state is self._committed.low_state
        assert stopped.workspace_state is self._committed.workspace_state
        assert stopped.step_index == self._committed.step_index
        self._committed = stopped
        return stopped

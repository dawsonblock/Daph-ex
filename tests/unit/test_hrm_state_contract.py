"""State-commit contract: the selected state is exactly the committed state."""

from __future__ import annotations

import pytest
import torch

from hrm_adaptive_memory.hrm.state import (
    ActionType,
    HRMState,
    StateCommitLedger,
    StepResult,
)


def base_state(**overrides):
    defaults = dict(
        high_state=torch.zeros(2, 4),
        low_state=torch.ones(2, 4),
        workspace_state=None,
        step_index=0,
        reasoning_depth=0,
        last_action=None,
        halted=False,
        signature={"hidden_size": 1536, "high_layers": 2, "low_layers": 3, "schema": "hrm-text"},
    )
    defaults.update(overrides)
    return HRMState(**defaults)


def candidate(state: HRMState, action: ActionType, fill: float) -> StepResult:
    return StepResult(
        output=f"out-{fill}",
        next_state=state.advanced(action=action, high_state=torch.full((2, 4), fill)),
        diagnostics={"fill": fill},
    )


def test_selected_state_is_committed():
    ledger = StateCommitLedger(base_state())
    chosen = candidate(ledger.committed, ActionType.THINK, 1.0)
    other = candidate(ledger.committed, ActionType.RETRIEVE, 2.0)
    key = ledger.propose(ActionType.THINK, chosen)
    ledger.propose(ActionType.RETRIEVE, other)
    ledger.select(key)
    committed = ledger.commit_selected()
    assert committed.state_hash() == chosen.next_state.state_hash()
    assert torch.equal(committed.high_state, torch.full((2, 4), 1.0))
    assert ledger.committed.state_hash() == chosen.next_state.state_hash()


def test_rejected_state_is_not_committed():
    ledger = StateCommitLedger(base_state())
    chosen = candidate(ledger.committed, ActionType.THINK, 1.0)
    rejected = candidate(ledger.committed, ActionType.RETRIEVE, 9.0)
    key = ledger.propose(ActionType.THINK, chosen)
    ledger.propose(ActionType.RETRIEVE, rejected)
    ledger.select(key)
    ledger.commit_selected()
    assert not torch.equal(ledger.committed.high_state, torch.full((2, 4), 9.0))
    # The discarded candidate cannot be smuggled in afterwards.
    with pytest.raises(ValueError, match="not the selected candidate"):
        ledger.commit(rejected.next_state)


def test_commit_without_selection_fails_closed():
    ledger = StateCommitLedger(base_state())
    ledger.propose(ActionType.THINK, candidate(ledger.committed, ActionType.THINK, 1.0))
    with pytest.raises(ValueError, match="No candidate selected"):
        ledger.commit_selected()


def test_stop_does_not_mutate_state():
    ledger = StateCommitLedger(base_state())
    before = ledger.committed
    ledger.propose(ActionType.THINK, candidate(before, ActionType.THINK, 5.0))
    stopped = ledger.stop()
    assert stopped.high_state is before.high_state
    assert stopped.low_state is before.low_state
    assert stopped.step_index == before.step_index
    assert stopped.reasoning_depth == before.reasoning_depth
    assert stopped.halted is True
    assert stopped.last_action == ActionType.STOP


def test_verify_commit_contract():
    """VERIFY may advance state only through the ledger's selected candidate."""

    ledger = StateCommitLedger(base_state())
    verified = candidate(ledger.committed, ActionType.VERIFY, 3.0)
    key = ledger.propose(ActionType.VERIFY, verified)
    ledger.select(key)
    committed = ledger.commit_selected()
    assert committed.last_action == ActionType.VERIFY
    assert committed.reasoning_depth == 0, "VERIFY is not internal reasoning"
    assert committed.step_index == 1


def test_retrieve_commit_contract():
    ledger = StateCommitLedger(base_state())
    retrieved = candidate(ledger.committed, ActionType.RETRIEVE, 4.0)
    key = ledger.propose(ActionType.RETRIEVE, retrieved)
    ledger.select(key)
    committed = ledger.commit_selected()
    assert committed.last_action == ActionType.RETRIEVE
    assert committed.reasoning_depth == 0, "external compute is not internal reasoning"
    assert torch.equal(committed.high_state, torch.full((2, 4), 4.0))


def test_think_increments_reasoning_depth_only():
    ledger = StateCommitLedger(base_state())
    for expected_depth in (1, 2, 3):
        key = ledger.propose(ActionType.THINK, candidate(ledger.committed, ActionType.THINK, expected_depth))
        ledger.select(key)
        state = ledger.commit_selected()
        assert state.reasoning_depth == expected_depth
        assert state.step_index == expected_depth


def test_discarded_candidates_leave_committed_state_untouched():
    ledger = StateCommitLedger(base_state())
    before = ledger.committed.state_hash()
    ledger.propose(ActionType.THINK, candidate(ledger.committed, ActionType.THINK, 7.0))
    ledger.discard_candidates()
    assert ledger.committed.state_hash() == before
    with pytest.raises(ValueError, match="No candidate selected"):
        ledger.commit_selected()


def test_state_hash_is_content_addressed():
    left, right = base_state(), base_state()
    assert left.state_hash() == right.state_hash()
    assert left.state_hash() != base_state(high_state=torch.ones(2, 4)).state_hash()
    assert left.state_hash() != base_state(step_index=1).state_hash()


def test_incompatible_schema_is_rejected_without_migration():
    with pytest.raises(ValueError, match="No silent migration"):
        base_state(state_schema_version="hrm-state-v0")


def test_signature_mismatch_fails_closed_on_resume():
    state = base_state()
    state.validate_signature({"hidden_size": 1536, "high_layers": 2, "low_layers": 3, "schema": "hrm-text"})
    with pytest.raises(ValueError, match="signature mismatch"):
        state.validate_signature({"hidden_size": 2048})


def test_candidate_must_advance_the_committed_state():
    ledger = StateCommitLedger(base_state(step_index=5))
    stale = StepResult(output="x", next_state=base_state(step_index=5))
    with pytest.raises(ValueError, match="does not advance"):
        ledger.propose(ActionType.THINK, stale)

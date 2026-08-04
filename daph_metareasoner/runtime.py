"""On-path execution: only the selected action runs, under hard loop/budget limits."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Sequence

from .collector import ReasoningAdapter
from .policy import ConservativeVOCPolicy, ControllerDecision
from .schema import Action, ReasoningState, StopReason, Task


@dataclass(frozen=True)
class RuntimeLimits:
    max_steps: int = 4
    max_tokens: int = 512
    max_latency_ms: float = 30_000.0
    max_cost: float = 0.20
    action_repeat_limit: int = 2
    unchanged_answer_limit: int = 2
    recurrence_cosine_threshold: float = 0.995


@dataclass(frozen=True)
class RuntimeStep:
    step: int
    state_id: str
    answer_before: str
    decision: ControllerDecision
    answer_after: str
    latency_ms: float
    tokens: int
    cost: float


@dataclass(frozen=True)
class RuntimeResult:
    task_id: str
    initial_answer: str
    answer: str
    stop_reason: str
    total_steps: int
    total_tokens: int
    total_latency_ms: float
    total_cost: float
    trace: Sequence[RuntimeStep]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class LoopGuard:
    def __init__(self, limits: RuntimeLimits) -> None:
        self.limits = limits

    def blocked_actions(self, states: Sequence[ReasoningState], actions: Sequence[str]) -> set[str]:
        blocked: set[str] = set()
        for action in set(actions):
            if actions.count(action) >= self.limits.action_repeat_limit:
                blocked.add(action)
        if len(actions) >= 4 and actions[-4] == actions[-2] and actions[-3] == actions[-1]:
            blocked.update(action.value for action in Action if action is not Action.STOP)
        if states:
            current = states[-1]
            if current.repeated_answer_count >= self.limits.unchanged_answer_limit:
                blocked.update(action.value for action in Action if action is not Action.STOP)
            if (
                not current.answer_changed
                and current.hidden_cosine_previous >= self.limits.recurrence_cosine_threshold
            ):
                blocked.update(action.value for action in Action if action is not Action.STOP)
        return blocked


class OnPathExecutor:
    def __init__(
        self,
        adapter: ReasoningAdapter,
        policy: ConservativeVOCPolicy,
        limits: RuntimeLimits = RuntimeLimits(),
    ) -> None:
        self.adapter = adapter
        self.policy = policy
        self.limits = limits
        self.guard = LoopGuard(limits)

    def run(self, task: Task) -> RuntimeResult:
        state = self.adapter.initial_state(task, budget=self.limits.max_cost)
        initial_answer = state.answer
        states = [state]
        actions: List[str] = []
        trace: List[RuntimeStep] = []
        total_tokens = float(state.initial_input_tokens + state.initial_output_tokens)
        total_latency = float(state.initial_latency_ms)
        total_cost = 0.0
        stop_reason = StopReason.NON_POSITIVE_VOC.value
        for step in range(self.limits.max_steps + 1):
            if (
                step >= self.limits.max_steps
                or total_tokens >= self.limits.max_tokens
                or total_latency >= self.limits.max_latency_ms
                or total_cost >= self.limits.max_cost
            ):
                stop_reason = StopReason.BUDGET.value
                break
            blocked = self.guard.blocked_actions(states, actions)
            decision = self.policy.decide(state, blocked_actions=blocked)
            if decision.action == Action.STOP.value:
                stop_reason = decision.stop_reason
                break
            action = Action(decision.action)
            action_cost = self.policy.utility.action_cost(action)
            if total_cost + action_cost > self.limits.max_cost:
                stop_reason = StopReason.BUDGET.value
                break
            branch = self.adapter.execute(task, state, action)
            # Only this selected branch has executed. No counterfactual table is consulted.
            total_tokens += branch.receipt.input_tokens + branch.receipt.output_tokens
            total_latency += branch.receipt.latency_ms
            total_cost += action_cost
            trace.append(RuntimeStep(
                step=step,
                state_id=state.state_id,
                answer_before=state.answer,
                decision=decision,
                answer_after=branch.next_state.answer,
                latency_ms=branch.receipt.latency_ms,
                tokens=branch.receipt.input_tokens + branch.receipt.output_tokens,
                cost=action_cost,
            ))
            actions.append(action.value)
            state = branch.next_state
            states.append(state)
        return RuntimeResult(
            task_id=task.task_id,
            initial_answer=initial_answer,
            answer=state.answer,
            stop_reason=stop_reason,
            total_steps=len(trace),
            total_tokens=int(total_tokens),
            total_latency_ms=float(total_latency),
            total_cost=float(total_cost),
            trace=tuple(trace),
        )

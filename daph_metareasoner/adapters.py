"""Frozen base-model adapters and multi-depth hidden-state capture."""

from __future__ import annotations

import hashlib
import json
import math
import platform
import time
from typing import Any, Dict, Mapping, Sequence

import torch
from torch import Tensor

from .schema import Action, ActionReceipt, BranchResult, ReasoningState, Task, canonical_digest


ACTION_PROMPTS: Mapping[Action, str] = {
    Action.THINK: (
        "{prompt}\nCurrent proposed answer: {answer}\n"
        "Reconsider the problem independently. Correct the answer if needed. "
        "Return only the final answer."
    ),
    Action.VERIFY: (
        "{prompt}\nCurrent proposed answer: {answer}\n"
        "Verify this answer using an independent check. If it is wrong, fix it. "
        "Return only the final answer."
    ),
    Action.DECOMPOSE: (
        "{prompt}\nCurrent proposed answer: {answer}\n"
        "Decompose the problem into smaller steps and solve them carefully. "
        "Return only the final answer."
    ),
}


def _cosine(left: Sequence[float], right: Sequence[float]) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0
    a = torch.tensor(left, dtype=torch.float32)
    b = torch.tensor(right, dtype=torch.float32)
    denominator = float(a.norm() * b.norm())
    return 0.0 if denominator == 0.0 else float(torch.dot(a, b) / denominator)


class HFCausalLMAdapter:
    """One frozen Hugging Face causal LM with four explicit actions.

    ``transformers`` is imported only by ``from_pretrained`` so the base test
    and package environment remains independent of the optional dependency.
    """

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        *,
        model_id: str,
        revision: str,
        max_new_tokens: int = 12,
        pool_tokens: int = 4,
        compute_normalization_tokens: int = 256,
        model_digest: str | None = None,
        action_budget_costs: Mapping[str, float] | None = None,
        use_chat_template: bool | None = None,
    ) -> None:
        self.model = model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        self.tokenizer = tokenizer
        self.model_id = model_id
        self.revision = revision
        self.max_new_tokens = int(max_new_tokens)
        self.pool_tokens = int(pool_tokens)
        self.compute_normalization_tokens = max(1, int(compute_normalization_tokens))
        self.action_budget_costs = dict(action_budget_costs or {
            Action.THINK.value: 0.02,
            Action.VERIFY.value: 0.04,
            Action.DECOMPOSE.value: 0.03,
        })
        self.use_chat_template = (
            bool(getattr(tokenizer, "chat_template", None))
            if use_chat_template is None else bool(use_chat_template)
        )
        if getattr(self.tokenizer, "pad_token_id", None) is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        config_payload = (
            self.model.config.to_dict() if hasattr(self.model.config, "to_dict")
            else str(self.model.config)
        )
        self.model_digest = model_digest or canonical_digest({
            "model_id": model_id,
            "revision": revision,
            "config": config_payload,
        })
        self.environment_digest = canonical_digest({
            "python": platform.python_version(),
            "torch": torch.__version__,
            "platform": platform.platform(),
            "model_id": model_id,
            "revision": revision,
        })

    @classmethod
    def from_pretrained(
        cls,
        model_id: str,
        revision: str,
        *,
        device: str = "auto",
        dtype: torch.dtype = torch.float32,
        full_model_digest: bool = True,
        **kwargs: Any,
    ) -> "HFCausalLMAdapter":
        from transformers import AutoModelForCausalLM, AutoTokenizer

        resolved = (
            "mps" if device == "auto" and torch.backends.mps.is_available()
            else "cuda" if device == "auto" and torch.cuda.is_available()
            else "cpu" if device == "auto" else device
        )
        tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
        model = AutoModelForCausalLM.from_pretrained(
            model_id, revision=revision, dtype=dtype,
        ).to(torch.device(resolved))
        digest = None
        if full_model_digest:
            from daph.counterfactual import full_state_dict_digest
            digest = full_state_dict_digest(model)
        return cls(
            model, tokenizer, model_id=model_id, revision=revision,
            model_digest=digest, **kwargs,
        )

    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device

    @torch.no_grad()
    def _run(self, prompt: str, action: Action) -> tuple[str, Dict[str, Any], ActionReceipt]:
        model_prompt = prompt
        if self.use_chat_template:
            model_prompt = self.tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
            )
        encoded = self.tokenizer(model_prompt, return_tensors="pt", add_special_tokens=False)
        input_ids = encoded["input_ids"].to(self.device)
        attention_mask = encoded.get("attention_mask", torch.ones_like(input_ids)).to(self.device)
        started = time.perf_counter()
        generated = self.model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=self.max_new_tokens,
            do_sample=False,
            return_dict_in_generate=True,
            output_scores=True,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        sequence = generated.sequences
        completion_ids = sequence[:, input_ids.size(1):]
        answer = self.tokenizer.decode(completion_ids[0], skip_special_tokens=True).strip()
        output = self.model(sequence, output_hidden_states=True, use_cache=False)
        hidden_states = output.hidden_states
        layer_count = len(hidden_states) - 1
        pooled: Dict[str, tuple[float, ...]] = {}
        for label, fraction in (("25", 0.25), ("50", 0.50), ("75", 0.75), ("100", 1.0)):
            index = min(layer_count, max(1, math.ceil(layer_count * fraction)))
            hidden = hidden_states[index][0]
            pooled[label] = tuple(
                float(value) for value in hidden[-min(self.pool_tokens, hidden.size(0)):].mean(dim=0).float().cpu()
            )
        final_token = tuple(float(value) for value in hidden_states[-1][0, -1].float().cpu())
        logprob_values, entropy_values = [], []
        for score, token in zip(generated.scores, completion_ids[0]):
            distribution = torch.log_softmax(score[0].float(), dim=-1)
            probability = distribution.exp()
            logprob_values.append(float(distribution[int(token)]))
            entropy_values.append(float(-(probability * distribution).sum()))
        mean_logprob = sum(logprob_values) / max(len(logprob_values), 1)
        mean_entropy = sum(entropy_values) / max(len(entropy_values), 1)
        elapsed = (time.perf_counter() - started) * 1000.0
        input_tokens = int(input_ids.numel())
        output_tokens = int(completion_ids.numel())
        compute = (input_tokens + output_tokens) / self.compute_normalization_tokens
        receipt = ActionReceipt(
            action=action.value,
            latency_ms=elapsed,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            normalized_compute=compute,
            model_digest=self.model_digest,
            environment_digest=self.environment_digest,
        )
        features = {
            "hidden_by_depth": pooled,
            "hidden_final_token": final_token,
            "answer_entropy": mean_entropy,
            "answer_logprob": mean_logprob,
            "answer_confidence": math.exp(mean_logprob) if logprob_values else 0.0,
            "token_count": input_tokens + output_tokens,
        }
        return answer, features, receipt

    def initial_state(self, task: Task, *, budget: float) -> ReasoningState:
        answer, features, receipt = self._run(
            task.prompt + "\nReturn only the final answer.", Action.STOP,
        )
        return ReasoningState.create(
            task_id=task.task_id,
            step=0,
            answer=answer,
            prompt=task.prompt,
            budget_remaining=float(budget),
            compute_spent=receipt.normalized_compute,
            initial_latency_ms=receipt.latency_ms,
            initial_input_tokens=receipt.input_tokens,
            initial_output_tokens=receipt.output_tokens,
            **features,
        )

    def execute(self, task: Task, state: ReasoningState, action: Action) -> BranchResult:
        if action is Action.STOP:
            receipt = ActionReceipt(
                action=action.value, latency_ms=0.0, input_tokens=0, output_tokens=0,
                normalized_compute=0.0, model_digest=self.model_digest,
                environment_digest=self.environment_digest,
            )
            return BranchResult(action, state.state_id, state, receipt)
        prior = ""
        if state.evidence:
            prior = "\nPrior computation:\n" + "\n".join(state.evidence)
        branch_prompt = ACTION_PROMPTS[action].format(
            prompt=task.prompt + prior, answer=state.answer,
        )
        answer, features, receipt = self._run(branch_prompt, action)
        current_hidden = features["hidden_by_depth"].get("100", ())
        previous_hidden = state.hidden_by_depth.get("100", ())
        changed = answer.strip() != state.answer.strip()
        repeated = 0 if changed else state.repeated_answer_count + 1
        next_state = ReasoningState.create(
            task_id=task.task_id,
            step=state.step + 1,
            answer=answer,
            prompt=task.prompt,
            evidence=state.evidence + (f"{action.value}:{answer}",),
            action_history=state.action_history + (action.value,),
            compute_spent=state.compute_spent + receipt.normalized_compute,
            budget_remaining=max(
                0.0, state.budget_remaining - self.action_budget_costs[action.value],
            ),
            answer_changed=changed,
            hidden_cosine_previous=_cosine(previous_hidden, current_hidden),
            confidence_delta=float(features["answer_confidence"]) - state.answer_confidence,
            repeated_answer_count=repeated,
            **features,
        )
        return BranchResult(action, state.state_id, next_state, receipt)

"""E2-first, verifier-driven hard-case mining for E3 curricula."""

from __future__ import annotations

import hashlib
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class HardCaseMiningConfig:
    hard_failure_ratio: float = 0.60
    hard_uncertain_ratio: float = 0.20
    easy_correct_ratio: float = 0.20
    entropy_threshold: Optional[float] = None
    seed: int = 42
    max_new_tokens: int = 16
    strict_category_availability: bool = True
    require_mixed_e2_outcomes: bool = True

    def validate(self) -> None:
        ratios = (self.hard_failure_ratio, self.hard_uncertain_ratio, self.easy_correct_ratio)
        if any(value < 0 for value in ratios):
            raise ValueError("Hard-case sampling ratios must be non-negative")
        if abs(sum(ratios) - 1.0) > 1e-8:
            raise ValueError("Hard-case sampling ratios must sum to one")
        if self.max_new_tokens < 1:
            raise ValueError("max_new_tokens must be positive")


@dataclass(frozen=True)
class HardCaseRecord:
    task_id: str
    category: str
    e2_correct: bool
    e2_verifier_reward: float
    e2_entropy: Optional[float]
    e2_confidence: Optional[float]
    e2_ce: Optional[float]
    e2_answer: Optional[str]
    task_family: Optional[str]
    difficulty: Optional[str]
    task_digest: str
    task_payload: Dict[str, Any]


VerifierMapping = Mapping[str, Any]
VerifierTuple = Tuple[float, str]
VerifierFn = Callable[
    [Mapping[str, Any], Mapping[str, Any]],
    Union[VerifierMapping, VerifierTuple],
]


class E3HardCaseMiner:
    def __init__(
        self,
        model: torch.nn.Module,
        verifier_fn: VerifierFn,
        config: HardCaseMiningConfig,
        *,
        tokenizer: Optional[Any] = None,
    ) -> None:
        config.validate()
        self.model = model
        self.verifier_fn = verifier_fn
        self.config = config
        self.tokenizer = tokenizer

    def _device(self) -> torch.device:
        try:
            return next(self.model.parameters()).device
        except StopIteration:
            return torch.device("cpu")

    def _verification_output(
        self,
        ids: torch.Tensor,
        mask: Optional[torch.Tensor],
        forward_output: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        if self.tokenizer is None:
            return forward_output
        if not hasattr(self.model, "generate"):
            raise RuntimeError("Verifier-driven hard-case generation requires model.generate()")
        generated = self.model.generate(
            ids,
            attention_mask=mask,
            effort_mode="fixed_2",
            max_new_tokens=self.config.max_new_tokens,
            tokenizer=self.tokenizer,
        )
        return generated

    @staticmethod
    def _normalize_verdict(
        raw: Union[VerifierMapping, VerifierTuple],
        verification_output: Mapping[str, Any],
    ) -> Tuple[bool, float, Optional[str]]:
        if isinstance(raw, Mapping):
            if "correct" not in raw:
                raise ValueError("Mapping verifier results must contain 'correct'")
            correct = bool(raw["correct"])
            reward = float(raw.get("reward", correct))
            answer = raw.get("answer")
            return correct, reward, None if answer is None else str(answer)
        if isinstance(raw, tuple) and len(raw) == 2:
            reward, status = raw
            if status not in {"CORRECT", "INCORRECT", "UNVERIFIABLE", "EXECUTION_ERROR", "TIMEOUT"}:
                raise ValueError(f"Unknown verifier status: {status!r}")
            if status not in {"CORRECT", "INCORRECT"}:
                raise ValueError(
                    "Hard-case mining requires a verifiable E2 result; "
                    f"got status={status!r}. Supply a tokenizer for decoded generation "
                    "or remove the task from the mining corpus."
                )
            texts = verification_output.get("generated_text")
            answer = texts[0] if isinstance(texts, (list, tuple)) and texts else texts
            return status == "CORRECT", float(reward), None if answer is None else str(answer)
        raise TypeError("Verifier results must be a mapping or a (quality, status) tuple")

    @torch.no_grad()
    def mine(self, tasks: Iterable[Mapping[str, Any]]) -> List[HardCaseRecord]:
        records: List[HardCaseRecord] = []
        device = self._device()
        for index, task in enumerate(tasks):
            ids = torch.as_tensor(task["input_ids"], dtype=torch.long, device=device)
            if ids.dim() == 1:
                ids = ids.unsqueeze(0)
            mask = task.get("attention_mask")
            mask_tensor = torch.as_tensor(mask, device=device) if mask is not None else None
            if mask_tensor is not None and mask_tensor.dim() == 1:
                mask_tensor = mask_tensor.unsqueeze(0)
            out = self.model(ids, attention_mask=mask_tensor, effort_mode="fixed_2")
            logits = out["logits"] if isinstance(out, dict) else out
            probs = torch.softmax(logits[:, -1].float(), dim=-1)
            entropy = float((-(probs * probs.clamp_min(1e-12).log()).sum(dim=-1)).mean().item())
            confidence = float(probs.max(dim=-1).values.mean().item())
            e2_ce = None
            if task.get("labels") is not None:
                labels = torch.as_tensor(task["labels"], dtype=torch.long, device=logits.device)
                if labels.dim() == 1:
                    labels = labels.unsqueeze(0)
                e2_ce = float(F.cross_entropy(
                    logits[:, :-1].reshape(-1, logits.size(-1)), labels[:, 1:].reshape(-1),
                    ignore_index=-100,
                ).item())
            forward_output = dict(out) if isinstance(out, Mapping) else {"logits": logits}
            verification_output = self._verification_output(ids, mask_tensor, forward_output)
            raw_verdict = self.verifier_fn(verification_output, task)
            correct, reward, answer = self._normalize_verdict(raw_verdict, verification_output)
            uncertain = self.config.entropy_threshold is not None and entropy >= self.config.entropy_threshold
            category = "HARD_FAILURE" if not correct else (
                "HARD_UNCERTAIN" if uncertain else "EASY_CORRECT"
            )
            digest_payload = {key: task.get(key) for key in ("task_id", "input_ids", "expected", "verifier_spec")}
            digest = hashlib.sha256(json.dumps(digest_payload, sort_keys=True, default=str).encode()).hexdigest()
            records.append(HardCaseRecord(
                task_id=str(task.get("task_id", index)), category=category,
                e2_correct=correct, e2_verifier_reward=reward, e2_entropy=entropy,
                e2_confidence=confidence, e2_ce=e2_ce,
                e2_answer=answer, task_family=task.get("task_family"),
                difficulty=task.get("difficulty_bucket"), task_digest=digest,
                task_payload=json.loads(json.dumps(dict(task), default=lambda value: value.tolist() if isinstance(value, torch.Tensor) else str(value))),
            ))
        return records

    def sample(self, records: Sequence[HardCaseRecord], count: int) -> List[HardCaseRecord]:
        sampled, _ = self.sample_with_manifest(records, count)
        return sampled

    def sample_with_manifest(
        self, records: Sequence[HardCaseRecord], count: int,
    ) -> Tuple[List[HardCaseRecord], Dict[str, Any]]:
        """Sample the declared curriculum and report requested versus realized mix."""
        if count < 1:
            raise ValueError("Hard-case sample count must be positive")
        rng = random.Random(self.config.seed)
        buckets: Dict[str, List[HardCaseRecord]] = {key: [] for key in ("HARD_FAILURE", "HARD_UNCERTAIN", "EASY_CORRECT")}
        for record in records:
            buckets[record.category].append(record)
        desired = {
            "HARD_FAILURE": self.config.hard_failure_ratio,
            "HARD_UNCERTAIN": self.config.hard_uncertain_ratio,
            "EASY_CORRECT": self.config.easy_correct_ratio,
        }
        exact = {category: count * ratio for category, ratio in desired.items()}
        quotas = {category: math.floor(value) for category, value in exact.items()}
        for category in sorted(exact, key=lambda key: (-(exact[key] - quotas[key]), key))[:count - sum(quotas.values())]:
            quotas[category] += 1
        missing = [category for category, quota in quotas.items() if quota and not buckets[category]]
        if missing and self.config.strict_category_availability:
            raise ValueError(
                "Hard-case curriculum cannot satisfy configured mix; missing categories: "
                + ", ".join(missing)
            )
        if self.config.require_mixed_e2_outcomes and count > 1:
            available_correctness = {record.e2_correct for record in records}
            if len(available_correctness) < 2:
                raise ValueError("Hard-case curriculum is degenerate: E2 outcomes are all correct or all incorrect")
        sampled: List[HardCaseRecord] = []
        for category in desired:
            take = quotas[category]
            values = buckets[category]
            if values:
                sampled.extend(rng.choice(values) for _ in range(take))
        all_records = list(records)
        while len(sampled) < count and all_records:
            sampled.append(rng.choice(all_records))
        rng.shuffle(sampled)
        sampled = sampled[:count]
        realized = {
            category: sum(record.category == category for record in sampled)
            for category in desired
        }
        manifest = {
            "config": asdict(self.config),
            "source_record_count": len(records),
            "source_task_ids": [record.task_id for record in records],
            "requested_count": count,
            "requested_quotas": quotas,
            "realized_counts": realized,
            "realized_ratios": {category: value / max(len(sampled), 1) for category, value in realized.items()},
            "e2_successes": sum(record.e2_correct for record in sampled),
            "e2_failures": sum(not record.e2_correct for record in sampled),
        }
        return sampled, manifest

    def save(self, records: Sequence[HardCaseRecord], output_dir: str) -> None:
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        with (output / "hard_cases.jsonl").open("w") as handle:
            for record in records:
                handle.write(json.dumps(asdict(record), sort_keys=True) + "\n")
        (output / "mining_manifest.json").write_text(json.dumps({
            "config": asdict(self.config),
            "records": len(records),
            "source_task_ids": [record.task_id for record in records],
            "counts": {category: sum(r.category == category for r in records) for category in (
                "HARD_FAILURE", "HARD_UNCERTAIN", "EASY_CORRECT"
            )},
        }, indent=2))

    def save_sample(self, records: Sequence[HardCaseRecord], count: int, output_dir: str) -> None:
        """Persist an exact curriculum sample and its requested/realized receipt."""
        sampled, manifest = self.sample_with_manifest(records, count)
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        with (output / "hardcase_sample.jsonl").open("w") as handle:
            for record in sampled:
                handle.write(json.dumps(asdict(record), sort_keys=True) + "\n")
        (output / "hardcase_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")

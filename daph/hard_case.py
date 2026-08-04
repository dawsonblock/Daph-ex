"""E2-first, verifier-driven hard-case mining for E3 curricula."""

from __future__ import annotations

import hashlib
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class HardCaseMiningConfig:
    hard_failure_ratio: float = 0.60
    hard_uncertain_ratio: float = 0.20
    easy_correct_ratio: float = 0.20
    entropy_threshold: Optional[float] = None
    seed: int = 42

    def validate(self) -> None:
        ratios = (self.hard_failure_ratio, self.hard_uncertain_ratio, self.easy_correct_ratio)
        if any(value < 0 for value in ratios):
            raise ValueError("Hard-case sampling ratios must be non-negative")
        if abs(sum(ratios) - 1.0) > 1e-8:
            raise ValueError("Hard-case sampling ratios must sum to one")


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


VerifierFn = Callable[[Mapping[str, Any], Mapping[str, Any]], Mapping[str, Any]]


class E3HardCaseMiner:
    def __init__(self, model: torch.nn.Module, verifier_fn: VerifierFn, config: HardCaseMiningConfig) -> None:
        config.validate()
        self.model = model
        self.verifier_fn = verifier_fn
        self.config = config

    @torch.no_grad()
    def mine(self, tasks: Iterable[Mapping[str, Any]]) -> List[HardCaseRecord]:
        records: List[HardCaseRecord] = []
        for index, task in enumerate(tasks):
            ids = torch.as_tensor(task["input_ids"], dtype=torch.long)
            if ids.dim() == 1:
                ids = ids.unsqueeze(0)
            mask = task.get("attention_mask")
            mask_tensor = torch.as_tensor(mask) if mask is not None else None
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
            verdict = dict(self.verifier_fn({"logits": logits}, task))
            correct = bool(verdict["correct"])
            reward = float(verdict.get("reward", correct))
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
                e2_answer=verdict.get("answer"), task_family=task.get("task_family"),
                difficulty=task.get("difficulty_bucket"), task_digest=digest,
                task_payload=json.loads(json.dumps(dict(task), default=lambda value: value.tolist() if isinstance(value, torch.Tensor) else str(value))),
            ))
        return records

    def sample(self, records: Sequence[HardCaseRecord], count: int) -> List[HardCaseRecord]:
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
        return sampled[:count]

    def save(self, records: Sequence[HardCaseRecord], output_dir: str) -> None:
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        with (output / "hard_cases.jsonl").open("w") as handle:
            for record in records:
                handle.write(json.dumps(asdict(record), sort_keys=True) + "\n")
        (output / "mining_manifest.json").write_text(json.dumps({
            "config": asdict(self.config),
            "records": len(records),
            "counts": {category: sum(r.category == category for r in records) for category in (
                "HARD_FAILURE", "HARD_UNCERTAIN", "EASY_CORRECT"
            )},
        }, indent=2))

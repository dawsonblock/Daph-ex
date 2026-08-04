"""Binary continuation probes and uncertainty-aware action-value ensembles."""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .analysis import group_by_state
from .features import FeatureNormalizer, StateVectorizer
from .schema import ALL_ACTIONS, Action, ExperienceRecord, ReasoningState, records_digest


ACTION_NAMES: Tuple[str, ...] = tuple(action.value for action in ALL_ACTIONS)


class ContinuationProbe(nn.Module):
    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, values: Tensor) -> Tensor:
        return self.linear(values).squeeze(-1)


class ActionValueModel(nn.Module):
    def __init__(
        self, input_dim: int, hidden_dim: int = 512,
        second_hidden_dim: int = 256, dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, second_hidden_dim),
            nn.SiLU(),
        )
        self.value_head = nn.Linear(second_hidden_dim, len(ACTION_NAMES))
        self.correctness_head = nn.Linear(second_hidden_dim, 1)
        nn.init.normal_(self.value_head.weight, std=1e-3)
        nn.init.zeros_(self.value_head.bias)
        nn.init.normal_(self.correctness_head.weight, std=1e-3)
        nn.init.zeros_(self.correctness_head.bias)

    def forward(self, values: Tensor) -> Tensor:
        value, _ = self.forward_all(values)
        return value

    def predict_correctness_logit(self, values: Tensor) -> Tensor:
        _, correctness = self.forward_all(values)
        return correctness

    def forward_all(self, values: Tensor) -> tuple[Tensor, Tensor]:
        encoded = self.encoder(values)
        return self.value_head(encoded), self.correctness_head(encoded).squeeze(-1)


@dataclass(frozen=True)
class ProbeTrainingConfig:
    epochs: int = 100
    lr: float = 1e-2
    weight_decay: float = 1e-4
    epsilon: float = 0.0
    seed: int = 42


@dataclass(frozen=True)
class ValueTrainingConfig:
    epochs: int = 150
    lr: float = 1e-3
    weight_decay: float = 1e-4
    hidden_dim: int = 512
    second_hidden_dim: int = 256
    dropout: float = 0.1
    batch_size: int = 256
    rank_weight: float = 0.25
    correctness_weight: float = 0.25
    rank_margin: float = 0.02
    ensemble_size: int = 5
    seed: int = 42


def _state_examples(
    records: Sequence[ExperienceRecord], vectorizer: StateVectorizer,
) -> tuple[Tensor, Tensor, Tensor, Tensor, List[ReasoningState]]:
    grouped = group_by_state(records)
    states, features, helps, targets, correctness = [], [], [], [], []
    for rows in grouped.values():
        state = next(iter(rows.values())).state
        states.append(state)
        features.append(vectorizer(state))
        helps.append(max(rows[action].delta_utility for action in ACTION_NAMES if action != Action.STOP.value))
        targets.append([rows[action].delta_quality for action in ACTION_NAMES])
        correctness.append(float(next(iter(rows.values())).quality_before > 0.0))
    widths = {len(row) for row in features}
    if len(widths) != 1:
        raise ValueError(f"Feature widths differ across states: {sorted(widths)}")
    return (
        torch.tensor(features, dtype=torch.float32),
        torch.tensor(helps, dtype=torch.float32),
        torch.tensor(targets, dtype=torch.float32),
        torch.tensor(correctness, dtype=torch.float32),
        states,
    )


def _binary_metrics(labels: Tensor, probabilities: Tensor) -> Dict[str, float | None]:
    labels = labels.float().cpu()
    probabilities = probabilities.float().cpu().clamp(0.0, 1.0)
    positives = int(labels.sum().item())
    negatives = int(labels.numel() - positives)
    if positives and negatives:
        positive_scores = probabilities[labels == 1]
        negative_scores = probabilities[labels == 0]
        comparisons = positive_scores[:, None] - negative_scores[None, :]
        auroc = float((
            (comparisons > 0).float() + 0.5 * (comparisons == 0).float()
        ).mean())
        order = torch.argsort(probabilities, descending=True)
        sorted_labels = labels[order]
        precision = torch.cumsum(sorted_labels, 0) / torch.arange(1, len(labels) + 1)
        auprc = float(precision[sorted_labels == 1].mean())
    else:
        auroc = auprc = None
    brier = float(torch.mean((probabilities - labels) ** 2))
    ece = 0.0
    for bin_index, lower in enumerate(torch.linspace(0.0, 0.9, 10)):
        upper = lower + 0.1
        mask = (probabilities >= lower) & (
            probabilities <= upper if bin_index == 9 else probabilities < upper
        )
        if bool(mask.any()):
            ece += float(mask.float().mean()) * abs(
                float(probabilities[mask].mean()) - float(labels[mask].mean())
            )
    return {"auroc": auroc, "auprc": auprc, "brier": brier, "ece": ece}


def train_probe(
    train_records: Sequence[ExperienceRecord],
    validation_records: Sequence[ExperienceRecord],
    vectorizer: StateVectorizer,
    config: ProbeTrainingConfig = ProbeTrainingConfig(),
) -> tuple[ContinuationProbe, FeatureNormalizer, Dict[str, float | None]]:
    torch.manual_seed(config.seed)
    train_x, train_help, _, _, _ = _state_examples(train_records, vectorizer)
    val_x, val_help, _, _, _ = _state_examples(validation_records, vectorizer)
    normalizer = FeatureNormalizer.fit(train_x)
    train_x, val_x = normalizer.transform(train_x), normalizer.transform(val_x)
    train_y = (train_help > config.epsilon).float()
    val_y = (val_help > config.epsilon).float()
    model = ContinuationProbe(train_x.size(1))
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    positive = float(train_y.sum())
    negative = float(train_y.numel() - positive)
    pos_weight = torch.tensor(
        negative / positive if positive > 0.0 and negative > 0.0 else 1.0
    )
    for _ in range(config.epochs):
        optimizer.zero_grad(set_to_none=True)
        loss = F.binary_cross_entropy_with_logits(model(train_x), train_y, pos_weight=pos_weight)
        loss.backward()
        optimizer.step()
    model.eval()
    with torch.no_grad():
        probability = torch.sigmoid(model(val_x))
    return model, normalizer, _binary_metrics(val_y, probability)


def _ranking_loss(prediction: Tensor, target: Tensor, margin: float) -> Tensor:
    best = target.argmax(dim=1)
    best_prediction = prediction.gather(1, best[:, None])
    mask = torch.ones_like(prediction, dtype=torch.bool)
    mask.scatter_(1, best[:, None], False)
    alternatives = prediction[mask].view(prediction.size(0), -1)
    return F.relu(margin - best_prediction + alternatives).mean()


class ActionValueEnsemble:
    def __init__(
        self, models: Sequence[ActionValueModel], normalizer: FeatureNormalizer,
        vectorizer: StateVectorizer,
    ) -> None:
        if not models:
            raise ValueError("ActionValueEnsemble requires at least one model")
        self.models = list(models)
        self.normalizer = normalizer
        self.vectorizer = vectorizer
        for model in self.models:
            model.eval()

    @torch.no_grad()
    def predict(self, state: ReasoningState) -> tuple[Dict[str, float], Dict[str, float]]:
        means, stds, _, _ = self.predict_state(state)
        return means, stds

    @torch.no_grad()
    def predict_state(
        self, state: ReasoningState,
    ) -> tuple[Dict[str, float], Dict[str, float], float, float]:
        raw = torch.tensor([self.vectorizer(state)], dtype=torch.float32)
        values = self.normalizer.transform(raw)
        outputs = [model.forward_all(values) for model in self.models]
        predictions = torch.stack([value[0] for value, _ in outputs])
        correctness = torch.stack([torch.sigmoid(logit)[0] for _, logit in outputs])
        means = predictions.mean(dim=0)
        stds = predictions.std(dim=0, unbiased=False)
        return (
            {action: float(means[index]) for index, action in enumerate(ACTION_NAMES)},
            {action: float(stds[index]) for index, action in enumerate(ACTION_NAMES)},
            float(correctness.mean()),
            float(correctness.std(unbiased=False)),
        )

    def save(
        self, path: str | Path, *, training_digest: str,
        config: ValueTrainingConfig, training_status: str = "UNVERIFIED_FIT",
        evaluation_digest: str = "", base_model_digest: str = "",
    ) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        state_dicts = [model.state_dict() for model in self.models]
        artifact = {
            "feature_spec": self.vectorizer.spec,
            "normalizer": self.normalizer.to_dict(),
            "input_dim": int(self.normalizer.mean.numel()),
            "hidden_dim": config.hidden_dim,
            "second_hidden_dim": config.second_hidden_dim,
            "dropout": config.dropout,
            "training_digest": training_digest,
            "training_config": asdict(config),
            "action_names": ACTION_NAMES,
            "training_status": training_status,
            "evaluation_digest": evaluation_digest,
            "base_model_digest": base_model_digest,
            "state_dicts_digest": _state_dicts_digest(state_dicts),
        }
        digest = hashlib.sha256(json.dumps(artifact, sort_keys=True).encode()).hexdigest()
        artifact["artifact_digest"] = digest
        torch.save({"artifact": artifact, "state_dicts": state_dicts}, destination)

    @classmethod
    def load(cls, path: str | Path) -> tuple["ActionValueEnsemble", Mapping[str, Any]]:
        payload = torch.load(path, map_location="cpu", weights_only=True)
        artifact = dict(payload["artifact"])
        claimed = artifact.pop("artifact_digest")
        actual = hashlib.sha256(json.dumps(artifact, sort_keys=True).encode()).hexdigest()
        if actual != claimed:
            raise ValueError("Controller artifact metadata digest mismatch")
        if _state_dicts_digest(payload["state_dicts"]) != artifact["state_dicts_digest"]:
            raise ValueError("Controller state_dict digest mismatch")
        artifact["artifact_digest"] = claimed
        models = []
        for state_dict in payload["state_dicts"]:
            model = ActionValueModel(
                artifact["input_dim"], artifact["hidden_dim"],
                artifact["second_hidden_dim"], artifact["dropout"],
            )
            model.load_state_dict(state_dict)
            models.append(model)
        ensemble = cls(
            models,
            FeatureNormalizer.from_dict(artifact["normalizer"]),
            StateVectorizer(artifact["feature_spec"]),
        )
        return ensemble, artifact


def train_value_ensemble(
    train_records: Sequence[ExperienceRecord],
    vectorizer: StateVectorizer,
    config: ValueTrainingConfig = ValueTrainingConfig(),
) -> ActionValueEnsemble:
    train_x, _, target, correctness, _ = _state_examples(train_records, vectorizer)
    normalizer = FeatureNormalizer.fit(train_x)
    train_x = normalizer.transform(train_x)
    models = []
    for member in range(config.ensemble_size):
        seed = config.seed + member
        torch.manual_seed(seed)
        rng = random.Random(seed)
        sample = torch.tensor([rng.randrange(train_x.size(0)) for _ in range(train_x.size(0))])
        x_member, y_member = train_x[sample], target[sample]
        correctness_member = correctness[sample]
        model = ActionValueModel(
            train_x.size(1), config.hidden_dim, config.second_hidden_dim, config.dropout,
        )
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=config.lr, weight_decay=config.weight_decay,
        )
        model.train()
        generator = torch.Generator().manual_seed(seed)
        for _ in range(config.epochs):
            permutation = torch.randperm(x_member.size(0), generator=generator)
            for start in range(0, x_member.size(0), config.batch_size):
                batch = permutation[start:start + config.batch_size]
                optimizer.zero_grad(set_to_none=True)
                prediction, correctness_logit = model.forward_all(x_member[batch])
                target_batch = y_member[batch]
                loss = F.mse_loss(prediction, target_batch)
                loss = loss + config.rank_weight * _ranking_loss(
                    prediction, target_batch, config.rank_margin,
                )
                loss = loss + config.correctness_weight * F.binary_cross_entropy_with_logits(
                    correctness_logit, correctness_member[batch],
                )
                loss.backward()
                optimizer.step()
        model.eval()
        models.append(model)
    return ActionValueEnsemble(models, normalizer, vectorizer)


def training_digest(records: Sequence[ExperienceRecord]) -> str:
    return records_digest(list(records))


def _state_dicts_digest(state_dicts: Sequence[Mapping[str, Tensor]]) -> str:
    digest = hashlib.sha256()
    for member, state_dict in enumerate(state_dicts):
        digest.update(str(member).encode())
        for name in sorted(state_dict):
            tensor = state_dict[name].detach().cpu().contiguous().reshape(-1)
            digest.update(name.encode())
            digest.update(str(tensor.dtype).encode())
            digest.update(str(tuple(tensor.shape)).encode())
            digest.update(bytes(tensor.view(torch.uint8).numpy()))
    return digest.hexdigest()

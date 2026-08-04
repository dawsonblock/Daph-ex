"""State feature extraction with explicit hidden and cheap-proxy variants."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, Tuple

import torch
from torch import Tensor

from .schema import ReasoningState


FEATURE_SPECS = (
    "cheap",
    "hidden_final",
    "hidden_multidepth",
    "hidden_runtime",
    "hidden_convergence",
)


@dataclass(frozen=True)
class StateVectorizer:
    spec: str = "hidden_runtime"

    def __post_init__(self) -> None:
        if self.spec not in FEATURE_SPECS:
            raise ValueError(f"Unknown feature spec {self.spec!r}; expected one of {FEATURE_SPECS}")

    def __call__(self, state: ReasoningState) -> Tuple[float, ...]:
        cheap = state.cheap_features()
        if self.spec == "cheap":
            return cheap
        if self.spec == "hidden_final":
            values = tuple(state.hidden_by_depth.get("100", state.hidden_final_token))
            if not values:
                raise ValueError(f"State {state.state_id} has no final hidden features")
            return values
        hidden = state.hidden_features()
        if not hidden:
            raise ValueError(f"State {state.state_id} has no multi-depth hidden features")
        if self.spec == "hidden_multidepth":
            return hidden
        if self.spec == "hidden_runtime":
            return hidden + cheap
        return hidden + cheap + (float(state.hidden_cosine_previous),)


@dataclass
class FeatureNormalizer:
    mean: Tensor
    scale: Tensor

    @classmethod
    def fit(cls, values: Tensor) -> "FeatureNormalizer":
        if values.dim() != 2 or values.size(0) < 1:
            raise ValueError("FeatureNormalizer expects a non-empty 2D tensor")
        mean = values.mean(dim=0)
        scale = values.std(dim=0, unbiased=False).clamp_min(1e-6)
        return cls(mean=mean, scale=scale)

    def transform(self, values: Tensor) -> Tensor:
        return (values - self.mean.to(values.device)) / self.scale.to(values.device)

    def to_dict(self) -> dict[str, list[float]]:
        return {"mean": self.mean.tolist(), "scale": self.scale.tolist()}

    @classmethod
    def from_dict(cls, payload: dict[str, Sequence[float]]) -> "FeatureNormalizer":
        return cls(
            mean=torch.tensor(payload["mean"], dtype=torch.float32),
            scale=torch.tensor(payload["scale"], dtype=torch.float32),
        )

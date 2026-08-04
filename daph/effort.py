"""
Adaptive Effort Controller + cost-aware loss + early-exit hooks.

Philosophy (K3 multi-effort RL inspiration, adapted to experimental scale):
  - Predict a soft distribution over discrete effort levels.
  - Train with a utility objective: Quality − λ · Cost.
  - At inference, threshold or sample to decide early-exit / skip expensive ops.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class EffortController(nn.Module):
    """
    Predicts soft effort over discrete levels.

    Level semantics (default):
      0 — minimal / early-exit candidate (cheap residual only)
      1 — recurrent + LatentMoE
      2 — + global attention
      3 — + extra compute (verifier, deeper MoE, tools — future)

    Can operate at sequence level (mean-pooled) or token level.
    """

    def __init__(
        self,
        hidden_size: int,
        num_levels: int = 4,
        hidden_router: int = 128,
        token_level: bool = False,
        base_costs: Optional[List[float]] = None,
    ) -> None:
        super().__init__()
        self.num_levels = num_levels
        self.token_level = token_level

        self.norm = nn.LayerNorm(hidden_size)
        self.net = nn.Sequential(
            nn.Linear(hidden_size, hidden_router),
            nn.GELU(),
            nn.Linear(hidden_router, num_levels),
        )

        costs = base_costs or [0.1, 1.0, 2.5, 5.0]
        costs = costs[:num_levels]
        while len(costs) < num_levels:
            costs.append(costs[-1] * 1.8)
        self.register_buffer(
            "base_costs",
            torch.tensor(costs, dtype=torch.float32),
        )


    @staticmethod
    def pool_last_valid(hidden: Tensor, attention_mask: Optional[Tensor] = None) -> Tensor:
        """Pool last valid token: (B, L, H) → (B, H)."""
        if hidden.dim() != 3:
            return hidden
        B, L, H = hidden.shape
        if attention_mask is None:
            return hidden[:, -1, :]
        lengths = attention_mask.long().sum(dim=1).clamp(min=1) - 1
        idx = lengths.view(B, 1, 1).expand(-1, 1, H)
        return hidden.gather(1, idx).squeeze(1)

    def forward(
        self,
        hidden_states: Tensor,
        difficulty: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        """
        Args:
            hidden_states: (B, L, H) or (B, H)
            difficulty: optional (B, L) or (B,) in [0, 1]

        Returns:
            dict with logits, probs, score, expected_cost
            (token-level shapes when token_level=True and input is 3-D)
        """
        # Accept (B, H) anchor directly (preferred for policy consistency)
        # or (B, L, H) with optional mask for last-valid / masked mean.
        if hidden_states.dim() == 2:
            features = hidden_states  # (B, H) — already pooled
        elif hidden_states.dim() == 3 and not self.token_level:
            features = hidden_states.mean(dim=1)  # legacy fallback
        else:
            features = hidden_states

        logits = self.net(self.norm(features))  # (..., levels)
        probs = F.softmax(logits, dim=-1)

        # Broadcast costs
        costs = self.base_costs
        while costs.dim() < probs.dim():
            costs = costs.unsqueeze(0)
        expected_cost = (probs * costs).sum(dim=-1)

        level_ids = torch.arange(
            self.num_levels, device=probs.device, dtype=probs.dtype
        )
        while level_ids.dim() < probs.dim():
            level_ids = level_ids.unsqueeze(0)
        effort_score = (probs * level_ids).sum(dim=-1) / max(self.num_levels - 1, 1)

        # Optional difficulty bias: higher difficulty → shift mass to higher levels
        if difficulty is not None:
            # soft push: add a small bias proportional to difficulty
            bias = difficulty.unsqueeze(-1) * 0.5  # scale
            # only bias the higher levels
            if logits.dim() == 2:
                logits = logits + bias * torch.linspace(0, 1, self.num_levels, device=logits.device)
            else:
                logits = logits + bias.unsqueeze(-1) * torch.linspace(
                    0, 1, self.num_levels, device=logits.device
                )
            probs = F.softmax(logits, dim=-1)
            expected_cost = (probs * costs).sum(dim=-1)
            effort_score = (probs * level_ids).sum(dim=-1) / max(self.num_levels - 1, 1)

        return {
            "effort_logits": logits,
            "effort_probs": probs,
            "effort_score": effort_score,
            "expected_cost": expected_cost,
        }

    def hard_level(self, probs: Tensor) -> Tensor:
        return probs.argmax(dim=-1)

    def should_early_exit(
        self,
        probs: Tensor,
        threshold: float = 0.55,
        min_level: int = 0,
    ) -> Tensor:
        """
        Returns a boolean mask (same leading dims as probs) indicating
        tokens/sequences that can early-exit (mass concentrated on low levels).
        """
        low_mass = probs[..., : min_level + 1].sum(dim=-1)
        return low_mass >= threshold

    def sample_level(
        self,
        probs: Tensor,
        temperature: float = 1.0,
        deterministic: bool = False,
    ) -> Tensor:
        if deterministic:
            return self.hard_level(probs)
        if temperature != 1.0:
            logits = torch.log(probs.clamp(min=1e-8)) / temperature
            probs = F.softmax(logits, dim=-1)
        return torch.multinomial(probs.view(-1, self.num_levels), 1).view(probs.shape[:-1])


def effort_cost_loss(
    effort_info: Dict[str, Tensor],
    quality_proxy: Tensor,
    lambda_cost: float = 0.1,
    target_effort: Optional[Tensor] = None,
) -> Dict[str, Tensor]:
    """
    Cost-aware auxiliary loss.

    U = Quality − λ · Cost
    We maximize U ⇒ minimize −Quality + λ · Cost.

    Args:
        effort_info: output of EffortController.forward
        quality_proxy: scalar or (B,) / (B,L) — higher is better
                       (e.g. −CE, reward, margin, confidence)
        lambda_cost: cost weight
        target_effort: optional soft target distribution for KL regularization

    Returns:
        dict with total aux loss and components
    """
    expected_cost = effort_info["expected_cost"]

    # Align shapes
    if quality_proxy.dim() == 0:
        quality_proxy = quality_proxy.expand_as(expected_cost)
    elif quality_proxy.shape != expected_cost.shape:
        # broadcast or mean-reduce as needed
        while quality_proxy.dim() < expected_cost.dim():
            quality_proxy = quality_proxy.unsqueeze(-1)
        if quality_proxy.shape != expected_cost.shape:
            quality_proxy = quality_proxy.mean(dim=tuple(range(quality_proxy.dim() - expected_cost.dim(), quality_proxy.dim())))

    # Maximize quality − λ cost  ≡  minimize −quality + λ cost
    utility_loss = -quality_proxy + lambda_cost * expected_cost
    loss = utility_loss.mean()

    components = {
        "effort_aux_loss": loss,
        "mean_expected_cost": expected_cost.mean().detach(),
        "mean_quality_proxy": quality_proxy.mean().detach(),
    }

    if target_effort is not None:
        # KL( current || target ) regularizer
        log_p = F.log_softmax(effort_info["effort_logits"], dim=-1)
        kl = F.kl_div(log_p, target_effort, reduction="batchmean", log_target=False)
        loss = loss + 0.05 * kl
        components["effort_kl"] = kl.detach()
        components["effort_aux_loss"] = loss

    return components


def early_exit_mask_from_effort(
    effort_info: Dict[str, Tensor],
    threshold: float = 0.55,
    max_exit_level: int = 0,
) -> Tensor:
    """
    Convenience: boolean mask of positions that may early-exit.
    """
    probs = effort_info["effort_probs"]
    low_mass = probs[..., : max_exit_level + 1].sum(dim=-1)
    return low_mass >= threshold

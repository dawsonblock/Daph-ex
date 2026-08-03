"""
Shared-weight latent refinement for effort level E3.

h^{t+1} = F_θ(h^t)  with the same parameters reused across steps.
Optional compact workspace slots for limited state carry-over.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor


class LatentRefineBlock(nn.Module):
    """
    Compact residual refinement used for E3 extended reasoning.

    Same weights applied for `num_steps` iterations.
    """

    def __init__(
        self,
        hidden_size: int,
        expansion: float = 2.0,
        dropout: float = 0.0,
        workspace_slots: int = 0,
    ) -> None:
        super().__init__()
        mid = int(hidden_size * expansion)
        self.norm = nn.LayerNorm(hidden_size)
        self.fc1 = nn.Linear(hidden_size, mid, bias=False)
        self.fc2 = nn.Linear(mid, hidden_size, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.workspace_slots = workspace_slots
        if workspace_slots > 0:
            self.workspace_proj = nn.Linear(hidden_size, workspace_slots, bias=False)
            self.workspace_out = nn.Linear(workspace_slots, hidden_size, bias=False)
        else:
            self.workspace_proj = None
            self.workspace_out = None

    def forward(
        self,
        hidden: Tensor,
        num_steps: int = 1,
        workspace: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """
        hidden: (B, L, H)
        num_steps: number of shared-weight refinement iterations
        workspace: optional (B, slots)

        Returns (refined_hidden, workspace)
        """
        if num_steps < 1:
            return hidden, workspace

        h = hidden
        for _ in range(num_steps):
            x = self.norm(h)
            x = self.fc2(torch.nn.functional.gelu(self.fc1(x)))
            x = self.dropout(x)
            if self.workspace_proj is not None:
                # pool last-token into workspace, inject back
                pooled = x[:, -1, :]  # (B, H)
                ws = torch.tanh(self.workspace_proj(pooled))  # (B, slots)
                if workspace is not None:
                    ws = 0.5 * ws + 0.5 * workspace
                inject = self.workspace_out(ws).unsqueeze(1)  # (B, 1, H)
                x = x + inject
                workspace = ws
            h = h + x
        return h, workspace

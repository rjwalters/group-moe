"""Symmetry Router.

A lightweight classifier that examines the current activation and decides:
1. Which group (if any) is active
2. Which element of that group applies
3. A confidence score for blending with pass-through
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..groups.representations import GroupRepresentation


@dataclass
class RoutingDecision:
    """Output of the symmetry router."""
    group_idx: torch.Tensor      # (batch,) which group (0 = pass-through)
    element_idx: torch.Tensor    # (batch,) which element within the group
    confidence: torch.Tensor     # (batch,) blend weight for group expert output
    logits: torch.Tensor         # (batch, n_options) raw logits for analysis


class SymmetryRouter(nn.Module):
    """Routes activations to the appropriate group expert.

    The router outputs a flat distribution over all options:
    [pass-through, group1_elem0, group1_elem1, ..., group2_elem0, ...]

    The pass-through option always exists — if no symmetry is detected,
    the activation passes through unchanged.
    """

    def __init__(
        self,
        d_model: int,
        groups: list[GroupRepresentation],
        hidden_dim: int | None = None,
    ):
        super().__init__()
        self.d_model = d_model
        self.groups = groups

        # Total options: 1 (pass-through) + sum of group orders
        self.n_options = 1 + sum(g.order for g in groups)

        # Build the index mapping: option_idx -> (group_idx, element_idx)
        # group_idx = 0 means pass-through
        self.option_map: list[tuple[int, int]] = [(0, 0)]  # pass-through
        for g_idx, group in enumerate(groups):
            for e_idx in range(group.order):
                self.option_map.append((g_idx + 1, e_idx))

        # Small MLP router
        h = hidden_dim or min(d_model // 4, 256)
        self.router = nn.Sequential(
            nn.Linear(d_model, h),
            nn.GELU(),
            nn.Linear(h, self.n_options),
        )

    def forward(self, x: torch.Tensor, temperature: float = 1.0) -> RoutingDecision:
        """Route each activation to a group expert or pass-through.

        Args:
            x: (batch, d_model) activations
            temperature: softmax temperature (lower = more decisive)

        Returns:
            RoutingDecision with hard routing (argmax) and soft confidence
        """
        logits = self.router(x)  # (batch, n_options)

        # Hard routing: pick the best option
        best = torch.argmax(logits, dim=-1)  # (batch,)

        # Soft confidence: how confident are we in this choice?
        probs = F.softmax(logits / temperature, dim=-1)
        confidence = probs.gather(1, best.unsqueeze(1)).squeeze(1)  # (batch,)

        # Map to (group_idx, element_idx)
        group_idx = torch.zeros_like(best)
        element_idx = torch.zeros_like(best)
        for i, (g, e) in enumerate(self.option_map):
            mask = best == i
            group_idx[mask] = g
            element_idx[mask] = e

        return RoutingDecision(
            group_idx=group_idx,
            element_idx=element_idx,
            confidence=confidence,
            logits=logits,
        )

    def routing_stats(self, decisions: RoutingDecision) -> dict[str, float]:
        """Summarize routing decisions for monitoring."""
        batch = decisions.group_idx.shape[0]
        stats = {
            "pass_through_rate": (decisions.group_idx == 0).float().mean().item(),
            "mean_confidence": decisions.confidence.mean().item(),
        }
        for g_idx, group in enumerate(self.groups):
            rate = (decisions.group_idx == g_idx + 1).float().mean().item()
            stats[f"{group.name}_rate"] = rate
        return stats

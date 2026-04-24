"""Group-MoE Layer.

The main module combining the symmetry router with group expert modules.
Drops in as a layer that can be inserted into a transformer.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ..groups.representations import GroupRepresentation
from .expert import GroupExpert
from .router import SymmetryRouter, RoutingDecision


class GroupMoELayer(nn.Module):
    """A mixture-of-experts layer where each expert implements a group representation.

    For each input activation:
    1. The router decides which group (if any) applies
    2. If a group is selected, the corresponding expert applies R(g) in a learned subspace
    3. The result is blended with the original via the confidence score
    4. If pass-through is selected, the activation passes unchanged

    This is a residual module: output = x + confidence * expert(x)
    """

    def __init__(
        self,
        d_model: int,
        groups: list[GroupRepresentation],
        router_hidden_dim: int | None = None,
    ):
        super().__init__()
        self.d_model = d_model
        self.groups = groups

        # Router
        self.router = SymmetryRouter(d_model, groups, hidden_dim=router_hidden_dim)

        # One expert per group
        self.experts = nn.ModuleList([
            GroupExpert(d_model, group) for group in groups
        ])

    def forward(
        self, x: torch.Tensor, temperature: float = 1.0
    ) -> tuple[torch.Tensor, RoutingDecision]:
        """Apply the Group-MoE layer.

        Args:
            x: (batch, d_model) or (batch, seq_len, d_model)

        Returns:
            (output, routing_decision)
        """
        # Handle sequence dimension
        has_seq = x.dim() == 3
        if has_seq:
            batch, seq_len, d = x.shape
            x_flat = x.reshape(batch * seq_len, d)
        else:
            x_flat = x

        # Route
        decision = self.router(x_flat, temperature=temperature)

        # Apply experts
        output = x_flat.clone()
        for g_idx, expert in enumerate(self.experts):
            # Find tokens routed to this group
            mask = decision.group_idx == (g_idx + 1)
            if not mask.any():
                continue

            x_group = x_flat[mask]
            elem_idx = decision.element_idx[mask]
            conf = decision.confidence[mask].unsqueeze(-1)

            # Apply expert and blend
            delta = expert(x_group, elem_idx) - x_group
            output[mask] = x_group + conf * delta

        # Reshape back
        if has_seq:
            output = output.reshape(batch, seq_len, d)

        return output, decision

    def param_summary(self) -> dict[str, int]:
        """Parameter count breakdown."""
        router_params = sum(p.numel() for p in self.router.parameters())
        expert_params = {
            g.name: sum(p.numel() for p in e.parameters())
            for g, e in zip(self.groups, self.experts)
        }
        total = router_params + sum(expert_params.values())
        full_matrix_equiv = sum(e.full_matrix_param_count for e in self.experts)
        return {
            "router": router_params,
            "experts": expert_params,
            "total": total,
            "full_matrix_equivalent": full_matrix_equiv,
            "compression_ratio": full_matrix_equiv / max(total, 1),
        }

"""Standard MoE Layer — same architecture as GroupMoE but with learned transforms.

This is the controlled comparison: same router, same project-inject structure,
same residual blending, but the expert applies a LEARNED k×k transformation
instead of a FIXED irrep matrix R(g). If Group-MoE outperforms this, the
advantage comes from the group structure, not just the routing architecture.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .router import RoutingDecision


class MLPExpert(nn.Module):
    """Expert with learned transform in a low-rank subspace.

    Architecture mirrors GroupExpert:
        x → project(d→k) → transform(k→k) → inject(k→d) → x'

    But instead of a fixed irrep matrix R(g), the transform is a learned
    linear layer. Multiple "slots" (analogous to group elements) are
    provided, each with its own learned k×k matrix.
    """

    def __init__(self, d_model: int, k: int, n_slots: int):
        super().__init__()
        self.d_model = d_model
        self.k = k
        self.n_slots = n_slots

        self.project = nn.Linear(d_model, k, bias=False)
        self.inject = nn.Linear(k, d_model, bias=False)

        # n_slots learned k×k transforms (analogous to R(g) for each element)
        self.transforms = nn.Parameter(torch.randn(n_slots, k, k) * 0.01)

    def forward(self, x: torch.Tensor, slot_idx: torch.Tensor) -> torch.Tensor:
        z = self.project(x)  # (batch, k)
        W = self.transforms[slot_idx]  # (batch, k, k)
        z_prime = torch.bmm(W, z.unsqueeze(-1)).squeeze(-1)  # (batch, k)
        return self.inject(z_prime)


class StandardMoELayer(nn.Module):
    """Standard MoE with same structure as GroupMoELayer.

    Args:
        d_model: activation dimension
        n_experts: number of expert modules
        slots_per_expert: number of routing slots per expert (like group elements)
        expert_dim: inner dimension of each expert (like irrep total_dim)
        router_hidden_dim: hidden dimension for the router MLP
    """

    def __init__(
        self,
        d_model: int,
        n_experts: int = 1,
        slots_per_expert: int = 6,
        expert_dim: int = 4,
        router_hidden_dim: int | None = None,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_experts = n_experts

        # Total routing options: 1 (pass-through) + sum of slots
        self.n_options = 1 + n_experts * slots_per_expert

        # Map option_idx → (expert_idx, slot_idx), 0 = pass-through
        self.option_map: list[tuple[int, int]] = [(0, 0)]
        for e in range(n_experts):
            for s in range(slots_per_expert):
                self.option_map.append((e + 1, s))

        # Router (same architecture as SymmetryRouter)
        h = router_hidden_dim or min(d_model // 4, 256)
        self.router = nn.Sequential(
            nn.Linear(d_model, h),
            nn.GELU(),
            nn.Linear(h, self.n_options),
        )

        # MLP experts
        self.experts = nn.ModuleList([
            MLPExpert(d_model, expert_dim, slots_per_expert)
            for _ in range(n_experts)
        ])

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, RoutingDecision]:
        has_seq = x.dim() == 3
        if has_seq:
            batch, seq_len, d = x.shape
            x_flat = x.reshape(batch * seq_len, d)
        else:
            x_flat = x

        # Route
        logits = self.router(x_flat)
        best = torch.argmax(logits, dim=-1)
        probs = F.softmax(logits, dim=-1)
        confidence = probs.gather(1, best.unsqueeze(1)).squeeze(1)

        # Map to (expert_idx, slot_idx)
        expert_idx = torch.zeros_like(best)
        slot_idx = torch.zeros_like(best)
        for i, (e, s) in enumerate(self.option_map):
            mask = best == i
            expert_idx[mask] = e
            slot_idx[mask] = s

        decision = RoutingDecision(
            group_idx=expert_idx,
            element_idx=slot_idx,
            confidence=confidence,
            logits=logits,
        )

        # Apply experts
        output = x_flat.clone()
        for e_idx, expert in enumerate(self.experts):
            mask = expert_idx == (e_idx + 1)
            if not mask.any():
                continue
            x_exp = x_flat[mask]
            s_idx = slot_idx[mask]
            conf = confidence[mask].unsqueeze(-1)
            delta = expert(x_exp, s_idx) - x_exp
            output[mask] = x_exp + conf * delta

        if has_seq:
            output = output.reshape(batch, seq_len, d)

        return output, decision

"""Per-atom symmetry-type router for the SO(3) Group-MoE.

In Paper 1's `SymmetryRouter`, the router emitted (group_idx, element_idx,
confidence) — meaningful for finite groups where the element is a discrete
rotation/permutation. For the continuous SO(3) case, the element index has no
analogue: every rotation is "valid" and the router can't usefully classify
over them. So this router emits *only* a categorical label over symmetry types
(plus pass-through), and the chosen expert handles the rotation internally.

Output shape: (n_atoms, K+1) logits, one per atom in the batched molecular
graph. K = number of expert symmetry types (e.g. 3 for tetrahedral / octahedral
/ planar) and the +1 is the pass-through option.

The router input is the atom's scalar (l=0) features only. We deliberately
withhold the l=1 (vector) features from the router so the routing decision is
rotation-invariant: rotating the molecule must not change which expert each
atom is routed to. (The expert applies the rotation; the router decides
*whether* to apply it.)
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


# Convention: index 0 = pass-through, indices 1..K = expert symmetry types.
PASS_THROUGH_IDX = 0


@dataclass
class MolecularRoutingDecision:
    """Per-atom routing for the SO(3) MoE layer."""
    expert_idx: torch.Tensor   # (n_atoms,) long — 0=pass-through, 1..K=expert i
    confidence: torch.Tensor   # (n_atoms,) float — softmax prob of chosen option (for blending)
    logits: torch.Tensor       # (n_atoms, K+1) raw logits (for load-balancing loss + analysis)


class MolecularRouter(nn.Module):
    """Routes each atom to one of K symmetry-type experts or pass-through.

    Parameters:
        scalar_dim: dimension of the per-atom l=0 (scalar) features fed in.
        n_experts: K = number of expert symmetry types. Total options = K+1.
        hidden_dim: width of the router's internal MLP. Defaults to scalar_dim // 2,
            capped at 128 — the router should be cheap relative to the experts.
    """

    def __init__(self, scalar_dim: int, n_experts: int, hidden_dim: int | None = None):
        super().__init__()
        self.scalar_dim = scalar_dim
        self.n_experts = n_experts
        self.n_options = n_experts + 1  # +1 for pass-through

        h = hidden_dim if hidden_dim is not None else min(scalar_dim // 2, 128)
        self.mlp = nn.Sequential(
            nn.Linear(scalar_dim, h),
            nn.GELU(),
            nn.Linear(h, self.n_options),
        )

    def forward(
        self, scalar_features: torch.Tensor, temperature: float = 1.0
    ) -> MolecularRoutingDecision:
        """Route each atom.

        Args:
            scalar_features: (n_atoms, scalar_dim) — l=0 features only.
            temperature: softmax temperature (lower = more decisive).

        Returns:
            MolecularRoutingDecision.
        """
        logits = self.mlp(scalar_features)  # (n_atoms, K+1)
        expert_idx = torch.argmax(logits, dim=-1)
        probs = F.softmax(logits / temperature, dim=-1)
        confidence = probs.gather(1, expert_idx.unsqueeze(1)).squeeze(1)
        return MolecularRoutingDecision(
            expert_idx=expert_idx,
            confidence=confidence,
            logits=logits,
        )

    def routing_stats(
        self, decision: MolecularRoutingDecision, expert_names: list[str] | None = None
    ) -> dict[str, float]:
        """Summarize routing distribution. Useful for monitoring router collapse."""
        n_atoms = decision.expert_idx.shape[0]
        stats = {
            "pass_through_rate": (decision.expert_idx == PASS_THROUGH_IDX).float().mean().item(),
            "mean_confidence": decision.confidence.mean().item(),
        }
        for k in range(self.n_experts):
            label = expert_names[k] if expert_names else f"expert_{k}"
            rate = (decision.expert_idx == (k + 1)).float().mean().item()
            stats[f"{label}_rate"] = rate
        return stats


def load_balancing_loss(logits: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Standard MoE load-balancing loss (Switch Transformer style).

    Encourages roughly uniform usage of experts (and pass-through). Without
    this, the router tends to collapse to one option early in training. The
    loss is the dot product of:
      - mean assignment probability per option (soft)
      - mean argmax-fraction per option (hard)
    times the number of options. Minimum at uniform usage.

    Args:
        logits: (n_atoms, n_options) router logits.
        eps: numerical floor.

    Returns:
        Scalar tensor — add to the main loss with a small weight (e.g. 0.01).
    """
    n_atoms, n_options = logits.shape
    probs = F.softmax(logits, dim=-1)               # (n, K+1)
    mean_prob = probs.mean(dim=0)                    # (K+1,) average soft prob per option

    hard = torch.argmax(logits, dim=-1)              # (n,)
    one_hot = F.one_hot(hard, num_classes=n_options).float()  # (n, K+1)
    mean_assign = one_hot.mean(dim=0)                # (K+1,) hard fraction per option

    # n_options × <mean_prob, mean_assign>; minimum (=1) at uniform usage.
    return n_options * (mean_prob * mean_assign).sum().clamp(min=eps)

"""Molecular Group-MoE layer for SO(3) experts.

Composes:
  - MolecularRouter: per-atom categorical decision over K+1 options
                     (K experts + pass-through), based on the atom's scalar features.
  - K SO3Experts:    SO(3)-equivariant residual blocks, each operating on the
                     full irrep features (scalar + vector + tensor).
  - Load-balancing loss to prevent router collapse.

Per-atom dispatch: each atom in the (PyG-batched) graph independently picks an
expert. This is the per-token-routing analogue from language MoE literature.

Design choice — shared irreps across all experts: every expert uses the same
working irreps (the union of the design-doc per-type specs). Specialization
emerges from the routing distribution and learned weights, not from fixed
irrep filters. Per-expert irreps with adapter layers is a clean follow-up
ablation; deferred to keep v1 simple.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ..groups.continuous import SymmetryType, shared_irreps
from .molecular_router import (
    MolecularRouter,
    MolecularRoutingDecision,
    PASS_THROUGH_IDX,
    load_balancing_loss,
)
from .so3_expert import SO3Expert


class MolecularMoE(nn.Module):
    """Group-MoE layer with SO(3) experts and per-atom routing.

    Parameters:
        scalar_dim:        l=0 feature dimension fed to the router
        symmetry_types:    list of K SymmetryType configs (defines the experts)
        load_balance_weight: coefficient on the load-balancing loss returned in
            forward(). Caller should add this term to the main loss.

    The layer's expert irreps are the union (`shared_irreps`) of all symmetry
    types' irreps. Inputs/outputs use that shape.
    """

    def __init__(
        self,
        scalar_dim: int,
        symmetry_types: list[SymmetryType],
        load_balance_weight: float = 0.01,
        random_route: bool = False,
    ):
        super().__init__()
        self.symmetry_types = symmetry_types
        self.n_experts = len(symmetry_types)
        self.load_balance_weight = load_balance_weight
        # Ablation flag — replace the learned router's argmax with a uniform random
        # assignment over (n_experts + 1) options at every forward pass. Tests whether
        # the gain over plain SchNet comes from the learned routing decisions or just
        # from having an MoE block with experts. Load-balancing loss is set to zero
        # in this mode (random routing trivially balances in expectation; the loss
        # term wouldn't drive learning anyway since the router isn't being used).
        self.random_route = random_route

        self.irreps = shared_irreps(symmetry_types)

        self.router = MolecularRouter(scalar_dim, n_experts=self.n_experts)
        self.experts = nn.ModuleList([SO3Expert(self.irreps) for _ in symmetry_types])

    def forward(
        self,
        irrep_features: torch.Tensor,
        scalar_features: torch.Tensor,
        temperature: float = 1.0,
    ) -> tuple[torch.Tensor, MolecularRoutingDecision, torch.Tensor]:
        """Apply per-atom routing + expert dispatch.

        Args:
            irrep_features: (n_atoms, irreps.dim) — full equivariant features
                (scalars + vectors + optional tensors). The lifting layer in
                `schnet_groupmoe.py` produces this shape from SchNet's scalar
                features + atomic coordinates.
            scalar_features: (n_atoms, scalar_dim) — l=0 components only,
                used as the router input. Decoupling router input from expert
                input keeps routing rotation-invariant.
            temperature: softmax temperature for confidence (lower = more decisive).

        Returns:
            output: (n_atoms, irreps.dim) — equivariantly updated features.
                Pass-through atoms keep their input unchanged. Expert atoms
                get `x + confidence * (expert(x) - x)`.
            decision: MolecularRoutingDecision (per-atom routing record).
            lb_loss: scalar load-balancing loss (already weighted by
                load_balance_weight; add directly to main loss).
        """
        if irrep_features.shape[0] != scalar_features.shape[0]:
            raise ValueError(
                f"n_atoms mismatch: irrep_features has {irrep_features.shape[0]}, "
                f"scalar_features has {scalar_features.shape[0]}"
            )

        decision = self.router(scalar_features, temperature=temperature)
        if self.random_route:
            n_atoms = scalar_features.shape[0]
            n_options = self.n_experts + 1
            decision.expert_idx = torch.randint(
                0, n_options, (n_atoms,),
                device=scalar_features.device, dtype=decision.expert_idx.dtype,
            )
            # Use full conviction so the residual blend uses the expert fully.
            decision.confidence = torch.ones_like(decision.confidence)

        # Pass-through atoms keep their input; only expert-routed atoms get updated.
        output = irrep_features.clone()
        for k, expert in enumerate(self.experts):
            expert_label = k + 1  # index 0 = pass-through
            mask = decision.expert_idx == expert_label
            if not mask.any():
                continue
            x_masked = irrep_features[mask]
            conf = decision.confidence[mask].unsqueeze(-1)
            transformed = expert(x_masked)  # SO3Expert is residual: returns x + delta
            delta = transformed - x_masked
            output[mask] = x_masked + conf * delta

        if self.random_route:
            # Don't apply load-balance pressure when router is random — random routing
            # is already balanced in expectation and the loss term wouldn't be informative.
            lb_loss = torch.zeros((), device=irrep_features.device)
        else:
            lb_loss = self.load_balance_weight * load_balancing_loss(decision.logits)
        return output, decision, lb_loss

    def param_summary(self) -> dict[str, int]:
        """Parameter count breakdown for paper tables / sanity checks."""
        router_params = sum(p.numel() for p in self.router.parameters())
        expert_params = {
            t.name: sum(p.numel() for p in e.parameters())
            for t, e in zip(self.symmetry_types, self.experts)
        }
        return {
            "router": router_params,
            "experts": expert_params,
            "total": router_params + sum(expert_params.values()),
            "irreps": str(self.irreps),
        }

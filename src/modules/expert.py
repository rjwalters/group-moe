"""Group Expert module.

Applies a group representation in a learned subspace of the full
activation space. The key component of Group-MoE.

Architecture:
    x ∈ R^d  →  project (P)  →  z ∈ R^k  →  R(g)  →  z' ∈ R^k  →  inject (P†)  →  x' ∈ R^d

Where:
    P: learned projection d → k (into symmetry-active subspace)
    R(g): irrep block-diagonal matrix (k × k, very sparse)
    P†: learned injection k → d (back to full space)
    k = sum of irrep dimensions (e.g., 4 for S_3)
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ..groups.representations import GroupRepresentation


class GroupExpert(nn.Module):
    """Expert module that applies group representations in a learned subspace.

    Parameters:
        d_model: dimension of the full activation space
        group: the group representation to use
        learn_projection: if True, learn the projection P; if False, use random fixed
    """

    def __init__(
        self,
        d_model: int,
        group: GroupRepresentation,
        learn_projection: bool = True,
    ):
        super().__init__()
        self.d_model = d_model
        self.group = group
        self.k = group.total_dim

        # Learned projection into symmetry subspace
        self.project = nn.Linear(d_model, self.k, bias=False)
        self.inject = nn.Linear(self.k, d_model, bias=False)

        if not learn_projection:
            self.project.weight.requires_grad_(False)
            self.inject.weight.requires_grad_(False)

        # Register irrep matrices as buffers (not parameters — fixed group structure)
        self.register_buffer(
            "irrep_matrices",
            group.all_matrices(),  # (order, k, k)
        )

    def forward(self, x: torch.Tensor, element_idx: torch.Tensor) -> torch.Tensor:
        """Apply group element to x in the learned subspace.

        Args:
            x: (batch, d_model) activation vectors
            element_idx: (batch,) integer indices into group elements

        Returns:
            x': (batch, d_model) transformed activations
        """
        # Project into symmetry subspace
        z = self.project(x)  # (batch, k)

        # Apply group element (gather the right matrix for each batch item)
        R = self.irrep_matrices[element_idx]  # (batch, k, k)
        z_prime = torch.bmm(R, z.unsqueeze(-1)).squeeze(-1)  # (batch, k)

        # Inject back to full space
        x_prime = self.inject(z_prime)  # (batch, d_model)

        return x_prime

    def compose(self, x: torch.Tensor, elem1: int, elem2: int) -> torch.Tensor:
        """Apply g1 * g2 by composing in irrep basis (exact by construction).

        This demonstrates the key advantage: composition is guaranteed correct.
        """
        z = self.project(x)
        R1 = self.irrep_matrices[elem1]
        R2 = self.irrep_matrices[elem2]
        # R(g1*g2) = R(g1) @ R(g2) — exact by group homomorphism
        z_prime = (R1 @ R2) @ z.unsqueeze(-1)
        return self.inject(z_prime.squeeze(-1))

    @property
    def param_count(self) -> int:
        """Number of learnable parameters (projection + injection)."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    @property
    def full_matrix_param_count(self) -> int:
        """Parameters that would be needed for a full d×d matrix per element."""
        return self.group.order * self.d_model ** 2

    @property
    def compression_ratio(self) -> float:
        """How much more efficient this is vs full matrices."""
        return self.full_matrix_param_count / max(self.param_count, 1)

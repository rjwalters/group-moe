"""Scalar → irrep lifting layer.

Converts SchNet's scalar (l=0) atom features into the (l=0 + l=1 + l=2 …)
features the SO(3) experts expect, using neighbor geometry. This is the
"option (a)" architecture from `docs/paper2_design.md` discussion: the
lifting lives in the host model (`schnet_groupmoe.py`) rather than inside
each expert, so the expert can stay clean.

This is a one-block PaiNN-style message pass:

  for each edge (i → j) within cutoff:
      Y_l(r̂_ij) = spherical harmonics of unit displacement   [equivariant]
      w_ij      = MLP(scalar_features_i)                       [learned, scalar]
      m_ij      = w_ij × Y_l(r̂_ij) × envelope(r_ij)           [equivariant]
  high-l_atom = scatter_sum_{i → j} m_ij over j

  l=0 atom_features = Linear(scalar_features)

  output = concat([l=0 part, high-l part])

Equivariance is structural: the only operation that touches geometry is
`spherical_harmonics`, which transforms correctly under rotation, and the
weights w_ij are scalars (l=0) so they don't break it.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
from e3nn import o3
from torch_scatter import scatter


def _split_irreps(irreps: o3.Irreps) -> tuple[o3.Irreps, o3.Irreps]:
    """Partition irreps into (scalar = l=0, high-l = l>0). Order preserved."""
    scalar = o3.Irreps([(mul, ir) for mul, ir in irreps if ir.l == 0])
    high_l = o3.Irreps([(mul, ir) for mul, ir in irreps if ir.l > 0])
    return scalar, high_l


class ScalarToIrrepLift(nn.Module):
    """Lift per-atom scalar features to per-atom (scalar + l>0) irrep features.

    Parameters:
        scalar_dim: input scalar feature width (e.g. SchNet's hidden_channels).
        irreps: target irreps for the output. Must be in standard order:
            scalars (l=0) first, then ascending l. (DEFAULT_SYMMETRY_TYPES
            presets all follow this.)
        cutoff: edge cutoff radius (Å) — must match the host SchNet's cutoff.

    The output `irreps.dim` matches what the molecular MoE expects.
    """

    def __init__(self, scalar_dim: int, irreps: o3.Irreps, cutoff: float = 5.0):
        super().__init__()
        self.irreps = irreps
        self.cutoff = cutoff

        scalar_irreps, high_l_irreps = _split_irreps(irreps)
        self.scalar_irreps = scalar_irreps
        self.high_l_irreps = high_l_irreps

        # Order check: standard layout has all scalars before any l>0
        for mul, ir in irreps[: len(scalar_irreps)]:
            if ir.l != 0:
                raise ValueError(
                    f"irreps must list all l=0 channels before higher l. Got: {irreps}"
                )

        # Scalar pathway: simple linear scalar_dim → scalar_irreps.dim
        self.linear_scalar = nn.Linear(scalar_dim, scalar_irreps.dim)

        # High-l pathway: per-edge weighted spherical harmonics aggregated to atoms.
        #
        # Per-edge weights = MLP(source atom's scalar features) → one scalar
        # per output channel of high_l_irreps (sum-of-multiplicities total).
        if high_l_irreps.num_irreps == 0:
            self.has_high_l = False
            return
        self.has_high_l = True

        n_high_l_channels = sum(mul for mul, _ in high_l_irreps)
        self.edge_weight_mlp = nn.Sequential(
            nn.Linear(scalar_dim, scalar_dim),
            nn.SiLU(),
            nn.Linear(scalar_dim, n_high_l_channels),
        )

        # Spherical harmonics produced per-edge with multiplicity 1 — TP expands
        # to multiplicity using the per-edge weights.
        self.sh_irreps = o3.Irreps([(1, ir) for _, ir in high_l_irreps])
        weight_irreps = o3.Irreps(f"{n_high_l_channels}x0e")
        self.tp = o3.FullyConnectedTensorProduct(
            irreps_in1=weight_irreps,
            irreps_in2=self.sh_irreps,
            irreps_out=high_l_irreps,
            internal_weights=True,
            shared_weights=True,
        )

    def forward(
        self,
        scalar_features: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor,
        edge_vec: torch.Tensor,
    ) -> torch.Tensor:
        """Build per-atom irrep features from scalars + edge geometry.

        Args:
            scalar_features: (n_atoms, scalar_dim).
            edge_index: (2, n_edges) — (src, dst) per edge.
            edge_weight: (n_edges,) — distances ||r_src - r_dst||.
            edge_vec: (n_edges, 3) — displacement r_src - r_dst.

        Returns:
            (n_atoms, irreps.dim) per-atom features in the irreps basis.
        """
        n_atoms = scalar_features.shape[0]
        atom_scalar = self.linear_scalar(scalar_features)  # (n_atoms, scalar_irreps.dim)

        if not self.has_high_l:
            return atom_scalar

        src, dst = edge_index[0], edge_index[1]

        # Unit edge vectors. Guard against tiny edges (self-loops zeroed).
        eps = 1e-8
        norm = edge_weight.clamp(min=eps).unsqueeze(-1)
        edge_unit = edge_vec / norm

        # Cosine cutoff envelope: smooth zero at r = cutoff.
        envelope = 0.5 * (torch.cos(math.pi * edge_weight / self.cutoff) + 1.0)
        envelope = envelope * (edge_weight < self.cutoff).float()

        # Per-edge weights from source atom's scalar features.
        w = self.edge_weight_mlp(scalar_features[src])  # (n_edges, n_high_l_channels)
        w = w * envelope.unsqueeze(-1)

        # Spherical harmonics per edge, single multiplicity per (l, parity).
        sh = o3.spherical_harmonics(
            self.sh_irreps, edge_unit, normalize=False, normalization="component"
        )  # (n_edges, sh_irreps.dim)

        # TP combines scalar weights with SH to expand to multi-mul high-l features.
        edge_high_l = self.tp(w, sh)  # (n_edges, high_l_irreps.dim)

        # Aggregate edge messages to atoms (sum over neighbors).
        atom_high_l = scatter(edge_high_l, dst, dim=0, dim_size=n_atoms, reduce="sum")

        # Concatenate. Output ordering matches `self.irreps` because of standard layout.
        return torch.cat([atom_scalar, atom_high_l], dim=-1)

"""SchNet + Group-MoE: SchNet with one inserted SO(3)-equivariant MoE block.

This is the headline model for Paper 2. Architecture (per `docs/paper2_design.md`):

    embedding(z)
      ↓
    SchNet interaction × moe_position    (invariant message passing)
      ↓
    ScalarToIrrepLift                    (build l=0+l=1+l=2 features from geometry)
      ↓
    MolecularMoE                         (per-atom: route to expert or pass-through)
      ↓
    scalar projection back               (reduce to scalars; SchNet's tail is scalar-only)
      ↓
    SchNet interaction × (num_interactions - moe_position)
      ↓
    SchNet readout (lin1 / act / lin2 / atomref / mean&std / sum-pool)

The MoE output is reduced back to scalars by taking the l=0 channels only and
projecting to SchNet's hidden width. This is the simplest reducer; future
ablation: include ||l=1|| and ||l=2|| norms (rotation-invariant scalars) for
a richer summary. Either way the equivariant block does meaningful work
because (a) the lifting layer creates l>0 features from geometry, (b) the
MoE's tensor-product layers mix scalars and vectors equivariantly, (c) the
resulting l=0 channels are nontrivial functions of molecular geometry that
pure-scalar SchNet can't construct.

Forward returns (energy, routing_decision, lb_loss). The training script must
add lb_loss to the main loss (it's already weighted by load_balance_weight).
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch_geometric.nn import SchNet

from ..groups.continuous import DEFAULT_SYMMETRY_TYPES, SymmetryType, shared_irreps
from ..modules.molecular_moe import MolecularMoE
from ..modules.molecular_router import MolecularRoutingDecision
from ..modules.scalar_to_irrep import ScalarToIrrepLift


class SchNetGroupMoE(SchNet):
    """SchNet baseline with a Group-MoE block inserted between interaction layers.

    All non-MoE arguments forward to `torch_geometric.nn.SchNet`. New args:
        symmetry_types: list of K SymmetryType configs (defaults to
            DEFAULT_SYMMETRY_TYPES = tetrahedral / octahedral / planar).
        moe_position: 1-indexed position of the MoE block in the interaction
            stack. `moe_position=k` means the MoE runs after k SchNet
            interactions, before the remaining `num_interactions - k`.
            Default: num_interactions // 2.
        load_balance_weight: coefficient on the load-balancing loss returned
            in forward(). Same convention as MolecularMoE.

    The model assumes `num_interactions >= 2` so there's at least one
    interaction on either side of the MoE.
    """

    def __init__(
        self,
        hidden_channels: int = 128,
        num_filters: int = 128,
        num_interactions: int = 6,
        num_gaussians: int = 50,
        cutoff: float = 5.0,
        interaction_graph: nn.Module | None = None,
        max_num_neighbors: int = 32,
        readout: str = "add",
        dipole: bool = False,
        mean: float | None = None,
        std: float | None = None,
        atomref: torch.Tensor | None = None,
        symmetry_types: list[SymmetryType] | None = None,
        moe_position: int | None = None,
        load_balance_weight: float = 0.01,
        include_irrep_norms: bool = False,
        random_route: bool = False,
    ):
        super().__init__(
            hidden_channels=hidden_channels,
            num_filters=num_filters,
            num_interactions=num_interactions,
            num_gaussians=num_gaussians,
            cutoff=cutoff,
            interaction_graph=interaction_graph,
            max_num_neighbors=max_num_neighbors,
            readout=readout,
            dipole=dipole,
            mean=mean,
            std=std,
            atomref=atomref,
        )
        if num_interactions < 2:
            raise ValueError("SchNetGroupMoE needs num_interactions >= 2")

        self.symmetry_types = symmetry_types or DEFAULT_SYMMETRY_TYPES
        self.moe_position = moe_position if moe_position is not None else num_interactions // 2
        if not (1 <= self.moe_position < num_interactions):
            raise ValueError(
                f"moe_position must be in [1, {num_interactions - 1}], got {self.moe_position}"
            )

        self.expert_irreps = shared_irreps(self.symmetry_types)
        self.lift = ScalarToIrrepLift(
            scalar_dim=hidden_channels, irreps=self.expert_irreps, cutoff=cutoff,
        )
        self.moe = MolecularMoE(
            scalar_dim=hidden_channels,
            symmetry_types=self.symmetry_types,
            load_balance_weight=load_balance_weight,
            random_route=random_route,
        )

        # Reducer: project rotation-invariant features of the MoE output back to
        # SchNet's scalar pipeline. Two variants, controlled by `include_irrep_norms`:
        #
        #  - False (default, v1 behavior): only the l=0 channels feed into the
        #    reducer. Higher-l information is computed by the experts but
        #    discarded at this boundary.
        #  - True (v2c): also include the L2 norms of each l>0 channel
        #    (||l=1||, ||l=2||, …). Norms are rotation-invariant, so they
        #    preserve invariance downstream while keeping a scalar summary of
        #    the equivariant work the experts did.
        #
        # Either way the linear is zero-initialized so the inserted block starts
        # as a near-identity (lets SchNet's existing dynamics dominate early).
        n_scalar_channels = sum(mul for mul, ir in self.expert_irreps if ir.l == 0)
        self._n_scalar_channels = n_scalar_channels
        self.include_irrep_norms = include_irrep_norms

        # Per-(l, parity) layout used by the norm-reducer forward path.
        # Each entry: (channel_offset_in_irreps_dim, mul, ir.dim) for l>0 channels.
        self._high_l_chunks: list[tuple[int, int, int]] = []
        offset = 0
        for mul, ir in self.expert_irreps:
            if ir.l > 0:
                self._high_l_chunks.append((offset, mul, ir.dim))
            offset += mul * ir.dim

        if include_irrep_norms:
            n_norm_channels = sum(mul for _, mul, _ in self._high_l_chunks)
            reducer_in = n_scalar_channels + n_norm_channels
        else:
            reducer_in = n_scalar_channels
        self.scalar_proj_back = nn.Linear(reducer_in, hidden_channels)
        nn.init.zeros_(self.scalar_proj_back.weight)
        nn.init.zeros_(self.scalar_proj_back.bias)
        self._reducer_in_dim = reducer_in

    def _build_reducer_input(self, irrep_h: torch.Tensor) -> torch.Tensor:
        """Project the MoE output into the rotation-invariant scalars the reducer expects.

        With `include_irrep_norms=False`: just the l=0 channels.
        With `include_irrep_norms=True`: l=0 channels concatenated with the
        L2 norm of each l>0 multiplicity channel (one scalar per (l, mul) pair).
        """
        scalar_part = irrep_h[:, : self._n_scalar_channels]
        if not self.include_irrep_norms:
            return scalar_part
        norms = []
        for offset, mul, ir_dim in self._high_l_chunks:
            chunk = irrep_h[:, offset : offset + mul * ir_dim]
            chunk = chunk.reshape(chunk.shape[0], mul, ir_dim)
            norms.append(chunk.norm(dim=-1))  # (n_atoms, mul)
        return torch.cat([scalar_part] + norms, dim=-1)

    def forward(
        self,
        z: torch.Tensor,
        pos: torch.Tensor,
        batch: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, MolecularRoutingDecision, torch.Tensor]:
        """Forward pass.

        Args:
            z: (n_atoms,) atomic numbers.
            pos: (n_atoms, 3) Cartesian coordinates.
            batch: (n_atoms,) molecule index per atom.

        Returns:
            energy: (n_molecules, 1) predicted energies after atomref / mean / std / readout.
            decision: per-atom routing record (see MolecularRoutingDecision).
            lb_loss: scalar load-balancing loss (already weighted; add to main loss).
        """
        if batch is None:
            batch = torch.zeros_like(z)

        h = self.embedding(z)
        edge_index, edge_weight = self.interaction_graph(pos, batch)
        edge_attr = self.distance_expansion(edge_weight)
        edge_vec = pos[edge_index[0]] - pos[edge_index[1]]

        decision: MolecularRoutingDecision | None = None
        lb_loss = torch.zeros((), device=z.device)

        for i, interaction in enumerate(self.interactions):
            h = h + interaction(h, edge_index, edge_weight, edge_attr)
            if i + 1 == self.moe_position:
                irrep_h = self.lift(h, edge_index, edge_weight, edge_vec)
                irrep_h, decision, lb_loss = self.moe(irrep_h, scalar_features=h)
                # Reduce back to scalar pipeline. zero-init projection means
                # this delta starts near zero and learns from there.
                reducer_input = self._build_reducer_input(irrep_h)
                h = h + self.scalar_proj_back(reducer_input)

        h = self.lin1(h)
        h = self.act(h)
        h = self.lin2(h)

        if not self.dipole and self.mean is not None and self.std is not None:
            h = h * self.std + self.mean
        if not self.dipole and self.atomref is not None:
            h = h + self.atomref(z)

        out = self.readout(h, batch, dim=0)
        if self.scale is not None:
            out = self.scale * out

        # decision/lb_loss must always be defined; moe_position is in [1, num_interactions-1]
        # so the loop body always runs the MoE branch exactly once.
        assert decision is not None
        return out, decision, lb_loss

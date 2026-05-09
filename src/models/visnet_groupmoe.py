"""ViSNet + Group-MoE: per-atom equivariant MoE inserted into ViSNet.

Background. The v2 sweep on QM9 showed SchNet+GroupMoE consistently
underperforms its SchNet host. The diagnosed cause was a *scalar bottleneck*:
the MoE block worked in irrep space, then had to reduce back to scalars to
feed SchNet's downstream layers, destroying most of the equivariant
computation. See `docs/paper2_routes_forward.md` Route 2.

This model addresses the bottleneck head-on. ViSNet maintains both scalar
features `x` and vector features `v` end-to-end. We insert the MoE block
between two of ViSNet's `ViS_MP` layers; the experts operate directly on
`(x, v)` and produce `(x, v)`. No reduction. The downstream ViSNet layers
consume the MoE output equivariantly.

Each expert is a `GatedEquivariantBlock` from ViSNet's own architecture
(ViSNet uses these in its output-reduction path; here we use them as
generic equivariant per-atom updaters). Each expert's final linear is
zero-initialized so the MoE block starts as near-identity — the
SchNet-with-zero-init-reducer trick that worked there.

Per-atom routing uses ViSNet's scalar features `x` only — keeps routing
rotation-invariant by construction.

Forward returns `(energy, decision, lb_loss)`. The training script must add
`lb_loss` (already weighted by `load_balance_weight`) to the main loss.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch_geometric.nn.models import ViSNet
from torch_geometric.nn.models.visnet import GatedEquivariantBlock, ViSNetBlock
from torch_geometric.utils import scatter

from ..modules.molecular_router import (
    MolecularRouter,
    MolecularRoutingDecision,
    load_balancing_loss,
)


class VisNetGroupMoEBlock(ViSNetBlock):
    """ViSNetBlock with a per-atom MoE inserted between two `ViS_MP` layers.

    Args (in addition to ViSNetBlock's):
        n_experts: K — number of expert symmetry types. Total options = K + 1
            (K experts + pass-through).
        moe_position: 1-indexed position of the MoE block. ``moe_position=k``
            inserts the MoE *after* the k-th ViS_MP layer. Default: half-way
            through the stack (`num_layers // 2`).
        load_balance_weight: coefficient on the load-balancing loss.
    """

    def __init__(
        self,
        *args,
        n_experts: int = 3,
        moe_position: int | None = None,
        load_balance_weight: float = 0.01,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)

        if moe_position is None:
            moe_position = self.num_layers // 2
        if not (1 <= moe_position <= self.num_layers):
            raise ValueError(
                f"moe_position must be in [1, {self.num_layers}], got {moe_position}"
            )
        self.moe_position = moe_position
        self.n_experts = n_experts
        self.load_balance_weight = load_balance_weight

        self.router = MolecularRouter(
            scalar_dim=self.hidden_channels, n_experts=n_experts,
        )
        self.experts = nn.ModuleList()
        for _ in range(n_experts):
            block = GatedEquivariantBlock(
                hidden_channels=self.hidden_channels,
                out_channels=self.hidden_channels,
            )
            # Zero-init the final linear so the block emits (0, 0) at start;
            # the MoE then acts as identity-plus-confidence-weighted-zero-delta
            # at training start, letting ViSNet's existing dynamics dominate
            # before the experts learn to contribute.
            nn.init.zeros_(block.update_net[2].weight)
            nn.init.zeros_(block.update_net[2].bias)
            self.experts.append(block)

    def _apply_moe(
        self, x: torch.Tensor, vec: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, MolecularRoutingDecision, torch.Tensor]:
        """Per-atom dispatch on (scalar, vector) features.

        Pass-through atoms unchanged; expert atoms get
        ``(x, v) ← (x, v) + confidence * (expert(x, v) - (x, v))``.
        """
        decision = self.router(x)
        x_out = x.clone()
        vec_out = vec.clone()

        for k in range(self.n_experts):
            mask = decision.expert_idx == (k + 1)
            if not mask.any():
                continue
            x_k = x[mask]
            vec_k = vec[mask]
            conf = decision.confidence[mask]
            new_x, new_vec = self.experts[k](x_k, vec_k)
            x_out[mask] = x_k + conf.unsqueeze(-1) * (new_x - x_k)
            vec_out[mask] = vec_k + conf.unsqueeze(-1).unsqueeze(-1) * (new_vec - vec_k)

        lb_loss = self.load_balance_weight * load_balancing_loss(decision.logits)
        return x_out, vec_out, decision, lb_loss

    def forward(
        self, z: torch.Tensor, pos: torch.Tensor, batch: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, MolecularRoutingDecision, torch.Tensor]:
        # Mirror ViSNetBlock.forward but insert MoE at `moe_position`.
        x = self.embedding(z)
        edge_index, edge_weight, edge_vec = self.distance(pos, batch)
        edge_attr = self.distance_expansion(edge_weight)
        mask = edge_index[0] != edge_index[1]
        edge_vec[mask] = edge_vec[mask] / torch.norm(edge_vec[mask], dim=1).unsqueeze(1)
        edge_vec = self.sphere(edge_vec)
        x = self.neighbor_embedding(z, x, edge_index, edge_weight, edge_attr)
        vec = torch.zeros(
            x.size(0), ((self.lmax + 1) ** 2) - 1, x.size(1),
            dtype=x.dtype, device=x.device,
        )
        edge_attr = self.edge_embedding(edge_index, edge_attr, x)

        decision: MolecularRoutingDecision | None = None
        lb_loss = torch.zeros((), device=x.device)

        for i, attn in enumerate(self.vis_mp_layers[:-1]):
            dx, dvec, dedge_attr = attn(
                x, vec, edge_index, edge_weight, edge_attr, edge_vec,
            )
            x = x + dx
            vec = vec + dvec
            edge_attr = edge_attr + dedge_attr
            if i + 1 == self.moe_position:
                x, vec, decision, lb_loss = self._apply_moe(x, vec)

        dx, dvec, _ = self.vis_mp_layers[-1](
            x, vec, edge_index, edge_weight, edge_attr, edge_vec,
        )
        x = x + dx
        vec = vec + dvec
        # Final-layer-position case (moe_position == num_layers):
        if self.moe_position == self.num_layers and decision is None:
            x, vec, decision, lb_loss = self._apply_moe(x, vec)

        x = self.out_norm(x)
        vec = self.vec_out_norm(vec)

        assert decision is not None  # moe_position is in [1, num_layers]
        return x, vec, decision, lb_loss


class VisNetGroupMoE(ViSNet):
    """ViSNet with `representation_model` replaced by `VisNetGroupMoEBlock`.

    Forward signature:
        ``forward(z, pos, batch) → (energy, decision, lb_loss)``

    All ViSNet args are forwarded to the parent. Additional args:
        n_experts (int, default 3)
        moe_position (int, default num_layers // 2)
        load_balance_weight (float, default 0.01)

    Note: this model does not currently support ViSNet's `derivative=True`
    direct-force path. For force prediction, autograd through the energy
    output the same way `train_md17_groupmoe.py` does for SchNetGroupMoE.
    """

    def __init__(
        self,
        *,
        n_experts: int = 3,
        moe_position: int | None = None,
        load_balance_weight: float = 0.01,
        **visnet_kwargs,
    ) -> None:
        if visnet_kwargs.get("derivative", False):
            raise NotImplementedError(
                "VisNetGroupMoE does not yet support ViSNet's derivative=True "
                "direct-force path. Use external autograd of energy w.r.t. pos "
                "(see scripts/train_md17_groupmoe.py for the pattern)."
            )
        super().__init__(**visnet_kwargs)

        # Reconstruct representation_model with our MoE-equipped block,
        # using the same kwargs ViSNet's __init__ passed.
        self.representation_model = VisNetGroupMoEBlock(
            lmax=visnet_kwargs.get("lmax", 1),
            vecnorm_type=visnet_kwargs.get("vecnorm_type", None),
            trainable_vecnorm=visnet_kwargs.get("trainable_vecnorm", False),
            num_heads=visnet_kwargs.get("num_heads", 8),
            num_layers=visnet_kwargs.get("num_layers", 6),
            hidden_channels=visnet_kwargs.get("hidden_channels", 128),
            num_rbf=visnet_kwargs.get("num_rbf", 32),
            trainable_rbf=visnet_kwargs.get("trainable_rbf", False),
            max_z=visnet_kwargs.get("max_z", 100),
            cutoff=visnet_kwargs.get("cutoff", 5.0),
            max_num_neighbors=visnet_kwargs.get("max_num_neighbors", 32),
            vertex=visnet_kwargs.get("vertex", False),
            n_experts=n_experts,
            moe_position=moe_position,
            load_balance_weight=load_balance_weight,
        )
        self.n_experts = n_experts
        self.moe_position = self.representation_model.moe_position
        self.load_balance_weight = load_balance_weight

    def forward(
        self,
        z: torch.Tensor,
        pos: torch.Tensor,
        batch: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, MolecularRoutingDecision, torch.Tensor]:
        """Compute energy, routing decision, and load-balancing loss."""
        if batch is None:
            batch = torch.zeros_like(z)

        x, v, decision, lb_loss = self.representation_model(z, pos, batch)
        x = self.output_model.pre_reduce(x, v)
        x = x * self.std
        if self.prior_model is not None:
            x = self.prior_model(x, z)
        y = scatter(x, batch, dim=0, reduce=self.reduce_op)
        y = y + self.mean
        return y, decision, lb_loss

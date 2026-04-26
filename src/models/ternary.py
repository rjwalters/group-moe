"""Ternary models for Group-MoE S_3 experiments.

Two models with a shared architecture base:
- TernaryGroupMoE: uses a GroupMoELayer with S_3 representation
- TernaryBaseline: uses a matched-parameter residual MLP instead

Both output a scalar regression prediction.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..groups import S3Representation
from ..modules import GroupMoELayer
from ..modules.standard_moe import StandardMoELayer


class ResidualMLPBlock(nn.Module):
    """Pre-norm residual MLP block."""

    def __init__(self, d_model: int, expansion: int = 4):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * expansion),
            nn.GELU(),
            nn.Linear(d_model * expansion, d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.mlp(self.norm(x))


class TernaryBase(nn.Module):
    """Shared base: embeddings + combine + residual MLP blocks."""

    def __init__(
        self,
        d_model: int = 128,
        n_numbers: int = 10,
        n_ops: int = 2,
        n_blocks: int = 2,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_embed = nn.Embedding(n_numbers, d_model)
        self.op_embed = nn.Embedding(n_ops, d_model)
        self.combine = nn.Linear(4 * d_model, d_model)
        self.blocks = nn.ModuleList(
            [ResidualMLPBlock(d_model) for _ in range(n_blocks)]
        )
        self.head = nn.Linear(d_model, 1)

    def encode(self, a: torch.Tensor, op: torch.Tensor, b: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        a_emb = self.num_embed(a)
        op_emb = self.op_embed(op)
        b_emb = self.num_embed(b)
        c_emb = self.num_embed(c)
        x = self.combine(torch.cat([a_emb, op_emb, b_emb, c_emb], dim=-1))
        x = F.gelu(x)
        for block in self.blocks:
            x = block(x)
        return x

    def count_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class TernaryGroupMoE(TernaryBase):
    """Ternary model with a GroupMoE layer (S_3 expert)."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.group_moe = GroupMoELayer(self.d_model, [S3Representation()])

    def forward(self, a, op, b, c):
        x = self.encode(a, op, b, c)
        x, decision = self.group_moe(x)
        return self.head(x).squeeze(-1), decision


class TernaryStandardMoE(TernaryBase):
    """Ternary model with a standard MoE layer (learned transforms, no group structure).

    Same routing architecture as GroupMoE — same number of options (7),
    same expert dimensions (k=4), same residual blending — but the expert
    applies a LEARNED k×k matrix instead of a fixed irrep matrix R(g).

    This is the controlled comparison: if GroupMoE outperforms this,
    the advantage comes from the group structure.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Match S_3 GroupMoE: 1 expert, 6 slots (like 6 group elements), dim 4
        self.standard_moe = StandardMoELayer(
            self.d_model,
            n_experts=1,
            slots_per_expert=6,
            expert_dim=4,
        )

    def forward(self, a, op, b, c):
        x = self.encode(a, op, b, c)
        x, decision = self.standard_moe(x)
        return self.head(x).squeeze(-1), decision


class TernaryBaseline(TernaryBase):
    """Ternary model with a residual MLP (no group structure).

    MLP dimensions chosen to approximately match GroupMoE parameter count.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        d = kwargs.get("d_model", 128)
        # S_3 expert has: project(d*4) + inject(4*d) + router(d*32 + 32*7)
        # ≈ 8d + 32d + 224 ≈ 40d + 224 ≈ 5344 for d=128
        # Match with a small MLP: d -> hidden -> d
        hidden = max(d // 5, 4)
        self.baseline_mlp = nn.Sequential(
            nn.LayerNorm(d),
            nn.Linear(d, hidden),
            nn.GELU(),
            nn.Linear(hidden, d),
        )

    def forward(self, a, op, b, c):
        x = self.encode(a, op, b, c)
        x = x + self.baseline_mlp(x)
        return self.head(x).squeeze(-1), None

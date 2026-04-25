"""Arithmetic models for Group-MoE experiments.

Two models with a shared architecture base:
- ArithmeticGroupMoE: uses a GroupMoELayer with S_2 representation
- ArithmeticBaseline: uses a matched-parameter residual MLP instead

Both output a scalar regression prediction.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..groups import S2Representation
from ..modules import GroupMoELayer


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


class ArithmeticBase(nn.Module):
    """Shared base: embeddings + combine + residual MLP blocks."""

    def __init__(
        self,
        d_model: int = 128,
        n_numbers: int = 100,
        n_ops: int = 2,
        n_blocks: int = 2,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_embed = nn.Embedding(n_numbers, d_model)
        self.op_embed = nn.Embedding(n_ops, d_model)
        self.combine = nn.Linear(3 * d_model, d_model)
        self.blocks = nn.ModuleList(
            [ResidualMLPBlock(d_model) for _ in range(n_blocks)]
        )
        self.head = nn.Linear(d_model, 1)

    def encode(self, a: torch.Tensor, op: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        a_emb = self.num_embed(a)
        op_emb = self.op_embed(op)
        b_emb = self.num_embed(b)
        x = self.combine(torch.cat([a_emb, op_emb, b_emb], dim=-1))
        x = F.gelu(x)
        for block in self.blocks:
            x = block(x)
        return x

    def count_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class ArithmeticGroupMoE(ArithmeticBase):
    """Arithmetic model with a GroupMoE layer (S_2 expert)."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.group_moe = GroupMoELayer(self.d_model, [S2Representation()])

    def forward(self, a, op, b):
        x = self.encode(a, op, b)
        x, decision = self.group_moe(x)
        return self.head(x).squeeze(-1), decision


class ArithmeticBaseline(ArithmeticBase):
    """Arithmetic model with a residual MLP (no group structure).

    The MLP dimensions are chosen to approximately match the parameter
    count of the GroupMoE layer.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        d = kwargs.get("d_model", 128)
        hidden = max(d // 7, 4)
        self.baseline_mlp = nn.Sequential(
            nn.LayerNorm(d),
            nn.Linear(d, hidden),
            nn.GELU(),
            nn.Linear(hidden, d),
        )

    def forward(self, a, op, b):
        x = self.encode(a, op, b)
        x = x + self.baseline_mlp(x)
        return self.head(x).squeeze(-1), None

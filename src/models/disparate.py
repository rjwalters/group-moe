"""Disparate-groups models: Z_2 + Z_3 (non-nested groups).

- DisparateGroupMoE: GroupMoELayer with Z_2 and Z_3 experts
- DisparateBaseline: matched-parameter residual MLP
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..groups import Z2Representation, Z3Representation
from ..modules import GroupMoELayer


class ResidualMLPBlock(nn.Module):
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


class DisparateBase(nn.Module):
    def __init__(
        self,
        d_model: int = 128,
        n_numbers: int = 15,
        n_ops: int = 3,
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

    def encode(self, a, op, b, c):
        a_emb = self.num_embed(a)
        op_emb = self.op_embed(op)
        b_emb = self.num_embed(b)
        c_emb = self.num_embed(c)
        x = self.combine(torch.cat([a_emb, op_emb, b_emb, c_emb], dim=-1))
        x = F.gelu(x)
        for block in self.blocks:
            x = block(x)
        return x

    def count_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class DisparateGroupMoE(DisparateBase):
    """Model with Z_2 and Z_3 group experts (non-nested)."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.group_moe = GroupMoELayer(
            self.d_model,
            [Z2Representation(), Z3Representation()],
        )

    def forward(self, a, op, b, c):
        x = self.encode(a, op, b, c)
        x, decision = self.group_moe(x)
        return self.head(x).squeeze(-1), decision


class DisparateBaseline(DisparateBase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        d = kwargs.get("d_model", 128)
        hidden = max(d // 4, 4)
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

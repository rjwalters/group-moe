"""Multi-group models for S_2 + S_3 routing experiments.

Two models with a shared architecture base:
- MultiGroupMoE: uses a GroupMoELayer with both S_2 and S_3 representations
- MultiGroupBaseline: uses a matched-parameter residual MLP instead

Both output a scalar regression prediction.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..groups import S2Representation, S3Representation
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


class MultiGroupBase(nn.Module):
    """Shared base: embeddings + combine + residual MLP blocks."""

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


class MultiGroupMoE(MultiGroupBase):
    """Model with both S_2 and S_3 group experts."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.group_moe = GroupMoELayer(
            self.d_model,
            [S2Representation(), S3Representation()],
        )

    def forward(self, a, op, b, c):
        x = self.encode(a, op, b, c)
        x, decision = self.group_moe(x)
        return self.head(x).squeeze(-1), decision


class MultiGroupBaseline(MultiGroupBase):
    """Baseline with a matched-parameter residual MLP."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        d = kwargs.get("d_model", 128)
        # S_2 expert: 2*d*2 = 4d params
        # S_3 expert: 2*d*4 = 8d params
        # Router: d*32 + 32*9 ≈ 32d + 288
        # Total ≈ 44d + 288 ≈ 5920 for d=128
        # Match with MLP: d -> hidden -> d
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

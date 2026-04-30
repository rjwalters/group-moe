"""General n-ary models for S_n scaling experiments.

Three variants with identical base architecture:
- NaryGroupMoE: GroupMoELayer with S_n representation
- NaryStandardMoE: StandardMoELayer with matched dimensions
- NaryBaseline: residual MLP (no expert)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..groups.symmetric import SnRepresentation
from ..modules import GroupMoELayer
from ..modules.standard_moe import StandardMoELayer


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


class NaryBase(nn.Module):
    """Shared base: n input embeddings + op embedding + MLP blocks."""

    def __init__(
        self,
        n: int = 3,
        d_model: int = 128,
        n_numbers: int = 10,
        n_ops: int = 2,
        n_blocks: int = 2,
    ):
        super().__init__()
        self.n = n
        self.d_model = d_model
        self.num_embed = nn.Embedding(n_numbers, d_model)
        self.op_embed = nn.Embedding(n_ops, d_model)
        self.combine = nn.Linear((n + 1) * d_model, d_model)
        self.blocks = nn.ModuleList(
            [ResidualMLPBlock(d_model) for _ in range(n_blocks)]
        )
        self.head = nn.Linear(d_model, 1)

    def encode(self, inputs: torch.Tensor, op: torch.Tensor) -> torch.Tensor:
        # inputs: (batch, n) long tensor
        input_embs = self.num_embed(inputs)  # (batch, n, d_model)
        batch = inputs.shape[0]
        input_flat = input_embs.reshape(batch, self.n * self.d_model)
        op_emb = self.op_embed(op)  # (batch, d_model)
        x = self.combine(torch.cat([input_flat, op_emb], dim=-1))
        x = F.gelu(x)
        for block in self.blocks:
            x = block(x)
        return x

    def count_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class NaryGroupMoE(NaryBase):
    """N-ary model with GroupMoE layer (S_n expert)."""

    def __init__(self, **kwargs):
        n = kwargs.get("n", 3)
        super().__init__(**kwargs)
        self.group_moe = GroupMoELayer(self.d_model, [SnRepresentation(n)])

    def forward(self, inputs, op):
        x = self.encode(inputs, op)
        x, decision = self.group_moe(x)
        return self.head(x).squeeze(-1), decision


class NaryStandardMoE(NaryBase):
    """N-ary model with StandardMoE (learned transforms, matched dimensions)."""

    def __init__(self, **kwargs):
        n = kwargs.get("n", 3)
        super().__init__(**kwargs)
        sn = SnRepresentation(n)
        self.standard_moe = StandardMoELayer(
            self.d_model,
            n_experts=1,
            slots_per_expert=sn.order,
            expert_dim=sn.total_dim,
        )

    def forward(self, inputs, op):
        x = self.encode(inputs, op)
        x, decision = self.standard_moe(x)
        return self.head(x).squeeze(-1), decision


class NaryBaseline(NaryBase):
    """N-ary model with residual MLP (no expert)."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        d = self.d_model
        hidden = max(d // 4, 4)
        self.baseline_mlp = nn.Sequential(
            nn.LayerNorm(d),
            nn.Linear(d, hidden),
            nn.GELU(),
            nn.Linear(hidden, d),
        )

    def forward(self, inputs, op):
        x = self.encode(inputs, op)
        x = x + self.baseline_mlp(x)
        return self.head(x).squeeze(-1), None

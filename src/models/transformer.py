"""Transformer models for Group-MoE experiments.

Small transformer encoder (4 layers, self-attention) with Group-MoE
as a drop-in FFN replacement. Each of (a, op, b, c) is a separate token
in a length-4 sequence; attention lets them interact before the MoE layer.

Three variants:
- TransformerGroupMoE: one FFN replaced with GroupMoELayer (S_3)
- TransformerStandardMoE: one FFN replaced with StandardMoELayer (learned)
- TransformerBaseline: all standard FFN blocks
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..groups import S3Representation
from ..modules import GroupMoELayer
from ..modules.standard_moe import StandardMoELayer


class TransformerBlock(nn.Module):
    """Standard pre-norm transformer encoder block."""

    def __init__(self, d_model: int, n_heads: int, ffn_expansion: int = 4, dropout: float = 0.0):
        super().__init__()
        self.attn_norm = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.ffn_norm = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * ffn_expansion),
            nn.GELU(),
            nn.Linear(d_model * ffn_expansion, d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        normed = self.attn_norm(x)
        attn_out, _ = self.attn(normed, normed, normed)
        x = x + attn_out
        x = x + self.ffn(self.ffn_norm(x))
        return x


class MoETransformerBlock(nn.Module):
    """Transformer block with MoE replacing the FFN.

    The MoE layer (GroupMoE or StandardMoE) is internally residual:
    moe_out = input + conf * delta. We extract the delta and add it
    as the outer residual.
    """

    def __init__(self, d_model: int, n_heads: int, moe_layer: nn.Module, dropout: float = 0.0):
        super().__init__()
        self.attn_norm = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.ffn_norm = nn.LayerNorm(d_model)
        self.moe_layer = moe_layer

    def forward(self, x: torch.Tensor):
        normed = self.attn_norm(x)
        attn_out, _ = self.attn(normed, normed, normed)
        x = x + attn_out

        normed = self.ffn_norm(x)
        moe_out, decision = self.moe_layer(normed)
        # MoE is internally residual (moe_out = normed + conf*delta).
        # Extract delta for the outer pre-norm residual.
        x = x + (moe_out - normed)
        return x, decision


class TransformerBase(nn.Module):
    """Shared base: token embeddings, positional embeddings, final head."""

    def __init__(
        self,
        d_model: int = 256,
        n_numbers: int = 15,
        n_ops: int = 2,
        n_heads: int = 4,
        n_layers: int = 4,
        ffn_expansion: int = 4,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.ffn_expansion = ffn_expansion
        self.dropout = dropout

        self.num_embed = nn.Embedding(n_numbers, d_model)
        self.op_embed = nn.Embedding(n_ops, d_model)
        self.pos_embed = nn.Embedding(4, d_model)

        self.final_norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, 1)

    def encode(self, a: torch.Tensor, op: torch.Tensor, b: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        a_emb = self.num_embed(a)
        op_emb = self.op_embed(op)
        b_emb = self.num_embed(b)
        c_emb = self.num_embed(c)

        x = torch.stack([a_emb, op_emb, b_emb, c_emb], dim=1)  # (batch, 4, d_model)
        positions = torch.arange(4, device=x.device)
        x = x + self.pos_embed(positions)
        return x

    def count_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class TransformerGroupMoE(TransformerBase):
    """Transformer with GroupMoELayer (S_3) replacing one FFN."""

    def __init__(self, moe_layer_idx: int = 2, **kwargs):
        super().__init__(**kwargs)
        d = self.d_model

        self.blocks = nn.ModuleList()
        self.moe_layer_idx = moe_layer_idx

        for i in range(self.n_layers):
            if i == moe_layer_idx:
                moe = GroupMoELayer(d, [S3Representation()])
                self.blocks.append(MoETransformerBlock(d, self.n_heads, moe, self.dropout))
            else:
                self.blocks.append(TransformerBlock(d, self.n_heads, self.ffn_expansion, self.dropout))

    def forward(self, a, op, b, c):
        x = self.encode(a, op, b, c)
        decision = None
        for block in self.blocks:
            if isinstance(block, MoETransformerBlock):
                x, decision = block(x)
            else:
                x = block(x)
        pooled = self.final_norm(x).mean(dim=1)
        return self.head(pooled).squeeze(-1), decision


class TransformerStandardMoE(TransformerBase):
    """Transformer with StandardMoELayer (learned W) replacing one FFN."""

    def __init__(self, moe_layer_idx: int = 2, **kwargs):
        super().__init__(**kwargs)
        d = self.d_model

        self.blocks = nn.ModuleList()
        self.moe_layer_idx = moe_layer_idx

        for i in range(self.n_layers):
            if i == moe_layer_idx:
                moe = StandardMoELayer(d, n_experts=1, slots_per_expert=6, expert_dim=4)
                self.blocks.append(MoETransformerBlock(d, self.n_heads, moe, self.dropout))
            else:
                self.blocks.append(TransformerBlock(d, self.n_heads, self.ffn_expansion, self.dropout))

    def forward(self, a, op, b, c):
        x = self.encode(a, op, b, c)
        decision = None
        for block in self.blocks:
            if isinstance(block, MoETransformerBlock):
                x, decision = block(x)
            else:
                x = block(x)
        pooled = self.final_norm(x).mean(dim=1)
        return self.head(pooled).squeeze(-1), decision


class TransformerBaseline(TransformerBase):
    """Transformer with all standard FFN blocks."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        d = self.d_model

        self.blocks = nn.ModuleList([
            TransformerBlock(d, self.n_heads, self.ffn_expansion, self.dropout)
            for _ in range(self.n_layers)
        ])

    def forward(self, a, op, b, c):
        x = self.encode(a, op, b, c)
        for block in self.blocks:
            x = block(x)
        pooled = self.final_norm(x).mean(dim=1)
        return self.head(pooled).squeeze(-1), None

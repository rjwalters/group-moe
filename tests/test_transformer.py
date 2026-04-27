"""Tests for transformer models."""

import torch
import pytest

from src.models.transformer import (
    TransformerBlock, MoETransformerBlock,
    TransformerGroupMoE, TransformerStandardMoE, TransformerBaseline,
)
from src.modules import GroupMoELayer
from src.groups import S3Representation


class TestTransformerBlocks:
    def test_transformer_block_shape(self):
        block = TransformerBlock(d_model=64, n_heads=4)
        x = torch.randn(8, 4, 64)
        out = block(x)
        assert out.shape == (8, 4, 64)

    def test_moe_block_shape(self):
        moe = GroupMoELayer(64, [S3Representation()])
        block = MoETransformerBlock(d_model=64, n_heads=4, moe_layer=moe)
        x = torch.randn(8, 4, 64)
        out, decision = block(x)
        assert out.shape == (8, 4, 64)
        assert decision is not None
        # 8 examples * 4 tokens = 32 routing decisions
        assert decision.group_idx.shape == (32,)


class TestTransformerModels:
    def test_groupmoe_forward_shape(self):
        model = TransformerGroupMoE(d_model=64, n_numbers=10, n_heads=4, n_layers=4)
        a = torch.randint(0, 10, (8,))
        op = torch.randint(0, 2, (8,))
        b = torch.randint(0, 10, (8,))
        c = torch.randint(0, 10, (8,))
        pred, decision = model(a, op, b, c)
        assert pred.shape == (8,)
        assert decision is not None
        assert decision.group_idx.shape == (32,)  # 8 * 4 tokens

    def test_standardmoe_forward_shape(self):
        model = TransformerStandardMoE(d_model=64, n_numbers=10, n_heads=4, n_layers=4)
        a = torch.randint(0, 10, (8,))
        op = torch.randint(0, 2, (8,))
        b = torch.randint(0, 10, (8,))
        c = torch.randint(0, 10, (8,))
        pred, decision = model(a, op, b, c)
        assert pred.shape == (8,)
        assert decision is not None

    def test_baseline_forward_shape(self):
        model = TransformerBaseline(d_model=64, n_numbers=10, n_heads=4, n_layers=4)
        a = torch.randint(0, 10, (8,))
        op = torch.randint(0, 2, (8,))
        b = torch.randint(0, 10, (8,))
        c = torch.randint(0, 10, (8,))
        pred, decision = model(a, op, b, c)
        assert pred.shape == (8,)
        assert decision is None

    def test_encode_produces_sequence(self):
        model = TransformerBaseline(d_model=64, n_numbers=10, n_heads=4, n_layers=4)
        a = torch.randint(0, 10, (4,))
        op = torch.randint(0, 2, (4,))
        b = torch.randint(0, 10, (4,))
        c = torch.randint(0, 10, (4,))
        x = model.encode(a, op, b, c)
        assert x.shape == (4, 4, 64)  # (batch, seq_len=4, d_model)

    def test_shared_num_embed(self):
        model = TransformerBaseline(d_model=64, n_numbers=10, n_heads=4, n_layers=4)
        # a and c should use the same embedding
        idx = torch.tensor([3])
        a_emb = model.num_embed(idx)
        c_emb = model.num_embed(idx)
        assert torch.equal(a_emb, c_emb)

    def test_moe_at_specified_layer(self):
        model = TransformerGroupMoE(d_model=64, n_numbers=10, n_heads=4, n_layers=4, moe_layer_idx=2)
        assert isinstance(model.blocks[2], MoETransformerBlock)
        assert isinstance(model.blocks[0], TransformerBlock)
        assert isinstance(model.blocks[1], TransformerBlock)
        assert isinstance(model.blocks[3], TransformerBlock)

    def test_groupmoe_routing_logits_shape(self):
        model = TransformerGroupMoE(d_model=64, n_numbers=10, n_heads=4, n_layers=4)
        a = torch.randint(0, 10, (4,))
        op = torch.randint(0, 2, (4,))
        b = torch.randint(0, 10, (4,))
        c = torch.randint(0, 10, (4,))
        _, decision = model(a, op, b, c)
        # 1 pass-through + 6 S_3 elements = 7 options
        assert decision.logits.shape == (16, 7)  # 4 examples * 4 tokens

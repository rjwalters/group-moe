"""Tests for Group-MoE modules."""

import torch
import pytest

from src.groups import S2Representation, S3Representation, Z2Representation
from src.modules import GroupExpert, SymmetryRouter, GroupMoELayer


class TestGroupExpert:
    def test_forward_shape(self):
        group = S3Representation()
        expert = GroupExpert(d_model=64, group=group)
        x = torch.randn(8, 64)
        elem_idx = torch.zeros(8, dtype=torch.long)
        out = expert(x, elem_idx)
        assert out.shape == (8, 64)

    def test_identity_element(self):
        """Applying the identity element should approximately preserve input."""
        group = S3Representation()
        expert = GroupExpert(d_model=64, group=group)
        x = torch.randn(4, 64)
        elem_idx = torch.zeros(4, dtype=torch.long)  # identity
        out = expert(x, elem_idx)
        # After project -> identity -> inject, should be close to project -> inject
        # (a rank-k approximation of x)
        z = expert.project(x)
        reconstructed = expert.inject(z)
        assert torch.allclose(out, reconstructed, atol=1e-5)

    def test_composition_exact(self):
        """Composing two elements via compose() should match direct application."""
        group = S3Representation()
        expert = GroupExpert(d_model=64, group=group)
        x = torch.randn(1, 64)

        # Apply (01) then (12) = (012)
        table = group.multiplication_table()
        composed_idx = table[1, 2].item()  # should be (012) = index 4

        result_composed = expert.compose(x, 1, 2)
        result_direct = expert(x, torch.tensor([composed_idx]))

        assert torch.allclose(result_composed, result_direct, atol=1e-5)

    def test_compression_ratio(self):
        """Group expert should be much more parameter-efficient."""
        group = S3Representation()
        expert = GroupExpert(d_model=896, group=group)
        assert expert.compression_ratio > 100


class TestSymmetryRouter:
    def test_forward_shape(self):
        groups = [S2Representation(), S3Representation()]
        router = SymmetryRouter(d_model=64, groups=groups)
        x = torch.randn(8, 64)
        decision = router(x)
        assert decision.group_idx.shape == (8,)
        assert decision.element_idx.shape == (8,)
        assert decision.confidence.shape == (8,)

    def test_n_options(self):
        groups = [S2Representation(), S3Representation()]
        router = SymmetryRouter(d_model=64, groups=groups)
        # 1 (pass-through) + 2 (S_2) + 6 (S_3) = 9
        assert router.n_options == 9


class TestGroupMoELayer:
    def test_forward_shape_2d(self):
        groups = [S2Representation(), S3Representation()]
        layer = GroupMoELayer(d_model=64, groups=groups)
        x = torch.randn(8, 64)
        out, decision = layer(x)
        assert out.shape == (8, 64)

    def test_forward_shape_3d(self):
        groups = [S2Representation()]
        layer = GroupMoELayer(d_model=64, groups=groups)
        x = torch.randn(2, 16, 64)
        out, decision = layer(x)
        assert out.shape == (2, 16, 64)

    def test_param_summary(self):
        groups = [S3Representation()]
        layer = GroupMoELayer(d_model=256, groups=groups)
        summary = layer.param_summary()
        assert summary["total"] > 0
        assert summary["compression_ratio"] > 10

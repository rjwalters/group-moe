"""Tests for disparate-groups dataset and models."""

import torch
import pytest

from src.data.disparate import (
    DisparateDataset, OP_Z2, OP_Z3, OP_NONE,
    fn_z2, fn_z3, fn_none,
)
from src.models.disparate import DisparateGroupMoE, DisparateBaseline


class TestDisparateFunctions:
    def test_fn_z2_swap_invariant(self):
        for a, b, c in [(3, 5, 7), (0, 1, 2), (9, 2, 4)]:
            assert fn_z2(a, b, c) == fn_z2(b, a, c)

    def test_fn_z3_cyclic_invariant(self):
        for a, b, c in [(3, 5, 7), (0, 1, 14), (9, 2, 4)]:
            val = fn_z3(a, b, c)
            assert fn_z3(b, c, a) == val, f"Cyclic rotation failed for ({a},{b},{c})"
            assert fn_z3(c, a, b) == val, f"Cyclic rotation failed for ({a},{b},{c})"

    def test_fn_z3_not_swap_invariant(self):
        # Should change sign under swap
        assert fn_z3(3, 5, 7) == -fn_z3(5, 3, 7)
        assert fn_z3(1, 2, 3) != fn_z3(2, 1, 3)

    def test_fn_z3_values(self):
        # a²(b-c) + b²(c-a) + c²(a-b) at (1,2,3)
        # = 1*(2-3) + 4*(3-1) + 9*(1-2) = -1 + 8 - 9 = -2
        assert fn_z3(1, 2, 3) == -2
        assert fn_z3(0, 0, 0) == 0

    def test_fn_none_not_symmetric(self):
        assert fn_none(3, 5, 7) != fn_none(5, 3, 7)


class TestDisparateDataset:
    def test_z2_complement_one_per_pair_c(self):
        ds = DisparateDataset(split="train", num_range=6)
        z2 = [(a, b, c) for a, op, b, c, _ in ds.examples if op == OP_Z2]
        seen = set()
        for a, b, c in z2:
            key = (min(a, b), max(a, b), c)
            assert key not in seen, f"Duplicate Z_2 pair {key}"
            seen.add(key)

    def test_z3_complement_one_per_orbit(self):
        ds = DisparateDataset(split="train", num_range=6)
        z3 = [(a, b, c) for a, op, b, c, _ in ds.examples if op == OP_Z3]
        seen = set()
        for a, b, c in z3:
            canonical = min((a, b, c), (b, c, a), (c, a, b))
            assert canonical not in seen, f"Duplicate Z_3 orbit {canonical}"
            seen.add(canonical)

    def test_z2_test_has_complements(self):
        train = DisparateDataset(split="train", num_range=6)
        test = DisparateDataset(split="test", num_range=6)
        train_z2 = train.triples_for_op(OP_Z2)
        for a, op, b, c, _ in test.examples:
            if op == OP_Z2:
                assert (b, a, c) in train_z2

    def test_z3_test_has_complements(self):
        train = DisparateDataset(split="train", num_range=6)
        test = DisparateDataset(split="test", num_range=6)
        train_z3 = train.triples_for_op(OP_Z3)
        for a, op, b, c, _ in test.examples:
            if op == OP_Z3:
                has = ((b, c, a) in train_z3 or (c, a, b) in train_z3
                       or (a, b, c) in train_z3)
                assert has, f"Z3 test ({a},{b},{c}) has no cyclic complement"

    def test_targets_correct(self):
        ds = DisparateDataset(split="train", num_range=8)
        for a, op, b, c, result in ds.examples:
            if op == OP_Z2:
                assert result == fn_z2(a, b, c)
            elif op == OP_Z3:
                assert result == fn_z3(a, b, c)
            else:
                assert result == fn_none(a, b, c)

    def test_three_ops_present(self):
        ds = DisparateDataset(split="train", num_range=8)
        ops = {op for _, op, _, _, _ in ds.examples}
        assert ops == {OP_Z2, OP_Z3, OP_NONE}


class TestDisparateModels:
    def test_groupmoe_forward_shape(self):
        model = DisparateGroupMoE(d_model=64, n_numbers=10)
        a = torch.randint(0, 10, (8,))
        op = torch.randint(0, 3, (8,))
        b = torch.randint(0, 10, (8,))
        c = torch.randint(0, 10, (8,))
        pred, decision = model(a, op, b, c)
        assert pred.shape == (8,)
        assert decision is not None

    def test_router_has_6_options(self):
        model = DisparateGroupMoE(d_model=64, n_numbers=10)
        # 1 pass-through + 2 Z_2 + 3 Z_3 = 6
        assert model.group_moe.router.n_options == 6

    def test_param_counts_comparable(self):
        moe = DisparateGroupMoE(d_model=128, n_numbers=15)
        base = DisparateBaseline(d_model=128, n_numbers=15)
        ratio = moe.count_params() / base.count_params()
        assert 0.8 < ratio < 1.2

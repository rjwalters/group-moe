"""Tests for multi-group dataset and models."""

import itertools

import torch
import pytest

from src.data.multigroup import (
    MultiGroupDataset, OP_S3, OP_S2, OP_NONE,
    fn_s3, fn_s2, fn_none,
)
from src.models.multigroup import MultiGroupMoE, MultiGroupBaseline


class TestMultiGroupFunctions:
    def test_fn_s3_invariant(self):
        for a, b, c in [(3, 5, 7), (0, 1, 2)]:
            val = fn_s3(a, b, c)
            for p in itertools.permutations([a, b, c]):
                assert fn_s3(*p) == val

    def test_fn_s2_invariant_under_ab_swap(self):
        for a, b, c in [(3, 5, 7), (0, 1, 2), (9, 2, 4)]:
            assert fn_s2(a, b, c) == fn_s2(b, a, c)

    def test_fn_s2_not_s3_invariant(self):
        # a+b+2c should change when c is swapped with a or b
        assert fn_s2(1, 2, 5) != fn_s2(5, 2, 1)
        assert fn_s2(1, 2, 5) != fn_s2(1, 5, 2)

    def test_fn_none_not_symmetric(self):
        assert fn_none(3, 5, 7) != fn_none(5, 3, 7)


class TestMultiGroupDataset:
    def test_s3_complement_one_per_multiset(self):
        ds = MultiGroupDataset(split="train", num_range=8)
        s3_examples = [(a, b, c) for a, op, b, c, _ in ds.examples if op == OP_S3]
        seen = set()
        for a, b, c in s3_examples:
            key = tuple(sorted([a, b, c]))
            assert key not in seen, f"Duplicate S3 multiset {key}"
            seen.add(key)

    def test_s2_complement_one_ordering_per_pair_c(self):
        ds = MultiGroupDataset(split="train", num_range=6)
        s2_examples = [(a, b, c) for a, op, b, c, _ in ds.examples if op == OP_S2]
        # For each unordered pair {a,b} and c, should have exactly one ordering
        seen = set()
        for a, b, c in s2_examples:
            key = (min(a, b), max(a, b), c)
            assert key not in seen, f"Duplicate S2 pair+c {key}"
            seen.add(key)

    def test_s3_test_has_complements(self):
        train = MultiGroupDataset(split="train", num_range=8)
        test = MultiGroupDataset(split="test", num_range=8)
        train_s3 = train.triples_for_op(OP_S3)

        for a, op, b, c, _ in test.examples:
            if op == OP_S3:
                has_comp = any(
                    (p[0], p[1], p[2]) in train_s3
                    for p in itertools.permutations([a, b, c])
                )
                assert has_comp, f"S3 test ({a},{b},{c}) has no complement"

    def test_s2_test_has_complements(self):
        train = MultiGroupDataset(split="train", num_range=8)
        test = MultiGroupDataset(split="test", num_range=8)
        train_s2 = train.triples_for_op(OP_S2)

        for a, op, b, c, _ in test.examples:
            if op == OP_S2:
                # The complement is (b, a, c) — swap first two
                assert (b, a, c) in train_s2, \
                    f"S2 test ({a},{b},{c}) has no complement (b,a,c)=({b},{a},{c})"

    def test_targets_correct(self):
        ds = MultiGroupDataset(split="train", num_range=8)
        for a, op, b, c, result in ds.examples:
            if op == OP_S3:
                assert result == fn_s3(a, b, c)
            elif op == OP_S2:
                assert result == fn_s2(a, b, c)
            else:
                assert result == fn_none(a, b, c)

    def test_deterministic(self):
        ds1 = MultiGroupDataset(split="train", seed=42)
        ds2 = MultiGroupDataset(split="train", seed=42)
        assert ds1.examples == ds2.examples

    def test_getitem_format(self):
        ds = MultiGroupDataset(split="train")
        item = ds[0]
        assert set(item.keys()) == {"a", "op", "b", "c", "target", "target_raw"}

    def test_three_ops_present(self):
        ds = MultiGroupDataset(split="train", num_range=8)
        ops = {op for _, op, _, _, _ in ds.examples}
        assert ops == {OP_S3, OP_S2, OP_NONE}


class TestMultiGroupModels:
    def test_multigroup_forward_shape(self):
        model = MultiGroupMoE(d_model=64, n_numbers=10)
        a = torch.randint(0, 10, (8,))
        op = torch.randint(0, 3, (8,))
        b = torch.randint(0, 10, (8,))
        c = torch.randint(0, 10, (8,))
        pred, decision = model(a, op, b, c)
        assert pred.shape == (8,)
        assert decision is not None

    def test_baseline_forward_shape(self):
        model = MultiGroupBaseline(d_model=64, n_numbers=10)
        a = torch.randint(0, 10, (8,))
        op = torch.randint(0, 3, (8,))
        b = torch.randint(0, 10, (8,))
        c = torch.randint(0, 10, (8,))
        pred, decision = model(a, op, b, c)
        assert pred.shape == (8,)
        assert decision is None

    def test_router_has_9_options(self):
        model = MultiGroupMoE(d_model=64, n_numbers=10)
        # 1 pass-through + 2 S_2 + 6 S_3 = 9
        assert model.group_moe.router.n_options == 9

    def test_routing_decision_fields(self):
        model = MultiGroupMoE(d_model=64, n_numbers=10)
        a = torch.randint(0, 10, (4,))
        op = torch.randint(0, 3, (4,))
        b = torch.randint(0, 10, (4,))
        c = torch.randint(0, 10, (4,))
        _, decision = model(a, op, b, c)
        assert decision.group_idx.shape == (4,)
        assert decision.logits.shape == (4, 9)

    def test_param_counts_comparable(self):
        moe = MultiGroupMoE(d_model=128, n_numbers=15)
        base = MultiGroupBaseline(d_model=128, n_numbers=15)
        ratio = moe.count_params() / base.count_params()
        assert 0.8 < ratio < 1.2, (
            f"moe={moe.count_params()}, base={base.count_params()}, ratio={ratio:.2f}"
        )

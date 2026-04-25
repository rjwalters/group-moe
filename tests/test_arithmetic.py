"""Tests for arithmetic dataset and models."""

import torch
import pytest

from src.data.arithmetic import ArithmeticDataset, OP_ADD, OP_SUB
from src.models.arithmetic import ArithmeticGroupMoE, ArithmeticBaseline


class TestArithmeticDataset:
    def test_pool_size(self):
        ds = ArithmeticDataset(split="train", num_range=10, train_frac=1.0)
        # 10*10 additions + 10*10 subtractions = 200
        assert len(ds) == 200

    def test_split_fractions(self):
        n = 100
        pool = 2 * n * n  # 20000
        train = ArithmeticDataset(split="train", num_range=n, train_frac=0.4)
        val = ArithmeticDataset(split="val", num_range=n, train_frac=0.4)
        test = ArithmeticDataset(split="test", num_range=n, train_frac=0.4)
        assert len(train) + len(val) + len(test) == pool

    def test_both_orderings_present(self):
        """Both (a,+,b) and (b,+,a) should be in the full pool."""
        ds = ArithmeticDataset(split="train", num_range=10, train_frac=1.0)
        pairs = {(a, op, b) for a, op, b, _ in ds.examples}
        assert (3, OP_ADD, 5) in pairs
        assert (5, OP_ADD, 3) in pairs

    def test_subtraction_targets(self):
        ds = ArithmeticDataset(split="train", num_range=20)
        for a, op, b, result in ds.examples:
            if op == OP_SUB:
                assert result == a - b

    def test_addition_targets(self):
        ds = ArithmeticDataset(split="train", num_range=20)
        for a, op, b, result in ds.examples:
            if op == OP_ADD:
                assert result == a + b

    def test_deterministic_splits(self):
        ds1 = ArithmeticDataset(split="train", seed=42)
        ds2 = ArithmeticDataset(split="train", seed=42)
        assert ds1.examples == ds2.examples

    def test_different_seed_gives_different_order(self):
        ds1 = ArithmeticDataset(split="train", seed=42)
        ds2 = ArithmeticDataset(split="train", seed=123)
        assert ds1.examples != ds2.examples

    def test_normalization_invertible(self):
        ds = ArithmeticDataset(split="train")
        sample = ds[0]
        raw = sample["target_raw"]
        normalized = sample["target"]
        recovered = ds.denormalize(normalized)
        assert torch.allclose(raw, recovered, atol=1e-4)

    def test_getitem_format(self):
        ds = ArithmeticDataset(split="train")
        item = ds[0]
        assert set(item.keys()) == {"a", "op", "b", "target", "target_raw"}
        assert item["a"].dtype == torch.long
        assert item["op"].dtype == torch.long
        assert item["b"].dtype == torch.long
        assert item["target"].dtype == torch.float32
        assert item["target_raw"].dtype == torch.float32

    def test_addition_pairs_method(self):
        ds = ArithmeticDataset(split="train", num_range=10, train_frac=1.0)
        pairs = ds.addition_pairs()
        # Should be a set of (a, b) tuples, all with op=ADD
        assert len(pairs) == 100  # 10*10
        assert (3, 5) in pairs


class TestArithmeticModels:
    def test_groupmoe_forward_shape(self):
        model = ArithmeticGroupMoE(d_model=64, n_numbers=20)
        a = torch.randint(0, 20, (8,))
        op = torch.randint(0, 2, (8,))
        b = torch.randint(0, 20, (8,))
        pred, decision = model(a, op, b)
        assert pred.shape == (8,)
        assert decision is not None

    def test_baseline_forward_shape(self):
        model = ArithmeticBaseline(d_model=64, n_numbers=20)
        a = torch.randint(0, 20, (8,))
        op = torch.randint(0, 2, (8,))
        b = torch.randint(0, 20, (8,))
        pred, decision = model(a, op, b)
        assert pred.shape == (8,)
        assert decision is None

    def test_param_counts_comparable(self):
        moe = ArithmeticGroupMoE(d_model=128)
        base = ArithmeticBaseline(d_model=128)
        ratio = moe.count_params() / base.count_params()
        assert 0.8 < ratio < 1.2, (
            f"Param counts too different: moe={moe.count_params()}, "
            f"base={base.count_params()}, ratio={ratio:.2f}"
        )

    def test_groupmoe_routing_decision_fields(self):
        model = ArithmeticGroupMoE(d_model=64, n_numbers=20)
        a = torch.randint(0, 20, (4,))
        op = torch.randint(0, 2, (4,))
        b = torch.randint(0, 20, (4,))
        _, decision = model(a, op, b)
        assert decision.group_idx.shape == (4,)
        assert decision.element_idx.shape == (4,)
        assert decision.confidence.shape == (4,)

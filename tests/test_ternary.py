"""Tests for ternary dataset and models."""

import itertools

import torch
import pytest

from src.data.ternary import TernaryDataset, OP_SYM, OP_NONSYM, symmetric_fn, nonsym_fn
from src.models.ternary import TernaryGroupMoE, TernaryBaseline


class TestTernaryFunctions:
    def test_symmetric_fn_invariant(self):
        """symmetric_fn should be invariant under all permutations."""
        for a, b, c in [(3, 5, 7), (0, 1, 2), (9, 9, 1)]:
            val = symmetric_fn(a, b, c)
            for p in itertools.permutations([a, b, c]):
                assert symmetric_fn(*p) == val

    def test_symmetric_fn_values(self):
        assert symmetric_fn(1, 2, 3) == 6
        assert symmetric_fn(0, 0, 0) == 0
        assert symmetric_fn(9, 9, 9) == 27

    def test_nonsym_not_symmetric(self):
        """nonsym_fn should NOT be invariant under permutations."""
        assert nonsym_fn(3, 5, 7) != nonsym_fn(7, 5, 3)
        assert nonsym_fn(1, 2, 3) == 2*1 - 2 + 3  # 3


class TestTernaryDataset:
    def test_random_pool_size(self):
        ds = TernaryDataset(split="train", num_range=5, train_frac=1.0,
                           split_mode="random")
        # 5^3 symmetric + 5^3 non-symmetric = 250
        assert len(ds) == 250

    def test_random_split_covers_pool(self):
        n = 5
        pool = 2 * n**3
        train = TernaryDataset(split="train", num_range=n, train_frac=0.5,
                              split_mode="random")
        val = TernaryDataset(split="val", num_range=n, train_frac=0.5,
                            split_mode="random")
        test = TernaryDataset(split="test", num_range=n, train_frac=0.5,
                             split_mode="random")
        assert len(train) + len(val) + len(test) == pool

    def test_symmetric_targets_correct(self):
        ds = TernaryDataset(split="train", num_range=10, split_mode="random",
                           train_frac=1.0)
        for a, op, b, c, result in ds.examples:
            if op == OP_SYM:
                assert result == symmetric_fn(a, b, c)

    def test_nonsym_targets_correct(self):
        ds = TernaryDataset(split="train", num_range=10, split_mode="random",
                           train_frac=1.0)
        for a, op, b, c, result in ds.examples:
            if op == OP_NONSYM:
                assert result == nonsym_fn(a, b, c)

    def test_complement_split_one_ordering_per_multiset(self):
        """In complement split, training symmetric set should have exactly
        one ordering per unordered multiset."""
        ds = TernaryDataset(split="train", num_range=8, split_mode="complement")
        sym_examples = [(a, b, c) for a, op, b, c, _ in ds.examples if op == OP_SYM]

        # Group by sorted tuple (the unordered multiset)
        seen_multisets = set()
        for a, b, c in sym_examples:
            key = tuple(sorted([a, b, c]))
            assert key not in seen_multisets, f"Duplicate multiset {key} in training"
            seen_multisets.add(key)

    def test_complement_split_test_has_complements(self):
        """Every symmetric test example should have a complement in train."""
        train = TernaryDataset(split="train", num_range=8, split_mode="complement")
        test = TernaryDataset(split="test", num_range=8, split_mode="complement")

        train_triples = train.symmetric_triples()

        for a, op, b, c, _ in test.examples:
            if op == OP_SYM:
                perms = list(itertools.permutations([a, b, c]))
                has_comp = any((p[0], p[1], p[2]) in train_triples for p in perms)
                assert has_comp, f"Test triple ({a},{b},{c}) has no complement in train"

    def test_complement_split_distinct_counts(self):
        """For num_range=5: C(5,3)=10 distinct triples, each with 6 orderings.
        Training should have 10 symmetric examples from distinct triples."""
        ds = TernaryDataset(split="train", num_range=5, split_mode="complement")
        sym = [(a, b, c) for a, op, b, c, _ in ds.examples if op == OP_SYM]
        distinct = [t for t in sym if len(set(t)) == 3]
        assert len(distinct) == 10  # C(5,3) = 10

    def test_complement_split_two_equal_counts(self):
        """For num_range=5: 5*4=20 {a,a,b} multisets, each with 3 orderings.
        Training should have 20 from two-equal."""
        ds = TernaryDataset(split="train", num_range=5, split_mode="complement")
        sym = [(a, b, c) for a, op, b, c, _ in ds.examples if op == OP_SYM]
        two_equal = [t for t in sym if len(set(t)) == 2]
        assert len(two_equal) == 20  # 5 * 4 = 20

    def test_composition_split_train_has_transpositions(self):
        """Composition split: train should have canonical + 3 transpositions."""
        ds = TernaryDataset(split="train", num_range=5, split_mode="composition")
        sym = [(a, b, c) for a, op, b, c, _ in ds.examples if op == OP_SYM]
        distinct = [t for t in sym if len(set(t)) == 3]
        # C(5,3)=10 triples × 4 orderings each = 40
        assert len(distinct) == 40

    def test_composition_split_test_has_3cycles(self):
        """Composition split: test should have only 3-cycle orderings."""
        ds = TernaryDataset(split="test", num_range=5, split_mode="composition")
        sym = [(a, b, c) for a, op, b, c, _ in ds.examples if op == OP_SYM]
        distinct = [t for t in sym if len(set(t)) == 3]
        # C(5,3)=10 triples × 2 three-cycles each = 20
        assert len(distinct) == 20

    def test_composition_split_test_only_3cycles(self):
        """Verify test orderings are actually 3-cycles of sorted canonical."""
        ds = TernaryDataset(split="test", num_range=6, split_mode="composition")
        for a, op, b, c, _ in ds.examples:
            if op != OP_SYM or len(set([a, b, c])) != 3:
                continue
            canonical = tuple(sorted([a, b, c]))
            # This ordering should be a 3-cycle of canonical, not a transposition
            ordering = (a, b, c)
            assert ordering in [
                (canonical[1], canonical[2], canonical[0]),  # (012)
                (canonical[2], canonical[0], canonical[1]),  # (021)
            ], f"Test ordering {ordering} is not a 3-cycle of {canonical}"

    def test_deterministic(self):
        ds1 = TernaryDataset(split="train", seed=42, split_mode="complement")
        ds2 = TernaryDataset(split="train", seed=42, split_mode="complement")
        assert ds1.examples == ds2.examples

    def test_normalization_invertible(self):
        ds = TernaryDataset(split="train")
        sample = ds[0]
        raw = sample["target_raw"]
        normalized = sample["target"]
        recovered = ds.denormalize(normalized)
        assert torch.allclose(raw, recovered, atol=1e-4)

    def test_getitem_format(self):
        ds = TernaryDataset(split="train")
        item = ds[0]
        assert set(item.keys()) == {"a", "op", "b", "c", "target", "target_raw"}
        assert item["a"].dtype == torch.long
        assert item["c"].dtype == torch.long


class TestTernaryModels:
    def test_groupmoe_forward_shape(self):
        model = TernaryGroupMoE(d_model=64, n_numbers=10)
        a = torch.randint(0, 10, (8,))
        op = torch.randint(0, 2, (8,))
        b = torch.randint(0, 10, (8,))
        c = torch.randint(0, 10, (8,))
        pred, decision = model(a, op, b, c)
        assert pred.shape == (8,)
        assert decision is not None

    def test_baseline_forward_shape(self):
        model = TernaryBaseline(d_model=64, n_numbers=10)
        a = torch.randint(0, 10, (8,))
        op = torch.randint(0, 2, (8,))
        b = torch.randint(0, 10, (8,))
        c = torch.randint(0, 10, (8,))
        pred, decision = model(a, op, b, c)
        assert pred.shape == (8,)
        assert decision is None

    def test_param_counts_comparable(self):
        moe = TernaryGroupMoE(d_model=128, n_numbers=10)
        base = TernaryBaseline(d_model=128, n_numbers=10)
        ratio = moe.count_params() / base.count_params()
        assert 0.8 < ratio < 1.2, (
            f"Param counts too different: moe={moe.count_params()}, "
            f"base={base.count_params()}, ratio={ratio:.2f}"
        )

    def test_groupmoe_routing_has_s3_options(self):
        model = TernaryGroupMoE(d_model=64, n_numbers=10)
        # S_3 has order 6, so router should have 7 options (1 pass-through + 6)
        assert model.group_moe.router.n_options == 7

    def test_groupmoe_routing_decision_fields(self):
        model = TernaryGroupMoE(d_model=64, n_numbers=10)
        a = torch.randint(0, 10, (4,))
        op = torch.randint(0, 2, (4,))
        b = torch.randint(0, 10, (4,))
        c = torch.randint(0, 10, (4,))
        _, decision = model(a, op, b, c)
        assert decision.group_idx.shape == (4,)
        assert decision.element_idx.shape == (4,)
        assert decision.confidence.shape == (4,)
        assert decision.logits.shape == (4, 7)

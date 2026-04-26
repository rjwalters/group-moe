"""Ternary dataset for Group-MoE S_3 experiments.

Generates (a, op, b, c) -> result examples where:
- op=0 (e_2): second elementary symmetric polynomial ab+ac+bc, S_3 invariant
- op=1 (nonsym): a*b - c, position-dependent, no symmetry

Two split modes:
- "random": random subsample
- "complement": for e_2, only ONE ordering per unordered multiset goes
  into training; remaining orderings go to test. This cleanly tests
  whether the S_3 expert enables transfer across permutations.

Uses regression (normalized targets) for smooth learning.
"""

from __future__ import annotations

import itertools

import numpy as np
import torch
from torch.utils.data import Dataset


OP_SYM = 0
OP_NONSYM = 1


def symmetric_fn(a: int, b: int, c: int) -> int:
    """Symmetric sum: a + b + c. Fully S_3-invariant."""
    return a + b + c


def nonsym_fn(a: int, b: int, c: int) -> int:
    """Non-symmetric function: 2*a - b + c. Position-dependent."""
    return 2 * a - b + c


class TernaryDataset(Dataset):
    """Dataset of ternary expressions (a, op, b, c) -> result.

    Args:
        split: one of 'train', 'val', 'test'
        num_range: numbers drawn from [0, num_range)
        seed: random seed for reproducible splits
        train_frac: fraction of non-symmetric examples used for training
        val_frac: fraction used for validation
        stats: pre-computed (mean, std) for normalization
        split_mode: "random" or "complement"
    """

    def __init__(
        self,
        split: str = "train",
        num_range: int = 10,
        seed: int = 42,
        train_frac: float = 0.5,
        val_frac: float = 0.1,
        stats: tuple[float, float] | None = None,
        split_mode: str = "complement",
    ):
        assert split in ("train", "val", "test")
        assert split_mode in ("random", "complement", "composition")
        self.split = split
        self.num_range = num_range

        if split_mode == "composition":
            self.examples = self._build_composition_split(
                split, num_range, seed, train_frac, val_frac,
            )
        elif split_mode == "complement":
            self.examples = self._build_complement_split(
                split, num_range, seed, train_frac, val_frac,
            )
        else:
            self.examples = self._build_random_split(
                split, num_range, seed, train_frac, val_frac,
            )

        raw_targets = np.array([ex[4] for ex in self.examples], dtype=np.float64)
        if stats is not None:
            self.mean, self.std = stats
        else:
            self.mean = float(raw_targets.mean())
            self.std = float(raw_targets.std()) if len(raw_targets) > 1 else 1.0

    @staticmethod
    def _build_random_split(
        split: str, num_range: int, seed: int,
        train_frac: float, val_frac: float,
    ) -> list[tuple[int, int, int, int, int]]:
        pool = []
        for a in range(num_range):
            for b in range(num_range):
                for c in range(num_range):
                    pool.append((a, OP_SYM, b, c, symmetric_fn(a, b, c)))
                    pool.append((a, OP_NONSYM, b, c, nonsym_fn(a, b, c)))

        rng = np.random.RandomState(seed)
        rng.shuffle(pool)

        n_train = int(len(pool) * train_frac)
        n_val = int(len(pool) * val_frac)

        if split == "train":
            return pool[:n_train]
        elif split == "val":
            return pool[n_train : n_train + n_val]
        else:
            return pool[n_train + n_val :]

    @staticmethod
    def _build_complement_split(
        split: str, num_range: int, seed: int,
        train_frac: float, val_frac: float,
    ) -> list[tuple[int, int, int, int, int]]:
        """For each unordered multiset {a,b,c}, one ordering trains, rest test.

        Handles three cases:
        - Distinct {a,b,c}: 6 orderings, 1 train / 5 test
        - Two-equal {a,a,b}: 3 orderings, 1 train / 2 test
        - All-equal {a,a,a}: 1 ordering, train only
        """
        rng = np.random.RandomState(seed)

        sym_train = []
        sym_test = []

        # Distinct triples: a < b < c
        for a in range(num_range):
            for b in range(a + 1, num_range):
                for c in range(b + 1, num_range):
                    orderings = list(itertools.permutations([a, b, c]))
                    rng.shuffle(orderings)
                    val = symmetric_fn(a, b, c)  # same for all orderings
                    sym_train.append((orderings[0][0], OP_SYM, orderings[0][1], orderings[0][2], val))
                    for o in orderings[1:]:
                        sym_test.append((o[0], OP_SYM, o[1], o[2], val))

        # Two-equal: {a,a,b} where a != b
        for a in range(num_range):
            for b in range(num_range):
                if a == b:
                    continue
                orderings = [(a, a, b), (a, b, a), (b, a, a)]
                rng.shuffle(orderings)
                val = symmetric_fn(a, a, b)
                sym_train.append((orderings[0][0], OP_SYM, orderings[0][1], orderings[0][2], val))
                for o in orderings[1:]:
                    sym_test.append((o[0], OP_SYM, o[1], o[2], val))

        # All-equal: {a,a,a}
        for a in range(num_range):
            val = symmetric_fn(a, a, a)
            sym_train.append((a, OP_SYM, a, a, val))

        # Non-symmetric op: split randomly
        nonsym_all = []
        for a in range(num_range):
            for b in range(num_range):
                for c in range(num_range):
                    nonsym_all.append((a, OP_NONSYM, b, c, nonsym_fn(a, b, c)))

        rng.shuffle(nonsym_all)
        n_nonsym_train = int(len(nonsym_all) * train_frac)
        n_nonsym_val = int(len(nonsym_all) * val_frac)
        nonsym_train = nonsym_all[:n_nonsym_train]
        nonsym_val = nonsym_all[n_nonsym_train : n_nonsym_train + n_nonsym_val]
        nonsym_test = nonsym_all[n_nonsym_train + n_nonsym_val :]

        # Combine
        if split == "train":
            examples = sym_train + nonsym_train
        elif split == "val":
            n_sym_val = max(1, int(len(sym_test) * val_frac))
            rng2 = np.random.RandomState(seed + 1)
            rng2.shuffle(sym_test)
            examples = sym_test[:n_sym_val] + nonsym_val
            # Restore sym_test for test split (this is a static method, so
            # each split call rebuilds independently)
        else:
            examples = sym_test + nonsym_test

        rng3 = np.random.RandomState(seed + 2)
        rng3.shuffle(examples)
        return examples

    @staticmethod
    def _build_composition_split(
        split: str, num_range: int, seed: int,
        train_frac: float, val_frac: float,
    ) -> list[tuple[int, int, int, int, int]]:
        """Compositional generalization split.

        For each distinct triple (a < b < c):
        - Train: canonical + all 3 transposition orderings (4 orderings)
        - Test: the 2 three-cycle orderings (compositions of transpositions)

        This tests whether the model can generalize from generators (transpositions)
        to composed elements (3-cycles). GroupMoE should generalize because
        R(3-cycle) = R(trans1) @ R(trans2) is pre-defined in the irrep basis.

        For non-distinct triples and non-symmetric op: same as complement split.
        """
        rng = np.random.RandomState(seed)

        sym_train = []
        sym_test = []

        # Distinct triples: a < b < c
        for a in range(num_range):
            for b in range(a + 1, num_range):
                for c in range(b + 1, num_range):
                    val = symmetric_fn(a, b, c)
                    # Canonical = sorted (a, b, c)
                    # Transpositions of canonical:
                    #   (01): (b, a, c)
                    #   (12): (a, c, b)
                    #   (02): (c, b, a)
                    # 3-cycles (compositions):
                    #   (012): (b, c, a)
                    #   (021): (c, a, b)
                    train_orderings = [(a,b,c), (b,a,c), (a,c,b), (c,b,a)]
                    test_orderings = [(b,c,a), (c,a,b)]

                    for o in train_orderings:
                        sym_train.append((o[0], OP_SYM, o[1], o[2], val))
                    for o in test_orderings:
                        sym_test.append((o[0], OP_SYM, o[1], o[2], val))

        # Two-equal {a,a,b}: cyclic rotations are (a,a,b), (a,b,a), (b,a,a)
        # (a,a,b) → (a,b,a) is like a transposition of positions 1,2
        # (a,a,b) → (b,a,a) is like a 3-cycle
        # Train: first 2, test: last 1
        for a in range(num_range):
            for b in range(num_range):
                if a == b:
                    continue
                val = symmetric_fn(a, a, b)
                orderings = [(a,a,b), (a,b,a), (b,a,a)]
                # First two are "generators", third is composition-like
                for o in orderings[:2]:
                    sym_train.append((o[0], OP_SYM, o[1], o[2], val))
                sym_test.append((orderings[2][0], OP_SYM, orderings[2][1], orderings[2][2], val))

        # All-equal: train only
        for a in range(num_range):
            sym_train.append((a, OP_SYM, a, a, symmetric_fn(a, a, a)))

        # Non-symmetric: random split
        nonsym_all = []
        for a in range(num_range):
            for b in range(num_range):
                for c in range(num_range):
                    nonsym_all.append((a, OP_NONSYM, b, c, nonsym_fn(a, b, c)))

        rng.shuffle(nonsym_all)
        n_train = int(len(nonsym_all) * train_frac)
        n_val = int(len(nonsym_all) * val_frac)
        nonsym_train = nonsym_all[:n_train]
        nonsym_val = nonsym_all[n_train : n_train + n_val]
        nonsym_test = nonsym_all[n_train + n_val :]

        if split == "train":
            examples = sym_train + nonsym_train
        elif split == "val":
            rng2 = np.random.RandomState(seed + 1)
            n_sym_val = max(1, int(len(sym_test) * val_frac))
            rng2.shuffle(sym_test)
            examples = sym_test[:n_sym_val] + nonsym_val
        else:
            examples = sym_test + nonsym_test

        rng3 = np.random.RandomState(seed + 2)
        rng3.shuffle(examples)
        return examples

    def get_stats(self) -> tuple[float, float]:
        return (self.mean, self.std)

    def denormalize(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor * self.std + self.mean

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        a, op, b, c, result = self.examples[idx]
        target_raw = float(result)
        target = (target_raw - self.mean) / self.std
        return {
            "a": torch.tensor(a, dtype=torch.long),
            "op": torch.tensor(op, dtype=torch.long),
            "b": torch.tensor(b, dtype=torch.long),
            "c": torch.tensor(c, dtype=torch.long),
            "target": torch.tensor(target, dtype=torch.float32),
            "target_raw": torch.tensor(target_raw, dtype=torch.float32),
        }

    def symmetric_triples(self) -> set[tuple[int, int, int]]:
        """Return the set of (a, b, c) triples for symmetric examples."""
        return {(a, b, c) for a, op, b, c, _ in self.examples if op == OP_SYM}

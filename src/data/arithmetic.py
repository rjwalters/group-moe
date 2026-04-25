"""Arithmetic dataset for Group-MoE experiments.

Generates (a, op, b) -> result examples where:
- op=0 (+): commutative, exhibits S_2 symmetry
- op=1 (-): non-commutative, no symmetry

Two split modes:
- "random": random subsample (original behavior)
- "complement": for addition, only ONE ordering per unordered pair
  goes into training; the reversed ordering goes to test. This cleanly
  tests whether the model can transfer (a,+,b) -> (b,+,a).

Uses regression (normalized targets) for smooth learning.
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset


OP_ADD = 0
OP_SUB = 1


class ArithmeticDataset(Dataset):
    """Dataset of arithmetic expressions (a, op, b) -> result.

    Args:
        split: one of 'train', 'val', 'test'
        num_range: numbers drawn from [0, num_range)
        seed: random seed for reproducible splits
        train_frac: fraction of pool used for training (random mode),
                    or fraction of subtraction pairs used for training
                    (complement mode)
        val_frac: fraction used for validation (remainder goes to test)
        stats: pre-computed (mean, std) for normalization; if None, computed from data
        split_mode: "random" or "complement"
    """

    def __init__(
        self,
        split: str = "train",
        num_range: int = 100,
        seed: int = 42,
        train_frac: float = 0.4,
        val_frac: float = 0.1,
        stats: tuple[float, float] | None = None,
        split_mode: str = "random",
    ):
        assert split in ("train", "val", "test")
        assert split_mode in ("random", "complement")
        self.split = split
        self.num_range = num_range

        if split_mode == "complement":
            self.examples = self._build_complement_split(
                split, num_range, seed, train_frac, val_frac,
            )
        else:
            self.examples = self._build_random_split(
                split, num_range, seed, train_frac, val_frac,
            )

        # Normalization stats
        raw_targets = np.array([ex[3] for ex in self.examples], dtype=np.float64)
        if stats is not None:
            self.mean, self.std = stats
        else:
            self.mean = float(raw_targets.mean())
            self.std = float(raw_targets.std())

    @staticmethod
    def _build_random_split(
        split: str, num_range: int, seed: int,
        train_frac: float, val_frac: float,
    ) -> list[tuple[int, int, int, int]]:
        pool = []
        for a in range(num_range):
            for b in range(num_range):
                pool.append((a, OP_ADD, b, a + b))
                pool.append((a, OP_SUB, b, a - b))

        rng = np.random.RandomState(seed)
        indices = rng.permutation(len(pool))
        pool = [pool[i] for i in indices]

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
    ) -> list[tuple[int, int, int, int]]:
        """Split where for each addition pair {a,b} (a!=b), only one
        ordering is in training and the reverse is in test.

        Diagonal additions (a+a) go to training.
        Subtraction pairs are split randomly by train_frac.
        """
        rng = np.random.RandomState(seed)

        # --- Addition ---
        # Diagonal (a+a): always train
        add_train = [(a, OP_ADD, a, 2 * a) for a in range(num_range)]

        # Off-diagonal: for each unordered pair {a,b}, randomly pick
        # which ordering goes to train vs test
        add_test = []
        for a in range(num_range):
            for b in range(a + 1, num_range):
                if rng.random() < 0.5:
                    add_train.append((a, OP_ADD, b, a + b))
                    add_test.append((b, OP_ADD, a, b + a))
                else:
                    add_train.append((b, OP_ADD, a, b + a))
                    add_test.append((a, OP_ADD, b, a + b))

        # --- Subtraction (no symmetry, split randomly) ---
        sub_all = []
        for a in range(num_range):
            for b in range(num_range):
                sub_all.append((a, OP_SUB, b, a - b))

        rng.shuffle(sub_all)
        n_sub_train = int(len(sub_all) * train_frac)
        n_sub_val = int(len(sub_all) * val_frac)
        sub_train = sub_all[:n_sub_train]
        sub_val = sub_all[n_sub_train : n_sub_train + n_sub_val]
        sub_test = sub_all[n_sub_train + n_sub_val :]

        # Combine
        if split == "train":
            examples = add_train + sub_train
        elif split == "val":
            # Val gets a small random subset of addition test for monitoring
            n_add_val = max(1, int(len(add_test) * val_frac))
            rng.shuffle(add_test)
            examples = add_test[:n_add_val] + sub_val
        else:
            examples = add_test + sub_test

        rng2 = np.random.RandomState(seed + 1)
        rng2.shuffle(examples)
        return examples

    def get_stats(self) -> tuple[float, float]:
        return (self.mean, self.std)

    def denormalize(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor * self.std + self.mean

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        a, op, b, result = self.examples[idx]
        target_raw = float(result)
        target = (target_raw - self.mean) / self.std
        return {
            "a": torch.tensor(a, dtype=torch.long),
            "op": torch.tensor(op, dtype=torch.long),
            "b": torch.tensor(b, dtype=torch.long),
            "target": torch.tensor(target, dtype=torch.float32),
            "target_raw": torch.tensor(target_raw, dtype=torch.float32),
        }

    def addition_pairs(self) -> set[tuple[int, int]]:
        """Return the set of (a, b) pairs for addition examples in this split."""
        return {(a, b) for a, op, b, _ in self.examples if op == OP_ADD}

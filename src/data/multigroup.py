"""Multi-group dataset for testing router discrimination between S_2 and S_3.

Three operations on (a, b, c) with different symmetries:
- op=0 (S_3): a + b + c — fully S_3-invariant
- op=1 (S_2): a + b + 2*c — invariant under a↔b swap only
- op=2 (none): 2*a - b + c — no symmetry

Complement split is per-operation:
- S_3 op: 1 ordering per unordered multiset, rest to test
- S_2 op: 1 ordering of (a,b) per pair+c, reverse to test
- None op: random split
"""

from __future__ import annotations

import itertools

import numpy as np
import torch
from torch.utils.data import Dataset


OP_S3 = 0
OP_S2 = 1
OP_NONE = 2


def fn_s3(a: int, b: int, c: int) -> int:
    """S_3-invariant: a + b + c."""
    return a + b + c


def fn_s2(a: int, b: int, c: int) -> int:
    """S_2-invariant (a↔b swap): a + b + 2*c."""
    return a + b + 2 * c


def fn_none(a: int, b: int, c: int) -> int:
    """Non-symmetric: 2*a - b + c."""
    return 2 * a - b + c


class MultiGroupDataset(Dataset):
    """Dataset with three operations at different symmetry levels.

    Args:
        split: one of 'train', 'val', 'test'
        num_range: numbers drawn from [0, num_range)
        seed: random seed
        train_frac: fraction for non-symmetric op random split
        val_frac: fraction for validation
        stats: pre-computed (mean, std) for normalization
    """

    def __init__(
        self,
        split: str = "train",
        num_range: int = 15,
        seed: int = 42,
        train_frac: float = 0.5,
        val_frac: float = 0.1,
        stats: tuple[float, float] | None = None,
    ):
        assert split in ("train", "val", "test")
        self.split = split
        self.num_range = num_range
        self.examples = self._build_split(
            split, num_range, seed, train_frac, val_frac,
        )

        raw_targets = np.array([ex[4] for ex in self.examples], dtype=np.float64)
        if stats is not None:
            self.mean, self.std = stats
        else:
            self.mean = float(raw_targets.mean())
            self.std = float(raw_targets.std()) if len(raw_targets) > 1 else 1.0

    @staticmethod
    def _build_split(
        split: str, num_range: int, seed: int,
        train_frac: float, val_frac: float,
    ) -> list[tuple[int, int, int, int, int]]:
        rng = np.random.RandomState(seed)

        # --- Op 0: S_3 complement split (same as ternary) ---
        s3_train, s3_test = [], []

        # Distinct triples
        for a in range(num_range):
            for b in range(a + 1, num_range):
                for c in range(b + 1, num_range):
                    orderings = list(itertools.permutations([a, b, c]))
                    rng.shuffle(orderings)
                    val = fn_s3(a, b, c)
                    s3_train.append((orderings[0][0], OP_S3, orderings[0][1], orderings[0][2], val))
                    for o in orderings[1:]:
                        s3_test.append((o[0], OP_S3, o[1], o[2], val))

        # Two-equal
        for a in range(num_range):
            for b in range(num_range):
                if a == b:
                    continue
                orderings = [(a, a, b), (a, b, a), (b, a, a)]
                rng.shuffle(orderings)
                val = fn_s3(a, a, b)
                s3_train.append((orderings[0][0], OP_S3, orderings[0][1], orderings[0][2], val))
                for o in orderings[1:]:
                    s3_test.append((o[0], OP_S3, o[1], o[2], val))

        # All-equal
        for a in range(num_range):
            s3_train.append((a, OP_S3, a, a, fn_s3(a, a, a)))

        # --- Op 1: S_2 complement split (swap a↔b) ---
        s2_train, s2_test = [], []

        for a in range(num_range):
            for b in range(a + 1, num_range):
                for c in range(num_range):
                    val = fn_s2(a, b, c)  # same as fn_s2(b, a, c)
                    if rng.random() < 0.5:
                        s2_train.append((a, OP_S2, b, c, val))
                        s2_test.append((b, OP_S2, a, c, val))
                    else:
                        s2_train.append((b, OP_S2, a, c, val))
                        s2_test.append((a, OP_S2, b, c, val))

        # a == b: only one ordering, goes to train
        for a in range(num_range):
            for c in range(num_range):
                s2_train.append((a, OP_S2, a, c, fn_s2(a, a, c)))

        # --- Op 2: non-symmetric, random split ---
        none_all = []
        for a in range(num_range):
            for b in range(num_range):
                for c in range(num_range):
                    none_all.append((a, OP_NONE, b, c, fn_none(a, b, c)))

        rng.shuffle(none_all)
        n_none_train = int(len(none_all) * train_frac)
        n_none_val = int(len(none_all) * val_frac)
        none_train = none_all[:n_none_train]
        none_val = none_all[n_none_train : n_none_train + n_none_val]
        none_test = none_all[n_none_train + n_none_val :]

        # --- Combine ---
        if split == "train":
            examples = s3_train + s2_train + none_train
        elif split == "val":
            rng2 = np.random.RandomState(seed + 1)
            n_s3_val = max(1, int(len(s3_test) * val_frac))
            n_s2_val = max(1, int(len(s2_test) * val_frac))
            rng2.shuffle(s3_test)
            rng2.shuffle(s2_test)
            examples = s3_test[:n_s3_val] + s2_test[:n_s2_val] + none_val
        else:
            examples = s3_test + s2_test + none_test

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

    def triples_for_op(self, op: int) -> set[tuple[int, int, int]]:
        """Return set of (a, b, c) triples for a given operation."""
        return {(a, b, c) for a, o, b, c, _ in self.examples if o == op}

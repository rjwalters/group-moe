"""Disparate-groups dataset: Z_2 vs Z_3 (non-nested groups).

Three operations on (a, b, c) with genuinely different symmetries:
- op=0 (Z_2): a + b + 2*c — invariant under a↔b swap
- op=1 (Z_3): a²(b-c) + b²(c-a) + c²(a-b) — invariant under cyclic shift,
               NOT invariant under swap (changes sign). Cubic.
- op=2 (none): 2*a - b + c — no symmetry

Z_2 and Z_3 are non-nested: neither is a subgroup of the other.
This tests whether the router can learn to dispatch to genuinely
different experts.

Complement splits:
- Z_2 op: swap a↔b complement
- Z_3 op: cyclic rotation complement (1 of 3 rotations train, 2 test)
- None op: random split
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset


OP_Z2 = 0
OP_Z3 = 1
OP_NONE = 2


def fn_z2(a: int, b: int, c: int) -> int:
    """Z_2-invariant (a↔b swap): a + b + 2*c."""
    return a + b + 2 * c


def fn_z3(a: int, b: int, c: int) -> int:
    """Z_3-invariant (cyclic shift), not S_3-invariant.
    f(a,b,c) = a²(b-c) + b²(c-a) + c²(a-b)
    """
    return a * a * (b - c) + b * b * (c - a) + c * c * (a - b)


def fn_none(a: int, b: int, c: int) -> int:
    """Non-symmetric: 2*a - b + c."""
    return 2 * a - b + c


class DisparateDataset(Dataset):
    """Dataset with Z_2 and Z_3 operations (non-nested groups).

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

        # --- Op 0: Z_2 complement (swap a↔b) ---
        z2_train, z2_test = [], []

        for a in range(num_range):
            for b in range(a + 1, num_range):
                for c in range(num_range):
                    val = fn_z2(a, b, c)  # same as fn_z2(b, a, c)
                    if rng.random() < 0.5:
                        z2_train.append((a, OP_Z2, b, c, val))
                        z2_test.append((b, OP_Z2, a, c, val))
                    else:
                        z2_train.append((b, OP_Z2, a, c, val))
                        z2_test.append((a, OP_Z2, b, c, val))

        # a == b: only one ordering
        for a in range(num_range):
            for c in range(num_range):
                z2_train.append((a, OP_Z2, a, c, fn_z2(a, a, c)))

        # --- Op 1: Z_3 complement (cyclic rotation) ---
        # Three cyclic rotations: (a,b,c), (b,c,a), (c,a,b)
        # For each unordered cyclic orbit, put 1 rotation in train, 2 in test
        z3_train, z3_test = [], []

        # Track which orbits we've already handled
        seen_orbits: set[tuple[int, int, int]] = set()

        for a in range(num_range):
            for b in range(num_range):
                for c in range(num_range):
                    # Canonical form: lexicographically smallest rotation
                    rotations = [(a, b, c), (b, c, a), (c, a, b)]
                    canonical = min(rotations)
                    if canonical in seen_orbits:
                        continue
                    seen_orbits.add(canonical)

                    # How many distinct rotations?
                    unique_rotations = list(dict.fromkeys(rotations))  # preserves order, deduplicates
                    val = fn_z3(a, b, c)  # same for all cyclic rotations

                    if len(unique_rotations) == 1:
                        # All same (a,a,a): just train
                        z3_train.append((a, OP_Z3, b, c, val))
                    else:
                        rng.shuffle(unique_rotations)
                        r = unique_rotations[0]
                        z3_train.append((r[0], OP_Z3, r[1], r[2], val))
                        for r in unique_rotations[1:]:
                            z3_test.append((r[0], OP_Z3, r[1], r[2], val))

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
            examples = z2_train + z3_train + none_train
        elif split == "val":
            rng2 = np.random.RandomState(seed + 1)
            n_z2_val = max(1, int(len(z2_test) * val_frac))
            n_z3_val = max(1, int(len(z3_test) * val_frac))
            rng2.shuffle(z2_test)
            rng2.shuffle(z3_test)
            examples = z2_test[:n_z2_val] + z3_test[:n_z3_val] + none_val
        else:
            examples = z2_test + z3_test + none_test

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

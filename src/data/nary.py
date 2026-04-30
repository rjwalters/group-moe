"""General n-ary dataset for S_n scaling experiments.

Generates (x_1, ..., x_n, op) -> result examples where:
- op=0 (symmetric): sum(x_1, ..., x_n) — fully S_n-invariant
- op=1 (non-symmetric): weighted sum with distinct weights — position-dependent

Complement split: for each unordered multiset, one ordering trains,
remaining orderings go to test. Generalizes from ternary.py to any n.
"""

from __future__ import annotations

from itertools import combinations_with_replacement, permutations

import numpy as np
import torch
from torch.utils.data import Dataset


OP_SYM = 0
OP_NONSYM = 1


def symmetric_fn(inputs: tuple[int, ...]) -> int:
    """S_n-invariant: sum of all inputs."""
    return sum(inputs)


def nonsym_fn(inputs: tuple[int, ...]) -> int:
    """Position-dependent: weighted sum with weights 1, 2, ..., n."""
    return sum((i + 1) * x for i, x in enumerate(inputs))


class NaryDataset(Dataset):
    """Dataset of n-ary expressions (x_1, ..., x_n, op) -> result.

    Args:
        n: number of inputs (arity)
        split: one of 'train', 'val', 'test'
        num_range: numbers drawn from [0, num_range)
        seed: random seed
        train_frac: fraction for non-symmetric op random split
        val_frac: fraction for validation
        stats: pre-computed (mean, std) for normalization
    """

    def __init__(
        self,
        n: int = 3,
        split: str = "train",
        num_range: int = 8,
        seed: int = 42,
        train_frac: float = 0.5,
        val_frac: float = 0.1,
        stats: tuple[float, float] | None = None,
    ):
        assert split in ("train", "val", "test")
        self.n = n
        self.split = split
        self.num_range = num_range

        # examples: list of (inputs_tuple, op, result)
        self.examples = self._build_split(
            n, split, num_range, seed, train_frac, val_frac,
        )

        raw_targets = np.array([ex[2] for ex in self.examples], dtype=np.float64)
        if stats is not None:
            self.mean, self.std = stats
        else:
            self.mean = float(raw_targets.mean())
            self.std = float(raw_targets.std()) if len(raw_targets) > 1 else 1.0

    @staticmethod
    def _build_split(
        n: int, split: str, num_range: int, seed: int,
        train_frac: float, val_frac: float,
    ) -> list[tuple[tuple[int, ...], int, int]]:
        rng = np.random.RandomState(seed)

        # --- Symmetric op: complement split ---
        sym_train: list[tuple[tuple[int, ...], int, int]] = []
        sym_test: list[tuple[tuple[int, ...], int, int]] = []

        for combo in combinations_with_replacement(range(num_range), n):
            # All unique orderings of this multiset
            orderings = list(set(permutations(combo)))
            rng.shuffle(orderings)
            val = symmetric_fn(combo)

            # First ordering to train, rest to test
            sym_train.append((orderings[0], OP_SYM, val))
            for o in orderings[1:]:
                sym_test.append((o, OP_SYM, val))

        # --- Non-symmetric op: random split ---
        nonsym_all: list[tuple[tuple[int, ...], int, int]] = []
        # Sample rather than enumerate for large n
        total_nonsym = num_range ** n
        if total_nonsym <= 50000:
            # Enumerate all
            for combo in _product_tuples(num_range, n):
                nonsym_all.append((combo, OP_NONSYM, nonsym_fn(combo)))
        else:
            # Sample to keep dataset manageable
            for _ in range(50000):
                combo = tuple(rng.randint(0, num_range, size=n).tolist())
                nonsym_all.append((combo, OP_NONSYM, nonsym_fn(combo)))

        rng.shuffle(nonsym_all)
        n_train = int(len(nonsym_all) * train_frac)
        n_val = int(len(nonsym_all) * val_frac)

        if split == "train":
            examples = sym_train + nonsym_all[:n_train]
        elif split == "val":
            rng2 = np.random.RandomState(seed + 1)
            n_sym_val = max(1, int(len(sym_test) * val_frac))
            rng2.shuffle(sym_test)
            examples = sym_test[:n_sym_val] + nonsym_all[n_train:n_train + n_val]
        else:
            examples = sym_test + nonsym_all[n_train + n_val:]

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
        inputs, op, result = self.examples[idx]
        target_raw = float(result)
        target = (target_raw - self.mean) / self.std
        return {
            "inputs": torch.tensor(inputs, dtype=torch.long),
            "op": torch.tensor(op, dtype=torch.long),
            "target": torch.tensor(target, dtype=torch.float32),
            "target_raw": torch.tensor(target_raw, dtype=torch.float32),
        }

    def symmetric_inputs(self) -> set[tuple[int, ...]]:
        """Return set of input tuples for symmetric examples."""
        return {inp for inp, op, _ in self.examples if op == OP_SYM}


def _product_tuples(num_range: int, n: int):
    """Generate all n-tuples from [0, num_range)."""
    if n == 0:
        yield ()
        return
    for x in range(num_range):
        for rest in _product_tuples(num_range, n - 1):
            yield (x,) + rest

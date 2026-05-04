"""QM9 data loader with the canonical 110k/10k/13k split.

Wraps torch_geometric.datasets.QM9. Targets are indexed as in PyG's QM9 docs;
target index 7 is U0 (internal energy at 0K), the headline property.

Atomic reference energies are subtracted so the model predicts the molecular
binding contribution rather than the sum of isolated-atom energies.
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch_geometric.datasets import QM9
from torch_geometric.loader import DataLoader


# Reference energies for isolated atoms (Hartree), QM9 convention.
# Source: standard QM9 atomref values used in SchNet / torch_geometric examples.
ATOMREF_U0 = {
    1: -0.500273,    # H
    6: -37.846772,   # C
    7: -54.583861,   # N
    8: -75.064579,   # O
    9: -99.718730,   # F
}

# Hartree → eV
HARTREE_TO_EV = 27.211386024367243


def get_atomref_tensor(max_z: int = 100) -> torch.Tensor:
    """Atomref as a (max_z,) tensor in eV, indexable by atomic number.

    Default size 100 matches torch_geometric.nn.SchNet's internal embedding.
    """
    ref = torch.zeros(max_z)
    for z, e in ATOMREF_U0.items():
        ref[z] = e * HARTREE_TO_EV
    return ref


def load_qm9(
    root: str | Path = "data/qm9",
    target_idx: int = 7,
    seed: int = 0,
) -> tuple[QM9, dict[str, torch.Tensor]]:
    """Load QM9 and produce a 110k/10k/13k random split.

    Returns the dataset and a dict of {train, val, test} index tensors.
    """
    dataset = QM9(root=str(root))
    n = len(dataset)
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=g)
    split = {
        "train": perm[:110_000],
        "val": perm[110_000:120_000],
        "test": perm[120_000:],
    }
    return dataset, split


def make_loaders(
    dataset: QM9,
    split: dict[str, torch.Tensor],
    batch_size: int = 64,
    num_workers: int = 0,
) -> dict[str, DataLoader]:
    """Build PyG DataLoaders for each split."""
    return {
        name: DataLoader(
            dataset[idx],
            batch_size=batch_size,
            shuffle=(name == "train"),
            num_workers=num_workers,
        )
        for name, idx in split.items()
    }

"""MD17 data loader for force prediction tasks.

MD17 (Chmiela et al. 2017) is a benchmark for ML potentials on molecular
dynamics trajectories. Each dataset is one molecule sampled from MD at 500K;
each conformation has positions, energy (kcal/mol), and per-atom forces
(kcal/mol/Å). The standard task is to predict both energy and forces.

Why this matters for Group-MoE: forces are *vectors*. Unlike QM9's energy
(invariant scalar), force prediction has equivariant outputs and is where
SO(3)-equivariant models outperform invariant ones (SchNet < PaiNN < MACE
on MD17). If selective equivariance has any traction, this is the task to
test on. See `docs/paper2_routes_forward.md` Route 1.

Standard split (Chmiela / SchNet convention):
    train: 1000 conformations
    val:   1000 conformations
    test:  rest (~hundreds of thousands)

We use original MD17 rather than rMD17 because the rMD17 download URL on
materialscloud is currently dead in PyG 2.7.0 (HTTP 404). Original MD17 is
slightly noisier but the headline benchmark numbers in the literature
predominantly use it.

PyG-MD17 conventions (verified via inspection):
    sample.z       (n_atoms,) atomic numbers (long)
    sample.pos     (n_atoms, 3) positions (Å)
    sample.energy  (1,) energy in kcal/mol
    sample.force   (n_atoms, 3) forces in kcal/mol/Å (note: PyG might
                                  use `force` or `dy`; we normalize to `force`)
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch_geometric.datasets import MD17
from torch_geometric.loader import DataLoader


# kcal/mol → eV (1 kcal/mol = 0.043361 eV). MD17 native unit is kcal/mol;
# we usually want eV (matches QM9 / our SchNet baseline) for cross-task comparability.
KCAL_PER_MOL_TO_EV = 0.043361254529175


# Available MD17 molecule names (per PyG MD17 file_names dict).
# We deliberately list the conventional benchmark molecules; CCSD(T) and
# revised variants are also available but not the primary targets.
MOLECULES_SCHNET_BENCHMARK = [
    "aspirin",
    "ethanol",
    "malonaldehyde",
    "naphthalene",
    "salicylic acid",
    "toluene",
    "uracil",
]


def load_md17(
    name: str = "aspirin",
    root: str | Path = "data/md17",
) -> MD17:
    """Load one MD17 molecule trajectory.

    Args:
        name: molecule name. One of `MOLECULES_SCHNET_BENCHMARK`, or any
            key from PyG's `MD17.file_names`.
        root: dataset root. PyG MD17 places raw and processed files under
            `<root>/<name>/`, so pass a single shared root for all molecules.

    Returns:
        PyG MD17 dataset object. `len(dataset)` is the number of conformations.
    """
    return MD17(root=str(root), name=name)


def make_split(
    n_conformations: int,
    n_train: int = 1000,
    n_val: int = 1000,
    seed: int = 0,
) -> dict[str, torch.Tensor]:
    """Standard SchNet-paper MD17 split: 1000 train / 1000 val / rest test.

    Random shuffle with the given seed. The remaining ~hundred-thousand-plus
    conformations form the test set.
    """
    if n_train + n_val > n_conformations:
        raise ValueError(
            f"Asked for {n_train}+{n_val} train+val samples but dataset has only "
            f"{n_conformations} conformations."
        )
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n_conformations, generator=g)
    return {
        "train": perm[:n_train],
        "val": perm[n_train : n_train + n_val],
        "test": perm[n_train + n_val :],
    }


def make_loaders(
    dataset: MD17,
    split: dict[str, torch.Tensor],
    batch_size: int = 4,
    num_workers: int = 0,
) -> dict[str, DataLoader]:
    """Build PyG DataLoaders. MD17 conformations are large in count but a single
    molecule, so per-batch graph size is fixed (one molecule per sample).

    Default batch_size=4 follows the SchNet/PaiNN papers. Larger batches help
    on big GPUs but force training is gradient-intensive (autograd of energy
    w.r.t. positions, with create_graph=True), so memory use scales steeply.
    """
    return {
        name: DataLoader(
            dataset[idx],
            batch_size=batch_size,
            shuffle=(name == "train"),
            num_workers=num_workers,
        )
        for name, idx in split.items()
    }


def normalize_force_field_name(sample) -> str:
    """Return the field name PyG used for forces ('force' or 'dy')."""
    if hasattr(sample, "force"):
        return "force"
    if hasattr(sample, "dy"):
        return "dy"
    raise AttributeError(f"MD17 sample has neither 'force' nor 'dy': {list(sample.keys())}")

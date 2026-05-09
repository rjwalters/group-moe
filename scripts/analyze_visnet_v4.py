"""Analyze v4 ViSNet's learned per-atom representations.

Step B of the post-v2-sweep plan (`docs/paper2_routes_forward.md`):
characterize how ViSNet uses its vector features. The key question for
deciding whether MoE-in-ViSNet has signal to learn:

  Are the vector norms ‖v_atom‖ varied across atoms (i.e., does ViSNet
  actually USE the equivariant pathway differently for different atoms)
  or roughly uniform (in which case selective compute couldn't help)?

If norms vary substantially with atom type / chemical environment, the
router has signal to learn and MoE-in-ViSNet is worth the $40 Lambda run.
If norms are uniform, the equivariance is mostly pro-forma and selective
equivariance can't gain much.

Usage:
    python scripts/analyze_visnet_v4.py --ckpt data/qm9/visnet_baseline_v4/best.pt
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.qm9 import get_atomref_tensor, load_qm9, make_loaders


# Element symbol lookup for atomic numbers we expect in QM9 (H, C, N, O, F).
ATOMIC_SYMBOL = {1: "H", 6: "C", 7: "N", 8: "O", 9: "F"}


def patch_visnet_block_to_return_intermediate(visnet):
    """Monkey-patch ViSNet's representation_model to return per-atom (x, v)
    instead of going through the output reduction. Returns a callable that
    takes (z, pos, batch) and returns (x_per_atom, v_per_atom).
    """
    rep = visnet.representation_model
    return lambda z, pos, batch: rep(z, pos, batch)  # ViSNetBlock.forward already returns (x, v)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True,
                        help="Path to v4 best.pt — produced by train_qm9_visnet.py.")
    parser.add_argument("--n-batches", type=int, default=20,
                        help="How many val batches to analyze (each is 100 molecules).")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    config = ckpt["config"]
    print(f"[load] checkpoint epoch={ckpt.get('epoch', '?')}, val_mae={ckpt.get('val_mae_eV', '?'):.5f} eV")
    print(f"[load] config: hidden={config['hidden']}, num_layers={config['num_layers']}, lmax={config['lmax']}")

    # Reconstruct ViSNet — must match training-time arch exactly
    from torch_geometric.nn.models import ViSNet
    atomref = get_atomref_tensor()
    model = ViSNet(
        lmax=config["lmax"],
        num_heads=config["num_heads"],
        num_layers=config["num_layers"],
        hidden_channels=config["hidden"],
        num_rbf=config["num_rbf"],
        cutoff=config["cutoff"],
        max_num_neighbors=config["max_num_neighbors"],
        atomref=atomref,
        mean=ckpt.get("mean_eV", 0.0),
        std=ckpt.get("std_eV", 1.0),
        derivative=False,
    )
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    print("[load] loading QM9 val split ...")
    dataset, split = load_qm9(seed=args.seed)
    loaders = make_loaders(dataset, split, batch_size=100)
    val_iter = iter(loaders["val"])

    # We only need the representation_model output (x, v), not the energy.
    rep = model.representation_model

    # Per-atom statistics, grouped by atomic number.
    # For each atom: ||v_atom|| — the L2 norm over (spatial × hidden) channels.
    norms_by_z: dict[int, list[float]] = defaultdict(list)
    x_norms_by_z: dict[int, list[float]] = defaultdict(list)

    with torch.no_grad():
        for i, batch in enumerate(val_iter):
            if i >= args.n_batches:
                break
            x, v = rep(batch.z, batch.pos, batch.batch)
            # x: (n_atoms, hidden), v: (n_atoms, (lmax+1)^2 - 1, hidden)
            v_norm_per_atom = v.norm(dim=(1, 2))   # (n_atoms,)
            x_norm_per_atom = x.norm(dim=1)        # (n_atoms,)
            for z, vn, xn in zip(batch.z.tolist(), v_norm_per_atom.tolist(), x_norm_per_atom.tolist()):
                norms_by_z[z].append(vn)
                x_norms_by_z[z].append(xn)

    # Report aggregate norm statistics per element.
    print()
    print(f"{'element':10s} {'count':>8s}  {'mean ||v||':>12s} {'std ||v||':>12s}  {'mean ||x||':>12s} {'std ||x||':>12s}")
    print("-" * 80)
    for z in sorted(norms_by_z.keys()):
        sym = ATOMIC_SYMBOL.get(z, f"Z={z}")
        v_arr = torch.tensor(norms_by_z[z])
        x_arr = torch.tensor(x_norms_by_z[z])
        print(
            f"{sym:10s} {v_arr.numel():>8d}  "
            f"{v_arr.mean().item():>12.4f} {v_arr.std().item():>12.4f}  "
            f"{x_arr.mean().item():>12.4f} {x_arr.std().item():>12.4f}"
        )

    # Within-element variation: even within (say) carbons, do norms vary widely?
    # High within-element variation → router could learn to discriminate sp³ vs sp² etc.
    print()
    print("[interpretation]")
    all_v_norms = torch.tensor([n for ns in norms_by_z.values() for n in ns])
    print(f"  overall ||v|| stats: mean={all_v_norms.mean().item():.4f}, std={all_v_norms.std().item():.4f}")
    # Coefficient of variation (CV = std/mean) — measure of "how much equivariance is being used"
    cv_overall = all_v_norms.std().item() / max(all_v_norms.mean().item(), 1e-9)
    print(f"  overall CV(||v||) = {cv_overall:.3f}")
    if 6 in norms_by_z:
        c_arr = torch.tensor(norms_by_z[6])
        cv_carbon = c_arr.std().item() / max(c_arr.mean().item(), 1e-9)
        print(f"  CV(||v||) within carbon atoms = {cv_carbon:.3f}")
    print()
    print("Reading:")
    print(" - Overall CV > 0.5 with high within-element CV (carbon especially) →")
    print("   ViSNet uses the equivariant pathway differently per atom; MoE-in-ViSNet")
    print("   has plenty of signal to learn from.")
    print(" - Low CV (<0.2) overall and within-element →")
    print("   equivariance is roughly uniform; routing has little to discriminate.")


if __name__ == "__main__":
    main()

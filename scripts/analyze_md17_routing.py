"""Per-atom routing analysis for SchNet+GroupMoE on MD17.

Loads a trained checkpoint, runs inference on N val/test conformations, and
records the router's per-atom expert choice. Outputs two views:

  1. **Per-element distribution** — for each Z (H, C, N, O, F, ...), what
     fraction of atoms go to pass-through / tetrahedral / octahedral / planar?
     This is the cleanest summary: does the router separate atom types?

  2. **Per-atom-position distribution** — MD17 holds the same molecule across
     all conformations, so each atom index has a fixed chemical role (e.g.
     atom 0 is always the same C). For each position, what fraction of
     conformations route it to which expert? This reveals position-specific
     specialization (aromatic ring C vs methyl C, etc.).

Run on the aspirin checkpoint first — it has the strongest GroupMoE win
(−22% F_MAE) and the most chemical heterogeneity (C9H8O4 = 6 ring carbons +
COOH + COCH3 + 4 distinct O environments).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.md17 import (
    load_md17,
    make_loaders,
    make_split,
)
from src.models.schnet_groupmoe import SchNetGroupMoE
from scripts.train_md17_groupmoe import CdistRadiusGraph


Z_TO_SYMBOL = {1: "H", 6: "C", 7: "N", 8: "O", 9: "F"}


def load_checkpoint(ckpt_path: Path, device: torch.device) -> tuple[SchNetGroupMoE, dict]:
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    interaction_graph = CdistRadiusGraph(cfg["cutoff"])
    model = SchNetGroupMoE(
        hidden_channels=cfg["hidden"],
        num_filters=cfg["hidden"],
        num_interactions=cfg["num_interactions"],
        num_gaussians=cfg["num_gaussians"],
        cutoff=cfg["cutoff"],
        interaction_graph=interaction_graph,
        atomref=None,
        # mean/std are baked into the state_dict's mean/std buffers; pass dummy
        # placeholders so the parent class wires them in.
        mean=0.0,
        std=1.0,
        moe_position=cfg.get("moe_position"),
        load_balance_weight=cfg.get("load_balance_weight", 0.01),
        include_irrep_norms=cfg.get("include_irrep_norms", False),
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, cfg


def collect_routing(
    model: SchNetGroupMoE,
    loader,
    device: torch.device,
    max_batches: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run inference and return (atomic_numbers, expert_idx, atom_position_in_mol).

    atom_position_in_mol is the within-molecule index (0..n_atoms-1), recovered
    from the PyG batch index. For a single-molecule MD17 dataset every batch
    has identical atom counts and orderings, so this is the chemical position.

    Routing is computed under no_grad — we only need the router's argmax, not
    the energy/force outputs.
    """
    expert_names = ["pass_through"] + [t.name for t in model.symmetry_types]
    atomic_nums = []
    expert_idx = []
    atom_pos = []
    n_batches = 0
    with torch.no_grad():
        for i, batch in enumerate(loader):
            if max_batches is not None and i >= max_batches:
                break
            z = batch.z.to(device)
            pos = batch.pos.to(device)
            batch_idx = batch.batch.to(device)
            _, decision, _ = model(z, pos, batch_idx)
            # within-batch atom index: counts atoms per molecule, restarting at 0
            # for each new molecule. Equivalent to per-molecule position index.
            n_per_mol = torch.bincount(batch_idx)
            pos_in_mol = torch.cat([torch.arange(n.item(), device=device) for n in n_per_mol])
            atomic_nums.append(z.cpu().numpy())
            expert_idx.append(decision.expert_idx.cpu().numpy())
            atom_pos.append(pos_in_mol.cpu().numpy())
            n_batches += 1
    print(f"[analyze] processed {n_batches} batches", flush=True)
    return (
        np.concatenate(atomic_nums),
        np.concatenate(expert_idx),
        np.concatenate(atom_pos),
    ), expert_names


def per_element_table(z: np.ndarray, idx: np.ndarray, expert_names: list[str]) -> dict:
    """Routing distribution per element (atomic number)."""
    table = {}
    for zi in sorted(set(z.tolist())):
        mask = z == zi
        n = mask.sum()
        if n == 0:
            continue
        dist = {}
        for k, name in enumerate(expert_names):
            dist[name] = float((idx[mask] == k).mean())
        table[Z_TO_SYMBOL.get(zi, f"Z{zi}")] = {
            "n_atoms": int(n),
            "distribution": dist,
        }
    return table


def per_position_table(
    z: np.ndarray, idx: np.ndarray, pos: np.ndarray, expert_names: list[str]
) -> dict:
    """Routing distribution per within-molecule atom position. MD17 has one
    molecule per dataset, so position = chemical identity (always the same atom).
    """
    table = {}
    for pi in sorted(set(pos.tolist())):
        mask = pos == pi
        n = mask.sum()
        if n == 0:
            continue
        # Element should be constant across conformations for a fixed position.
        elem_vals = z[mask]
        elem = elem_vals[0]
        elem_consistent = bool((elem_vals == elem).all())
        dist = {}
        for k, name in enumerate(expert_names):
            dist[name] = float((idx[mask] == k).mean())
        # Modal expert + how peaky the distribution is
        modal_idx = int(np.argmax([dist[n] for n in expert_names]))
        modal_name = expert_names[modal_idx]
        modal_rate = dist[modal_name]
        table[int(pi)] = {
            "element": Z_TO_SYMBOL.get(int(elem), f"Z{int(elem)}"),
            "element_consistent": elem_consistent,
            "n_conformations": int(n),
            "distribution": dist,
            "modal_expert": modal_name,
            "modal_rate": modal_rate,
        }
    return table


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=str,
                        default="data/md17/results/groupmoe_md17_aspirin_alc6")
    parser.add_argument("--molecule", type=str, default=None,
                        help="MD17 molecule name; default = read from checkpoint config.")
    parser.add_argument("--max-batches", type=int, default=200,
                        help="200 batches × 4 = 800 conformations. Enough for stable stats.")
    parser.add_argument("--split", choices=["val", "test"], default="test")
    parser.add_argument("--out", type=str, default=None,
                        help="Output JSON path; default = <run-dir>/routing_analysis.json")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    ckpt_path = run_dir / "best.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"No checkpoint at {ckpt_path}")

    device = torch.device("cuda" if torch.cuda.is_available()
                          else "mps" if torch.backends.mps.is_available()
                          else "cpu")
    print(f"[analyze] device={device}", flush=True)
    print(f"[analyze] loading checkpoint: {ckpt_path}", flush=True)
    model, cfg = load_checkpoint(ckpt_path, device)
    print(f"[analyze] symmetry_types: {[t.name for t in model.symmetry_types]}", flush=True)
    print(f"[analyze] moe_position: {model.moe_position}/{cfg['num_interactions']}", flush=True)

    molecule = args.molecule or cfg["molecule"]
    print(f"[analyze] loading MD17 ({molecule}) ...", flush=True)
    dataset = load_md17(molecule)
    split = make_split(len(dataset), n_train=cfg["n_train"], n_val=cfg["n_val"], seed=cfg["seed"])
    loaders = make_loaders(dataset, split, batch_size=cfg["batch_size"])
    print(f"[analyze] molecule has {dataset[0].z.numel()} atoms; "
          f"using {args.split} split with {args.max_batches} batches max.", flush=True)

    (z, idx, pos), expert_names = collect_routing(
        model, loaders[args.split], device, max_batches=args.max_batches
    )
    print(f"[analyze] total atoms observed: {len(z):,}", flush=True)

    by_element = per_element_table(z, idx, expert_names)
    by_position = per_position_table(z, idx, pos, expert_names)

    # Pretty-print per-element table
    print("\n=== Per-element routing distribution ===")
    header = f"{'elem':>6s}  {'n_atoms':>9s}  " + "  ".join(f"{n:>14s}" for n in expert_names)
    print(header)
    for elem, info in by_element.items():
        row = f"{elem:>6s}  {info['n_atoms']:>9d}  " + "  ".join(
            f"{info['distribution'][n]:>14.3f}" for n in expert_names
        )
        print(row)

    # Pretty-print per-position table
    print("\n=== Per-position routing modal expert ===")
    print(f"{'pos':>4s}  {'elem':>4s}  {'modal_expert':>14s}  {'modal_rate':>10s}  " +
          "  ".join(f"{n:>10s}" for n in expert_names))
    for pi, info in sorted(by_position.items()):
        elem = info["element"]
        dist = info["distribution"]
        print(
            f"{pi:>4d}  {elem:>4s}  {info['modal_expert']:>14s}  {info['modal_rate']:>10.3f}  " +
            "  ".join(f"{dist[n]:>10.3f}" for n in expert_names)
        )

    out_path = Path(args.out) if args.out else run_dir / "routing_analysis.json"
    out_path.write_text(json.dumps({
        "molecule": molecule,
        "split": args.split,
        "n_conformations_used": int(len(z) / dataset[0].z.numel()),
        "expert_names": expert_names,
        "per_element": by_element,
        "per_position": by_position,
    }, indent=2))
    print(f"\n[analyze] wrote {out_path}")


if __name__ == "__main__":
    main()

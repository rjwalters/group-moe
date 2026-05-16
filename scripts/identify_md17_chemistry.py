"""Identify chemical role of each atom position in an MD17 molecule.

MD17 stores fixed atom ordering across conformations. We assign each atom a
human-readable chemical role label (e.g. "ring-C", "methyl-C", "carbonyl-O")
by analyzing bond topology on the first conformation:

  1. Build pairwise distances; threshold at element-pair-specific bond cutoffs
     to get the bond graph (C-C ~1.5Å, C=O ~1.2Å, C-O ~1.4Å, O-H ~1.0Å, etc.).
  2. Hydrogens are leaves: classify by what their parent atom looks like.
  3. Carbons: count neighbors. sp3 (4 nbr) → tetrahedral / methyl; sp2 (3 nbr,
     with one C=O) → carbonyl; sp2 in aromatic ring → ring-C.
  4. Oxygens: =O (one short C=O) → carbonyl-O; -O-H → hydroxyl-O; -O- between
     two C's → ester-O.

This is conformation-1 bond topology; MD17 atom indices are fixed across
the trajectory, so the label transfers to all conformations.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.md17 import load_md17


Z_TO_SYMBOL = {1: "H", 6: "C", 7: "N", 8: "O", 9: "F"}


# Conservative covalent-bond cutoffs (Å). Anything closer is "bonded".
# Single-vs-double distinction by length: short C-O (<1.3) is C=O; etc.
BOND_CUTOFFS = {
    frozenset({1, 1}): 0.0,    # no H-H
    frozenset({1, 6}): 1.25,
    frozenset({1, 8}): 1.20,
    frozenset({1, 7}): 1.20,
    frozenset({6, 6}): 1.70,
    frozenset({6, 8}): 1.55,
    frozenset({6, 7}): 1.60,
    frozenset({8, 8}): 1.55,
}
DOUBLE_C_O = 1.30   # below this, C-O is C=O


def build_bonds(z: np.ndarray, pos: np.ndarray) -> list[set[int]]:
    """Return adjacency list: bonds[i] = set of atom indices bonded to atom i."""
    n = len(z)
    bonds = [set() for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            cutoff = BOND_CUTOFFS.get(frozenset({int(z[i]), int(z[j])}), 0)
            if cutoff <= 0:
                continue
            d = np.linalg.norm(pos[i] - pos[j])
            if d < cutoff:
                bonds[i].add(j)
                bonds[j].add(i)
    return bonds


def label_atoms(z: np.ndarray, pos: np.ndarray) -> list[str]:
    """Assign a chemistry label to each atom position."""
    n = len(z)
    bonds = build_bonds(z, pos)
    labels: list[str] = [""] * n

    # First pass: classify heavy atoms (C, N, O)
    for i in range(n):
        zi = int(z[i])
        if zi == 1:
            continue  # H later, depends on parent

        nbrs = bonds[i]
        nbr_elements = [int(z[j]) for j in nbrs]
        n_c = sum(1 for e in nbr_elements if e == 6)
        n_h = sum(1 for e in nbr_elements if e == 1)
        n_o = sum(1 for e in nbr_elements if e == 8)

        if zi == 6:  # carbon
            o_dists = sorted(np.linalg.norm(pos[i] - pos[j]) for j in nbrs if int(z[j]) == 8)
            has_double_bond_o = len(o_dists) > 0 and o_dists[0] < DOUBLE_C_O
            if has_double_bond_o and n_o >= 2:
                labels[i] = "C(carboxyl/ester carbonyl)"  # both C=O and C-O
            elif has_double_bond_o:
                labels[i] = "C(carbonyl)"  # just C=O
            elif n_h == 3:
                labels[i] = "C(methyl)"
            elif n_h == 2 and len(nbrs) == 4:
                labels[i] = "C(methylene)"
            elif len(nbrs) == 3 and n_c >= 2:
                # 3 nbrs and at least 2 are C → likely aromatic ring.
                # Substituted ring carbons (ipso position) have no H neighbor;
                # unsubstituted ring carbons carry one ring H. Worth distinguishing
                # because they're chemically and electronically different.
                if n_h == 0:
                    labels[i] = "C(ring, ipso/substituted)"
                else:
                    labels[i] = "C(ring, with H)"
            else:
                labels[i] = f"C(deg{len(nbrs)})"

        elif zi == 8:  # oxygen
            c_dists = sorted(np.linalg.norm(pos[i] - pos[j]) for j in nbrs if int(z[j]) == 6)
            has_double_bond_c = len(c_dists) > 0 and c_dists[0] < DOUBLE_C_O
            if n_h == 1:
                labels[i] = "O(hydroxyl, -OH)"
            elif has_double_bond_c:
                labels[i] = "O(carbonyl, =O)"
            elif n_c == 2:
                labels[i] = "O(ester linker, -O-)"
            else:
                labels[i] = f"O(deg{len(nbrs)})"

        elif zi == 7:  # nitrogen
            labels[i] = f"N(deg{len(nbrs)})"

    # Second pass: hydrogens classified by parent
    parent_label_to_h_label = {
        "C(ring, with H)": "H(aromatic, ring)",
        "C(methyl)": "H(methyl, -CH3)",
        "C(methylene)": "H(methylene, -CH2-)",
        "O(hydroxyl, -OH)": "H(hydroxyl, -OH)",
    }
    for i in range(n):
        if int(z[i]) != 1:
            continue
        if not bonds[i]:
            labels[i] = "H(isolated?)"
            continue
        parent = next(iter(bonds[i]))
        parent_label = labels[parent]
        labels[i] = parent_label_to_h_label.get(parent_label, f"H(via {parent_label})")

    return labels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--molecule", type=str, default="aspirin")
    parser.add_argument("--out", type=str, default=None,
                        help="JSON output. Default: data/md17/<mol>/chemistry_labels.json")
    args = parser.parse_args()

    print(f"[chem] loading MD17 ({args.molecule})", flush=True)
    dataset = load_md17(args.molecule)
    sample = dataset[0]
    z = sample.z.numpy()
    pos = sample.pos.numpy()
    print(f"[chem] {len(z)} atoms: {[Z_TO_SYMBOL.get(int(zi), '?') for zi in z]}", flush=True)

    labels = label_atoms(z, pos)

    print("\nIndex | Z | Element | Label")
    for i, (zi, lab) in enumerate(zip(z, labels)):
        print(f"{i:>5d} | {int(zi):>2d} | {Z_TO_SYMBOL.get(int(zi),'?'):>7s} | {lab}")

    out_path = Path(args.out) if args.out else Path(f"data/md17/{args.molecule}/chemistry_labels.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({
        "molecule": args.molecule,
        "atomic_numbers": [int(zi) for zi in z],
        "labels": labels,
    }, indent=2))
    print(f"\n[chem] wrote {out_path}")


if __name__ == "__main__":
    main()

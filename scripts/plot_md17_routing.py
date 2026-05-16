"""Plot per-position routing for an MD17 GroupMoE run.

Stacked bar chart: rows = atom positions, x = fraction routed to each expert.
Companion to `analyze_md17_routing.py` — reads its routing_analysis.json output.

Output PNG saved to <run-dir>/routing_<molecule>.png by default.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# Color order: pass-through (gray), tetrahedral (blue), octahedral (orange), planar (green).
# Stable across plots so the legend reads the same.
COLORS = {
    "pass_through": "#888888",
    "tetrahedral": "#1f77b4",
    "octahedral": "#ff7f0e",
    "planar": "#2ca02c",
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis", type=str,
                        default="data/md17/results/groupmoe_md17_aspirin_alc6/routing_analysis.json")
    parser.add_argument("--out", type=str, default=None,
                        help="Output PNG path. Default: <run-dir>/routing_<molecule>.png")
    args = parser.parse_args()

    data = json.loads(Path(args.analysis).read_text())
    molecule = data["molecule"]
    expert_names = data["expert_names"]
    per_pos = data["per_position"]
    per_elem = data["per_element"]

    # Sort positions numerically; keys are stringified ints in JSON.
    positions = sorted(per_pos.keys(), key=int)
    n_pos = len(positions)

    fig, axes = plt.subplots(1, 2, figsize=(13, max(5, n_pos * 0.28)),
                             gridspec_kw={"width_ratios": [3, 1]})

    # -------- Left: per-position stacked horizontal bars --------
    ax = axes[0]
    y = np.arange(n_pos)
    left = np.zeros(n_pos)
    for name in expert_names:
        widths = np.array([per_pos[p]["distribution"][name] for p in positions])
        ax.barh(y, widths, left=left, color=COLORS[name], label=name, edgecolor="white", linewidth=0.5)
        left += widths

    # Y-tick labels: position index + element symbol
    labels = [f"{p}  ({per_pos[p]['element']})" for p in positions]
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlim(0, 1)
    ax.set_xlabel("Fraction of conformations routed to expert", fontsize=10)
    ax.set_ylabel("Atom position (chemical role fixed across conformations)", fontsize=10)
    ax.set_title(f"Per-position routing — {molecule}", fontsize=11)
    ax.invert_yaxis()  # position 0 at top
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.06),
              ncol=len(expert_names), fontsize=9, frameon=False)
    ax.grid(axis="x", alpha=0.3, linestyle="--")

    # -------- Right: per-element summary bars --------
    ax2 = axes[1]
    elements = list(per_elem.keys())
    n_elem = len(elements)
    y2 = np.arange(n_elem)
    left2 = np.zeros(n_elem)
    for name in expert_names:
        widths = np.array([per_elem[e]["distribution"][name] for e in elements])
        ax2.barh(y2, widths, left=left2, color=COLORS[name], edgecolor="white", linewidth=0.5)
        left2 += widths
    ax2.set_yticks(y2)
    ax2.set_yticklabels(elements, fontsize=10)
    ax2.set_xlim(0, 1)
    ax2.set_xlabel("Fraction", fontsize=10)
    ax2.set_title(f"Per-element\nrouting", fontsize=11)
    ax2.invert_yaxis()
    ax2.grid(axis="x", alpha=0.3, linestyle="--")

    plt.tight_layout()
    out_path = Path(args.out) if args.out else Path(args.analysis).parent / f"routing_{molecule}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()

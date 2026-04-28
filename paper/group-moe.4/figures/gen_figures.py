"""Generate all paper figures from existing experimental data."""

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 9,
    'figure.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.labelsize': 10,
    'axes.titlesize': 11,
    'legend.fontsize': 8,
})

COLORS = {
    'groupmoe': '#1f77b4',
    'standardmoe': '#ff7f0e',
    'baseline': '#2ca02c',
}
LABELS = {
    'groupmoe': 'Group-MoE (irrep R(g))',
    'standardmoe': 'Standard MoE (learned W)',
    'baseline': 'Baseline (no expert)',
}

figdir = Path(__file__).parent


def fig1_architecture():
    """Architecture diagram."""
    fig, ax = plt.subplots(figsize=(7, 2.5))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 3)
    ax.axis('off')

    boxes = [
        (0.5, 1.0, 1.5, 1.0, 'Input\n$x \\in \\mathbb{R}^d$', '#e8e8e8'),
        (2.5, 1.0, 1.5, 1.0, 'Standard\nLayers', '#d4e6f1'),
        (4.5, 1.8, 1.2, 0.8, 'Router', '#fdebd0'),
        (4.5, 0.4, 1.2, 0.8, 'Pass-\nthrough', '#e8e8e8'),
        (6.2, 1.8, 1.5, 0.8, 'Group Expert\n$P \\to R(g) \\to P^\\dagger$', '#d5f5e3'),
        (8.2, 1.0, 1.5, 1.0, 'Standard\nLayers', '#d4e6f1'),
    ]

    for x, y, w, h, label, color in boxes:
        rect = mpatches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.05",
                                        facecolor=color, edgecolor='black', linewidth=0.8)
        ax.add_patch(rect)
        ax.text(x + w/2, y + h/2, label, ha='center', va='center', fontsize=7.5)

    # Arrows
    arrow_kw = dict(arrowstyle='->', color='black', linewidth=1.0)
    ax.annotate('', xy=(2.5, 1.5), xytext=(2.0, 1.5), arrowprops=arrow_kw)
    ax.annotate('', xy=(4.5, 2.2), xytext=(4.0, 1.7), arrowprops=arrow_kw)
    ax.annotate('', xy=(4.5, 0.8), xytext=(4.0, 1.3), arrowprops=arrow_kw)
    ax.annotate('', xy=(6.2, 2.2), xytext=(5.7, 2.2), arrowprops=arrow_kw)
    ax.annotate('', xy=(8.2, 1.5), xytext=(7.7, 1.8), arrowprops=arrow_kw)
    ax.annotate('', xy=(8.2, 1.3), xytext=(5.7, 0.8), arrowprops=arrow_kw)

    ax.text(4.3, 2.7, 'Symmetry detected?', fontsize=7, ha='center', style='italic', color='#666')
    ax.text(5.1, 2.6, 'yes', fontsize=7, color='#27ae60')
    ax.text(4.2, 0.2, 'no', fontsize=7, color='#c0392b')

    fig.savefig(figdir / 'fig1_architecture.pdf')
    fig.savefig(figdir / 'fig1_architecture.png')
    plt.close()
    print("Generated fig1_architecture")


def fig2_s2_heatmaps():
    """S_2 complement transfer heatmaps from existing analysis data."""
    analysis_dir = Path(__file__).parent.parent.parent.parent / 'data' / 'analysis'
    if not (analysis_dir / 'complement_analysis.png').exists():
        print("SKIP fig2_s2_heatmaps — run scripts/analyze_complement.py first")
        return

    # Copy the existing analysis figure
    import shutil
    shutil.copy(analysis_dir / 'complement_analysis.png', figdir / 'fig2_s2_heatmaps.png')
    print("Copied fig2_s2_heatmaps from existing analysis")


def fig3_threeway():
    """Three-way comparison bar chart (complement + composition)."""
    fig, axes = plt.subplots(1, 2, figsize=(7, 3))

    models = ['groupmoe', 'standardmoe', 'baseline']
    complement = [92.1, 88.2, 86.1]
    composition = [99.0, 98.1, 96.2]

    x = np.arange(len(models))
    width = 0.6

    for ax, data, title, ylim_low in [(axes[0], complement, 'Complement Split (1→5)', 82),
                                       (axes[1], composition, 'Composition Split (4→2)', 94)]:
        bars = ax.bar(x, data, width, color=[COLORS[m] for m in models], edgecolor='black', linewidth=0.5)
        ax.set_ylabel('Complement Accuracy (%)')
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(['Group-\nMoE', 'Standard\nMoE', 'Baseline'], fontsize=8)
        ax.set_ylim(ylim_low, 100.5)
        ax.axhline(y=100, color='gray', linestyle=':', alpha=0.3)

        # Value labels
        for bar, val in zip(bars, data):
            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.3,
                   f'{val:.1f}%', ha='center', va='bottom', fontsize=8, fontweight='bold')

        # Decomposition annotations
        if title.startswith('Complement'):
            ax.annotate('', xy=(0, 88.2), xytext=(0, 92.1),
                       arrowprops=dict(arrowstyle='<->', color='#c0392b', linewidth=1.5))
            ax.text(-0.55, 90.1, '+3.9pp\ngroup', fontsize=6.5, color='#c0392b', ha='center')
            ax.annotate('', xy=(2, 86.1), xytext=(2, 88.2),
                       arrowprops=dict(arrowstyle='<->', color='#7f8c8d', linewidth=1.0))
            ax.text(2.55, 87.1, '+2.1pp\nrouting', fontsize=6.5, color='#7f8c8d', ha='center')

    plt.tight_layout()
    fig.savefig(figdir / 'fig3_threeway.pdf')
    fig.savefig(figdir / 'fig3_threeway.png')
    plt.close()
    print("Generated fig3_threeway")


def fig4_router_discrimination():
    """Router discrimination: nested vs non-nested groups."""
    fig, axes = plt.subplots(1, 2, figsize=(7, 3))

    # Nested: S_2 + S_3
    ax = axes[0]
    ops = ['S₃ op', 'S₂ op', 'none op']
    pass_rate = [4.7, 5.8, 15.2]
    s2_rate = [25.1, 22.7, 31.2]
    s3_rate = [70.1, 71.6, 53.7]

    x = np.arange(len(ops))
    w = 0.25
    ax.bar(x - w, pass_rate, w, label='Pass-through', color='#95a5a6', edgecolor='black', linewidth=0.5)
    ax.bar(x, s2_rate, w, label='S₂ expert', color='#3498db', edgecolor='black', linewidth=0.5)
    ax.bar(x + w, s3_rate, w, label='S₃ expert', color='#e74c3c', edgecolor='black', linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(ops, fontsize=8)
    ax.set_ylabel('Routing Rate (%)')
    ax.set_title('Nested: S₂ ⊂ S₃')
    ax.legend(fontsize=7, loc='upper left')
    ax.set_ylim(0, 85)

    # Non-nested: Z_2 + Z_3
    ax = axes[1]
    ops = ['Z₂ op', 'Z₃ op', 'none op']
    pass_rate = [0.0, 66.0, 0.0]
    z2_rate = [75.7, 14.8, 0.7]
    z3_rate = [24.3, 19.2, 99.3]

    ax.bar(x - w, pass_rate, w, label='Pass-through', color='#95a5a6', edgecolor='black', linewidth=0.5)
    ax.bar(x, z2_rate, w, label='Z₂ expert', color='#3498db', edgecolor='black', linewidth=0.5)
    ax.bar(x + w, z3_rate, w, label='Z₃ expert', color='#e74c3c', edgecolor='black', linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(ops, fontsize=8)
    ax.set_title('Non-nested: Z₂, Z₃')
    ax.legend(fontsize=7, loc='upper left')
    ax.set_ylim(0, 110)

    plt.tight_layout()
    fig.savefig(figdir / 'fig4_router.pdf')
    fig.savefig(figdir / 'fig4_router.png')
    plt.close()
    print("Generated fig4_router")


def fig5_complement_split_diagram():
    """Visual explanation of the complement split."""
    fig, ax = plt.subplots(figsize=(6, 2.5))
    ax.axis('off')
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 4)

    # S_3 example: {1,2,3} with 6 orderings
    orderings = ['(1,2,3)', '(1,3,2)', '(2,1,3)', '(2,3,1)', '(3,1,2)', '(3,2,1)']
    perm_types = ['e', '(12)', '(01)', '(012)', '(021)', '(02)']

    ax.text(5, 3.7, 'Complement Split for {1, 2, 3}', ha='center', fontsize=11, fontweight='bold')
    ax.text(2, 3.2, 'TRAIN (1 ordering)', ha='center', fontsize=9, color='#27ae60')
    ax.text(7, 3.2, 'TEST (5 orderings)', ha='center', fontsize=9, color='#c0392b')

    # Train box
    rect = mpatches.FancyBboxPatch((0.8, 1.5), 2.4, 1.2, boxstyle="round,pad=0.1",
                                    facecolor='#d5f5e3', edgecolor='#27ae60', linewidth=1.5)
    ax.add_patch(rect)
    ax.text(2, 2.4, orderings[0], ha='center', fontsize=10, fontfamily='monospace')
    ax.text(2, 1.8, 'identity', ha='center', fontsize=7, color='#666')

    # Test boxes
    for i, (o, p) in enumerate(zip(orderings[1:], perm_types[1:])):
        x_pos = 4.5 + i * 1.1
        rect = mpatches.FancyBboxPatch((x_pos - 0.45, 1.5), 0.9, 1.2, boxstyle="round,pad=0.05",
                                        facecolor='#fadbd8', edgecolor='#c0392b', linewidth=1.0)
        ax.add_patch(rect)
        ax.text(x_pos, 2.4, o, ha='center', fontsize=7.5, fontfamily='monospace')
        ax.text(x_pos, 1.8, p, ha='center', fontsize=6, color='#666')

    ax.text(5, 0.8, 'All orderings compute the same value: f(1,2,3) = 1+2+3 = 6',
            ha='center', fontsize=8, style='italic', color='#555')
    ax.text(5, 0.3, 'Group-MoE transfers via R(g); baseline must learn each independently',
            ha='center', fontsize=8, style='italic', color='#555')

    fig.savefig(figdir / 'fig5_complement_split.pdf')
    fig.savefig(figdir / 'fig5_complement_split.png')
    plt.close()
    print("Generated fig5_complement_split")


if __name__ == '__main__':
    fig1_architecture()
    fig2_s2_heatmaps()
    fig3_threeway()
    fig4_router_discrimination()
    fig5_complement_split_diagram()
    print("\nAll figures generated in", figdir)

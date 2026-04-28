"""Regenerate fig3 with 5-seed means and error bars."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 9,
    'figure.dpi': 300,
    'savefig.bbox': 'tight',
})

fig, axes = plt.subplots(1, 2, figsize=(7, 3))

models = ['Group-\nMoE', 'Standard\nMoE', 'Baseline']
colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

# 5-seed data
comp_means = [89.2, 90.2, 87.0]
comp_stds = [3.1, 2.0, 1.6]
cpos_means = [98.7, 98.6, 97.4]
cpos_stds = [0.8, 0.8, 1.1]

x = np.arange(3)
width = 0.6

for ax, means, stds, title, ylim_low in [
    (axes[0], comp_means, comp_stds, 'Complement Split (1→5)', 80),
    (axes[1], cpos_means, cpos_stds, 'Composition Split (4→2)', 94),
]:
    bars = ax.bar(x, means, width, yerr=stds, capsize=4,
                  color=colors, edgecolor='black', linewidth=0.5,
                  error_kw={'linewidth': 1.2})
    ax.set_ylabel('Complement Accuracy (%)')
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=8)
    ax.set_ylim(ylim_low, 102)
    ax.axhline(y=100, color='gray', linestyle=':', alpha=0.3)

    for bar, m, s in zip(bars, means, stds):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + s + 0.5,
               f'{m:.1f}%', ha='center', va='bottom', fontsize=8, fontweight='bold')

# Decomposition annotations on complement
ax = axes[0]
ax.annotate('', xy=(1, 87.0), xytext=(1, 90.2),
           arrowprops=dict(arrowstyle='<->', color='#7f8c8d', linewidth=1.5))
ax.text(1.55, 88.5, '+3.2pp\nrouting', fontsize=6.5, color='#7f8c8d', ha='center')

ax.text(0.5, 82, '5 seeds each\nmean ± std', fontsize=7, color='#888', ha='center', style='italic')

plt.tight_layout()
fig.savefig('fig3_threeway.pdf')
fig.savefig('fig3_threeway.png')
plt.close()
print("Generated updated fig3_threeway")

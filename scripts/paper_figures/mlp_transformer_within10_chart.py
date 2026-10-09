"""
MLP vs. Transformer, interpolation vs. extrapolation: R-squared and
within-+/-10% accuracy, built directly from this paper's own Table 10
(NRC-derived data). No new numbers; this is a grouped-bar view of a table
already in the paper text, added specifically to make the within-+/-10%
comparison (the practically relevant "how often is this close enough"
metric) visible as a chart rather than only as table cells.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import sys
sys.path.insert(0, "scripts/paper_figures")
from style_common import set_pub_style, add_outer_border
set_pub_style()

INK = "#0b0b0b"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"
SURFACE = "#fcfcfb"

COLORS = {"blue": "#2a78d6", "orange": "#eb6834"}


def style_axes(ax):
    ax.set_facecolor(SURFACE)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    for spine in ["left", "bottom"]:
        ax.spines[spine].set_color(BASELINE)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)


splits = ["Interpolation\n(n=2,459)", "Extrapolation\n(n=13,748)"]
mlp_r2 = [0.951, 0.838]
tf_r2 = [0.957, 0.844]
mlp_within10 = [47.3, 25.8]
tf_within10 = [47.7, 20.2]
mlp_mape = [17.9, 44.6]
tf_mape = [16.5, 49.5]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 5.2), dpi=200)

x = np.arange(len(splits))
width = 0.32

bars1a = ax1.bar(x - width / 2, mlp_r2, width, color=COLORS["blue"], label="MLP", zorder=3,
                  edgecolor=INK, linewidth=0.5)
bars1b = ax1.bar(x + width / 2, tf_r2, width, color=COLORS["orange"], label="Transformer", zorder=3,
                  edgecolor=INK, linewidth=0.5, hatch="///")
for b, v in zip(list(bars1a) + list(bars1b), mlp_r2 + tf_r2):
    ax1.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.3f}", ha="center", fontsize=8.5, color=INK)
ax1.set_xticks(x)
ax1.set_xticklabels(splits, fontsize=9.5)
ax1.set_ylim(0, 1.08)
ax1.set_ylabel("R-squared", fontsize=10.5, color=INK)
ax1.set_title("R-squared", fontsize=13, fontweight="bold", color=INK)
style_axes(ax1)
ax1.legend(loc="lower left", fontsize=9, frameon=False)

bars2a = ax2.bar(x - width / 2, mlp_within10, width, color=COLORS["blue"], label="MLP", zorder=3,
                  edgecolor=INK, linewidth=0.5)
bars2b = ax2.bar(x + width / 2, tf_within10, width, color=COLORS["orange"], label="Transformer", zorder=3,
                  edgecolor=INK, linewidth=0.5, hatch="///")
for b, v in zip(list(bars2a) + list(bars2b), mlp_within10 + tf_within10):
    ax2.text(b.get_x() + b.get_width() / 2, v + 1, f"{v:.1f}%", ha="center", fontsize=8.5, color=INK)
ax2.set_xticks(x)
ax2.set_xticklabels(splits, fontsize=9.5)
ax2.set_ylim(0, 62)
ax2.set_ylabel("Predictions within +/-10% of measured CHF", fontsize=10, color=INK)
ax2.set_title("Within +/-10% accuracy", fontsize=13, fontweight="bold", color=INK)
style_axes(ax2)
ax2.legend(loc="upper right", fontsize=9, frameon=False)

fig.suptitle("MLP vs. Transformer: interpolation vs. extrapolation (Table 10)",
              fontsize=16.5, fontweight="bold", color=INK, y=1.02)
fig.text(0.5, -0.02,
          "Source: this paper's Table 10, NRC-derived interpolation and extrapolation splits. The "
          "Transformer's R-squared edge over the MLP (about 0.005-0.006) does not carry over to "
          "within-+/-10% accuracy, where the MLP matches or beats it under extrapolation "
          "(25.8% vs. 20.2%), for a fraction of the training cost (Table 10).",
          ha="center", fontsize=7.5, color=MUTED, style="italic")
fig.tight_layout()
add_outer_border(fig)
fig.savefig("docs/manuscript/fig_mlp_transformer_within10.png", facecolor=SURFACE, bbox_inches="tight", dpi=300)
plt.close(fig)
print("saved fig_mlp_transformer_within10.png")

"""
Accuracy vs. training cost, built directly from this paper's own Table 9
(R-squared, MAPE, and training time under Strategy 1 / random split, on the
seven-source merged dataset). No new numbers, just a different view of an
already-published table: is the most accurate model also the cheapest to
train, or is there a trade-off to argue about?
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, "scripts/paper_figures")
from style_common import set_pub_style, add_outer_border, MARKERS
set_pub_style()

INK = "#0b0b0b"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"
SURFACE = "#fcfcfb"

COLORS = {
    "blue": "#2a78d6", "orange": "#eb6834", "aqua": "#1baf7a",
    "violet": "#4a3aa7", "red": "#e34948",
}


def style_axes(ax):
    ax.set_facecolor(SURFACE)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    for spine in ["left", "bottom"]:
        ax.spines[spine].set_color(BASELINE)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.grid(True, color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)


# Table 9: Model, R2, MAPE (%), training time (s)
models = [
    ("Gradient\nBoosting", 0.968, 9.22, 1.9, COLORS["blue"]),
    ("Random\nForest", 0.965, 9.36, 8.8, COLORS["aqua"]),
    ("ANN", 0.921, 13.92, 4.2, COLORS["orange"]),
    ("GPR", 0.872, 17.43, 0.8, COLORS["violet"]),
    ("PINN", 0.876, 19.25, 11.5, COLORS["red"]),
]

fig, ax = plt.subplots(figsize=(8.8, 6.2), dpi=200)

for i, (name, r2, mape, t, color) in enumerate(models):
    size = 300 + mape * 30
    ax.scatter(t, r2, s=size, color=color, alpha=0.85, edgecolor=INK, linewidth=0.8, zorder=3,
               marker=MARKERS[i])
    ax.annotate(f"{name.replace(chr(10), ' ')}\nMAPE {mape:.1f}%", (t, r2),
                textcoords="offset points", xytext=(0, -34 if name != "GPR" else 20),
                ha="center", fontsize=8.5, color=INK)

ax.set_xscale("log")
ax.set_xlabel("Training time, seconds (log scale)", fontsize=10.5, color=INK)
ax.set_ylabel("R-squared (Strategy 1, random split)", fontsize=10.5, color=INK)
ax.set_title("Accuracy vs. training cost: the best model is also nearly the cheapest",
              fontsize=15, fontweight="bold", color=INK, pad=14)
ax.set_ylim(0.84, 1.0)
style_axes(ax)

fig.text(0.5, -0.01,
          "Source: this paper's Table 9. Marker size scales with MAPE (larger = worse percentage "
          "error). Gaussian Process Regression trains fastest but is among the least accurate; "
          "gradient boosting is both the most accurate model and nearly the fastest to train.",
          ha="center", fontsize=7.5, color=MUTED, style="italic")
fig.tight_layout(rect=[0, 0.05, 1, 1])
add_outer_border(fig)
fig.savefig("docs/manuscript/fig_accuracy_vs_cost.png", facecolor=SURFACE, bbox_inches="tight", dpi=300)
plt.close(fig)
print("saved fig_accuracy_vs_cost.png")

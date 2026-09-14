"""
Two figures built directly from this paper's own already-verified real result
tables (Table 2 and Table 4 in the paper text): a model-comparison chart across
the four validation strategies, and an R-squared / within-10% comparison for
the proposed pipeline on entirely unseen datasets. No numbers here are
invented; they are the same figures already cited as text and in Word tables
elsewhere in the paper.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np

plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman", "DejaVu Serif"]

INK = "#0b0b0b"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"
SURFACE = "#fcfcfb"

COLORS = {
    "blue": "#2a78d6", "orange": "#eb6834", "aqua": "#1baf7a",
    "yellow": "#eda100", "magenta": "#e87ba4", "green": "#008300",
    "violet": "#4a3aa7", "red": "#e34948",
}


def style_axes(ax):
    ax.set_facecolor(SURFACE)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    for spine in ["left", "bottom"]:
        ax.spines[spine].set_color(BASELINE)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)


# ---------- Figure 2: Table 2 model comparison across four strategies ----------
models = ["Gradient\nBoosting", "Random\nForest", "ANN", "GPR", "PINN"]
strategies = ["Strategy 1\n(Random)", "Strategy 2\n(Condition-wise)",
              "Strategy 3\n(Surface-wise)", "Strategy 4\n(Leave-one-\nsource-out)"]
r2_data = {
    "Gradient\nBoosting": [0.968, 0.890, 0.173, 0.784],
    "Random\nForest":     [0.965, 0.875, 0.219, 0.269],
    "ANN":                [0.921, 0.515, -4132.9, -5.150],
    "GPR":                [0.872, -0.051, -0.813, 0.539],
    "PINN":               [0.876, -64.9, -3906.9, -6.022],
}
model_colors = [COLORS["blue"], COLORS["aqua"], COLORS["orange"],
                COLORS["violet"], COLORS["red"]]

fig, ax = plt.subplots(figsize=(9.5, 5.2), dpi=200)
x = np.arange(len(strategies))
n_models = len(models)
width = 0.15
for i, m in enumerate(models):
    offset = (i - (n_models - 1) / 2) * width
    vals = r2_data[m]
    bars = ax.bar(x + offset, vals, width=width, color=model_colors[i],
                   label=m.replace("\n", " "), zorder=3, edgecolor=SURFACE, linewidth=0.5)
    for b, v in zip(bars, vals):
        label = f"{v:.2f}" if abs(v) < 10 else f"{v:.0f}"
        y = b.get_height()
        va = "bottom" if y >= 0 else "top"
        ax.text(b.get_x() + b.get_width() / 2, y, label, ha="center", va=va,
                fontsize=6.5, color=INK, rotation=90 if abs(v) > 10 else 0)

ax.set_yscale("symlog", linthresh=1)
ax.axhline(0, color=BASELINE, linewidth=1.0, zorder=2)
ax.set_xticks(x)
ax.set_xticklabels(strategies, fontsize=9)
ax.set_ylabel("R-squared (symmetric log scale)", fontsize=10, color=INK)
ax.set_title("Model accuracy collapses under harder validation strategies",
             fontsize=18, fontweight="bold", color=INK, pad=14)
style_axes(ax)
ax.legend(loc="lower left", fontsize=8, frameon=False, ncol=5,
          bbox_to_anchor=(0.0, -0.32))
fig.text(0.5, 0.01,
         "Source: this project's own model bake-off on the seven-source, 28,470-row merged dataset (Table 2). "
         "Y-axis uses a symmetric log scale because scores range from 0.97 to -4133.",
         ha="center", fontsize=7.5, color=MUTED, style="italic")
fig.tight_layout(rect=[0, 0.08, 1, 1])
fig.savefig("docs/manuscript/fig_model_comparison_strategies.png", facecolor=SURFACE)
plt.close(fig)
print("saved fig_model_comparison_strategies.png")


# ---------- Figure 3: Table 4, entirely-unseen-dataset performance ----------
datasets = ["KAERI\nnon-uniform", "NRC", "Hardik\nstraight", "KAERI\nuniform",
            "Hardik\nhelical", "Helical\nR-123", "Pioro\nR-134a", "Zhao\n2020"]
physics_r2 = [0.797, 0.926, 0.636, 0.620, -0.161, 0.051, -0.097, -0.437]
ml_r2 = [0.841, 0.242, -868.6, 0.852, -1.214, -307.7, -423.6, -1.433]
pipeline_r2 = [0.970, 0.944, 0.793, 0.735, 0.687, 0.449, -0.065, -0.429]
within10 = [47.5, 47.0, 33.8, 58.9, 29.7, 14.8, 16.0, 7.7]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 5.0), dpi=200)

x = np.arange(len(datasets))
width = 0.26
ax1.bar(x - width, physics_r2, width=width, color=COLORS["violet"], label="Physics alone", zorder=3)
ax1.bar(x, ml_r2, width=width, color=COLORS["red"], label="ML, no physics", zorder=3)
ax1.bar(x + width, pipeline_r2, width=width, color=COLORS["blue"], label="Proposed hybrid", zorder=3)
ax1.set_yscale("symlog", linthresh=1)
ax1.axhline(0, color=BASELINE, linewidth=1.0, zorder=2)
ax1.set_xticks(x)
ax1.set_xticklabels(datasets, fontsize=7.5)
ax1.set_ylabel("R-squared (symmetric log scale)", fontsize=10, color=INK)
ax1.set_title("R-squared on entirely unseen datasets", fontsize=11.5, fontweight="bold", color=INK)
style_axes(ax1)
ax1.legend(loc="lower left", fontsize=7.5, frameon=False)

bars2 = ax2.bar(x, within10, width=0.55, color=COLORS["aqua"], zorder=3)
for b, v in zip(bars2, within10):
    ax2.text(b.get_x() + b.get_width() / 2, v + 1, f"{v:.1f}%", ha="center",
              fontsize=8, color=INK)
ax2.set_xticks(x)
ax2.set_xticklabels(datasets, fontsize=7.5)
ax2.set_ylim(0, 68)
ax2.set_ylabel("Predictions within +/-10% of measured CHF (%)", fontsize=10, color=INK)
ax2.set_title("Proposed hybrid: within +/-10% accuracy", fontsize=11.5, fontweight="bold", color=INK)
style_axes(ax2)

fig.suptitle("Physics alone vs. data-driven ML vs. the proposed hybrid, on data none of them trained on",
             fontsize=18, fontweight="bold", color=INK, y=1.03)
fig.text(0.5, -0.03,
         "Source: this paper's Table 4, flow-boiling-only dataset (eight sources, 28,582 rows). "
         "The unanchored ML model reaches R2 = -868.6 on one dataset; the hybrid's worst case is -0.429.",
         ha="center", fontsize=7.5, color=MUTED, style="italic")
fig.tight_layout()
fig.savefig("docs/manuscript/fig_unseen_dataset_performance.png", facecolor=SURFACE, bbox_inches="tight")
plt.close(fig)
print("saved fig_unseen_dataset_performance.png")

"""
"Which model or approach actually wins, scenario by scenario" -- built purely
by taking the column-wise maximum of numbers already published in this
paper's own Table 2 (model comparison across four validation strategies) and
Table 4 (physics alone / ML alone / proposed hybrid on eight entirely unseen
datasets). No new experiments, no new numbers: this figure only re-reads and
re-presents values that are already sitting in Word tables elsewhere in the
paper, so that the "who wins where" pattern is visible at a glance instead
of requiring a reader to scan two tables by eye.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
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
    ax.grid(axis="x", color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)


# ---- Panel A: winning model per validation strategy (from Table 2) ----
model_color = {
    "Gradient\nBoosting": COLORS["blue"],
    "Random\nForest": COLORS["aqua"],
}
strategy_winners = [
    ("Strategy 1: Random", "Gradient\nBoosting", 0.968),
    ("Strategy 2: Condition-wise", "Gradient\nBoosting", 0.890),
    ("Strategy 3: Surface-wise", "Random\nForest", 0.219),
    ("Strategy 4: Leave-one-\nsource-out", "Gradient\nBoosting", 0.784),
]

# ---- Panel B: winning approach per unseen dataset (from Table 4) ----
approach_color = {
    "Physics alone": COLORS["violet"],
    "ML, no physics": COLORS["red"],
    "Proposed hybrid": COLORS["blue"],
}
dataset_winners = [
    ("KAERI non-uniform\n[non-uniform axial heating]", "Proposed hybrid", 0.970),
    ("NRC\n[uniform axial heating]", "Proposed hybrid", 0.944),
    ("Hardik straight tube", "Proposed hybrid", 0.793),
    ("KAERI uniform\n[uniform axial heating]", "ML, no physics", 0.852),
    ("Hardik helical coil", "Proposed hybrid", 0.687),
    ("Helical R-123", "Proposed hybrid", 0.449),
    ("Pioro R-134a", "Proposed hybrid", -0.065),
    ("Zhao 2020", "Proposed hybrid", -0.429),
]

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10.5, 8.8), dpi=200,
                                gridspec_kw={"height_ratios": [1, 1.7]})

# Panel A
labels_a = [s[0] for s in strategy_winners]
vals_a = [s[2] for s in strategy_winners]
colors_a = [model_color[s[1]] for s in strategy_winners]
y1 = np.arange(len(labels_a))
bars1 = ax1.barh(y1, vals_a, color=colors_a, zorder=3, edgecolor=SURFACE, linewidth=0.5)
for b, (label, model, v) in zip(bars1, strategy_winners):
    ax1.text(v + 0.015, b.get_y() + b.get_height() / 2,
              f"{model.replace(chr(10), ' ')}  (R2 = {v:.3f})", va="center", fontsize=9, color=INK)
ax1.set_yticks(y1)
ax1.set_yticklabels(labels_a, fontsize=9.5)
ax1.set_xlim(0, 1.32)
ax1.set_xlabel("R-squared of the winning model", fontsize=10, color=INK)
ax1.set_title("A. Best model per validation strategy (Table 2)", fontsize=12.5,
               fontweight="bold", color=INK, loc="left")
style_axes(ax1)
ax1.invert_yaxis()

# Panel B
labels_b = [d[0] for d in dataset_winners]
vals_b = [d[2] for d in dataset_winners]
colors_b = [approach_color[d[1]] for d in dataset_winners]
y2 = np.arange(len(labels_b))
bars2 = ax2.barh(y2, vals_b, color=colors_b, zorder=3, edgecolor=SURFACE, linewidth=0.5)
ax2.axvline(0, color=BASELINE, linewidth=1.0, zorder=2)
for b, (label, approach, v) in zip(bars2, dataset_winners):
    x_text = v + 0.03 if v >= 0 else v - 0.03
    ha = "left" if v >= 0 else "right"
    ax2.text(x_text, b.get_y() + b.get_height() / 2,
              f"{approach}  (R2 = {v:.3f})", va="center", ha=ha, fontsize=9, color=INK)
ax2.set_yticks(y2)
ax2.set_yticklabels(labels_b, fontsize=9.5)
ax2.set_xlim(-1.05, 1.55)
ax2.set_xlabel("R-squared of the winning approach", fontsize=10, color=INK)
ax2.set_title("B. Best approach per entirely unseen dataset (Table 4)", fontsize=12.5,
               fontweight="bold", color=INK, loc="left")
style_axes(ax2)
ax2.invert_yaxis()

handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in approach_color.values()]
ax2.legend(handles, approach_color.keys(), loc="lower right", fontsize=8.5, frameon=False)

fig.suptitle("Which model or approach actually wins, scenario by scenario",
              fontsize=17, fontweight="bold", color=INK, y=0.995)
fig.text(0.5, 0.005,
          "Source: this paper's own Table 2 and Table 4; each bar is simply the column-wise best of numbers "
          "already reported there. Random Forest, not Gradient Boosting, wins Strategy 3 specifically; "
          "ML with no physics, not the hybrid, wins on KAERI uniform specifically.",
          ha="center", fontsize=7.5, color=MUTED, style="italic")
fig.tight_layout(rect=[0, 0.02, 1, 0.97])
fig.savefig("docs/manuscript/fig_best_by_scenario.png", facecolor=SURFACE, bbox_inches="tight")
plt.close(fig)
print("saved fig_best_by_scenario.png")

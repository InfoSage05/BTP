"""
Database composition by boiling regime and heating configuration, built
directly from this project's own dataset manifest
(CHF_datasets_share/share/MANIFEST.csv). Row counts are read straight from
the manifest, not invented; the only "computation" here is grouping the
already-listed per-source row counts into four categories: flow boiling with
uniform axial heating, flow boiling with non-uniform axial heating, flow
boiling with an unspecified/other heating regime, and pool boiling /
engineered surfaces.
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


# Row counts taken directly from CHF_datasets_share/share/MANIFEST.csv (one row per source file).
sources = [
    ("NRC Groeneveld\n(24,579 rows)", 24579, "Flow boiling,\nuniform axial heating"),
    ("KAERI uniform\n(651 rows)", 651, "Flow boiling,\nuniform axial heating"),
    ("KAERI non-uniform\n(888 rows)", 888, "Flow boiling,\nnon-uniform axial heating"),
    ("Zhao 2020\n(1,865 rows)", 1865, "Flow boiling,\nheating regime not specified"),
    ("Helical R-123\n(257 rows)", 257, "Flow boiling,\nheating regime not specified"),
    ("Hardik helical,\nwater (156 rows)", 156, "Flow boiling,\nheating regime not specified"),
    ("Hardik straight,\nR-123 (55 rows)", 55, "Flow boiling,\nheating regime not specified"),
    ("Pioro R-134a\nholdout (268 rows)", 268, "Flow boiling,\nheating regime not specified"),
    ("Pin-fin surfaces\n(175 rows)", 175, "Pool boiling /\nengineered surfaces"),
    ("Surface master\n(100 rows)", 100, "Pool boiling /\nengineered surfaces"),
    ("Strip heater\n(55 rows)", 55, "Pool boiling /\nengineered surfaces"),
]

category_order = [
    "Flow boiling,\nuniform axial heating",
    "Flow boiling,\nnon-uniform axial heating",
    "Flow boiling,\nheating regime not specified",
    "Pool boiling /\nengineered surfaces",
]
category_colors = {
    "Flow boiling,\nuniform axial heating": COLORS["blue"],
    "Flow boiling,\nnon-uniform axial heating": COLORS["orange"],
    "Flow boiling,\nheating regime not specified": COLORS["aqua"],
    "Pool boiling /\nengineered surfaces": COLORS["magenta"],
}

totals = {c: 0 for c in category_order}
for _, n, cat in sources:
    totals[cat] += n

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 5.6), dpi=200,
                                gridspec_kw={"width_ratios": [1.55, 1]})

# ---- Panel 1: per-source row counts, colored by category ----
sources_sorted = sorted(sources, key=lambda t: t[1])
labels = [s[0] for s in sources_sorted]
counts = [s[1] for s in sources_sorted]
colors = [category_colors[s[2]] for s in sources_sorted]
y = np.arange(len(labels))
bars = ax1.barh(y, counts, color=colors, zorder=3, edgecolor=SURFACE, linewidth=0.5)
for b, v in zip(bars, counts):
    ax1.text(v * 1.03, b.get_y() + b.get_height() / 2, f"{v:,}", va="center",
              fontsize=8, color=INK)
ax1.set_xscale("log")
ax1.set_yticks(y)
ax1.set_yticklabels(labels, fontsize=8.5)
ax1.set_xlabel("Rows (log scale)", fontsize=10, color=INK)
ax1.set_title("Rows per source", fontsize=12.5, fontweight="bold", color=INK)
style_axes(ax1)

# ---- Panel 2: totals by category ----
cat_labels = category_order
cat_counts = [totals[c] for c in cat_labels]
y2 = np.arange(len(cat_labels))
bars2 = ax2.barh(y2, cat_counts, color=[category_colors[c] for c in cat_labels],
                  zorder=3, edgecolor=SURFACE, linewidth=0.5)
for b, v in zip(bars2, cat_counts):
    ax2.text(v * 1.03, b.get_y() + b.get_height() / 2, f"{v:,}", va="center",
              fontsize=9, color=INK, fontweight="bold")
ax2.set_xscale("log")
ax2.set_yticks(y2)
ax2.set_yticklabels(cat_labels, fontsize=9)
ax2.set_xlabel("Total rows (log scale)", fontsize=10, color=INK)
ax2.set_title("Rows by category", fontsize=12.5, fontweight="bold", color=INK)
style_axes(ax2)

fig.suptitle("Database composition: boiling regime and heating configuration",
              fontsize=17, fontweight="bold", color=INK, y=1.02)
fig.text(0.5, -0.02,
          "Source: CHF_datasets_share/share/MANIFEST.csv, one row per source file. "
          "Includes the Pioro R-134a set, held out and never used for training.",
          ha="center", fontsize=7.5, color=MUTED, style="italic")
fig.tight_layout()
fig.savefig("docs/manuscript/fig_dataset_composition.png", facecolor=SURFACE, bbox_inches="tight")
plt.close(fig)
print("saved fig_dataset_composition.png")

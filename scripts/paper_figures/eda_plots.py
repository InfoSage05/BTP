"""
Exploratory data analysis figures built directly from the real merged CHF
dataset (CHF_datasets_share/share/01_merged/master_chf_dataset.csv, 28,470
rows, seven sources). No synthetic or invented values; every bar and every
histogram bin comes from the actual dataset used throughout this paper.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys
sys.path.insert(0, "scripts/paper_figures")
from style_common import set_pub_style, add_outer_border, MARKERS
set_pub_style()

INK = "#0b0b0b"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"
SURFACE = "#fcfcfb"
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]


def style_axes(ax):
    ax.set_facecolor(SURFACE)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    for spine in ["left", "bottom"]:
        ax.spines[spine].set_color(BASELINE)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.grid(axis="y", color=GRID, linewidth=0.7, zorder=0)
    ax.set_axisbelow(True)


df = pd.read_csv("CHF_datasets_share/share/01_merged/master_chf_dataset.csv", low_memory=False)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5.0), dpi=200)

# Panel 1: log10(CHF) distribution
log_chf = np.log10(df["CHF_kW_m2"].dropna())
ax1.hist(log_chf, bins=50, color=COLORS[0], edgecolor=SURFACE, linewidth=0.3, zorder=3)
ax1.set_xlabel("log10(CHF), CHF in kW/m2", fontsize=10, color=INK)
ax1.set_ylabel("Number of rows", fontsize=10, color=INK)
ax1.set_title("CHF spans about three orders of magnitude", fontsize=12, fontweight="bold", color=INK)
style_axes(ax1)
ax1.text(0.03, 0.95,
         f"n = {len(log_chf)}\nmin = {df['CHF_kW_m2'].min():.0f} kW/m2\n"
         f"median = {df['CHF_kW_m2'].median():.0f} kW/m2\nmax = {df['CHF_kW_m2'].max():.0f} kW/m2",
         transform=ax1.transAxes, ha="left", va="top", fontsize=8.5, color=INK,
         bbox=dict(boxstyle="round,pad=0.4", facecolor=SURFACE, edgecolor=BASELINE, linewidth=0.8))

# Panel 2: row counts per geometry family
counts = df["geometry_family"].value_counts()
labels = [l.replace("_", " ") for l in counts.index]
bars = ax2.bar(range(len(counts)), counts.values, color=COLORS[:len(counts)], zorder=3)
ax2.set_yscale("log")
ax2.set_xticks(range(len(counts)))
ax2.set_xticklabels(labels, rotation=30, ha="right", fontsize=8.5)
ax2.set_ylabel("Number of rows (log scale)", fontsize=10, color=INK)
ax2.set_title("Rows per geometry family (highly imbalanced)", fontsize=12, fontweight="bold", color=INK)
style_axes(ax2)
for b, v in zip(bars, counts.values):
    ax2.text(b.get_x() + b.get_width() / 2, v * 1.15, f"{v}", ha="center", fontsize=8, color=INK)

fig.suptitle("Exploratory data analysis: the merged CHF dataset (28,470 rows, seven sources)",
             fontsize=18, fontweight="bold", color=INK, y=1.03)
fig.text(0.5, -0.04,
         "Source: CHF_datasets_share/share/01_merged/master_chf_dataset.csv, as documented in "
         "01_merged/merge_report.md.",
         ha="center", fontsize=7.5, color=MUTED, style="italic")
fig.tight_layout()
add_outer_border(fig)
fig.savefig("docs/manuscript/fig_eda_distribution_geometry.png", facecolor=SURFACE, bbox_inches="tight", dpi=300)
plt.close(fig)
print("saved fig_eda_distribution_geometry.png")

# ---------- Second EDA figure: pressure vs CHF, colored by geometry family ----------
fig2, ax = plt.subplots(figsize=(8.5, 6.0), dpi=200)
top_families = counts.index[:3].tolist()  # keep only top 3 for scatter (all-pairs color rule)
for i, fam in enumerate(top_families):
    sub = df[df["geometry_family"] == fam]
    ax.scatter(sub["pressure_kPa"], sub["CHF_kW_m2"], s=12, alpha=0.35,
               color=COLORS[i], label=fam.replace("_", " "), edgecolors="none",
               marker=MARKERS[i])
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("Pressure (kPa, log scale)", fontsize=10, color=INK)
ax.set_ylabel("CHF (kW/m2, log scale)", fontsize=10, color=INK)
ax.set_title("CHF vs. pressure, top three geometry families",
             fontsize=18, fontweight="bold", color=INK)
style_axes(ax)
ax.legend(loc="upper left", fontsize=9, frameon=False)
fig2.text(0.5, -0.02,
          "Source: master_chf_dataset.csv. Only the three largest geometry families are shown, "
          "per this project's own accessible-color guidance for scatter plots with more than three series.",
          ha="center", fontsize=7.5, color=MUTED, style="italic")
fig2.tight_layout()
add_outer_border(fig2)
fig2.savefig("docs/manuscript/fig_eda_pressure_vs_chf.png", facecolor=SURFACE, bbox_inches="tight", dpi=300)
plt.close(fig2)
print("saved fig_eda_pressure_vs_chf.png")

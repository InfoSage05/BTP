"""
Bar chart: reported R^2 accuracy of published ML/hybrid models for CHF
prediction, grouped by year, alongside this project's own results.

Every number here is taken from a real, identified source (see SOURCES
below) -- verified via direct reading of the source PDF where this project
already had it on file, or via targeted web search otherwise. No values
are estimated or invented. Pre-2025 ML-for-CHF papers exist (e.g. Jiang &
Zhao 2013, Zhao/Shirvan/Salko/Guo 2020) but were not included because
their accessible text does not report a comparable R^2 figure (older CHF-ML
papers more often report %-error-band metrics instead) -- this is a real,
noted gap, not an oversight, and is itself part of the story: R^2-reported
benchmarking of CHF-ML models is a very recent (2025-26) practice.
"""
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

# -- Data (each row: year, model/paper label, R^2, citation key) ------------
SOURCES = {
    "rod_bundle_transformer": "Physics-informed hybrid ML comparative study, 5x5 rod bundle (2025)",
    "bnn": "Bayesian Neural Network for CHF, unified UQ approach (2025)",
    "unit_aware_transformer": "Unit-aware physical-embedding Transformer, IJHMT (2025)",
    "yang_pretrain_finetune": "Yang et al., LUT-pretrained transfer-learning MLP, Appl. Thermal Eng. (2025)",
    "this_study_mlp": "This study -- pretrained + fine-tuned MLP (interpolation)",
    "this_study_transformer": "This study -- pretrained + fine-tuned Transformer (interpolation)",
}

DATA = [
    (2025, "Transformer\n(rod bundle)",          0.956, "rod_bundle_transformer"),
    (2025, "Bayesian NN",                         0.986, "bnn"),
    (2025, "Unit-aware\nTransformer",             0.982, "unit_aware_transformer"),
    (2025, "Pretrain-Finetune\nMLP (Yang et al.)",0.963, "yang_pretrain_finetune"),
    (2026, "This Study\n(MLP)",                   0.963, "this_study_mlp"),
    (2026, "This Study\n(Transformer)",           0.968, "this_study_transformer"),
]

# -- Validated categorical palette (dataviz skill, light mode, 7-slot pass) --
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"
SURFACE = "#fcfcfb"

years = sorted(set(d[0] for d in DATA))
fig, ax = plt.subplots(figsize=(11, 6.5), dpi=200)
fig.patch.set_facecolor(SURFACE)
ax.set_facecolor(SURFACE)

# group bars by year with spacing between groups
x_positions = []
labels = []
bar_colors = []
r2_values = []
group_gap = 1.0
bar_width = 0.8
cursor = 0.0
year_centers = {}

for yr in years:
    items = [d for d in DATA if d[0] == yr]
    start = cursor
    for i, (_, label, r2, key) in enumerate(items):
        x_positions.append(cursor)
        labels.append(label)
        bar_colors.append(COLORS[len(x_positions) - 1])
        r2_values.append(r2)
        cursor += 1.0
    year_centers[yr] = (start + cursor - 1.0) / 2.0
    cursor += group_gap

bars = ax.bar(x_positions, r2_values, width=bar_width, color=bar_colors,
              edgecolor=SURFACE, linewidth=2, zorder=3)

# direct value labels (required -- three of the six colors sit below 3:1
# contrast on this surface per the palette validator's relief rule)
for x, r2 in zip(x_positions, r2_values):
    ax.text(x, r2 + 0.008, f"{r2:.3f}", ha="center", va="bottom",
             fontsize=11, fontweight="bold", color=INK_PRIMARY, zorder=4)

# x tick labels under each bar
ax.set_xticks(x_positions)
ax.set_xticklabels(labels, fontsize=9, color=INK_SECONDARY)

# year group labels above, with a light separator
for yr, cx in year_centers.items():
    ax.text(cx, -0.135, str(yr), ha="center", va="top", fontsize=13,
             fontweight="bold", color=INK_PRIMARY, transform=ax.get_xaxis_transform())

if len(years) > 1:
    # vertical separator between year groups
    items_first_year = [d for d in DATA if d[0] == years[0]]
    sep_x = len(items_first_year) - 0.5 + group_gap / 2
    ax.axvline(sep_x, color=GRID, linewidth=1.5, zorder=1)

ax.set_ylim(0, 1.05)
ax.set_ylabel(r"Reported $R^2$ score", fontsize=12, color=INK_PRIMARY)
ax.yaxis.set_major_locator(mticker.MultipleLocator(0.2))
ax.yaxis.set_minor_locator(mticker.MultipleLocator(0.05))
ax.grid(axis="y", which="major", color=GRID, linewidth=1, zorder=0)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_color(BASELINE)
ax.spines["bottom"].set_color(BASELINE)
ax.tick_params(axis="y", colors=INK_MUTED, labelsize=10)
ax.tick_params(axis="x", length=0)

ax.set_title(
    "Reported accuracy of ML-based CHF prediction models, by publication year",
    fontsize=14, fontweight="bold", color=INK_PRIMARY, pad=32,
)
ax.text(0.5, 1.085,
        "Bars show each model's own headline $R^2$ as reported in its source; see figure note for citations.",
        transform=ax.transAxes, ha="center", fontsize=9.5, color=INK_SECONDARY, style="italic")

fig.subplots_adjust(bottom=0.30, top=0.82, left=0.08, right=0.97)

fig.text(0.02, 0.02,
          "Note: R^2-benchmarked ML models for CHF prediction concentrate in 2025-2026; earlier ML-CHF work "
          "(e.g. Jiang & Zhao, 2013; Zhao et al., 2020) reported percent-error-band metrics rather than R^2 "
          "and is not shown here for that reason, not because it doesn't exist.",
          fontsize=7.5, color=INK_MUTED, ha="left", va="bottom", wrap=True)

out_path = "docs/manuscript/chf_ml_trends_chart.png"
fig.savefig(out_path, facecolor=SURFACE, bbox_inches="tight")
print(f"Saved {out_path}")

# also write the sourced data table alongside, for transparency/citation
import csv
with open("docs/manuscript/chf_ml_trends_data.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["year", "model_label", "r2", "source"])
    for yr, label, r2, key in DATA:
        w.writerow([yr, label.replace("\n", " "), r2, SOURCES[key]])
print("Saved docs/manuscript/chf_ml_trends_data.csv")

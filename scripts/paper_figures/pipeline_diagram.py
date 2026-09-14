"""
Schematic diagram of the proposed model's five-layer hierarchy (Table 6 in
the paper text). This is an architecture illustration, not a data chart, but
follows the same legibility rules: one hue per layer in the project's fixed
categorical order, text labels rather than color-only distinctions, and a
plain, uncluttered layout.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman", "DejaVu Serif"]

INK = "#0b0b0b"
SURFACE = "#fcfcfb"
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]

layers = [
    ("Layer 0", "Fluid properties &\ndimensionless groups",
     "Fluid name + pressure -> CoolProp properties ->\n"
     "inverse Weber number, density ratio, L/D,\nsubcooling ratio, reduced pressure"),
    ("Layer 1", "Physics backbone\n(zero fitted parameters)",
     "Katto-Ohno / Biasi / 2006 look-up table,\nauto-selected by lowest training error"),
    ("Layer 2", "Pooled learned correction",
     "Single tree model predicts\nlog(measured CHF / backbone prediction)"),
    ("Layer 3", "Trust-scaled correction limit",
     "Full trust: up to 10x correction\nLow trust: clipped to 1.5x, decays to backbone"),
    ("Layer 4", "Conformal interval + regime label",
     "Calibrated uncertainty interval;\nlabel: interpolating / extrapolating / fallback"),
]

fig, ax = plt.subplots(figsize=(8.0, 11.0), dpi=200)
ax.set_xlim(0, 10)
ax.set_ylim(0, len(layers) * 2.2 + 1)
ax.axis("off")
fig.patch.set_facecolor(SURFACE)
ax.set_facecolor(SURFACE)

box_h = 1.7
gap = 0.5
y = len(layers) * (box_h + gap)

for i, (tag, title, detail) in enumerate(layers):
    y_top = y
    box = FancyBboxPatch((0.5, y_top - box_h), 9.0, box_h,
                          boxstyle="round,pad=0.08,rounding_size=0.12",
                          linewidth=1.2, edgecolor=INK, facecolor=COLORS[i], alpha=0.85, zorder=3)
    ax.add_patch(box)
    ax.text(1.0, y_top - 0.35, f"{tag}: {title}", fontsize=13, fontweight="bold",
            color="white", va="top", ha="left", zorder=4)
    ax.text(1.0, y_top - 0.95, detail, fontsize=9.5, color="white", va="top", ha="left", zorder=4)

    if i < len(layers) - 1:
        arrow = FancyArrowPatch((5.0, y_top - box_h - 0.02), (5.0, y_top - box_h - gap + 0.05),
                                 arrowstyle="-|>", mutation_scale=18, color=INK, linewidth=1.4, zorder=5)
        ax.add_patch(arrow)
    y = y_top - box_h - gap

ax.text(5.0, len(layers) * (box_h + gap) + 0.55, "Raw inputs\n(pressure, mass flux, quality, diameter, fluid)",
        fontsize=10.5, color=INK, ha="center", va="bottom", style="italic")
arrow_in = FancyArrowPatch((5.0, len(layers) * (box_h + gap) + 0.5),
                            (5.0, len(layers) * (box_h + gap) - 0.02),
                            arrowstyle="-|>", mutation_scale=18, color=INK, linewidth=1.4, zorder=5)
ax.add_patch(arrow_in)

last_box_bottom = y + gap
arrow_out = FancyArrowPatch((5.0, last_box_bottom), (5.0, last_box_bottom - 0.5),
                            arrowstyle="-|>", mutation_scale=18, color=INK, linewidth=1.4, zorder=5)
ax.add_patch(arrow_out)
ax.text(5.0, last_box_bottom - 0.6, "Final prediction with uncertainty interval",
        fontsize=11, fontweight="bold", color=INK, ha="center", va="top")
ax.set_ylim(last_box_bottom - 1.2, ax.get_ylim()[1])

fig.suptitle("Proposed model hierarchy", fontsize=18, fontweight="bold", color=INK, y=0.99)
fig.tight_layout()
fig.savefig("docs/manuscript/fig_pipeline_hierarchy.png", facecolor=SURFACE, bbox_inches="tight")
plt.close(fig)
print("saved fig_pipeline_hierarchy.png")

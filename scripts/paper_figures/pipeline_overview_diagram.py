"""
High-level, end-to-end pipeline diagram: raw data sources through to a final
prediction. This complements the existing five-layer hierarchy diagram
(pipeline_diagram.py / fig_pipeline_hierarchy.png), which zooms into the
model itself (Table 6 in the paper text). This figure instead shows the
bigger picture described across Sections 2, 3.1, 3.3, and 3.4: where the
data comes from, how it is prepared, what the model hierarchy does with it,
how it is validated, and what comes out the other end. No new claims are
made here; every stage names a section already written elsewhere in the
paper.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman", "DejaVu Serif"]

INK = "#0b0b0b"
SURFACE = "#fcfcfb"
COLORS = ["#4a3aa7", "#2a78d6", "#1baf7a", "#eda100", "#eb6834"]

stages = [
    ("Stage 1", "Raw CHF data\nsources\n(Section 2)",
     "11 source files:\nflow boiling (uniform\nand non-uniform axial\nheating) and pool\nboiling / engineered\nsurfaces; six CHF\ndetection criteria"),
    ("Stage 2", "Merge and\npreprocessing\n(Section 3.1)",
     "51-column merged\nschema; missing-value\npolicy, audited\noutliers, log(CHF)\ntarget, dimensionless\nphysical groups"),
    ("Stage 3", "Physics + learned\nhierarchy\n(Section 3.3,\nFigure 11)",
     "Layer 0 fluid groups\n-> Layer 1 physics\nbackbone -> Layer 2\npooled correction ->\nLayer 3 trust limit ->\nLayer 4 conformal\ninterval"),
    ("Stage 4", "Validation\nstrategies I-IV\n(Section 3.4)",
     "Random,\ncondition-wise,\nsurface-wise, and\nleave-one-source-out,\napplied identically to\nevery model compared"),
    ("Stage 5", "Prediction\noutput",
     "CHF estimate,\ncalibrated uncertainty\ninterval, and an\ninterpolating /\nextrapolating / fallback\nregime label"),
]

fig, ax = plt.subplots(figsize=(12.5, 4.6), dpi=200)
ax.set_xlim(0, len(stages) * 2.55 + 0.3)
ax.set_ylim(0, 3.9)
ax.axis("off")
fig.patch.set_facecolor(SURFACE)
ax.set_facecolor(SURFACE)

box_w = 2.25
box_h = 3.3
gap = 0.3
x = 0.4

for i, (tag, title, detail) in enumerate(stages):
    box = FancyBboxPatch((x, 0.9), box_w, box_h,
                          boxstyle="round,pad=0.06,rounding_size=0.1",
                          linewidth=1.2, edgecolor=INK, facecolor=COLORS[i],
                          alpha=0.85, zorder=3)
    ax.add_patch(box)
    ax.text(x + box_w / 2, 0.9 + box_h - 0.3, tag, fontsize=11, fontweight="bold",
            color="white", va="top", ha="center", zorder=4)
    ax.text(x + box_w / 2, 0.9 + box_h - 0.7, title, fontsize=9.5, fontweight="bold",
            color="white", va="top", ha="center", zorder=4)
    ax.text(x + box_w / 2, 0.9 + box_h - 1.9, detail, fontsize=8.0,
            color="white", va="top", ha="center", zorder=4, linespacing=1.5)

    if i < len(stages) - 1:
        arrow = FancyArrowPatch((x + box_w + 0.02, 0.9 + box_h / 2),
                                 (x + box_w + gap - 0.02, 0.9 + box_h / 2),
                                 arrowstyle="-|>", mutation_scale=16, color=INK,
                                 linewidth=1.4, zorder=5)
        ax.add_patch(arrow)
    x += box_w + gap

fig.suptitle("End-to-end pipeline: from raw CHF measurements to a calibrated prediction",
             fontsize=16.5, fontweight="bold", color=INK, y=0.985)
fig.tight_layout(rect=[0, 0, 1, 0.94])
fig.savefig("docs/manuscript/fig_pipeline_overview.png", facecolor=SURFACE, bbox_inches="tight")
plt.close(fig)
print("saved fig_pipeline_overview.png")

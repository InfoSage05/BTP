"""
Figure 7: pairwise correlation matrix for the NRC/Groeneveld sub-dataset
(24,579 rows). Rebuilt directly from the already-computed
results/eda/nrc_groeneveld_chf_database/correlation_matrix.csv (same
correlation table used in the original figure) -- no new computation,
just a restyled, non-truncated render with every cell's real value
annotated, so the figure stays readable once color is stripped for
black-and-white printing.
"""
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, "scripts/paper_figures")
from style_common import set_pub_style, add_outer_border, INK, MUTED, SURFACE
set_pub_style()

df = pd.read_csv("results/eda/nrc_groeneveld_chf_database/correlation_matrix.csv", index_col=0)
labels = list(df.columns)
vals = df.to_numpy()

fig, ax = plt.subplots(figsize=(8.5, 7.4), dpi=300)
im = ax.imshow(vals, cmap="RdBu_r", vmin=-1, vmax=1)

ax.set_xticks(range(len(labels)))
ax.set_xticklabels(labels, rotation=90, fontsize=10, color=INK)
ax.set_yticks(range(len(labels)))
ax.set_yticklabels(labels, fontsize=10, color=INK)

for i in range(len(labels)):
    for j in range(len(labels)):
        v = vals[i, j]
        # text color flips for readability once the diverging colormap is dark
        txt_color = "white" if abs(v) > 0.6 else INK
        ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8.5, color=txt_color)

cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cbar.ax.tick_params(labelsize=9, colors=MUTED)

fig.suptitle("NRC/Groeneveld 24,579-pt CHF Database: Correlation Matrix",
             fontsize=15, fontweight="bold", color=INK, y=0.995)
fig.text(0.5, 0.01,
         "Source: results/eda/nrc_groeneveld_chf_database/correlation_matrix.csv. "
         "Every cell value is annotated directly so the sign and magnitude of each "
         "correlation remain legible once printed in black and white.",
         ha="center", fontsize=7.5, color=MUTED, style="italic")
fig.tight_layout(rect=[0, 0.03, 1, 0.96])
add_outer_border(fig)
fig.savefig("docs/manuscript/fig_correlation_heatmap_nrc.png",
            facecolor=SURFACE, bbox_inches="tight", dpi=300)
plt.close(fig)
print("saved fig_correlation_heatmap_nrc.png")

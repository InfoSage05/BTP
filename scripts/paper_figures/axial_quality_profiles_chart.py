"""
Figure 3: axial local-quality profile structure for the KAERI TR-1665
non-uniform heating dataset. Rebuilt directly from the real
kaeri_tr1665_nonuniform.xml source (same file and same per-test profile
reconstruction as scripts/build_eda_notebooks.py's build_kaeri_nonuniform
cell) so every point plotted is a real measured value, not synthetic.
"""
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, "scripts/paper_figures")
from style_common import set_pub_style, add_outer_border, style_axes, PALETTE, MARKERS, LINESTYLES, INK, MUTED, SURFACE
set_pub_style()

XML_PATH = Path("data/raw/fine_tuning/kaeri_tr1665_nonuniform.xml")

tree = ET.parse(XML_PATH)
profiles = []
for ds in tree.getroot().findall("dataset"):
    test_id = ds.find("TestID").text
    qualities = [float(q.text) for q in ds.findall("Quality")]
    positions = [float(p.text) for p in ds.findall("QualityPosition")]
    profiles.append({"TestID": test_id, "n_points": len(qualities),
                      "qualities": qualities, "positions": positions})
profiles_df = pd.DataFrame(profiles)

fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), dpi=300)

axes[0].hist(profiles_df["n_points"], bins=range(1, int(profiles_df["n_points"].max()) + 2),
             color=PALETTE[0], edgecolor=SURFACE, linewidth=0.3, zorder=3)
axes[0].set_title("Axial profile length (# points per test)", fontsize=12, fontweight="bold", color=INK)
axes[0].set_xlabel("n_points", fontsize=10, color=INK)
axes[0].set_ylabel("Number of tests", fontsize=10, color=INK)
style_axes(axes[0])

rng = np.random.default_rng(0)
sample_idx = rng.choice(len(profiles_df), size=min(8, len(profiles_df)), replace=False)
for j, i in enumerate(sample_idx):
    row = profiles_df.iloc[i]
    axes[1].plot(row["positions"], row["qualities"],
                 marker=MARKERS[j % len(MARKERS)], markersize=4,
                 linestyle=LINESTYLES[j % len(LINESTYLES)], linewidth=1.3,
                 color=PALETTE[j % len(PALETTE)], alpha=0.9,
                 label=f"Test {row['TestID']}")
axes[1].set_xlabel("Axial position", fontsize=10, color=INK)
axes[1].set_ylabel("Local quality", fontsize=10, color=INK)
axes[1].set_title("Sample axial quality profiles", fontsize=12, fontweight="bold", color=INK)
axes[1].legend(fontsize=7, ncol=2, frameon=False)
style_axes(axes[1])

fig.suptitle("KAERI TR-1665 Non-Uniform Heating: Axial Quality Profile Structure",
             fontsize=15.5, fontweight="bold", color=INK, y=1.03)
fig.text(0.5, -0.02,
         "Source: data/raw/fine_tuning/kaeri_tr1665_nonuniform.xml, per-test Quality/QualityPosition "
         "sequences; 8 of the dataset's tests shown as a representative sample.",
         ha="center", fontsize=7.5, color=MUTED, style="italic")
fig.tight_layout()
add_outer_border(fig)
fig.savefig("docs/manuscript/fig_axial_quality_profiles_nonuniform.png",
            facecolor=SURFACE, bbox_inches="tight", dpi=300)
plt.close(fig)
print("saved fig_axial_quality_profiles_nonuniform.png")

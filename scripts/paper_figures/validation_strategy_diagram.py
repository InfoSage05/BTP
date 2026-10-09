"""
Schematic diagram of the four validation strategies (Table 8 in the paper
text): what "train" and "test" actually mean under each one. This is an
explainer diagram, not a data chart -- it encodes no new numeric results,
only the already-written construction rules from Table 8, drawn so a reader
can see the difference between the four strategies at a glance rather than
parsing four rows of a text table.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch
import sys
sys.path.insert(0, "scripts/paper_figures")
from style_common import set_pub_style, add_outer_border
set_pub_style()

TEST_HATCH = "///"

INK = "#0b0b0b"
MUTED = "#898781"
SURFACE = "#fcfcfb"
TRAIN = "#c7d9f2"
TEST = "#2a78d6"
SOURCE_COLORS = ["#4a3aa7", "#1baf7a", "#eda100", "#e87ba4", "#eb6834", "#008300"]

n_sources = 6

fig, axes = plt.subplots(4, 1, figsize=(9.5, 8.2), dpi=200)
fig.patch.set_facecolor(SURFACE)

strategies = [
    ("Strategy I: Random split",
     "Rows from every source are shuffled together; train/test assignment "
     "ignores which source a row came from."),
    ("Strategy II: Condition-wise split",
     "Within each source, the highest-pressure rows are held out as test; "
     "the rest of that same source trains the model."),
    ("Strategy III: Surface-wise split",
     "One or more entire sources (surface types / geometries) are held out "
     "as test in full; the model never sees them during training."),
    ("Strategy IV: Leave-one-source-out",
     "Each source takes a turn as the sole test source while all others "
     "train; results are pooled across every fold."),
]

for ax, (title, desc) in zip(axes, strategies):
    ax.set_xlim(0, n_sources)
    ax.set_ylim(0, 1.6)
    ax.axis("off")
    ax.set_facecolor(SURFACE)
    ax.text(0, 1.55, title, fontsize=12.5, fontweight="bold", color=INK, va="top", ha="left")
    ax.text(0, 1.18, desc, fontsize=8.8, color=MUTED, va="top", ha="left")

def block(color, hatch=""):
    return dict(facecolor=color, edgecolor=INK, linewidth=0.6, hatch=hatch)


for s in range(n_sources):
    axes[0].add_patch(Rectangle((s, 0.15), 0.85, 0.6, facecolor=SOURCE_COLORS[s], alpha=0.35,
                                 edgecolor=INK, linewidth=0.6, zorder=2))
    for k in range(5):
        is_test = (s + k) % 3 == 0
        axes[0].add_patch(Rectangle((s + 0.02 + k * 0.16, 0.2), 0.14, 0.5, zorder=3,
                                     **block(TEST if is_test else TRAIN, TEST_HATCH if is_test else "")))

for s in range(n_sources):
    axes[1].add_patch(Rectangle((s, 0.2), 0.6, 0.5, zorder=3, **block(TRAIN)))
    axes[1].add_patch(Rectangle((s + 0.6, 0.2), 0.25, 0.5, zorder=3, **block(TEST, TEST_HATCH)))
    axes[1].text(s + 0.42, 0.75, f"S{s+1}", fontsize=8, color=MUTED, ha="center")

held_out = {4, 5}
for s in range(n_sources):
    is_test = s in held_out
    axes[2].add_patch(Rectangle((s, 0.2), 0.85, 0.5, zorder=3,
                                 **block(TEST if is_test else TRAIN, TEST_HATCH if is_test else "")))
    axes[2].text(s + 0.42, 0.75, f"S{s+1}", fontsize=8, color=MUTED, ha="center")

fold_y = [0.78, 0.42, 0.06]
fold_labels = ["Fold 1", "Fold 2", "...Fold 6"]
for fi, (fy, flabel) in enumerate(zip(fold_y, fold_labels)):
    axes[3].text(-0.05, fy + 0.15, flabel, fontsize=8, color=MUTED, ha="left", va="center")
    for s in range(n_sources):
        is_test = (s == fi) if fi < 2 else (s == n_sources - 1)
        axes[3].add_patch(Rectangle((s + 0.9, fy), 0.75, 0.28, zorder=3,
                                     **block(TEST if is_test else TRAIN, TEST_HATCH if is_test else "")))

handles = [Rectangle((0, 0), 1, 1, **block(TRAIN)), Rectangle((0, 0), 1, 1, **block(TEST, TEST_HATCH))]
fig.legend(handles, ["Train", "Test"], loc="lower center", ncol=2, frameon=False,
           fontsize=10, bbox_to_anchor=(0.5, -0.01))

fig.suptitle("The four validation strategies: what counts as train vs. test",
             fontsize=16.5, fontweight="bold", color=INK, y=0.995)
fig.tight_layout(rect=[0, 0.03, 1, 0.965])
add_outer_border(fig)
fig.savefig("docs/manuscript/fig_validation_strategies.png", facecolor=SURFACE, bbox_inches="tight", dpi=300)
plt.close(fig)
print("saved fig_validation_strategies.png")

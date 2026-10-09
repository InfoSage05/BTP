"""
Shared publication style for every figure in CHF_Research_Paper.docx.

Applies the formatting feedback from the September 17, 2026 team meeting:
  - consistent serif font family and sizing across all figures
  - a visible outer border on every saved figure, for print clarity
  - a B&W-safe qualitative palette (distinct even once color is stripped),
    paired with hatch patterns (bars) / marker shapes + linestyles
    (scatter and line plots) so series stay distinguishable in grayscale
  - PNG output at a print-safe 300 dpi
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

INK = "#0b0b0b"
MUTED = "#5a5952"
GRID = "#d8d7cf"
BASELINE = "#8c8b82"
SURFACE = "#fcfcfb"
BORDER = "#000000"

# Qualitative palette chosen so relative luminance (grayscale value) is
# spread out, not clustered -- each color remains distinguishable from its
# neighbors once desaturated.
PALETTE = [
    "#1f5fa8",  # dark blue    (luminance ~0.30)
    "#d9731f",  # burnt orange (luminance ~0.47)
    "#1b8a5a",  # deep green   (luminance ~0.45) -- paired with hatch/marker, not relied on alone
    "#7a2f8f",  # purple       (luminance ~0.22)
    "#c23b3b",  # brick red    (luminance ~0.35)
    "#b89a1f",  # dark gold    (luminance ~0.58)
]

# Matching hatch patterns / marker shapes / linestyles, same order as PALETTE,
# so each series carries a second, color-independent channel of identity.
HATCHES = ["", "///", "...", "xxx", "\\\\\\", "ooo"]
MARKERS = ["o", "s", "^", "D", "v", "P"]
LINESTYLES = ["-", "--", "-.", ":", (0, (3, 1, 1, 1)), (0, (5, 1))]


def set_pub_style():
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = ["Times New Roman", "DejaVu Serif"]
    plt.rcParams["axes.unicode_minus"] = True
    plt.rcParams["figure.facecolor"] = SURFACE
    plt.rcParams["savefig.facecolor"] = SURFACE


def style_axes(ax, grid=True):
    ax.set_facecolor(SURFACE)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color(BASELINE)
        spine.set_linewidth(0.8)
    ax.tick_params(colors=MUTED, labelsize=9)
    if grid:
        ax.grid(color=GRID, linewidth=0.7, zorder=0)
    ax.set_axisbelow(True)


def add_outer_border(fig, linewidth=1.6, pad=0.004):
    """Draws a border around the full figure canvas, for print clarity."""
    rect = plt.Rectangle(
        (pad, pad), 1 - 2 * pad, 1 - 2 * pad,
        transform=fig.transFigure, fill=False,
        edgecolor=BORDER, linewidth=linewidth, zorder=1000,
    )
    fig.patches.append(rect)


def savefig(fig, path, **kwargs):
    add_outer_border(fig)
    kwargs.setdefault("dpi", 300)
    kwargs.setdefault("facecolor", SURFACE)
    kwargs.setdefault("bbox_inches", "tight")
    fig.savefig(path, **kwargs)

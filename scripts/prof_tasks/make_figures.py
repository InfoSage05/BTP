"""Figures for the flow-boiling comparison.

Colour follows the job each measure does:

  R2          has a meaningful zero (0 = no better than predicting the mean), so
              it gets a DIVERGING blue<->red ramp with a neutral grey midpoint at
              zero. Blue = better than the mean, red = worse.
  within+/-10%  is a pure magnitude on [0, 1], so it gets a SEQUENTIAL single-hue
              blue ramp, light -> dark.

No rainbow anywhere. Grid and axes are recessive; all text sits in ink tokens
rather than in a series colour.
"""

from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402
import numpy as np                        # noqa: E402
import pandas as pd                       # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm  # noqa: E402

OUT = os.path.join(os.path.dirname(__file__), "..", "..", "results", "prof_tasks")
FIG = os.path.join(OUT, "figures")
os.makedirs(FIG, exist_ok=True)

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
GRID = "#e3e2df"

BLUE = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
SEQ = LinearSegmentedColormap.from_list("seq_blue", BLUE)
DIV = LinearSegmentedColormap.from_list(
    "div_bluered", ["#8c1d1d", "#d03b3b", "#e79a9a", "#f0efec",
                    "#9ec5f4", "#3987e5", "#184f95"])


def _style(ax):
    ax.set_facecolor(SURFACE)
    for s in ax.spines.values():
        s.set_color(GRID)
        s.set_linewidth(0.8)
    ax.tick_params(colors=INK2, labelsize=8, length=0)


def heatmap(piv, title, subtitle, fname, kind="div", fmt="{:.2f}"):
    if piv.empty:
        return
    fig, ax = plt.subplots(figsize=(1.05 * len(piv.columns) + 4.2,
                                    0.42 * len(piv.index) + 2.4))
    fig.patch.set_facecolor(SURFACE)
    v = piv.to_numpy(float)
    if kind == "div":
        vis = np.clip(v, -1.0, 1.0)          # R2 can be -1000s; clip for colour only
        norm = TwoSlopeNorm(vmin=-1.0, vcenter=0.0, vmax=1.0)
        im = ax.imshow(vis, cmap=DIV, norm=norm, aspect="auto")
        cbar_label = "R²  (clipped to [-1, 1] for colour; 0 = predicting the mean)"
    else:
        im = ax.imshow(v, cmap=SEQ, vmin=0, vmax=1, aspect="auto")
        cbar_label = "fraction of predictions within ±10%"

    ax.set_xticks(range(len(piv.columns)))
    ax.set_xticklabels(piv.columns, rotation=40, ha="right")
    ax.set_yticks(range(len(piv.index)))
    ax.set_yticklabels(piv.index)
    _style(ax)
    ax.set_xticks(np.arange(-.5, len(piv.columns), 1), minor=True)
    ax.set_yticks(np.arange(-.5, len(piv.index), 1), minor=True)
    ax.grid(which="minor", color=SURFACE, linewidth=2)   # 2px surface gap
    ax.tick_params(which="minor", length=0)

    # a heatmap is a table: label every cell so identity is never colour-alone
    for i in range(v.shape[0]):
        for j in range(v.shape[1]):
            val = v[i, j]
            if not np.isfinite(val):
                continue
            shade = im.norm(np.clip(val, -1, 1)) if kind == "div" else val
            txt = fmt.format(val) if abs(val) < 100 else f"{val:.0f}"
            ax.text(j, i, txt, ha="center", va="center", fontsize=7,
                    color="#ffffff" if (shade < 0.22 or shade > 0.78) else INK)

    ax.set_title(title, fontsize=12, color=INK, pad=16, loc="left", weight="bold")
    ax.text(0, 1.012, subtitle, transform=ax.transAxes, fontsize=8.5, color=INK2)
    cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.015)
    cb.set_label(cbar_label, fontsize=8, color=INK2)
    cb.ax.tick_params(colors=INK2, labelsize=7, length=0)
    cb.outline.set_edgecolor(GRID)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, fname), dpi=190, facecolor=SURFACE)
    plt.close(fig)
    print("wrote", fname)


def main():
    g = pd.read_csv(os.path.join(OUT, "grid_results.csv"))

    for form in ("reduced", "local"):
        for split, tag in (("80/20", "80_20"), ("LOSO", "loso")):
            d = g[(g.form == form) & (g.split == split)]
            if d.empty:
                continue
            for val, kind, fn, lab in (
                    ("R2_mean", "div", f"heatmap_R2_{form}_{tag}.png", "R²"),
                    ("within_10pct_mean", "seq",
                     f"heatmap_within10_{form}_{tag}.png", "within ±10%")):
                piv = d.pivot_table(index="dataset", columns="model", values=val)
                piv = piv.reindex(sorted(piv.index))
                heatmap(piv, f"{lab} by model and dataset — {form} form, {split} split",
                        "flow boiling only; models trained on log(CHF), scored on raw kW/m²",
                        fn, kind=kind, fmt="{:.2f}" if kind == "div" else "{:.2f}")

    # ---- formulation gap: best-model R2 per dataset, per formulation
    d = g[g.split == "80/20"]
    best = (d.sort_values("R2_mean", ascending=False)
              .groupby(["dataset", "form"], as_index=False).head(1))
    piv = best.pivot_table(index="dataset", columns="form", values="R2_mean")
    order = [c for c in ("local", "inlet", "reduced") if c in piv.columns]
    piv = piv[order].reindex(sorted(piv.index))
    if not piv.empty:
        fig, ax = plt.subplots(figsize=(9.5, 4.6))
        fig.patch.set_facecolor(SURFACE)
        x = np.arange(len(piv.index))
        w = 0.8 / len(order)
        hues = {"local": "#9ec5f4", "inlet": "#3987e5", "reduced": "#184f95"}
        for k, form in enumerate(order):
            ax.bar(x + k * w - 0.4 + w / 2, piv[form].clip(lower=-0.2), w * 0.92,
                   label=form, color=hues[form], edgecolor=SURFACE, linewidth=2)
        ax.axhline(0, color=INK2, linewidth=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(piv.index, rotation=25, ha="right")
        ax.set_ylabel("best-model R²  (80/20)", fontsize=9, color=INK2)
        ax.set_title("The formulation gap: what outlet quality is worth",
                     fontsize=12, color=INK, pad=16, loc="left", weight="bold")
        ax.text(0, 1.02, "local includes outlet quality X and is circular; "
                         "reduced drops it; inlet substitutes inlet conditions",
                transform=ax.transAxes, fontsize=8.5, color=INK2)
        ax.grid(axis="y", color=GRID, linewidth=0.7)
        ax.set_axisbelow(True)
        _style(ax)
        leg = ax.legend(frameon=False, fontsize=8.5, ncol=3, loc="lower right")
        for t in leg.get_texts():
            t.set_color(INK2)
        fig.tight_layout()
        fig.savefig(os.path.join(FIG, "formulation_gap.png"), dpi=190,
                    facecolor=SURFACE)
        plt.close(fig)
        print("wrote formulation_gap.png")


if __name__ == "__main__":
    main()

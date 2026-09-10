"""The five figures specified in the manuscript outline (literature survey, sec. 3).

  Fig 2  Global model performance and generalisation gap
  Fig 3  Local error topography over (G, x) slices
  Fig 4  Multi-seed prediction spread and stochastic stability
  Fig 5  Extrapolation domain and operational reliability boundary

Figure 1 in the outline is a literature-coverage map built from the reference
library rather than from model output, so it is produced separately.

Colour follows the job each measure does. R2 has a meaningful zero -- zero means
"no better than predicting the mean" -- so it gets a diverging blue/red ramp
centred there. Fractions and errors are pure magnitudes and get a sequential
single-hue ramp. No rainbow anywhere.
"""

from __future__ import annotations

import os
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402
import numpy as np                        # noqa: E402
import pandas as pd                       # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm  # noqa: E402

import physics_arms as P                  # noqa: E402
import pipeline as PL                     # noqa: E402
import run_pipeline as RP                 # noqa: E402

warnings.filterwarnings("ignore")
FIG = os.path.join(P.OUT, "figures_outline")
os.makedirs(FIG, exist_ok=True)

SURF, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e3e2df"
BLUE = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
SEQ = LinearSegmentedColormap.from_list("seq", BLUE)
DIV = LinearSegmentedColormap.from_list(
    "div", ["#8c1d1d", "#d03b3b", "#e79a9a", "#f0efec", "#9ec5f4", "#3987e5", "#184f95"])
SHORT = {"D1_NRC": "NRC", "D2_Zhao2020": "Zhao", "D3_KAERI_uniform": "KAERI unif",
         "D4_KAERI_nonuniform": "KAERI non-unif", "D5_Helical_R123": "Helical R123",
         "D6_Hardik_helical_water": "Hardik helical",
         "D7_Hardik_straight_R123": "Hardik straight", "D8_Pioro_R134a": "Pioro R134a"}


def _ax(ax):
    ax.set_facecolor(SURF)
    for s in ax.spines.values():
        s.set_color(GRID)
        s.set_linewidth(0.8)
    ax.tick_params(colors=INK2, labelsize=8.5, length=0)


def _save(fig, name):
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, name), dpi=190, facecolor=SURF)
    plt.close(fig)
    print("wrote", name)


def fig2(res):
    """Generalisation gap: same-dataset accuracy against unseen-dataset accuracy."""
    a = res[res.split == "80/20"].set_index("dataset")
    b = res[res.split.str.startswith("LODO")].set_index("dataset")
    ds = [d for d in SHORT if d in a.index and d in b.index]
    x = np.arange(len(ds))
    fig, axes = plt.subplots(1, 2, figsize=(12.6, 4.6))
    fig.patch.set_facecolor(SURF)
    for ax, col, lab, hi in ((axes[0], "R2_mean", "R²", 1.0),
                             (axes[1], "within_10pct_mean", "fraction within ±10%", 1.0)):
        v1 = [a.loc[d, col] for d in ds]
        v2 = [b.loc[d, col] for d in ds]
        ax.bar(x - 0.21, np.clip(v1, -0.25, hi), 0.4, label="same dataset (80/20)",
               color="#9ec5f4", edgecolor=SURF, linewidth=1.6)
        ax.bar(x + 0.21, np.clip(v2, -0.25, hi), 0.4, label="unseen dataset",
               color="#184f95", edgecolor=SURF, linewidth=1.6)
        for i, (p, q) in enumerate(zip(v1, v2)):
            if q < -0.25:
                ax.text(i + 0.21, -0.23, f"{q:.2f}", ha="center", va="bottom",
                        fontsize=7, color="#a8342f")
        ax.axhline(0, color=INK2, linewidth=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels([SHORT[d] for d in ds], rotation=32, ha="right")
        ax.set_ylabel(lab, fontsize=9, color=INK2)
        ax.grid(axis="y", color=GRID, linewidth=0.7)
        ax.set_axisbelow(True)
        _ax(ax)
        leg = ax.legend(frameon=False, fontsize=8.5, loc="lower left")
        for t in leg.get_texts():
            t.set_color(INK2)
    fig.suptitle("Figure 2   Generalisation gap: familiar data against an unseen dataset",
                 fontsize=12.5, color=INK, x=0.01, ha="left", weight="bold")
    _save(fig, "fig2_generalisation_gap.png")


def fig3(df, pipe):
    """Local error topography of the physics backbone over (G, x)."""
    d = df[df.dataset == "D1_NRC"].copy()
    raw = RP.D.LOADERS["D1_NRC"]().df
    d = d.reset_index(drop=True)
    d["G"] = raw["G_kg_m2s"].to_numpy()[: len(d)]
    d["x"] = raw["X"].to_numpy()[: len(d)]
    err = np.abs(d.phys_katto.to_numpy() - d.CHF_kW_m2.to_numpy()) / d.CHF_kW_m2.to_numpy()
    gb = np.linspace(0, 5000, 26)
    xb = np.linspace(-0.5, 1.0, 26)
    gi = np.clip(np.digitize(d.G, gb) - 1, 0, 24)
    xi = np.clip(np.digitize(d.x, xb) - 1, 0, 24)
    grid = np.full((25, 25), np.nan)
    cnt = np.zeros((25, 25))
    for a, b, e in zip(xi, gi, err):
        cnt[a, b] += 1
        grid[a, b] = e if np.isnan(grid[a, b]) else grid[a, b] + e
    grid = np.where(cnt > 0, grid / np.maximum(cnt, 1), np.nan)
    fig, ax = plt.subplots(figsize=(8.4, 5.4))
    fig.patch.set_facecolor(SURF)
    im = ax.imshow(np.clip(grid, 0, 1), origin="lower", aspect="auto", cmap=SEQ,
                   extent=[gb[0], gb[-1], xb[0], xb[-1]], vmin=0, vmax=1)
    ax.set_xlabel("mass flux G  (kg m⁻² s⁻¹)", fontsize=9, color=INK2)
    ax.set_ylabel("thermodynamic quality x", fontsize=9, color=INK2)
    _ax(ax)
    cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cb.set_label("mean relative error of the physics backbone", fontsize=8.5, color=INK2)
    cb.ax.tick_params(colors=INK2, labelsize=7.5, length=0)
    cb.outline.set_edgecolor(GRID)
    ax.set_title("Figure 3   Where the correlation fails, over the (G, x) plane — NRC",
                 fontsize=12, color=INK, loc="left", pad=14, weight="bold")
    ax.text(0, 1.015, "blank cells carry no measurements", transform=ax.transAxes,
            fontsize=8.5, color=INK2)
    _save(fig, "fig3_error_topography.png")


def fig4(res):
    """Stochastic stability: seed-to-seed spread of every reported figure."""
    d = res[res.split.isin(["80/20", "70/30"]) |
            res.split.str.startswith("LODO")].copy()
    d["lab"] = d.dataset.map(SHORT) + "  " + d.split.str.replace(" (unseen dataset)", "", regex=False)
    d = d.sort_values("R2_sd", ascending=True)
    fig, ax = plt.subplots(figsize=(8.6, 8.2))
    fig.patch.set_facecolor(SURF)
    y = np.arange(len(d))
    ax.barh(y, d.R2_sd, color="#3987e5", edgecolor=SURF, linewidth=1.4, height=0.72)
    ax.set_yticks(y)
    ax.set_yticklabels(d.lab, fontsize=7.6)
    ax.set_xlabel("standard deviation of R² across seeds", fontsize=9, color=INK2)
    ax.set_xscale("symlog", linthresh=0.001)
    ax.grid(axis="x", color=GRID, linewidth=0.7)
    ax.set_axisbelow(True)
    _ax(ax)
    ax.set_title("Figure 4   Stochastic stability — seed spread of every reported result",
                 fontsize=12, color=INK, loc="left", pad=14, weight="bold")
    ax.text(0, 1.008, "a result whose bar is long should never be quoted from one seed",
            transform=ax.transAxes, fontsize=8.5, color=INK2)
    _save(fig, "fig4_seed_stability.png")


def fig5(df, pipe):
    """Reliability boundary: how trust and error move together as data gets novel."""
    rows = []
    for held in RP.FLOW:
        tr, te = df[df.dataset != held], df[df.dataset == held]
        p = PL.CHFPipeline(seed=0).fit(tr)
        r = p.predict(te)
        y = te.CHF_kW_m2.to_numpy(float)
        rel = np.abs(r.chf.to_numpy() - y) / y
        rows.append(pd.DataFrame({"trust": r.trust.to_numpy(), "rel": rel,
                                  "dataset": held}))
    a = pd.concat(rows, ignore_index=True)
    bins = np.linspace(0, 1, 11)
    a["b"] = np.clip(np.digitize(a.trust, bins) - 1, 0, 9)
    g = a.groupby("b").agg(med=("rel", "median"), w10=("rel", lambda v: (v <= .1).mean()),
                           n=("rel", "size"))
    fig, ax = plt.subplots(figsize=(9.0, 5.0))
    fig.patch.set_facecolor(SURF)
    c = (bins[:-1] + bins[1:]) / 2
    ax.bar(c[g.index], g.med * 100, 0.085, color="#9ec5f4", edgecolor=SURF,
           linewidth=1.5, label="median relative error (%)")
    ax2 = ax.twiny()
    ax2.axis("off")
    ax.plot(c[g.index], g.w10 * 100, "o-", color="#0d366b", linewidth=2, markersize=7,
            label="predictions within ±10% (%)")
    for i, n in zip(g.index, g.n):
        ax.text(c[i], 2, f"n={n}", ha="center", fontsize=6.8, color=INK2, rotation=90)
    ax.set_xlabel("trust  (0 = correlation fallback, 1 = interpolating)", fontsize=9, color=INK2)
    ax.set_ylabel("per cent", fontsize=9, color=INK2)
    ax.grid(axis="y", color=GRID, linewidth=0.7)
    ax.set_axisbelow(True)
    _ax(ax)
    leg = ax.legend(frameon=False, fontsize=8.5, loc="upper center")
    for t in leg.get_texts():
        t.set_color(INK2)
    ax.set_title("Figure 5   Reliability boundary — does the trust signal track real error?",
                 fontsize=12, color=INK, loc="left", pad=14, weight="bold")
    ax.text(0, 1.012, "pooled over all eight unseen-dataset folds",
            transform=ax.transAxes, fontsize=8.5, color=INK2)
    _save(fig, "fig5_reliability_boundary.png")


def main():
    res = pd.read_csv(os.path.join(P.OUT, "FINAL_results.csv"))
    df = RP.build_corpus()
    pipe = PL.CHFPipeline().fit(df)
    fig1(df)
    fig2(res)
    fig3(df, pipe)
    fig4(res)
    fig5(df, pipe)
    print(f"\nfigures in {FIG}")




def fig1(df):
    """Parameter-space coverage: which dataset covers which operating region.

    The outline's Figure 1 is a *literature* methodology map -- which published
    study used which validation protocol. Building that would require asserting
    what other papers did from their abstracts, which is not verifiable here, so
    it is left for the manuscript. This is the verifiable counterpart: coverage of
    the operating envelope by the data actually in hand, which is what determines
    where any model can be trusted.
    """
    import datasets as D
    axes_spec = [("P_kPa", "pressure (kPa)", True), ("G_kg_m2s", "mass flux (kg m⁻² s⁻¹)", True),
                 ("D_mm", "diameter (mm)", True), ("L_mm", "heated length (mm)", True),
                 ("CHF_kW_m2", "CHF (kW m⁻²)", True)]
    names = [n for n in SHORT]
    fig, axs = plt.subplots(1, len(axes_spec), figsize=(15.5, 4.9), sharey=True)
    fig.patch.set_facecolor(SURF)
    for ax, (col, lab, logx) in zip(axs, axes_spec):
        for i, n in enumerate(names):
            d = D.LOADERS[n]().df
            if col not in d:
                ax.text(0.5, i, "not reported", transform=ax.get_yaxis_transform(),
                        ha="center", va="center", fontsize=7, color="#a8342f", style="italic")
                continue
            v = pd.to_numeric(d[col], errors="coerce").dropna()
            v = v[v > 0] if logx else v
            if v.empty:
                continue
            ax.plot([v.min(), v.max()], [i, i], linewidth=7, solid_capstyle="butt",
                    color="#3987e5", alpha=.85)
            ax.plot([v.median()], [i], "o", color="#0d366b", markersize=5, zorder=3)
        if logx:
            ax.set_xscale("log")
        ax.set_xlabel(lab, fontsize=8.5, color=INK2)
        ax.grid(axis="x", color=GRID, linewidth=0.7)
        ax.set_axisbelow(True)
        _ax(ax)
    axs[0].set_yticks(range(len(names)))
    axs[0].set_yticklabels([SHORT[n] for n in names], fontsize=8.5)
    axs[0].set_ylim(-0.7, len(names) - 0.3)
    fig.suptitle("Figure 1   Operating-envelope coverage of the eight flow-boiling datasets",
                 fontsize=12.5, color=INK, x=0.006, ha="left", weight="bold")
    fig.text(0.006, 0.925, "bar spans min to max, dot marks the median; log scale",
             fontsize=8.5, color=INK2)
    _save(fig, "fig1_parameter_coverage.png")


if __name__ == "__main__":
    main()

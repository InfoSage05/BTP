"""
New correlation parity plots -- predicted vs. measured CHF -- for Random
Forest, Gradient Boosting, and XGBoost, requested at the Sept 17, 2026
team meeting to sit alongside the existing MLP parity plot (Figure 15).
Built from real per-row test-set predictions produced by
generate_treemodel_predictions.py (same interpolation/extrapolation test
splits as the MLP figure, so all four models are directly comparable).
Nothing here is synthetic.
"""
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, "scripts/paper_figures")
from style_common import set_pub_style, add_outer_border, style_axes, PALETTE, INK, MUTED, SURFACE
set_pub_style()

OUT_DIR = "data/processed/stage1"

MODELS = [
    ("random_forest", "Random Forest", PALETTE[2], "docs/manuscript/fig_parity_plot_random_forest.png"),
    ("gradient_boosting", "Gradient Boosting", PALETTE[3], "docs/manuscript/fig_parity_plot_gradient_boosting.png"),
    ("xgboost", "XGBoost", PALETTE[4], "docs/manuscript/fig_parity_plot_xgboost.png"),
]


def metrics(df):
    true = df["true_CHF_kW_m2"].to_numpy()
    pred = df["pred_CHF_kW_m2"].to_numpy()
    rel_err = (pred - true) / true
    ss_res = np.sum((true - pred) ** 2)
    ss_tot = np.sum((true - true.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot
    mape = np.mean(np.abs(rel_err)) * 100
    within10 = np.mean(np.abs(rel_err) <= 0.10) * 100
    return r2, mape, within10


def parity_panel(ax, df, color, title, r2, mape, within10):
    true = df["true_CHF_kW_m2"].to_numpy()
    pred = df["pred_CHF_kW_m2"].to_numpy()
    lo = min(true.min(), pred.min()) * 0.8
    hi = max(true.max(), pred.max()) * 1.2

    ax.scatter(true, pred, s=10, color=color, alpha=0.35, edgecolors="none", zorder=3)
    diag = np.array([lo, hi])
    ax.plot(diag, diag, color=INK, linewidth=1.3, zorder=4, label="Perfect prediction")
    ax.plot(diag, diag * 1.10, color=MUTED, linewidth=1.0, linestyle="--", zorder=2)
    ax.plot(diag, diag * 0.90, color=MUTED, linewidth=1.0, linestyle="--", zorder=2,
            label="+/-10% band")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("Measured CHF (kW/m2)", fontsize=10, color=INK)
    ax.set_ylabel("Predicted CHF (kW/m2)", fontsize=10, color=INK)
    ax.set_title(title, fontsize=12, fontweight="bold", color=INK)
    style_axes(ax)
    ax.text(0.03, 0.97,
            f"n = {len(true)}\nR2 = {r2:.3f}\nMAPE = {mape:.1f}%\nwithin +/-10% = {within10:.1f}%",
            transform=ax.transAxes, ha="left", va="top", fontsize=9, color=INK,
            bbox=dict(boxstyle="round,pad=0.4", facecolor=SURFACE, edgecolor=MUTED, linewidth=0.8))
    ax.legend(loc="lower right", fontsize=8, frameon=False)


for key, label, color, out_path in MODELS:
    interp = pd.read_csv(f"{OUT_DIR}/{key}_predictions_interp.csv")
    extrap = pd.read_csv(f"{OUT_DIR}/{key}_predictions_extrap.csv")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5.2), dpi=300)
    r2_i, mape_i, w10_i = metrics(interp)
    r2_e, mape_e, w10_e = metrics(extrap)
    parity_panel(ax1, interp, color, "Interpolation test split", r2_i, mape_i, w10_i)
    parity_panel(ax2, extrap, color, "Extrapolation test split (pressure > 14 MPa)", r2_e, mape_e, w10_e)

    fig.suptitle(f"Predicted vs. measured CHF, {label} (real test-set predictions)",
                 fontsize=18, fontweight="bold", color=INK, y=1.02)
    fig.text(0.5, -0.02,
             f"Source: {key}_predictions_{{interp,extrap}}.csv (generate_treemodel_predictions.py), "
             "a fresh fit on the same real NRC train/test splits used for the MLP parity plot (Figure 15), "
             "so all four models are directly comparable.",
             ha="center", fontsize=7.5, color=MUTED, style="italic")
    fig.tight_layout()
    add_outer_border(fig)
    fig.savefig(out_path, facecolor=SURFACE, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"{label}: interp R2={r2_i:.4f} MAPE={mape_i:.2f}% | extrap R2={r2_e:.4f} MAPE={mape_e:.2f}%")
    print(f"saved {out_path}")

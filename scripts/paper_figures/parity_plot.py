"""
Real parity plots (predicted vs. measured CHF) for this project's own MLP
model, built from actual model predictions on the Stage-1 interpolation and
extrapolation test sets (see generate_mlp_predictions.py, which re-runs the
real pretrain-then-fine-tune pipeline and saves per-row true/predicted
values). Nothing here is synthetic or invented; every point is a real test-set
row and a real model prediction.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman", "DejaVu Serif"]

INK = "#0b0b0b"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"
SURFACE = "#fcfcfb"
BLUE = "#2a78d6"
ORANGE = "#eb6834"


def style_axes(ax):
    ax.set_facecolor(SURFACE)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    for spine in ["left", "bottom"]:
        ax.spines[spine].set_color(BASELINE)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.grid(color=GRID, linewidth=0.7, zorder=0)
    ax.set_axisbelow(True)


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
            bbox=dict(boxstyle="round,pad=0.4", facecolor=SURFACE, edgecolor=BASELINE, linewidth=0.8))
    ax.legend(loc="lower right", fontsize=8, frameon=False)


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


interp = pd.read_csv("data/processed/stage1/mlp_predictions_interp.csv")
extrap = pd.read_csv("data/processed/stage1/mlp_predictions_extrap.csv")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5.2), dpi=200)
r2_i, mape_i, w10_i = metrics(interp)
r2_e, mape_e, w10_e = metrics(extrap)
parity_panel(ax1, interp, BLUE, "Interpolation test split", r2_i, mape_i, w10_i)
parity_panel(ax2, extrap, ORANGE, "Extrapolation test split (pressure > 14 MPa)", r2_e, mape_e, w10_e)

fig.suptitle("Predicted vs. measured CHF, this project's own MLP (real test-set predictions)",
             fontsize=18, fontweight="bold", color=INK, y=1.02)
fig.text(0.5, -0.02,
         "Source: a fresh run of this project's own pretrain-then-fine-tune MLP pipeline "
         "(scripts/chf_pipeline/train_stage1.py architecture), evaluated on the real NRC test splits. "
         "Single-seed numbers vary somewhat from the multi-seed means reported elsewhere in this paper "
         "(Section 1.8's own point about needing multi-seed reporting).",
         ha="center", fontsize=7.5, color=MUTED, style="italic")
fig.tight_layout()
fig.savefig("docs/manuscript/fig_parity_plot_mlp.png", facecolor=SURFACE, bbox_inches="tight")
plt.close(fig)
print(f"interp: R2={r2_i:.4f} MAPE={mape_i:.2f}% within10={w10_i:.2f}%")
print(f"extrap: R2={r2_e:.4f} MAPE={mape_e:.2f}% within10={w10_e:.2f}%")
print("saved fig_parity_plot_mlp.png")

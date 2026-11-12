"""GIF 3 - Start from physics, let ML fix the error.

Real data: CHF_manuscript_revision_2026-11-05/supporting_data/audit_v2_features.csv,
feature set F0_base5 (three-seed leave-one-dataset-out comparison, Table 5 of the
manuscript). Arms: pooled_ML (direct ML), katto_alone (physics), PIPE (bounded
physics + ML residual correction). Bars morph ML -> physics -> physics+ML -> back.
"""
import sys
import numpy as np
import pandas as pd
from gifkit import *

CSV, OUT = sys.argv[1], sys.argv[2]
FPS = 12
LABELS = {
    "D1_NRC": "NRC tubes · water",
    "D2_Zhao2020": "Zhao 2020 · water",
    "D3_KAERI_uniform": "KAERI uniform heating",
    "D4_KAERI_nonuniform": "KAERI non-uniform",
    "D5_Helical_R123": "Helical coils · R-123",
    "D6_Hardik_helical_water": "Hardik & Prabhu coils",
    "D7_Hardik_straight_R123": "Hardik & Prabhu tubes",
    "D8_Pioro_R134a": "Pioro · R-134a",
}
STAGES = [
    ("pooled_ML", "ML alone", "learns CHF directly from the data"),
    ("katto_alone", "Physics alone", "classic Katto–Ohno correlation, no learning"),
    ("PIPE", "Physics + ML", "ML learns only the correlation's error"),
]
LO = -1.0  # axis floor; worse values are clipped and labelled

d = pd.read_csv(CSV)
d = d[d.set == "F0_base5"]
names = list(LABELS)
vals = [np.array([float(d[(d.held_out == n) & (d.arm == a)].R2.iloc[0]) for n in names])
        for a, _, _ in STAGES]
meds = [float(np.median(v)) for v in vals]
wins = int(np.sum(vals[2] > vals[1]))
assert wins == 7 and abs(meds[0] + 2.061) < 0.01 and abs(meds[2] - 0.708) < 0.01, (wins, meds)
print("medians", [round(m, 3) for m in meds], "physics+ML beats physics on", wins, "of 8")

HOLD, MORPH = 26, 10
seq = []
for i in range(3):
    seq += [(i, i, 0.0)] * HOLD
    seq += [(i, (i + 1) % 3, (k + 1) / MORPH) for k in range(MORPH)]


def ease(t):
    return t * t * (3 - 2 * t)


def fmt(x):
    return f"{x:.2f}" if x > -10 else f"{x:.0f}"


def render(a, b, t):
    e = ease(t)
    ca, cb = np.clip(vals[a], LO, 1), np.clip(vals[b], LO, 1)
    v = ca + (cb - ca) * e
    m = meds[a] + (meds[b] - meds[a]) * e
    stage = b if e > 0.5 else a
    raw = vals[stage]
    fig = new_fig()
    header(fig, 3, "Start from physics, let ML fix the error")
    fig.text(0.06, 0.885, "Each dataset hidden in turn · how well do we predict it? (R²)",
             fontsize=15, color=MUTED, va="center")
    for i, (_, name, _) in enumerate(STAGES):
        x = 0.06 + i * 0.30
        active = i == stage
        fig.text(x + 0.135, 0.82, name, ha="center", va="center", fontsize=14,
                 weight="bold" if active else "medium", color="white" if active else MUTED,
                 bbox=dict(boxstyle="round,pad=0.45,rounding_size=0.8",
                           fc=TEAL if active else "#F1F3F5", ec="none"))
    fig.text(0.5, 0.765, STAGES[stage][2], ha="center", fontsize=13, color=MUTED, style="italic")
    col = RED if m < 0 else (INK if m < 0.5 else TEAL)
    fig.text(0.06, 0.685, f"{m:.2f}", fontsize=52, weight="bold", color=col, va="center")
    fig.text(0.33, 0.70, "median R²  (1 = perfect)", fontsize=15, color=INK, va="center")

    ax = fig.add_axes([0.40, 0.13, 0.54, 0.50])
    y = np.arange(len(names))[::-1]
    ax.axvline(0, color=MUTED, lw=1.2)
    ax.axvspan(LO, 1.0, color="#F7F8FA", zorder=0)
    cols = [TEAL if x >= 0.5 else (INK if x >= 0 else RED) for x in v]
    ax.barh(y, v, color=cols, height=0.62, zorder=2)
    for yi, vi, r in zip(y, v, raw):
        clipped = r < LO
        if vi >= 0:
            ax.text(vi + 0.04, yi, fmt(r) if t == 0 else "", va="center", fontsize=13,
                    weight="demibold", color=INK)
        elif t == 0 and clipped:
            ax.text(LO + 0.04, yi, "◀ " + fmt(r), va="center", fontsize=12.5,
                    weight="bold", color="white", zorder=3)
        elif t == 0:
            ax.text(0.04, yi, fmt(r), va="center", fontsize=13, weight="demibold", color=RED)
    for yi, n in zip(y, names):
        fig.text(0.385, 0.13 + 0.50 * (yi + 0.5) / len(names), LABELS[n], ha="right",
                 va="center", fontsize=13.5)
    ax.set_xlim(LO - 0.02, 1.25); ax.set_ylim(-0.5, len(names) - 0.5)
    ax.set_yticks([]); ax.set_xticks([-1, 0, 1])
    ax.tick_params(axis="x", labelsize=12, length=0)
    for sp in ["top", "right", "left"]:
        ax.spines[sp].set_visible(False)
    if stage == 2 and t == 0:
        fig.text(0.06, 0.075, f"Physics + ML beats physics alone on {wins} of 8 datasets",
                 fontsize=16, weight="bold", color=TEAL)
    elif stage == 0 and t == 0:
        fig.text(0.06, 0.075, f"ML alone: negative R² on {int(np.sum(vals[0] < 0))} of 8 unseen datasets",
                 fontsize=16, weight="bold", color=RED)
    fig.text(0.94, 0.03, "3 seeds · bars below −1 clipped (◀) · physics = Katto–Ohno",
             fontsize=11, color=MUTED, ha="right")
    return fig_to_image(fig)


frames = [render(*s) for s in seq]
save_gif(frames, OUT, FPS, holds={HOLD - 1: 600, MORPH + 2 * HOLD - 1: 600,
                                  2 * MORPH + 3 * HOLD - 1: 1400},
         start=2 * MORPH + 3 * HOLD - 1)
print("frames", len(frames))

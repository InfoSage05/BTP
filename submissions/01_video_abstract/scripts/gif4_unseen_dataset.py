"""GIF 3 - The harder the test, the lower the score.

Real data: CHF_manuscript_revision_2026-11-05/supporting_data/FINAL_results.csv
(within +-10% of measured CHF, mean over seeds, for the eight-dataset corpus).
Bars morph: random 80/20 -> held-out lab/group -> held-out whole dataset -> back.
"""
import sys
import numpy as np
import pandas as pd
from matplotlib.patches import FancyBboxPatch
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
    ("80/20", "Random 80/20 split", "test points mixed in with training points"),
    ("GROUPED", "Unseen lab or group", "a whole lab, source or group left out of training"),
    ("LODO", "Unseen dataset", "a whole dataset left out of training"),
]

d = pd.read_csv(CSV)
names = list(LABELS)
vals = []
for key, _, _ in STAGES:
    row = []
    for n in names:
        r = d[(d.dataset == n) & d.split.str.startswith(key)]
        row.append(float(r.within_10pct_mean.iloc[0]) * 100 if len(r) else np.nan)
    vals.append(np.array(row))
means = [np.nanmean(v) for v in vals]
print("means", [round(m, 1) for m in means], "n", [int(np.sum(~np.isnan(v))) for v in vals])

HOLD, MORPH = 26, 10
seq = []  # (stage_a, stage_b, t)
for i in range(3):
    seq += [(i, i, 0.0)] * HOLD
    seq += [(i, (i + 1) % 3, (k + 1) / MORPH) for k in range(MORPH)]


def ease(t):
    return t * t * (3 - 2 * t)


def render(a, b, t):
    e = ease(t)
    va, vb = np.nan_to_num(vals[a]), np.nan_to_num(vals[b])
    v = va + (vb - va) * e
    m = means[a] + (means[b] - means[a]) * e
    stage = b if e > 0.5 else a
    fig = new_fig()
    header(fig, 4, "Still honest: harder tests, lower scores")
    fig.text(0.06, 0.885, "Our physics + ML model · predictions within ±10% of measured CHF",
             fontsize=15, color=MUTED, va="center")
    # stage pills
    for i, (_, name, _) in enumerate(STAGES):
        x = 0.06 + i * 0.30
        active = i == stage
        fig.text(x + 0.135, 0.82, name, ha="center", va="center", fontsize=13.5,
                 weight="bold" if active else "medium",
                 color="white" if active else MUTED,
                 bbox=dict(boxstyle="round,pad=0.45,rounding_size=0.8",
                           fc=TEAL if active else "#F1F3F5", ec="none"))
    fig.text(0.5, 0.765, STAGES[stage][2], ha="center", fontsize=13, color=MUTED, style="italic")
    # big number
    col = TEAL if m > 55 else (INK if m > 40 else RED)
    fig.text(0.06, 0.685, f"{m:.0f}%", fontsize=52, weight="bold", color=col, va="center")
    fig.text(0.30, 0.70, "average across datasets", fontsize=15, color=INK, va="center")
    # bars
    ax = fig.add_axes([0.40, 0.10, 0.54, 0.53])
    y = np.arange(len(names))[::-1]
    ax.barh(y, [100] * len(y), color="#F1F3F5", height=0.62)
    cols = [TEAL if x >= 55 else (INK if x >= 30 else RED) for x in v]
    ax.barh(y, v, color=cols, height=0.62)
    for yi, vi, n, ra, rb in zip(y, v, names, vals[a], vals[b]):
        missing = (np.isnan(ra) and e < 0.5) or (np.isnan(rb) and e >= 0.5)
        ax.text(min(vi, 100) + 2, yi, "n/a" if missing else f"{vi:.0f}%", va="center",
                fontsize=13, color=MUTED if missing else INK, weight="demibold")
        fig.text(0.385, 0.10 + 0.53 * (yi + 0.5) / len(names), LABELS[n], ha="right",
                 va="center", fontsize=13.5)
    ax.set_xlim(0, 112); ax.set_ylim(-0.5, len(names) - 0.5)
    ax.axis("off")
    fig.text(0.94, 0.035, "Mean over seeds · n/a: no group split possible for that dataset",
             fontsize=11, color=MUTED, ha="right")
    return fig_to_image(fig)


frames = [render(*s) for s in seq]
save_gif(frames, OUT, FPS, holds={HOLD - 1: 600, MORPH + 2 * HOLD - 1: 600, 2 * MORPH + 3 * HOLD - 1: 1200},
         start=2 * MORPH + 3 * HOLD - 1)
print("frames", len(frames))

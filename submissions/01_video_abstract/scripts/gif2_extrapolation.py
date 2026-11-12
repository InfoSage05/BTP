"""GIF 2 - Accurate inside the data, wrong outside it.

Real data: 2006 Groeneveld CHF look-up table (data/chf_long_clean.csv), Split C:
train on P <= 16 MPa, test on 17-21 MPa (2,310 rows). Models are refit here with
the same settings as the verified README table (ExtraTrees R2 0.4335,
degree-2 Ridge on log CHF R2 0.7547); the script asserts those values.
"""
import sys
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from gifkit import *

DATA, OUT = sys.argv[1], sys.argv[2]
G_SL, X_SL = 500.0, -0.2
FPS = 12

df = pd.read_csv(DATA)
df = df[(df.CHF > 0) & (df.X < 1.0)]
tr, te = df[df.P <= 16000], df[df.P >= 17000]
F = ["P", "G", "X"]
et = ExtraTreesRegressor(300, random_state=0, n_jobs=-1).fit(tr[F], tr.CHF)
rd = make_pipeline(StandardScaler(), PolynomialFeatures(2), Ridge(1.0)).fit(tr[F], np.log(tr.CHF))
r2_et = r2_score(te.CHF, et.predict(te[F]))
r2_rd = r2_score(te.CHF, np.exp(rd.predict(te[F])))
assert abs(r2_et - 0.4335) < 0.01 and abs(r2_rd - 0.7547) < 0.005, (r2_et, r2_rd)

sl = df[(df.G == G_SL) & (df.X == X_SL) & (df.P >= 4000)].sort_values("P")
P = sl.P.values / 1000
Y = sl.CHF.values / 1000
Pd = np.linspace(4000, 21000, 300)
q = pd.DataFrame({"P": Pd, "G": G_SL, "X": X_SL})
ET = et.predict(q) / 1000
RD = np.exp(rd.predict(q)) / 1000
Pd /= 1000
YMAX = float(np.ceil(Y.max() * 1.1))

N_DRAW, N_HOLD, N_FADE = 72, 26, 10
P_START, P_END = 4.0, 21.0


def render(f):
    fig = new_fig()
    header(fig, 2, "Right inside the data, wrong outside")
    fig.text(0.06, 0.885, "Models trained only up to 16 MPa, then asked about 17–21 MPa",
             fontsize=15, color=MUTED, va="center")
    ax = fig.add_axes([0.13, 0.30, 0.82, 0.52])
    cur = P_START + (P_END - P_START) * min(1.0, f / (N_DRAW - 1))
    ax.axvspan(3.5, 16.5, color=TEAL_LIGHT, lw=0)
    ax.axvline(16.5, color=TEAL, lw=1.5, ls=(0, (4, 3)))
    ax.text(4.0, 0.3, "training data", ha="left", color=TEAL, fontsize=15, weight="demibold")
    ax.text(18.9, YMAX * 0.93, "never seen", ha="center", color=RED, fontsize=15, weight="demibold")
    m = Pd <= cur
    ax.plot(Pd[m], RD[m], color=TEAL, lw=3, ls=(0, (5, 3)), zorder=3)
    ax.plot(Pd[m], ET[m], color=RED, lw=3.4, zorder=4)
    pm = P <= cur + 1e-9
    ax.plot(P[pm], Y[pm], "o", ms=10, color=INK, mec="white", mew=1.6, zorder=5)
    if cur > 16.6:
        ax.plot([cur], [np.interp(cur, Pd, ET)], "o", ms=9, color=RED, zorder=6)
    done = f >= N_DRAW
    if done:
        ax.text(20.9, 3.32, "tree model\nstays flat", color=RED, fontsize=14,
                ha="right", va="bottom", weight="bold", linespacing=1.0)
        ax.text(21.3, 0.2, "smooth model\nfollows the trend", color=TEAL, fontsize=13,
                ha="right", va="bottom", weight="demibold", linespacing=1.0)
    ax.set_xlim(3.5, 21.5); ax.set_ylim(0, YMAX)
    ax.set_xlabel("Pressure (MPa)", fontsize=15)
    ax.set_ylabel("Critical heat flux (MW/m²)", fontsize=15)
    ax.grid(axis="y", color=GRID); ax.set_axisbelow(True)
    for sp in ["top", "right"]:
        ax.spines[sp].set_visible(False)
    # legend
    lx = fig.add_axes([0.06, 0.13, 0.9, 0.08]); lx.axis("off"); lx.set_xlim(0, 1); lx.set_ylim(0, 1)
    lx.plot([0.01], [0.75], "o", ms=10, color=INK); lx.text(0.04, 0.75, "look-up table value", va="center", fontsize=13)
    lx.plot([0.47, 0.54], [0.75, 0.75], color=RED, lw=3.4); lx.text(0.56, 0.75, "tree ensemble", va="center", fontsize=13)
    lx.plot([0.47, 0.54], [0.15, 0.15], color=TEAL, lw=3, ls=(0, (5, 3))); lx.text(0.56, 0.15, "polynomial ridge (log)", va="center", fontsize=13)
    a = min(1.0, max(0.0, (f - N_DRAW) / 6))
    if a > 0:
        fig.text(0.06, 0.065, f"All 2,310 unseen points:  tree R² = {r2_et:.2f}   ·   ridge R² = {r2_rd:.2f}",
                 fontsize=15, weight="bold", color=INK, alpha=a)
    fig.text(0.94, 0.025, f"2006 CHF look-up table · G = {int(G_SL)} kg/m²s, x = {X_SL}",
             fontsize=11, color=MUTED, ha="right")
    return fig_to_image(fig)


frames = [render(f) for f in range(N_DRAW + N_HOLD)]
frames = crossfade(frames, N_FADE)
save_gif(frames, OUT, FPS, holds={N_DRAW + N_HOLD - 1: 1500}, start=N_DRAW + N_HOLD - 1)
print("R2", r2_et, r2_rd, "frames", len(frames))

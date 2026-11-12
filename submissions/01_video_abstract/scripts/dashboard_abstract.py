"""One-page animated graphical abstract (1920x1080 GIF), hydrogel-poster style.

Every number is read from the project files and asserted:
  - 2006 LUT Split C (data/chf_long_clean.csv): ExtraTrees R2 0.43, Ridge-log R2 0.75
  - FINAL_results.csv: within +-10%, random / grouped / leave-one-dataset-out
  - audit_v2_features.csv (F0_base5): median R2 ML -2.06, physics 0.34, physics+ML 0.71, 7/8
  - manuscript: 28,582 records, 8 datasets, 53.2% median interval coverage
  - docs/CONFOUNDING_FINDINGS.md: ATF surface ICC 0.815
Usage (from the repo root): python submissions/01_video_abstract/scripts/dashboard_abstract.py . submissions/01_video_abstract/CHF_graphical_abstract.gif
"""
import sys, textwrap
import numpy as np
import pandas as pd
from matplotlib.patches import FancyBboxPatch, Rectangle, Circle, Polygon, Wedge
from matplotlib.colors import to_rgb
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
from gifkit import INK, MUTED, TEAL, RED, WATER, GRID, fig_to_image, save_gif_delta

ROOT, OUT = sys.argv[1], sys.argv[2]
W, H = 1920, 1080
PAGE = "#F4F2EE"
CARD_EC = "#E2DED7"
NAVY = "#1F3B57"
FPS, N = 12, 96   # 8 s loop

# ------------------------------------------------------------------ data
lut = pd.read_csv(f"{ROOT}/data/chf_long_clean.csv")
lut = lut[(lut.CHF > 0) & (lut.X < 1)]
tr, te = lut[lut.P <= 16000], lut[lut.P >= 17000]
F = ["P", "G", "X"]
et = ExtraTreesRegressor(300, random_state=0, n_jobs=-1).fit(tr[F], tr.CHF)
rd = make_pipeline(StandardScaler(), PolynomialFeatures(2), Ridge(1.0)).fit(tr[F], np.log(tr.CHF))
R2_ET = r2_score(te.CHF, et.predict(te[F]))
R2_RD = r2_score(te.CHF, np.exp(rd.predict(te[F])))
assert abs(R2_ET - 0.4335) < 0.01 and abs(R2_RD - 0.7547) < 0.005
sl = lut[(lut.G == 500) & (lut.X == -0.2) & (lut.P >= 4000)].sort_values("P")
SP, SY = sl.P.values / 1000, sl.CHF.values / 1000
Pd = np.linspace(4000, 21000, 200)
q = pd.DataFrame({"P": Pd, "G": 500.0, "X": -0.2})
ET_L, RD_L, Pd = et.predict(q) / 1000, np.exp(rd.predict(q)) / 1000, Pd / 1000

M = f"{ROOT}/docs/manuscript_revision_2026-11-05"
fin = pd.read_csv(f"{ROOT}/results/prof_tasks/FINAL_results.csv")
W10 = [fin[fin.split.str.startswith(k)].within_10pct_mean.mean() * 100 for k in ["80/20", "GROUPED", "LODO"]]
assert [round(x) for x in W10] == [68, 39, 32], W10
ab = pd.read_csv(f"{ROOT}/results/prof_tasks/audit_v2_features.csv")
ab = ab[ab.set == "F0_base5"]
MED = [ab[ab.arm == a].R2.median() for a in ["pooled_ML", "katto_alone", "PIPE"]]
pv = ab.pivot_table(index="held_out", columns="arm", values="R2")
WINS = int((pv.PIPE > pv.katto_alone).sum())
assert WINS == 7 and abs(MED[0] + 2.061) < 1e-3 and abs(MED[2] - 0.708) < 1e-3
paper = open(f"{M}/_work/paper_text.md").read()
assert "28,582 records" in paper and "53.2%" in paper and "eight flow-boiling datasets" in paper
conf = open(f"{ROOT}/docs/CONFOUNDING_FINDINGS.md").read()
assert "0.815" in conf
COVER = 53.2

# ------------------------------------------------------------------ helpers
def ramp(p, a, b):
    return float(np.clip((p - a) / (b - a), 0, 1))

def ease(t):
    return t * t * (3 - 2 * t)

def mix(c1, c2, t):
    a, b = np.array(to_rgb(c1)), np.array(to_rgb(c2))
    return tuple(a + (b - a) * np.clip(t, 0, 1))

def sub(fig, x, y, w, h):
    """axes at pixel box (top-left origin)"""
    return fig.add_axes([x / W, 1 - (y + h) / H, w / W, h / H])

def card(o, x, y, w, h, kicker, title, kc=TEAL):
    o.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=14",
                               fc="white", ec=CARD_EC, lw=1.4))
    o.text(x + 22, y + 30, kicker, fontsize=11, color=kc, weight="bold", va="center")
    o.text(x + 22, y + 60, title, fontsize=17, color=INK, weight="bold", va="center")

def para(o, x, y, s, width, size=12, color=INK, gap=1.45, weight="regular"):
    for i, line in enumerate(textwrap.wrap(s, width)):
        o.text(x, y + i * size * gap * 1.39, line, fontsize=size, color=color, va="top", weight=weight)

rng = np.random.default_rng(3)
BUB = [(rng.uniform(0.3, 9.7), rng.uniform(0, 1), i / 40) for i in range(40)]
BIG = [(rng.uniform(0.6, 9.4), rng.uniform(0, 1)) for _ in range(7)]
APP = [[(rng.uniform(0, 1), rng.uniform(0, 1)) for _ in range(14)] for _ in range(3)]
TAGS = ["NRC", "Zhao", "KAERI-U", "KAERI-N", "Coils R-123", "H&P coils", "H&P tubes", "Pioro"]

# ------------------------------------------------------------------ panels
def p_chf(fig, o, p):
    x, y = 40, 165
    card(o, x, y, 600, 290, "THE PROBLEM", "What is critical heat flux?", RED)
    heat = ramp(p, 0, 0.5) * (1 - ramp(p, 0.86, 0.98))
    film = ramp(p, 0.5, 0.62) * (1 - ramp(p, 0.84, 0.96))
    hot = ramp(p, 0.58, 0.68) * (1 - ramp(p, 0.84, 0.95))
    ax = sub(fig, x + 20, y + 85, 300, 185); ax.axis("off")
    ax.set_xlim(0, 10); ax.set_ylim(-0.6, 7.4)
    ax.add_patch(Rectangle((0, 1), 10, 6.4, fc=WATER, ec="none"))
    for bx, bo, rank in BUB:
        if rank < 0.08 + 0.92 * heat * (1 - film):
            fr = (p * 4 + bo) % 1
            ax.add_patch(Circle((bx + 0.15 * np.sin(fr * 9), 1.25 + fr * 6.0), 0.14 + 0.28 * fr,
                                fc="white", ec=TEAL, lw=1.1))
    if film > 0:
        xx = np.linspace(0, 10, 120)
        top = 1 + 0.55 * film + 0.12 * film * np.sin(xx * 2 + p * 40)
        ax.add_patch(Polygon(np.vstack([np.c_[xx, top], [[10, 1], [0, 1]]]), fc="white", ec=TEAL, lw=1.2))
        for bx, bo in BIG:
            fr = (p * 3 + bo) % 1
            ax.add_patch(Circle((bx, 1.9 + fr * 5), 0.5, fc=(1, 1, 1, film), ec=(*to_rgb(TEAL), film), lw=1.1))
    wall = mix("#9AA5B1", "#E8743B", heat * 0.7) if hot == 0 else mix("#E8743B", "#C81D25", hot)
    ax.add_patch(Rectangle((0, 0), 10, 1, fc=wall, ec="none"))
    ax.text(0.2, 0.5, "heated wall", color="white", fontsize=10, weight="bold", va="center")
    for xa in np.linspace(1, 9, 6):
        ax.annotate("", (xa, -0.05), (xa, -0.6), arrowprops=dict(arrowstyle="-|>", color=RED, lw=1.6))
    # wall temperature gauge
    gx = x + 340
    o.add_patch(FancyBboxPatch((gx, y + 90), 22, 145, boxstyle="round,pad=0,rounding_size=10", fc="#EEF0F2", ec="none"))
    lvl = 0.18 + 0.22 * heat + 0.6 * hot
    o.add_patch(FancyBboxPatch((gx, y + 90 + 145 * (1 - lvl)), 22, 145 * lvl, boxstyle="round,pad=0,rounding_size=10",
                               fc=mix(TEAL, RED, hot), ec="none"))
    o.text(gx + 11, y + 243, "wall\ntemp", fontsize=9.5, color=MUTED, ha="center", va="top", linespacing=1.0)
    msg = ("More heat, more bubbles" if film < 0.3 else
           ("Bubbles merge into a vapour film" if hot < 0.5 else "Wall temperature jumps"))
    o.text(x + 385, y + 100, msg, fontsize=12.5, weight="bold", va="top",
           color=RED if hot >= 0.5 else (INK if film >= 0.3 else TEAL))
    para(o, x + 385, y + 135, "Boiling carries heat off a hot wall. Past a limit, vapour blankets the "
         "wall and cooling collapses. That limit is the critical heat flux (CHF).", 27, size=11.5)


def p_apps(fig, o, p):
    x, y = 660, 165
    card(o, x, y, 600, 290, "WHY IT MATTERS", "Boiling cools the hottest machines")
    labels = ["Nuclear fuel rod", "EV battery cold plate", "Electronics chip"]
    for k in range(3):
        ax = sub(fig, x + 22 + k * 190, y + 85, 175, 150); ax.axis("off")
        ax.set_xlim(0, 10); ax.set_ylim(0, 10)
        o.text(x + 22 + k * 190 + 87, y + 250, labels[k], fontsize=11.5, ha="center", weight="demibold")
        if k == 0:
            ax.add_patch(Rectangle((1, 0), 8, 10, fc=WATER, ec="none"))
            ax.add_patch(FancyBboxPatch((4.2, 0.4), 1.6, 9.2, boxstyle="round,pad=0,rounding_size=0.8",
                                        fc="#E8743B", ec="none"))
            for i, (a, b) in enumerate(APP[0]):
                fr = (p * 2 + b) % 1
                side = 3.5 if i % 2 else 6.5
                ax.add_patch(Circle((side + 0.4 * np.sin(fr * 12 + a * 6), 0.6 + fr * 9), 0.22 + 0.25 * fr,
                                    fc="white", ec=TEAL, lw=1))
            ax.annotate("", (8.8, 9.6), (8.8, 0.4), arrowprops=dict(arrowstyle="-|>", color=TEAL, lw=1.4))
        elif k == 1:
            for c in range(4):
                ax.add_patch(FancyBboxPatch((0.6 + c * 2.3, 4.2), 1.9, 5, boxstyle="round,pad=0,rounding_size=0.3",
                                            fc="#3C4F63", ec="none"))
                ax.add_patch(Rectangle((1.25 + c * 2.3, 9.2), 0.6, 0.45, fc="#3C4F63"))
                ax.text(1.55 + c * 2.3, 6.7, "+", color="white", ha="center", va="center", fontsize=13, weight="bold")
            ax.add_patch(Rectangle((0.2, 1.0), 9.6, 2.8, fc="#C9D3DC", ec="none"))
            ax.add_patch(Rectangle((0.2, 1.7), 9.6, 1.4, fc=WATER, ec="none"))
            for i, (a, b) in enumerate(APP[1]):
                fr = (p * 2 + b) % 1
                ax.add_patch(Circle((0.3 + fr * 9.4, 2.0 + 0.8 * a), 0.18 + 0.2 * fr, fc="white", ec=TEAL, lw=1))
            ax.annotate("", (9.9, 0.5), (0.1, 0.5), arrowprops=dict(arrowstyle="-|>", color=TEAL, lw=1.4))
        else:
            ax.add_patch(Rectangle((0, 0), 10, 10, fc=WATER, ec="none"))
            ax.add_patch(Rectangle((0, 0), 10, 1.6, fc="#2E7D4F", ec="none"))
            ax.add_patch(Rectangle((2.5, 1.6), 5, 1.4, fc="#2B2F36", ec="none"))
            for i, (a, b) in enumerate(APP[2]):
                fr = (p * 3 + b) % 1
                ax.add_patch(Circle((2.8 + 4.4 * a + 0.3 * np.sin(fr * 10), 3.2 + fr * 6.5), 0.18 + 0.3 * fr,
                                    fc="white", ec=TEAL, lw=1))
    o.text(x + 300, y + 272, "Push past CHF and the surface can overheat and fail.", fontsize=12,
           color=RED, ha="center", weight="demibold")


def p_data(fig, o, p):
    x, y = 1280, 165
    card(o, x, y, 600, 290, "OUR DATA", "Two benchmarks, eight datasets")
    stats = [("11,088", "points in the 2006 CHF look-up table"), ("28,582", "records in 8 flow-boiling datasets"),
             ("3 fluids", "water · R-123 · R-134a")]
    for i, (big, small) in enumerate(stats):
        o.text(x + 22, y + 102 + i * 44, big, fontsize=21, color=TEAL, weight="bold", va="center")
        o.text(x + 175, y + 102 + i * 44, small, fontsize=12.5, va="center")
    k = int(p * 8) % 8
    for i, t in enumerate(TAGS):
        cx, cy = x + 22 + (i % 4) * 142, y + 232 + (i // 4) * 32
        held = i == k
        o.add_patch(FancyBboxPatch((cx, cy - 12), 130, 24, boxstyle="round,pad=0,rounding_size=11",
                                   fc=RED if held else "#E4F1F2", ec="none"))
        o.text(cx + 65, cy, ("held out: " if held else "") + t, fontsize=9.8, ha="center", va="center",
               color="white" if held else TEAL, weight="bold")


def p_extrap(fig, o, p):
    x, y = 40, 500
    card(o, x, y, 600, 290, "FINDING", "Right inside the data, wrong outside", RED)
    o.text(x + 22, y + 85, "Trained on pressures ≤ 16 MPa, then asked about 17–21 MPa", fontsize=11.5, color=MUTED, va="center")
    ax = sub(fig, x + 75, y + 102, 335, 128)
    cur = 4 + 17 * ease(ramp(p, 0.02, 0.62))
    fade = 1 - 0.85 * ramp(p, 0.9, 1.0)
    ax.axvspan(3.5, 16.5, color="#E4F1F2", lw=0); ax.axvline(16.5, color=TEAL, ls=(0, (3, 3)), lw=1.2)
    m = Pd <= cur
    ax.plot(Pd[m], RD_L[m], color=TEAL, lw=2.3, ls=(0, (4, 2.5)), alpha=fade)
    ax.plot(Pd[m], ET_L[m], color=RED, lw=2.6, alpha=fade)
    k = SP <= cur + 1e-9
    ax.plot(SP[k], SY[k], "o", ms=5.5, color=INK, alpha=fade)
    ax.set_xlim(3.5, 21.5); ax.set_ylim(0, 8)
    ax.set_xticks([4, 8, 12, 16, 20]); ax.set_yticks([0, 4, 8])
    ax.tick_params(labelsize=9.5, length=0)
    ax.set_xlabel("Pressure (MPa)", fontsize=10.5); ax.set_ylabel("CHF (MW/m²)", fontsize=10.5)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    ax.text(10, 0.6, "training data", color=TEAL, fontsize=9.5, ha="center", weight="bold")
    ax.text(19, 7.0, "never seen", color=RED, fontsize=9.5, ha="center", weight="bold")
    lx = x + 430
    o.plot([lx, lx + 26], [y + 125, y + 125], color=RED, lw=2.6)
    o.text(lx + 34, y + 125, f"tree model\nR² = {R2_ET:.2f}", fontsize=11, va="center", linespacing=1.1)
    o.plot([lx, lx + 26], [y + 175, y + 175], color=TEAL, lw=2.3, ls=(0, (4, 2.5)))
    o.text(lx + 34, y + 175, f"polynomial ridge\nR² = {R2_RD:.2f}", fontsize=11, va="center", linespacing=1.1)
    o.plot([lx + 13], [y + 222], "o", ms=5.5, color=INK)
    o.text(lx + 34, y + 222, "look-up table", fontsize=11, va="center")
    o.text(lx, y + 258, "R² on 2,310 unseen points", fontsize=9.5, color=MUTED, va="center")


def bar_h(o, x0, y0, width, val, lo, hi, color):
    zero = x0 + width * (0 - lo) / (hi - lo)
    end = x0 + width * (np.clip(val, lo, hi) - lo) / (hi - lo)
    o.add_patch(Rectangle((min(zero, end), y0 - 13), abs(end - zero), 26, fc=color, ec="none"))
    return zero, end


def p_fix(fig, o, p):
    x, y = 660, 500
    card(o, x, y, 600, 290, "OUR FIX", "Start from physics, let ML fix the error")
    o.text(x + 22, y + 85, "Each dataset hidden in turn · median R² (1 = perfect)", fontsize=11.5, color=MUTED, va="center")
    g = ease(ramp(p, 0.04, 0.4)) * (1 - ramp(p, 0.9, 1.0))
    names = ["ML alone", "Physics alone", "Physics + ML"]
    cols = [RED, INK, TEAL]
    bx, bw = x + 190, 300
    lo, hi = -1.0, 1.0
    zero = bx + bw * 0.5
    o.plot([zero, zero], [y + 105, y + 245], color=MUTED, lw=1)
    for i in range(3):
        cy = y + 125 + i * 45
        o.text(x + 175, cy, names[i], fontsize=12.5, ha="right", va="center", weight="demibold")
        v = MED[i] * g
        _, end = bar_h(o, bx, cy, bw, v, lo, hi, cols[i])
        lab = f"{MED[i]:.2f}".replace("-", "−") if g > 0.95 else ""
        if MED[i] < lo and g > 0.95:
            o.text(bx + 4, cy, "◀ " + lab, fontsize=11.5, color="white", va="center", weight="bold")
        elif MED[i] >= 0:
            o.text(end + 8, cy, lab, fontsize=12.5, va="center", weight="bold", color=cols[i])
    o.text(bx, y + 255, "−1", fontsize=9.5, color=MUTED, ha="center"); o.text(zero, y + 255, "0", fontsize=9.5, color=MUTED, ha="center")
    o.text(bx + bw, y + 255, "1", fontsize=9.5, color=MUTED, ha="center")
    a = ramp(p, 0.42, 0.5) * (1 - ramp(p, 0.9, 1.0))
    o.text(x + 22, y + 272, f"Physics + ML beats physics alone on {WINS} of 8 datasets", fontsize=12.5,
           color=TEAL, weight="bold", va="center", alpha=max(a, 0.001))


def p_honest(fig, o, p):
    x, y = 1280, 500
    card(o, x, y, 600, 290, "HONEST CHECK", "Harder tests, lower scores", RED)
    o.text(x + 22, y + 85, "Our physics + ML model · predictions within ±10% of measured CHF",
           fontsize=11.5, color=MUTED, va="center")
    labels = ["random\n80/20 split", "unseen lab\nor group", "unseen\ndataset"]
    cols = [TEAL, INK, RED]
    base, top = y + 240, y + 110
    fade = 1 - ramp(p, 0.9, 1.0)
    for i in range(3):
        g = ease(ramp(p, 0.05 + 0.15 * i, 0.22 + 0.15 * i)) * fade
        bx = x + 60 + i * 180
        hgt = (base - top) * W10[i] / 100 * g
        o.add_patch(Rectangle((bx, base - (base - top)), 110, base - top, fc="#F1F3F5", ec="none"))
        o.add_patch(Rectangle((bx, base - hgt), 110, hgt, fc=cols[i], ec="none"))
        if g > 0.9:
            o.text(bx + 55, base - hgt - 14, f"{W10[i]:.0f}%", fontsize=15, ha="center", weight="bold", color=cols[i])
        o.text(bx + 55, base + 8, labels[i], fontsize=10.5, ha="center", va="top", linespacing=1.05)


def p_method(fig, o, p):
    x, y = 40, 805
    card(o, x, y, 1220, 235, "OUR METHOD", "Physics-anchored machine learning")
    steps = [("Inputs", "Weber number, density ratio,\nL/D, inlet subcooling,\nreduced pressure"),
             ("Physics", "Katto–Ohno CHF\ncorrelation gives a\nfirst estimate"),
             ("ML", "ExtraTrees learns the\nlog error of that\nestimate"),
             ("Trust bound", "correction is capped\nwhen the case looks\nunfamiliar"),
             ("Output", "CHF prediction\nwith a prediction\ninterval")]
    bw, gap = 210, 27
    act = int(p * 5) % 5
    t = (p * 5) % 1
    for i, (h, s) in enumerate(steps):
        bx = x + 22 + i * (bw + gap)
        on = i == act
        o.add_patch(FancyBboxPatch((bx, y + 85), bw, 130, boxstyle="round,pad=0,rounding_size=10",
                                   fc="#E4F1F2" if on else "#F7F8FA", ec=TEAL if on else CARD_EC, lw=2.2 if on else 1.2))
        o.text(bx + 14, y + 108, h, fontsize=13.5, weight="bold", color=TEAL if on else INK, va="center")
        o.text(bx + 14, y + 128, s, fontsize=10.8, va="top", linespacing=1.25)
        if i < 4:
            ax0 = bx + bw + 3
            o.annotate("", (ax0 + gap - 6, y + 150), (ax0, y + 150),
                       arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=1.5))
    # travelling pulse
    if act < 4:
        px = x + 22 + act * (bw + gap) + bw + 3 + (gap - 6) * ease(ramp(t, 0.55, 1.0))
        if t > 0.55:
            o.plot([px], [y + 150], "o", ms=8, color=RED, zorder=5)


def p_limits(fig, o, p):
    x, y = 1280, 805
    card(o, x, y, 600, 235, "LIMITS WE REPORT", "What still fails", RED)
    cx, cy, r = x + 95, y + 155, 58
    g = ease(ramp(p, 0.05, 0.45)) * (1 - ramp(p, 0.9, 1.0))
    o.add_patch(Wedge((cx, cy), r, 0, 360, width=16, fc="#F1F3F5", ec="none"))
    o.add_patch(Wedge((cx, cy), r, 90 - 360 * COVER / 100 * g, 90, width=16, fc=RED, ec="none"))
    ang = np.deg2rad(90 - 360 * 0.9)
    o.plot([cx + (r - 20) * np.cos(ang), cx + (r + 4) * np.cos(ang)],
           [cy - (r - 20) * np.sin(ang), cy - (r + 4) * np.sin(ang)], color=INK, lw=2.4)
    o.text(cx, cy, f"{COVER * g:.0f}%", fontsize=16, weight="bold", ha="center", va="center", color=RED)
    o.text(cx, cy + r + 10, "black tick = 90% target", fontsize=9.5, ha="center", va="top", color=MUTED)
    bullets = ["Intervals meant to cover 90% cover only 53%",
               "Zhao and Pioro datasets: R² still below zero",
               "PINN physics penalties: no measurable gain",
               "Lab identity alone: 82% of CHF variance",
               "   (in one engineered-surface dataset)"]
    for i, b in enumerate(bullets):
        by = y + 100 + i * 27 - (6 if i == 4 else 0)
        if not b.startswith(" "):
            o.add_patch(Rectangle((x + 190, by - 4), 8, 8, fc=RED, ec="none"))
        o.text(x + 207, by, b.strip(), fontsize=11.5 if i < 4 else 10.5, va="center",
               color=INK if i < 4 else MUTED)


def render(f):
    p = f / N
    fig = plt.figure(figsize=(W / 100, H / 100), dpi=100)
    fig.patch.set_facecolor(PAGE)
    o = fig.add_axes([0, 0, 1, 1]); o.set_xlim(0, W); o.set_ylim(H, 0); o.axis("off")
    # header
    o.text(40, 42, "BTP PROJECT · VIDEO ABSTRACT", fontsize=11.5, color=TEAL, weight="bold", va="center")
    o.text(40, 92, "Machine Learning for Critical Heat Flux", fontsize=36, weight="bold", color=INK, va="center")
    o.text(40, 140, "Can ML predict the boiling limit outside the data it learned from?", fontsize=16, color="#4A5663", va="center")
    names = [("Mauryavardhan Singh", "23ME10048"), ("Mayank Nagar", "23ME10049"),
             ("Ayushman Paul", "23MF3IM02"), ("Hardik Mahawar", "23ME31037")]
    for i, (n, r) in enumerate(names):
        cx = 1300 + (i % 2) * 310
        cy = 40 + (i // 2) * 32
        o.text(cx + 175, cy, n, fontsize=14, weight="bold", ha="right", va="center")
        o.text(cx + 185, cy, r, fontsize=13, color=MUTED, va="center")
    o.text(1880, 108, "Prof. Chirodeep Bakli (Professor)  ·  Prof. Bikash Pattanayak (Supervisor)",
           fontsize=12.5, ha="right", va="center")
    o.text(1880, 136, "School of Energy Science and Engineering · IIT Kharagpur", fontsize=12.5,
           color=TEAL, ha="right", va="center")
    o.plot([40, 1880], [158, 158], color=CARD_EC, lw=1.2)
    o.text(40, 478, "WHAT WE FOUND  ·  A MODEL MUST BE TESTED OUTSIDE ITS TRAINING DATA",
           fontsize=11.5, color=TEAL, weight="bold", va="center")
    p_chf(fig, o, p); p_apps(fig, o, p); p_data(fig, o, p)
    p_extrap(fig, o, p); p_fix(fig, o, p); p_honest(fig, o, p)
    p_method(fig, o, p); p_limits(fig, o, p)
    o.text(40, 1062, "Data: Groeneveld et al. (2007) 2006 CHF look-up table · eight published flow-boiling datasets · "
           "Katto & Ohno (1984) correlation · all numbers from our own results files",
           fontsize=10, color=MUTED, va="center")
    return fig_to_image(fig, bg=PAGE)


if __name__ == "__main__":
    only = [int(a) for a in sys.argv[3].split(",")] if len(sys.argv) > 3 else None
    frames = [render(f) for f in (only or range(N))]
    if only:
        for f, im in zip(only, frames):
            im.save(OUT.replace(".gif", f"_f{f:02d}.png"))
    else:
        save_gif_delta(frames, OUT, FPS)
    print("done", len(frames))

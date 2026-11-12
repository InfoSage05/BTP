"""GIF 1 - What is critical heat flux? (schematic, no measured values shown)."""
import sys
import numpy as np
from matplotlib.patches import Rectangle, Circle, FancyArrowPatch, Polygon
from matplotlib.colors import to_rgb
from gifkit import *

OUT = sys.argv[1]
FPS = 12
N_RISE, N_MERGE, N_HOT, N_FADE = 66, 18, 30, 10
N = N_RISE + N_MERGE + N_HOT

rng = np.random.default_rng(7)
SITES = np.linspace(0.7, 9.3, 13) + rng.uniform(-0.15, 0.15, 13)
WALL_TOP = 1.0


def mix(c1, c2, t):
    a, b = np.array(to_rgb(c1)), np.array(to_rgb(c2))
    return tuple(a + (b - a) * np.clip(t, 0, 1))


# --- schematic boiling curve: q vs log10(wall superheat) ---
xs = np.linspace(0.0, 3.0, 400)
def q_of(x):
    nuc = np.where(x < 1.45, 0.06 + 0.94 * ((x / 1.45) ** 4), np.nan)
    trans = np.where((x >= 1.45) & (x < 2.25),
                     1.0 - 0.72 * np.sin((x - 1.45) / 0.8 * np.pi / 2) ** 2, np.nan)
    film = np.where(x >= 2.25, 0.28 + 0.72 * ((x - 2.25) / 0.75) ** 1.6, np.nan)
    return np.nan_to_num(nuc) + np.nan_to_num(trans) + np.nan_to_num(film)
QS = q_of(xs)
X_CHF = 1.45
X_FILM = 3.0  # film branch reaches q = 1 at x = 3


# --- bubble simulation (deterministic) ---
bubbles = []  # dict(x, y, r, rmax, attached, vy)
states = []
for f in range(N):
    if f < N_RISE:
        s = f / (N_RISE - 1)          # heat-flux level 0..1
    else:
        s = 1.0
    film = 0.0 if f < N_RISE else min(1.0, (f - N_RISE) / N_MERGE)
    # spawn
    rate = 0.03 + 0.55 * s ** 1.5
    if film < 0.6:
        for x0 in SITES:
            busy = any(b["attached"] and abs(b["x"] - x0) < 0.05 for b in bubbles)
            if not busy and rng.random() < rate:
                bubbles.append(dict(x=x0, y=WALL_TOP, r=0.05,
                                    rmax=0.16 + 0.22 * s + rng.uniform(0, 0.06),
                                    attached=True, vy=0.0, wob=rng.uniform(0, 6.28)))
    else:
        if rng.random() < 0.45:
            x0 = rng.uniform(0.6, 9.4)
            bubbles.append(dict(x=x0, y=WALL_TOP + 0.35, r=0.3 + rng.uniform(0, 0.15),
                                rmax=0.5, attached=False, vy=0.09, wob=rng.uniform(0, 6.28)))
    # update
    for b in bubbles:
        if b["attached"]:
            b["r"] += 0.035 + 0.04 * s
            b["y"] = WALL_TOP + b["r"] * 0.85
            if b["r"] >= b["rmax"]:
                b["attached"] = False
                b["vy"] = 0.06
        else:
            b["vy"] = min(b["vy"] + 0.012, 0.22)
            b["y"] += b["vy"]
            b["x"] += 0.03 * np.sin(b["wob"] + b["y"] * 2)
            b["r"] = min(b["r"] * 1.004, 0.6)
    # merge: during film formation attached bubbles flatten into film
    bubbles = [b for b in bubbles if b["y"] < 6.4 and not (film > 0.5 and b["attached"])]
    states.append(([dict(b) for b in bubbles], s, film, f))


def render(state):
    bubs, s, film, f = state
    fig = new_fig()
    header(fig, 1, "What is critical heat flux?")
    if f < N_RISE:
        msg, col = "More heat → more bubbles carry it away", TEAL
    elif f < N_RISE + N_MERGE:
        msg, col = "At CHF the bubbles merge into a vapour film", INK
    else:
        msg, col = "Vapour insulates → wall temperature jumps", RED
    fig.text(0.06, 0.885, msg, fontsize=16, color=col, weight="demibold", va="center")

    # ---- scene ----
    ax = fig.add_axes([0.06, 0.44, 0.88, 0.40])
    ax.set_xlim(0, 10); ax.set_ylim(-0.9, 6.2); ax.axis("off")
    ax.add_patch(Rectangle((0, WALL_TOP), 10, 5.2, fc=WATER, ec="none"))
    hot = 0.0 if f < N_RISE else min(1.0, (f - N_RISE - N_MERGE * 0.4) / 14)
    wall_c = mix("#9AA5B1", "#E8743B", s * 0.6) if hot <= 0 else mix("#E8743B", "#C81D25", hot)
    ax.add_patch(Rectangle((0, 0), 10, WALL_TOP, fc=wall_c, ec="none"))
    ax.text(0.15, 0.48, "Heated wall", fontsize=13, color="white", weight="bold", va="center")
    # heat arrows under wall
    for xa in np.linspace(1.2, 8.8, 7):
        ln = 0.25 + 0.45 * s
        ax.add_patch(FancyArrowPatch((xa, -0.85), (xa, -0.85 + ln), arrowstyle="-|>",
                                     mutation_scale=14, lw=2.2, color=RED))
    ax.text(9.95, -0.55, "heat in", fontsize=12, color=RED, ha="right", va="center")
    # vapour film
    if film > 0:
        h = 0.08 + 0.32 * film
        xx = np.linspace(0, 10, 200)
        top = WALL_TOP + h + 0.08 * film * np.sin(xx * 2.2 + f * 0.4)
        poly = np.vstack([np.c_[xx, top], [[10, WALL_TOP], [0, WALL_TOP]]])
        ax.add_patch(Polygon(poly, closed=True, fc="white", ec=TEAL, lw=1.4))
        if film >= 1:
            ax.text(5, WALL_TOP + 0.2, "vapour blanket", fontsize=12, color=TEAL,
                    ha="center", va="bottom", style="italic")
    for b in bubs:
        ax.add_patch(Circle((b["x"], b["y"]), b["r"], fc="white", ec=TEAL, lw=1.4))

    # ---- boiling curve ----
    ax2 = fig.add_axes([0.13, 0.08, 0.80, 0.30])
    ax2.plot(xs, QS, color=GRID, lw=3, solid_capstyle="round")
    if f < N_RISE:
        xd = 0.15 + (X_CHF - 0.15) * s
        m = xs <= xd
        ax2.plot(xs[m], QS[m], color=TEAL, lw=3.5, solid_capstyle="round")
        yd = q_of(np.array([xd]))[0]
    else:
        m = xs <= X_CHF
        ax2.plot(xs[m], QS[m], color=TEAL, lw=3.5)
        t = min(1.0, (f - N_RISE - N_MERGE * 0.5) / 10) if f >= N_RISE + N_MERGE * 0.5 else 0
        xd = X_CHF + (X_FILM - X_CHF) * t
        yd = 1.0
        if t > 0:
            ax2.annotate("", xy=(xd, 1.0), xytext=(X_CHF, 1.0),
                         arrowprops=dict(arrowstyle="-|>", color=RED, lw=2.4,
                                         linestyle=(0, (4, 3)), mutation_scale=16))
        if t >= 1:
            ax2.text(2.25, 0.86, "wall overheats\n(burnout risk)", color=RED, fontsize=13,
                     ha="center", va="top", weight="demibold")
    ax2.plot([xd], [yd], "o", ms=13, color=RED, mec="white", mew=2, zorder=5)
    ax2.text(X_CHF, 1.09, "CHF", ha="center", fontsize=14, weight="bold", color=INK)
    ax2.set_xlim(0, 3.1); ax2.set_ylim(0, 1.2)
    ax2.set_xticks([]); ax2.set_yticks([])
    for sp in ["top", "right"]:
        ax2.spines[sp].set_visible(False)
    ax2.set_xlabel("Wall temperature above boiling  →", fontsize=13)
    ax2.set_ylabel("Heat flux  →", fontsize=13)
    ax2.text(3.08, 0.02, "schematic boiling curve", fontsize=11, color=MUTED, ha="right")
    return fig_to_image(fig)


frames = [render(st) for st in states]
frames = crossfade(frames, N_FADE)
save_gif(frames, OUT, FPS, holds={N - 1: 1200}, start=N - 1)
print("frames", len(frames))

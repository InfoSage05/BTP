"""Shared style and GIF writer for the video-abstract GIFs."""
import io
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from PIL import Image

INK = "#13293D"
MUTED = "#6B7785"
TEAL = "#1A7F86"
TEAL_LIGHT = "#DDEFF0"
RED = "#D64550"
WATER = "#E3F1F7"
GRID = "#E6E9ED"
BG = "#FFFFFF"

for path in ["/System/Library/Fonts/Avenir Next.ttc"]:
    try:
        font_manager.fontManager.addfont(path)
    except Exception:
        pass
plt.rcParams.update({
    "font.family": ["Avenir Next", "Helvetica", "DejaVu Sans"],
    "font.size": 15,
    "text.color": INK,
    "axes.edgecolor": MUTED,
    "axes.labelcolor": INK,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "figure.facecolor": BG,
    "axes.facecolor": BG,
})

SIZE_IN = 8.0   # 8 in x 100 dpi = 800 px square
DPI = 100


def new_fig():
    return plt.figure(figsize=(SIZE_IN, SIZE_IN), dpi=DPI)


def save_gif_delta(frames, path, fps):
    """Large GIF writer: shared 255-colour palette, and every pixel unchanged from the
    previous frame is written as transparent, so static regions cost almost nothing."""
    import numpy as np
    w, h = frames[0].size
    sample = Image.new("RGB", (w, h * 3))
    for i in range(3):
        sample.paste(frames[i * (len(frames) - 1) // 2], (0, h * i))
    pal = sample.quantize(colors=255, method=Image.Quantize.MEDIANCUT)
    idx = [np.array(f.quantize(palette=pal, dither=Image.Dither.NONE)) for f in frames]
    palette = pal.getpalette()[:765] + [255, 0, 255]
    out = []
    for i, a in enumerate(idx):
        b = a.copy()
        if i > 0:
            b[a == idx[i - 1]] = 255
        im = Image.fromarray(b.astype("uint8"), "P")
        im.putpalette(palette)
        out.append(im)
    out[0].save(path, save_all=True, append_images=out[1:], duration=int(round(1000 / fps)),
                loop=0, disposal=1, transparency=255, optimize=False)


def fig_to_image(fig, bg=BG):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=DPI, facecolor=bg)
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def header(fig, number, title, subtitle=None):
    fig.text(0.06, 0.945, f"{number}", fontsize=17, weight="bold", color="white",
             ha="center", va="center",
             bbox=dict(boxstyle="circle,pad=0.32", fc=TEAL, ec="none"))
    fig.text(0.105, 0.945, title, fontsize=23, weight="bold", va="center")
    if subtitle:
        fig.text(0.06, 0.895, subtitle, fontsize=15, color=MUTED, va="center")


def crossfade(frames, n):
    """Append n frames blending the last frame back into the first (seamless loop)."""
    a, b = frames[-1], frames[0]
    for i in range(1, n + 1):
        frames.append(Image.blend(a, b, i / (n + 1)))
    return frames


def save_gif(frames, path, fps, holds=None, start=0):
    """Write an infinitely looping GIF with one shared palette (no colour flicker).
    holds: optional {frame_index: extra_ms} to pause on key frames.
    start: rotate the loop so this frame is shown first (static previews / PDFs
    show frame 0, so start on the most informative frame)."""
    n = len(frames)
    holds = {(i - start) % n: v for i, v in (holds or {}).items()}
    frames = frames[start:] + frames[:start]
    w, h = frames[0].size
    sample = Image.new("RGB", (w, h * 4))
    picks = [frames[int(i * (len(frames) - 1) / 3)] for i in range(4)]
    for i, f in enumerate(picks):
        sample.paste(f, (0, h * i))
    pal = sample.quantize(colors=128, method=Image.Quantize.MEDIANCUT)
    q = [f.quantize(palette=pal, dither=Image.Dither.NONE) for f in frames]
    base = int(round(1000 / fps))
    durations = [base + (holds or {}).get(i, 0) for i in range(len(q))]
    q[0].save(path, save_all=True, append_images=q[1:], duration=durations,
              loop=0, optimize=True, disposal=1)

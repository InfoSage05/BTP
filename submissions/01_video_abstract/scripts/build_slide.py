"""Put the three GIFs on the professor's Master Template slide (keeps background, logos, footer)."""
import sys
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR

TEMPLATE, GIF_DIR, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
TITLE = "Machine Learning Prediction of Critical Heat Flux: Testing Beyond the Training Data"
CAPTIONS = [
    "Too much heat turns bubbles into an insulating vapour film",
    "Tree models go flat beyond the pressures they were trained on",
    "Starting from physics, ML improves predictions on 7 of 8 unseen datasets",
]
MEMBERS = [("Mauryavardhan Singh", "23ME10048"), ("Mayank Nagar", "23ME10049"),
           ("Ayushman Paul", "23MF3IM02"), ("Hardik Mahawar", "23ME31037")]
GUIDES = "Professor: Prof. Chirodeep Bakli   ·   Supervisor: Prof. Bikash Pattanayak"
INK = RGBColor(0x13, 0x29, 0x3D)
MUTED = RGBColor(0x55, 0x60, 0x6C)
FONT = "Times New Roman"

prs = Presentation(TEMPLATE)
slide = prs.slides[0]
shapes = {s.name: s for s in slide.shapes}


def set_text(shape, lines, size, bold=False, color=INK, align=PP_ALIGN.CENTER):
    tf = shape.text_frame
    tf.clear(); tf.word_wrap = True
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    for i, (txt, sz, b, c) in enumerate(lines):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align
        r = p.add_run(); r.text = txt
        r.font.size = Pt(sz); r.font.bold = b; r.font.name = FONT
        r.font.color.rgb = c


# title
t = shapes["TextBox 5"]
t.left, t.top, t.width, t.height = Inches(0.6), Inches(0.18), Inches(12.13), Inches(1.0)
set_text(t, [(TITLE, 30, True, INK)], 30)

# remove the three "Enter Media" placeholders
for n in ["TextBox 16", "TextBox 17", "TextBox 18"]:
    el = shapes[n]._element; el.getparent().remove(el)

# GIFs: three 3.3 in squares, evenly spaced
side, top = 3.3, 1.28
gap = (13.333 - 3 * side) / 4
for i, name in enumerate(["gif1_what_is_chf.gif", "gif2_extrapolation.gif", "gif3_physics_anchor.gif"]):
    left = gap + i * (side + gap)
    slide.shapes.add_picture(f"{GIF_DIR}/{name}", Inches(left), Inches(top), Inches(side), Inches(side))
    cap = slide.shapes.add_textbox(Inches(left - 0.1), Inches(top + side + 0.02), Inches(side + 0.2), Inches(0.5))
    set_text(cap, [(CAPTIONS[i], 13, False, MUTED)], 13)

# members + guides
g = shapes["TextBox 6"]
g.left, g.top, g.width, g.height = Inches(0.5), Inches(5.1), Inches(12.33), Inches(0.9)
names = "  ·  ".join(f"{n} ({r})" for n, r in MEMBERS)
set_text(g, [(names, 13, True, INK), (GUIDES, 13, False, INK)], 13)

# footer stays; nudge it slightly to avoid overlap
f = shapes["TextBox 15"]
f.top = Inches(6.62)

# ---- Keynote compatibility ----
# The template's background is the TeNSORE logo stretched over the slide at 6% opacity
# (alphaModFix). Keynote ignores that opacity and shows it at full strength, so bake the
# fade into a new image and drop the alpha setting. The image part is shared with the
# bottom-right logo, so a new part is added rather than editing the shared one.
import io, os
from PIL import Image
from pptx.oxml.ns import qn

R_EMBED = qn("r:embed")
bg_blip = slide._element.find(".//" + qn("p:bg")).find(".//" + qn("a:blip"))
alpha = bg_blip.find(qn("a:alphaModFix"))
amt = int(alpha.get("amt")) / 100000 if alpha is not None else 1.0
src = slide.part.related_part(bg_blip.get(R_EMBED))
w_px, h_px = 2400, int(2400 * prs.slide_height / prs.slide_width)
logo = Image.open(io.BytesIO(src.blob)).convert("RGB").resize((w_px, h_px), Image.LANCZOS)
faded = Image.blend(Image.new("RGB", logo.size, "white"), logo, amt)
bg_path = os.path.join(os.path.dirname(os.path.abspath(OUT)), "_bg_faded.jpg")
faded.save(bg_path, quality=90)
_, rid = slide.part.get_or_add_image_part(bg_path)
bg_blip.set(R_EMBED, rid)
if alpha is not None:
    bg_blip.remove(alpha)

# The IIT logo is SVG-only; Keynote needs a PNG fallback in the same picture.
iit = shapes["Graphic 14"]._element.find(".//" + qn("a:blip"))
_, rid = slide.part.get_or_add_image_part(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "iit_logo.png"))
iit.set(R_EMBED, rid)

prs.save(OUT)
os.remove(bg_path)
print("saved", OUT, "background opacity baked:", amt)

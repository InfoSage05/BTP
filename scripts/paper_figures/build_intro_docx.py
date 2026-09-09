from docx import Document
from docx.shared import Pt, Inches
from docx.enum.text import WD_ALIGN_PARAGRAPH

doc = Document()
style = doc.styles['Normal']
style.font.name = 'Calibri'
style.font.size = Pt(11)

doc.add_heading('Critical Heat Flux Prediction Using Machine Learning: A Physics-Informed Approach', level=0)
doc.add_heading('1. Introduction', level=1)
doc.add_heading('1.1 Nuclear Relevance of Critical Heat Flux', level=2)

doc.add_paragraph(
    "Nuclear reactors generate enormous amounts of heat inside a very small volume, and the only "
    "thing standing between that heat and a serious accident is the coolant's ability to carry it "
    "away fast enough. In water-cooled reactors, this heat removal happens mostly through boiling: "
    "as water flows past the hot fuel surface, it picks up energy and starts forming vapor bubbles "
    "that carry heat away far more efficiently than the flowing liquid alone. This process, called "
    "nucleate boiling, is remarkably good at removing heat, but only within a limit. Push the heat "
    "flux higher, and there comes a point where the bubbles at the surface stop escaping efficiently, "
    "start merging into an insulating vapor layer, and the wall temperature spikes almost instantly. "
    "This tipping point is the Critical Heat Flux (CHF), and it represents the ceiling on how much "
    "heat a boiling surface can safely handle."
)

doc.add_paragraph(
    "In pressurized water reactors (PWRs), this same phenomenon is usually described as Departure "
    "from Nucleate Boiling (DNB): the fuel surface loses direct contact with the liquid and gets "
    "blanketed by a vapor film. In boiling water reactors (BWRs), where a large fraction of the "
    "coolant is already vapor by the time it reaches the top of the core, the equivalent failure "
    "mode is dryout, where the thin liquid film clinging to the fuel rod simply runs out and the "
    "surface is left dry. Different names, same underlying danger. Once CHF is exceeded, the "
    "heat transfer coefficient collapses by an order of magnitude or more, and the fuel cladding "
    "temperature can rise by hundreds of degrees within seconds. If sustained, this can damage or "
    "fail the fuel cladding, which is precisely the barrier that keeps radioactive material "
    "contained. This is why CHF is not just a thermal hydraulics curiosity; it is one of the "
    "defining safety limits that governs how much power a nuclear reactor is allowed to produce."
)

doc.add_heading('1.2 Why Predicting CHF Matters', level=2)

doc.add_paragraph(
    "Because CHF sets a hard ceiling on reactor operation, being able to predict it accurately, "
    "across the full range of pressures, flow rates, and geometries a reactor might see, is "
    "essential, and the consequences of getting it wrong run in both directions. Underestimate "
    "CHF, and a plant is operated more conservatively than it needs to be, at the cost of power "
    "output and efficiency. Overestimate it, and the margin to a real safety limit shrinks or "
    "disappears, which is obviously the more dangerous mistake."
)

doc.add_paragraph("Accurate CHF prediction is essential in the following areas:")

bullets = [
    "Nuclear reactor fuel cooling. CHF sets the fundamental thermal limit on how much power a fuel "
    "assembly can safely generate before the cooling mechanism itself breaks down.",
    "Pressurized water reactors (PWRs). CHF prediction directly determines the minimum DNB ratio a "
    "plant must maintain during both normal operation and postulated accidents, shaping core design "
    "and operating limits from day one.",
    "Boiling water reactors (BWRs). Dryout prediction plays the equivalent role here, defining the "
    "critical power that a fuel bundle can sustain.",
    "Accident-tolerant cooling concepts. New cladding materials, coatings, and enhanced surfaces "
    "designed to buy operators more time during a loss-of-coolant event all need to be evaluated "
    "against how they shift the CHF limit, so any new design has to be tested against, or predicted "
    "by, a reliable CHF model before it can be trusted in a real core.",
    "High-heat-flux thermal systems. Beyond nuclear reactors, the same underlying physics governs "
    "the limits of any high-heat-flux boiling system, from electronics cooling to industrial "
    "boilers, which makes CHF prediction a problem whose stakes extend well past the nuclear "
    "industry alone.",
]
for b in bullets:
    doc.add_paragraph(b, style='List Bullet')

doc.add_heading('1.3 Limitations of Conventional CHF Correlations', level=2)

doc.add_paragraph(
    "Given how much depends on getting this number right, the natural question is how CHF is "
    "actually predicted in practice today. For decades, the standard tools have been empirical "
    "correlations and look-up tables, such as the widely used 2006 Groeneveld look-up table, built "
    "by fitting mathematical relationships directly to large collections of experimental "
    "measurements. These tools work well within the specific range of pressures, flow rates, "
    "geometries, and fluids they were built from, since they were tuned on exactly that data. The "
    "problem shows up at the edges. Step outside the original database, into a new fluid, an "
    "unusual tube diameter, a different surface condition, or a pressure range with little "
    "experimental coverage, and these correlations can degrade sharply or fail outright, precisely "
    "because they have no underlying mechanistic understanding of the boiling crisis to fall back "
    "on. They interpolate well; they extrapolate poorly. This gap between the conditions a real "
    "reactor might encounter and the narrow window a correlation was actually validated against is "
    "exactly the opening that data-driven and machine learning methods have started to fill, and "
    "it is the motivation behind the growing body of CHF research summarized in Figure 1."
)

doc.add_paragraph()
fig_para = doc.add_paragraph()
fig_para.alignment = WD_ALIGN_PARAGRAPH.CENTER
run = fig_para.add_run()
run.add_picture('docs/manuscript/chf_ml_trends_chart.png', width=Inches(6.3))

caption = doc.add_paragraph()
caption.alignment = WD_ALIGN_PARAGRAPH.CENTER
cap_run = caption.add_run(
    "Figure 1. Reported R-squared accuracy of machine-learning-based CHF prediction models by "
    "publication year (2025 to 2026), including this study's results. Sourced from Yang et al. "
    "(2025), a physics-informed hybrid comparative study (2025), a Bayesian neural network study "
    "(2025), a unit-aware embedding Transformer study (2025), and this study's own pretrained and "
    "fine-tuned MLP and Transformer models (2026). See docs/manuscript/chf_ml_trends_data.csv for "
    "the full sourced dataset."
)
cap_run.italic = True
cap_run.font.size = Pt(9)


doc.add_heading('1.4 Influence of Surface Characteristics', level=2)

doc.add_paragraph(
    "The correlations described above are built almost entirely around bulk flow conditions: "
    "pressure, mass flux, and quality. But boiling does not happen in the bulk fluid. It happens "
    "at the surface, one nucleation site at a time, and the condition of that surface turns out to "
    "shape CHF just as strongly as the flow conditions do. This is why engineered surfaces (surfaces "
    "deliberately roughened, coated, or textured to control how boiling behaves on them) have become "
    "one of the most active areas of CHF research in recent years. Five surface properties in "
    "particular keep showing up across the literature: roughness, wettability, contact angle, "
    "surface material, and orientation. Each one acts through a distinct physical mechanism, and "
    "together they determine when and where a dry spot first forms."
)

doc.add_paragraph(
    "Surface roughness and nucleation site density. A rougher surface has more cavities where "
    "vapor can get trapped and grow into a bubble, so rougher surfaces generally support a higher "
    "density of active nucleation sites and, through that, a higher CHF. Roughness has been reported "
    "to raise the boiling heat transfer coefficient by close to 19 percent on typical roughened "
    "surfaces. But roughness alone is not the whole story: chromium coatings with very low roughness "
    "have still produced CHF increases of 32 to 79 percent through superhydrophilicity alone, and at "
    "the nanoscale, dense structuring can actually work against CHF, because very fine features can "
    "trap vapor so effectively that they interfere with rewetting right when it matters most, near "
    "the CHF point itself. In practice, the surfaces that perform best tend to combine micro-scale "
    "roughness (which drives nucleation and provides a wicking path for liquid) rather than relying "
    "on nanoscale texture alone.",
    style='List Bullet'
)

doc.add_paragraph(
    "Wettability and contact angle. Wettability, measured through the contact angle a liquid makes "
    "with the surface, controls how aggressively liquid is drawn back toward a growing bubble. More "
    "wettable (lower contact angle) surfaces have been shown experimentally to reach higher CHF, and "
    "the mechanism is well characterized: CHF is triggered at a local dry spot where the liquid can "
    "no longer keep the surface wetted, and where it is instead violently scattered by strong "
    "evaporation right at the three-phase contact line. Interestingly, the specific way this failure "
    "happens depends on how wettable the surface is. On less wettable, non-wetting surfaces, a dry "
    "patch becomes unrecoverable once the rate of liquid supplied by capillary action falls below the "
    "rate of evaporation at its edge (an evaporation-limited failure, typically seen with water). On "
    "highly wettable surfaces, the dry patch can instead become unrecoverable once the surface "
    "temperature underneath it crosses the Leidenfrost point, so the liquid film simply cannot "
    "re-establish contact even where it is otherwise available (a levitation-limited failure, more "
    "typical of fluids like FC-72). These are two different physical routes to the same outcome, and "
    "which one applies depends directly on the wettability of the surface in question. As a further "
    "detail worth noting, experiments near the boiling crisis itself have found that the apparent "
    "contact angle tends to sit close to 90 degrees right at the point of triggering, regardless of "
    "the surface's static contact angle far from CHF.",
    style='List Bullet'
)

doc.add_paragraph(
    "Microlayer evaporation, liquid replenishment, and dry spot formation. When a bubble nucleates, "
    "it leaves behind a thin liquid microlayer trapped between the growing bubble and the surface. "
    "Because this microlayer is so thin, heat conducts through it very efficiently, and its rapid "
    "evaporation supplies a large share of the energy needed for the bubble to keep growing. This is "
    "also where the dry spot begins: as the microlayer evaporates, the three-phase contact line "
    "advances outward, and the dry patch underneath the bubble spreads with it. Under normal nucleate "
    "boiling, the bubble eventually detaches and rises under buoyancy, the contact line retreats "
    "back toward the center, and liquid rewets the dry patch before it can do any damage. CHF is "
    "simply the point where this rewetting step fails to keep up. More wettable surfaces support a "
    "larger microlayer area to begin with, and surfaces with micro-scale roughness add a further "
    "advantage: their capillary wicking action pulls in extra liquid to the retreating contact line, "
    "which offsets the evaporation rate and buys the surface more time to rewet before a dry spot can "
    "lock in and spread out of control.",
    style='List Bullet'
)

doc.add_paragraph(
    "Surface material. Beyond its influence on wettability, the material a surface is made of also "
    "affects CHF through its thermal properties. A material with higher thermal effusivity conducts "
    "heat into the thin liquid microlayer more readily, which changes how quickly that microlayer can "
    "evaporate and how the local wall temperature responds as a dry spot begins to form. This is part "
    "of why the same roughness or coating can produce different CHF outcomes on different base "
    "materials: the geometry of the surface and the thermal behavior of the material underneath it "
    "act together, not independently.",
    style='List Bullet'
)

doc.add_paragraph(
    "Surface orientation, bubble departure, and vapor escape. Orientation determines how gravity and "
    "buoyancy act on a growing bubble. On a vertical, upward-facing surface, bubbles detach cleanly "
    "and rise away from the heated wall, clearing the way for liquid to return. On horizontal or "
    "inclined surfaces, bubbles are more likely to slide along the surface, coalesce with neighboring "
    "bubbles, and merge into a continuous vapor layer that is far harder for liquid to break through. "
    "This is the same underlying competition described in classical hydrodynamic instability models "
    "of CHF, where the boiling crisis is framed as a competition between the rate at which vapor can "
    "escape from the surface and the rate at which liquid can flow back in to replace it: orientation "
    "shifts that balance directly, by changing how easily bubbles depart and how quickly the vapor "
    "generated at the surface is able to escape rather than accumulate.",
    style='List Bullet'
)

doc.add_paragraph(
    "Taken together, nucleation site density, bubble departure behavior, liquid replenishment, "
    "microlayer evaporation, and vapor escape are not independent side details. They are the actual "
    "mechanisms that determine when and where a dry spot first appears and whether it grows into full "
    "dryout, and every one of the five surface properties discussed above acts by shifting one or "
    "more of these mechanisms. Any CHF prediction approach that ignores surface condition is, in "
    "effect, ignoring the mechanism itself and hoping that a bulk flow correlation happens to average "
    "it out."
)

doc.save("CHF_Research_Paper.docx")
print("saved")

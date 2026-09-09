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
    "While CHF is most commonly discussed in terms of bulk flow conditions, such as pressure, mass "
    "flux, and quality, the condition of the heated surface itself turns out to matter just as "
    "much, and this is where a large part of recent CHF research has shifted its attention. "
    "Boiling is fundamentally a surface phenomenon: bubbles form at specific sites on the surface, "
    "grow, detach, and are replaced by fresh liquid, and every one of those steps depends on how "
    "the surface interacts with the fluid sitting on top of it."
)

doc.add_paragraph(
    "Surface roughness is one of the clearest examples. A rougher surface offers more cavities and "
    "imperfections where vapor can get trapped and nucleate into bubbles, which increases the "
    "density of active nucleation sites and generally pushes CHF higher, up to a point, beyond "
    "which very rough or re-entrant structures can trap vapor too effectively and actually hurt "
    "performance. Wettability, usually described through the contact angle a water droplet makes "
    "with the surface, controls how readily liquid spreads back over the surface to replace what "
    "is consumed by evaporation. A highly wettable (low contact angle) surface encourages fast "
    "liquid replenishment beneath and around growing bubbles, delaying the point at which a dry "
    "patch forms and stabilizes, which is, mechanistically, what CHF actually is: the onset of a "
    "dry spot that fails to rewet. Surface material affects this too, both through its own "
    "intrinsic wettability and through its thermal properties, which influence how quickly heat is "
    "conducted into the thin liquid microlayer trapped beneath a growing bubble, a layer whose "
    "evaporation is itself a major contributor to the total heat removed during nucleate boiling. "
    "Even surface orientation changes the picture, since gravity and buoyancy affect how easily "
    "vapor bubbles can detach and escape upward versus sliding and coalescing along an inclined or "
    "horizontal surface, which in turn changes how quickly a stable vapor blanket can form."
)

doc.add_paragraph(
    "Taken together, nucleation site density, bubble departure behavior, liquid replenishment, "
    "microlayer evaporation, and vapor escape are not independent side details. They are the "
    "actual mechanisms that determine when and where a dry spot first appears and whether it grows "
    "into full dryout. Any CHF prediction approach that ignores surface condition is, in effect, "
    "ignoring the mechanism itself and hoping that a bulk flow correlation happens to average it out."
)

doc.save("CHF_Research_Paper_NEW.docx")
print("saved")

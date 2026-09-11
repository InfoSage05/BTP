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
    "correlations and look up tables, such as the widely used 2006 Groeneveld look up table, built "
    "by fitting mathematical relationships directly to large collections of experimental "
    "measurements. These correlations work, and they have served the industry for decades, but they "
    "carry a specific set of limitations that are worth naming individually, since each one shapes "
    "exactly where a data driven alternative could actually help."
)

doc.add_paragraph(
    "Empirical nature. Correlations such as the Groeneveld look up table are curve fits to measured "
    "data, not derivations from a physical model of the boiling crisis. The functional form used, "
    "often a power law or a piecewise multiplicative combination of pressure, mass flux, and "
    "quality, is chosen because it fits the available data reasonably well, not because it follows "
    "from the underlying physics of bubble growth, microlayer evaporation, or vapor escape. This "
    "means the correlation carries no built in understanding of why CHF happens, only a description "
    "of what it looked like in the specific experiments it was fitted to.",
    style='List Bullet'
)

doc.add_paragraph(
    "Narrow experimental ranges. Every correlation is only as good as the data it was built from, "
    "and that data was collected within a specific window of pressure, mass flux, quality, and tube "
    "diameter. Even the 2006 Groeneveld look up table, despite being one of the broadest CHF "
    "databases ever assembled, still contains entire regions that are interpolated or extrapolated "
    "rather than measured directly, simply because no experiment has ever been run at every "
    "combination of conditions a reactor could see.",
    style='List Bullet'
)

doc.add_paragraph(
    "Dependence on specific fluids. Correlations built from water data do not transfer cleanly to "
    "other fluids, even when the flow conditions look similar on paper. Pioro et al. (2002) compared "
    "CHF under matched flow conditions for water and for R-134a, a common refrigerant used as a "
    "water surrogate in scaled experiments, and found that the effect of surface orientation on CHF "
    "was measurably stronger for R-134a than for water under otherwise equivalent conditions. A "
    "correlation tuned only on water data has no way of knowing this discrepancy exists, let alone "
    "correcting for it.",
    style='List Bullet'
)

doc.add_paragraph(
    "Dependence on particular geometries. A CHF correlation built for one channel geometry, a round "
    "tube, an annulus, a rod bundle, does not automatically apply to another. The usual fix is a "
    "geometry correction factor, but even this correction is not constant. Tanase et al. (2009) "
    "tabulated a diameter correction exponent for CHF across 24 separate combinations of pressure, "
    "mass flux, and quality, and found the exponent itself ranging from about minus 0.3 to plus 0.6 "
    "depending on which regime a given condition fell into. In other words, there is no single "
    "number that describes how CHF scales with tube diameter, only a patchwork of regime specific "
    "values.",
    style='List Bullet'
)

doc.add_paragraph(
    "Difficulty incorporating surface characteristics. As the next section describes in detail, "
    "conventional correlations are built almost entirely around bulk flow variables, pressure, mass "
    "flux, and quality, and have no natural input slot for surface roughness, wettability, coating, "
    "or material. A correlation can be refit for one specific engineered surface, but it cannot "
    "generalize to a new surface it has never seen, because surface condition was never part of its "
    "functional form to begin with.",
    style='List Bullet'
)

doc.add_paragraph(
    "Poor extrapolation. Because these correlations are fitted rather than derived, they interpolate "
    "well within their original database and extrapolate poorly outside it. Step into a new fluid, "
    "an unusual diameter, or a pressure range with little experimental coverage, and a correlation "
    "can degrade sharply or fail outright, precisely because it has no mechanistic understanding of "
    "the boiling crisis to fall back on once the data runs out.",
    style='List Bullet'
)

doc.add_paragraph(
    "Nonlinear coupling among parameters. CHF does not respond to pressure, mass flux, diameter, and "
    "quality independently. It responds to specific combinations of them, and the way these "
    "parameters interact changes from one flow regime to another. The same Tanase et al. (2009) "
    "diameter exponent data makes this concrete: the exponent describing how CHF scales with "
    "diameter is not one number but 24 different numbers, each valid only for a specific combination "
    "of pressure, mass flux, and quality. Capturing this kind of regime dependent, multi parameter "
    "interaction with a single tidy equation is exactly the sort of problem that empirical "
    "correlations, built around simple multiplicative forms, are poorly suited to represent.",
    style='List Bullet'
)

doc.add_paragraph(
    "This gap between the conditions a real reactor might encounter and the narrow, fluid specific, "
    "geometry specific window a correlation was actually validated against is exactly the opening "
    "that data driven and machine learning methods have started to fill, and it is the motivation "
    "behind the growing body of CHF research summarized in Figure 1."
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


doc.add_heading('1.4 Machine Learning as a Potential Solution', level=2)

doc.add_paragraph(
    "Machine learning offers a genuinely different way of building a CHF model, and it is worth "
    "being precise about exactly which of the seven limitations above it addresses directly, which "
    "it improves only partially, and which it does not solve just by virtue of being machine "
    "learning. Taken point by point against the same list:"
)

doc.add_paragraph(
    "Empirical nature. Machine learning models are also fitted to data rather than derived from "
    "first principles, so in a strict sense they do not remove the empirical nature of CHF "
    "prediction. What changes is the flexibility of the fit. A correlation is locked into a "
    "functional form chosen in advance, usually a power law or a fixed multiplicative combination, "
    "while a neural network can represent a much wider family of shapes, including sharp transitions "
    "and interactions that a person would not think to write down as a single equation. This "
    "project's own models reached an R-squared of 0.963 (MLP) and 0.968 (Transformer) under "
    "interpolation testing, a level of fit that a fixed-form correlation is not built to achieve "
    "across such a wide range of conditions at once.",
    style='List Bullet'
)

doc.add_paragraph(
    "Narrow experimental ranges. Merging many independent datasets into one much larger training set "
    "directly widens the range of conditions a model has actually seen. A collaborator on this "
    "project combined seven independent CHF sources into a single 28,470 row dataset spanning tubes, "
    "annuli, helical coils, and pin-fin surfaces, a far broader coverage than any single "
    "correlation's original database. This genuinely helps, but it is worth being honest about what "
    "it does not do: it pushes the boundary of the covered range outward, it does not remove the "
    "boundary. A model trained on this larger dataset can still only interpolate within it.",
    style='List Bullet'
)

doc.add_paragraph(
    "Dependence on specific fluids. Rather than needing a separate correlation refit for every "
    "fluid, a machine learning model can take fluid properties directly as input features, things "
    "like reduced pressure or other dimensionless groups that describe a fluid's thermodynamic state "
    "relative to its critical point. This project's own feature set is built this way, and the per "
    "domain fine-tuning results, R-squared values ranging from about 0.48 to 0.91 depending on the "
    "fluid and geometry combination, show a single underlying model architecture adapting across "
    "genuinely different fluids, rather than requiring a separate hand-built correlation for each "
    "one.",
    style='List Bullet'
)

doc.add_paragraph(
    "Dependence on particular geometries. The same logic applies to geometry. Instead of a single "
    "diameter correction exponent applied uniformly, a model can take geometry descriptors, "
    "diameter, geometry family, hydraulic diameter, as direct inputs and learn how CHF depends on "
    "them jointly with the flow conditions, rather than through a bolted-on correction factor. The "
    "cross-geometry fine-tuning results from this project's own testing, spanning five different "
    "geometry and fluid combinations, are a direct test of this idea.",
    style='List Bullet'
)

doc.add_paragraph(
    "Difficulty incorporating surface characteristics. Surface descriptors, roughness, contact "
    "angle, coating type, can simply be added as extra columns in the input data, something a bulk "
    "flow correlation has no mechanism for at all. The pin-fin pool-boiling dataset used in this "
    "project's own testing includes exactly this kind of surface-specific information, and a model "
    "trained on it can, in principle, learn how CHF shifts with surface condition directly from the "
    "data rather than needing a person to encode that relationship by hand.",
    style='List Bullet'
)

doc.add_paragraph(
    "Poor extrapolation. This is the point where it is important not to overclaim. Plain machine "
    "learning does not fix poor extrapolation just by being machine learning; in several respects it "
    "can be worse than a classical correlation outside its training range, since a neural network "
    "has no physical constraint forcing it to behave sensibly once it leaves the data it was trained "
    "on. Under a hard pressure-based extrapolation split, this project's own testing found "
    "random-forest and gradient-boosted tree models collapsing to an R-squared of roughly 0.41 to "
    "0.45, and a separate pretrain-and-fine-tune transfer learning study reported a standard neural "
    "network reaching a negative R-squared of minus 6.56 under a comparable pressure-based "
    "extrapolation test. What actually helps is a more deliberate training strategy, not simply "
    "switching from a correlation to a neural network. This project's own model, pretrained on a "
    "large synthetic dataset spanning a wide parameter space and then fine-tuned on real "
    "experimental data with a deliberately held-out high-pressure extrapolation region, reached an "
    "R-squared of 0.916 (MLP) and 0.953 (Transformer) on that held-out extrapolation test, far above "
    "the tree-based results under the same split. Yang et al. (2025) report a similar pattern, "
    "reaching an R-squared of 0.9632 with a physics-informed hybrid approach rather than a plain "
    "data fit. The lesson is specific rather than general: machine learning does not solve "
    "extrapolation automatically, but pretraining, transfer learning, and physics-informed "
    "structure, used deliberately, can measurably narrow the gap in a way that neither a classical "
    "correlation nor a naively trained neural network achieves on its own.",
    style='List Bullet'
)

doc.add_paragraph(
    "Nonlinear coupling among parameters. This is arguably where machine learning has the clearest "
    "structural advantage. A neural network is, by construction, a universal function approximator: "
    "it does not need a person to pre-specify how pressure, mass flux, diameter, and quality "
    "interact, or to split the parameter space into 24 separate regimes the way the Tanase et al. "
    "(2009) diameter exponent table does. It learns the interactions directly from the data. The "
    "practical result in this project's own testing is a single model, not two dozen regime-specific "
    "equations, reaching an R-squared above 0.96 on interpolation testing across the full combined "
    "range of pressure, mass flux, quality, and diameter, without any manual splitting of the "
    "parameter space into separate regimes.",
    style='List Bullet'
)

doc.add_paragraph(
    "None of this means machine learning is a drop-in replacement for physical understanding, or "
    "that it should be trusted blindly outside the range of conditions it has actually seen. But "
    "point by point, it directly addresses five of the seven limitations above, fluid dependence, "
    "geometry dependence, surface characteristics, nonlinear coupling, and to a real extent the "
    "narrow-range problem, offers a genuine but carefully qualified improvement on a sixth, poor "
    "extrapolation, when combined with pretraining and transfer learning rather than used naively, "
    "and is honest, rather than evasive, about the seventh, its own empirical nature. That is the "
    "case, grounded in the results already discussed above and summarized in Figure 1, for treating "
    "machine learning as a genuinely promising direction for CHF prediction, rather than as a "
    "buzzword substitute for the correlations it is trying to improve on."
)


doc.add_heading('1.5 Influence of Surface Characteristics', level=2)

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


doc.add_heading('1.6 Limitations of Current CHF Prediction Approaches', level=2)

doc.add_paragraph(
    "It is worth being direct about something the field does not always state plainly: predicting "
    "CHF accurately across the full range of fluids, geometries, surfaces, and flow regimes that "
    "real engineering applications actually encounter is not a solved problem, no matter how the "
    "headline accuracy numbers in recent papers might read. A model that reports an R-squared of "
    "0.95 or higher is reporting the truth, but that number is almost always measured on a test set "
    "drawn from the same distribution as its training data. What that number does not tell you is "
    "how the same model performs the moment it is asked to generalize beyond that distribution, and "
    "the gap between the two can be enormous."
)

doc.add_paragraph(
    "Interpolation is not extrapolation. When a model is evaluated on a random split of a large, "
    "well-populated dataset, it is effectively being asked to fill in gaps between points it has "
    "already seen many close neighbors of. This is a genuinely easier task than predicting CHF for "
    "a pressure, geometry, or surface condition that lies outside anything in the training data, "
    "and the difference in reported accuracy between the two settings is not a small effect. A "
    "recent pretrain-and-fine-tune transfer learning study found that a standard neural network "
    "achieved a negative R-squared value of minus 6.56 when tested on pressures above a cutoff that "
    "excluded only the highest end of its training range, even though the same architecture scored "
    "well above 0.9 on a random split of the identical dataset. Random-forest and gradient-boosted "
    "tree ensembles, evaluated in this project's own earlier testing under a hard pressure-based "
    "extrapolation split, showed the same pattern, collapsing from near-perfect interpolation scores "
    "to an R-squared of roughly 0.41 to 0.45 once asked to predict outside the pressure range they "
    "were trained on. This happens because tree-based models are fundamentally piecewise-constant: "
    "they can only predict values close to what they have already seen, and have no mechanism for "
    "recognizing that a physical trend should continue smoothly past the edge of their training data.",
    style='List Bullet'
)

doc.add_paragraph(
    "Generalizing across surfaces and sources is harder still. A collaborator on this project "
    "recently assembled a unified, 28,470-row dataset merging seven independent point-level CHF "
    "sources, tubes, annuli, helical coils, and pin-fin pool-boiling surfaces among them, and "
    "evaluated the same model bake-off (a standard neural network, random forest, gradient-boosted "
    "trees, Gaussian process regression, and a physics-informed neural network) under four "
    "increasingly demanding test strategies. On a random stratified split, every model looked "
    "strong, with the best reaching an R-squared of 0.968. On a condition-wise split, where only the "
    "highest-pressure portion of each source was held out, the best models still managed roughly "
    "0.89, but the physics-informed neural network collapsed to an R-squared of minus 64.9, far "
    "worse than simply predicting the average CHF value every time. On a surface-wise split, where "
    "two entire geometries (pin-fin pool boiling and helical coils) were withheld completely rather "
    "than just a condition range, even the best-performing tree models dropped to an R-squared of "
    "about 0.17 to 0.22, while the standard neural network and the physics-informed model both "
    "produced results so unstable that their R-squared values fell below minus 3900. Finally, under "
    "a leave-one-source-out cross-validation (systematically holding out each of the seven sources "
    "in turn and pooling the results), the best model reached an R-squared of about 0.78, a real and "
    "useful number, but still a clear step down from the 0.97 the same approach reported on the "
    "random split of the same underlying data.",
    style='List Bullet'
)

doc.add_paragraph(
    "Physics-informed does not automatically mean robust. One of the more counterintuitive findings "
    "in the results above is that the physics-informed neural network was not the most reliable "
    "model under distribution shift; in several cases it was the least reliable, producing the "
    "single worst results of any model tested. This matters because it is tempting to assume that "
    "adding physical structure to a model automatically buys generalization ability. It does not, "
    "on its own. How the physics is incorporated (as a hard constraint, a soft penalty, a pretrained "
    "prior, or a residual correction) determines whether it actually helps outside the training "
    "distribution, and a poorly chosen physics-informed architecture can perform worse than a "
    "plain data-driven model with no physical structure at all.",
    style='List Bullet'
)

doc.add_paragraph(
    "Taken together, these results point to a conclusion that is easy to state but important not to "
    "gloss over: a single reported R-squared value, without knowing exactly how the test set "
    "relates to the training data, says very little about whether a CHF prediction model is "
    "actually reliable for a new fluid, a new geometry, or a new surface it has never encountered. "
    "Interpolation performance and extrapolation performance are, in effect, two different "
    "questions, and a model that answers one well can fail the other by several orders of "
    "magnitude. Any CHF prediction approach that claims broad, large-scale applicability needs to be "
    "evaluated against genuinely held-out conditions and sources, not just a random split of its own "
    "training database, before that claim can be trusted."
)

doc.save("CHF_Research_Paper.docx")
print("saved")

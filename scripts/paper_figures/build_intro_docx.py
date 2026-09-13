from docx import Document
from docx.shared import Pt, Inches
from docx.enum.text import WD_ALIGN_PARAGRAPH

doc = Document()
style = doc.styles['Normal']
style.font.name = 'Calibri'
style.font.size = Pt(11)


def add_table(doc, headers, data_rows):
    table = doc.add_table(rows=1, cols=len(headers))
    table.style = 'Light Grid Accent 1'
    hdr_cells = table.rows[0].cells
    for i, h in enumerate(headers):
        hdr_cells[i].text = h
        hdr_cells[i].paragraphs[0].runs[0].bold = True
        hdr_cells[i].paragraphs[0].runs[0].font.size = Pt(9)
    for row in data_rows:
        cells = table.add_row().cells
        for i, val in enumerate(row):
            cells[i].text = str(val)
            for p in cells[i].paragraphs:
                for r in p.runs:
                    r.font.size = Pt(9)
    doc.add_paragraph()


def add_caption(doc, text_str):
    cap = doc.add_paragraph()
    run = cap.add_run(text_str)
    run.italic = True
    run.font.size = Pt(9)

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
    "one. This solution carries its own honest caveat, discussed further in Section 1.8: the same "
    "dimensionless representation that lets a model transfer across fluids can also make an unseen "
    "fluid look, numerically, like one the model has already seen, which is a real cost of this "
    "approach and not a minor detail.",
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


doc.add_heading('1.5 Existing ML-based CHF Prediction', level=2)

doc.add_paragraph(
    "With the case for machine learning established, it is worth looking at what has actually been "
    "tried so far. Over roughly the last decade and a half, a fairly consistent set of algorithms "
    "keeps showing up across the CHF literature: artificial neural networks, random forest, gradient "
    "boosting, XGBoost, support vector regression, Gaussian process regression, and deeper neural "
    "network architectures. Reviewing what each of these has actually achieved, including where this "
    "project's own testing produced comparable numbers, gives a clearer picture of where the field "
    "genuinely stands."
)

doc.add_paragraph(
    "Artificial Neural Networks (ANN). A standard fully connected neural network is the most common "
    "starting point in CHF machine learning studies, and it generally performs well on a like-for-"
    "like test. This project's own testing found an ANN reaching an R-squared of 0.921 on a random "
    "split of the collaborator's unified 28,470-row dataset, and Yang et al. (2025) report an "
    "R-squared of 0.9632 with a physics-informed architecture built on the same underlying network "
    "family. That accuracy is not guaranteed to hold up once the test conditions shift: the same ANN "
    "in this project's own testing dropped to an R-squared of 0.515 under a condition-wise split, "
    "and collapsed to an R-squared of roughly minus 4133 once two entire surface types were withheld "
    "under a surface-wise split.",
    style='List Bullet'
)

doc.add_paragraph(
    "Random Forest. A random forest builds many decision trees on random subsets of the data and "
    "features, then averages their predictions, which tends to make it more stable than a single "
    "model. In this project's own testing, random forest reached an R-squared of 0.965 on a random "
    "split, still a respectable 0.875 under a condition-wise split, but only 0.219 once two surface "
    "types were withheld entirely, and 0.269 under a leave-one-source-out test across all seven "
    "sources.",
    style='List Bullet'
)

doc.add_paragraph(
    "Gradient Boosting. Gradient boosting builds trees sequentially, with each new tree correcting "
    "the errors left by the ones before it. In this project's own testing, a gradient-boosted tree "
    "model reached an R-squared of 0.968 on a random split, the best of the five algorithms compared, "
    "and it remained the strongest or tied-strongest model under every one of the four test "
    "strategies used, including an R-squared of 0.784 under the leave-one-source-out test, the "
    "hardest of the four.",
    style='List Bullet'
)

doc.add_paragraph(
    "XGBoost. This specific, heavily optimized implementation of gradient boosting was not part of "
    "this project's own model comparison, but it appears repeatedly in the wider literature. A 2025 "
    "study of CHF prediction in a 5x5 rod bundle assembly, using over 6000 datapoints and fifteen "
    "input features, compared a Transformer, XGBoost, random forest, KNN, SVM, and AdaBoost, and "
    "found XGBoost reaching an R-squared above 0.94, just behind the Transformer's 0.956, while "
    "needing substantially less computation. Its SHAP-based feature ranking in that study identified "
    "heated rod length as the single most influential input.",
    style='List Bullet'
)

doc.add_paragraph(
    "Support Vector Regression (SVR). SVR looks for the flattest function that still fits the "
    "training data within a set error margin, which can make it comparatively resistant to "
    "overfitting on smaller datasets. A 2024 comparison study evaluated a neural network, Gaussian "
    "process regression, and nu-support vector regression on a large CHF database and reported the "
    "predicted-to-measured ratio standard deviation at 12.3 percent for the neural network, about "
    "three times tighter than the 2006 look up table, and 17.7 percent for both GPR and nu-SVR, "
    "about two times tighter than the look up table. A separate study on narrow rectangular channel "
    "CHF, comparing a back-propagation neural network, random forest, SVR, and a long short-term "
    "memory model, found the neural network reaching the lowest prediction error of the four.",
    style='List Bullet'
)

doc.add_paragraph(
    "Gaussian Process Regression (GPR). This project's own testing of GPR tells a cautionary story "
    "about how much an algorithm's usefulness depends on how it is set up. GPR reached an R-squared "
    "of 0.872 on a random split, but only a slightly negative R-squared of about minus 0.05 under a "
    "condition-wise split, meaning it performed no better than simply predicting the average CHF "
    "value once asked to extrapolate to higher pressures. A separate kernel-comparison study reported "
    "that GPR with a properly chosen anisotropic kernel can outperform both ANN and SVR for CHF "
    "prediction, which points to kernel choice, not the algorithm family itself, as the deciding "
    "factor.",
    style='List Bullet'
)

doc.add_paragraph(
    "Deep Neural Networks (DNN). Architectures with more layers and specialized structure, rather "
    "than a single hidden layer, have pushed reported accuracy higher still in recent CHF studies. A "
    "Bayesian neural network study reported an R-squared of 0.986, and a Transformer architecture "
    "using unit-aware input embeddings reached 0.9818 on a large public tube database, both notably "
    "higher than the plain ANN figures above. This is consistent with the overall trend shown earlier "
    "in Figure 1: as architectures have grown deeper and more specialized, reported interpolation "
    "accuracy has climbed fairly steadily over the last few years.",
    style='List Bullet'
)

doc.add_paragraph(
    "Across all seven of these approaches, a consistent and important pattern emerges, one that is "
    "easy to miss if the only number reported is a single headline R-squared. Every one of them "
    "looks strong on a random split of a training-like dataset, and every one of them can lose most "
    "or all of that accuracy once tested outside that distribution, whether that means a higher "
    "pressure range, a different flow condition, or, most severely, a surface type it has never "
    "encountered before. What is largely missing across the great majority of these studies, this "
    "project's own model comparison included, is a systematic test of generalization: results are "
    "almost always reported for a single train-test split, usually random, without checking whether "
    "the model has actually learned the underlying physical relationship or has simply learned the "
    "specific surfaces and conditions represented in its own training data. This gap, a strong focus "
    "on maximizing accuracy over demonstrating genuine generalization and physical consistency across "
    "unseen surface conditions, is exactly the gap the remainder of this project's own testing was "
    "designed to expose and measure directly, rather than assume away."
)


doc.add_heading('1.6 Influence of Surface Characteristics', level=2)

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


doc.add_heading('1.7 Limitations of Current CHF Prediction Approaches', level=2)

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


doc.add_heading('1.8 Research Gap', level=2)

doc.add_paragraph(
    "Given everything discussed above, the specific gaps this project focuses on can be stated "
    "plainly, in four parts, each grounded in a pattern that showed up repeatedly in this project's "
    "own testing rather than assumed in the abstract."
)

doc.add_paragraph(
    "Gap 1: Existing CHF correlations inadequately account for engineered surface characteristics. "
    "As detailed in Section 1.6, roughness, wettability, contact angle, surface material, and "
    "orientation are established, physically important drivers of CHF, yet the correlations "
    "discussed in Section 1.3 were built almost entirely around bulk flow variables, pressure, mass "
    "flux, and quality, and have no input slot for any of them. This omission is not just "
    "structural, it can actively mislead a simple statistical analysis of surface effects: when "
    "roughness and CHF were correlated after pooling data across several engineered-surface studies, "
    "the relationship came out negative, roughness associated with lower CHF, but the same "
    "relationship measured within each individual study came out positive, roughness associated with "
    "higher CHF, matching the capillary-wicking mechanism the source papers themselves describe. "
    "Pooling surface data from different laboratories without accounting for which laboratory each "
    "row came from can reverse the sign of a real physical relationship, not merely blur it.",
    style='List Bullet'
)

doc.add_paragraph(
    "Gap 2: Existing ML models frequently rely on randomly divided datasets, allowing samples from "
    "the same surface or operating condition to appear in both training and testing sets, and the "
    "actual scale of the resulting inflation turned out to be far larger than a single example can "
    "convey. On the smallest of the engineered-surface datasets used in this project, averaging only "
    "about six or seven rows per publication, knowing only which laboratory produced a given row "
    "explained roughly 80.6 percent of the variance in log CHF, more than the roughness and contact "
    "angle values the paper was actually about, which added only a further 4.1 percentage points "
    "once laboratory identity was already known. The same measurement, repeated across five datasets "
    "of increasing size, showed a clean, decreasing pattern: 80.6 percent of variance explained by "
    "source identity alone at roughly 7 rows per publication, falling to 73.5 percent at 81 rows per "
    "publication, 65.9 percent at 11, and down to 34.5 and 27.3 percent on the two largest, "
    "best-populated datasets. A further, more literal version of the same leakage was found hiding "
    "inside individual sources even after grouping by source: 578 rows across this project's merged "
    "dataset turned out to be exact or near-exact duplicates of other rows in feature space, "
    "concentrated overwhelmingly in one compilation where 39 percent of all rows duplicated another "
    "row already in the set, and correcting for this alone flipped that dataset's apparent "
    "random-split score from a positive 0.066 to a negative 0.016. The most severe version of this "
    "problem was not a duplicate row at all: in several of these datasets, the reported flow quality "
    "is itself calculated from the measured CHF value through a standard energy balance, so using it "
    "as a model input lets a model reconstruct the answer arithmetically rather than predict it. "
    "Feeding that quality value back in as an input reproduced CHF on one dataset's 24,443 rows with "
    "an R-squared of 0.999771 using nothing but a textbook energy-balance equation and no fitted "
    "parameters at all, a direct demonstration of how a model can report a near-perfect score while "
    "having learned essentially nothing about CHF. A random split cannot tell any of these three "
    "shortcuts apart from genuine physical learning, because it never asks a model to generalize "
    "past what it has effectively already been shown.",
    style='List Bullet'
)

doc.add_paragraph(
    "Gap 3: Cross-surface generalization remains insufficiently demonstrated. When two entire "
    "surface types, pin-fin pool boiling and helical coils, were withheld completely from training "
    "rather than just a condition range, even the best-performing tree-based model in this project's "
    "own testing dropped to an R-squared of about 0.17 to 0.22, and the standard neural network and "
    "the physics-informed neural network both produced R-squared values below minus 3900. Under a "
    "comparably unseen test on axially non-uniform heating conditions, a closed-form physical "
    "correlation with no fitted parameters at all reached an R-squared of 0.821, outperforming every "
    "machine learning model tested, the best of which reached only minus 0.077. This project's later "
    "testing attempted to address cross-surface generalization directly, using low-rank adapters and "
    "a mixture-of-experts architecture meant to route pool-boiling inputs to a specialized pathway, "
    "but the routing gate itself did not learn that distinction reliably, activating correctly on "
    "only about 11 percent of genuine pool-boiling rows while activating on 61 percent of ordinary "
    "flow rows, close to the opposite of the pattern it was meant to detect. A related and, on "
    "reflection, more fundamental obstacle showed up when this project tried to build a system that "
    "could at least recognize when it was being asked about a genuinely new fluid rather than a new "
    "surface: representing fluids through dimensionless physical groups, the same approach used in "
    "Section 1.4 to address fluid dependence, placed an unseen refrigerant's operating conditions "
    "inside the same numerical region as the water data the model had already trained on, so a "
    "novelty detector built on that representation flagged zero percent of that refrigerant's data "
    "points as unfamiliar, even though the model had never seen that fluid during training. The same "
    "representation that helps a model transfer across fluids and surfaces is, by construction, the "
    "representation that hides the fact that a fluid or surface has changed, and no solution to that "
    "specific tension has yet been found.",
    style='List Bullet'
)

doc.add_paragraph(
    "Gap 4: ML predictions are often treated as black boxes, with limited interpretation of how "
    "surface and thermal-hydraulic parameters control CHF. This project's own experience with a "
    "physics-informed neural network is a direct illustration of how this gap shows up in practice, "
    "not just in principle. The model was originally built around a small, physically motivated set "
    "of three core input features, but extending it to ingest the wider and more varied feature sets "
    "present across different real experimental datasets, so that it could be tested against data it "
    "was not originally designed for, made its internal behavior considerably harder to trace back "
    "to any specific physical mechanism. The interpretability problem runs deeper than any single "
    "model's design: feature-importance rankings computed under an optimistic random split and under "
    "an honest, source-grouped split came out essentially unchanged in this project's own testing, "
    "even though the honest split's actual predictive accuracy was far worse, which means a "
    "feature-importance ranking alone cannot be trusted to reveal whether a model has learned "
    "something real or something that will not survive contact with new data. A related blind spot "
    "showed up in the models' uncertainty estimates rather than their point predictions: an ensemble "
    "of models built specifically to report calibrated 95 percent confidence intervals contained the "
    "true CHF value only 9 to 22 percent of the time when tested on genuinely new data, and even "
    "after switching to a more principled conformal prediction approach, coverage that reached 92 to "
    "96 percent on familiar, in-distribution data fell to 57 percent on a laboratory the model had "
    "never seen. A model that is confidently wrong about its own uncertainty is arguably more "
    "concerning for a safety-relevant quantity like CHF than one that is simply less accurate, "
    "because it gives a false sense of how much a given prediction should be trusted.",
    style='List Bullet'
)

doc.add_paragraph(
    "Beyond these four formal gaps, two practical difficulties stood out repeatedly over the course "
    "of this project's own experiments, and both are worth stating plainly rather than glossing over. "
    "The first is that assembling usable data was, in practice, harder than training any individual "
    "model, and not in a vague sense. Concrete bugs kept surfacing well after the data looked clean: "
    "one exported file had its units row sitting directly beneath the header, so an entire pressure "
    "column was read as the text string \"kPa\" instead of a number until an explicit check caught "
    "it; one dataset's diameter column was silently replaced by a hard-coded 8 millimeters whenever "
    "the real value was missing, and that placeholder went unnoticed for 1,865 rows spanning true "
    "diameters from 1 to 37.5 millimeters, until it was caught by asking why that one dataset was the "
    "pipeline's single worst performer; and a property-lookup routine silently substituted water's "
    "density and latent heat for any fluid it did not recognize, so a request for an unfamiliar fluid "
    "returned a confident, wrong number rather than an error. None of these were modeling failures, "
    "and every one of them would have quietly lowered, or in the diameter case actively corrupted, "
    "whatever a model built on top of that data claimed to have learned. Finding a single "
    "architecture that performed acceptably across every one of these differing feature sets and "
    "geometries was similarly difficult, but it turned out to matter less than getting the data "
    "itself right: on one of the hardest test splits, correcting unit errors and a poorly populated "
    "subcooling column, with no change to the model or any physics assumption, moved that split's "
    "accuracy from an R-squared of 0.173 to 0.711, a bigger jump than any single modeling idea tried "
    "afterward managed on its own."
)

doc.add_paragraph(
    "The second difficulty is the one already illustrated by the numbers throughout this "
    "introduction, and it is worth stating the interpretation directly. Every tree ensemble and "
    "every plain neural network tested in this project's own work reached strong, sometimes "
    "excellent, accuracy under interpolation, R-squared values above 0.9 were common, but the same "
    "models performed far worse, in several cases catastrophically, once tested under extrapolation. "
    "Tree-based models settled around an R-squared of 0.41 to 0.45 under a pressure-based "
    "extrapolation split, while standard neural networks and the physics-informed neural network "
    "produced sharply negative R-squared values under condition-wise and surface-wise splits, and in "
    "one leave-one-publication-out test a plain linear regression outperformed considerably more "
    "sophisticated methods, including gradient-boosted trees and Gaussian process regression, a "
    "further sign that model complexity does not, by itself, buy generalization. A pattern this "
    "consistent, strong performance next to seen data and weak or collapsing performance beyond it, "
    "is hard to read as anything other than evidence that these models are not learning the "
    "underlying physical relationship governing CHF. What they appear to be doing instead is closer "
    "to locating a new point relative to the nearby points they have already seen and interpolating "
    "between them, which works well exactly as long as a new point actually has close neighbors in "
    "the training data, and fails as soon as it does not."
)

doc.add_paragraph(
    "Combining multiple models did not fix this either, which matters because stacking or "
    "ensembling is often the first thing reached for when a single model looks unreliable. Under an "
    "honest, source-grouped evaluation in this project's own testing, a stacked ensemble of several "
    "model types scored an R-squared of 0.459, against 0.945 for a single tree-based model evaluated "
    "the same way, because the ensemble's own combination weights were fit on the training "
    "laboratories and simply learned to trust whichever base model looked best on data it had "
    "already seen. Adding models on top of models does not add physical understanding; it adds "
    "another layer that can itself overfit to source identity."
)

doc.add_paragraph(
    "This project's own attempts at pretraining and fine-tuning, applied to both MLP and Transformer "
    "architectures in an effort to give the models a head start through transfer learning, ran into a "
    "related version of the same problem. Models trained this way performed well on data that "
    "resembled their pretraining distribution, but noticeably worse on genuinely unseen real "
    "experimental data carrying feature combinations the model had not encountered in that particular "
    "combination before. Pretraining and fine-tuning narrowed the interpolation-extrapolation gap "
    "relative to training from scratch, as shown by the improved extrapolation R-squared values "
    "reported earlier in this introduction, but it did not close it."
)

doc.add_paragraph(
    "Part of the underlying difficulty is that many of the input features used across these datasets "
    "are not actually independent of one another. Density, pressure, and the forces acting on a "
    "growing bubble, for instance, are connected through well-established physical relationships, not "
    "coincidence. A model that receives these quantities simply as separate numeric columns has no "
    "built-in way of knowing that such relationships exist. It can only pick up on them indirectly, "
    "if the training data happens to cover enough combinations for the relationship to become visible "
    "in the numbers, and it does not carry that relationship forward reliably into combinations of "
    "conditions it has not seen. In that sense, the models used across this field, including in this "
    "project's own testing, are considerably less sophisticated than they can appear from a single "
    "strong headline accuracy score. They are good at recognizing patterns in data they have already "
    "been shown, and considerably less capable of anything that could reasonably be called physical "
    "reasoning."
)
doc.add_heading('1.9 Objectives', level=2)

doc.add_paragraph(
    "Pulling together everything discussed so far, the aim of this work can be stated simply: to "
    "build a single, general machine learning pipeline for CHF prediction that a practicing engineer "
    "can actually trust and reuse, one that does not need to be redesigned, retrained from scratch, "
    "or handed a bespoke set of input features every time it meets a new fluid, geometry, or surface. "
    "A useful CHF model should behave less like a correlation tuned to one specific dataset and more "
    "like a tool: something that takes whatever thermal-hydraulic and surface information is "
    "available for a given case and returns a prediction that is at least as reliable as, and ideally "
    "considerably better than, the correlations and look-up tables currently in routine use, without "
    "asking the user to understand the details of how it was built."
)

doc.add_paragraph(
    "This objective did not start out this way, and the path that led to it is itself part of the "
    "motivation for this paper. The early phase of this work set out to do what most of the "
    "literature reviewed in Section 1.5 does: maximize a single reported R-squared value. That "
    "effort succeeded on its own terms, this project's own pretrained and fine-tuned models reached "
    "an R-squared of 0.963 (MLP) and 0.968 (Transformer) when interpolating within the range of a "
    "single look-up-table-style dataset. The problem appeared once the same models were tested "
    "against the collaborator's independently assembled, multi-source experimental data described "
    "throughout this introduction. Accuracy dropped sharply and, in several configurations, collapsed "
    "outright, exactly the pattern documented in Sections 1.7 and 1.8. The conclusion was hard to "
    "avoid: a high R-squared on one dataset had been mistaken for a model that understood CHF, when "
    "what had actually happened was a model fitted closely, and somewhat narrowly, to the specific "
    "numerical patterns of that one dataset."
)

doc.add_paragraph(
    "That realization reframed the objective. Rather than continuing to treat CHF prediction as a "
    "straightforward regression problem, feed in pressure, mass flux, and quality, and fit the "
    "output, the goal shifted toward architectures that have some chance of capturing the underlying "
    "physical structure of the problem: pretraining on a wide synthetic parameter space before "
    "fine-tuning on real data, using physically motivated and dimensionless input features rather "
    "than raw values alone, and testing generalization deliberately rather than assuming it. "
    "Creating genuinely new benchmarks along the way turned out to be harder than expected, mainly "
    "because reconciling the different input feature sets, units, and geometry conventions used "
    "across independent CHF datasets, discussed in Section 1.8, is itself a substantial undertaking, "
    "and finding one architecture that behaves consistently well across every one of them remains an "
    "open and only partially solved problem. That open problem is precisely what the remainder of "
    "this paper works on."
)

doc.add_paragraph("With that context, the specific objectives of this paper are:")

objectives = [
    "Develop a single machine learning pipeline for CHF prediction that can be applied across "
    "multiple fluids, geometries, and surface conditions without requiring a bespoke model or a "
    "hand-picked feature set for each new case.",
    "Evaluate that pipeline honestly, using random, condition-wise, surface-wise, and "
    "leave-one-source-out validation strategies side by side, rather than reporting a single "
    "random-split R-squared as if it represented general reliability.",
    "Move the model's behavior closer to physical understanding rather than blind curve fitting, "
    "through pretraining on a broad synthetic parameter space, physically motivated input features, "
    "and transfer learning, and measure directly how much this narrows the gap between interpolation "
    "and extrapolation performance rather than assuming that it does.",
    "Establish a general-purpose CHF prediction approach that a non-specialist user can apply with "
    "reasonable confidence, one whose accuracy has been demonstrated on data it was not trained on, "
    "not just on the dataset it was built from.",
]
for o in objectives:
    doc.add_paragraph(o, style='List Bullet')

doc.add_paragraph(
    "Everything that follows in this paper, the dataset construction, the model architectures, the "
    "validation strategy, and the interpretability analysis, is organized around testing whether "
    "these four objectives can actually be met, not just claimed."
)

doc.add_heading('2. Experimental Database and CHF Measurements', level=1)
doc.add_heading('2.1 CHF Detection Criterion', level=2)

doc.add_paragraph(
    "Before any of the modeling described later in this paper is possible, every experimental CHF "
    "value in the underlying dataset first has to be identified from raw sensor readings, and that "
    "identification step is not as clean cut as it might sound. Different experimental facilities "
    "use different rules for deciding, from a stream of pressure, temperature, heat flux, acoustic, "
    "or image measurements, exactly which data point counts as CHF. The outline for this project "
    "names four criteria, and two more, drawn from the wider CHF literature, are worth adding "
    "because they show up increasingly often in recent experimental and machine-learning-adjacent "
    "work."
)

doc.add_paragraph(
    "Criterion 1, wall temperature excursion. This is the most common general-purpose criterion: "
    "heat flux or heater power is raised gradually while wall temperature is continuously logged, "
    "and CHF is identified as the point where the wall superheat time series shows the onset of a "
    "sustained upward excursion rather than settling to a new steady value, the sign that nucleate "
    "boiling has broken down and the surface is no longer being cooled effectively. Because it is the "
    "excursion itself, not a single instantaneous reading, that defines the point, most facilities "
    "record the full time history of heat flux, wall superheat, pressure, and mass flux around the "
    "event and locate CHF from where that excursion begins, rather than from any one isolated "
    "measurement.",
    style='List Bullet'
)

doc.add_paragraph(
    "Criterion 2, sudden wall-temperature increase. A closely related but more automatable version "
    "of Criterion 1 sets a numerical trip threshold rather than relying on inspection of the "
    "excursion curve after the fact. Rod bundle CHF experiments commonly trigger detection when a "
    "monitored wall temperature rises faster than about 10 degrees Celsius per second, or crosses an "
    "absolute ceiling such as 500 degrees Celsius, whichever comes first. A similar trip criterion "
    "has been used in downward-facing boiling surface experiments relevant to severe accident "
    "research, where a thermocouple reaching a preset value of 160 degrees Celsius within 10 to 15 "
    "seconds of the excursion beginning is used to automatically or manually cut heater power before "
    "the test section is physically damaged. This distinction matters for a dataset: a criterion "
    "built primarily as a safety trip is tuned to protect the equipment first and capture the exact "
    "CHF condition second, so its reported CHF value can carry a small, systematic bias toward "
    "slightly exceeding the true critical point.",
    style='List Bullet'
)

doc.add_paragraph(
    "Criterion 3, heat-flux reduction. In test setups where the heater surface temperature, rather "
    "than the heat flux, is the directly controlled variable, CHF shows up differently. As surface "
    "temperature is increased in steps, heat flux initially increases with it, tracing out the "
    "familiar boiling curve, but at CHF the curve turns over: heat flux stops increasing, and can "
    "even decrease, even as the wall temperature keeps rising. This turnover point in the boiling "
    "curve has been used directly as a CHF criterion in several studies, since it does not require "
    "detecting a temperature excursion at all, only a change in the slope of the heat-flux to "
    "wall-superheat relationship.",
    style='List Bullet'
)

doc.add_paragraph(
    "Criterion 4, visual detection. On transparent or optically accessible test sections, CHF can be "
    "identified directly by observing the boiling process itself rather than inferring it from a "
    "temperature or heat-flux signal. Total reflection imaging can detect the dry patches that form "
    "underneath vapor bubbles on a transparent heated surface, and high-speed and infrared imaging "
    "have been used to track dry-spot growth and bubble coalescence frame by frame as CHF is "
    "approached. This approach has also produced a finding that complicates the purely thermal "
    "criteria above: direct visual observation in water has found that irreversible dry spots first "
    "form at a surface temperature of around 134 degrees Celsius, well below the Leidenfrost "
    "temperature that a simple thermal criterion would predict, a reminder that a temperature-based "
    "definition of CHF is a practical proxy for the underlying physical event, not a first-principles "
    "description of it.",
    style='List Bullet'
)

doc.add_paragraph(
    "Criterion 5, acoustic emission. Bubble nucleation, growth, and collapse generate small, rapid "
    "pressure fluctuations in the liquid, and these can be picked up non-intrusively by a "
    "microphone or transducer mounted outside the test section rather than in contact with the "
    "heated surface at all. As boiling approaches CHF, the character of this acoustic signal "
    "changes in ways that correlate with the transition from nucleate boiling to film boiling, and "
    "recent work has combined externally measured acoustic emission signals with deep learning "
    "models to detect CHF directly from sound, tracked continuously from natural convection through "
    "to the critical point. This criterion is particularly useful where thermocouples or optical "
    "access are impractical, for example in harsh or enclosed industrial equipment.",
    style='List Bullet'
)

doc.add_paragraph(
    "Criterion 6, image-based machine-learning classification. This criterion is a direct "
    "extension of Criterion 4 rather than a wholly separate physical signal: instead of a person or "
    "a fixed image-processing rule identifying dry-patch formation from high-speed video, a trained "
    "deep learning model is used to classify boiling-regime images or bubble morphology frame by "
    "frame and flag the onset of CHF automatically. Reported results show that such models can "
    "correlate bubble morphology with the actual measured heat flux even when the resolution or "
    "frame rate of the images is reduced, which makes this approach attractive as a way of "
    "processing very large volumes of high-speed video without a human reviewing every frame by "
    "eye, and it is the one criterion on this list that is itself a small CHF prediction model "
    "rather than a fixed physical rule.",
    style='List Bullet'
)

doc.add_paragraph(
    "These six criteria do not always identify exactly the same physical instant, and switching "
    "between them, even applied to the same raw data, can shift the reported CHF value by a small "
    "but nonzero amount. Because the dataset used later in this paper is merged from multiple "
    "independent sources, each of which may have used a different one of these four criteria, this "
    "is itself a real, concrete source of the label noise and cross-source inconsistency discussed "
    "elsewhere in this paper, and one more reason why a CHF prediction model needs to be evaluated "
    "with some tolerance for the fact that its training labels were not all produced by an identical "
    "measurement procedure. The two additional criteria are also worth flagging for a different "
    "reason: a model trained to detect CHF from acoustic or image data is, in effect, a separate "
    "machine learning system sitting upstream of the CHF prediction models discussed later in this "
    "paper, and any bias or error in that upstream detector becomes part of the ground truth the "
    "downstream CHF model is trained against, whether or not that error is ever reported."
)

doc.add_heading('3. Machine-Learning Methodology', level=1)
doc.add_heading('3.1 Data Preprocessing', level=2)

doc.add_paragraph(
    "Merging independent CHF datasets into a single modeling table, described in Section 2, does "
    "not by itself produce something a model can be trained on safely. The merged schema used in "
    "this project carries 51 columns once every source's fields are combined, and getting from that "
    "raw merge to a trustworthy training set involved five distinct kinds of preprocessing work: "
    "deciding what to do about missing values, deciding which unusual points were real and which "
    "were errors, choosing how to scale the inputs and the target, checking which features actually "
    "carried independent information, and working out what the effective dimensionality of the "
    "problem really was."
)

doc.add_paragraph(
    "Missing-value treatment. Column coverage across the merged dataset is extremely uneven. Core "
    "flow variables such as CHF itself, mass flux, and geometry family are present on effectively "
    "every row, heated length, quality, and pressure are present on more than 99 percent of rows, "
    "and diameter is present on 98.3 percent, but surface-specific columns such as roughness factor "
    "and surface material are present on well under 1 percent of rows, simply because the great "
    "majority of sources never measured them. The rule adopted throughout this project is to never "
    "impute a value a source does not report, rather than filling a gap with a column mean, a "
    "model-based estimate, or any other guess. This matters because two different kinds of "
    "missingness are easy to confuse: a pool-boiling dataset has no mass flux or quality because "
    "there is no forced flow to measure, which is a physically meaningful absence, not missing data "
    "in the usual sense, while a flow-boiling paper that simply never reported subcooling is missing "
    "data that does, in principle, exist. Only the second kind is a candidate for careful imputation "
    "at all, and even there this project chose to leave the gap as a genuine missing value rather "
    "than invent a number, so that no model result can be quietly resting on an assumption dressed "
    "up as a measurement.",
    style='List Bullet'
)

doc.add_paragraph(
    "Outlier detection. The instruction not to remove outliers blindly is worth taking seriously, "
    "because this dataset contains at least three genuinely different situations that a single "
    "statistical outlier rule would treat identically and should not. First, CHF itself spans three "
    "orders of magnitude, from about 15 to over 44,000 kW per square meter, so a small number of "
    "very high or very low values are real physical outcomes at extreme pressure, mass flux, or "
    "subcooling combinations, not errors, and a naive standard-deviation filter would delete exactly "
    "the high-CHF conditions that matter most for margin calculations. Second, some points are "
    "genuinely suspect due to how they were collected rather than what they measure: one dataset "
    "used in this project was digitized by eye from a published scatter plot, which carries an "
    "estimated 5 to 10 percent error on both axes, and this project's response was not to delete "
    "those points but to keep them clearly labeled as a lower-trust source used only for "
    "out-of-distribution testing, never for training. Third, some apparent outliers are unresolved "
    "disagreements in the literature itself: one pool-boiling study's own published text gives CHF "
    "values for its coated surfaces that do not match the range quoted for the same study in a later "
    "review paper, and rather than guessing which source was right, the values were kept exactly as "
    "printed in the original paper and the discrepancy was documented rather than silently resolved. "
    "A fourth case is not an outlier at all but a data-entry defect masquerading as one: a diameter "
    "column that had silently defaulted to a fixed placeholder value whenever the true diameter was "
    "missing was only caught because the affected dataset was the pipeline's worst performer, and "
    "the fix there was to correct the underlying value, not to remove the rows.",
    style='List Bullet'
)

doc.add_paragraph(
    "Normalization and standardization. Because CHF spans three orders of magnitude, every model in "
    "this project is trained on the logarithm of CHF rather than the raw value, with predictions "
    "exponentiated back before any accuracy metric is computed. For the input side, an early "
    "baseline simply standardized the raw physical variables to zero mean and unit variance before "
    "fitting, which is a reasonable default but treats pressure, mass flux, and fluid identity as "
    "arbitrary numbers to be rescaled rather than physical quantities with known relationships to "
    "each other. The approach carried into the final pipeline instead collapses the fluid and flow "
    "variables into a small set of dimensionless physical groups, an inverse Weber number, a "
    "density ratio, a length-to-diameter ratio, a subcooling ratio, and a reduced pressure, so that "
    "water at 15 megapascals and a refrigerant at 1 megapascal that are in physically equivalent "
    "states map to the same point in the input space. This is not just a scaling convenience: tested "
    "directly, moving from raw standardized inputs to this dimensionless formulation improved "
    "cross-fluid prediction error by a factor of roughly 10 to 32, while leaving within-fluid "
    "accuracy essentially unchanged, 0.851 against 0.849 R-squared, which is the control that shows "
    "the gain is coming from better cross-fluid normalization and not from some unrelated side "
    "effect.",
    style='List Bullet'
)

doc.add_paragraph(
    "Feature correlation analysis. Run naively on data pooled from many independent laboratories, a "
    "feature correlation analysis in this project turned out to be actively misleading rather than "
    "simply noisy. Computed across pooled engineered-surface data, the correlation between surface "
    "roughness and CHF came out negative, but computed within each individual study it came out "
    "positive, matching the physical mechanism the source papers themselves describe. Extending the "
    "same kind of analysis to ask how much variance a purely administrative variable, which "
    "laboratory produced a given row, explains on its own, rather than only checking physical "
    "features against the target, is what first revealed that source identity alone can account for "
    "the majority of the apparent signal in a small dataset. Correlation analysis was also used "
    "directly as a leakage-detection tool rather than only a feature-selection one: several columns "
    "in the merged data, including one combination of electrical power, perimeter, and flow area in "
    "one dataset, and a heater-power column in another, reconstructed the CHF target almost exactly "
    "through simple algebra once checked, and were excluded once that reconstruction was confirmed. "
    "In this project, feature correlation analysis removed at least as much as it selected.",
    style='List Bullet'
)

doc.add_paragraph(
    "Dimensionality analysis. The physical problem itself is low-dimensional: CHF in flow boiling is "
    "conventionally described by four or five variables, pressure, mass flux, quality, diameter, and "
    "heated length, plus fluid identity, so this project never faced the classic high-dimensional "
    "setting that would call for something like principal component analysis. The dimensionality "
    "problem that did show up came from merging, not from the physics: once seven independent "
    "sources were combined, the resulting schema carried 51 columns, because each source contributed "
    "its own geometry- or surface-specific descriptors, and the great majority of those columns are "
    "populated on well under 10 percent of rows. The practical dimensionality question in this "
    "project was therefore not how to compress a rich feature space, but how to decide, column by "
    "column, which of these sparsely populated fields could be used at all without either discarding "
    "most of the dataset by requiring every column to be present, or inventing values for rows that "
    "never measured that quantity in the first place. The dimensionless feature construction "
    "described above served this purpose as well, replacing a long, sparse tail of source-specific "
    "raw columns with a small, fixed set of variables that apply to every row regardless of which "
    "source it came from.",
    style='List Bullet'
)

doc.add_paragraph(
    "None of these five preprocessing decisions were free of consequence, and it is worth restating "
    "the finding from Section 1.8 in this context: on one of the hardest test splits used in this "
    "project, correcting a handful of data-preprocessing issues, unit errors and a poorly populated "
    "column, with no change to the model or to any physics assumption, moved that split's accuracy "
    "from an R-squared of 0.173 to 0.711. In this project, the preprocessing choices described above "
    "turned out to matter at least as much as which model was ultimately trained on the result."
)

doc.add_heading('3.2 ML Models', level=2)

doc.add_paragraph(
    "The outline for this paper asks for a comparison of at least five algorithms. This project "
    "ended up testing considerably more than five, across several distinct phases of "
    "experimentation, and it is more honest to describe the five headline models below alongside a "
    "summary of everything else that was tried and set aside, rather than presenting only a tidy "
    "five-way comparison as if it were the whole story."
)

doc.add_paragraph(
    "Model 1, Artificial Neural Network (ANN / MLP). A fully connected multilayer perceptron, "
    "trained on the logarithm of CHF rather than the raw value, was this project's primary "
    "nonlinear baseline throughout every phase of the work, and a Transformer encoder variant was "
    "added later specifically to test whether attention-based architecture offered any advantage "
    "over a plain feedforward network. The earliest test of this model, on the 2006 Groeneveld "
    "look-up table alone, already showed the pattern that shaped the rest of the project: excellent "
    "accuracy under an easy split, and a sharp decline under a genuinely hard one. Table 1 shows "
    "this alongside three other baselines evaluated the same way, using three splits of increasing "
    "difficulty: Split A, an ordinary random 80/20 split; Split B, an interior pressure holdout; and "
    "Split C, an edge-of-range pressure extrapolation, described further in Section 3.4."
)

add_caption(doc, "Table 1. Early baseline comparison on the 2006 Groeneveld look-up table (11,088 usable rows), before any real experimental data was introduced. MAPE is mean absolute percentage error, computed on Split C.")
add_table(doc,
    ["Model", "Split A (R2)", "Split B (R2)", "Split C (R2)", "MAPE on Split C"],
    [
        ["Grid interpolation (physical baseline)", "N/A", "0.999", "0.841", "20.8%"],
        ["Degree-2 log ridge regression", "0.847", "0.871", "0.755", "35.8%"],
        ["Neural network, ANN (log target)", "0.991", "0.989", "0.628 +/- 0.072", "40.0%"],
        ["Tree ensembles (Random Forest, gradient boosting variants)", "0.999", "0.988", "about 0.43", "about 42%"],
        ["Pressure-gated blend (raw target)", "0.999", "0.999", "0.466 +/- 0.228", "67.1%"],
    ]
)

doc.add_paragraph(
    "Two things from this early phase carried forward into everything that followed. First, "
    "deterministic grid interpolation, simply reading the value off the look-up table itself with "
    "no learning involved, held its accuracy far better than any learned model under genuine "
    "extrapolation, the first sign that a physical anchor might matter more than model choice. "
    "Second, an internal review had initially ranked a different blend first, at an R-squared of "
    "0.855 on Split C, but multi-seed verification showed that result did not replicate, the "
    "underlying model averaged 0.547 across 30 random seeds and not one seed reached 0.84 on its "
    "own. The claim was withdrawn, and from that point on, every number in this project is reported "
    "as a multi-seed mean, never a single run."
)

doc.add_paragraph(
    "Model 2, Random Forest. A random forest builds a large number of decision trees on randomly "
    "resampled subsets of the data and features, then averages their predictions, which the outline "
    "correctly identifies as useful for capturing nonlinear relationships, ranking feature "
    "importance, and providing a fairly robust baseline that rarely fails outright. This project's "
    "own testing on the seven-source, 28,470-row merged dataset confirmed the first two points and "
    "complicated the third: random forest was consistently strong under easier splits but degraded "
    "considerably faster than gradient boosting once the test conditions became genuinely unseen."
)

doc.add_paragraph(
    "Model 3, Gradient Boosting. Gradient boosting builds trees sequentially, with each new tree "
    "correcting the errors left by the ones before it, and the outline specifically names XGBoost as "
    "a strong candidate for structured, tabular experimental data such as this project's own. The "
    "implementation actually trained and evaluated in this project's own bake-off was a histogram-"
    "based gradient boosting model rather than the XGBoost package specifically, and it is worth "
    "being precise about that distinction rather than blurring the two together. Section 1.5 already "
    "reviews XGBoost results from the wider published literature, where it reaches an R-squared above "
    "0.94 on rod-bundle CHF data, close behind a Transformer architecture's 0.956. Within this "
    "project's own testing, the gradient boosting model it did train turned out to be the strongest "
    "or joint-strongest performer of every model compared, under every one of the four validation "
    "strategies used, as Table 2 shows."
)

add_caption(doc, "Table 2. Model comparison across four validation strategies, evaluated on the same seven-source, 28,470-row merged dataset (includes pin-fin and helical-coil pool-boiling surfaces). Strategies are defined in Section 3.4.")
add_table(doc,
    ["Model", "Strategy 1: Random (R2)", "Strategy 2: Condition-wise (R2)", "Strategy 3: Surface-wise (R2)", "Strategy 4: Leave-one-source-out (R2)"],
    [
        ["Gradient boosting (histogram-based)", "0.968", "0.890", "0.173", "0.784"],
        ["Random Forest", "0.965", "0.875", "0.219", "0.269"],
        ["Neural network (ANN)", "0.921", "0.515", "-4132.9", "-5.150"],
        ["Gaussian Process Regression", "0.872", "-0.051", "-0.813", "0.539"],
        ["Physics-informed neural network (PINN)", "0.876", "-64.9", "-3906.9", "-6.022"],
    ]
)

doc.add_paragraph(
    "The physics-informed neural network in Table 2 deserves a brief, honest note, since this "
    "project spent considerable effort on it before setting it aside. Early results looked "
    "promising when the target was reparameterized as a residual correction, reaching an R-squared "
    "of 0.847 against a deterministic baseline of 0.841. On closer inspection, however, the "
    "monotonicity penalty that was supposed to enforce physical behavior read exactly zero "
    "throughout training, meaning it was never actually active, and the best-performing "
    "configuration had a fitted weight of zero on both of its physics penalty terms. In other "
    "words, the apparent gain came from how the target was framed, not from any physics constraint "
    "actually doing work, and Table 2 shows the same model collapsing to an R-squared of minus 64.9 "
    "and minus 3900 under harder splits. Combined with the fact that CHF has no governing partial "
    "differential equation for a physics-informed network to enforce in the first place, this "
    "project moved away from the physics-informed neural network as a standalone architecture and "
    "toward the residual-correction-on-a-correlation approach described under Model 5 below, which "
    "keeps the physical anchor but does not depend on an unstable penalty term."
)

doc.add_paragraph(
    "Model 4, physics correlation baseline (zero fitted parameters). Alongside the four learned "
    "models above, this project also carried a purely physical model through every comparison: an "
    "empirical CHF correlation, such as Katto and Ohno's, the Biasi correlation, or the 2006 "
    "Groeneveld look-up table itself, chosen automatically for whichever training subset was being "
    "evaluated, with no fitted parameters of its own. In a later, more tightly scoped phase of this "
    "project, restricted specifically to flow boiling and using a different, eight-source, "
    "28,582-row dataset than Table 2, this physics-only baseline was compared against several "
    "learned models under the same four validation strategies, shown in Table 3."
)

add_caption(doc, "Table 3. Physics-only baseline versus learned models, flow-boiling-only dataset (eight sources, 28,582 rows), same four validation strategies as Table 2.")
add_table(doc,
    ["Model", "Strategy 1: Random (R2)", "Strategy 2: Condition-wise (R2)", "Strategy 3: Surface-wise (R2)", "Strategy 4: Leave-one-source-out (R2)"],
    [
        ["Physics only (Katto-Ohno), no learning", "0.779", "0.508", "0.505", "0.785"],
        ["No physics, raw features (data-driven only)", "0.979", "0.903", "-17.997", "0.719"],
    ]
)

doc.add_paragraph(
    "The pattern is the same one seen throughout this paper: the purely data-driven model wins "
    "comfortably when the split is easy and can collapse badly when it is not, in this case to an "
    "R-squared of minus 18 under a surface-wise split, while the physics-only baseline never scores "
    "as high but never collapses either, staying within a narrow band of 0.5 to 0.79 across every "
    "split tested. That stability, not raw accuracy, is exactly what a physical floor is meant to "
    "provide, and it is the reason Model 4 was kept as a component of Model 5 rather than replaced "
    "by it."
)

doc.add_paragraph(
    "Model 5, Proposed hybrid architecture. Rather than choosing between a data-driven model and a "
    "physics correlation, the model proposed in this paper combines both, using the physics "
    "correlation as a floor and a learned model only to correct it. The full hierarchy is described "
    "in Section 3.3. Evaluated on the same entirely unseen datasets used to test generalization "
    "elsewhere in this paper, this hybrid consistently outperformed both of its own ingredients "
    "taken alone, as Table 4 shows."
)

add_caption(doc, "Table 4. Performance on entirely unseen datasets: physics alone, a purely data-driven model with no physics, and the proposed hybrid pipeline, evaluated dataset by dataset (flow-boiling-only data, eight sources).")
add_table(doc,
    ["Unseen dataset", "Physics alone (R2)", "ML, no physics (R2)", "Proposed hybrid (R2)", "Within +/-10%"],
    [
        ["KAERI non-uniform", "0.797", "0.841", "0.970", "47.5%"],
        ["NRC", "0.926", "0.242", "0.944", "47.0%"],
        ["Hardik straight tube", "0.636", "-868.6", "0.793", "33.8%"],
        ["KAERI uniform", "0.620", "0.852", "0.735", "58.9%"],
        ["Hardik helical coil", "-0.161", "-1.214", "0.687", "29.7%"],
        ["Helical R-123", "0.051", "-307.7", "0.449", "14.8%"],
        ["Pioro R-134a", "-0.097", "-423.6", "-0.065", "16.0%"],
        ["Zhao 2020", "-0.437", "-1.433", "-0.429", "7.7%"],
        ["Median across all eight", "+0.336", "-1.323", "+0.711", "31.8%"],
    ]
)

doc.add_paragraph(
    "The headline result in Table 4 is not the median score by itself, it is the absence of "
    "catastrophe. An unanchored data-driven model reaches an R-squared of minus 868 on one unseen "
    "dataset, a result with no practical meaning beyond signaling total failure, while the proposed "
    "hybrid's worst case across all eight unseen datasets is minus 0.429. A model that is "
    "occasionally mediocre but never that catastrophic is a fundamentally different, and for a "
    "safety-relevant quantity like CHF, considerably more useful thing than a model that is usually "
    "excellent but occasionally worthless."
)

doc.add_paragraph(
    "Other models and configurations explored. The five models above are the headline comparison, "
    "but they are far from everything this project tried. Across several rounds of hyperparameter "
    "tuning and architecture search, involving roughly thirty configurations in the earliest phase "
    "alone and a further 720-configuration search specifically for the physics-informed network, a "
    "considerable number of ideas were tested and set aside once they failed to improve genuinely "
    "held-out accuracy. Table 5 summarizes the ones most worth recording, precisely because a "
    "negative result, tested honestly, is still useful evidence about what does not work for this "
    "problem."
)

add_caption(doc, "Table 5. Additional models, hyperparameter variations, and architectural ideas tested over the course of this project, beyond the five headline models above.")
add_table(doc,
    ["Technique tried", "What it was meant to test", "Outcome"],
    [
        ["Linear / Ridge regression", "A simple baseline, expected to be outperformed", "In one leave-one-publication-out test, plain linear regression beat gradient boosting, XGBoost, and Gaussian process regression; kept as a standing baseline"],
        ["k-Nearest Neighbors (k=3, distance-weighted)", "A grid-memorization diagnostic, not a serious final model", "Useful for showing how much of a dataset's apparent accuracy comes from simply having a close neighbor nearby"],
        ["Degree-2 / degree-4 polynomial ridge", "A simple nonlinear baseline on standardized inputs", "Reasonable inside the training range, error grows sharply outside it once expanded to higher-degree terms"],
        ["Stacked ensemble of multiple model types", "Whether combining models improves robustness", "Scored an R-squared of 0.459 against 0.945 for a single tree model under honest, source-grouped evaluation; the ensemble's own weights overfit to the training sources"],
        ["Physics-informed neural network (PINN)", "Whether enforcing physical penalties during training improves generalization", "Apparent gains traced to target reparameterization, not the physics penalty, which was inactive; catastrophic instability under distribution shift (see Table 2); not carried forward as a standalone model"],
        ["Deep ensembles for uncertainty", "Calibrated confidence intervals from ensemble spread", "95%-nominal intervals contained the true value only 9 to 22% of the time on unseen data; replaced with conformal prediction"],
        ["Pretraining and fine-tuning (MLP and Transformer)", "Transfer learning as a head start for real-data fine-tuning", "Improved extrapolation R-squared over training from scratch, but pooling all fine-tuning domains together underperformed separate per-domain fine-tuning in 8 of 10 comparisons"],
        ["LoRA (low-rank adapters)", "Cheaper fine-tuning with fewer trainable parameters", "Reached an R-squared of 0.685 using only 8.8% of parameters trainable, close to full fine-tuning's 0.696"],
        ["Mixture-of-Experts routing", "Route pool-boiling and flow-boiling inputs to specialized sub-models", "The routing gate did not learn the distinction reliably; abandoned in favor of a single pooled correction (Section 3.3)"],
        ["Inverse dataset-size weighting", "Reduce the influence of one very large source on training", "Median R-squared fell from 0.777 to 0.767; rejected"],
        ["Absolute-error loss instead of squared error", "Match the loss function to the reported accuracy metric", "Median R-squared fell to 0.700; rejected"],
        ["Correlation stacking (Biasi / LUT as extra features)", "Give the model a physics correlation's output as an input", "Median R-squared fell to 0.727; rejected"],
        ["Missing-value indicator column", "Flag rows with incomplete data explicitly", "Made results worse (0.660 vs 0.727); the indicator effectively leaked fluid identity, since it fires only on non-water rows"],
        ["Weber number on diameter / boiling-number target", "Reformulate inputs or target in the correlation literature's native dimensionless form", "Median R-squared 0.731 and 0.703 respectively; both rejected in favor of the simpler dimensionless formulation in Section 3.3"],
        ["Backbone ensembling across physics correlations", "Avoid betting on a single physics correlation", "Helped one dataset, hurt another; no single global choice served both, so a single best-correlation rule was kept instead"],
    ]
)

doc.add_paragraph(
    "The one modification out of eleven tested in this particular round that was actually adopted "
    "was not a modeling idea at all: it was correcting a fabricated placeholder value in one "
    "dataset's diameter column, already described in Section 1.8. Across this entire project, that "
    "pattern held up more often than not: fixing what the data actually said tended to matter more "
    "than which model, loss function, or architecture was used to fit it."
)

doc.add_heading('3.3 Proposed Model Hierarchy', level=2)

doc.add_paragraph(
    "The model proposed in this paper, referred to as Model 5 above, is not a single network but a "
    "hierarchy of five layers, each one added specifically because an earlier, simpler design failed "
    "in a documented way somewhere in this paper. Table 6 lists the five layers in order."
)

add_caption(doc, "Table 6. The proposed model's layer hierarchy, from raw inputs to a final prediction with an uncertainty interval.")
add_table(doc,
    ["Layer", "Component", "Purpose"],
    [
        ["Layer 0", "Fluid properties and dimensionless groups", "Converts fluid name and pressure into density, surface tension, and latent heat, then collapses the inputs into a small set of dimensionless numbers (inverse Weber number, density ratio, length-to-diameter ratio, subcooling ratio, reduced pressure), so the model never sees a raw fluid label"],
        ["Layer 1", "Physics backbone (zero fitted parameters)", "An empirical correlation such as Katto-Ohno, Biasi, or the 2006 look-up table, selected automatically by whichever has the lowest error on the training rows only; this is the floor that prevents catastrophic failure"],
        ["Layer 2", "Pooled learned correction", "A single tree-based model predicting the logarithm of the ratio between measured CHF and the Layer 1 backbone prediction, trained once on all sources pooled together rather than one model per dataset"],
        ["Layer 3", "Trust-scaled correction limit", "Scales how far the Layer 2 correction is allowed to move the Layer 1 prediction: full trust allows up to a ten-fold correction, low trust clips the correction to one and a half times, so an unfamiliar input decays gracefully back toward the physics baseline rather than the learned correction running unchecked"],
        ["Layer 4", "Conformal prediction interval and regime label", "Produces an uncertainty interval calibrated on a held-out calibration slice rather than on ensemble spread, and labels each prediction as interpolating, extrapolating, or falling back to the correlation alone"],
    ]
)

doc.add_paragraph(
    "This hierarchy exists specifically because the alternative, simpler designs tested earlier in "
    "this paper did not hold up. A router that tried to classify an input as familiar or unfamiliar "
    "and send it to either a learned model or the physics baseline was tested and rejected: even its "
    "best version reached only a median R-squared of 0.136 on entirely unseen datasets, worse than "
    "the physics baseline alone at 0.336, because the routing gate could not reliably tell an unseen "
    "fluid apart from a familiar one, the same fluid-identity blind spot discussed in Section 1.8. "
    "Layer 3 replaces that router with a dial rather than a switch: trust scales how much the "
    "correction is allowed to move the prediction, but it never chooses between two entirely "
    "different models the way a router does, so a wrong trust estimate degrades the result gradually "
    "rather than catastrophically."
)

doc.add_paragraph(
    "Layer 2's correction is deliberately pooled across all sources rather than trained separately "
    "per dataset or routed through a mixture-of-experts gate, because both alternatives were tried "
    "first and both underperformed a single pooled model, as Table 5 already summarizes. Layer 2 "
    "can, however, be adapted efficiently to a new domain using low-rank adapters rather than full "
    "retraining, reaching an R-squared of 0.685 while updating only 8.8 percent of its parameters, "
    "against 0.696 for updating all of them, which matters in practice because it makes fine-tuning "
    "the correction on a new fluid or surface considerably cheaper without giving up much accuracy."
)

doc.add_paragraph(
    "The inference procedure built on top of this hierarchy also encodes several of this paper's own "
    "findings as explicit refusals rather than silent extrapolation: it declines to produce a "
    "prediction for zero mass flux, since that is pool boiling and the physics backbone here is not "
    "valid there, for negative or physically impossible geometry, or for a fluid the underlying "
    "property routine does not recognize, closing off exactly the kind of silent, confident, wrong "
    "answer described as a property-lookup failure in Section 1.8."
)

doc.add_heading('3.4 Data Splitting Strategy', level=2)

doc.add_paragraph(
    "A large part of this paper is really an argument about how a CHF model should be evaluated, "
    "and that argument starts with a plain warning worth stating in the outline's own terms: do not "
    "rely only on an ordinary 80 percent training, 20 percent testing random split. If the same "
    "surface, the same laboratory, or even the same narrow condition range has rows in both the "
    "training set and the testing set, a model can score very well by learning the characteristics "
    "of that specific surface or source rather than the underlying physics of CHF, and a random "
    "split has no way of preventing this, since it does not look at where a row came from before "
    "deciding which side of the split it lands on. Every result in Table 2, Table 3, and Table 4 "
    "above exists specifically to demonstrate how large that effect actually is, and this section "
    "collects the split definitions themselves in one place."
)

doc.add_paragraph(
    "This project used two related sets of splits at different stages. The earliest phase, working "
    "only with the 2006 look-up table, used three splits of increasing severity, summarized in "
    "Table 7."
)

add_caption(doc, "Table 7. Split definitions used in this project's earliest phase, on the 2006 Groeneveld look-up table alone.")
add_table(doc,
    ["Split", "Construction", "Question it answers"],
    [
        ["Split A", "Random 80/20", "Interpolation inside the grid (the optimistic case)"],
        ["Split B", "Interior pressure holdout", "A moderately difficult, still largely interpolative case"],
        ["Split C", "Edge-of-range pressure extrapolation", "The honest test of whether a model generalizes at all"],
    ]
)

doc.add_paragraph(
    "Once real, multi-source experimental data replaced the look-up table as the main training "
    "target, this project moved to four validation strategies, matching the outline's own Strategy "
    "I through Strategy IV and used consistently for every result in Table 2 and Table 3 above."
)

add_caption(doc, "Table 8. The four validation strategies used from Section 1.5 onward, applied consistently across every model comparison in this paper.")
add_table(doc,
    ["Strategy", "Construction", "Question it answers"],
    [
        ["Strategy I: Random split", "Rows assigned to train or test at random, stratified by source", "Interpolation; what most published CHF machine learning work actually reports"],
        ["Strategy II: Condition-wise split", "The highest-pressure portion of each source held out, rest used for training", "Can the model handle operating conditions it has not directly seen, within a familiar source?"],
        ["Strategy III: Surface-wise split", "One or more entire surface types or geometries withheld completely from training", "Can the model handle a surface or geometry it has never seen at all?"],
        ["Strategy IV: Leave-one-source-out cross-validation", "Each independent source held out in turn, with results pooled across all folds", "Can the model handle an entirely new laboratory, repeated across every source in the dataset?"],
    ]
)

doc.add_paragraph(
    "Two further points about how these strategies were implemented are worth recording, because "
    "both were discovered as mistakes and then fixed, and both are the kind of error that is easy "
    "to make without noticing. First, after building these splits, this project added an automatic "
    "check for degenerate groupings: a Strategy III or Strategy IV split where every group happens "
    "to contain only one row is silently identical to a random split, and a split where one group "
    "exceeds 95 percent purity from a single source is effectively Strategy IV wearing Strategy III's "
    "label. Both of these had, in fact, occurred earlier in this project and been reported as if they "
    "were the intended, harder protocol, until the automatic check caught them. Second, three "
    "grouped splits used at one point were labelled as testing generalization to an unseen "
    "laboratory when the grouping column they used actually corresponded to something else entirely, "
    "a pressure range in one case and individual coil geometries in another, both audited and "
    "renamed once discovered. Neither error changed the paper's overall conclusions, since the same "
    "pattern, strong scores on easy splits and weak or collapsing scores on hard ones, showed up "
    "consistently regardless, but both are recorded here because a validation strategy that is not "
    "actually testing what its name claims is a quiet way for an otherwise careful project to end up "
    "overstating how well a model generalizes."
)

doc.add_paragraph(
    "The practical rule that came out of all four strategies, repeated across every model in Tables "
    "2 through 4, is simple to state and easy to forget: a single reported R-squared value is only "
    "meaningful once the split it was measured on is stated alongside it. The same model can score "
    "above 0.9 and below minus 100 depending entirely on which of the four strategies in Table 8 was "
    "used to test it, and any CHF prediction result presented in this paper always names which "
    "strategy produced it for exactly this reason."
)

doc.save("CHF_Research_Paper.docx")
print("saved")

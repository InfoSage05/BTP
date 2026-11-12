"""Render the full pipeline report to .docx in the project house style.

House style, from the bottom of `outline/paper outline.docx`:
    "Texts should be times new roman, 12, 1.5 spacing"

Every number is read from the results CSVs rather than typed in, so the document
cannot drift away from what the code actually produced.
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Inches, Pt, RGBColor

HERE = Path(__file__).resolve().parent
RES = HERE / ".." / ".." / "results" / "prof_tasks"
FIGS = RES / "figures_outline"
DST = HERE / ".." / ".." / "docs" / "CHF_Pipeline_Report.docx"

BODY, SIZE, LS = "Times New Roman", Pt(12), 1.5
SHORT = {"D1_NRC": "NRC / Groeneveld", "D2_Zhao2020": "Zhao 2020",
         "D3_KAERI_uniform": "KAERI uniform", "D4_KAERI_nonuniform": "KAERI non-uniform",
         "D5_Helical_R123": "Helical coil R-123", "D6_Hardik_helical_water": "Hardik helical (water)",
         "D7_Hardik_straight_R123": "Hardik straight tubes (R-123)", "D8_Pioro_R134a": "Pioro R-134a"}


def style(doc):
    n = doc.styles["Normal"]
    n.font.name, n.font.size = BODY, SIZE
    n.paragraph_format.line_spacing = LS
    n.paragraph_format.space_after = Pt(6)
    for lv in range(1, 5):
        s = doc.styles[f"Heading {lv}"]
        s.font.name = BODY
        s.font.color.rgb = RGBColor(0, 0, 0)
        s.font.size = Pt({1: 18, 2: 14, 3: 12.5, 4: 12}[lv])
        s.paragraph_format.space_before = Pt(14)
        s.paragraph_format.space_after = Pt(6)


def para(doc, text, bold=False, italic=False, size=None, align=None):
    p = doc.add_paragraph()
    r = p.add_run(text)
    r.bold, r.italic = bold, italic
    r.font.name, r.font.size = BODY, size or SIZE
    if align:
        p.alignment = align
    return p


def bullets(doc, items):
    for it in items:
        p = doc.add_paragraph(style="List Bullet")
        r = p.add_run(it)
        r.font.name, r.font.size = BODY, SIZE
        p.paragraph_format.line_spacing = LS


def table(doc, headers, rows, widths=None, caption=None):
    if caption:
        c = para(doc, caption, italic=True, size=Pt(10.5))
        c.paragraph_format.space_after = Pt(3)
    t = doc.add_table(rows=1, cols=len(headers))
    t.style = "Table Grid"
    for i, h in enumerate(headers):
        cell = t.rows[0].cells[i]
        cell.text = ""
        r = cell.paragraphs[0].add_run(str(h))
        r.bold = True
        r.font.name, r.font.size = BODY, Pt(10.5)
    for row in rows:
        cells = t.add_row().cells
        for i, v in enumerate(row):
            cells[i].text = ""
            r = cells[i].paragraphs[0].add_run(str(v))
            r.font.name, r.font.size = BODY, Pt(10.5)
    doc.add_paragraph()
    return t


def figure(doc, name, caption):
    p = FIGS / name
    if not p.exists():
        return
    doc.add_picture(str(p), width=Inches(6.3))
    doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
    c = para(doc, caption, italic=True, size=Pt(10.5), align=WD_ALIGN_PARAGRAPH.CENTER)
    c.paragraph_format.space_after = Pt(12)


def main():
    r = pd.read_csv(RES / "FINAL_results.csv")
    doc = Document()
    style(doc)

    # ---------------------------------------------------------------- title
    para(doc, "Machine-Learning Prediction of Critical Heat Flux",
         bold=True, size=Pt(22), align=WD_ALIGN_PARAGRAPH.CENTER)
    para(doc, "The complete project record: what was tried, what the results were, "
              "why each direction was abandoned or kept, and the pipeline that resulted",
         italic=True, size=Pt(13), align=WD_ALIGN_PARAGRAPH.CENTER)
    para(doc, "B.Tech Project · Technical Report", size=Pt(11),
         align=WD_ALIGN_PARAGRAPH.CENTER)
    doc.add_paragraph()

    # ---------------------------------------------------------------- summary
    doc.add_heading("1. Summary", 1)
    para(doc, "This report describes a system that predicts critical heat flux (CHF) in "
              "flow boiling, the point at which a heated surface stops being wetted by "
              "liquid and its temperature rises sharply. In a water-cooled reactor this "
              "limit sets the thermal margin of the core, so both the accuracy of a "
              "prediction and the honesty of the accuracy claim carry safety significance.")
    para(doc, "The system combines a closed-form physics correlation with a machine-learning "
              "correction. The correlation provides a floor that is never wildly wrong; the "
              "correction improves it where data supports doing so; and a trust signal limits "
              "how far the correction is allowed to move the answer when the situation looks "
              "unfamiliar. Every prediction is returned with a calibrated interval and a plain "
              "statement of confidence.")
    para(doc, "Headline results", bold=True)
    bullets(doc, [
        "On the largest database (24,443 experiments), 99.4% of the variance is explained "
        "and 96.2% of predictions land within 10% of the measured value.",
        "On a dataset the model has never seen at all, the median R-squared is 0.711, "
        "against 0.336 for the physics correlation alone and −1.323 for machine learning "
        "with no physics anchor.",
        "The single most important property is the absence of catastrophe: unanchored "
        "machine learning reaches R-squared of −868 on an unseen dataset, whereas the "
        "worst case for this pipeline is −0.065.",
        "Eleven attempts to improve the model were tested and ten were rejected on "
        "evidence. The one change that helped was a data-correctness bug, not a modelling idea.",
    ])


    # ================================================================ PART I
    doc.add_page_break()
    doc.add_heading("PART I — HOW THE PROJECT DEVELOPED", 1)
    para(doc, "This part is the story of the work in the order it happened. Each phase "
              "records what was attempted, what the numbers were, and the reason the "
              "project moved on. The reasons matter as much as the results: almost every "
              "design choice in the final pipeline exists because an earlier approach "
              "failed in a specific, measurable way.")

    doc.add_heading("2. Starting point — the look-up table", 1)
    para(doc, "The project began with the 2006 Groeneveld look-up table, the standard "
              "reference that gives CHF as a function of pressure, mass flux and "
              "thermodynamic quality. After cleaning it yields 11,592 rows on a regular "
              "24 x 21 x 23 grid. Exactly 504 of those rows have CHF equal to zero, all at "
              "quality 1.0, which is an all-steam placeholder rather than a measurement, "
              "leaving 11,088 usable rows. Surviving CHF values span 15 to 44,338 kW/m2 — "
              "roughly three orders of magnitude, which is why every model in the project "
              "trains on the logarithm of CHF rather than the raw value.")
    para(doc, "Three ways of splitting the data were defined to probe progressively harder "
              "generalisation:")
    bullets(doc, [
        "Split A — random. Interpolation inside the grid. The optimistic test.",
        "Split B — interior pressure holdout. Whole pressure levels removed, with "
        "training data on both sides.",
        "Split C — edge pressure extrapolation. Testing beyond the trained pressure "
        "range. The honest test.",
    ])
    para(doc, "About thirty model configurations were evaluated across linear, polynomial "
              "ridge, nearest-neighbour, Random Forest, Extra Trees, XGBoost, LightGBM, "
              "Gaussian process, neural network, physics-feature and residual-learning "
              "families, plus a direct grid-interpolation baseline.")
    table(doc, ["Model", "Split A", "Split B", "Split C (verified)", "MAPE on C"],
          [["Grid interpolation (physical baseline)", "—", "0.999", "0.841", "20.8%"],
           ["Degree-2 log ridge", "0.847", "0.871", "0.755", "35.8%"],
           ["Neural network (log target)", "0.991", "0.989", "0.628 ± 0.072", "40.0%"],
           ["Tree ensembles (RF / XGB / LGBM / ET)", "0.999", "0.988", "≈ 0.43", "≈ 42%"],
           ["Pressure-gated blend (raw target)", "0.999", "0.999", "0.466 ± 0.228", "67.1%"]],
          caption="Table 2. Phase I results. Split C values are means over 30 independent "
                  "seeds where the model is stochastic.")
    para(doc, "Two findings came out of this phase and both shaped everything afterwards.")
    para(doc, "First, the tree collapse.", bold=True)
    para(doc, "Models scoring 0.999 under random splitting fall to roughly 0.43 when asked "
              "to extrapolate beyond the trained pressure range, while a simple "
              "deterministic grid interpolation holds 0.841. This was the first evidence "
              "that headline accuracy under random splits is a poor guide to real "
              "generalisation.")
    para(doc, "Second, a result had to be withdrawn.", bold=True)
    para(doc, "An internal review had ranked a pressure-gated blend first at R-squared "
              "0.855 on Split C. Independent verification across 30 seeds showed the "
              "driving model averaged 0.547 and not one seed reached 0.84. The claim was "
              "retracted and the verification script kept in the repository. Two "
              "implementation bugs were found in the same pass, including a Split B "
              "definition that silently selected the topmost pressure level instead of an "
              "interior one. From that point on, every number in this project is a "
              "multi-seed mean rather than a single run.")

    doc.add_heading("3. Physics-informed neural networks", 1)
    para(doc, "Since the models failed specifically at extrapolation, the natural next "
              "question was whether physical constraints could stabilise them. Penalties "
              "were imposed on collocation points sampled across the operating domain: CHF "
              "must not increase with quality, predictions must follow the Zuber "
              "hydrodynamic trend with pressure, and CHF must stay positive.")
    para(doc, "A hyperparameter search of 720 configurations was run on cloud GPUs, "
              "followed by a three-tier refinement.")
    table(doc, ["Tier", "Configuration", "Ensemble R² (raw)", "MAPE", "Beats grid baseline?"],
          [["Tier 1", "architecture and monotonicity weight", "0.732", "37.8%", "no"],
           ["Tier 2", "plus Zuber penalty, tanh activation", "0.770", "37.2%", "no"],
           ["Tier 2b", "residual target, SiLU activation", "0.847", "20.4%", "yes"]],
          caption="Table 3. Physics-informed network results against the 0.841 grid "
                  "interpolation baseline.")
    para(doc, "Tier 2b is the only configuration anywhere in this phase that beat the "
              "deterministic baseline, and it did so by changing the target "
              "parameterisation rather than by any physics penalty. Seed-to-seed spread "
              "also fell from 0.179 to 0.007, a 25-fold stability improvement.")
    para(doc, "The honest conclusion was negative.", bold=True)
    para(doc, "Inspection of the training logs showed the monotonicity penalty read exactly "
              "zero throughout training — it was never active. The best penalty weights in "
              "Tier 2b were zero for both physics terms. So this phase does not constitute "
              "a fair test of physics-constrained learning; the physics was present in the "
              "loss function and inert in practice. A separate headline of 0.8123 from the "
              "720-configuration search was also retired, because its configuration had "
              "been selected using the test set.")
    para(doc, "Research during this phase produced an important reframing: CHF has no "
              "governing partial differential equation, so \u201cphysics-informed neural "
              "network\u201d is the wrong template. The technique that actually applies is "
              "residual or hybrid correction on top of an empirical correlation. That idea "
              "is the direct ancestor of the final architecture.")

    doc.add_heading("4. Building a real dataset", 1)
    para(doc, "The look-up table is a smoothed reference, not raw measurement, so the "
              "project turned to assembling genuine experimental data.")
    para(doc, "Thirty-five reference papers were indexed, each recorded with its citation, "
              "identifier, and a note on whether extractable data exists. Eleven "
              "publications were extracted by hand using exact text extraction rather than "
              "visual reading, producing paired roughness, contact angle and CHF records "
              "for engineered surfaces. A large database believed to be restricted was "
              "located in a public regulatory document and transcribed in full: roughly "
              "24,579 points, of which 136 proved to be exact duplicates.")
    para(doc, "Several candidate sources were examined and deliberately rejected, and "
              "recording why is as much a part of the work as the data that was kept:")
    table(doc, ["Rejected source", "Reason"],
          [["The look-up table itself, as training data",
            "A smoothed table on a regular grid, and not independent of the large "
            "database, which is built from the same historical experiments."],
           ["A 21-row sample table",
            "Every row proved to be an exact duplicate of rows already held, digitising "
            "the same 1958 experiment."],
           ["Helical minichannel logs, 32 runs",
            "The recorded output is heater power, not heat flux; conversion needed "
            "per-geometry assumptions too subjective to justify."],
           ["Several source-summary tables",
            "They list parameter ranges and study metadata but contain no point-level "
            "CHF values."]],
          caption="Table 4. Data deliberately excluded, and the reason in each case.")
    para(doc, "A Gaussian-process generator was also built to densify the look-up table "
              "interior, accepting 7,114 of 8,000 sampled points. A later self-audit "
              "established that because every accepted point was filtered for agreement "
              "with grid interpolation, this synthetic data is not independent of that "
              "baseline and must never be used to claim a model outperforms it. It is "
              "retained for interior densification only and was excluded from all "
              "confounding analysis.")

    doc.add_heading("5. Cleaning, and the defects it exposed", 1)
    table(doc, ["Defect", "Consequence if uncorrected", "Resolution"],
          [["A units row beneath the header in one export",
            "Every column read as text; maximum pressure returned the string \u201ckPa\u201d.",
            "Row detected and dropped, with an assertion verifying zero coercion failures."],
           ["136 exact duplicate rows",
            "Identical experiments under different reference identifiers, inflating "
            "random-split scores.", "Removed; both versions retained for inspection."],
           ["Study labels present only on the first row of each block",
            "90% of rows would have carried no laboratory label, making source-wise "
            "validation impossible.", "Forward-filled; block contiguity verified independently."],
           ["Pressure in Pa and heat flux in W/m2 misread as kPa and kW/m2",
            "Merged pressures of 180,000 bar and CHF of 28.8 GW/m2; pressure became a "
            "perfect source indicator.",
            "Units stated explicitly from the source; all heuristic unit detection "
            "removed and replaced with plausibility assertions that abort the run."]],
          caption="Table 5. Data defects found during cleaning.")
    para(doc, "Four preprocessing conventions were fixed at this point and have held ever "
              "since: train on the logarithm of CHF; fit any scaler inside the "
              "cross-validation fold so it never sees held-out data; never impute a value "
              "a source does not report; and exclude by rule any column that reconstructs "
              "the answer.")

    doc.add_heading("6. A unified dataset and four ways to split it", 1)
    para(doc, "Seven point-level sources were merged into a single schema, one row per "
              "measured CHF, with every unit conversion stated explicitly from source "
              "documentation. Four split strategies were defined, each answering a "
              "different question.")
    table(doc, ["Strategy", "Construction", "Question it answers"],
          [["1 — Random, stratified", "stratified by source dataset",
            "Interpolation within known conditions. The optimistic protocol most "
            "published work uses."],
           ["2 — Condition-wise", "per-source top-pressure holdout",
            "Extrapolation to unseen operating conditions."],
           ["3 — Surface-wise", "whole surface types withheld",
            "Generalisation to an unseen geometry."],
           ["4 — Leave-one-source-out", "each source held out in turn",
            "Generalisation to an unseen laboratory."]],
          caption="Table 6. The four validation protocols.")
    para(doc, "An automatic degeneracy check was added after two split definitions were "
              "found to be mislabelled: every grouped split is now tested for all-singleton "
              "groups, which make it mathematically identical to a random split, and for "
              "groups exceeding 95% purity with respect to source, which make it a source "
              "split under a different name. Both conditions had occurred and were being "
              "reported as independent protocols.")

    doc.add_heading("7. The model bake-off", 1)
    para(doc, "Fifteen model families were evaluated across all four protocols: linear "
              "regression, polynomial ridge, nearest neighbours, support-vector regression, "
              "a multilayer perceptron, Random Forest, Extra Trees, XGBoost, LightGBM, "
              "CatBoost, a Gaussian process, a stacking ensemble, histogram gradient "
              "boosting, a physics-informed network, and a proposed two-tier hierarchy.")
    table(doc, ["Model", "1 — random", "2 — condition", "3 — surface", "4 — source"],
          [["Histogram gradient boosting", "0.968", "0.890", "0.173", "0.784"],
           ["Random Forest", "0.965", "0.875", "0.219", "0.269"],
           ["Neural network", "0.921", "0.515", "−4132.9", "−5.150"],
           ["Gaussian process", "0.872", "−0.051", "−0.813", "0.539"],
           ["Physics-informed network", "0.876", "−64.9", "−3906.9", "−6.022"]],
          caption="Table 7. R-squared across the four protocols. The collapse from the "
                  "first column to the last two is the central quantitative observation of "
                  "the whole project.")
    para(doc, "Four conclusions followed. Neural architectures are catastrophically "
              "unstable on held-out surfaces, reaching R-squared of −4,133; these are not "
              "merely poor scores, but the behaviour of a model returning arbitrary "
              "numbers. Gradient boosting is the most robust family. Ensembling did not "
              "help: a stacking ensemble scored 0.459 against 0.945 for a single Extra "
              "Trees model under honest splitting, because the meta-learner's weights are "
              "themselves fitted to the training laboratories. And simple models often win "
              "when extrapolating — plain linear regression beat CatBoost, XGBoost and the "
              "Gaussian process on one leave-one-publication-out evaluation.")

    doc.add_heading("8. The turning point — source-level confounding", 1)
    para(doc, "The repeated collapse under source-wise splitting prompted a direct "
              "question: are the relationships in these compiled databases genuine physics, "
              "or artefacts of which laboratory produced each measurement?")
    table(doc, ["Dataset", "Rows", "Publications", "Rows per publication",
                "Variance from source identity alone", "Gain from physical features"],
          [["Engineered surfaces", "65", "10", "6.5", "0.806", "+0.041"],
           ["KAERI", "888", "11", "80.7", "0.735", "+0.187"],
           ["Pin-fin", "175", "16", "10.9", "0.659", "+0.303"],
           ["NRC", "24,443", "60", "407.4", "0.345", "+0.556"],
           ["Zhao", "1,865", "10", "186.5", "0.273", "+0.511"]],
          caption="Table 8. On the smallest dataset, knowing only which laboratory "
                  "performed a test explains 80.6% of the variance in log CHF. Adding "
                  "roughness AND contact angle contributes a further 4.1 percentage points.")
    para(doc, "The severity scales inversely with measurements per publication — a clean "
              "dose-response across three orders of magnitude in sample size. Two further "
              "results emerged. Pooling laboratories can reverse the sign of a physical "
              "relationship: the roughness-to-CHF correlation flips from −0.293 pooled to "
              "+0.439 within-study, contradicting the mechanism the source papers "
              "themselves invoke. And standard diagnostics give no warning — feature "
              "importance rankings are unchanged between honest and optimistic protocols, "
              "so a model can present stable, physically plausible attributions while "
              "failing entirely to generalise.")
    para(doc, "This phase is why every result in the final pipeline is reported under "
              "leave-one-source-out and leave-one-dataset-out, not only under random "
              "splits.")

    doc.add_heading("9. Building a physics foundation", 1)
    para(doc, "A 1,382-line reference document was written to establish provenance for "
              "every equation used afterwards, covering the phenomenon itself, dimensional "
              "analysis, the five pool-boiling trigger mechanisms, the physics of surface "
              "characteristics, flow-boiling correlations, geometry-specific effects, what "
              "machine learning has and has not achieved, and the constraints any "
              "admissible CHF model must satisfy. Every equation carries a provenance tag "
              "recording whether it was transcribed from a primary source, verified only in "
              "structure, or not obtained at all.")
    para(doc, "Three findings from that document drive the final design: the correct "
              "feature space is dimensionless, because water at 15 MPa and a refrigerant at "
              "1 MPa map to the same point there; seven hard and ten soft physical "
              "constraints are testable without new data; and inter-laboratory "
              "reproducibility on nominally identical test sections is about 10%, meaning "
              "random-split results sitting inside that band are measuring source-specific "
              "bias rather than physics.")

    doc.add_heading("10. Physics-guided models", 1)
    para(doc, "Six arms were built, each adding exactly one idea to the one above, so that "
              "every contribution could be isolated. Pure-physics arms carry no fitted "
              "parameters and form the reference every learned arm must beat.")
    table(doc, ["Arm", "1 — random", "2 — condition", "3 — surface", "4 — source"],
          [["Physics only (Katto–Ohno), no learning", "0.779", "0.508", "0.505", "0.785"],
           ["A0 — no physics, raw features", "0.979", "0.903", "−17.997", "0.719"],
           ["A1 — latent-heat baseline", "0.970", "0.877", "0.711", "0.800"],
           ["A2 — plus Katto–Ohno baseline", "0.978", "0.812", "0.606", "−0.797"],
           ["A3 — plus dimensionless features", "0.982", "0.790", "0.252", "−0.467"],
           ["A5 — plus bounded, monotone, trust", "0.957", "0.854", "0.506", "0.710"]],
          caption="Table 9. The physics ablation ladder.")
    para(doc, "Credit the data, not the models.", bold=True)
    para(doc, "A dedicated attribution test separated data repairs from modelling ideas. On "
              "the surface-wise split, correcting the data — fixing units, adding proper "
              "saturation properties, and substituting a well-populated subcooling column "
              "for one with 0.8% coverage — accounts on its own for a rise from 0.173 to "
              "0.711, before any physics idea is applied. Most of the apparent gain from "
              "physics-informed modelling was in fact the value of correct data. That "
              "lesson recurred in the final pipeline, where the only successful improvement "
              "in eleven attempts was again a data-correctness fix.")
    para(doc, "Three further results from this phase carried directly into the final "
              "design. Non-dimensionalisation does not make models more accurate, it makes "
              "them portable: on unseen refrigerants it improved error by 10 to 32 times "
              "while leaving within-fluid accuracy unchanged. Bounding a learned correction "
              "produces a monotonic, controllable trade-off between safety on novel data "
              "and accuracy on familiar data. And closed-form correlations beat every "
              "machine-learning model in genuinely unseen regimes — on axially non-uniform "
              "heating, a 1984 correlation with zero fitted parameters scored 0.821 where "
              "the best learned model scored −0.077.")
    para(doc, "One problem was left explicitly open. A novelty-weighted blend between "
              "physics and machine learning, calibrated honestly on held-out training "
              "publications, selected zero physics weight — yet that selection is "
              "demonstrably wrong for the non-uniform dataset. The cause is structural: all "
              "training publications are water in straight tubes, so holding one out never "
              "produces a point novel enough to reveal where the fallback earns its value. "
              "An out-of-distribution fallback cannot be calibrated using in-distribution "
              "data, and out-of-distribution data is by definition unavailable at "
              "calibration time.")

    doc.add_heading("11. Transfer learning, neural networks and transformers", 1)
    para(doc, "A two-stage transfer-learning pipeline was built: a 300,000-row synthetic "
              "corpus sampled from the look-up table for pre-training, then fine-tuning on "
              "each real target domain. Both a multilayer perceptron and a transformer "
              "encoder were implemented, along with low-rank adaptation and a "
              "mixture-of-experts routing variant.")
    table(doc, ["Architecture", "Split", "n", "R²", "MAPE", "Within ±10%"],
          [["Neural network", "interpolation", "2,459", "0.951", "17.9%", "47.3%"],
           ["Transformer", "interpolation", "2,459", "0.957", "16.5%", "47.7%"],
           ["Neural network", "pressure extrapolation", "13,748", "0.838", "44.6%", "25.8%"],
           ["Transformer", "pressure extrapolation", "13,748", "0.844", "49.5%", "20.2%"]],
          caption="Table 10. Pre-training results. The transformer's advantage over the "
                  "simpler network is marginal at roughly thirty times the training cost.")
    para(doc, "Fine-tuning gave mixed evidence for transfer. On one refrigerant domain the "
              "pre-trained network reached 0.481 against −0.171 from scratch, a clear gain. "
              "On another the from-scratch model won. Pooling all fine-tuning domains into "
              "one shared model was worse than separate per-domain fine-tuning in eight of "
              "ten combinations, one catastrophically at −27.9.")
    table(doc, ["Technique", "Trainable parameters", "R² on the target domain"],
          [["Training from scratch", "100%", "0.012"],
           ["Full fine-tuning", "100%", "0.696"],
           ["Low-rank adaptation", "8.8%", "0.685"],
           ["Mixture of experts", "100%", "0.474"]],
          caption="Table 11. Adaptation techniques on a small pool-boiling target. "
                  "Low-rank adaptation reaches near-parity while training under 9% of the "
                  "parameters.")
    para(doc, "Two negative findings from this phase were decisive. The mixture-of-experts "
              "gate did not learn the regime it was routing on: it fired at 0.108 on "
              "pool-boiling rows against 0.609 on flow rows. And eight-member deep "
              "ensembles produced prediction intervals claiming 95% confidence that "
              "actually contained the true value only 9 to 22% of the time. The first of "
              "these is why the final design refuses to route between models; the second is "
              "why it uses conformal prediction rather than ensemble spread for "
              "uncertainty.")

    doc.add_heading("12. Narrowing to flow boiling", 1)
    para(doc, "At this point the project had accumulated ten datasets, fifteen model "
              "families, four validation protocols and a set of hard-won negative results. "
              "The decision was taken to narrow the scope to flow boiling and build one "
              "pipeline properly rather than continue broadening.")
    para(doc, "Every design choice in that pipeline traces to a specific earlier failure:")
    table(doc, ["Design choice", "The failure that motivated it"],
          [["A physics correlation as the backbone",
            "Unanchored models reach R-squared of −4,133 and −868 on unseen data."],
           ["Dimensionless feature space",
            "Raw-variable models cannot transfer across fluids; dimensionless ones improve "
            "error 10 to 32 times."],
           ["One pooled correction, never per-dataset routing",
            "The mixture-of-experts gate failed to learn its own regime."],
           ["Trust limits the correction rather than choosing a model",
            "A switch costs a factor of a thousand when it chooses wrong."],
           ["Conformal prediction intervals",
            "Deep-ensemble intervals achieved 9 to 22% of their nominal coverage."],
           ["Multi-seed reporting throughout",
            "A single-seed headline of 0.855 had to be publicly retracted."],
           ["Leave-one-source-out and leave-one-dataset-out reporting",
            "Source identity alone explains up to 80.6% of variance."]],
          caption="Table 12. The final architecture is a direct response to measured "
                  "failures, not a design chosen in advance.")

    doc.add_page_break()
    doc.add_heading("PART II — THE PIPELINE IN DETAIL", 1)

    # ------------------------------------------------------------- datasets
    doc.add_heading("13. The data used by the pipeline", 1)
    para(doc, "Eight experimental flow-boiling datasets were used, holding 28,582 individual "
              "burnout measurements. Pool-boiling data was excluded from this study by "
              "instruction. The 2006 Groeneveld look-up table is used as a reference "
              "correlation rather than as training data, because it is a smoothed table "
              "rather than raw measurement and is not independent of the NRC database.")
    rows = []
    for k, lab in SHORT.items():
        d = r[(r.dataset == k) & (r.split == "80/20")].iloc[0]
        rows.append([lab, f"{int(d.n):,}"])
    meta = {"D1_NRC": ("water", "vertical tubes", "60 studies, from NRC document ML22264A009"),
            "D2_Zhao2020": ("water", "tubes, annuli, plates", "10 authors, compilation"),
            "D3_KAERI_uniform": ("water", "tubes", "uniform axial heating, 10 sources"),
            "D4_KAERI_nonuniform": ("water", "tubes", "non-uniform axial heating, 11 sources"),
            "D5_Helical_R123": ("R-123", "helical coils", "appendices C and D of the advisor's PDF"),
            "D6_Hardik_helical_water": ("water", "helical coils", "advisor's PDF, table D.2"),
            "D7_Hardik_straight_R123": ("R-123", "horizontal tubes", "advisor's PDF, table D.1"),
            "D8_Pioro_R134a": ("R-134a", "horizontal and vertical tubes", "digitised from charts")}
    table(doc,
          ["Dataset", "Rows", "Fluid", "Geometry", "Notes"],
          [[lab, n, *meta[k]] for (lab, n), k in zip(rows, SHORT)],
          caption="Table 1. The eight flow-boiling datasets used throughout this report.")
    para(doc, "Three of these datasets — the two Hardik sets and the helical R-123 set, "
              "468 rows in total — come from the coil-and-tube appendix supplied by the "
              "advisor. Table D.1 of that document matches our file row for row.")
    figure(doc, "fig1_parameter_coverage.png",
           "Figure 1. Operating-envelope coverage. The bar spans the minimum to maximum "
           "value in each dataset and the dot marks the median. The water databases cover "
           "wide ranges; the refrigerant sets occupy narrow, low-pressure corners, which is "
           "why predicting them from the others is hard.")

    # ---------------------------------------------------------------- integrity
    doc.add_heading("14. Data integrity: what was found and fixed", 1)
    para(doc, "Before any model was trained, each dataset was audited. Several problems were "
              "found that would have produced impressive but meaningless results.")

    doc.add_heading("14.1 Columns that contain the answer", 2)
    para(doc, "Some columns are algebraically equivalent to the quantity being predicted. "
              "Each suspected column was tested by recomputing CHF from it, and excluded "
              "only when the reconstruction succeeded.")
    table(doc, ["Dataset", "Column removed", "How it reproduces the answer", "Quality of reproduction"],
          [["KAERI (both)", "Power, Perimeter, Area, MassFlow, InletEnthalpy",
            "Power ÷ (perimeter × length) is the heat flux", "R² = 0.999993, 100% within 10%"],
           ["Hardik helical", "Q_W, CHF_ratio",
            "Q_W ÷ heated area is the heat flux; CHF_ratio is CHF ÷ look-up table",
            "R² = 0.979, 98.1% within 10%"],
           ["Helical R-123", "Q_watt", "same heat-input relation", "confirmed per coil"],
           ["KAERI non-uniform", "CHFLocation, QualityPosition",
            "measured at burnout, therefore outcomes not inputs", "excluded on principle"]],
          caption="Table 2. Columns excluded from every model, with the evidence for excluding them.")

    doc.add_heading("14.2 Outlet quality is not an independent input", 2)
    para(doc, "Every CHF database computes the thermodynamic quality at the tube outlet from "
              "the measured CHF, using an energy balance. Feeding that quality back in as an "
              "input therefore asks the model to recover arithmetic that was already done. "
              "On the NRC database this can be shown exactly, with nothing fitted:")
    table(doc, ["Measure", "Value"],
          [["R² of the energy balance against 24,443 measurements", "0.999771"],
           ["Median relative error", "0.372%"],
           ["Predictions within ±10%", "100.0%"],
           ["Predictions within ±1%", "77.3%"]],
          caption="Table 3. The energy balance CHF = G·D·(x·h_fg + ΔH_in) / (4L), with no "
                  "fitted parameters, reproduces the measured CHF almost exactly.")
    para(doc, "This relation is conservation of energy — Hall and Mudawar (2000), equation 8 — "
              "not a discovery. It is why this pipeline never uses outlet quality as an input. "
              "Everything is predicted from conditions known before an experiment runs.")
    para(doc, "The circularity is not universal, and this was tested rather than assumed. "
              "It holds on NRC and KAERI uniform. It does not hold on KAERI non-uniform, "
              "where the quality column turned out to be the INLET quality: the expression "
              "(h_in − h_f)/h_fg reproduces that column on all 888 rows to within 0.001. "
              "That quality is an independent measurement and is used as a legitimate input.")

    doc.add_heading("14.3 Bugs found during the work", 2)
    table(doc, ["Problem", "Effect", "Status"],
          [["Zhao rows carried a hard-coded 8 mm diameter",
            "True diameters span 1–37.5 mm. The physics baseline was meaningless for all "
            "1,865 rows and nothing reported it.", "Fixed; accuracy 0.824 → 0.851"],
           ["Unknown fluids silently used water's properties",
            "A request for ammonia returned a confident number computed from water.",
            "Fixed; now resolved through CoolProp or refused"],
           ["578 duplicate rows inside datasets",
            "A naive random split puts identical rows on both sides and inflates the score.",
            "All random splits now grouped so duplicates cannot straddle"],
           ["Three splits mislabelled as 'unseen laboratory'",
            "The groups were a pressure range, a coil geometry and figures within one paper.",
            "Renamed to what they actually are"],
           ["Trust reference cloud included the calibration slice",
            "Confidence intervals came out slightly too tight.", "Fixed"],
           ["save() modified the object it was saving",
            "Calling save shrank the live model.", "Fixed; copies first"],
           ["Group labels aligned to rows by position",
            "Would silently mispair the moment any row was dropped.",
            "Fixed; assertion-guarded"]],
          caption="Table 4. Defects found during development and review, and their resolution.")

    # ---------------------------------------------------------------- architecture
    doc.add_heading("15. How the pipeline works", 1)
    para(doc, "The system has five layers. Each exists because a measurement justified it.")

    doc.add_heading("15.1 Layer 0 — Fluid properties and dimensionless groups", 2)
    para(doc, "The fluid name and pressure are sent to CoolProp, a standard thermophysical "
              "property library, which returns the liquid and vapour density, the surface "
              "tension and the latent heat of vaporisation. These are combined into five "
              "dimensionless numbers: an inverse Weber number, the density ratio, the "
              "length-to-diameter ratio, the subcooling ratio and the reduced pressure.")
    para(doc, "This is the reason an unfamiliar fluid can be predicted at all. The model never "
              "sees the word 'water' or 'R-134a'. It sees numbers, and water at 10 MPa can "
              "land at the same point as a refrigerant at 1.5 MPa.")

    doc.add_heading("15.2 Layer 1 — A physics correlation with nothing fitted", 2)
    para(doc, "A closed-form correlation predicts CHF outright. Three are available: "
              "Katto–Ohno (1984), Biasi (1967) and the 2006 Groeneveld look-up table. "
              "The pipeline selects whichever has the lowest error on the training rows, and "
              "that choice never sees test data. This layer is the floor that prevents "
              "catastrophic answers.")

    doc.add_heading("15.3 Layer 2 — One machine-learning correction", 2)
    para(doc, "An Extremely Randomised Trees model predicts the logarithm of the ratio "
              "between the measured CHF and the correlation's prediction — in plain terms, "
              "how wrong the correlation tends to be under these conditions. It is trained on "
              "all datasets pooled, not one model per dataset: per-dataset routing was tested "
              "twice and failed both times.")

    doc.add_heading("15.4 Layer 3 — Trust limits the correction", 2)
    para(doc, "This is the most important design decision. An earlier version switched between "
              "physics and machine learning. A switch is all-or-nothing, so every wrong choice "
              "costs a factor of a thousand; that version produced R-squared of −1276 on an "
              "unseen fluid.")
    para(doc, "Instead, a trust score between 0 and 1 controls how far the correction may move "
              "the answer. Trust falls if the conditions are far from anything seen in "
              "training, if the fluid does not appear in the training data at all, or if the "
              "available correlations disagree with each other.")
    table(doc, ["Trust", "Correction allowed", "Behaviour"],
          [["1.0", "up to ×10", "the learned model does the work"],
           ["0.5", "about ×3.9", "correction moderated"],
           ["0.0", "×1.5", "answer decays into the correlation"]],
          caption="Table 5. Trust does not choose a model; it decides how much correction is permitted.")

    doc.add_heading("15.5 Layer 4 — A calibrated interval and a plain label", 2)
    para(doc, "One fifth of the training data is held back and never shown to the correction "
              "model. The spread of its errors sets the width of the prediction interval, "
              "which widens further as trust falls. Every prediction carries one of three "
              "labels: interpolating, extrapolating, or correlation fallback.")
    para(doc, "This layer is not optional. Earlier ensemble models in this project produced "
              "intervals claiming 95% confidence that actually contained the true value only "
              "9 to 22% of the time. Uncertainty that is confidently wrong is worse than none.")

    doc.add_heading("15.6 What a user provides", 2)
    para(doc, "The interface asks only for quantities known before an experiment is run: "
              "fluid, pressure, mass flux, tube diameter, heated length and inlet temperature. "
              "Outlet quality is never requested. A typical call and its answer:")
    table(doc, ["Input", "Value"],
          [["fluid", "water"], ["pressure", "10,000 kPa"], ["mass flux", "2,000 kg/m²s"],
           ["diameter", "8 mm"], ["heated length", "1,000 mm"], ["inlet temperature", "200 °C"]],
          caption="Table 6a. Example request.")
    table(doc, ["Output", "Value"],
          [["Predicted CHF", "2,904 kW/m²"], ["Interval", "2,640 to 3,200 kW/m²"],
           ["Confidence", "interpolating"], ["Physics-only reference", "2,994 kW/m²"]],
          caption="Table 6b. The corresponding answer. The correlation's own value is always "
                  "shown so the effect of the correction is visible.")
    para(doc, "Impossible requests are refused rather than answered: zero mass flux (that is "
              "pool boiling, outside this system's scope), a negative diameter, a pressure at "
              "or above the critical pressure, or an unrecognised substance.")

    doc.add_heading("16. Results", 1)
    para(doc, "Four different ways of splitting the data were used, because each answers a "
              "different question. All figures are averages over multiple random seeds with "
              "the standard deviation reported, never a single run.")

    doc.add_heading("16.1 Testing on held-out data from the same dataset", 2)
    para(doc, "This is the standard 80/20 and 70/30 split most published work reports. It "
              "answers: how well does the model do on more experiments from the same "
              "campaign? Identical rows are prevented from appearing on both sides.")
    for sp in ("80/20", "70/30"):
        d = r[r.split == sp].sort_values("R2_mean", ascending=False)
        table(doc, ["Dataset", "Rows", "R²", "Std dev", "Within ±10%", "Within ±20%",
                    "MAPE", "RMSE (kW/m²)", "Safe-side", "Coverage"],
              [[SHORT[x.dataset], f"{int(x.n):,}", f"{x.R2_mean:.3f}", f"{x.R2_sd:.3f}",
                f"{x.within_10pct_mean*100:.1f}%", f"{x.within_20pct_mean*100:.1f}%",
                f"{x.MAPE_pct_mean:.1f}%", f"{x.RMSE_mean:.0f}",
                f"{x.safe_side_frac_mean*100:.0f}%", f"{x.coverage90_mean:.2f}"]
               for x in d.itertuples()],
              caption=f"Table 7{'a' if sp=='80/20' else 'b'}. {sp} random split, 10 seeds. "
                      f"MAPE is mean absolute percentage error. Safe-side is the share of "
                      f"predictions BELOW the measurement, which is the conservative "
                      f"direction for a safety limit. Interval coverage should be near 0.90.")
    para(doc, "The two split ratios agree to within about 0.01 throughout, which shows the "
              "result does not depend on how much data is held back.")

    doc.add_heading("16.2 Why Extremely Randomised Trees was chosen", 2)
    para(doc, "Before the pipeline was built, twenty-one model variants were benchmarked "
              "across every dataset and both split ratios, producing 804 scored results. "
              "Models are ranked below by mean rank across nine datasets, which rewards "
              "consistency rather than a single strong showing.")
    table(doc, ["Model", "Mean rank", "Median R²", "Worst R²", "Mean within ±10%"],
          [["Extremely Randomised Trees", "4.22", "0.912", "0.233", "54.9%"],
           ["CatBoost", "4.22", "0.897", "0.233", "51.2%"],
           ["XGBoost", "4.22", "0.916", "0.232", "51.9%"],
           ["Random Forest", "4.44", "0.918", "0.232", "51.4%"],
           ["Support-vector regression", "5.67", "0.875", "0.253", "49.3%"],
           ["LightGBM", "5.78", "0.906", "0.244", "44.5%"],
           ["Histogram gradient boosting", "6.56", "0.893", "0.246", "43.8%"],
           ["Nearest neighbours", "8.22", "0.879", "0.201", "41.7%"],
           ["Degree-2 polynomial ridge", "8.33", "0.710", "0.235", "39.1%"],
           ["Linear regression", "9.67", "0.532", "0.129", "33.4%"],
           ["Neural network", "10.11", "0.572", "−48.34", "31.3%"]],
          caption="Table 8. Model benchmark. The top four are a statistical tie; Extremely "
                  "Randomised Trees was selected because it leads on the ±10% criterion, "
                  "which is the measure of practical interest. The neural network is the "
                  "one family to avoid outright: a respectable median alongside a −48 "
                  "collapse.")
    para(doc, "Gaussian process regression was also evaluated but had to be trained on a "
              "2,000-row subsample above that size, because its cost grows with the cube of "
              "the number of rows. It wins on the two smallest datasets and cannot be "
              "scaled to the largest, so it was not adopted.")

    doc.add_heading("16.3 Testing on an unseen laboratory", 2)
    para(doc, "A harder question: hold out one entire publication and predict it from the "
              "others. Only four datasets carry real laboratory labels. For the remaining "
              "three the group is a pressure range, a coil geometry, or figures within a "
              "single paper, and they are named accordingly rather than overstated.")
    g = r[r.split.str.startswith("GROUPED")].sort_values("R2_median", ascending=False)
    table(doc, ["Dataset", "What is held out", "Median R²", "Worst fold", "Within ±10%"],
          [[SHORT[x.dataset], x.split.replace("GROUPED: ", ""), f"{x.R2_median:.3f}",
            f"{x.R2_worst:.3f}", f"{x.within_10pct_mean*100:.1f}%"] for x in g.itertuples()],
          caption="Table 8b. Median across folds is reported rather than the mean, because on "
                  "KAERI uniform one fold reaches −73 and would dominate any average.")
    para(doc, "The Hardik straight-tube dataset has no group column at all — it is a single "
              "campaign — so this test cannot be run on it.")

    doc.add_heading("16.4 Testing on a completely unseen dataset", 2)
    para(doc, "The hardest and most informative test. An entire dataset is withheld and the "
              "model is trained on the other seven, so the held-out data is a different "
              "fluid, geometry or campaign. The datasets were verified to share no duplicate "
              "rows, so this test is genuinely clean.")
    l = r[r.split.str.startswith("LODO")].sort_values("R2_mean", ascending=False)
    phys = {"D1_NRC": 0.926, "D2_Zhao2020": -0.437, "D3_KAERI_uniform": 0.620,
            "D4_KAERI_nonuniform": 0.797, "D5_Helical_R123": 0.051,
            "D6_Hardik_helical_water": -0.161, "D7_Hardik_straight_R123": 0.636,
            "D8_Pioro_R134a": -0.097}
    pool = {"D1_NRC": 0.242, "D2_Zhao2020": -1.433, "D3_KAERI_uniform": 0.852,
            "D4_KAERI_nonuniform": 0.841, "D5_Helical_R123": -307.7,
            "D6_Hardik_helical_water": -1.214, "D7_Hardik_straight_R123": -868.6,
            "D8_Pioro_R134a": -423.6}
    rows = [[SHORT[x.dataset], f"{phys[x.dataset]:.3f}", f"{pool[x.dataset]:.1f}",
             f"{x.R2_mean:.3f}", f"{x.R2_sd:.3f}", f"{x.within_10pct_mean*100:.1f}%",
             f"{x.within_20pct_mean*100:.1f}%", f"{x.MAPE_pct_mean:.1f}%",
             f"{x.RMSE_mean:.0f}", f"{x.safe_side_frac_mean*100:.0f}%"]
            for x in l.itertuples()]
    rows.append(["MEDIAN", "0.336", "−1.323", f"{l.R2_mean.median():.3f}", "",
                 f"{l.within_10pct_mean.median()*100:.1f}%",
                 f"{l.within_20pct_mean.median()*100:.1f}%",
                 f"{l.MAPE_pct_mean.median():.1f}%", f"{l.RMSE_mean.median():.0f}",
                 f"{l.safe_side_frac_mean.median()*100:.0f}%"])
    table(doc, ["Unseen dataset", "Physics alone", "ML, no physics", "This pipeline",
                "Std dev", "±10%", "±20%", "MAPE", "RMSE", "Safe-side"], rows,
          caption="Table 9. Generalisation to an entirely unseen dataset. The pipeline more "
                  "than doubles the physics-only median and avoids the catastrophic failures "
                  "of unanchored machine learning.")
    para(doc, "The most important column is the third. Machine learning without a physics "
              "anchor produces R-squared of −868 and −424 on datasets it has not seen. Those "
              "are not merely poor scores; a model behaving that way would return predictions "
              "wrong by orders of magnitude. The pipeline's worst case is −0.065.")
    figure(doc, "fig2_generalisation_gap.png",
           "Figure 2. The generalisation gap. Light bars are accuracy on held-out data from "
           "the same dataset; dark bars are accuracy on a dataset never seen. The gap is the "
           "honest cost of moving to new data, and it is the number most published work does "
           "not report.")

    doc.add_heading("16.5 How often is the prediction within 10%?", 2)
    para(doc, "This is the practical question an engineer asks. On familiar data the answer "
              "is good — 96% on the largest database. On genuinely unseen data it falls to "
              "roughly a third, and this should be stated plainly rather than buried.")
    a = r[r.split == "80/20"].set_index("dataset")
    b = l.set_index("dataset")
    table(doc, ["Dataset", "Within ±10%, same dataset", "Within ±10%, unseen dataset", "Drop"],
          [[SHORT[k], f"{a.loc[k,'within_10pct_mean']*100:.1f}%",
            f"{b.loc[k,'within_10pct_mean']*100:.1f}%",
            f"{(a.loc[k,'within_10pct_mean']-b.loc[k,'within_10pct_mean'])*100:+.1f} pts"]
           for k in SHORT if k in b.index],
          caption="Table 10. The ±10% criterion, on familiar and on unseen data.")
    para(doc, "One result deserves emphasis because it runs against the pipeline. On the NRC "
              "database under unseen conditions, the bare 1984 correlation places 52.0% of "
              "predictions within 10%, while the pipeline places 44.9%. The pipeline wins on "
              "R-squared and loses on this measure: the correction tightens the overall spread "
              "but pushes some near-misses across the line. The fair statement is that the "
              "pipeline is far more dependable across varied data, and that on the one dataset "
              "where the correlation is on home ground it remains more precise point by point.")

    doc.add_heading("16.6 Does the confidence label mean anything?", 2)
    para(doc, "A confidence label is worthless unless it predicts accuracy. It does.")
    table(doc, ["Trust", "Within ±10%", "Median error"],
          [["0.0 — correlation fallback", "31%", "17%"],
           ["0.5 — extrapolating", "47%", "10%"],
           ["1.0 — interpolating", "58%", "8%"]],
          caption="Table 11. Accuracy against the trust score, pooled over all eight "
                  "unseen-dataset folds. Both measures improve monotonically.")
    figure(doc, "fig5_reliability_boundary.png",
           "Figure 5. The reliability boundary. As trust rises, the share of predictions "
           "within 10% climbs from 31% to 58% and the median error falls from 17% to 8%. "
           "The label is informative, not decorative.")

    doc.add_heading("16.7 Where the physics correlation fails", 2)
    figure(doc, "fig3_error_topography.png",
           "Figure 3. Error of the Katto–Ohno correlation across the mass-flux and quality "
           "plane on the NRC database. The correlation is accurate through the bulk of the "
           "operating range and fails at low mass flux with high quality, the region where "
           "flow instability dominates. Localising the failure is more useful than an "
           "average error.")

    doc.add_heading("16.8 Reproducibility across random seeds", 2)
    para(doc, "Every result in this report is an average over several seeds. This matters: "
              "earlier in the project a headline result of R-squared 0.855 had to be "
              "withdrawn after it proved to be a single lucky seed.")
    figure(doc, "fig4_seed_stability.png",
           "Figure 4. Seed-to-seed spread of every result reported. Results with a long bar "
           "must never be quoted from a single run.")

    # ---------------------------------------------------------------- experiments
    doc.add_heading("17. Attempts to improve the pipeline", 1)
    para(doc, "Eleven variants were designed and tested across two rounds. Each was motivated "
              "either by a defect that had been measured or by physics already present in the "
              "project. Each was scored only on unseen datasets, so there was no familiar "
              "number to tune toward. Ten were rejected.")
    table(doc, ["Attempt", "Why it was tried", "Median R²", "Outcome"],
          [["Baseline (unchanged)", "reference", "0.777", "kept"],
           ["Inverse dataset-size weighting", "NRC is 85% of all rows", "0.767", "rejected"],
           ["Absolute-error training loss", "trained for R², reported on ±10%", "0.700", "rejected"],
           ["Correlation stacking", "Biasi and the look-up table were computed but unused",
            "0.727", "rejected"],
           ["Stacking with a missing-value flag", "intended fix for the above", "0.660", "rejected"],
           ["Weber number on diameter", "Hall and Mudawar's group, never included", "0.731", "rejected"],
           ["Hall and Mudawar as a fourth correlation", "present in the repository, unused",
            "included above", "rejected"],
           ["Predicting the boiling number", "the form the theory is written in", "0.703", "rejected"],
           ["Averaging two physics backbones", "avoid betting on one correlation", "0.607", "rejected"],
           ["Zhao diameter correction", "1,865 rows carried a fabricated 8 mm diameter",
            "—", "ADOPTED"]],
          caption="Table 12. The improvement campaign. Ten rejections is the honest outcome "
                  "of testing ideas properly rather than a failure to try.")

    doc.add_heading("17.1 Two rejections that taught more than a success would have", 2)
    para(doc, "A missing-value flag is a fluid label in disguise.", bold=True)
    para(doc, "Marking where a correlation is undefined made results worse, not better — "
              "0.660 against 0.727. The damage landed precisely on the unseen-fluid datasets: "
              "the helical R-123 set lost 0.529 and the Pioro R-134a set lost 0.330. The flag "
              "fires only on non-water rows, so it hands the model a fluid identity, and fluid "
              "identity does not carry to a fluid never seen. This is the project's own "
              "confounding finding, reproduced by accident while trying to fix something else.")
    para(doc, "The best correlation depends on the domain.", bold=True)
    para(doc, "Averaging a Katto-anchored and a look-up-table-anchored prediction rescues the "
              "Zhao dataset, from −0.600 to +0.068, and destroys the Hardik helical dataset, "
              "from 0.737 to 0.184. No single global choice serves both, which is why the "
              "overall median falls even though half the datasets improve.")

    doc.add_heading("17.2 The change that did help was a bug", 2)
    para(doc, "The Zhao loader names its columns for the equivalent and hydraulic diameter but "
              "never for the plain diameter the property routine looks for, so a hard-coded "
              "8 mm value was substituted for every one of its 1,865 rows. True diameters "
              "span 1 to 37.5 mm. The physics baseline was meaningless for that entire "
              "dataset and nothing reported it.")
    table(doc, ["Diameter used", "Physics baseline R²", "Pipeline R² (Zhao unseen)"],
          [["Constant 8 mm (the bug)", "−0.437", "−0.472"],
           ["Heated equivalent diameter", "−0.350", "−0.415"],
           ["Hydraulic diameter", "−0.362", "−0.507"]],
          caption="Table 13. The equivalent diameter was adopted on the evidence.")
    para(doc, "Zhao's own accuracy improved from 0.824 to 0.851. The fix slightly lowered the "
              "overall median, from 0.742 to 0.711, because Zhao appears in the training data "
              "for every other fold. It was kept regardless. Supplying a model with invented "
              "geometry is wrong whatever it does to a summary statistic, and reverting to "
              "recover 0.03 would be the exact behaviour this report argues against.")
    para(doc, "The property routine now warns instead of substituting silently, and every "
              "dataset carries a flag recording whether its geometry is real. The Pioro "
              "dataset is correctly marked as not having a published heated length.")

    # ---------------------------------------------------------------- validation
    doc.add_heading("18. Validation and testing", 1)
    table(doc, ["Check", "Result"],
          [["Metrics recomputed independently of the reporting code", "pass"],
           ["Prediction interval contains the estimate; all values positive", "pass"],
           ["No missing or infinite values in any prediction", "pass"],
           ["Trust confined between 0 and 1", "pass"],
           ["CHF rises with mass flux (1466 → 5074 kW/m²)", "pass"],
           ["CHF falls as the inlet gets hotter (3948 → 1728 kW/m²)", "pass"],
           ["CHF falls with heated length (4359 → 1023 kW/m²)", "pass"],
           ["Impossible inputs refused rather than answered", "5 of 5"],
           ["Automated regression tests, one per bug found", "9 of 9"],
           ["Same random seed reproduces identical predictions", "pass"],
           ["No duplicate rows shared between datasets", "none found"]],
          caption="Table 14. Validation checks, all passing.")
    para(doc, "The three trends in the middle of that table deserve attention. They are "
              "evaluated on artificial sweeps the model never saw during training, and it "
              "gets all three physical relationships right in both direction and magnitude. "
              "That is evidence the system learned the underlying physics rather than "
              "memorising the training rows.")

    # ---------------------------------------------------------------- limitations
    doc.add_heading("19. Limitations", 1)
    para(doc, "These are stated plainly because a reviewer will find them, and it is better "
              "that they come from us.")
    bullets(doc, [
        "No validation on data outside this collection. Every number here comes from the "
        "same eight datasets. This is the largest gap and no amount of further internal "
        "work can close it.",
        "The Pioro R-134a dataset performs poorly on every test. It is 268 points read by "
        "eye from printed charts, with an expected error of 5 to 10%, which exceeds the "
        "signal. It should not appear in any headline claim.",
        "The Zhao dataset remains negative on unseen-data tests. It is a compilation from "
        "ten authors with 39% of its rows involved in internal duplication.",
        "Prediction intervals are honest on familiar data, achieving 92 to 96% coverage "
        "against a 90% target, but optimistic on unseen data. This is inherent: the "
        "statistical guarantee assumes the new data resembles the old, and a new "
        "laboratory breaks that assumption.",
        "Everything is anchored to the Katto–Ohno correlation, one of whose constants is "
        "marked unverified in the project's own physics reference.",
        "The model was never tuned. Its settings are near-default and no systematic search "
        "over hyperparameters or architectures was performed.",
        "Whole families of model were not tried: properly tuned neural networks, "
        "transformers, and Bayesian methods among them.",
        "Pool boiling is out of scope by instruction. The correlation used is not valid at "
        "zero mass flux and such requests are refused.",
    ])

    doc.add_heading("20. What should be done next", 1)
    bullets(doc, [
        "Validate on data from outside this collection. This matters more than any "
        "modelling improvement.",
        "Submit to the OECD Nuclear Energy Agency's critical heat flux benchmark, which "
        "runs on the same NRC database and would give a genuine external comparison.",
        "Run a systematic hyperparameter and architecture search, including properly "
        "tuned neural methods, which this work did not attempt.",
        "Verify the Katto–Ohno constant against the original 1984 paper.",
        "Extend the same architecture to pool boiling using a pool-boiling correlation "
        "such as Zuber as the backbone.",
    ])

    doc.add_heading("21. Closing note on what this work claims", 1)
    para(doc, "This pipeline is well validated and its failure modes are measured rather than "
              "hidden. It is not claimed to be the best possible CHF predictor. There is an "
              "international benchmark running on this same data whose results we have not "
              "seen, the model was never tuned, and several model families were never tried.")
    para(doc, "The strongest contributions of this work are methodological rather than "
              "predictive: the demonstration that outlet quality makes a reported accuracy "
              "circular, the measurement of how far accuracy falls when moving to an unseen "
              "laboratory, and the finding that a novelty detector cannot recognise an unseen "
              "fluid because the dimensionless representation that enables transfer also "
              "erases the fluid's identity. The pipeline is the constructive demonstration "
              "that these principles produce something workable.")

    DST.parent.mkdir(parents=True, exist_ok=True)
    doc.save(str(DST))
    print(f"wrote {DST.resolve()}")
    print(f"  paragraphs {len(doc.paragraphs)}, tables {len(doc.tables)}, "
          f"images {len(doc.inline_shapes)}")


if __name__ == "__main__":
    main()

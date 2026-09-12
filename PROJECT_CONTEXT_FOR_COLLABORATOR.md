# CHF Project — Full Context, Problems Faced, and How They Were Tackled

**Purpose of this document.** A collaborator is writing the *research gap* and
*problems and difficulties* sections of the paper. This file is the complete
working context of the project: what was attempted, what broke, how each problem
was diagnosed, what fixed it, and what is still unsolved. It is written to be
read by someone who was not present for the work.

**Subject.** Machine-learning prediction of Critical Heat Flux (CHF) in flow
boiling. CHF is the heat flux at which a heated surface stops being wetted by
liquid — a vapour film forms, heat transfer collapses, and wall temperature rises
sharply. In a water-cooled reactor this limit sets the thermal margin of the
core, so both the accuracy of a prediction and the honesty of the accuracy claim
carry safety significance.

**Scope of the final phase.** Flow boiling only. Pool boiling was explicitly
excluded. Eight experimental datasets, 28,582 measured CHF points, plus the 2006
Groeneveld look-up table (11,088 usable rows) used as a reference correlation
rather than as training data.

---

# PART 1 — THE CHRONOLOGY

Each phase records what was attempted, the numbers, and **why the project moved
on**. The reasons matter more than the results: nearly every design choice in the
final pipeline exists because an earlier approach failed in a specific,
measurable way.

## Phase I — The look-up table

Started with the 2006 Groeneveld CHF look-up table: CHF as a function of
pressure, mass flux and thermodynamic quality. 11,592 grid rows (24 × 21 × 23).
504 rows have CHF = 0, all at quality 1.0 — an all-steam placeholder, not a
measurement — leaving **11,088 usable rows**. CHF spans 15 to 44,338 kW/m², three
orders of magnitude, which is why every model in the project trains on log(CHF).

Three splits were defined:

| Split | Construction | Question |
|---|---|---|
| A | random 80/20 | interpolation inside the grid (optimistic) |
| B | interior pressure holdout | moderate difficulty |
| C | edge pressure extrapolation | the honest test |

~30 model configurations evaluated.

| Model | Split A | Split B | Split C (verified) | MAPE on C |
|---|---|---|---|---|
| Grid interpolation (physical baseline) | — | 0.999 | **0.841** | 20.8% |
| Degree-2 log ridge | 0.847 | 0.871 | 0.755 | 35.8% |
| Neural network (log target) | 0.991 | 0.989 | 0.628 ± 0.072 | 40.0% |
| Tree ensembles (RF/XGB/LGBM/ET) | 0.999 | 0.988 | ≈0.43 | ≈42% |
| Pressure-gated blend (raw) | 0.999 | 0.999 | 0.466 ± 0.228 | 67.1% |

**Finding 1 — the tree collapse.** Models scoring 0.999 under random splitting
fall to ~0.43 under pressure extrapolation, while deterministic grid
interpolation holds 0.841. First evidence that random-split accuracy is a poor
guide to real generalisation.

**Finding 2 — a retraction.** An internal review had ranked a pressure-gated
blend first at R² = 0.855 on Split C. Multi-seed verification showed the driving
model averaged 0.547 across 30 seeds and **not one seed reached 0.84**. The claim
was withdrawn. Two implementation bugs were found in the same pass, including a
Split B definition that silently selected the topmost pressure level instead of
an interior one.

**Consequence:** from this point every number in the project is a multi-seed mean
with standard deviation, never a single run.

## Phase II — Physics-informed neural networks

Since models failed specifically at extrapolation, the question was whether
physical constraints could stabilise them. Penalties on collocation points: CHF
must not increase with quality; predictions must follow the Zuber pressure trend;
CHF must stay positive. A 720-configuration search ran on cloud GPUs, then three
refinement tiers.

| Tier | Configuration | Ensemble R² | MAPE | Beats 0.841 baseline? |
|---|---|---|---|---|
| 1 | architecture + monotonicity weight | 0.732 | 37.8% | no |
| 2 | + Zuber penalty, tanh | 0.770 | 37.2% | no |
| 2b | residual target, SiLU | **0.847** | **20.4%** | **yes** |

Tier 2b is the only configuration anywhere in this phase that beat the
deterministic baseline — and it did so by changing the **target
parameterisation**, not through any physics penalty. Seed spread also fell from
0.179 to 0.007 (25× more stable).

**The honest conclusion was negative.** Training logs showed the monotonicity
penalty read exactly `0.0000` throughout — it was never active. The best penalty
weights in Tier 2b were **zero for both physics terms**. So this phase is *not* a
fair test of physics-constrained learning. A separate headline of R² = 0.8123
from the 720-config search was also retired because its configuration had been
selected using the test set.

**Key reframing discovered here:** CHF has no governing partial differential
equation, so "physics-informed neural network" is the wrong template. The
technique that actually applies in this field is **residual/hybrid correction on
top of an empirical correlation**. This is the direct ancestor of the final
architecture.

## Phase III — Building a real dataset

35 reference papers indexed with citation, identifier, and whether extractable
data exists. Eleven publications extracted by hand using `pdftotext -layout`
(exact text extraction, not visual reading of rendered pages). A large database
believed restricted to benchmark participants was located in a public regulatory
document (NRC ADAMS ML22264A009, 457 pages) and transcribed in full — ~24,579
points, of which 136 proved to be exact duplicates.

**Sources deliberately rejected, and why:**

| Rejected | Reason |
|---|---|
| The look-up table as training data | Smoothed, interpolated, on a regular grid; also **not independent** of the NRC database, which is the data it was built from. Merging would double-count. |
| NUREG Table 4-1 sample (21 rows) | All 21 verified as exact duplicates of rows already held — both digitise the same Lowdermilk 1958 experiment. Including them would put identical rows under two source labels. |
| Helical minichannel logs (32 runs) | Recorded output is heater *power*, not heat flux; conversion needs per-geometry area assumptions too subjective to justify. |
| Source-summary tables (Serrao, Liang, supercritical, narrow-channel) | List parameter ranges and study metadata, no point-level CHF. |

A Gaussian-process generator densified the look-up table interior (7,114 of 8,000
points accepted). **Self-audit finding:** because every accepted point was
filtered for agreement with grid interpolation, this synthetic data is *not
independent* of that baseline and must never be used to claim a model outperforms
it. Retained for interior densification only.

## Phase IV — Cleaning, and what it exposed

| Defect | Consequence if uncorrected | Resolution |
|---|---|---|
| Units row beneath the header in an export | Every column read as text; max pressure returned the string "kPa" | Row detected and dropped, with an assertion verifying zero coercion failures |
| 136 exact duplicate rows | Identical experiments under different reference IDs, inflating random-split scores | Removed; both versions retained |
| Study labels only on the first row of each block | 90% of rows would carry no laboratory label, making source-wise validation impossible | Forward-filled; block contiguity verified independently |
| Pressure in Pa and heat flux in W/m² misread as kPa and kW/m² | Merged pressures of 180,000 bar, CHF of 28.8 GW/m²; **pressure became a perfect source indicator** | Units stated explicitly from source; all heuristic unit detection removed, replaced with plausibility assertions that abort the run |

Four conventions fixed here and held since: train on log(CHF); fit scalers inside
the CV fold; never impute a value a source does not report; exclude by rule any
column that reconstructs the answer.

## Phase V — Unified dataset and four split strategies

Seven point-level sources merged into one schema, one row per measured CHF, every
unit conversion stated explicitly from source documentation.

| Strategy | Construction | Question |
|---|---|---|
| 1 — random stratified | stratified by source | interpolation (what most published work reports) |
| 2 — condition-wise | per-source top-pressure holdout | unseen operating conditions |
| 3 — surface-wise | whole surface types withheld | unseen geometry |
| 4 — leave-one-source-out | each source held out in turn | unseen laboratory |

**Degeneracy detection added** after two split definitions were found to be
mislabelled. Every grouped split is now tested for all-singleton groups (which
make it identical to a random split) and for groups exceeding 95% source purity
(which make it a source split under a different name). **Both conditions had
occurred and were being reported as independent protocols.**

## Phase VI — The model bake-off

15 model families × 4 protocols.

| Model | 1 random | 2 condition | 3 surface | 4 source |
|---|---|---|---|---|
| Histogram gradient boosting | 0.968 | 0.890 | 0.173 | 0.784 |
| Random Forest | 0.965 | 0.875 | 0.219 | 0.269 |
| Neural network | 0.921 | 0.515 | **−4132.9** | −5.150 |
| Gaussian process | 0.872 | −0.051 | −0.813 | 0.539 |
| Physics-informed network | 0.876 | −64.9 | **−3906.9** | −6.022 |

Four conclusions:

1. **Neural architectures are catastrophically unstable on held-out surfaces**
   (−4,133). Not merely poor — the behaviour of a model returning arbitrary numbers.
2. **Gradient boosting is the most robust family.**
3. **Ensembling did not help.** A stacking ensemble scored 0.459 against 0.945 for
   a single Extra Trees model under honest splitting, because the meta-learner's
   weights are themselves fitted to the training laboratories.
4. **Simple models often win when extrapolating** — plain linear regression beat
   CatBoost, XGBoost and GP on one leave-one-publication-out evaluation.

## Phase VII — The turning point: source-level confounding

Direct question: are the relationships in these compiled databases genuine
physics, or artefacts of which laboratory produced each measurement?

| Dataset | Rows | Publications | Rows/pub | Variance from **source identity alone** | Gain from physical features |
|---|---|---|---|---|---|
| Engineered surfaces | 65 | 10 | 6.5 | **0.806** | +0.041 |
| KAERI | 888 | 11 | 80.7 | 0.735 | +0.187 |
| Pin-fin | 175 | 16 | 10.9 | 0.659 | +0.303 |
| NRC | 24,443 | 60 | 407.4 | 0.345 | +0.556 |
| Zhao | 1,865 | 10 | 186.5 | 0.273 | +0.511 |

On the smallest dataset, knowing **only which laboratory ran the test** explains
80.6% of the variance in log CHF. Adding roughness *and* contact angle — the
physical variables the paper is about — contributes a further 4.1 percentage
points.

**Severity scales inversely with measurements per publication** — a clean
dose–response across three orders of magnitude in sample size.

Two further results:

- **Pooling can reverse the sign of a physical relationship.** Roughness-to-CHF
  correlation flips from −0.293 pooled to +0.439 within-study, contradicting the
  capillary-wicking mechanism the source papers themselves invoke.
- **Standard diagnostics give no warning.** Feature-importance rankings are
  unchanged between honest and optimistic protocols. A model can present stable,
  physically plausible attributions while failing entirely to generalise.

**This phase is why every later result is reported under leave-one-source-out and
leave-one-dataset-out, not only random splits.**

## Phase VIII — Physics foundation

A 1,382-line reference document establishing provenance for every equation used
afterwards. Every equation carries a tag: transcribed verbatim from a primary
source, structure-verified only, or not obtained.

Three findings that drive the final design:

1. **The correct feature space is dimensionless** — water at 15 MPa and R-123 at
   1 MPa map to the same point there.
2. **Seven hard and ten soft physical constraints** are testable without new data.
3. **Inter-laboratory reproducibility is about 10%** (two independent multi-lab
   studies, 1970 and 1984/85). Random-split results sitting inside that band are
   measuring source-specific bias, not physics.

## Phase IX — Physics-guided models

Six arms, each adding exactly one idea to the one above.

| Arm | 1 random | 2 condition | 3 surface | 4 source |
|---|---|---|---|---|
| Physics only (Katto–Ohno), no learning | 0.779 | 0.508 | 0.505 | 0.785 |
| A0 — no physics, raw features | 0.979 | 0.903 | **−17.997** | 0.719 |
| A1 — latent-heat baseline | 0.970 | 0.877 | 0.711 | 0.800 |
| A2 — + Katto–Ohno baseline | 0.978 | 0.812 | 0.606 | −0.797 |
| A3 — + dimensionless features | 0.982 | 0.790 | 0.252 | −0.467 |
| A5 — + bounded, monotone, trust | 0.957 | 0.854 | 0.506 | 0.710 |

**Credit the data, not the models.** A dedicated attribution test separated data
repairs from modelling ideas. On the surface-wise split, correcting the data
(units, saturation properties, substituting a well-populated subcooling column
for one with 0.8% coverage) accounts on its own for **0.173 → 0.711**, before any
physics idea is applied. *Most of the apparent gain from physics-informed
modelling was in fact the value of correct data.* This lesson recurred in the
final phase.

Three results carried into the final design:

- **Non-dimensionalisation does not make models more accurate, it makes them
  portable.** On unseen refrigerants it improved error by 10–32×, while leaving
  within-fluid accuracy unchanged (R² 0.851 vs 0.849 — the control that proves the point).
- **Bounding a learned correction gives a monotonic, controllable trade-off**
  between safety on novel data and accuracy on familiar data.
- **Closed-form correlations beat every ML model in genuinely unseen regimes.**
  On axially non-uniform heating, a 1984 correlation with zero fitted parameters
  scored 0.821 where the best learned model scored −0.077.

**An open problem was left explicitly.** A novelty-weighted blend, calibrated
honestly on held-out training publications, **selected zero physics weight** —
yet that selection is demonstrably wrong for the non-uniform dataset. Cause is
structural: all training publications are water in straight tubes, so holding one
out never produces a point novel enough to reveal where the fallback earns its
value. **An out-of-distribution fallback cannot be calibrated using
in-distribution data, and out-of-distribution data is by definition unavailable
at calibration time.**

## Phase X — Transfer learning, ANNs and transformers

Two-stage pipeline: 300,000-row synthetic corpus from the look-up table for
pre-training, then fine-tuning on each real domain. Both an MLP and a transformer
encoder, plus LoRA and a mixture-of-experts variant.

| Architecture | Split | n | R² | MAPE | Within ±10% |
|---|---|---|---|---|---|
| MLP | interpolation | 2,459 | 0.951 | 17.9% | 47.3% |
| Transformer | interpolation | 2,459 | 0.957 | 16.5% | 47.7% |
| MLP | pressure extrapolation | 13,748 | 0.838 | 44.6% | 25.8% |
| Transformer | pressure extrapolation | 13,748 | 0.844 | 49.5% | 20.2% |

The transformer's advantage is marginal at ~30× the training cost.

Fine-tuning gave mixed evidence for transfer: on one refrigerant domain the
pre-trained network reached 0.481 against −0.171 from scratch; on another,
from-scratch won. **Pooling all fine-tuning domains was worse than separate
per-domain fine-tuning in 8 of 10 combinations**, one catastrophically at −27.9.

| Technique | Trainable parameters | R² on target |
|---|---|---|
| From scratch | 100% | 0.012 |
| Full fine-tune | 100% | 0.696 |
| **LoRA** | **8.8%** | **0.685** |
| Mixture of experts | 100% | 0.474 |

**Two decisive negative findings:**

1. **The MoE gate did not learn the regime it was routing on** — it fired at
   0.108 on pool-boiling rows against 0.609 on flow rows.
2. **Eight-member deep ensembles produced 95%-nominal intervals that actually
   contained the true value only 9–22% of the time.**

The first is why the final design refuses to route between models. The second is
why it uses conformal prediction rather than ensemble spread.

## Phase XI — Narrowing to flow boiling

Decision taken to stop broadening and build one pipeline properly. Every design
choice traces to a specific earlier failure:

| Design choice | The failure that motivated it |
|---|---|
| A physics correlation as backbone | Unanchored models reach −4,133 and −868 on unseen data |
| Dimensionless feature space | Raw-variable models cannot transfer across fluids; dimensionless improve error 10–32× |
| One pooled correction, never per-dataset routing | The MoE gate failed to learn its own regime |
| Trust limits the *correction*, never chooses a *model* | A switch costs a factor of 1000 when it chooses wrong |
| Conformal prediction intervals | Deep-ensemble intervals achieved 9–22% of nominal coverage |
| Multi-seed reporting throughout | A single-seed headline of 0.855 had to be retracted |
| Leave-one-source-out reporting | Source identity alone explains up to 80.6% of variance |

---

# PART 2 — PROBLEMS FACED AND HOW THEY WERE TACKLED

This is the section most relevant to a "problems and difficulties" write-up. Each
entry: **what the problem was → how it was discovered → how it was tackled →
outcome.**

## Problem 1 — Reported accuracy was circular

**Problem.** Every CHF database computes thermodynamic outlet quality *from* the
measured CHF via an energy balance. Feeding that quality back in as a model input
asks the model to recover arithmetic already done.

**Discovery.** Tested directly on the NRC database using the textbook energy
balance with real latent heat from CoolProp at each row's pressure, nothing fitted:

```
CHF = G · D · (x · h_fg + ΔH_in,sub) / (4L)
```

| Measure | Value |
|---|---|
| R² against 24,443 measurements | **0.999771** |
| Median relative error | 0.372% |
| Within ±10% | **100.0%** |
| Within ±1% | 77.3% |

**Tackled.** Three feature formulations were defined and run separately: *local*
(includes outlet quality — comparable to the literature but circular), *inlet*
(uses inlet subcooling/temperature instead — genuinely predictive), and *reduced*
(drops quality with no substitute — available everywhere).

**Important nuance discovered later.** The circularity is **not universal** and
was tested rather than assumed:

| Dataset | Does the identity close? | Quality circular? |
|---|---|---|
| NRC | Yes — R² 0.9998, 0.37% error | **Yes, proven** |
| KAERI uniform | Yes — R² 0.9894, 0.31% error | **Yes, proven** |
| KAERI non-uniform | **No** — R² −0.56 | **No** |
| 2006 LUT | Cannot be tested (no D, no L) | **No** |

**Outcome.** The final pipeline never uses outlet quality. Its NRC result
(R² 0.994, 96.2% within ±10%) is structurally non-circular.

**This is the single strongest methodological contribution of the project and the
most likely basis for a publishable claim.** Note for the paper: the identity
itself is Hall & Mudawar (2000b) Eq. 8 — conservation of energy — **not a novel
finding**. The contribution is measuring what it costs to remove it, dataset by
dataset.

## Problem 2 — Quality columns do not mean the same thing across datasets

**Problem.** We assumed all "quality" columns were outlet quality.

**Discovery.** On KAERI non-uniform the energy balance failed at ~100% median
error. Investigating, the column proved to be the **inlet** quality:
`(h_in − h_f)/h_fg` reproduces it on **all 888 rows to within 0.001**
(r = 0.999999). The same test on KAERI uniform misses by 0.375.

**Tackled.** Every dataset's quality column was verified individually — by the
energy balance where possible, by the source table header otherwise. The advisor's
appendix PDF literally prints "Exit Quality", settling three datasets.

**Outcome.** KAERI non-uniform's quality is an independent inlet measurement, not
circular, and was restored as a legitimate feature. Earlier treatment had been
discarding good information.

## Problem 3 — Columns that silently contain the answer

**Problem.** Several datasets carry columns algebraically equivalent to the target.

**Discovery/tackled.** Each suspected column was tested by recomputing CHF from it.

| Dataset | Column | Reconstruction | Quality |
|---|---|---|---|
| KAERI (both) | Power, Perimeter, Area, MassFlow, InletEnthalpy | `Power/(Perimeter×L)` | R² 0.999993, 100% within 10% |
| Hardik helical | `Q_W`, `CHF_ratio` | `Q_W/(πdL)`; CHF_ratio is CHF ÷ LUT | R² 0.979, 98.1% |
| Helical R-123 | `Q_watt` | same heat-input relation | confirmed per coil |

**Outcome.** All excluded **on evidence rather than suspicion**. A bonus: inverting
`Q_watt/(πdL) = CHF` per coil recovered tube diameters the extraction never had
(9.5, 7.5, 5.5, 9.5, 7.5, 5.4 mm, σ ≈ 0.002 mm). Diameter is a design parameter
known before the experiment, so using it is legitimate; `Q_watt` stays banned.

## Problem 4 — A fabricated geometry went unnoticed for the whole project

**Problem.** The Zhao loader names its columns `De_mm` and `Dh_mm` but never
`D_mm`, and the property routine silently substituted a hard-coded **8 mm** when
the column was absent.

**Discovery.** Found while auditing why Zhao was the pipeline's only persistent
failure. All **1,865 rows** carried a constant 8 mm against true diameters
spanning **1–37.5 mm**. The physics baseline was meaningless for that entire
dataset and nothing reported it.

**Tackled.** Added `D_mm`; tested equivalent vs hydraulic diameter on evidence.

| Diameter used | Physics baseline R² | Pipeline R² (Zhao unseen) |
|---|---|---|
| Constant 8 mm (the bug) | −0.437 | −0.472 |
| **Heated equivalent (adopted)** | **−0.350** | **−0.415** |
| Hydraulic | −0.362 | −0.507 |

**Outcome.** Zhao's own accuracy improved 0.824 → 0.851. Property defaults now
**warn instead of applying silently**, and every frame carries a `geom_known`
flag. Pioro is correctly flagged false — it genuinely publishes no heated length.

**Important honest note.** The fix *lowered* the headline median from 0.742 to
0.711, because Zhao sits in the training set for every other fold. **It was kept
anyway** — supplying a model with invented geometry is wrong regardless of what
removing it does to a summary statistic.

## Problem 5 — Unknown fluids silently used water's properties

**Problem.** The property routine had `fluid if fluid in CP else "water"`. A
request for ammonia returned a confident number computed from **water's**
density, surface tension and latent heat.

**Discovery.** Found during adversarial edge-case testing of the inference API.

**Tackled.** First fix attempt *didn't work* — the value came back byte-identical
(1960.0), which is how the failure was caught. Fixed at source.

**Outcome.** Verified by physics: CHF now tracks latent heat across fluids
(water 1960 kW/m² at h_fg 2015 kJ/kg; ammonia 916 at 1166; CO₂ 404 at 323;
R-134a 149 at 164). Side benefit: the pipeline now accepts **any CoolProp fluid**,
flagged as having no training data.

## Problem 6 — Duplicate rows inflating random splits

**Problem.** 578 rows inside datasets are identical in the model's feature space.
A naive random split puts some on both sides.

**Discovery.** Systematic duplicate audit during senior review.

| Dataset | Rows involved |
|---|---|
| Zhao 2020 | **722 of 1,864 (39%)** |
| Pioro | 74 |
| NRC | 62 |
| KAERI non-uniform | 46 |
| KAERI uniform | 12 |

**Tackled.** All random splits now grouped on a duplicate signature so identical
rows cannot straddle train and test.

**Outcome.** Measured inflation on Zhao: +0.033. **Pioro flipped from +0.066 to
−0.016** — its apparent positive score was entirely duplicates. Cross-dataset
duplicates were checked separately and found to be **zero**, so
leave-one-dataset-out is genuinely clean.

## Problem 7 — Splits that were not what they claimed

**Problem.** Three grouped splits were labelled "unseen laboratory" when the
group was something else entirely.

**Discovery.** Audited what each group column actually contains.

| Dataset | Group column | What it really is |
|---|---|---|
| NRC, Zhao, KAERI ×2 | reference / author / source | **genuine unseen laboratory** |
| Helical R-123 | appendix C vs D | pressure range |
| Hardik helical | Coil_1…6 | coil geometry |
| Pioro | Fig2a…Fig7c | figures within one paper |

**Outcome.** Renamed to what they are. This is the same degeneracy the project
had already documented once in Phase V — it recurred.

## Problem 8 — A routing architecture that failed twice

**Problem.** The natural design is a router: classify input as seen/unseen, send
seen data to the best model and unseen data to physics.

**Discovery.** Built and tested under leave-one-dataset-out with a
**pre-registered bar**: must beat both a global model and pure physics on every
held-out dataset.

| | Median R² |
|---|---|
| Pooled ML, no routing | −0.948 |
| Router v1 (continuous, dimensionless distance) | 0.099 |
| Router v2 (fluid/geometry identity gate) | 0.136 |
| **Pure physics** | **0.336** |
| Oracle (perfect routing, upper bound) | 0.831 |

**Root cause found.** The novelty gate is **blind to an unseen fluid**. Pioro's
R-134a appears in no training row, yet **neither** dimensionless-space nor
raw-unit distance flags a single one of its 268 rows:

| Held out | Fluid in training? | Flagged, dimensionless | Flagged, raw units |
|---|---|---|---|
| **Pioro R-134a** | **No** | **0.0%** | **0.0%** |
| Hardik straight R-123 | yes | 0.0% | 30.9% |
| Hardik helical | yes | 100.0% | 12.8% |

Its pressures, mass fluxes and diameter sit inside the water envelope, so every
continuous-distance measure calls it interpolation. **And dimensionless space
fails here by design** — its entire purpose is to make water at 15 MPa and R-123
at 1 MPa land on the same point. That is exactly right for *transfer* and exactly
wrong for *novelty detection*. **The two goals pull in opposite directions.**

**Tackled.** Abandoned routing. Replaced the switch with a **dial**: trust scales
a *bound* on the correction rather than choosing a model.

**Outcome.** Median 0.099 → 0.711. **This finding — that a novelty gate cannot
detect an unseen fluid because the representation enabling transfer erases fluid
identity — appears to be novel and is a strong candidate for the paper.**

**Methodological note worth recording:** iteration was stopped after two gate
designs deliberately. Building a v3 by combining what worked on each dataset
would be fitting the gate to the same held-out sets used to judge it.

## Problem 9 — Confidently wrong uncertainty

**Problem.** Deep ensembles produced 95%-nominal intervals containing the true
value only 9–22% of the time. Uncertainty that is confidently wrong is worse than
none.

**Tackled.** Replaced with split-conformal prediction on a held-out calibration
slice, widened by trust.

**Outcome, honestly stated.**

| Condition | Target coverage | Actual |
|---|---|---|
| Random splits (in-distribution) | 0.90 | **0.92–0.96** ✓ |
| Unseen dataset (out-of-distribution) | 0.90 | **0.57** ✗ |

Conformal guarantees rest on exchangeability, which a new laboratory violates.
**This is inherent, not a tuning failure, and remains an open problem.**

## Problem 10 — Improvements that did not work

Eleven variants tested over two rounds, each judged **only** on unseen datasets so
there was no familiar number to tune toward. **Ten rejected.**

| Attempt | Motivation | Median R² | Outcome |
|---|---|---|---|
| Baseline | reference | **0.777** | kept |
| Inverse dataset-size weighting | NRC is 85% of all rows | 0.767 | rejected |
| Absolute-error loss | trained for R², reported ±10% | 0.700 | rejected |
| Correlation stacking | Biasi/LUT computed but unused | 0.727 | rejected |
| Stacking + missing indicator | fix for the above | 0.660 | rejected |
| Weber number on diameter | Hall & Mudawar's group | 0.731 | rejected |
| Boiling-number target | the theory's native form | 0.703 | rejected |
| Backbone ensembling | avoid betting on one correlation | 0.607 | rejected |
| **Zhao diameter correction** | 1,865 rows had fabricated geometry | — | **ADOPTED** |

**Two rejections taught more than a success would have:**

1. **A missing-value indicator is a fluid label in disguise.** Adding one made
   results *worse* (0.660 vs 0.727), and the damage landed precisely on
   unseen-fluid datasets — Helical −0.529, Pioro −0.330. The indicator fires only
   on non-water rows, so it hands the model a fluid identity, and fluid identity
   does not transfer. **This project's own confounding result, reproduced by
   accident while trying to fix something else.**
2. **The best physics backbone is domain-dependent.** Ensembling rescues Zhao
   (−0.600 → +0.068) and destroys Hardik helical (0.737 → 0.184). No single global
   choice serves both.

**The headline lesson:** the only successful improvement in eleven attempts was a
**data-correctness bug**, not a modelling idea. This echoes Phase IX's attribution
finding exactly.

---

# PART 3 — THE FINAL PIPELINE

Five layers. Each exists because a measurement justified it.

**Layer 0 — Properties and dimensionless groups.** Fluid name + pressure → CoolProp
→ liquid/vapour density, surface tension, latent heat. Collapsed into five
dimensionless numbers: inverse Weber number (on heated length), density ratio,
L/D, subcooling ratio, reduced pressure. *The model never sees the word "water".*

**Layer 1 — Physics backbone, zero fitted parameters.** Katto–Ohno / Biasi / 2006
LUT, whichever has lowest median log error **on training rows only**. The floor
that prevents catastrophe.

**Layer 2 — One pooled correction.** ExtraTrees predicting `log(CHF/backbone)`.
Pooled, not per-dataset.

**Layer 3 — Trust scales the clip.** Trust 1.0 → correction allowed up to ×10;
trust 0.0 → clipped to ×1.5, so the prediction decays into the correlation.

**Layer 4 — Conformal interval + label** (*interpolating* / *extrapolating* /
*correlation fallback*).

**Inference API** takes only pre-experiment quantities. Refuses G = 0 (pool
boiling), negative geometry, supercritical pressure, unknown fluids.

## Results — same dataset, 80/20 random (10 seeds, duplicate-grouped)

| Dataset | n | R² | ±10% | ±20% | MAPE |
|---|---|---|---|---|---|
| NRC | 24,443 | **0.994** | **96.2%** | 99.0% | — |
| KAERI non-uniform | 888 | 0.973 | 81.4% | 93.1% | — |
| KAERI uniform | 651 | 0.928 | 88.5% | 92.6% | — |
| Hardik helical | 156 | 0.901 | 77.8% | 90.0% | — |
| Helical R-123 | 257 | 0.889 | 64.0% | 84.4% | — |
| Hardik straight | 55 | 0.883 | 58.2% | 81.8% | — |
| Zhao 2020 | 1,864 | 0.851 | 63.5% | 84.7% | — |
| Pioro R-134a | 268 | −0.016 | 13.7% | 25.5% | — |

70/30 agrees within ~0.01 throughout.

## Results — entirely unseen dataset

| Unseen | Physics alone | ML, no physics | **Pipeline** | ±10% |
|---|---|---|---|---|
| KAERI non-uniform | 0.797 | 0.841 | **0.970** | 47.5% |
| NRC | 0.926 | 0.242 | **0.944** | 47.0% |
| Hardik straight | 0.636 | **−868.6** | **0.793** | 33.8% |
| KAERI uniform | 0.620 | 0.852 | **0.735** | 58.9% |
| Hardik helical | −0.161 | −1.214 | **0.687** | 29.7% |
| Helical R-123 | 0.051 | **−307.7** | **0.449** | 14.8% |
| Pioro R-134a | −0.097 | **−423.6** | −0.065 | 16.0% |
| Zhao 2020 | −0.437 | −1.433 | −0.429 | 7.7% |
| **median** | **+0.336** | **−1.323** | **+0.711** | 31.8% |

**The headline is the absence of catastrophe**, not the median. Unanchored ML
reaches −868; the pipeline's worst case is −0.429.

## Validation

11 stress tests pass, including three physical-monotonicity probes on synthetic
sweeps the model never saw: CHF **rises** with mass flux (1466→5074), **falls** as
inlet warms (3948→1728), **falls** with heated length (4359→1023). 9 regression
tests, one per bug found. Same-seed reproducibility bit-identical.

**Trust is informative:** as trust goes 0→1, ±10% climbs 31%→58% and median error
falls 17%→8%.

---

# PART 4 — OPEN PROBLEMS (the actual research gaps)

These are unsolved and are the honest content of a "research gap" section.

1. **No external validation.** Every number comes from the same eight datasets.
   The single largest gap; no internal work can close it.
2. **Out-of-distribution uncertainty is unsolved.** Conformal coverage falls from
   0.92–0.96 in-distribution to 0.57 on an unseen laboratory. Exchangeability
   fails and there is no accepted fix.
3. **An OOD fallback cannot be honestly calibrated.** Calibration needs
   out-of-distribution data, which is by definition unavailable at calibration
   time. Documented in Phase IX, still open.
4. **Novelty detection cannot see an unseen fluid.** The dimensionless
   representation that enables transfer erases fluid identity. The two objectives
   are in direct tension and we have no solution.
5. **The best physics backbone is domain-dependent** and no global selection rule
   works.
6. **Two datasets remain unpredictable.** Zhao (a compilation, 39% internal
   duplication, 10 authors) and Pioro (268 eye-digitised points whose ±5–10%
   noise exceeds the signal).
7. **Anchored to an unverified constant.** Katto–Ohno's `C_Kc` is tagged
   `[S] UNVERIFIED` in the project's own physics reference.
8. **The model was never tuned.** Near-default settings, no hyperparameter or
   architecture search.
9. **Model families untried:** properly tuned neural networks, transformers within
   this pipeline, Bayesian methods.
10. **Not benchmarked externally.** The OECD/NEA runs a formal CHF benchmark on
    this same NRC data. We have not entered it and cannot compare.
11. **Pool boiling out of scope.** Katto–Ohno is invalid at G = 0; such requests
    are refused. Extending would need a pool-boiling backbone (Zuber).

---

# PART 5 — WHAT TO CLAIM, AND WHAT NOT TO

**Defensible claims:**

- Source-level confounding quantified across five datasets with a dose–response
  against rows-per-publication (0.806 at 6.5 rows/pub → 0.273 at 186.5).
- The local-vs-inlet formulation gap measured dataset by dataset — everyone knows
  the identity exists; quoting the honest cost of removing it is the contribution.
- A novelty gate cannot detect an unseen fluid because dimensionless space erases
  fluid identity. **Not seen elsewhere in the literature.**
- Closed-form correlations beating all ML in genuinely unseen regimes (0.821 vs
  −0.077 on non-uniform heating).
- 10–32× cross-fluid improvement from dimensionless formulation, with a
  within-fluid control showing no change.

**Do NOT claim:**

- That the energy-balance identity is a discovery. It is Hall & Mudawar (2000b)
  Eq. 8, conservation of energy, already in the project's own foundation document
  tagged as transcribed from a primary source. A reviewer at *Nuclear Engineering
  and Design* will recognise it instantly.
- That physics-anchored residual learning is novel. It is published (PIMLAF, and
  campaign-disjoint validation work).
- Best-in-class predictive performance. Not benchmarked externally, never tuned,
  several model families untried.

**Suggested framing.** A validation-methodology paper — *how CHF machine-learning
results should be evaluated* — with the pipeline as the constructive
demonstration, rather than a paper claiming a better predictor.

---

# APPENDIX — Datasets

| Dataset | Rows | Fluid | Geometry | Provenance |
|---|---|---|---|---|
| NRC / Groeneveld | 24,443 | water | vertical tubes | NRC ADAMS ML22264A009, 60 studies, deduplicated from 24,579 |
| Zhao 2020 | 1,864 | water | tubes, annuli, plates | 10-author compilation |
| KAERI non-uniform | 888 | water | tubes | TR-1665, non-uniform axial heating, 11 sources |
| KAERI uniform | 651 | water | tubes | TR-1665, uniform heating, 10 sources |
| Helical R-123 | 257 | R-123 | helical coils | advisor's PDF, appendices C + D (140 + 117) |
| Hardik helical | 156 | water | helical coils | advisor's PDF, table D.2 |
| Hardik straight | 55 | R-123 | horizontal tubes | advisor's PDF, table D.1 |
| Pioro R-134a | 268 | R-134a | horizontal + vertical | chart-digitised, ±5–10% |
| 2006 LUT | 11,088 | water | normalised 8 mm | reference correlation only, not training data |

Excluded as pool boiling: pin-fin (175), engineered surfaces (100), strip
experiments (55).

**Code:** `scripts/prof_tasks/` — `datasets.py`, `physics_arms.py`, `pipeline.py`,
`run_pipeline.py`, `final_review.py`, `test_pipeline.py`, `outline_figures.py`,
`improve_ablation.py`, `improve_round2.py`.
**Results:** `results/prof_tasks/` — `FINAL_results.csv`, `improve_*.csv`,
`figures_outline/`.

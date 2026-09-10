# Machine-Learning Prediction of Critical Heat Flux — Project Record

**B.Tech Thesis Project · Progress report to date**

A complete account of the work carried out: data acquisition and extraction, cleaning and preprocessing, model development across fifteen architecture families, four validation protocols, the discovery of source-level confounding, and the physics-guided methods developed in response.

**Scale of work:** 10 datasets integrated · 28,470 experimental records · 11 source publications extracted by hand · 35 reference papers reviewed · 15 model families evaluated · 13 exploratory analysis notebooks · 10 model-test notebooks

---

## 1. Objective and target

The objective, following the manuscript outline, is machine-learning prediction of the Critical Heat Flux (CHF) over **engineered surfaces** under diverse thermal-hydraulic conditions, using surface characteristics — roughness, wettability, contact angle, material, orientation — alongside operating conditions, with rigorous validation and explainability. The target venue is *Nuclear Engineering and Design*.

CHF is the limiting heat-transfer condition in a water-cooled reactor. Below it, nucleate boiling removes heat efficiently; above it a vapour film blankets the surface, heat transfer collapses and cladding temperature rises rapidly. Predicting it accurately sets the thermal margin of the core, so both the accuracy and the honesty of the accuracy claim carry direct safety significance.

The closest prior art was identified early and has shaped the whole programme: **Serrao et al. (2025), *Nuclear Engineering and Design* 435, 113924** — published in the target journal — builds a 101-point ATF pool-boiling database from 14 studies and trains SVR, Random Forest, MLP and Gaussian Process models on it, selecting Random Forest as best. Their reported validation uses a **random 80/20 split**. That single methodological detail became the pivot of this project's contribution, for reasons set out in Section 8.

---

## 2. Phase I — Look-up table modelling

The programme began with the **2006 Groeneveld CHF look-up table**, a standard tabulated reference giving CHF as a function of pressure, mass flux and thermodynamic quality. After cleaning this yields **11,592 rows** on a perfectly regular 24 x 21 x 23 grid, with 504 rows at quality = 1.0 carrying CHF = 0 correctly excluded as a placeholder boundary rather than a measurement. The surviving CHF range spans 15 to 44,338 kW/m2, roughly three orders of magnitude, which is why every model in the programme is trained on log(CHF) rather than raw values.

Three validation splits were defined to probe progressively harder generalisation:

- **Split A — random.** Interpolation within the grid.
- **Split B — interior pressure holdout.** Whole interior pressure levels removed from training.
- **Split C — edge pressure extrapolation.** Testing beyond the trained pressure range.

Roughly thirty model configurations were evaluated across linear, polynomial ridge, k-nearest-neighbour, Random Forest, Extra Trees, XGBoost, LightGBM, Gaussian Process, multilayer perceptron, physics-feature and residual-learning families, plus a grid-interpolation baseline.

### 2.1 Headline outcome and collapse under extrapolation

| Model | Split A | Split B | Split C (verified) | MAPE |
|---|---|---|---|---|
| Grid interpolation (physical baseline) | — | 0.999 | 0.841 | 20.8% |
| Degree-2 log-ridge | 0.847 | 0.871 | 0.755 | 35.8% |
| MLP (log target) | 0.991 | 0.989 | 0.628 +/- 0.072 | 40.0% |
| Tree ensembles (RF / XGB / LGBM / ET) | 0.999 | 0.988 | ~0.43 | ~42% |
| Pressure-gated blend (raw MLP) | 0.999 | 0.999 | 0.466 +/- 0.228 | 67.1% |

Split C values are means over 30 independent seeds where the model is stochastic. Grid interpolation and ridge are deterministic.

### 2.2 Retraction recorded during this phase

An earlier version of the internal review ranked the pressure-gated blend first at **R2 = 0.855** on Split C. Independent multi-seed verification showed this to be a single-seed artefact: across 30 seeds the driving model averaged **0.547**, and not one seed reached 0.84. The claim was withdrawn and the verification script retained in the repository. Two implementation bugs were also found and fixed in the same pass, including a Split B definition that silently selected the topmost pressure level rather than an interior one.

The durable finding from Phase I is the **tree-collapse contrast**: models scoring 0.999 under random splitting fall to roughly 0.43 under pressure extrapolation, while a simple deterministic grid interpolation holds 0.841. This was the first evidence that headline accuracy under random splits is a poor guide to real generalisation — a theme that recurs decisively in Section 8.

---

## 3. Phase II — Physics-informed neural networks

Following the extrapolation failures, physics-informed neural networks were developed to test whether soft physical constraints could stabilise the models. Penalties were imposed on collocation points sampled across the operating domain:

- **Monotonicity penalty** — CHF must not increase with thermodynamic quality.
- **Zuber pressure-trend penalty** — predictions must follow the hydrodynamic-limit trend with pressure.
- **Positivity** — CHF must remain physically positive.

A three-tier hyperparameter search was executed on cloud GPU infrastructure, covering architecture depth, activation function, target parameterisation (direct versus residual), penalty weights, collocation count and learning rate.

| Tier | Configuration explored | Best ensemble R2 (raw) | Log R2 |
|---|---|---|---|
| Tier 1 | Architecture and monotonicity weight; best (16,8), lambda_mono = 0.3 | 0.732 | 0.843 |
| Tier 2 | Plus Zuber penalty lambda = 0.1, tanh activation, direct target | 0.770 | — |
| Tier 2b | Residual target, SiLU activation | — | — |

### 3.1 Documented limitation

Review of the training logs established that the **monotonicity penalty read exactly 0.0000 throughout training** — it was non-binding and never activated. The physics constraint was therefore present in the loss function but inert in practice. This is recorded because it means Phase II does *not* constitute a fair test of physics-constrained learning; that test was only properly conducted in Phase IX.

A further scoping observation was made: the Groeneveld 2007 source paper documents a **Limiting Quality Region**, a steep non-smooth transition zone where the table's own authors report elevated entry-to-entry variation. No model in this phase was evaluated specifically inside that region, and it is flagged as a natural target for error characterisation.

---

## 4. Phase III — Data acquisition

### 4.1 Literature review

Thirty-five reference papers were assembled and indexed, each recorded with citation, DOI or publisher identifier, and a note stating whether extractable data exists and where any extraction was stored. Four papers assigned directly by the supervisor were audited first; three contained extractable appendix tables (55, 156 and 257 rows), and the fourth was confirmed to contain no data table at all — it is a wall-temperature and dryout-quality modelling paper whose only numeric content is one worked example and a correlation.

### 4.2 The engineered-surface extraction

The manuscript outline requires surface characteristics, which the large flow-boiling databases do not carry. Eleven publications were therefore extracted by hand, using exact text extraction rather than visual reading, to produce paired roughness, contact angle and CHF records.

| Source publication | Rows | Validation against independently published ranges |
|---|---|---|
| He, Ali and Chen, IJHMT 196 (2022) | 40 | Ra 1.55–25.21 match; CA 54–82 match; CHF 381–922 match |
| Ahn et al., NED 240 (2010) | 10 | Ra 0.05–0.32 match; CA 0–49 match; CHF 1004–1924 match |
| Yeom et al., NED 370 (2020) | 9 | Ra 0.07–5.72 match; CA 33–107 match; CHF 454–931 match |
| Ali et al., NED 362 (2020) | 8 | CA 68–83 match; CHF lower bound 677 match |
| Kim, Son and Kim, IJHMT 144 (2019) | 7 | Ra 0.093–0.107 match; per-specimen CHF figure-only |
| Kam et al., Annals of Nuclear Energy 76 (2015) | 6 | Ra match; CA 19–81 match; **CHF disagrees — see below** |
| Zhang et al., Nature Communications 14 (2023) | 6 | CA 0–85 match; CHF 980–2220 match; open access |
| Jo et al., NED 354 (2019) | 4 | Ra 0.067–0.12 match; CA 64–89 match; CHF 571–709 match |
| Cheol Lee et al., Annals of Nuclear Energy 126 (2019) | 4 | Ra 0.154–0.207 match; CA 0–65 match; CHF 804–1004 match |
| Ali et al., NED 338 (2018) | 4 | CHF 784–1089 match |
| Seo, Jeun and Kim, ETFS 64 (2015) | 2 | Ra 0.088/0.107 match; CA 85/93 match; CHF 635/1037 match |

Consolidated into a 100-row surface master file; 65 rows carry roughness, contact angle and CHF simultaneously. Ten of eleven extractions matched their independently published ranges exactly.

**Unresolved discrepancy — Kam et al. 2015.** The paper's own text reports CHF of 1020 (bare stainless steel), 940 (Zircaloy-4), 1230 (SiC 400 nm) and 1470 kW/m2 (SiC 1 micron). The independently published range for the same study quotes 660–1223. Values recorded here are verbatim from the source, so the disagreement is upstream and unresolved; resolving it requires digitising the paper's Figure 7. Neither number should be relied upon until then.

### 4.3 Large database recovery

The approximately 24,579-point NRC/Groeneveld CHF database, widely cited in recent literature as the largest public compilation, was initially believed to be restricted to registered participants of the OECD/NEA benchmark. It was subsequently located in a **public NRC ADAMS document** (accession ML22264A009, 457 pages) and transcribed in full. This correction is recorded because the earlier assessment appeared in project documentation and was wrong.

| Dataset | Rows | Sources | Character |
|---|---|---|---|
| NRC / Groeneveld database | 24,443 | 60 | Uniformly heated vertical water tubes; deduplicated from 24,579 |
| Zhao 2020 compilation | 1,865 | 10 | Tubes, annuli and plates; multi-author compilation |
| KAERI TR-1665 non-uniform | 888 | 11 | Non-uniform axial power profile |
| KAERI TR-1665 uniform | 651 | 10 | Uniform axial power |
| Helical coil R-123 | 257 | 2 | Coiled geometry, refrigerant |
| Pioro 2002 R-134a | 268 | 1 | Chart-digitised; horizontal and vertical orientation |
| Pin-fin surfaces | 175 | 16 | Water and FC-72; micro-fin geometry, the only large surface-feature set |
| Supervisor strip experiments | 55 | 1 | Pool boiling, water, orientation 0/90/180 degrees |
| Engineered-surface extraction | 100 | 11 | ATF claddings and coatings; roughness and contact angle |

### 4.4 Synthetic augmentation

A Gaussian-Process generator was built to densify the look-up table interior, fitting a Matern-5/2 kernel on 2,000 real points and sampling the posterior at 8,000 off-grid query points, filtered against a grid-interpolation reference. 7,114 points were accepted (88.9%). A subsequent self-audit established that because every accepted point was filtered for agreement with grid interpolation, **the synthetic data is not independent of that baseline** and must never be used to claim a model outperforms it. It is retained strictly for interior densification, and was excluded from all confounding analysis since synthetic rows have no laboratory of origin.

### 4.5 Deliberate exclusions

| Excluded | Reason |
|---|---|
| Groeneveld look-up table from the merged corpus | A smoothed, interpolated table on a perfectly regular grid, not independent raw measurement. It is also not independent of the NRC database, being constructed from the same historical experiments. Merging would double-count and bias training. |
| NUREG Table 4-1 sample (21 rows) | Merge validation established that all 21 rows are exact-value duplicates already present in the NRC database; both digitise the same Lowdermilk 1958 experiment. Inclusion would place identical rows under two source labels, a direct leakage risk for source-wise splits. |
| "CHF Dataset.csv" | Byte-identical duplicate of the pin-fin file. |
| Helical minichannel evaporator logs (32 runs) | Raw sensor time-series were parsed and a burnout detector applied, but the recorded output is heater power, not heat flux; conversion required per-geometry heated-area assumptions too subjective to justify for 32 rows. |
| Source-summary tables (Serrao, Liang nanofluid, supercritical, narrow-channel) | These list parameter ranges and study metadata but contain no point-level CHF. Retained as provenance and target lists only. |

---

## 5. Phase IV — Cleaning and preprocessing

### 5.1 Data defects found and fixed

| Defect | Consequence if uncorrected | Resolution |
|---|---|---|
| Units row beneath the header in the NRC export | Every column silently read as text; aggregate statistics returned nonsense, for example maximum pressure returning the string "kPa" | Row detected and dropped, with an assertion verifying zero coercion failures. Later replaced by a source-level fix; the reader now detects the row rather than assuming it, so it is correct on both file versions |
| 136 exact duplicate rows in the NRC database | Identical experiments appearing under different reference IDs, inflating random-split scores | Removed; 24,579 to 24,443. Both versions retained for inspection |
| Pin-fin study labels present only on the first row of each block | 90% of rows would have carried no laboratory label, making source-wise validation impossible | Forward-filled; block contiguity independently verified as 16 blocks summing to 175 rows with no label spanning two blocks |
| KAERI pressure in Pa and heat flux in W/m2 misread as kPa and kW/m2 | Merged pressures of 180,000 bar and CHF of 28.8 GW/m2; pressure became a perfect source indicator | Units stated explicitly from the source; all heuristic unit detection removed and replaced with physical plausibility assertions that abort the run |
| Mixed-type study labels across sources | Grouped cross-validation crashed on unsortable group arrays | Source-prefixed string labels throughout |

### 5.2 Preprocessing conventions

- **Target transform.** All models train on log(CHF). Values span three orders of magnitude; raw-target training lets large values dominate the loss.
- **Scaling.** Standardisation applied inside cross-validation pipelines for scale-sensitive learners (SVR, GPR, k-NN, neural networks, linear models) so the scaler is refitted per fold and never sees held-out data. Tree ensembles are scale-invariant and receive raw inputs.
- **Missing values.** Where a source genuinely does not report a quantity the field is left empty, never imputed. Pool-boiling records have no mass flux because nothing flows; that is a structural absence, not a missing measurement.
- **Fluid properties.** Saturation properties computed from CoolProp for water, R-134a and R-123, with a tabulated FC-72 supplement.
- **Leakage control.** Answer-side quantities excluded by rule: the KAERI Power, Area and Perimeter columns reconstruct heat flux exactly and are never used as inputs.

---

## 6. Phase V — Unified dataset and split strategies

Seven point-level sources were merged into a single canonical schema of one row per measured CHF, with unit harmonisation to metres, kPa, kg/m2s and kW/m2. Every conversion factor is stated explicitly from the source documentation; no unit is inferred. The merged corpus holds **28,470 experimental records** across 50 canonical columns including computed fluid properties.

| Strategy | Construction | Question it answers |
|---|---|---|
| 1 — Random, stratified | Stratified by source dataset | Interpolation within known conditions. The optimistic protocol used by most published work |
| 2 — Condition-wise | Per-source top-pressure holdout | Extrapolation to unseen operating conditions |
| 3 — Surface-wise | Whole surface types withheld (pin-fin, helical) | Generalisation to an unseen surface or geometry |
| 4 — Leave-one-source-out | Each source dataset held out in turn | Generalisation to an unseen laboratory |

### 6.1 Degeneracy detection

An automated check was added after two split definitions were found to be mislabelled. Every grouped split is now tested for all-singleton groups, which make a grouped split mathematically identical to a random one, and for groups exceeding 95% purity with respect to source labels, which make it a source split wearing a different name. Both conditions had occurred and were being reported as independent protocols; both are now flagged automatically in the results table.

---

## 7. Phase VI — Model bake-off

Models evaluated: linear regression, degree-2 polynomial ridge, k-nearest neighbours, support-vector regression with RBF kernel, multilayer perceptron, Random Forest, Extra Trees, XGBoost, LightGBM, CatBoost, Gaussian Process with Matern-5/2 kernel, a stacking ensemble, a histogram gradient-boosting machine, a physics-informed network, and a proposed two-tier hierarchy. Metrics recorded throughout: R2, RMSE, MAE, MAPE and training time.

| Model | S1 random | S2 condition | S3 surface | S4 leave-one-source-out |
|---|---|---|---|---|
| HistGB | 0.968 | 0.890 | 0.173 | 0.784 |
| Random Forest | 0.965 | 0.875 | 0.219 | 0.269 |
| Proposed hierarchy | 0.968 | 0.890 | 0.173 | 0.784 |
| ANN | 0.921 | 0.515 | -4132.9 | -5.150 |
| GPR | 0.872 | -0.051 | -0.813 | 0.539 |
| PINN | 0.876 | -64.9 | -3906.9 | -6.022 |

The collapse from column one to columns three and four is the central quantitative observation of the programme.

### 7.1 Findings

- **Neural architectures are catastrophically unstable on held-out surfaces**, reaching R2 of -4,133 (ANN) and -3,907 (PINN) under surface-wise splitting. These are not merely poor scores; a deployed model behaving this way would produce arbitrary predictions.
- **Gradient boosting is the most robust family**, holding 0.784 under leave-one-source-out where Random Forest falls to 0.269.
- **Ensembling did not help.** A stacking ensemble of Random Forest, XGBoost and SVR performed worse than a single Extra Trees model under honest splits (0.459 versus 0.945 on pin-fin), because the meta-learner's weights are themselves fitted to the training laboratories.
- **Simple models often win when extrapolating.** On the KAERI leave-one-publication-out evaluation, plain linear regression (0.605) outperformed CatBoost, XGBoost and Gaussian Process.

---

## 8. Phase VII — Source-level confounding

The repeated collapse under source-wise splitting prompted a direct investigation: are the surface-to-CHF relationships in compiled databases genuine physics, or artefacts of which laboratory produced each measurement?

### 8.1 Variance attributable to laboratory identity

| Dataset | Rows | Publications | Rows per publication | Variance from source identity | Gain from physical features |
|---|---|---|---|---|---|
| Engineered surface (ATF) | 65 | 10 | 6.5 | 0.806 | +0.041 |
| KAERI | 888 | 11 | 80.7 | 0.735 | +0.187 |
| Pin-fin | 175 | 16 | 10.9 | 0.659 | +0.303 |
| NRC | 24,443 | 60 | 407.4 | 0.345 | +0.556 |
| Zhao | 1,865 | 10 | 186.5 | 0.273 | +0.511 |

On the engineered-surface data, knowing only which laboratory performed the test explains 80.6% of the variance in log CHF. Adding roughness *and* contact angle contributes a further 4.1 percentage points.

### 8.2 Inversion of the physical relationship

Pooling laboratories does not merely weaken the surface signal; in two cases it reverses its sign.

| Dataset | Feature | Pooled correlation | Within-study correlation | Verdict | Studies agreeing |
|---|---|---|---|---|---|
| ATF surface | Roughness | -0.293 | +0.439 | sign reversal | 5 of 5 |
| Pin-fin | Subcooling | -0.092 | +0.537 | sign reversal | 6 of 6 |
| Pin-fin | Roughness factor | +0.156 | +0.466 | 3x understated | 13 of 13 |
| NRC | all seven features | — | — | no reversals | — |

The pooled roughness result contradicts the capillary-wicking mechanism invoked by every one of the source papers. Within each individual study the physically expected positive relationship is recovered.

### 8.3 Generalisation to an unseen laboratory

| Dataset | Random split R2 | Leave-one-publication-out R2 | Optimism gap |
|---|---|---|---|
| ATF surface | 0.776 | -0.043 | 0.802 |
| KAERI | 0.990 | 0.605 | 0.385 |
| Zhao | 0.945 | 0.599 | 0.346 |
| Pin-fin | 0.975 | 0.945 | 0.030 |
| NRC | 0.995 | 0.963 | 0.032 |

On the engineered-surface data, no held-out laboratory is predicted better than the global mean. Under a random split the same data appears to achieve R2 of approximately 0.78.

### 8.4 Mechanism and robustness

- A classifier recovers the **source publication from the input features alone at 83.1% accuracy**, against a 41.5% majority baseline. The features encode laboratory identity directly.
- The result survives independent reimplementation from scratch (-0.0735 against -0.073), true leave-one-group-out (-0.041), merging same-laboratory publications (-0.045), and stripping to only the two raw measured features (0.645 to -0.065).
- Proper mixed-effects estimation reverses the roughness coefficient from -0.009 (p = 0.22, not significant) to **+0.017 (p = 0.001, significant)**, while the contact-angle coefficient shrinks by 68%.
- **Feature-importance rankings do not change** between protocols. A model can present stable, physically plausible attributions while failing entirely to generalise, so importance plots offer no warning of this failure.

### 8.5 Stated limitations

The roughness reversal is sensitive to the high-roughness sandblasted samples: excluding Ra above 5 microns removes it. Cluster-bootstrap confidence intervals over ten publications are wide, and the contact-angle correlation is not statistically distinguishable from zero in either pooled or within-study form. The grouping variable identifies publications rather than laboratories; two pairs in the corpus share an author group, and two publications report identical CHF values, indicating a shared experimental campaign.

---

## 9. Phase VIII — Physics foundation

A 1,382-line foundation document was written to establish provenance for every equation subsequently used. Its structure covers the phenomenon itself; dimensional analysis as the basis of generalisation; the five pool-boiling trigger mechanisms; the physics of surface characteristics; flow-boiling correlations; geometry-specific effects; a sourced review of what machine learning has and has not achieved; the physical constraints any admissible CHF model must satisfy; and a proposed physics-first architecture.

| Correlation | Regime | Physical content |
|---|---|---|
| Zuber (1959) | Pool boiling | Hydrodynamic instability of vapour columns. Contains no surface terms, which is precisely why it removes almost none of the laboratory effect (0.820 to 0.799) |
| Kutateladze | Pool boiling | Dimensionless form of the hydrodynamic limit; Zuber's 0.131 constant is its flat-plate value |
| Kandlikar (2001) | Pool boiling | Force balance including contact angle and orientation. Reduces the laboratory-variance share from 0.820 to 0.474 |
| Haramura-Katto (1983) | Pool boiling | Macrolayer dryout beneath a coalesced bubble |
| Katto-Ohno (1984) | Flow boiling | Generalised correlation using the Weber number on heated length, density ratio and L/D. The principal transferability mechanism |
| Hall-Mudawar | Flow boiling | Alternative flow-boiling baseline for regime dispatch |
| Biasi (1967) | Flow boiling, water | Two quality branches; water-specific |
| Tanase diameter correction, coil factor | Geometry | Diameter normalisation and helical-coil adjustment |

A constraint scorecard (C1 to C7, S1 to S10) probes each trained model on synthetic parameter sweeps it never saw during training, verifying that learned functions respect known physical behaviour.

---

## 10. Phase IX — Physics-guided models

### 10.1 The ablation ladder

Six arms, each adding exactly one idea to the arm above, so that every contribution is isolated. Pure-physics arms carry no fitted parameters and form the reference every learned arm must beat.

| Arm | S1 random | S2 condition | S3 surface | S4 source |
|---|---|---|---|---|
| Physics only (Katto-Ohno), no learning | 0.779 | 0.508 | 0.505 | 0.785 |
| Physics only (gated), no learning | 0.733 | 0.809 | 0.506 | 0.751 |
| A0 — no physics, raw features | 0.979 | 0.903 | -17.997 | 0.719 |
| A1 — latent-heat baseline | 0.970 | 0.877 | 0.711 | 0.800 |
| A2 — plus Katto-Ohno baseline | 0.978 | 0.812 | 0.606 | -0.797 |
| A3 — plus dimensionless features | 0.982 | 0.790 | 0.252 | -0.467 |
| A4 — plus mechanism gating | 0.979 | 0.827 | 0.315 | 0.620 |
| A5 — plus bounded, monotone, trust | 0.957 | 0.854 | 0.506 | 0.710 |

**Attribution — credit the data, not the models.** A dedicated attribution test separates data repairs from modelling ideas. On the surface-wise split, the Stage-0 data repair, FC-72 saturation properties and the substitution of a well-populated subcooling column for one with 0.8% coverage account on their own for a rise from 0.173 to 0.711, before any physics idea is applied. Most of the apparent gain from physics-informed modelling is in fact the value of correct data.

### 10.2 Dimensionless formulation and cross-fluid transfer

Models were trained exclusively on water and tested on refrigerants never seen in training, comparing raw-variable prediction against prediction of the Boiling number from dimensionless groups.

| Held-out test | Raw-variable error | Dimensionless error | Improvement |
|---|---|---|---|
| Pioro R-134a (268 rows, unseen fluid) | 3,632% | 114% | 32x |
| Helical R-123 (257 rows, unseen fluid and geometry) | 796% | 81% | 10x |
| Within water (control) | R2 0.851 | R2 0.849 | none |

The control is the essential result: non-dimensionalisation does not make models more accurate, it makes them portable. Within a single fluid there are no property differences to cancel, and the two formulations are indistinguishable.

### 10.3 Bounded corrections and the fallback trade-off

The learned correction to a physical baseline was clipped to a maximum multiplicative factor, so that far from the training data the correction saturates and the prediction degrades gracefully into the correlation.

| Bound on learned correction | Novel data (non-uniform heating) | Familiar data |
|---|---|---|
| 1.5x (tightest) | +0.70 | -3.93 |
| 2x | +0.33 | -1.77 |
| 3x | -0.19 | +0.06 |
| Unbounded | -0.31 | +0.99 |

Strictly monotonic in the bound, and in opposite directions on the two data types. This quantifies the trade-off between physical safety and data-driven accuracy on a single controllable parameter.

### 10.4 Where physics wins outright

| Held-out dataset | Best method | R2 | MAPE | Runner-up |
|---|---|---|---|---|
| NUREG | ML on Katto groups | 0.993 | 3.1% | Katto alone -8.20 |
| KAERI uniform | Biasi correlation | 0.628 | 30.4% | ML 0.537 |
| KAERI non-uniform | **Katto-Ohno, no fitting** | 0.821 | 21.8% | ML -0.077 |
| Helical R-123 | ML | 0.205 | 34.6% | Katto -15.7 |

On axially non-uniform heating, a regime absent from training, a 1984 correlation with zero fitted parameters outperforms every machine-learning model by a wide margin. Conversely Katto fails on helical coils, correctly so: it is derived for straight tubes and does not model secondary swirl flows.

### 10.5 Adaptive fallback and an open problem

A novelty-weighted blend was constructed, measuring Mahalanobis distance from the training distribution and blending machine learning with physics in log space so that familiar points trust the learned model and novel points fall back continuously to the correlation. The blend width was calibrated honestly on held-out training publications, with test data playing no part.

The calibration selected **zero physics weight**, that is pure machine learning, because performance on held-out training publications increased monotonically with blend width. Yet that selection is demonstrably wrong for the non-uniform-heating dataset, where physics beats machine learning 0.82 to -0.08. The cause is structural: the training publications are all water in straight tubes, so holding one out never produces a point novel enough to reveal where the fallback earns its value. **An out-of-distribution fallback cannot be honestly calibrated using in-distribution data**, and by construction out-of-distribution data is unavailable at calibration time. The novelty measure itself is sound, correctly ranking the non-uniform dataset as most novel, but the calibration procedure discards that signal.

### 10.6 Methods evaluated and their verdicts

| Method | Verdict | Evidence |
|---|---|---|
| Dimensionless groups (Katto formulation) | Strong | R2 0.993 at 3.1% error; 10 to 32x cross-fluid improvement |
| Physics correlation as backbone | Strong | Wins outright where ML collapses (0.821 versus -0.077) |
| Bounded residual correction | Works as designed | Monotonic trade-off; catastrophic divergence made structurally impossible |
| Conservative quantile prediction | Conditional | 80% safe-side coverage in-distribution, falling to 2–20% out-of-distribution |
| Regime-specific monotonicity | Partial | Better than global constraints (0.77 versus 0.50) but still below unconstrained |
| Global monotonicity constraints | Harmful | Degrades performance broadly; the enforced direction is false near dryout |
| Mixed-effects machine learning | No advantage | Best case 0.70, below plain dimensionless 0.80 |
| Stacking ensemble | Harmful | 0.459 against 0.945 for a single Extra Trees under honest splitting |
| Automatic fallback calibration | Open problem | Selects zero physics weight; see 10.5 |

**Safety finding.** A quantile model trained to predict the 10th percentile, intended as a conservative design limit, achieves approximately 80% safe-side coverage on familiar data but only 2 to 20% on genuinely novel data. A statistically calibrated safety margin therefore provides false assurance precisely where it is most needed. This has direct implications for any deployment of learned CHF limits.

---

## 11. Phase X — Transfer learning

A two-stage transfer-learning pipeline was built: a 300,000-row synthetic corpus sampled from the look-up table for pretraining, followed by fine-tuning on each real target domain. Both a multilayer perceptron and a transformer encoder were implemented, along with LoRA low-rank adaptation and a mixture-of-experts routing variant.

| Architecture | Split | n | R2 | MAPE | Within +/-10% |
|---|---|---|---|---|---|
| MLP | Interpolation | 2,459 | 0.951 | 17.9% | 47.3% |
| Transformer | Interpolation | 2,459 | 0.957 | 16.5% | 47.7% |
| MLP | Pressure extrapolation | 13,748 | 0.838 | 44.6% | 25.8% |
| Transformer | Pressure extrapolation | 13,748 | 0.844 | 49.5% | 20.2% |

Extrapolation trains only on pressures at or below 8 MPa and tests entirely above it. The transformer's advantage over the MLP is marginal at roughly 30 times the training cost.

### 11.1 Fine-tuning against training from scratch

| Target domain | n | Pretrained and fine-tuned | From scratch | Benefit of pretraining |
|---|---|---|---|---|
| Hardik 2016 coils, R-123 (transformer) | 31 | 0.748 | 0.671 | positive |
| Hardik 2017 tubes, R-123 (transformer) | 11 | -0.053 | -1.143 | positive |
| Hardik 2016 coils, R-123 (MLP) | 31 | 0.562 | 0.633 | negative |
| KAERI uniform, water (MLP) | 130 | 0.908 | — | — |
| Pool boiling (transformer) | 11 | 0.775 | 0.774 | negligible |

**Caveats on this phase.** Stage-1 scores are measured on synthetic data drawn from the same corpus used for pretraining, so they quantify the ability to interpolate a smooth table rather than to generalise. The pool-boiling comparison rests on eleven test rows, where R2 is highly unstable. On that set, pretraining on 300,000 synthetic rows improved the transformer by 0.001 over training from scratch; the transfer hypothesis is not supported there, though it does hold on the two refrigerant domains.

---

## 12. Verification and quality control

An independent adversarial audit was commissioned against the analysis pipeline, instructed to write its own verification code rather than trust the existing scripts. It reproduced the headline result exactly (-0.0735 against a reported -0.073) and identified six defects, all since corrected.

| Defect | Severity | Effect if uncorrected | Status |
|---|---|---|---|
| KAERI units read as kPa and kW/m2 instead of Pa and W/m2 | Critical | Merged R2 overstated at 0.972 against a true 0.897; a reported condition-wise result of -18.0 was a pure artefact of the 1000x offset | Fixed; plausibility assertions added |
| Energy-balance recoverability in the NRC feature set | Major | CHF reconstructible from inputs at R2 0.9998 with no fitting, since quality is itself computed from CHF | Ablation added; local and inlet formulations now reported separately |
| Two split definitions mislabelled | Major | A surface-wise split was mathematically identical to a random split; a condition-wise split was a single-laboratory holdout | Automatic degeneracy detection added |
| Fluid indicator acting as a source proxy | Major | Pin-fin score of 0.945 fell to 0.620 once the water/FC-72 flag was removed | Ablation reported alongside |
| Grouping variable identifies publications, not laboratories | Major | Two publication pairs share an author group; two report identical CHF values | Terminology corrected throughout |
| Material-to-effusivity encoding order- and whitespace-dependent | Major | Coated samples assigned substrate properties; "SS 304" and "SS304" resolved differently, and the error tracked study membership | Rewritten; unknowns now return missing rather than a fabricated default |

### 12.1 Errors found and corrected internally

- **Weber number defined on capillary length instead of heated length** in the Katto correlation, an approximately 800x error that made the physics baseline appear useless and produced an incorrect early conclusion that physics did not help.
- **A fabricated receding-contact-angle conversion factor** (0.75x) with no literature basis, which degraded results from 44% to 72% error before being removed.
- **Fabricated Katto coefficients** in an early implementation, later replaced with the published form.
- **Biasi unit divisor** of 10,000 instead of 1,000, calibrated empirically against 4,000 training rows to a median ratio of 1.000.
- **Row-count discrepancy** in project documentation, 190 recorded against 175 actual.

Each is recorded because the pattern matters: every one produced plausible-looking numbers that would have survived casual review. The controls now in place — explicit unit declarations, physical plausibility assertions, degeneracy detection, range validation against independently published values, and adversarial audit — exist because these specific failures occurred.

---

## 13. Current position

### 13.1 Established findings

1. **Source-level confounding in compiled CHF databases is real, quantified and replicated across five datasets.** Its severity scales inversely with measurements per publication, from 0.806 at 6.5 rows per publication to 0.273 at 186.5.
2. **Random-split validation substantially overstates performance** on sparse multi-source data: R2 0.776 falling to -0.043 on the engineered-surface corpus.
3. **Pooling can invert the sign of a physical relationship.** The roughness-to-CHF relationship reverses direction, contradicting the mechanism the source papers themselves invoke.
4. **Standard diagnostics provide no warning.** Feature-importance rankings are unchanged between honest and optimistic protocols.
5. **Dimensionless formulation enables cross-fluid transfer**, improving error by 10 to 32 times on unseen refrigerants while leaving within-fluid accuracy unchanged.
6. **Closed-form correlations outperform all machine learning in genuinely unseen regimes**: 0.821 against -0.077 on non-uniform axial heating, with zero fitted parameters.
7. **The physics-accuracy trade-off is controllable** through a single bound on the learned correction, monotonic and in opposite directions on familiar and novel data.

### 13.2 Open problems

- **Automatic fallback calibration.** Selecting when to trust physics over machine learning cannot be calibrated on in-distribution data, and out-of-distribution data is unavailable at calibration time.
- **The engineered-surface corpus remains small**: 65 fully characterised rows across 10 publications, of which one contributes 40%. Two publications contribute a single row each.
- **Two directory conventions coexist** in the repository, a transfer-learning organisation and a held-out-evaluation organisation. These should be reconciled before publication.
- **The Kam 2015 CHF discrepancy** requires figure digitisation to resolve.
- **Helical-coil geometry** is not covered by any available straight-tube correlation and remains the weakest predicted regime.

### 13.3 Immediate next step

The physics machinery developed in Phase IX — regime-dispatched baselines, dimensionless formulation, bounded corrections, the Kutateladze pool-boiling form — has been validated on flow-boiling data but has not yet been applied to the engineered-surface corpus that is the manuscript's actual subject. Approximately 155 pool-boiling rows are now available for that purpose: the 100-row eleven-publication extraction together with the supervisor's 55-row strip experiments, carrying orientation, roughness and contact angle. Applying the validated toolkit to that corpus is the highest-value remaining work.

---

*Compiled from the project repository: 10 integrated datasets, 28,470 experimental records, 35 reference publications, 13 exploratory analysis notebooks, 10 model-test notebooks, and the full analysis and physics pipelines. All figures quoted are reproducible from the scripts and result files held in the repository.*

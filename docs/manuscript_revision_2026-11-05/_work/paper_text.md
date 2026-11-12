# Abstract
Accurate prediction of critical heat flux requires validation under changes in experimental source and operating conditions. This study retrospectively examines physics-anchored machine learning on eight flow-boiling datasets comprising 28,582 records for water, R-123 and R-134a. A physical baseline is combined with an ExtraTrees model of the logarithmic prediction residual using five dimensionless inputs. A heuristic trust score controls the allowed correction, and residual quantiles provide multiplicative prediction intervals. Random, grouped and leave-one-dataset-out evaluations are reported separately. In the five-seed dataset-holdout results, median R² is 0.711 and the median fraction within ±10% of measured CHF is 31.8%; Zhao and Pioro yield negative R². In a separate three-seed comparison, median R² changes from −2.061 for direct learning to 0.713 for an unbounded anchored correction and 0.708 for its bounded counterpart. Thus, anchoring is more strongly supported than the particular bounding rule. Nominal 90% intervals achieve only 53.2% median dataset-holdout coverage. Pairwise source-exclusion calibration improves median coverage but retains severe failures on individual datasets. Quality-dependent auxiliary calculations, reconstructed geometry, missing inlet conditions and digitized data restrict interpretation of these results as prospective validation. The findings support source-aware reporting and explicit assessment of interval coverage, while defining the limits of the evaluated implementation rather than establishing a universal CHF predictor.

Keywords: Critical heat flux; Flow boiling; Residual learning; Dataset shift; Prediction intervals

# 1 Introduction
Critical heat flux (CHF) marks a deterioration in boiling heat transfer that can cause a rapid increase in heated-surface temperature. Reliable CHF estimation is consequently relevant to nuclear fuel cooling and other high-heat-flux systems. The relationship between CHF and pressure, mass flux, thermal state and geometry depends on the applicable boiling regime. Departure from nucleate boiling and liquid-film dryout do not imply one universal response to all inputs. A model that reproduces measurements within one operating envelope therefore requires separate evidence before being used for a different geometry, fluid or experimental source.

Established approaches incorporate this dependence through empirical correlations and lookup tables. The 2006 CHF lookup table is normalized to a vertical water-cooled tube of 8 mm diameter and tabulates CHF as a function of pressure, mass flux and thermodynamic quality (Groeneveld et al., 2007). The Katto–Ohno formulation relates forced-convection CHF in uniformly heated vertical tubes to dimensionless groups and inlet subcooling (Katto and Ohno, 1984). These methods provide useful physical structure, but their original domains must be distinguished from additional domains in which they are subsequently evaluated.

Machine-learning models can represent nonlinear interactions among operating variables, and physics-assisted CHF prediction is already established in the literature. Zhao et al. (2020) combined physical prediction with machine learning, while Yang et al. (2026) investigated lookup-table pretraining followed by application-specific fine-tuning. Mondal and Sharma (2025) examined Bayesian neural prediction and uncertainty, including performance outside the original training and test population. These studies address different datasets, information sets and validation tasks; their reported coefficients of determination cannot be ranked as if they originated from one controlled benchmark.

Validation design is especially consequential for compiled CHF databases. Multiple nearby measurements can originate from a common test section or experimental campaign. A random record split can place related observations in both training and test sets, whereas withholding a campaign or source changes the prediction task. The OECD/NEA CHF benchmark provides a relevant setting for examining machine-learning methodology in nuclear engineering (Le Corre et al., 2024). More recently, Saraswat (2026) reported campaign-disjoint CHF validation, excluded outcome-derived inputs and evaluated physics-anchored residual transfer. The present study therefore does not claim priority for residual learning or source-aware CHF validation.

A second distinction concerns what information is available when a prediction is made. Local quality at CHF can contain information about the measured heat flux through an energy balance. Removing an explicit quality column from a regressor does not necessarily eliminate this dependence if quality also enters a physical baseline, a trust score or reconstructed metadata. Such pathways are examples of the broader need to examine the entire prediction procedure rather than only the named feature list (Kapoor and Narayanan, 2023).

This study evaluates the behavior and limitations of a physics-anchored implementation across eight heterogeneous flow-boiling datasets. The objectives are to quantify the difference between within-dataset and dataset-holdout performance, distinguish the effect of anchoring from that of bounding the learned correction, and assess empirical interval coverage under source shift. The analysis is retrospective. It does not establish an inlet-only prospective predictor, a pool-boiling model, a surface-morphology model or a reactor safety-margin method. Earlier lookup-table, neural-transfer and merged-data experiments are distinct research stages; supporting benchmark results are reported separately in the supplementary material.

# 2 Data and prediction scope
## 2.1 Flow-boiling corpus
The principal corpus contains 28,582 records distributed across eight datasets (Table 1). It includes water, R-123 and R-134a; straight and helical channels; uniform and nonuniform heating; and a small digitized orientation dataset. Figure 1 shows the imbalance in source sizes. The NRC subset contributes 85.5% of the records, so pooled row-level performance would be dominated by that population. Dataset-wise results and unweighted medians across datasets are therefore reported alongside sample counts.

{{TABLE1}}

The NRC loader uses 24,443 rows after deduplication of the 24,579-row experimental compilation. The Zhao loader retains 1,864 of 1,865 records after excluding one zero-mass-flux row and maps reported geometry to the enrichment variables. The two KAERI populations contain 651 and 888 records. Their source report concerns the effect of axial heat-flux distribution (Park, 2000), and the corresponding data collection is documented by Bourisaw et al. (2026). Counts refer to the local snapshots used in the stored experiments; they do not establish absence of overlap among the original experiments underlying different compilations.

The 156-row Hardik helical dataset uses water (Hardik and Prabhu, 2017). An earlier neural-transfer file name and loader incorrectly identified these same rows as R-123. Those affected neural results are excluded from the present flow analysis; the flow loader uses the correct water identity. The 55-row straight-tube dataset uses R-123 (Hardik et al., 2017). The 257-row helical R-123 population originates from an extended appendix collection associated with the helical-coil investigation; it is not represented as 257 points tabulated in the journal article itself (Hardik and Prabhu, 2018).

{{FIG1}}

The Pioro population consists of 268 points manually read from figures for R-134a in horizontal and vertical tubes (Pioro et al., 2002). The extraction record estimates approximately 5–10% reading uncertainty and notes that the extracted point density differs from the original measurements. This estimate is a provenance warning, not a calibrated measurement-error model. These rows are included in the recorded training sets when another dataset is held out and in within-dataset evaluation. They are not a pristine independent external-validation dataset.

## 2.2 Units and incomplete metadata
The scalar response is CHF in kW m⁻². Loader interfaces express pressure in kPa, mass flux in kg m⁻² s⁻¹, and diameter and heated length in mm. Geometry is converted to metres and enthalpy to J kg⁻¹ for physical calculations. Saturated liquid and vapour properties are evaluated through CoolProp (Bell et al., 2014). Fluid identity determines the property model, while reduced pressure and property ratios enter the numerical representation. The code clips property-evaluation pressure below the critical pressure and can return missing values when property evaluation fails; these operations are computational handling rules, not evidence that the original condition lies within a correlation's validity range.

Missing information is not uniformly removed. When inlet temperature or inlet enthalpy deficit is unavailable, the implementation assumes zero inlet subcooling. Missing heated length is assigned 1 m; this affects the digitized Pioro population. Shared enrichment code also provides an 8 mm diameter default when a diameter column is absent, although the corrected Zhao loader supplies its geometry explicitly. Nonfinite numerical inputs to the residual model are replaced with zero. The interpretation of the fitted model is consequently conditional on these preprocessing choices.

For the helical R-123 population, fixed per-coil diameters were reconstructed by inverting the heated-area relationship using heat input and measured CHF. A constant recovered from redundant measurements may correspond to a true geometric parameter, but this does not establish that it was independently available before observing held-out targets. Independent source confirmation of the coil mapping is required before these results can support a leakage-free prospective claim. The residual model also omits coil curvature, orientation and the full axial heat-flux distribution. These omissions matter when interpreting transfer across the corresponding datasets.

Source-version control is another limitation. The archived KAERI uniform XML matches the checksum of the retrieved dataset release, whereas the archived nonuniform XML differs from the updated release. This comparison establishes a version difference, not its numerical effect on the reported scores. All results in this paper refer to the archived snapshots. File checksums and the experiment-to-output mapping are supplied with the supplementary provenance material.

## 2.3 Local quality and the prospective prediction task
Quality has source-specific meaning. In the KAERI nonuniform loader, the reported quality variable denotes inlet quality; other sources provide local or exit quality. These quantities must not be treated as interchangeable simply because they share a column name. For a uniformly heated channel, an illustrative energy balance is

{{EQ1}}

where x is equilibrium exit quality, Δh_sub is inlet enthalpy deficit relative to saturated liquid, h_fg is latent heat, G is mass flux, L is heated length and D is the diameter used in the heated-area relation. Heat flux in this equation is in W m⁻². The exact relation depends on the geometry and heating assumptions. It is not asserted for every nonuniform or noncircular configuration in the corpus.

Equation (1) shows why quality measured at the CHF condition may encode the target. The implemented residual learner excludes quality from its five direct inputs, but some auxiliary physics arms still use it. Accordingly, the archived evaluations are described as retrospective prediction with the recorded input and auxiliary-calculation pathways. Their performance is not reinterpreted as the performance of a fully inlet-only deployment system.

# 3 Methods
## 3.1 Physical representation and baseline selection
The residual model uses the dimensionless vector

{{EQ2}}

where σ is surface tension, ρ_l and ρ_g are saturated liquid and vapour densities, and P_c is the critical pressure of the specified fluid. The first component is an inverse Weber-type group based on heated length. The five inputs describe a limited subset of the physics. They do not explicitly include viscosity, conductivity, contact angle, roughness, curvature or orientation. Additional transport-property groups are considered only in a separate sensitivity analysis.

The available physical predictions comprise the implemented Katto–Ohno correlation, a water-specific Biasi arm and the water CHF lookup-table arm. A candidate is eligible if it provides valid positive predictions for at least 90% of the outer training population. The baseline B is selected by minimizing the median absolute logarithmic error on that population. The selected-baseline and fixed-Katto columns coincide in all eight folds of the retained three-seed base-feature comparison. This observation avoids attributing the gain in that comparison to baseline switching.

The piecewise length-dependent coefficient in the Katto implementation is 0.25 for L/D below 50, increases linearly to 0.34 between 50 and 150, and is 0.34 above 150. These values agree with the original source (Katto and Ohno, 1984). Complete equivalence of the implemented branch-selection rules to the original conditional formulation has not been established. The baseline is therefore identified as the implemented Katto arm rather than an independently certified reproduction over every operating regime. Its use for coils and nonuniform heating also extends beyond the original uniformly heated vertical-tube setting.

## 3.2 Learning and bounding the correction
The learned response is the logarithmic ratio between measured CHF and the selected baseline,

{{EQ3}}

An ExtraTrees regressor with 300 trees, the recorded random seed and parallel tree construction estimates this residual (Geurts et al., 2006). The implementation uses the estimator's remaining defaults rather than an independently optimized architecture for every source. Approximately 20% of eligible outer-training rows, with a minimum of 30, are allocated to residual calibration after shuffling with the run seed. The regressor is fitted on the remaining rows. No neural network or LoRA adapter is part of this residual model.

The final point prediction is

{{EQ4}}

with the trust-dependent correction cap

{{EQ5}}

Here t lies between zero and one. At full trust, the allowed ratio of corrected prediction to baseline lies between 0.1 and 10. At zero trust, it lies between 1/1.5 and 1.5. Thus, the bound limits departures from the baseline but does not force an exact physics-only prediction. It also does not establish a lower bound on true CHF, conservatism with respect to measured CHF, or monotonic variation with operating variables.

## 3.3 Trust score and auxiliary information
The trust score combines distance from the fitting population, disagreement between available physics predictions and a fluid-identity check. Distance is calculated in log-transformed, clipped dimensionless coordinates using a median centre and regularized covariance. Its 50th and 95th training percentiles define a clipped normalization. Physics disagreement is the standard deviation of logarithmic arm predictions and is normalized using analogous percentiles. An unseen fluid name produces an additional novelty signal. The resulting score has the form

{{EQ6}}

where u_d and u_s are normalized distance and spread, and u_f is the unseen-fluid indicator. Omitting fluid identity from the residual feature vector therefore does not make the complete system independent of fluid labels. The thresholds t = 0.33 and 0.66 divide three descriptive trust bands; they are heuristic labels rather than validated probability statements.

The spread calculation can depend on local quality through the Biasi and LUT arms even if Katto is the selected baseline. Figure 2 includes this pathway explicitly. A later single-query interface closes a selected local-quality arm against an energy balance when measured quality is absent. That closure solves the baseline relation before applying the learned correction and does not recompute all disagreement terms afterward. Its implementation is not evidence that the archived batch results reproduce the prospective interface.

{{FIG2}}

## 3.4 Residual intervals and source-exclusion calibration
For each calibration row, the nonconformity score is the absolute logarithmic prediction error. The default implementation uses its empirical 90th percentile q_0.9 and increases the interval width as trust decreases:

{{EQ7}}

This is a multiplicative prediction interval for an observation, not a confidence interval for a mean response. The procedure is motivated by residual calibration, but the implementation does not establish the finite-sample guarantee of standard split-conformal prediction. Baseline selection precedes the calibration split, the ordinary empirical quantile is used rather than the explicit finite-sample order statistic, and the trust multiplier is heuristic. Exchangeability between calibration and future observations is additionally questionable under dataset shift (Lei et al., 2018).

A separate stored experiment uses pairwise source exclusion. For each target dataset d and each other dataset e, a model is fitted after excluding both d and e, and residuals are recorded on e. Pooling only those residuals for a fixed d prevents the target dataset from influencing its own calibration residuals through the fitting population. The symmetric implementation requires 28 pair-exclusion fits, followed by eight leave-one-dataset-out point-prediction fits. The audit uses seed 0. A further variant multiplies the cross-dataset residual quantile by 1 + (1 − t). These variants are evaluated empirically and are not claimed to guarantee dataset-conditional coverage.

## 3.5 Evaluation protocols
Table 2 distinguishes the evaluation protocols. Within-dataset random experiments use 80/20 and 70/30 partitions with ten seeds. The outer split groups duplicate signatures formed from the five dimensionless inputs and CHF, rounded to six decimals. This keeps identical input–target signatures together, but does not establish independence of nearby campaign observations or rows with identical inputs and different measured targets. Calibration within the training partition remains row-based.

{{TABLE2}}

The grouped evaluation uses the recorded source labels with at most ten GroupKFold folds and a model seed of zero. NRC has 60 reference groups, but the implementation evaluates ten folds rather than 60 individual leave-reference-out fits. Group meanings vary: authors or source identifiers for some water compilations, pressure bands for helical R-123, coil identity for Hardik water and digitized figure group for Pioro. No grouped result is available for the 55-row straight R-123 dataset. Grouped means and standard deviations describe folds, not repeated random seeds.

Leave-one-dataset-out evaluation withholds each complete dataset and trains on the other seven. Five seeds are retained for the main point-prediction summary. A separate three-seed comparison uses the same five numerical inputs to compare direct learning of log CHF, the selected physical baseline, an unbounded residual correction and the bounded correction. The direct learner uses all outer-training rows, while the residual learner reserves a calibration slice. The comparison is therefore aligned in source partition and feature set, but not in fitting-row budget. The unbounded and bounded residual predictions do use the same fitted residual model, differing only in the correction cap.

## 3.6 Performance measures and aggregation
For observed CHF y_i, predicted CHF ŷ_i and test-set mean ȳ, the coefficient of determination is

{{EQ8}}

RMSE and MAE are reported in kW m⁻². The mean absolute percentage error and fraction within ±10% are

{{EQ9}}

Empirical coverage is the fraction of observed values inside the reported interval. Negative R² indicates that squared prediction error exceeds the denominator defined by the test-set mean; it is not a percentage accuracy. A high R² can coexist with poor relative-error accuracy when the test population spans a broad CHF range.

Results are first summarized within each dataset and protocol. The median across eight datasets then assigns equal weight to each dataset, unlike a pooled row-level metric. Five-seed standard deviations quantify variation under the recorded seeds; they do not include uncertainty from dataset selection, digitization or measurement error. The grouped-fold standard deviation has a different interpretation. No common multi-seed claim is made across the distinct experiments.

## 3.7 Experimental provenance
The numerical tables are derived from stored result files and, where available, saved row-level predictions. The seven-source supporting benchmark was separately checked by recalculating its metrics from those predictions. This verification is not a retraining reproduction. The principal five-seed flow results, three-seed ablation and single-seed interval audit are kept as separate experiment families. Their values are not combined into a single matched model-ranking table.

The implementation uses Python, NumPy, pandas, scikit-learn and CoolProp; additional experiments use PyTorch and other learners. Scikit-learn provides the tree and validation implementations (Pedregosa et al., 2011). Exact environment versions and hardware were not preserved consistently for every historical run, and the repository dependency list specifies lower bounds. Consequently, this paper does not claim a controlled computational-cost ranking or full environment-level reproducibility. Supplementary material identifies the relevant source files, seeds and output checksums.

# 4 Results
## 4.1 Random and grouped evaluation
Table 3 summarizes R² for the 80/20, grouped and dataset-holdout protocols. Figure 3 gives the corresponding within-10% fractions. The protocols differ in both information retained from a source and training-population composition; their differences should therefore be read as changes in prediction task rather than as an isolated causal effect of one split operator.

{{TABLE3}}

{{FIG3}}

For NRC, mean R² decreases from 0.9935 under the 80/20 split to 0.9759 under grouped validation and 0.9443 when NRC is withheld as a dataset. The corresponding within-10% fractions decrease from 96.2% to 83.9% and 47.0%. The high dataset-holdout R² therefore masks a substantial loss of relative-error accuracy. The 70/30 NRC result remains strong within the same source population, with mean R² of 0.9932, but does not resolve this source-transfer gap.

Zhao behaves differently. Its mean R² is approximately 0.850 under the 80/20 split, whereas the mean across grouped folds is −2.233 and the dataset-holdout mean is −0.429. The grouped median is positive while one fold has a very large negative value. Reporting only the median or only the random-split result would hide this instability. The stored results demonstrate sensitivity to which source information is available during fitting.

## 4.2 Leave one dataset out performance
The five-seed dataset-holdout results are listed in Table 4 and plotted in Figure 4. Median R² across datasets is 0.7108, while the median fraction within ±10% is 31.8%. These summaries describe different aspects of error and should not be interpreted interchangeably. The median MAPE and absolute-error measures are supplied in the accompanying numerical provenance tables rather than inferred from R².

{{TABLE4}}

The largest R² values occur for KAERI nonuniform and NRC, while the strongest within-10% fraction occurs for KAERI uniform. Helical R-123 retains positive R² but only 14.8% of predictions lie within ±10%. Hardik water and straight R-123 have positive R², with relative-error accuracy remaining below 35%. Zhao and Pioro have negative R². These outcomes do not support a claim that one learned correction is reliable for every unseen source, fluid or geometry.

The reported five-seed values also do not establish independence from the originating experiments. Some source files are compilations, and cross-file overlap at campaign level has not been fully resolved. The dataset-holdout protocol is therefore more accurately described as withholding source files than as guaranteeing wholly unseen laboratories.

{{FIG4}}

## 4.3 Anchoring and correction bounds
Table 5 reports the three-seed base-feature experiment. The selected physical baseline is Katto in each fold. The direct ExtraTrees model predicts log CHF; the residual models predict log CHF relative to the physical baseline. Figure 5 compares the baseline and the two residual variants without compressing their useful range to accommodate the extreme direct-model failures.

Across datasets, median R² is approximately 0.336 for the baseline, −2.061 for direct learning, 0.713 for the unbounded correction and 0.708 for the bounded correction. The direct model has very poor transfer on several small refrigerant populations. Anchoring changes the learning task from predicting the absolute CHF scale to estimating a multiplicative correction and is associated with a much smaller failure magnitude in this experiment.

{{TABLE5}}

{{FIG5}}

The incremental effect of the bound is mixed. It improves Zhao and Pioro relative to the unbounded correction, is almost inactive on several datasets, and slightly reduces the Hardik-water score. The bounded method is best among the baseline, direct and bounded predictions on six of eight datasets; direct learning is better on KAERI uniform and the baseline is better on Zhao. The bound improves the worst R² relative to the unbounded correction, but it does not improve the median. The evidence supports describing the bound as a heuristic control on extreme departures, rather than as an established optimum or a source of universal predictive gain.

## 4.4 Empirical interval coverage
The nominal 90% intervals in the five-seed dataset-holdout experiment have median coverage of 53.2%. Coverage ranges from 13.7% for Zhao to 99.5% for NRC. Overcoverage on one large dataset does not compensate for undercoverage on another when the intended use is prediction for a new source. These intervals must therefore be reported with their measured coverage rather than described simply as calibrated 90% uncertainty estimates.

The separate source-exclusion audit is summarized in Table 6 and Figure 6. Its seed-0 default results differ slightly from the five-seed means. Pairwise source-exclusion calibration raises median coverage from 56.4% to 88.5%. Additional trust widening raises the median to 97.2%, but the minimum remains only 43.7%, on Zhao. Pioro remains below the nominal target even after widening. This is evidence that the calibration population matters, but not evidence that source-exclusion calibration solves arbitrary distribution shift.

{{TABLE6}}

The supplementary audit does not preserve a common interval-width score for all variants. Consequently, the increased coverage is not claimed to establish superior interval efficiency. Wider intervals can increase coverage without providing sufficiently precise information for engineering decisions. Both sharpness and source-specific coverage would be necessary for a complete comparison.

{{FIG6}}

## 4.5 Sensitivity and related analyses
The existing feature study tests extensions involving Reynolds, Prandtl, diameter-based Weber, Bond and Froude numbers and a viscosity ratio. Adding all available groups improves the direct learner substantially relative to its five-input form, while the anchored model does not show a universal improvement. Removing geometry-related information also changes performance. These observations indicate dependence on the representation, but do not by themselves prove or disprove target leakage in reconstructed geometry. The variants share the same sources and are not independent replications.

A separate comparison of ten residual learners finds a median R² of about 0.725 for CatBoost, 0.710 for XGBoost and 0.708 for ExtraTrees. Their worst-dataset results differ, and some learners use different seed counts or training caps. These comparisons support treating ExtraTrees as a practical selected implementation, not as a uniquely optimal learner. The complete stored summary is supplied in the supplementary material, without a claim that a nonsignificant difference establishes equivalence.

The leave-fluid-out audit contains 312 R-123 records and 268 R-134a records. The anchored model gives R² of approximately 0.464 and −0.061, respectively, with within-10% fractions of approximately 16.5% and 15.9%. All held-fluid observations receive low trust in that experiment. Because an explicit unseen-fluid indicator is part of the trust score, this result is not evidence that distance in dimensionless space alone detects every novel fluid.

# 5 Discussion
## 5.1 What the benchmark establishes
The principal finding is a separation between strong within-source interpolation and uneven transfer across source files. The physics anchor is associated with a large reduction in the severe failures of direct learning on the retained five-feature benchmark. This is a useful empirical result, but its interpretation is limited to the observed datasets and implementation. It does not show that every physical baseline is beneficial, that a particular learner is optimal, or that the same model can cover pool boiling and engineered surfaces.

The distinction between R² and relative error is particularly relevant for engineering interpretation. NRC retains R² above 0.94 when held out as a dataset, yet fewer than half of its predictions are within ±10%. Conversely, a low-variance subset may have negative R² even when some absolute errors are modest. Reporting absolute errors, relative errors and source-specific coverage together makes the limitations visible. A single pooled score cannot serve as a safety-margin criterion.

These findings complement existing physics-assisted CHF and campaign-validation studies rather than replacing them. Zhao et al. (2020), Yang et al. (2026) and Saraswat (2026) examine related ways to incorporate physical knowledge under different data and validation conditions. The contribution here is the documented comparison across heterogeneous source files, including the distinction between anchoring and bounds and the observed interval failures. It is not a controlled performance comparison against those publications.

## 5.2 Information availability and physical assumptions
The present results retain several pathways that complicate prospective interpretation. Quality-dependent auxiliary arms can influence trust even when quality is absent from the residual input vector. Source-specific definitions of quality differ. Fixed geometry was partly reconstructed using a relation involving measured CHF, and missing inlet conditions were treated as saturated inlet conditions. These choices can alter the apparent difficulty of transfer and the physical scale used for correction. They must remain visible in the interpretation rather than being hidden by the label physics-guided.

The digitized Pioro population adds uncertainty in both CHF and quality and combines orientation effects that are not explicit residual inputs. Its missing length is replaced by a default. The poor held-out score therefore cannot be attributed uniquely to fluid identity or to the regression algorithm. Likewise, the helical populations introduce curvature effects that the five-input representation does not describe explicitly. A correlation originally developed for straight vertical tubes should not be presumed adequate for these regimes without separate validation.

The model's positive output and finite multiplicative bound are algebraic properties. They do not imply physical correctness outside the baseline domain or conservative prediction relative to the true CHF. The recorded underprediction fraction, where used, is only the fraction of predictions below measured CHF. Establishing a nuclear-design margin would require a separately defined acceptance criterion, applicable validation evidence and an uncertainty allowance relevant to that decision.

## 5.3 Evaluation and reproducibility limits
The historical experiment campaign involved repeated inspection of held-out performance while preprocessing and model choices evolved. The existing holdouts therefore serve as development benchmarks as well as evaluation sets. They should not be presented as untouched confirmatory external tests. Source overlap remains incompletely resolved, and the small number of datasets limits statistical generalization. The seed standard deviations quantify one component of variability, not uncertainty over future experimental campaigns.

The scoring utility used by the five-seed summary contains a nonfinite-prediction replacement based on the median of the supplied true targets. If activated, that rule would introduce target information into the score. Persisted row-level predictions for those final runs are insufficient to determine whether it was activated. The separate three-seed ablation computes R² directly without that replacement, but shares other data and preprocessing limitations. This unresolved distinction prevents treating every archived summary as an independently validated deployment estimate.

Complete physical-formula verification also remains limited by the branch-selection question described in Section 3.1. Dataset checksums identify what was analyzed, but do not establish that an older source snapshot is free from errors subsequently corrected by its maintainers. Exact software versions, hardware and complete per-run predictions were not consistently archived. These limitations explain why the present analysis makes no claim of full retraining reproducibility or fair end-to-end runtime superiority.

## 5.4 Intervals and the role of trust
The interval results demonstrate that a calibration slice drawn from the fitting population can be inadequate for a new dataset. Standard conformal guarantees concern a specified probabilistic setting and should not be transferred automatically to a heuristic interval under distribution shift (Lei et al., 2018). The observed improvement from pairwise source exclusion is consistent with using residuals from a more relevant transfer task, but the remaining failures show that one pooled residual distribution can still inadequately represent particular targets.

Trust should therefore be interpreted as a model-specific diagnostic. It combines numerical distance, physics disagreement and fluid identity, and its thresholds have not been calibrated as probabilities of accuracy or safety. Low trust limits the correction but does not remove it. High trust does not guarantee accurate prediction or interval coverage. For the present implementation, the most defensible use of trust is to accompany measured performance and flag limited support, not to certify a prediction.

## 5.5 Scope of supporting experiments
The earlier seven-source benchmark contains pool-boiling and pin-fin data and uses a different target normalization, feature vector and set of split strategies. Its results are retained separately for traceability. They do not enlarge the physical scope of the five-input flow model. The earlier neural-transfer experiments likewise use an MLP or Transformer and different pressure cutoffs, training budgets and adaptation procedures. LoRA and mixture-of-experts routing from that work are not components of the ExtraTrees method reported here.

No final-model SHAP ranking is asserted. Existing random-forest SHAP analyses in the wider project concern different datasets and models, and an anticipated ranking of roughness or contact angle cannot explain a model that does not receive those inputs. Similarly, between-study variance fractions describe association with source identity; they do not isolate laboratory bias from operating conditions and geometry. Restricting these interpretations keeps the engineering conclusions aligned with the implemented experiment.

# 6 Conclusions
The retrospective evaluation of eight flow-boiling datasets shows that strong random-split CHF performance does not establish reliable source transfer. The five-seed leave-one-dataset-out summary has median R² of 0.711 and median within-10% accuracy of 31.8%, with negative R² on Zhao and Pioro. Dataset-specific errors remain essential even where aggregate scores appear favourable.

In the separate three-seed comparison, physics anchoring substantially improves the median and worst-dataset behavior relative to direct learning on the five base inputs. The particular correction bound improves some failures but does not improve median R² over the unbounded anchored model. Its role is a heuristic restriction on departures from the baseline, not an established physical guarantee.

Nominal 90% intervals show marked undercoverage under dataset shift. Pairwise source-exclusion calibration improves median coverage, but neither this procedure nor additional trust widening reaches the nominal target for every dataset. Quality-dependent auxiliary information, reconstructed geometry, missing metadata, source-version differences and incomplete run provenance further limit prospective interpretation. The work supports physics-anchored prediction as a useful research direction while identifying the evidence still needed before the evaluated implementation can support engineering deployment.

# Funding
This research received no specific grant from funding agencies in the public, commercial or not-for-profit sectors.

# Data and code availability
The source publications and dataset records are identified in the references and supplementary provenance material. The authors intend to release the code publicly. A public repository identifier has not yet been assigned. The compiled-data release is subject to confirmation of the original source terms and is not represented here as an unrestricted public deposit.

# Declaration of generative AI and AI assisted technologies
OpenAI Codex was used to assist with manuscript restructuring, literature and result cross-checking, language editing, and programmatic preparation of figures from stored numerical outputs. Claude Code was used during software development. The figures contain computed representations of the recorded results and schematic method descriptions; no experimental observations were generated by an image-generation model. Responsibility for the scientific content, interpretation and final approval remains with the human authors.

# References
{{REFERENCES}}

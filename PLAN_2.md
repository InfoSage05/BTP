# PLAN_2 - Leakage-safe CHF model improvement and external validation

## Status

Implemented as a reproducible first pass by `scripts/plan2_pipeline.py`. The
runner reads the two supplied files without modifying either source, writes
auditable artifacts under `results/plan2/`, and keeps the mentor workbook out of
the external evaluation.

## What the current model tells us

The existing model is good for interpolation inside the dense Groeneveld water
lookup table, but its headline random-split scores are not evidence of physical
generalization. The current honest high-pressure split is the relevant baseline:
raw GridInterp R2 = 0.8415 and MAPE = 20.85%; log GridInterp R2 = 0.8040. Tree
models collapse under pressure extrapolation, and neural-network scores are
seed-sensitive. The current best implemented source candidate is the residual-
on-GridInterp SiLU ensemble (reported log-target R2 about 0.892 on Split C),
but it must remain secondary until rerun with the same leakage controls and
multi-seed protocol as the deterministic baselines.

## Data contract and leakage controls

The workbook has 55 populated mentor experiments. The green columns found in
`Final Master file` are:

`Angle`, `L_effective(mm)`, `Width(mm)`, `Pnet`, `Tsat-Tpool`, `Surface tension`,
`rho_l`, `Cp`, `Kl`, `l/w`, `mu_l`, `alpha`, `Ja`, and `R`.

The label is `CHF(MW/m^2)`. The audit proves, row by row, that

`CHF = Pnet / (L_effective * Width)`

with zero maximum error. Therefore the literal all-green model is a leakage
control, not a deployable predictor. The primary model excludes `Pnet` and uses
the other 13 green fields only. Any future feature audit must also reject target,
prediction, fitted-correlation, or post-test quantities even if they are colored
green. The exact identity result is saved in `results/plan2/summary.json`.

## Implemented mentor-data evaluation

`scripts/plan2_pipeline.py` extracts the green rows and evaluates log-target
models using five-fold grouped cross-validation. Geometry groups normalize
orientation and combine orientation, angle, effective length, and width; there
are 24 groups. Random row splitting is not used as the primary estimate.

The first run produced:

| Model | Grouped R2 | RMSE (MW/m2) | MdAPE | Within 20% |
|---|---:|---:|---:|---:|
| Median baseline | -0.096 | 1.600 | 51.0% | 12.7% |
| Ridge, log target | 0.802 | 0.679 | 16.6% | 60.0% |
| Polynomial degree-2 Ridge, log target | **0.914** | **0.448** | 9.7% | 81.8% |
| Elastic Net, log target | 0.811 | 0.664 | 18.9% | 58.2% |
| Extra Trees control | 0.809 | 0.667 | **9.6%** | 74.5% |
| `Pnet/(L*Width)` leakage control | 1.000 | 0.000 | 0.0% | 100% |

The degree-2 Ridge result is a promising small-data model, not proof of final
performance: grouped five-fold uncertainty and the small sample size must be
reported. The next model-selection pass should use nested grouped CV, compare
raw versus log target, add bootstrap confidence intervals, and run leave-one-
geometry-out on the selected candidate. No tuning may use the PDF labels.

## Implemented PDF extraction and external test

The PDF contains four appendix tables and 468 records:

- D.1: 55 R123 straight-tube records.
- D.2: 156 water helical-coil records.
- D.3: 140 R123 low-pressure helical-coil records.
- D.4: 117 R123 high-pressure helical-coil records.

The parser handles continuation rows where the serial number or coil label is
visually omitted. It writes `results/plan2/external_pdf_data.csv` and verifies
the expected table counts. It also writes a diagnostic zero-shot evaluation of
the existing water LUT in `external_validation_report.csv`, using PDF pressure
converted from bar to kPa. The results are intentionally not presented as a
validated universal model: the LUT is a water, vertical 8-mm-normalized source,
whereas the appendix includes R123, helical coils, different diameters and
lengths, and different pressure/quality conventions. The diagnostic R2 is very
poor, as expected; this is evidence of domain shift, not evidence that the
parser or mentor model has failed.

The mentor-specific 13-feature model is not applied to the PDF because the PDF
does not contain those same thermophysical and geometric fields. Imputing them
would manufacture a result. A true unified model requires recovering tube and
coil geometry and fluid properties from the source paper/data record, or adding
those fields to the external table with traceable provenance.

## Next improvement stages

1. Freeze the leakage-safe data schema and add unit/range checks. Keep units
   explicit: workbook CHF in MW/m2, PDF CHF in kW/m2, pressure in bar in the
   appendix and kPa in the LUT.
2. Re-run the source LUT baselines, including raw/log GridInterp and the
   residual-on-GridInterp SiLU ensemble, across fixed seeds. Select using only
   a pressure proxy validation band, then evaluate Split C once.
3. On the mentor data, compare degree-2 log-Ridge, Ridge/Elastic Net, PLS, and
   an ARD Matérn Gaussian process inside nested grouped CV. Keep Extra Trees as
   an interpolation control, not as the extrapolation default.
4. Build the unified physics schema around pressure, mass flux, quality, heated
   length, tube diameter, coil diameter/ratio, orientation, fluid identity, and
   thermophysical properties. Derive dimensionless quantities only from inputs
   available before CHF is measured. In particular, never use Boiling number,
   supplied heat, or a computed CHF proxy if it contains the target.
5. Add valid-regime physical priors (Biasi/Bowring/Katto/Hardik-Prabhu or other
   source-appropriate correlations) only where their published inputs and
   validity ranges exist. Train a bounded log residual on the prior; compare
   against direct models and the prior alone. Do not extrapolate a correlation
   beyond its stated regime without an OOD flag.
6. Keep D.1-D.4 as zero-shot external tests first. Only afterward consider a
   leave-one-table-out experiment where one appendix table enters training;
   label it adaptation, not zero-shot validation. Mentor data may be used later
   for calibration/fine-tuning only after the zero-shot report is frozen.
7. Add uncertainty using bootstrap/deep ensembles or a calibrated Gaussian
   process. Report prediction-interval coverage and width, not just R2. An OOD
   prediction must carry a warning and should not be interpreted as calibrated.

## Research basis

The Groeneveld LUT literature describes a large, normalized water CHF database
and its applicability limits ([Groeneveld 2006](https://www.sciencedirect.com/science/article/pii/S0029549307002002);
[NRC NUREG/KM-0011](https://www.nrc.gov/reading-rm/doc-collections/nuregs/knowledge/km0011/index)).
The supplied coil appendix corresponds to the 413-point helical-coil dataset
described in the IIT/Mendeley record ([IIT source record](https://dspace.library.iitb.ac.in/jspui/handle/100/35944);
[Mendeley data](https://data.mendeley.com/datasets/f775wmnpkv/1)).
Residual learning against a domain-knowledge prior is supported by the hybrid
CHF framework of Zhao et al. ([ORNL publication](https://www.ornl.gov/publication/prediction-critical-heat-flux-using-physics-informed-machine-learning-aided-framework)),
while transfer learning from a LUT and uncertainty-aware ensembles are useful
follow-on methods ([transfer-learning framework](https://www.sciencedirect.com/science/article/pii/S1359431125040438);
[physics/UQ hybrid](https://arxiv.org/abs/2502.19357)). Dimensionless feature
engineering is promising only when every dimensionless group is computable
without the measured target ([2024 CHF ML study](https://doi.org/10.1016/j.ijheatmasstransfer.2024.125441)).

## Reproducibility and acceptance checks

Run `python scripts/plan2_pipeline.py`. Acceptance checks are: 55 mentor rows,
14 audited green columns, 24 geometry groups, zero target-identity error, PDF
counts 55/156/140/117, no duplicate parsed rows, positive finite model outputs,
and separate metrics for D.1-D.4. Future changes must also test missing geometry,
quality outside the LUT range, zero mass flux, pressure outside training range,
and that no target-derived feature enters `predict(df)`.

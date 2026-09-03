# Unified CHF Pipeline — Merge, Splits & Models

Self-contained folder for the variable-input CHF prediction pipeline. Builds
one master dataset out of all the point-level raw CHF sources, produces the
4 train/test split strategies needed to honestly evaluate cross-surface
generalization, and trains/evaluates the model bake-off (ANN, Random Forest,
HistGB, GPR, and the proposed Tier-0+Tier-1 hierarchy) against them.

This folder does not depend on `notebooks/`, `results/`, or `scripts/` at the
repo root — it reads directly from `data/raw/` and writes its own outputs
here.

## Folder layout

```
unified_chf_pipeline/
├── scripts/
│   ├── recipes.py         # one loader per raw source -> canonical columns
│   ├── merge_datasets.py  # runs all loaders, concatenates, validates
│   ├── build_splits.py    # builds the 4 split strategies
│   ├── features.py        # core feature matrix: CoolProp fluid properties +
│   │                       # geometry_family/fluid one-hot + sample weights
│   ├── models.py           # ANN / RandomForest / HistGB / GPR + the
│   │                       # Tier-0+Tier-1 hierarchy
│   ├── metrics_utils.py    # R2 / RMSE / MAE / MAPE / training time
│   └── run_experiments.py  # trains + evaluates everything, writes results/
├── data/
│   ├── master_chf_dataset.csv   # the merged output (one row per experiment)
│   └── merge_report.md          # row counts, coverage %, duplicate checks
├── results/
│   ├── metrics_summary.csv      # every (model x strategy[x fold]) row
│   ├── loso_fold_details.csv    # strategy-4 per-fold breakdown
│   ├── results_report.md        # formatted summary tables
│   └── predictions/             # y_true vs y_pred per model/strategy
└── splits/
    ├── strategy1_random_stratified.csv
    ├── strategy2_condition_wise.csv
    ├── strategy3_surface_wise.csv
    ├── strategy4_leave_one_source_out.csv
    └── splits_report.md
```

Run order:
```
python scripts/merge_datasets.py
python scripts/build_splits.py
```

## What got merged, and what didn't

**Merged (7 sources, 28,470 rows):** `nrc_groeneveld_24579pt`, `zhao2020`,
`kaeri_uniform`, `kaeri_nonuniform`, `pinfin_chf_water_fc72`,
`helical_coil_r123`, `mentor_master` — every one of these is real,
point-level experimental data (one row = one measured CHF).

**Deliberately excluded:**
- `CHF Dataset.csv` — byte-identical duplicate of `pinfin_chf_water_fc72.csv`.
- `nureg_km0011_table4-1_SAMPLE_ONLY.csv` — verified during merge validation
  that **all 21 of its rows** are exact-value duplicates already present in
  `nrc_groeneveld_24579pt` (both digitize the same Lowdermilk 1958
  experiment). Including it would have let identical rows appear under two
  `source_dataset` labels, a direct leakage risk for the surface-wise and
  leave-one-source-out splits.
- `data/chf_long_clean.csv`, `chf_long_with_gridbase.csv`,
  `groeneveld_2006_chf_lookup_table.xlsx` — this is the smoothed/interpolated
  Groeneveld **look-up table**, not independent raw measurements (verified:
  exactly 24 × 21 × 23 = 11,592 rows, a perfectly regular grid). Mixing a
  smoothed table in with raw scatter data would quietly bias training.
- All the "range/summary" files (`narrowchannel_chf_ml2025_source_summary.csv`,
  `furlong2025_nrc_vs_debortoli_range_comparison.csv`,
  `supercritical_co2/water_source_summary_luo2020.csv`,
  `nureg_km0011_table4-2_source_dataset_ranges.csv`,
  `tanase2009_diameter_correction_exponent_grid.csv`) — these describe
  *ranges* covered by cited sources, not individual data points; there is
  nothing row-level to merge.

## Canonical schema

Every row has: `row_id`, `source_dataset`, `geometry_family`, `data_type`,
`fluid`, `pressure_kPa`, `mass_flux_kg_m2s`, `quality`, `diameter_mm`,
`heated_length_mm`, `CHF_kW_m2` filled where the source reports it (NaN
otherwise), plus ~35 sparse optional columns (fin geometry, roughness,
coil info, thermophysical properties, etc.) that only a subset of sources
fill in. See `CANONICAL_COLUMNS` in `scripts/merge_datasets.py` for the full
list, and the `assumptions` column in the data itself for any value that had
to be assumed rather than read directly from a source file (e.g. pinfin's
pressure isn't reported — pool boiling, left blank rather than guessed).

`geometry_family` values: `tube`, `annulus`, `plate`, `helical_coil`,
`pin_fin_pool_boiling`, `flat_heater_pool_boiling`.

Note: **mandatory fields are per-`geometry_family`, not global** — e.g.
`diameter_mm` is essential for `tube`/`helical_coil` but doesn't apply to
`flat_heater_pool_boiling` (which uses `heated_length_mm`/`heater_width_mm`
instead). Any model built on top of this table should read mandatory-field
rules per family, not assume one fixed list for every row.

## The 4 split strategies

Each strategy is a small CSV keyed by `row_id` — never a copy of the data
itself. To use one: `pd.read_csv(master_path).merge(pd.read_csv(strategy_path), on="row_id")`.

1. **Random, stratified by source** (`strategy1_random_stratified.csv`) —
   80/20 split done independently per `source_dataset` so small sources
   aren't lost from the test set. The optimistic baseline.
2. **Condition-wise** (`strategy2_condition_wise.csv`) — per source, the top
   20% of `pressure_kPa` is held out as test (an extrapolation test).
   Sources with fewer than 30 rows with pressure data (`mentor_master`,
   `pinfin_chf_water_fc72` — neither reports pressure at all) go entirely to
   train.
3. **Surface-wise** (`strategy3_surface_wise.csv`) — `pinfin_chf_water_fc72`
   and `helical_coil_r123` are held out whole as test; everything else is
   train. Tests generalization to a surface type never seen in training.
4. **Leave-one-source-out** (`strategy4_leave_one_source_out.csv`) — records
   `fold = source_dataset` for every row; use
   `sklearn.model_selection.LeaveOneGroupOut(groups=master['source_dataset'])`
   (or filter `fold == X` vs `fold != X`) to rotate through all 7 sources as
   the held-out fold. `nrc_groeneveld_24579pt` (86% of rows) and
   `mentor_master`/`pinfin_chf_water_fc72` (< 1% each) will have very
   different statistical reliability as a single fold — report per-fold, not
   just averaged.

See `splits/splits_report.md` for the exact train/test row counts per
source for every strategy.

## Models

Run: `python scripts/run_experiments.py` (after the merge/splits scripts
above). Trains and evaluates 5 models on every split, writes
`results/metrics_summary.csv` (R2, RMSE, MAE, MAPE%, training time per model
x strategy), `results/loso_fold_details.csv`, `results/results_report.md`,
and per-row predictions under `results/predictions/`.

**Feature set** (`scripts/features.py`): pressure, mass flux, quality,
subcooling, diameter, heated length; fluid thermophysical properties
(density, viscosity, conductivity, Cp, surface tension, latent heat) computed
via CoolProp from `(fluid, pressure)` rather than using the fluid as a bare
label; and one-hot `geometry_family`/`fluid` flags. That last point was a
significant fix made mid-project: the model initially had **no explicit
signal for which physical regime a row belonged to** (tube vs. pin-fin pool
boiling vs. helical coil), and had to infer it weakly from which numeric
columns happened to be NaN. Adding it directly turned `helical_coil_r123`
from R2 = -10.3 into R2 = 0.93-0.96 on the random-split test, with no other
change.

**Models**: ANN (`MLPRegressor`, imputed+scaled), Random Forest (native NaN
handling), HistGradientBoosting (native NaN handling — also the Tier-0
algorithm below), GPR (imputed+scaled, capped to 800 training rows since fit
cost is O(n^3)), and the **proposed Tier-0 + Tier-1 hierarchy**: a
HistGradientBoosting baseline on the core feature set trained on every row,
plus one small GPR correction model per geometry family that has real
optional surface features (pin-fin, helical coil, flat-heater pool boiling),
predicting the residual on top of the Tier-0 baseline. Families with no
Tier-1 training data in a given split fall back to Tier-0 alone
automatically (verified: identical numbers to plain HistGB in the
strategy-3 test, where the held-out families have zero Tier-1 training data).

All models except ANN/GPR train with inverse-sqrt-frequency sample weights
(`sample_weights_by_source`) so the 86%-share `nrc_groeneveld_24579pt`
source doesn't drown out minority regimes during training (MLPRegressor and
GaussianProcessRegressor don't support `sample_weight` at all).

**Target transform — physics-normalized, not just log-transformed.**
Every model trains on `log(CHF / physics_baseline)` and predicts back via
`exp(.) * physics_baseline` (`features.compute_physics_baseline_kw_m2`,
`models._to_log_ratio` / `_from_log_ratio`), where `physics_baseline` is a
closed-form per-row physical CHF scale: the classical **Zuber correlation**
for pool-boiling rows, and `G * h_fg` (a Boiling-number-style scale) for
flow-boiling rows, both computed purely from CoolProp fluid properties --
needing zero training data for the target fluid. This replaced a simpler
plain `log1p(CHF)` transform, which helped in-distribution accuracy but did
nothing for cross-fluid extrapolation: a model predicting raw CHF has to
relearn each fluid's absolute magnitude from scratch, and R123's latent heat
is ~9x lower than water's -- almost exactly the size of the water-trained
model's original CHF overprediction on R123 test rows. Dividing by a
physics-computed scale before log-space training removes that mismatch up
front. (Two sources, `pinfin_chf_water_fc72` and `mentor_master`, report zero
system pressure at all -- both are open pool-boiling rigs, so
`ASSUMED_POOL_BOILING_PRESSURE_KPA = 101.325` is used **only** for this
physics-property lookup, never for the real `pressure_kPa` feature.)

Every prediction is clipped to `[0, 30,000]` kW/m^2 (`models.CHF_CLIP_MAX`)
-- log-space training makes an unbounded model (ANN, and to a lesser extent
GPR) capable of producing an astronomically wrong real-space prediction
under extreme extrapolation (observed during testing: a single such
blow-up turned one fold's R2 into -1,600,000 before this clip existed).
Tree models can't do this on their own (a tree's prediction is always
bounded by its training leaf values), which is part of why they're the more
robust choice here.

**Honest results, not just the good ones.** On the random-split test
(strategy 1), RandomForest/HistGB/Hierarchy land around R2 = 0.96-0.97.
The real target of this work was strategy 3 (surface types held out
completely) and leave-one-source-out -- and the physics-normalized target
moved those from uniformly catastrophic to a mix of genuinely positive and
still-hard results:

| Test | Before physics-normalization | After |
|---|---|---|
| Strategy 3 (RandomForest) | R2 = -16.4 | **R2 = +0.22** |
| Strategy 3 (HistGB / Hierarchy) | R2 = -14.3 | **R2 = +0.17** |
| LOSO `helical_coil_r123` (HistGB) | R2 = -409 | **R2 = +0.24** |
| LOSO `mentor_master` (HistGB) | R2 = -0.64 | **R2 = +0.03** |
| LOSO `pinfin_chf_water_fc72` (HistGB) | R2 = -2.01 | R2 = -0.04 (still negative, much closer) |
| Pooled LOSO (HistGB / Hierarchy) | R2 = 0.687 | **R2 = 0.784** |

Not every model benefited equally: **ANN remains catastrophically unstable**
under extrapolation regardless of clipping (pooled LOSO R2 = -5.15) and
should not be trusted for out-of-distribution predictions; RandomForest and
GPR show some fold-specific regressions under this transform (e.g.
RandomForest's `nrc_groeneveld_24579pt` LOSO fold, pooled R2 dropped to
0.269) that would need their own hyperparameter tuning to fully benefit --
not chased further here. **HistGB / Proposed_Hierarchy is the clear,
consistent winner** across both accuracy and robustness and is the
recommended model. See `results/results_report.md` and
`results/metrics_summary.csv` for the complete numbers.

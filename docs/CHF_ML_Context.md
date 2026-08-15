# Context: CHF Prediction from the 2006 Groeneveld Look-Up Table

## 1. Project goal

Build and rigorously validate several machine-learning models that predict Critical
Heat Flux (CHF) from three inputs — pressure (P), mass flux (G), thermodynamic
quality (X) — using the 2006 Groeneveld CHF Look-Up Table as the training data.

This is a Bachelor's Thesis Project (BTP). The deliverable for **this specific task**
is a well-documented, reproducible **Jupyter notebook** (`.ipynb`) that trains
multiple model families, evaluates them under both an "easy" (random split) and
an "honest" (structured, generalization-testing) protocol, and reports the results
in tables and plots. This is a scoped sub-task of a larger thesis; do not attempt
the full thesis roadmap (PINN discussion, external experimental validation, etc.)
unless explicitly asked — focus on producing the modeling notebook described below.

## 2. Files provided alongside this document

| File | What it is | Use it for |
|---|---|---|
| `chf_long_clean.csv` | Already-cleaned, long-format dataset: columns `P, G, X, CHF` | **Primary data source — load this directly.** |
| `2006_CHF_Lookup_Table.xlsx` | Original raw digitized table (source of truth) | Provenance only. Do not re-parse unless verifying `prepare_data.py`. |
| `prepare_data.py` | Tested script that turns the raw Excel into `chf_long_clean.csv` | Reference only — the CSV was already generated with it and verified. Re-run it only if you need to regenerate the CSV from scratch. |

**Do not write a new Excel parser from scratch.** The raw sheet has real quirks
(merged pressure cells requiring forward-fill, blank spacer rows between pressure
blocks, a repeated header row partway down the sheet). `prepare_data.py` already
handles all of this and has been verified to produce exactly 11,592 grid rows
with zero duplicates and zero nulls. Just load `chf_long_clean.csv`.

## 3. Dataset facts (already verified — use these to sanity-check your own loading)

- Grid shape: **24 pressures × 21 mass fluxes × 23 qualities = 11,592 rows**, no
  duplicates, no missing values.
- Columns: `P` (kPa, range 100–21,000), `G` (kg m⁻² s⁻¹, range 0–8,000, where
  G=0 is pool boiling), `X` (thermodynamic quality, range −0.50 to 1.00, negative
  = subcooled), `CHF` (kW m⁻²).
- **504 rows have CHF = 0, and every single one of them is at X = 1.0.** This is
  a physical boundary/placeholder (all-steam condition), not a real trainable
  CHF value. **Exclude all X = 1.0 rows before training** (`df[df.X != 1.0]`),
  leaving 11,088 usable rows.
- Non-zero CHF spans **15 to 44,338 kW/m²** — about 3 orders of magnitude. Because
  of this spread, **always test a log-target variant** (train on `log(CHF)`,
  exponentiate predictions back before computing metrics) alongside the raw-target
  version for every model. In initial testing, log-target consistently reduced
  MAPE by roughly 3–5x with little or no R² cost.
- The data is a **structured, near-fully-populated grid**, not scattered
  experimental points. This has a major consequence for evaluation — see
  Section 5.

## 4. Models to implement

Implement all of the following, each with both a raw-target and log-target
variant where relevant (trees generally don't need the log variant to work
well, but include it anyway for a complete comparison):

1. **Linear regression** (floor baseline — expect it to perform poorly, that's
   fine and expected, report it anyway).
2. **Polynomial regression** (degree 2 and degree 4, Ridge-regularized) on
   standardized inputs.
3. **k-Nearest Neighbors** regression (k=3, distance-weighted) — this is a useful
   "grid memorization" diagnostic, not a serious final model.
4. **Random Forest** and **Extra Trees** regressors.
5. **XGBoost** and **LightGBM** regressors.
6. **Gaussian Process Regression** with a Matern-5/2 kernel (ARD length scales
   per input dimension). Full GPR is O(n^3) — if training on the full ~9k-point
   training fold is too slow, subsample to ~2,000 training points for the GPR
   fit specifically, and note this clearly in the notebook.
7. **A compact MLP / feedforward neural network**: 2 hidden layers, 32–64 units
   each, tanh or ReLU activation, trained with Adam, early stopping on a held-out
   validation slice. Keep it small deliberately — do not use a large/deep network;
   justify in a markdown cell why complexity is kept low (3 inputs, ~9-11k
   training points, smoothed target).
8. **Trilinear grid interpolation** baseline using
   `scipy.interpolate.RegularGridInterpolator` on the full (P, G, X) grid. This
   is the "the look-up table interpolating itself" baseline and is expected to
   be extremely strong inside the training domain. Use `bounds_error=False,
   fill_value=None` so it also produces (linear-extrapolated) predictions for
   points outside the training grid, for use in the extrapolation test below.

All models must use `StandardScaler` (or equivalent) fitted **only on the
training fold** for any model that needs scaled inputs (linear, poly, kNN, MLP,
GPR). Tree models (RF/ET/XGBoost/LightGBM) do not need scaling.

## 5. Validation protocol — this is the most important part

**Do not evaluate any model with only a single random 80/20 split.** A random
split on a structured grid like this one is an "easy" interpolation test and
will make almost every model look excellent (R² > 0.99) regardless of whether
it has actually learned anything generalizable. This has already been confirmed
empirically on this exact dataset. Implement all three of the following splits
for every model, and present them as three separate result tables/sections in
the notebook — do not average or blend them together:

### Split A — Random 80/20 (labeled explicitly as "optimistic / interpolation test")
- Standard `train_test_split(test_size=0.2)`.
- **Run this with 5 different random seeds (e.g., 0–4) and report mean ± standard
  deviation of R² and MAPE for every model**, not a single-run number. This was
  already validated: e.g., Extra Trees gave R² = 0.9991 ± 0.0001 and Random
  Forest R² = 0.9987 ± 0.0002 across 5 seeds on this dataset — use this as a
  rough sanity check that your pipeline is working correctly, not as a target
  to hit exactly (small differences from hyperparameter choices are fine).

### Split B — Interior pressure-level holdout ("moderate honesty test")
- Hold out every 4th pressure level from the sorted unique pressure list (so the
  held-out levels have neighboring pressure levels on both sides still in
  training).
- Train on the rest, test only on the held-out pressure levels.
- Expect most models to still do reasonably well here since the held-out levels
  are "sandwiched" — this split mainly separates genuinely smooth models from
  ones overfitting to grid artifacts.

### Split C — Edge extrapolation holdout ("the honest test")
- Train only on P ≤ 16,000 kPa. Test only on the 5 highest pressure levels
  (17,000–21,000 kPa), which have **no neighboring training pressure on one
  side** — this is genuine extrapolation, not interpolation.
- **This is expected to be dramatic and is the most important result in the
  notebook.** In prior testing on this exact dataset: tree-based models (Random
  Forest, Extra Trees, LightGBM, XGBoost) collapsed to R² ≈ 0.10–0.44, because
  tree ensembles cannot predict outside the range of values they were trained
  on (they output a constant from the nearest leaf). Smooth models did far
  better: compact MLP (log-target) reached R² ≈ 0.74, degree-2 polynomial
  (log-target) ≈ 0.75, and trilinear/linear-extrapolated grid interpolation
  reached R² ≈ 0.84. **Reproduce this contrast and discuss it explicitly in a
  markdown cell** — it is the central, thesis-relevant finding, not a bug to
  "fix."

For every split, report both **R²** and **MAPE** (mean absolute percentage
error), computed on the raw (non-log) CHF scale even for log-target models
(exponentiate predictions first).

## 6. Reproducibility requirements

- Set and log every random seed used (numpy, sklearn, xgboost, lightgbm).
- Print library versions in the first notebook cell (`sklearn.__version__`,
  `xgboost.__version__`, `lightgbm.__version__`, `pandas.__version__`).
- Never tune any hyperparameter using the Split C (extrapolation) test data —
  that would be leakage into the very test meant to be honest. If you need a
  validation set for early stopping or hyperparameter choices, carve it out of
  the training portion only.
- Save all numeric results (not just print them) to a `results/` folder as CSV
  or JSON so they can be reloaded without re-running the whole notebook.

## 7. Plots to include

1. Predicted-vs-actual scatter (parity plot) for each model family, for Split A
   and Split C side by side (same model, two panels) — this visually makes the
   generalization-gap point.
2. A bar chart comparing R² across all models, faceted by split (A/B/C) — the
   single most important summary figure.
3. For at least the best tree model and the best smooth model: a 1-D slice plot
   of predicted vs. true CHF against quality X, at a fixed (P, G) inside Split C's
   held-out pressure region — shows qualitatively *how* each model fails or
   succeeds at extrapolation, not just the aggregate number.

## 8. Do NOT do these things

- Do not report only the random-split (Split A) results — always show all three
  splits together, with Split C given equal prominence, not buried.
- Do not tune hyperparameters on Split C.
- Do not train on the X = 1.0 rows (they're placeholder zeros, not real CHF
  values) — filter them out first.
- Do not claim the model "beats the look-up table" — the table itself is one of
  the baselines being compared, not an opponent to be defeated; the table's own
  interpolation is expected to win or tie on Splits A and B.
- Do not use a large/deep neural network "to be safe" — justify network size
  given the small input dimensionality and dataset size.
- Do not silently drop the log-target variants — the ~3-order-of-magnitude CHF
  range makes them relevant to every applicable model.

## 9. Suggested notebook structure

```
01_load_and_verify_data.ipynb section:
    - load chf_long_clean.csv, print shape, verify against Section 3 facts above
    - filter X == 1.0 rows, confirm 11,088 rows remain

02_eda section:
    - CHF vs X at a few fixed (P,G); CHF vs G at fixed (P,X); CHF vs P at fixed (G,X)
    - one 3D surface plot CHF(G,X) at a fixed P

03_models_split_A (random, 5 seeds, mean+-std table)

04_models_split_B (interior pressure holdout)

05_models_split_C (edge extrapolation holdout) -- most important section

06_comparison_and_plots (the bar chart + parity plots + slice plot)

07_conclusions (markdown cell summarizing: which model/family is best under
    which definition of "best", and why trees vs smooth models diverge so much
    under Split C)
```

A single well-organized `.ipynb` following this structure is the deliverable.
Splitting into multiple notebook files is acceptable if preferred, but keep a
single top-level notebook that runs everything and produces the final
comparison table and plots.

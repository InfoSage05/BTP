# data/ — folder map

## Actively used by the current training pipeline (don't move these)
- `chf_long_clean.csv` — the main training set: 11,592 rows, P/G/X -> CHF
  (from the 2006 Groeneveld LUT). Used directly by most notebooks/scripts.
- `chf_long_with_gridbase.csv` — same data + grid-interpolation baseline
  columns, used by the grid-interpolation model tests.

## data/raw/ — organized by intended usage, not by source
- `raw/pretraining/` — the single-lineage core dataset only: the
  Groeneveld LUT spreadsheet + the 24,579-row raw NRC tube database (same
  fluid, same geometry, same source lineage). Stage 1 / core training
  data. See its own README.
- `raw/fine_tuning/` — every other real (non-digitized) dataset, each a
  distinct fluid/geometry/heating regime from the core: the two Hardik &
  Prabhu R123 papers, the R123 helical-coil appendix data, pin-fin
  surfaces (FC-72), KAERI uniform/non-uniform heating, and the Zhao2020
  compilation. Stage 2 / transfer-learning fine-tuning targets — size
  varies a lot (55 to 1,865 rows), homogeneity with the core is what
  matters, not row count. See its own README.
- `raw/testing/` — **holdout only, never train or fine-tune on these.**
  Digitized scatter-plot points (~5-10% error), tiny illustrative samples,
  source-range/metadata tables with no point-level rows, and mentor-
  provided material kept separate from the main pipeline. See its own
  README for exactly why each file is there.

## data/processed/ — cleaned, derived, ready-to-train outputs
- (currently empty — nothing derived yet needs its own processed folder)

## Rule of thumb
- Need to add a new dataset from a paper/source? Decide its role first:
  - Real, complete, point-level data meant for the core model -> `raw/pretraining/`.
  - Real, small, distinct target-domain data meant for fine-tuning -> `raw/fine_tuning/`.
  - Digitized, sample-only, metadata-only, or otherwise not trustworthy
    for training -> `raw/testing/`, holdout only, never merged into
    training or fine-tuning.
- Need to derive/clean something from a raw file? Output goes in
  `data/processed/<dataset_name>/`, with its own short README explaining
  what to actually train on.

## Dropped
- Helical minichannel evaporator raw logs + processed CHF summary — removed.
  Output was heater power (W), not true heat flux; converting it required
  per-geometry heated-area assumptions that were too subjective to be worth
  it for 32 rows. Not used in training.

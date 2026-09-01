# data/ — folder map

## Actively used by the current training pipeline (don't move these)
- `chf_long_clean.csv` — the main training set: 11,592 rows, P/G/X -> CHF
  (from the 2006 Groeneveld LUT). Used directly by most notebooks/scripts.
- `chf_long_with_gridbase.csv` — same data + grid-interpolation baseline
  columns, used by the grid-interpolation model tests.

## data/raw/ — original, untouched source files
- `groeneveld_2006_chf_lookup_table.xlsx` — source spreadsheet for the two
  files above.
- `mentor_master_experiments.xlsx`, `external_coil_tube_chf_appendix.pdf` —
  mentor-provided source material (see PLAN_2.md).
- `raw/external/` — 17 extra CHF datasets/tables gathered from papers and
  public databases this session (tubes, helical coils, pin-fin surfaces,
  supercritical, diameter-correction factors, etc). Each file is a separate,
  distinctly-named dataset — never merged. Point-per-row training data vs.
  reference/range tables is not distinguished by folder, only by name/content
  — check each file's columns before using it.

## data/processed/ — cleaned, derived, ready-to-train outputs
- (currently empty — nothing derived yet needs its own processed folder)

## Rule of thumb
- Need to add a new dataset from a paper/source? Put the raw file in
  `data/raw/external/` with a distinctive name, never merged into another
  dataset.
- Need to derive/clean something from a raw file? Output goes in
  `data/processed/<dataset_name>/`, with its own short README explaining
  what to actually train on.

## Dropped
- Helical minichannel evaporator raw logs + processed CHF summary — removed.
  Output was heater power (W), not true heat flux; converting it required
  per-geometry heated-area assumptions that were too subjective to be worth
  it for 32 rows. Not used in training.

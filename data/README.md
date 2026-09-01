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
- `raw/external/` — real, ready-to-train CHF datasets (tubes, pin-fin
  surfaces, non/uniform heating, multi-source compilations). Each file is a
  separate, distinctly-named dataset — never merged. Check each file's
  columns before using it.
- `raw/external/paper_extracted_test_only/` — **testing/pipeline-validation
  only, do not train the reported model on these.** Everything here was
  pulled out of a research paper PDF (table-parsed or, for one file,
  digitized by eye off a scatter plot) rather than downloaded as a published
  dataset — small samples, source-range metadata tables, or approximate
  digitized points. See its own README for exactly why each file is there.

## data/processed/ — cleaned, derived, ready-to-train outputs
- (currently empty — nothing derived yet needs its own processed folder)

## Rule of thumb
- Need to add a new dataset from a paper/source?
  - A real, complete, ready-to-train dataset -> `data/raw/external/`.
  - Anything extracted/digitized from a PDF (table parse or chart
    digitization) -> `data/raw/external/paper_extracted_test_only/`, for
    pipeline testing only, never merged into the training set.
- Need to derive/clean something from a raw file? Output goes in
  `data/processed/<dataset_name>/`, with its own short README explaining
  what to actually train on.

## Dropped
- Helical minichannel evaporator raw logs + processed CHF summary — removed.
  Output was heater power (W), not true heat flux; converting it required
  per-geometry heated-area assumptions that were too subjective to be worth
  it for 32 rows. Not used in training.

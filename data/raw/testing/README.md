# data/raw/testing/

**Holdout / pipeline-validation only — never used for pretraining or
fine-tuning.**

Nothing here is trustworthy enough as training input: it's either
manually digitized (real error), a tiny illustrative sample, a
source-range/metadata table with no point-level rows, or mentor-provided
material kept separate from the main pipeline.

## Manually digitized (~5-10% error expected)
- `pioro2002_r134a_horizontal_vertical_chf_DIGITIZED.csv` — 268 points read
  by eye off scatter-plot figures. See `pioro2002_r134a_DIGITIZED_README.md`.

## Small samples / metadata / range tables (no usable point-level rows)
- `nureg_km0011_table4-1_SAMPLE_ONLY.csv` (21 illustrative rows only)
- `nureg_km0011_table4-2_source_dataset_ranges.csv`
- `tanase2009_diameter_correction_exponent_grid.csv`
- `supercritical_water_source_summary_luo2020.csv`
- `supercritical_co2_source_summary_luo2020.csv`
- `narrowchannel_chf_ml2025_source_summary.csv`
- `furlong2025_nrc_vs_debortoli_range_comparison.csv`

## Mentor-provided source material
- `mentor_master_experiments.xlsx`, `external_coil_tube_chf_appendix.pdf` —
  still referenced by `scripts/plan2_pipeline.py` (a separate exploratory
  pipeline, not the main training pipeline).

## Checked, nothing extractable
- Kefer, Kohler & Kastner (1989), *CHF and post-CHF heat transfer in
  horizontal and inclined evaporator tubes* — no data table, figures are
  wall-temperature profiles, not CHF-vs-quality curves.

## What to use this folder for
Pipeline/architecture testing, and as a genuine out-of-distribution
evaluation set once you've settled on a train/fine-tune split — never
train or fine-tune the reported model on anything here.

# paper_extracted_test_only/

**Do not use these for training the main model, and do not use them for
transfer learning either. Testing/pipeline-validation only.**

Every file in this folder was pulled out of a research paper (PDF) by me —
either by parsing a table embedded in the PDF text, or (for the Pioro 2002
R-134a file) by visually digitizing points off a printed scatter-plot
figure — or is mentor-provided source material kept separate for the same
reason. None of it was downloaded as a ready-made, author-published dataset
the way the files one level up (`kaeri_tr1665_*`, `pinfin_chf_water_fc72`,
`zhao2020_chf_flowboiling_tubes`, `nrc_groeneveld_24579pt_chf_database`)
were.

## Genuine point-level data (real tabulated numbers, PDF-extracted)
- `helical_coil_r123_appendixCD.csv` (+ source PDF) — 257 rows, R123 helical
  coil, Appendices C/D of a Mendeley-hosted "Research Data" supplement.
- `hardik2016_helical_coils_r123_lowpressure_chf.csv` — 156 rows, R123
  helical coils, 6 coils, from Hardik & Prabhu, *Critical heat flux in
  helical coils at low pressure*, Appl. Thermal Eng. 112 (2017) 1223-1239
  (PII S1359431116325650). Parsed from the paper's own Appendix A; row
  count verified exactly against the paper's per-coil totals in Table 4
  (47+29+18+22+31+9 = 156).
- `hardik2017_straight_tubes_r123_chf.csv` — 55 rows, R123 straight
  horizontal tubes, 7 tube geometries, from Hardik, Kumar & Prabhu,
  *Boiling pressure drop... in horizontal straight tubes*, Int. J. Heat
  Mass Transfer 113 (2017) 466-481 (PII S0017931016340443). Parsed from
  the paper's own Appendix A; row count verified against Table 2's stated
  total (8+7+10+11+8+5+6 = 55) — note the last 8 rows were on a page whose
  text layer the PDF parser initially dropped, so they were re-entered by
  reading the rendered page image directly.

## Manually digitized (not measured/tabulated — real ~5-10% error)
- `pioro2002_r134a_horizontal_vertical_chf_DIGITIZED.csv` — 268 points read
  by eye off scatter-plot figures (Pioro et al. 2002, R-134a, horizontal vs.
  vertical tubes). See `pioro2002_r134a_DIGITIZED_README.md` for full
  methodology and caveats.

## Small samples / metadata / range tables (no usable point-level rows)
`nureg_km0011_table4-1_SAMPLE_ONLY.csv` (21 illustrative rows only),
`nureg_km0011_table4-2_*`, `tanase2009_diameter_correction_exponent_grid.csv`,
`supercritical_water_source_summary_luo2020.csv`,
`supercritical_co2_source_summary_luo2020.csv`,
`narrowchannel_chf_ml2025_source_summary.csv`, and
`furlong2025_nrc_vs_debortoli_range_comparison.csv` all give per-source
point counts and min/max condition ranges, not individual
(features -> CHF) rows.

## Mentor-provided source material (kept here, not in the main pipeline)
- `mentor_master_experiments.xlsx`, `external_coil_tube_chf_appendix.pdf` —
  moved from `data/raw/`. Still referenced by `scripts/plan2_pipeline.py`
  (paths updated), which is a separate exploratory pipeline, not the main
  training pipeline.

## Checked, nothing extractable
- Kefer, Kohler & Kastner, *CHF and post-CHF heat transfer in horizontal
  and inclined evaporator tubes*, Int. J. Multiphase Flow 15 (1989) 385-392
  (PII 0301932289900086, `docs/references/CRITICAL_HEAT_FLUX_CHF_AND_POST-
  CHF_HEAT.pdf`). No data table, no appendix, and its figures are wall-
  temperature-distribution plots, not CHF-vs-quality curves — there was
  nothing to digitize without fabricating numbers.

## What to use them for
Pipeline/architecture testing: does your training code run against a
different schema, a smaller sample, a different fluid/orientation, without
crashing or silently misbehaving? Sanity-checking model behavior outside
your main training distribution. Not for training the model you intend to
report results from, and not for transfer learning either.

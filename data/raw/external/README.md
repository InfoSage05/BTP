# External data index

Everything in this folder is candidate data for the engineered-surface / multi-source phase described in `docs/manuscript/paper_outline.docx`. Two kinds of file live here and it matters which is which:

- **Point-level data** — real rows you can train/test/validate on.
- **Source-list summaries** — a list of *where* more data could come from (paper name, sample size, parameter ranges), with no actual data points. These are menus, not meals; do not mistake a large row count in one of these for usable data.

| File | Kind | Rows | What it actually is | Status |
|---|---|---|---|---|
| `helical_coil_r123_appendixCD.csv` | Point-level | 257 | Full R123 helical-coil dataset (140 low-pressure + 117 high-pressure) | **Duplicate of existing extraction.** Verified row-for-row identical to `results/plan2/external_pdf_data.csv` tables `D.3`/`D.4` (same source: `Hardik_Prabhu_2018_CHF_helical_coils_R123_IJTS.pdf`, appendix C/D). Kept here as an independent manual cross-check of the pipeline's PDF-parsing output — **not** consumed by `scripts/plan2_pipeline.py`, which parses the source PDF directly. If you need this data, use `results/plan2/external_pdf_data.csv` (the reproducible, pipeline-generated version); treat this CSV as a verification artifact only. |
| `pinfin_chf_water_fc72.csv` | Point-level | 190 | Pin-fin pool-boiling surfaces: fin shape/width/height/spacing, coverage, porosity, roughness factor, surface material, fluid (water or FC-72) → CHF | **The only genuine engineered-surface dataset in this repository.** Real per-surface features (roughness factor, porosity, material) matching what the advisor's outline asks for — but a different physical configuration (pool boiling on lab-fabricated micro-surfaces) than the flow-boiling tube LUT, so it can't be merged into the same model without separate justification. Candidate seed data for a surface-characteristic sub-study. |
| `nureg_km0011_table4-1_SAMPLE_ONLY.csv` | Point-level | 22 | One source (Lowdermilk 1958) manually transcribed from the NUREG/KM-0011 report (`docs/references/ML19029B306.pdf`) | Real but tiny — filename says `SAMPLE_ONLY` deliberately. This is *not* the ~25,000-point NRC/OECD-NEA database; see the note below on why that full database isn't a simple download. |
| `nureg_km0011_table4-2_source_dataset_ranges.csv` | Source-list summary | 73 | Every historical experiment (Lowdermilk, Bergles, Celata, Tong, Mudawar, etc.) that makes up the full NUREG/KM-0011 compilation, with size and parameter range only | **No point data.** Use this as a target list: the largest named sources (Zenkevich et al. 1974 = 823 pts, Kureta 1997 = 913 pts, Hewitt et al. 1965 = 442 pts, Alessandrini 1963 = 753 pts) are the highest-value candidates for manual transcription from `ML19029B306.pdf` if more real data is needed. |
| `narrowchannel_chf_ml2025_source_summary.csv` | Source-list summary | 5 | Sources behind `Critical heat flux prediction through machine learning model for narrow.pdf` | No point data — same caveat as above. |
| `supercritical_co2_source_summary_luo2020.csv` | Source-list summary | 14 | Sources behind the supercritical-CO2 CHF-analogue reference | No point data. Also note: supercritical heat-transfer deterioration is a related but distinct phenomenon from subcritical CHF/dryout — don't merge without checking whether the underlying physics is comparable. |
| `supercritical_water_source_summary_luo2020.csv` | Source-list summary | 18 | Sources behind the supercritical-water CHF-analogue reference | Same caveats as above. |
| `tanase2009_diameter_correction_exponent_grid.csv` | Reference table, not CHF data | 24 | The (8/D)ⁿ exponent grid from `Diameter effect on CHF.pdf`, used to correct the 8mm-normalized LUT to other tube diameters | Not a dataset to train on — a correction methodology. Relevant if the project ever needs to de-normalize LUT predictions to real tube diameters. |

## Why the full ~25,000-point NRC/OECD-NEA CHF database isn't just sitting here

It's a compilation of ~70 separate historical experiments (see `table4-2` above), several from the 1950s–1990s, not a single publicly downloadable file. The clean, ready-to-use version used by papers like Wang et al. (2026) is typically distributed only to teams registered for the formal OECD/NEA AI/ML CHF Benchmark (spec document: `docs/references/Benchmark on Artificial Intelligence and Machine Learning...Phase 1...pdf`). Getting more of it requires either (a) registering for that benchmark, or (b) manually transcribing more named sources from `ML19029B306.pdf` using the target list above.

## 2026-08-31 cleanup log

No files were deleted from this folder — every CSV here is either genuinely unique data or a documented, deliberately-kept cross-check. See the table above for what to actually use vs. what to treat as reference-only.

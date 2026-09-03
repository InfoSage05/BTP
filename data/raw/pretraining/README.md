# data/raw/pretraining/

**Stage 1: the single-lineage core dataset — pretrain here, and only here.**

Kept deliberately narrow: both files trace back to the same experimental
lineage (the Groeneveld tube CHF database), same fluid (water), same
geometry (round tubes). This is the large, homogeneous "core" a model
should learn its baseline physics from before seeing anything else.

| File | Rows | Notes |
|---|---|---|
| `groeneveld_2006_chf_lookup_table.xlsx` | - | source spreadsheet for `data/chf_long_clean.csv` (top-level, 11,592 rows) |
| `nrc_groeneveld_24579pt_chf_database.csv` | 24,579 | raw (non-gridded) NRC tube database behind the 2006 LUT |

Everything else that's real experimental data but a *different* fluid,
geometry, or heating regime lives in `data/raw/fine_tuning/` instead —
even if it's a few hundred or a couple thousand rows. Size isn't the
criterion here; single-lineage homogeneity is.

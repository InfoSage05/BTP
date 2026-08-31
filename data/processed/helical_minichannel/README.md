# Helical Minichannel Evaporator — Processed Data

This is the cleaned, trainable version of the raw logs in
`data/raw/helical_minichannel/`. Built by `scripts/preprocess_minichannel_data.py`.

## >>> The one file to use for training <<<

**`minichannel_chf_summary.csv`** — 32 rows, one per test run.

| Column | Meaning |
|---|---|
| `material`, `channel_pitch_mm`, `insert_type` | geometry input features |
| `mass_flow_avg_kg_s_at_chf`, `suction_pressure_bar_at_chf`, `high_pressure_bar_at_chf`, `subcooling_T_C_at_chf` | operating-condition input features, sampled at the detected burnout point |
| `power_actual_W_at_chf` | **output/target** — heater power (W) at burnout. This is a CHF *proxy*, not true heat flux in W/m², because no heated-area value exists anywhere in the source data. If you can find/measure the heated area for each geometry, multiply it out to get real W/m². |
| `chf_detection_confidence` | `high` / `low` / `none` — see below, **read this before trusting a row** |

### Confidence — don't skip this
The burnout point in each run was found automatically (a heuristic: sustained
abnormal temperature rise in a wall thermocouple while power isn't
decreasing). It is **not** a validated ground-truth label.

- **20 rows, `high`** — all aluminium runs. Clean stepped power ramps, the
  detector's confident and a spot-check looked right. Safe to start with.
- **6 rows, `low`** + **6 rows, `none`** — mostly copper runs. These show
  noisy multi-cycle power search behaviour (the operator ramped up, backed
  off, and re-ramped), so there's no single clean burnout spike to find
  automatically. `none` rows have blank CHF-point columns entirely.

**Recommended first pass**: filter to `chf_detection_confidence == "high"`
(20 rows) for initial model training. For the other 12, open the matching
plot in `figures/` and pick the CHF point by eye if you want to recover
them — the automated guess isn't reliable there.

## The other two folders (you normally won't need to touch these)

- **`timeseries/`** — the full cleaned time series for all 32 runs (English
  column names, proper numeric types, `,`/`;` mess fixed). Only useful if
  you want to re-derive the CHF point yourself, or train a time-series model
  instead of a point-CHF model.
- **`figures/`** — one diagnostic plot per run (heater power + 5 wall
  thermocouples vs. time, detected CHF point marked with a dashed line).
  Use these to visually check/correct the `low`/`none` rows above.

## Honest limitation

32 rows is small, and the "output" is heater power, not heat flux — this
dataset is best used as one more *geometry/condition-diversity* source
alongside the tube/coil/pin-fin datasets in `data/raw/external/`, not as a
standalone training set on its own.

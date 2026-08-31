# Helical Minichannel Evaporator — Raw Test-Rig Logs

This folder contains the **original, untouched** sensor logs from 32 CHF
burnout experiments on a helical minichannel evaporator test rig. Nothing in
here has been cleaned or modified — treat it as read-only source material.

## What's here

- `CHF_Experiments_Raw_Data/Test Run 1/` ... `Test Run 32/` — one folder per
  experiment, each holding a single raw CSV log.
- `INDEX.csv` — **start here.** One row per test run, decoded from the
  (fairly cryptic) folder/file names: material, channel pitch (aluminium
  series) or tube-insert type (copper series), run version, test date, and
  row count. This is what makes the raw folder navigable without opening 32
  individual files.

## Why the raw files themselves are confusing

- German-locale sensor export: `;`-separated, `,` as the decimal separator,
  `cp1252` encoding (so `ü`, `ö`, `°` show up as `�` in a plain text editor
  or UTF-8 tool).
- ~30 columns per file, mostly raw sensor channels (mass flow, several
  pressures, ~10 temperature sensors, applied heater power) logged at roughly
  2 Hz — **not** one-row-per-experiment. A single burnout test produces
  thousands of time-series rows.
- No single "this is the CHF value" column. Burnout (CHF) has to be
  identified as an event *within* each run's temperature time series.
- No heated-area value anywhere in the data or filenames, so the sensor logs
  can only give heater power (W), not true heat flux (W/m²).

Two experiment series are mixed together here:
- **Aluminium series** (Nov 2021): varies helical channel pitch — 2, 3, 5, 6,
  7 mm — 20 runs total.
- **Copper series** (Mar 2021): fixed tube, varies the internal insert —
  swirl insert, plastic insert (normal/optimized), no insert (baseline),
  long/short tube — 12 runs total.

## Where the usable, trainable data actually is

This raw folder is **not** meant to be fed to a model directly. It has been
processed into a trainable form at:

    data/processed/helical_minichannel/

See the README there — in short, `minichannel_chf_summary.csv` in that
folder is the one file you actually train on.

"""
prepare_nrc_data.py
--------------------
Cleans data/raw/external/nrc_groeneveld_24579pt_chf_database.csv into a
ready-to-use CSV, the same way prepare_data.py turns the raw Groeneveld LUT
Excel into chf_long_clean.csv.

Source: NRC ADAMS ML22264A009 (public document, no paywall/membership),
docs/references/ML22264A009.pdf -- a 457-page point-level CHF table compiled
from 60 historical experimental campaigns (uniformly heated vertical water
tubes). This is the ~24,579-point database referenced throughout the recent
CHF-ML literature (e.g. Wang et al. 2026, Yang et al. 2026) as the largest
public compilation, distinct from and much larger than the 2006 Groeneveld
look-up table already used as this project's primary dataset.

Bug found and fixed here: the raw CSV, as extracted, has a *units* row
("-, -, m, m, kPa, kg/m^2/s, -, kJ/kg, C, kW/m^2, kW/m^2") immediately after
the header row. pandas reads this as data, which silently corrupts every
column to dtype=object (so e.g. Pressure.max() returns the string "kPa"
instead of a number). This script drops that row explicitly and verifies the
resulting cast to numeric introduces zero NaNs (i.e. every remaining row was
genuinely numeric) before writing the cleaned file.
"""
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = REPO_ROOT / "data" / "raw" / "external" / "nrc_groeneveld_24579pt_chf_database.csv"
OUT_PATH = REPO_ROOT / "data" / "nrc_chf_clean.csv"

NUMERIC_COLS = [
    "Tube Diameter", "Heated Length", "Pressure", "Mass Flux",
    "Outlet Quality", "Inlet Subcooling", "Inlet Temperature", "CHF",
]

RENAME = {
    "Reference ID": "ref_id",
    "Tube Diameter": "D_m",
    "Heated Length": "L_m",
    "Pressure": "P_kPa",
    "Mass Flux": "G_kg_m2s",
    "Outlet Quality": "X",
    "Inlet Subcooling": "dHin_sub_kJkg",
    "Inlet Temperature": "Tin_C",
    "CHF": "CHF_kW_m2",
}


def _has_units_row(path):
    """True if the second line of the file is the units row rather than data.

    The raw export originally shipped with a units row ("-,-,m,m,kPa,...")
    directly under the header. That row was later stripped at source on the
    `main` branch. Hard-coding skiprows=[1] is therefore unsafe in both
    directions: on the old file it is required, on the new file it would
    silently delete the first genuine measurement. Detect instead of assume.
    """
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        fh.readline()                       # header
        second = fh.readline().split(",")
    # a data row starts with an integer index; the units row does not
    return not second[0].strip().lstrip("-").isdigit()


def main():
    skip = [1] if _has_units_row(RAW_PATH) else None
    print(f"units row detected: {skip is not None}")
    df = pd.read_csv(RAW_PATH, skiprows=skip)

    # 'CHF Result' is an unused placeholder column present only in the older
    # export; it was removed at source on main. Drop it if present, after
    # confirming it really is empty.
    if "CHF Result" in df.columns:
        assert df["CHF Result"].notna().sum() == 0, (
            "CHF Result column expected to be entirely empty (unused placeholder "
            "column in the source table) -- found non-null values, investigate "
            "before proceeding."
        )
        df = df.drop(columns=["CHF Result"])

    for col in NUMERIC_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    n_bad = df[NUMERIC_COLS].isnull().sum().sum()
    assert n_bad == 0, f"{n_bad} values failed numeric coercion -- inspect before proceeding"

    df = df.rename(columns=RENAME).drop(columns=["Number"])

    n_before = len(df)
    dup_mask = df.drop(columns=["ref_id"]).duplicated(keep="first")
    n_dupes = int(dup_mask.sum())
    df_deduped = df[~dup_mask].reset_index(drop=True)

    df.to_csv(OUT_PATH.with_name("nrc_chf_clean_with_duplicates.csv"), index=False)
    df_deduped.to_csv(OUT_PATH, index=False)

    print(f"Read {n_before} rows from source (after dropping the units row).")
    print(f"Found {n_dupes} exact duplicate rows (same D/L/P/G/X/subcooling/Tin/CHF, "
          f"different Reference ID) -- {n_dupes/n_before:.2%} of the dataset.")
    print(f"Wrote {len(df_deduped)}-row deduplicated file: {OUT_PATH}")
    print(f"Wrote {len(df)}-row file with duplicates retained (for anyone who wants "
          f"to inspect them): {OUT_PATH.with_name('nrc_chf_clean_with_duplicates.csv')}")
    print(f"Unique source studies (ref_id): {df['ref_id'].nunique()}")
    print()
    print("Ranges (deduplicated):")
    print(df_deduped[["D_m", "L_m", "P_kPa", "G_kg_m2s", "X", "CHF_kW_m2"]].describe())


if __name__ == "__main__":
    main()

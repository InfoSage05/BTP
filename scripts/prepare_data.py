"""
prepare_data.py
----------------
Parses the raw '2006_CHF_Lookup_Table.xlsx' (as digitized from Groeneveld et al. 2007,
Appendix B) into a clean long-format CSV: columns [P, G, X, CHF].

Handles known quirks of this specific file:
  - Row 3 (index 2) holds the X (quality) column headers.
  - Data starts at row 4 (index 3).
  - Pressure (P) is only written on the first row of each pressure block
    (merged-cell style) -> must be forward-filled.
  - Each pressure block is followed by a blank spacer row (G is NaN) -> must be dropped.
  - The sheet repeats the header row ("Pressure [kPa]", "Mass Flux...") partway down
    (approx every 22 rows) -> must be filtered out before numeric conversion.

Verified output (as of the source file used to write this script):
  - 24 unique pressures, 21 unique mass fluxes, 23 unique qualities
  - 24 * 21 * 23 = 11,592 total (P,G,X) grid cells, zero duplicates, zero nulls
  - 504 cells have CHF == 0, all at X == 1.0 (physically a placeholder/boundary,
    NOT a real trainable target -> excluded downstream, not here, so the raw
    grid is preserved for anyone who wants it)
  - CHF range (excluding the X==1.0 placeholders): ~15 to ~44,338 kW/m^2

Usage:
    python scripts/prepare_data.py --input data/raw/groeneveld_2006_chf_lookup_table.xlsx --output data/chf_long_clean.csv
"""
import argparse
import pandas as pd


def parse_lut(xlsx_path: str) -> pd.DataFrame:
    raw = pd.read_excel(xlsx_path, sheet_name="CHF Lookup Table", header=None)

    # Row index 2 holds the X (quality) values, columns 3..25 (23 columns)
    x_vals = raw.iloc[2, 3:26].astype(float).tolist()

    data = raw.iloc[3:].reset_index(drop=True)
    data.columns = ["P", "G", "blank"] + [f"X_{x}" for x in x_vals]
    data = data.drop(columns="blank")

    # Drop repeated header rows that appear mid-sheet
    data = data[data["P"] != "Pressure\\n[kPa]"].reset_index(drop=True)

    data["P"] = pd.to_numeric(data["P"], errors="coerce")
    data["G"] = pd.to_numeric(data["G"], errors="coerce")

    # Pressure is merged-cell style -> forward fill
    data["P"] = data["P"].ffill()

    # Drop spacer rows between pressure blocks (G is NaN there)
    data = data.dropna(subset=["G"]).reset_index(drop=True)

    # Sanity check: expect exactly 24 * 21 = 504 unique (P, G) rows
    n_p, n_g = data["P"].nunique(), data["G"].nunique()
    assert len(data) == n_p * n_g, (
        f"Unexpected row count after cleaning: {len(data)} rows but "
        f"{n_p} unique P x {n_g} unique G = {n_p * n_g} expected. "
        f"Re-check the raw sheet structure before trusting this output."
    )

    long_df = data.melt(id_vars=["P", "G"], var_name="X", value_name="CHF")
    long_df["X"] = long_df["X"].str.replace("X_", "").astype(float)

    # Final sanity checks
    assert long_df.duplicated(subset=["P", "G", "X"]).sum() == 0, "duplicate grid cells found"
    assert long_df["CHF"].isna().sum() == 0, "unexpected nulls in CHF"

    return long_df.sort_values(["P", "G", "X"]).reset_index(drop=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="data/raw/groeneveld_2006_chf_lookup_table.xlsx")
    ap.add_argument("--output", default="data/chf_long_clean.csv")
    args = ap.parse_args()

    df = parse_lut(args.input)
    print(f"Parsed grid: {df.shape[0]} rows "
          f"({df['P'].nunique()} P x {df['G'].nunique()} G x {df['X'].nunique()} X)")
    print(f"CHF==0 count (expected 504, all at X=1.0): {(df['CHF'] == 0).sum()}")
    print(f"Non-zero CHF range: {df.loc[df.CHF > 0, 'CHF'].min()} "
          f"to {df.loc[df.CHF > 0, 'CHF'].max()} kW/m^2")

    df.to_csv(args.output, index=False)
    print(f"Saved -> {args.output}")

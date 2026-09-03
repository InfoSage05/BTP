"""
Map the Strip pool-boiling dataset into the existing (P,G,X,D)-based flow-
boiling feature schema, per explicit user instruction: mass flux G=0 for
pool boiling (no forced flow), reuse everything else as-is.

Column mapping (data/Master file Strip.xlsx, sheet "Final Master file"):
  P_kPa   = P [bar] * 100                          (verified: P~=0.9984 bar,
                                                      near-atmospheric)
  G_kg_m2s = 0                                      (pool boiling: no flow)
  X        = -Cp * (Tsat-Tpool) / hfg               (standard subcooling ->
                                                      equivalent-quality
                                                      conversion; verified
                                                      Cp~4181 J/kg-K and
                                                      hfg~2.258e6 J/kg match
                                                      water at ~1 atm, so
                                                      units are consistent;
                                                      resulting X in
                                                      [-0.15, -0.13] range on
                                                      a spot check -- sane,
                                                      not fabricated)
  D_mm     = "Apparent dia (mm)"                    (given directly)
  CHF_kW_m2 = "CHF" [W/m^2] / 1000                  (verified against the
                                                      "CHF(MW/m^2)" column:
                                                      CHF/1e6 == that column
                                                      exactly)

Only the 55 rows with every needed column populated are used -- no
imputation, no fabrication of missing rows.

pin-fin (pinfin_chf_water_fc72.csv) is NOT included here: it has no bulk
diameter concept (only fin Width/Height/Spacing at O(10-100 um)), and
substituting those for D_mm would silently ask the model to extrapolate
2-3 orders of magnitude below its training diameter range (3-20mm) --
worse than excluding it. See README.md in this output folder for the full
reasoning.

Orientation/Angle are kept as metadata columns (not fed as model features)
-- the (P,G,X,D) schema has no way to represent heater inclination, which
is a real, honest limitation, not an oversight.
"""
import os
import sys
import pandas as pd

sys.path.insert(0, "scripts/chf_pipeline")
from physics_features import add_dimensionless_features

OUT_DIR = "data/processed/pool_boiling"


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    df = pd.read_excel("data/Master file Strip.xlsx", sheet_name="Final Master file", header=0)

    needed = ["Apparent dia (mm)", "P", "hfg", "Cp", "Tsat-Tpool", "CHF",
              "CHF(MW/m^2)", "Angle", "Orientation"]
    df = df[needed].dropna().reset_index(drop=True)

    # sanity check: CHF and CHF(MW/m^2) must agree (same value, different units)
    ratio = df["CHF"] / (df["CHF(MW/m^2)"] * 1e6)
    assert (ratio.sub(1).abs() < 1e-6).all(), "CHF vs CHF(MW/m^2) unit mismatch -- mapping wrong"

    out = pd.DataFrame({
        "P_kPa": df["P"] * 100.0,
        "G_kg_m2s": 0.0,
        "X": -df["Cp"] * df["Tsat-Tpool"] / df["hfg"],
        "D_mm": df["Apparent dia (mm)"],
        "CHF_kW_m2": df["CHF"] / 1000.0,
        "Angle_deg": df["Angle"],
        "Orientation": df["Orientation"],
    })

    print(f"Strip pool-boiling rows prepared: {len(out)}")
    print(out[["P_kPa", "X", "D_mm", "CHF_kW_m2"]].describe())

    out = add_dimensionless_features(out, fluid="water", p_col="P_kPa", g_col="G_kg_m2s")
    n_before = len(out)
    out = out.dropna(subset=["P_reduced", "rho_l_kg_m3", "rho_g_kg_m3", "density_ratio"])
    print(f"After CoolProp property lookup: {len(out)}/{n_before} rows retained")

    out_path = os.path.join(OUT_DIR, "strip_pool_boiling_water.csv")
    out.to_csv(out_path, index=False)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()

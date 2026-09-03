"""
build_surface_master.py
-----------------------
Phase 0 of the confounding study: assemble every engineered-surface CHF
extraction in data/raw/external/ into ONE analysis table with an explicit
`study` label per row, plus a pin-fin table with the same schema contract.

Why a script and not a notebook: every downstream result in the paper depends
on exactly which rows enter the pooled analysis. That decision has to be
reproducible and reviewable, not buried in notebook state.

Outputs
    data/processed/surface_master.csv        ATF/coating family, one row per tested surface
    data/processed/pinfin_master.csv         pin-fin family, same contract
    results/confounding/dataset_inventory.csv

Schema contract (both tables):
    study        str   source publication -- the grouping variable for LOSO
    CHF_kW_m2    float target
    Ra_um        float arithmetic mean roughness (or roughness factor for pin-fin)
    CA_deg       float static/apparent contact angle (NaN where not published)
    material     str
    geometry     str   flat_plate | horizontal_tube | vertical_tube | rodlet | pin_fin
    fluid        str
    P_kPa        float system pressure where reported
    orient_deg   float surface orientation (0 = upward facing)

Rows are kept even when a feature is missing; filtering to the modellable
subset is done explicitly downstream so the drop is always visible and counted.
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
EXT = ROOT / "data" / "raw" / "external"
OUT = ROOT / "data" / "processed"
RES = ROOT / "results" / "confounding"

COLS = ["study", "CHF_kW_m2", "Ra_um", "CA_deg", "material",
        "geometry", "fluid", "P_kPa", "orient_deg"]

# (file, study label, geometry, column mapping)
ATF_SOURCES = [
    ("he2022_horizontal_tube_pool_boiling_chf.csv", "He2022", "horizontal_tube",
     dict(CHF_kW_m2="CHF_kW_m2", Ra_um="roughness_Ra_um", CA_deg="static_contact_angle_deg",
          material="material", P_kPa="pressure_kPa", orient_deg="surface_orientation_deg")),
    ("ahn2010_zircaloy4_anodized_pool_boiling_chf.csv", "Ahn2010", "flat_plate",
     dict(CHF_kW_m2="CHF_kW_m2", Ra_um="roughness_Ra_um", CA_deg="static_contact_angle_deg",
          material="material", orient_deg="surface_orientation_deg")),
    ("yeom2020_atf_cladding_pool_boiling_chf.csv", "Yeom2020", "rodlet",
     dict(CHF_kW_m2="CHF_kW_m2", Ra_um="roughness_Ra_um", CA_deg="static_contact_angle_deg",
          material="heater_material", orient_deg="surface_orientation_deg")),
    ("ali2020_cr_coated_zirc4_ion_irradiation_chf.csv", "Ali2020", "flat_plate",
     dict(CHF_kW_m2="CHF_kW_m2", Ra_um="roughness_Ra_um", CA_deg="static_contact_angle_deg",
          material="surface", orient_deg="surface_orientation_deg")),
    ("ali2018_fecral_atf_pool_boiling_chf.csv", "Ali2018", "flat_plate",
     dict(CHF_kW_m2="CHF_kW_m2", Ra_um="roughness_Ra_um", CA_deg="static_contact_angle_deg",
          material="material")),
    ("kam2015_sic_cr_coated_plates_chf.csv", "Kam2015", "flat_plate",
     dict(CHF_kW_m2="CHF_kW_m2", Ra_um="roughness_Ra_um", CA_deg="static_contact_angle_deg",
          material="surface")),
    ("zhang2023_boiling_crisis_sapphire_surfaces_chf.csv", "Zhang2023", "flat_plate",
     dict(CHF_kW_m2="CHF_kW_m2", Ra_um="roughness_Ra_um", CA_deg="static_contact_angle_deg",
          material="surface_coating", P_kPa=None, orient_deg="surface_orientation_deg")),
    ("kim2019_cr_coated_oxidized_roughness.csv", "Kim2019", "vertical_tube",
     dict(CHF_kW_m2="CHF_kW_m2", Ra_um="roughness_Ra_um", CA_deg="static_contact_angle_deg",
          material="surface")),
    # --- added 2026-09-02, second extraction round ---
    ("jo2019_atf_coating_pool_boiling_chf.csv", "Jo2019", "flat_plate",
     dict(CHF_kW_m2="CHF_kW_m2", Ra_um="roughness_Ra_um", CA_deg="static_contact_angle_deg",
          material="material", orient_deg="surface_orientation_deg")),
    ("cheollee2019_zrsi2_coated_zircaloy_chf.csv", "CheolLee2019", "flat_plate",
     dict(CHF_kW_m2="CHF_kW_m2", Ra_um="roughness_Ra_um", CA_deg="static_contact_angle_deg",
          material="material", orient_deg="surface_orientation_deg")),
    ("seo2015_zircaloy_sic_cladding_chf.csv", "Seo2015", "vertical_tube",
     dict(CHF_kW_m2="CHF_kW_m2", Ra_um="roughness_Ra_um", CA_deg="static_contact_angle_deg",
          material="material", orient_deg="surface_orientation_deg")),
]


def _take(df, mapping, study, geometry, fluid="water"):
    out = pd.DataFrame(index=df.index)
    for target, src in mapping.items():
        out[target] = df[src] if (src and src in df.columns) else np.nan
    out["study"] = study
    out["geometry"] = geometry
    out["fluid"] = fluid
    for c in COLS:
        if c not in out.columns:
            out[c] = np.nan
    return out[COLS]


def build_atf():
    frames = []
    for fname, study, geom, mapping in ATF_SOURCES:
        df = pd.read_csv(EXT / fname)
        frames.append(_take(df, mapping, study, geom))
    m = pd.concat(frames, ignore_index=True)
    # Zhang2023 reports pressure in MPa in its own file; normalise to kPa.
    z = m.study == "Zhang2023"
    m.loc[z, "P_kPa"] = 101.2
    m["P_kPa"] = m["P_kPa"].fillna(101.325)  # all others: atmospheric, stated in source
    m["orient_deg"] = m["orient_deg"].fillna(0.0)
    return m


def build_pinfin():
    d = pd.read_csv(EXT / "pinfin_chf_water_fc72.csv")
    # 'Source' is only populated on the first row of each study block.
    d["Source"] = d["Source"].ffill()
    study = (d["Source"].astype(str).str.split("\n").str[0]
             .str.split("http").str[0].str.strip())
    out = pd.DataFrame({
        "study": study,
        "CHF_kW_m2": d["CHF(kW/m2)"],
        "Ra_um": d["Roughness Factor"],   # dimensionless area ratio, NOT microns -- see note below
        "CA_deg": np.nan,                 # pin-fin compilation reports no contact angle
        "material": d["Surface Material"],
        "geometry": "pin_fin",
        "fluid": d["Fluid"],
        "P_kPa": 101.325,
        "orient_deg": 0.0,
    })
    # carry the extra pin-fin geometry features; they are real predictors here
    for c in ["Subcooling(K)", "Width(um)", "Height(um)", "Spacing(um)",
              "Coverage", "Porosity", "MBL, Lateral (um)", "MBL, Total (um)"]:
        out[c] = d[c]
    return out


def inventory(name, df, feat):
    n_complete = df[feat + ["CHF_kW_m2"]].notna().all(axis=1).sum()
    return dict(dataset=name, rows=len(df), studies=df.study.nunique(),
                rows_with_CHF=int(df.CHF_kW_m2.notna().sum()),
                modellable_rows=int(n_complete),
                median_rows_per_study=float(df.study.value_counts().median()),
                max_study_share=float(df.study.value_counts().max() / len(df)))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    RES.mkdir(parents=True, exist_ok=True)

    atf = build_atf()
    pin = build_pinfin()
    atf.to_csv(OUT / "surface_master.csv", index=False)
    pin.to_csv(OUT / "pinfin_master.csv", index=False)

    inv = pd.DataFrame([
        inventory("ATF_surface", atf, ["Ra_um", "CA_deg"]),
        inventory("pin_fin", pin, ["Ra_um"]),
    ])
    inv.to_csv(RES / "dataset_inventory.csv", index=False)

    print(f"wrote {OUT/'surface_master.csv'}  ({len(atf)} rows, {atf.study.nunique()} studies)")
    print(f"wrote {OUT/'pinfin_master.csv'}   ({len(pin)} rows, {pin.study.nunique()} studies)")
    print()
    print(inv.to_string(index=False))
    print()
    print("ATF modellable (Ra+CA+CHF) per study:")
    c = atf.dropna(subset=["Ra_um", "CA_deg", "CHF_kW_m2"])
    print(c.study.value_counts().to_string())
    print()
    print("NOTE: pin-fin 'Ra_um' column holds a dimensionless roughness FACTOR, not")
    print("microns. The two families are never pooled into one regression -- they are")
    print("analysed separately and compared only at the level of the confounding metric.")


if __name__ == "__main__":
    main()

"""
build_model_ready.py
--------------------
Collect every dataset the study actually uses into ONE place with ONE schema,
so that what is being trained on and what is held out is unambiguous.

Writes:
    data/model_ready/train/*.csv      datasets the model may learn from
    data/model_ready/test/*.csv       held out, touched once, at the end
    data/model_ready/README.md        what each file is and where it came from
    data/model_ready/feature_matrix.csv   which columns exist in which dataset

THE SCHEMA PROBLEM AND ITS ANSWER
Different sources report different columns. Rather than guess or impute across
sources, every file is rewritten into one common schema and a column that a
source genuinely does not report is left as NaN -- never filled, never faked.

The intersection turns out to be exactly (P, G, X): pressure, mass flux and
thermodynamic quality are the only quantities present in every flow-boiling
source. That is not a coincidence -- it is why the 2006 Groeneveld look-up
table is built on those three. So:

    CORE feature set     P, G, X            available in 100% of flow sources
    EXTENDED feature set P, G, X, D, L      available where geometry is reported

The core model is what gets tested on every held-out set. The extended model is
tested only where D and L exist. Both are reported.

Units are harmonised to: m, kPa, kg/m2s, kJ/kg, kW/m2. Every conversion below is
explicit -- no heuristic unit sniffing (that produced a 1000x error previously).
"""
from pathlib import Path
import shutil

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "model_ready"
SCHEMA = ["dataset", "study", "boiling_mode", "fluid",
          "D_m", "L_m", "P_kPa", "G_kg_m2s", "X", "dHin_kJkg",
          "Ra_um", "CA_deg", "orient_deg", "CHF_kW_m2"]


def blank(n):
    return pd.DataFrame({c: [np.nan] * n for c in SCHEMA})


def finish(df, name, mode, fluid, study=None):
    df["dataset"] = name
    df["boiling_mode"] = mode
    df["fluid"] = fluid
    if study is not None:
        df["study"] = study
    df = df[SCHEMA]
    return df[df.CHF_kW_m2.notna() & (df.CHF_kW_m2 > 0)].reset_index(drop=True)


# ---------------------------------------------------------------- TRAIN
def nrc():
    d = pd.read_csv(ROOT / "data/nrc_chf_clean.csv")
    o = blank(len(d))
    o["D_m"], o["L_m"] = d.D_m, d.L_m
    o["P_kPa"], o["G_kg_m2s"], o["X"] = d.P_kPa, d.G_kg_m2s, d.X
    o["dHin_kJkg"], o["CHF_kW_m2"] = d.dHin_sub_kJkg, d.CHF_kW_m2
    return finish(o, "NRC", "flow", "water", d.ref_id.astype(str))


def zhao():
    d = pd.read_csv(ROOT / "data/raw/external/zhao2020_chf_flowboiling_tubes.csv")
    o = blank(len(d))
    o["D_m"] = pd.to_numeric(d["D_e [mm]"], errors="coerce") / 1000.0
    o["L_m"] = pd.to_numeric(d["length [mm]"], errors="coerce") / 1000.0
    o["P_kPa"] = pd.to_numeric(d["pressure [MPa]"], errors="coerce") * 1000.0
    o["G_kg_m2s"] = pd.to_numeric(d["mass_flux [kg/m2-s]"], errors="coerce")
    o["X"] = pd.to_numeric(d["x_e_out [-]"], errors="coerce")
    o["CHF_kW_m2"] = pd.to_numeric(d["chf_exp [MW/m2]"], errors="coerce") * 1000.0
    return finish(o, "Zhao2020", "flow", "water", d.author.astype(str))


def pinfin():
    d = pd.read_csv(ROOT / "data/processed/pinfin_master.csv")
    o = blank(len(d))
    o["Ra_um"] = d.Ra_um                    # dimensionless roughness FACTOR here
    o["P_kPa"] = d.P_kPa
    o["orient_deg"] = d.orient_deg
    o["CHF_kW_m2"] = d.CHF_kW_m2
    o = pd.concat([o, d[["Subcooling(K)", "Width(um)", "Height(um)", "Spacing(um)",
                         "Coverage", "Porosity"]]], axis=1)
    o["dataset"] = "PinFin"; o["boiling_mode"] = "pool"
    o["fluid"] = d.fluid; o["study"] = d.study
    keep = SCHEMA + ["Subcooling(K)", "Width(um)", "Height(um)", "Spacing(um)",
                     "Coverage", "Porosity"]
    o = o[keep]
    return o[o.CHF_kW_m2.notna() & (o.CHF_kW_m2 > 0)].reset_index(drop=True)


# ---------------------------------------------------------------- TEST
def kaeri(fname, label):
    d = pd.read_csv(ROOT / "data/raw/external" / fname)
    q = "EquilibriumQuality" if "EquilibriumQuality" in d.columns else "Quality"
    o = blank(len(d))
    o["D_m"] = pd.to_numeric(d.Diameter, errors="coerce")
    o["L_m"] = pd.to_numeric(d.Length, errors="coerce")
    o["P_kPa"] = pd.to_numeric(d.Pressure, errors="coerce") / 1000.0     # Pa -> kPa
    o["G_kg_m2s"] = pd.to_numeric(d.MassFlux, errors="coerce")
    o["X"] = pd.to_numeric(d[q], errors="coerce")
    o["CHF_kW_m2"] = pd.to_numeric(d.HeatFlux, errors="coerce") / 1000.0  # W -> kW
    return finish(o, label, "flow", "water", d.Source.astype(str))


def pioro():
    p = "data/raw/external/paper_extracted_test_only/pioro2002_r134a_horizontal_vertical_chf_DIGITIZED.csv"
    d = pd.read_csv(ROOT / p)
    o = blank(len(d))
    o["D_m"] = pd.to_numeric(d.D_mm, errors="coerce") / 1000.0
    o["P_kPa"] = pd.to_numeric(d.P_MPa, errors="coerce") * 1000.0
    o["G_kg_m2s"] = pd.to_numeric(d.G_kg_m2s, errors="coerce")
    o["X"] = pd.to_numeric(d.x_cr, errors="coerce")
    o["orient_deg"] = np.where(d.orientation.astype(str).str.contains("horiz", case=False), 90.0, 0.0)
    o["CHF_kW_m2"] = pd.to_numeric(d.CHF_kW_m2, errors="coerce")
    return finish(o, "Pioro2002_R134a", "flow", "R-134a", "Pioro2002")   # no heated length reported


def hardik2016():
    p = "data/raw/external/paper_extracted_test_only/hardik2016_helical_coils_r123_lowpressure_chf.csv"
    d = pd.read_csv(ROOT / p)
    o = blank(len(d))
    o["L_m"] = pd.to_numeric(d.L_mm, errors="coerce") / 1000.0
    o["P_kPa"] = pd.to_numeric(d.P_bar, errors="coerce") * 100.0          # bar -> kPa
    o["G_kg_m2s"] = pd.to_numeric(d.G_kg_m2s, errors="coerce")
    o["X"] = pd.to_numeric(d.xe, errors="coerce")
    o["CHF_kW_m2"] = pd.to_numeric(d.CHF_kW_m2, errors="coerce")
    return finish(o, "Hardik2016_helical", "flow", "R-123", "Hardik2016")  # no tube diameter reported


def helical_appendix():
    p = "data/raw/external/paper_extracted_test_only/helical_coil_r123_appendixCD.csv"
    d = pd.read_csv(ROOT / p)
    o = blank(len(d))
    o["L_m"] = pd.to_numeric(d.Lh_mm, errors="coerce") / 1000.0
    o["P_kPa"] = pd.to_numeric(d.Psys_bar, errors="coerce") * 100.0
    o["G_kg_m2s"] = pd.to_numeric(d.G_kg_m2s, errors="coerce")
    o["X"] = pd.to_numeric(d.xe, errors="coerce")
    o["CHF_kW_m2"] = pd.to_numeric(d.CHF_kW_m2, errors="coerce")
    return finish(o, "Helical_R123_appendix", "flow", "R-123",
                  "Hardik_Prabhu_" + d.appendix.astype(str))


def nureg_sample():
    p = "data/raw/external/paper_extracted_test_only/nureg_km0011_table4-1_SAMPLE_ONLY.csv"
    d = pd.read_csv(ROOT / p)
    o = blank(len(d))
    o["D_m"] = pd.to_numeric(d.D_m, errors="coerce")
    o["L_m"] = pd.to_numeric(d.L_m, errors="coerce")
    o["P_kPa"] = pd.to_numeric(d.P_kPa, errors="coerce")
    o["G_kg_m2s"] = pd.to_numeric(d.G_kg_m2s, errors="coerce")
    o["X"] = pd.to_numeric(d.Xchf, errors="coerce")
    o["dHin_kJkg"] = pd.to_numeric(d.DHin_kJkg, errors="coerce")
    o["CHF_kW_m2"] = pd.to_numeric(d.CHF_kW_m2, errors="coerce")
    return finish(o, "NUREG_Lowdermilk", "flow", "water", d.Reference.astype(str))


def atf_surface():
    d = pd.read_csv(ROOT / "data/processed/surface_master.csv")
    o = blank(len(d))
    o["Ra_um"], o["CA_deg"] = d.Ra_um, d.CA_deg
    o["orient_deg"], o["P_kPa"] = d.orient_deg, d.P_kPa
    o["CHF_kW_m2"] = d.CHF_kW_m2
    o["material"] = d.material; o["geometry"] = d.geometry
    o = o[SCHEMA + ["material", "geometry"]]
    o["dataset"] = "ATF_surface"; o["boiling_mode"] = "pool"; o["fluid"] = "water"
    o["study"] = d.study
    return o[o.CHF_kW_m2.notna() & (o.CHF_kW_m2 > 0)].reset_index(drop=True)


def main():
    if OUT.exists():
        shutil.rmtree(OUT)
    (OUT / "train").mkdir(parents=True)
    (OUT / "test").mkdir(parents=True)

    train = {"01_NRC_flow.csv": nrc(), "02_Zhao2020_flow.csv": zhao(),
             "03_PinFin_pool.csv": pinfin()}
    test = {"L1_NUREG_Lowdermilk.csv": nureg_sample(),
            "L2_KAERI_uniform.csv": kaeri("kaeri_tr1665_uniform_chf.csv", "KAERI_uniform"),
            "L3_KAERI_nonuniform.csv": kaeri("kaeri_tr1665_nonuniform_chf.csv", "KAERI_nonuniform"),
            "L4_Pioro2002_R134a.csv": pioro(),
            "L5_Hardik2016_helical.csv": hardik2016(),
            "L6_Helical_R123.csv": helical_appendix(),
            "L7_ATF_surface.csv": atf_surface()}

    rows = []
    for folder, group in (("train", train), ("test", test)):
        for fname, df in group.items():
            df.to_csv(OUT / folder / fname, index=False)
            core = ["P_kPa", "G_kg_m2s", "X"]
            ext = core + ["D_m", "L_m"]
            rows.append(dict(role=folder, file=fname, dataset=df.dataset.iloc[0],
                             rows=len(df), studies=df.study.nunique(),
                             mode=df.boiling_mode.iloc[0], fluid=df.fluid.iloc[0],
                             has_CORE=bool(df[core].notna().all(axis=1).any()),
                             has_EXTENDED=bool(df[ext].notna().all(axis=1).any()),
                             missing=", ".join([c for c in ext if df[c].isna().all()]) or "-"))
    fm = pd.DataFrame(rows)
    fm.to_csv(OUT / "feature_matrix.csv", index=False)
    pd.set_option("display.width", 200)
    print(fm.to_string(index=False))
    print(f"\nTotal train rows: {sum(len(d) for d in train.values())}")
    print(f"Total test  rows: {sum(len(d) for d in test.values())}")


if __name__ == "__main__":
    main()

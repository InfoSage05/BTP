"""Evidence, not assumption: prove every banned column reconstructs the target.

Each check recomputes CHF from the columns we refuse to use. A high R2 here is
what justifies the exclusion. Also prints Exhibit A -- the energy-balance identity
that makes outlet quality circular as a predictor.
"""

from __future__ import annotations

import os
import numpy as np
import pandas as pd
from CoolProp.CoolProp import PropsSI

import datasets as D

FLOW = D.FLOW
SHARE = D.SHARE


def r2(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float)
    return 1.0 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum()


def band(y, p, tol=0.10):
    y, p = np.asarray(y, float), np.asarray(p, float)
    return float((np.abs(p - y) / np.abs(y) <= tol).mean())


def main():
    rows = []

    # -- KAERI: Power / (Perimeter * Length) IS the heat flux -----------------
    for tag, fn in (("D3_KAERI_uniform", "kaeri_tr1665_uniform_chf.csv"),
                    ("D4_KAERI_nonuniform", "kaeri_tr1665_nonuniform_chf.csv")):
        d = pd.read_csv(os.path.join(FLOW, fn), encoding="utf-8-sig")
        rec = d["Power"] / (d["Perimeter"] * d["Length"])
        rows.append(dict(dataset=tag, banned="Power,Perimeter(,Length)",
                         formula="Power/(Perimeter*Length)",
                         R2=r2(d["HeatFlux"], rec), within10=band(d["HeatFlux"], rec),
                         median_ratio=float(np.median(rec / d["HeatFlux"]))))

    # -- Helical R-123: Q_watt over heated area ------------------------------
    d = pd.read_csv(os.path.join(FLOW, "helical_coil_r123_appendixCD.csv"), encoding="utf-8-sig")
    # diameter unpublished for this paper; recover the implied constant per coil
    implied = d["Q_watt"] / (np.pi * (d["Lh_mm"] / 1000.0) * d["CHF_kW_m2"] * 1000.0)
    rows.append(dict(dataset="D5_Helical_R123", banned="Q_watt",
                     formula="Q_watt/(pi*d*Lh) with d implied per coil",
                     R2=np.nan, within10=np.nan, median_ratio=float(np.median(implied))))

    # -- Hardik helical water: Q_W over heated area, and CHF_ratio -----------
    d = pd.read_csv(os.path.join(FLOW, "hardik2017_helical_coils_water_lowpressure_chf.csv"),
                    encoding="utf-8-sig")
    dia = d["Coil_No"].map(D.COIL_D_MM) / 1000.0
    rec = d["Q_W"] / (np.pi * dia * d["L_mm"] / 1000.0) / 1000.0
    rows.append(dict(dataset="D6_Hardik_helical_water", banned="Q_W",
                     formula="Q_W/(pi*d*L)", R2=r2(d["CHF_kW_m2"], rec),
                     within10=band(d["CHF_kW_m2"], rec),
                     median_ratio=float(np.median(rec / d["CHF_kW_m2"]))))
    rows.append(dict(dataset="D6_Hardik_helical_water", banned="CHF_ratio",
                     formula="CHF_ratio = CHF / LUT(P,G,x)", R2=np.nan, within10=np.nan,
                     median_ratio=float(np.median(d["CHF_ratio"]))))

    tbl = pd.DataFrame(rows)
    print("\n=== Leakage evidence: what the banned columns reconstruct ===")
    print(tbl.to_string(index=False, float_format=lambda v: f"{v:.6f}"))

    # -- Exhibit A: the energy-balance identity on NRC -----------------------
    n = pd.read_csv(os.path.join(FLOW, "nrc_groeneveld_deduplicated_24443.csv"),
                    encoding="utf-8-sig")
    hfg = {p: (PropsSI("H", "P", p * 1e3, "Q", 1, "water")
               - PropsSI("H", "P", p * 1e3, "Q", 0, "water")) / 1e3
           for p in n["P_kPa"].unique()}
    h = n["P_kPa"].map(hfg)
    pred = n["G_kg_m2s"] * n["D_m"] * (n["X"] * h + n["dHin_sub_kJkg"]) / (4 * n["L_m"])
    rel = np.abs(pred - n["CHF_kW_m2"]) / n["CHF_kW_m2"]
    exhibit = dict(
        rows=len(n), R2=r2(n["CHF_kW_m2"], pred),
        median_abs_rel_err_pct=float(np.median(rel) * 100),
        within_10pct=band(n["CHF_kW_m2"], pred),
        within_1pct=band(n["CHF_kW_m2"], pred, 0.01),
        median_ratio=float(np.median(pred / n["CHF_kW_m2"])),
    )
    print("\n=== Exhibit A: CHF = G*D*(X*h_fg + dHin)/(4L), zero fitted parameters ===")
    for k, v in exhibit.items():
        print(f"   {k:26} {v}")

    os.makedirs(D.os.path.join(os.path.dirname(__file__), "..", "..",
                               "results", "prof_tasks"), exist_ok=True)
    out = os.path.join(os.path.dirname(__file__), "..", "..", "results", "prof_tasks")
    tbl.to_csv(os.path.join(out, "leakage_evidence.csv"), index=False)
    pd.DataFrame([exhibit]).to_csv(os.path.join(out, "exhibitA_energy_balance.csv"), index=False)

    # -- schema summary across every dataset ---------------------------------
    recs = []
    for name, ds in D.load_all().items():
        recs.append(dict(
            dataset=name, rows=ds.n, fluid=ds.fluid, geometry=ds.geometry,
            n_groups=(ds.df[ds.group].nunique() if ds.group else 0),
            local_features=len(ds.local),
            inlet_features=(len(ds.inlet) if ds.inlet else 0),
            inlet_form="yes" if ds.inlet else "NOT POSSIBLE",
            banned=";".join(ds.banned) or "-",
            CHF_min=round(float(ds.df["CHF_kW_m2"].min()), 1),
            CHF_max=round(float(ds.df["CHF_kW_m2"].max()), 1),
            note=ds.note,
        ))
    sch = pd.DataFrame(recs)
    print("\n=== Dataset schema ===")
    print(sch.drop(columns=["note"]).to_string(index=False))
    sch.to_csv(os.path.join(out, "dataset_schema.csv"), index=False)
    print("\nwrote leakage_evidence.csv, exhibitA_energy_balance.csv, dataset_schema.csv")


if __name__ == "__main__":
    main()

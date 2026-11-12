"""Sensitivity of every downstream result to the UNVERIFIED Katto constant C_Kc.

The code carries C_Kc = 0.25 (L/D < 50) rising linearly to 0.34 (L/D > 150),
tagged [S] UNVERIFIED. The NASA source the rest of the correlation was
transcribed from (NTRS 20230009827) gives Eqs. 8-18 in full but never states
C_Kc numerically -- confirmed by direct text extraction.

Rather than leave that as an open risk, measure it:
  1. How often is the branch containing C_Kc (q_co,2) actually the minimum?
     C_Kc is irrelevant on any row where another branch wins.
  2. Sweep C_Kc over a wide range and record physics-alone and pipeline LODO R2.

If results barely move, the unverified tag is immaterial and can be retired.
"""
from __future__ import annotations
import os, sys, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "..", "physics_pipeline", "scripts"))
import datasets as D, physics_arms as P, pipeline as PL
from physics import correlations as C

OUT = P.OUT
FLOW = ["D1_NRC", "D2_Zhao2020", "D3_KAERI_uniform", "D4_KAERI_nonuniform",
        "D5_Helical_R123", "D6_Hardik_helical_water", "D7_Hardik_straight_R123",
        "D8_Pioro_R134a"]

def r2(y, p):
    y = np.asarray(y, float); p = np.nan_to_num(np.asarray(p, float), nan=np.nanmedian(y))
    ss = ((y - y.mean()) ** 2).sum()
    return float(1 - ((y - p) ** 2).sum() / ss) if ss > 0 else np.nan

def main():
    raw = {n: PL.prepare(D.LOADERS[n]()) for n in FLOW}

    # ---- 1. branch activation: does C_Kc even matter on these rows?
    print("=" * 78)
    print("BRANCH ACTIVATION -- C_Kc only matters where q_co,2 is the minimum")
    print("=" * 78)
    tot_rows = tot_c = 0
    for n, df in raw.items():
        _, branch, low = C.katto_ohno_chf(
            df["G_kg_m2s"].to_numpy(), df["_D_m"].to_numpy(), df["_L_m"].to_numpy(),
            df["rho_l"].to_numpy(), df["rho_g"].to_numpy(), df["sigma"].to_numpy(),
            df["h_fg"].to_numpy(), df["dh_sub"].to_numpy(), return_branches=True)
        frac = float((branch == 0).mean())     # branch 0 == q_co,2, the C_Kc branch
        tot_rows += len(df); tot_c += int((branch == 0).sum())
        print(f"  {n:26s} q_co,2 selected on {frac*100:5.1f}% of rows")
    print(f"  {'CORPUS':26s} q_co,2 selected on {tot_c/tot_rows*100:5.1f}% of rows "
          f"({tot_c}/{tot_rows})")

    # ---- 2. sweep C_Kc
    print("\n" + "=" * 78)
    print("C_Kc SWEEP -- physics-alone R2 per dataset, and pipeline LODO median")
    print("=" * 78)
    base_small, base_large = C.KATTO_C_SMALL, C.KATTO_C_LARGE
    scales = [0.5, 0.75, 0.9, 1.0, 1.1, 1.25, 1.5, 2.0]
    rows = []
    for sc in scales:
        C.KATTO_C_SMALL, C.KATTO_C_LARGE = base_small * sc, base_large * sc
        phys_r2, corpus = {}, []
        for n, df in raw.items():
            d = df.copy()
            d["phys_katto"] = np.clip(P.phys_katto(d), 1e-3, None)
            arms = d[["phys_katto", "phys_biasi", "phys_lut"]].to_numpy(float)
            with np.errstate(all="ignore"):
                la = np.log(np.where(arms > 0, arms, np.nan))
            d["corr_spread"] = pd.Series(np.nanstd(la, axis=1)).fillna(0.0).to_numpy()
            d["dataset"] = n
            d["fluid_name"] = D.LOADERS[n]().fluid
            phys_r2[n] = r2(d["CHF_kW_m2"], d["phys_katto"])
            corpus.append(d.reindex(columns=["dataset", "fluid_name", "CHF_kW_m2",
                                             "phys_katto", "phys_biasi", "phys_lut",
                                             "corr_spread"] + P.DIMLESS))
        cp = pd.concat(corpus, ignore_index=True).dropna(
            subset=["CHF_kW_m2", "phys_katto"]).reset_index(drop=True)
        lodo = []
        for held in FLOW:
            tr, te = cp[cp.dataset != held], cp[cp.dataset == held]
            pipe = PL.CHFPipeline(seed=0).fit(tr)
            lodo.append(r2(te["CHF_kW_m2"].to_numpy(float),
                           pipe.predict(te)["chf"].to_numpy()))
        rec = dict(scale=sc, C_small=base_small * sc, C_large=base_large * sc,
                   phys_median=float(np.median(list(phys_r2.values()))),
                   pipe_median=float(np.median(lodo)), pipe_worst=float(np.min(lodo)),
                   **{f"phys_{k}": v for k, v in phys_r2.items()})
        rows.append(rec)
        print(f"  scale {sc:4.2f}  C=({base_small*sc:.3f},{base_large*sc:.3f})  "
              f"physics median R2 {rec['phys_median']:+.4f}   "
              f"PIPE LODO median {rec['pipe_median']:+.4f}  worst {rec['pipe_worst']:+.4f}",
              flush=True)
    C.KATTO_C_SMALL, C.KATTO_C_LARGE = base_small, base_large
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "audit_katto_c.csv"), index=False)
    sp = df.pipe_median.max() - df.pipe_median.min()
    print(f"\nPipeline LODO median varies by {sp:.4f} R2 across a 4x range of C_Kc")
    print("wrote audit_katto_c.csv")

if __name__ == "__main__":
    main()

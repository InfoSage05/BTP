"""Leave-one-FLUID-out: the test that prior campaign-disjoint work cannot run.

Context. A 2026 study (campaign-disjoint validation of ML CHF models) already
establishes the outlet-quality leakage identity, campaign-disjoint validation,
and anchored multiplicative log-residual learning -- on a single water /
uniformly-heated-vertical-tube database. Those claims are therefore NOT ours.

What that design cannot test is cross-FLUID transfer, because it has only one
fluid. Our corpus has three (water, R-123, R-134a) and five geometries, so we
can hold out an entire working fluid. That is a strictly harder shift than
holding out a campaign within one homogeneous database, and it is where the
dimensionless representation either earns its keep or does not.

Also measures the transfer/detection tension directly: when a fluid is held out,
what fraction of its rows does the novelty gate actually flag?
"""
from __future__ import annotations
import os, sys, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sklearn.ensemble import ExtraTreesRegressor
import datasets as D, physics_arms as P, pipeline as PL

OUT = P.OUT
FLOW = ["D1_NRC", "D2_Zhao2020", "D3_KAERI_uniform", "D4_KAERI_nonuniform",
        "D5_Helical_R123", "D6_Hardik_helical_water", "D7_Hardik_straight_R123",
        "D8_Pioro_R134a"]

def r2(y, p):
    y = np.asarray(y, float); p = np.nan_to_num(np.asarray(p, float), nan=np.nanmedian(y))
    ss = ((y - y.mean()) ** 2).sum()
    return float(1 - ((y - p) ** 2).sum() / ss) if ss > 0 else np.nan
def w10(y, p):
    return float((np.abs(np.asarray(p, float) - y) / np.abs(y) <= 0.10).mean()) * 100

def build():
    frames = []
    for n in FLOW:
        ds = D.LOADERS[n]()
        f = PL.prepare(ds); f["dataset"] = n; f["fluid_name"] = ds.fluid
        frames.append(f)
    keep = (["dataset", "fluid_name", "CHF_kW_m2", "phys_katto", "phys_biasi",
             "phys_lut", "corr_spread"] + P.DIMLESS)
    return (pd.concat([f.reindex(columns=keep) for f in frames], ignore_index=True)
              .dropna(subset=["CHF_kW_m2", "phys_katto"]).reset_index(drop=True))

def main():
    df = build()
    print(f"corpus {len(df)} rows")
    print(df.groupby("fluid_name").size().to_string(), "\n")

    rows = []
    print("=" * 100)
    print("LEAVE-ONE-FLUID-OUT  (every row of that fluid is unseen, in every geometry)")
    print("=" * 100)
    print(f"{'held-out fluid':<14}{'n_test':>8}{'n_train':>9}{'physics':>10}"
          f"{'pooledML':>11}{'PIPE':>9}{'PIPE w10':>10}{'meanTrust':>11}{'flagged%':>10}")
    for fl in ["R-123", "R-134a"]:
        tr, te = df[df.fluid_name != fl], df[df.fluid_name == fl]
        if len(te) < 5:
            continue
        y = te["CHF_kW_m2"].to_numpy(float)
        rs, ws, ts, fg = [], [], [], []
        pool_rs = []
        for s in (0, 1, 2):
            pipe = PL.CHFPipeline(alpha=0.10, seed=s).fit(tr)
            r = pipe.predict(te)
            rs.append(r2(y, r["chf"].to_numpy())); ws.append(w10(y, r["chf"].to_numpy()))
            t = r["trust"].to_numpy(); ts.append(float(t.mean()))
            # "flagged" = the gate calls it low-trust, i.e. correlation fallback
            fg.append(float((t < 0.33).mean()) * 100)
            m = ExtraTreesRegressor(n_estimators=300, n_jobs=-1, random_state=s)
            m.fit(np.nan_to_num(tr[P.DIMLESS].to_numpy(float), nan=0.0),
                  np.log(tr["CHF_kW_m2"].to_numpy(float)))
            pool_rs.append(r2(y, np.exp(m.predict(
                np.nan_to_num(te[P.DIMLESS].to_numpy(float), nan=0.0)))))
        ph = r2(y, te["phys_katto"].to_numpy(float))
        print(f"{fl:<14}{len(te):>8}{len(tr):>9}{ph:>10.4f}{np.mean(pool_rs):>11.4f}"
              f"{np.mean(rs):>9.4f}{np.mean(ws):>9.1f}%{np.mean(ts):>11.3f}{np.mean(fg):>9.1f}%")
        rows.append(dict(held_fluid=fl, n_test=len(te), n_train=len(tr),
                         physics_R2=ph, pooled_R2=float(np.mean(pool_rs)),
                         PIPE_R2=float(np.mean(rs)), PIPE_R2_sd=float(np.std(rs)),
                         PIPE_w10=float(np.mean(ws)), mean_trust=float(np.mean(ts)),
                         pct_flagged_low_trust=float(np.mean(fg))))

    # --- the tension, stated numerically
    print("\n" + "=" * 100)
    print("TRANSFER / DETECTION TENSION")
    print("=" * 100)
    for rec in rows:
        print(f"  {rec['held_fluid']}: absent from ALL {rec['n_train']} training rows, "
              f"yet only {rec['pct_flagged_low_trust']:.1f}% of its {rec['n_test']} rows "
              f"are flagged low-trust (mean trust {rec['mean_trust']:.3f}).")
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "audit_fluid_transfer.csv"), index=False)
    print("\nwrote audit_fluid_transfer.csv")

if __name__ == "__main__":
    main()

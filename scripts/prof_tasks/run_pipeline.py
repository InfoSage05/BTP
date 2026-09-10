"""Evaluate the CHF pipeline.

Three questions, in order of how much they matter:

 1. Does it clear the pre-registered bar -- beat BOTH pure physics AND a plain
    pooled model on held-out sources? (leave-one-dataset-out)
 2. Does the correlation-disagreement signal actually help? (ablation: the same
    pipeline with that signal switched off)
 3. Do the conformal intervals cover what they claim, in-distribution and out?

Also reports the per-dataset best R2 under 80/20 and 70/30, which is what gets
quoted per dataset.
"""

from __future__ import annotations

import os
import sys
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import train_test_split

import datasets as D
import physics_arms as P
import pipeline as PL

warnings.filterwarnings("ignore")
OUT = P.OUT
FLOW = ["D1_NRC", "D2_Zhao2020", "D3_KAERI_uniform", "D4_KAERI_nonuniform",
        "D5_Helical_R123", "D6_Hardik_helical_water", "D7_Hardik_straight_R123",
        "D8_Pioro_R134a"]


def build_corpus() -> pd.DataFrame:
    frames = []
    for name in FLOW:
        ds = D.LOADERS[name]()
        f = PL.prepare(ds)
        f["dataset"] = name
        f["fluid_name"] = ds.fluid
        frames.append(f)
    keep = (["dataset", "fluid_name", "CHF_kW_m2", "phys_katto", "phys_biasi",
             "phys_lut", "corr_spread"] + PL.DIM)
    df = pd.concat([f.reindex(columns=keep) for f in frames], ignore_index=True)
    return df.dropna(subset=["CHF_kW_m2", "phys_katto"]).reset_index(drop=True)


def coverage(y, lo, hi):
    return float(((y >= lo) & (y <= hi)).mean())


def plain_pooled(tr, te):
    """Competitor: one pooled model on the same features, no physics, no bound."""
    m = ExtraTreesRegressor(n_estimators=300, n_jobs=-1, random_state=0)
    X = np.nan_to_num(tr[PL.DIM].to_numpy(float), nan=0.0)
    m.fit(X, np.log(tr["CHF_kW_m2"].to_numpy(float)))
    return np.exp(m.predict(np.nan_to_num(te[PL.DIM].to_numpy(float), nan=0.0)))


def main():
    df = build_corpus()
    print(f"corpus: {len(df)} rows, {df.dataset.nunique()} datasets, "
          f"fluids {sorted(df.fluid_name.unique())}\n")
    rows, cov_rows = [], []

    # ---------------- 1 & 2: leave-one-dataset-out, with and without the spread signal
    print("=" * 92)
    print("LEAVE-ONE-DATASET-OUT  (an unseen row really is unseen)")
    print("=" * 92)
    print(f"{'held out':<26}{'physics':>9}{'pooledML':>10}{'PIPE':>9}"
          f"{'PIPE-nospread':>15}{'trust':>8}{'cover90':>9}")
    for held in FLOW:
        tr = df[df.dataset != held]
        te = df[df.dataset == held]
        if len(te) < 5:
            continue
        y = te["CHF_kW_m2"].to_numpy(float)
        p_phys = te["phys_katto"].to_numpy(float)
        p_pool = plain_pooled(tr, te)

        out = {}
        for tag, spread in (("PIPE", True), ("PIPE_nospread", False)):
            pipe = PL.CHFPipeline(alpha=0.10, seed=0, use_spread=spread).fit(tr)
            r = pipe.predict(te)
            out[tag] = r
            rows.append(dict(held_out=held, arm=tag, **P.metrics(y, r["chf"], len(tr)),
                             mean_trust=float(r["trust"].mean())))
            cov_rows.append(dict(held_out=held, arm=tag, target=0.90,
                                 actual=coverage(y, r["lo"], r["hi"])))
        for tag, pred in (("pure_physics", p_phys), ("pooled_ML", p_pool)):
            rows.append(dict(held_out=held, arm=tag, **P.metrics(y, pred, len(tr))))

        g = lambda a: [r for r in rows if r["held_out"] == held and r["arm"] == a][0]["R2"]
        print(f"{held:<26}{g('pure_physics'):>9.3f}{g('pooled_ML'):>10.3f}"
              f"{g('PIPE'):>9.3f}{g('PIPE_nospread'):>15.3f}"
              f"{out['PIPE']['trust'].mean():>8.2f}"
              f"{cov_rows[-2]['actual']:>9.2f}")

    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(OUT, "pipeline_loso.csv"), index=False)
    piv = res.pivot_table(index="held_out", columns="arm", values="R2")
    n = len(piv)
    beats_phys = int((piv["PIPE"] > piv["pure_physics"]).sum())
    beats_pool = int((piv["PIPE"] > piv["pooled_ML"]).sum())
    print(f"\nmedian R2  physics {piv['pure_physics'].median():+.3f} | "
          f"pooledML {piv['pooled_ML'].median():+.3f} | "
          f"PIPE {piv['PIPE'].median():+.3f} | "
          f"PIPE-nospread {piv['PIPE_nospread'].median():+.3f}")
    print(f"PIPE beats pure physics on {beats_phys}/{n};  beats pooled ML on {beats_pool}/{n}")
    print("VERDICT:", "PASS" if (beats_phys == n and beats_pool == n)
          else "FAIL against the pre-registered bar")
    d = (piv["PIPE"] - piv["PIPE_nospread"])
    print(f"correlation-disagreement signal: helps on {int((d>0).sum())}/{n} datasets, "
          f"median delta R2 {d.median():+.3f}")

    cov = pd.DataFrame(cov_rows)
    cov.to_csv(os.path.join(OUT, "pipeline_coverage.csv"), index=False)
    print(f"\nconformal 90% intervals, actual coverage on UNSEEN datasets: "
          f"median {cov[cov.arm=='PIPE'].actual.median():.2f}")

    # ---------------- 3: per-dataset random splits (the numbers quoted per dataset)
    print("\n" + "=" * 92)
    print("PER-DATASET, random splits (pipeline trained on that dataset alone)")
    print("=" * 92)
    print(f"{'dataset':<26}{'split':>7}{'R2':>8}{'within10':>10}{'within20':>10}"
          f"{'cover90':>9}")
    per = []
    for name in FLOW:
        sub = df[df.dataset == name].reset_index(drop=True)
        for label, ts in (("80/20", 0.2), ("70/30", 0.3)):
            acc, cvg = [], []
            for s in range(10):
                tr_i, te_i = train_test_split(np.arange(len(sub)), test_size=ts,
                                              random_state=s)
                pipe = PL.CHFPipeline(alpha=0.10, seed=s).fit(sub.iloc[tr_i])
                r = pipe.predict(sub.iloc[te_i])
                yy = sub["CHF_kW_m2"].to_numpy(float)[te_i]
                acc.append(P.metrics(yy, r["chf"], len(tr_i)))
                cvg.append(coverage(yy, r["lo"], r["hi"]))
            a = pd.DataFrame(acc)
            per.append(dict(dataset=name, split=label, n=len(sub),
                            **{c: a[c].mean() for c in a.columns},
                            coverage90=float(np.mean(cvg))))
            print(f"{name:<26}{label:>7}{a.R2.mean():>8.3f}"
                  f"{a.within_10pct.mean()*100:>9.1f}%{a.within_20pct.mean()*100:>9.1f}%"
                  f"{np.mean(cvg):>9.2f}")
    pd.DataFrame(per).to_csv(os.path.join(OUT, "pipeline_per_dataset.csv"), index=False)
    print("\nwrote pipeline_loso.csv, pipeline_coverage.csv, pipeline_per_dataset.csv")


if __name__ == "__main__":
    main()

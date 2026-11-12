"""Strict version of the cross-dataset conformal test -- closing a leakage hole.

THE HOLE. In the fast version, the interval for held-out dataset d is set from
residuals recorded when each other dataset e was held out. But those models were
trained on data that INCLUDED d. So d influenced the very residuals used to size
d's own interval. That is a (mild) form of leakage and a referee would find it.

Direction of the bias: including d gives those models more data, so they predict
e slightly BETTER, so the residuals are slightly SMALLER, so the quantile is
slightly TOO NARROW -> the fast version should if anything UNDER-cover. Observed
coverage was already 0.885, so the argument says the strict number should be at
least as good. That is an argument, not a measurement. Measure it.

STRICT PROTOCOL. For each ordered pair (d, e), d != e, train on ALL datasets
except BOTH d and e, predict e, keep those residuals. d never touches the model.
By symmetry the model for {d,e} serves both orderings, so this is 28 fits, not 56.
"""
from __future__ import annotations
import os, sys, itertools, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import datasets as D, physics_arms as P, pipeline as PL

OUT = P.OUT
FLOW = ["D1_NRC", "D2_Zhao2020", "D3_KAERI_uniform", "D4_KAERI_nonuniform",
        "D5_Helical_R123", "D6_Hardik_helical_water", "D7_Hardik_straight_R123",
        "D8_Pioro_R134a"]
ALPHA = 0.10

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

def logres(y, p):
    return np.abs(np.log(np.clip(y, 1e-9, None)) - np.log(np.clip(p, 1e-9, None)))

def main():
    df = build()
    print(f"corpus {len(df)} rows\n", flush=True)

    # pair_res[(d,e)] = residuals on e from a model that saw NEITHER d NOR e
    pair_res: dict = {}
    for i, (a, b) in enumerate(itertools.combinations(FLOW, 2), 1):
        tr = df[(df.dataset != a) & (df.dataset != b)]
        pipe = PL.CHFPipeline(alpha=ALPHA, seed=0).fit(tr)
        for held, other in ((a, b), (b, a)):
            te = df[df.dataset == held]
            r = pipe.predict(te)
            pair_res[(other, held)] = logres(te["CHF_kW_m2"].to_numpy(float),
                                             r["chf"].to_numpy())
        print(f"  [{i:2d}/28] excluded {a} + {b}", flush=True)

    # true LODO predictions for each d (model trained on all but d)
    print("\nfitting the 8 true leave-one-dataset-out models ...", flush=True)
    rows = []
    for d in FLOW:
        tr, te = df[df.dataset != d], df[df.dataset == d]
        pipe = PL.CHFPipeline(alpha=ALPHA, seed=0).fit(tr)
        r = pipe.predict(te)
        y = te["CHF_kW_m2"].to_numpy(float)
        res = logres(y, r["chf"].to_numpy())
        trust = r["trust"].to_numpy()
        cov_ship = float(((y >= r["lo"].to_numpy()) & (y <= r["hi"].to_numpy())).mean())
        # STRICT pooled residuals: only from models that never saw d
        pool = np.concatenate([pair_res[(d, e)] for e in FLOW if e != d])
        q = float(np.quantile(pool, 1 - ALPHA))
        cov_strict = float((res <= q).mean())
        cov_strict_t = float((res <= q * (1 + 1.0 * (1 - trust))).mean())
        rows.append(dict(held_out=d, n=len(y), cov_shipped=cov_ship,
                         cov_strict=cov_strict, cov_strict_trust=cov_strict_t,
                         q_strict=q, q_in=pipe.q_global_))
        print(f"  {d:26s} shipped {cov_ship:.3f}  STRICT {cov_strict:.3f}  "
              f"+trust {cov_strict_t:.3f}  q={q:.3f}", flush=True)

    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(OUT, "audit_ood_conformal_strict.csv"), index=False)
    print("\n" + "-" * 88)
    for c, lab in (("cov_shipped", "shipped (in-distribution)"),
                   ("cov_strict", "STRICT cross-dataset (no leakage)"),
                   ("cov_strict_trust", "STRICT cross-dataset + trust widening")):
        v = res[c]
        print(f"{lab:40s} median {v.median():.3f}  min {v.min():.3f}  "
              f"mean |cov-0.90| {np.abs(v-0.9).mean():.3f}")
    print("\nwrote audit_ood_conformal_strict.csv")

if __name__ == "__main__":
    main()

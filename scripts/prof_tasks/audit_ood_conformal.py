"""Fix attempt for the project's worst weakness: out-of-distribution coverage.

THE PROBLEM. The shipped pipeline calibrates its conformal interval on a 20%
slice held out of the TRAINING data. That slice is exchangeable with the
training data, so the resulting quantile answers "how wrong am I on rows like
the ones I trained on?". Measured coverage against a 90% target on a genuinely
unseen dataset ranged from 12.4% to 99.5%. The guarantee is void exactly where
it is needed.

THE IDEA (cross-dataset / jackknife-over-datasets conformal). Do not ask how
wrong the model is on familiar rows. Ask how wrong it has historically been on
*entire unseen datasets*, and use THAT distribution.

Concretely, to set the interval for held-out dataset d:
    for every other dataset e, we already know the residuals the pipeline
    produced when e itself was held out. Pool those, take the quantile, and
    use it as d's interval width. d's own residuals are never used, so the
    procedure stays honest.

This replaces an exchangeability assumption that is FALSE (new dataset ~ training
rows) with one that is weaker and more nearly true (new datasets resemble other
new datasets).

Also evaluated: a trust-scaled variant, and the shipped in-distribution method,
on identical predictions, so the only thing that differs is the interval.
"""
from __future__ import annotations
import os, sys, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import datasets as D, physics_arms as P, pipeline as PL

OUT = P.OUT
FLOW = ["D1_NRC", "D2_Zhao2020", "D3_KAERI_uniform", "D4_KAERI_nonuniform",
        "D5_Helical_R123", "D6_Hardik_helical_water", "D7_Hardik_straight_R123",
        "D8_Pioro_R134a"]
ALPHA = 0.10          # target 90% coverage


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
    print(f"corpus {len(df)} rows\n", flush=True)

    # ---- pass 1: standard LODO. Record predictions, trust, and the shipped width.
    store = {}
    for held in FLOW:
        tr, te = df[df.dataset != held], df[df.dataset == held]
        pipe = PL.CHFPipeline(alpha=ALPHA, seed=0).fit(tr)
        r = pipe.predict(te)
        y = te["CHF_kW_m2"].to_numpy(float)
        p = r["chf"].to_numpy()
        store[held] = dict(
            y=y, p=p, trust=r["trust"].to_numpy(),
            lo_ship=r["lo"].to_numpy(), hi_ship=r["hi"].to_numpy(),
            resid=np.abs(np.log(np.clip(y, 1e-9, None)) - np.log(np.clip(p, 1e-9, None))),
            q_in=pipe.q_global_)
        print(f"  {held:26s} n={len(y):6d} shipped q={pipe.q_global_:.4f} "
              f"median|log err|={np.median(store[held]['resid']):.4f}", flush=True)

    # ---- pass 2: cross-dataset conformal
    print("\n" + "=" * 96)
    print("COVERAGE at 90% target -- shipped (in-distribution) vs cross-dataset")
    print("=" * 96)
    print(f"{'held out':<26}{'shipped':>10}{'X-data':>10}{'X-trust':>10}"
          f"{'q_in':>9}{'q_xd':>9}{'width x':>9}")
    rows = []
    for held in FLOW:
        s = store[held]
        # pooled residuals from every OTHER dataset's own held-out run
        pool = np.concatenate([store[o]["resid"] for o in FLOW if o != held])
        q_xd = float(np.quantile(pool, 1 - ALPHA))
        cov_ship = float(((s["y"] >= s["lo_ship"]) & (s["y"] <= s["hi_ship"])).mean())
        cov_xd = float((s["resid"] <= q_xd).mean())
        # trust-scaled variant of the cross-dataset width
        w_t = q_xd * (1.0 + 1.0 * (1.0 - s["trust"]))
        cov_xt = float((s["resid"] <= w_t).mean())
        rows.append(dict(held_out=held, n=len(s["y"]), target=1 - ALPHA,
                         cov_shipped=cov_ship, cov_crossdataset=cov_xd,
                         cov_crossdataset_trust=cov_xt,
                         q_in=s["q_in"], q_xd=q_xd,
                         median_width_ratio=float(q_xd / max(s["q_in"], 1e-9))))
        print(f"{held:<26}{cov_ship:>10.3f}{cov_xd:>10.3f}{cov_xt:>10.3f}"
              f"{s['q_in']:>9.3f}{q_xd:>9.3f}{q_xd/max(s['q_in'],1e-9):>9.2f}")

    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(OUT, "audit_ood_conformal.csv"), index=False)
    print("\n" + "-" * 96)
    for c, lab in (("cov_shipped", "shipped (in-distribution calibration)"),
                   ("cov_crossdataset", "cross-dataset calibration"),
                   ("cov_crossdataset_trust", "cross-dataset + trust widening")):
        v = res[c]
        print(f"{lab:42s} median {v.median():.3f}  min {v.min():.3f}  "
              f"max {v.max():.3f}  mean |cov-0.90| {np.abs(v-0.9).mean():.3f}")
    print("\nwrote audit_ood_conformal.csv")


if __name__ == "__main__":
    main()

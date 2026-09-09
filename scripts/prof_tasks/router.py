"""Phase 4: the seen/unseen router, tested against a pre-registered bar.

The proposal: classify an incoming row as seen or unseen; send seen rows to the
model that did best on that data, and unseen rows to a physics-based fallback.

Two design choices, both deliberate:

  * Novelty is measured in DIMENSIONLESS space (Katto number, density ratio,
    L/D, subcooling ratio, reduced pressure), not in raw engineering units.
    Water at 15 MPa and R-123 at 1 MPa land on the same point there, so the
    question being asked is "have I seen this physics?" rather than "have I
    seen this rig?". Raw-unit distance answers the second question, which is
    the source-confounding effect wearing a disguise.

  * Evaluation is leave-one-DATASET-out on the merged flow corpus, so an
    "unseen" row really is unseen -- a different fluid, geometry or campaign.

PRE-REGISTERED BAR, fixed before the numbers were seen: the router must beat
BOTH the single global model AND pure physics on held-out sources. Matching
either one means the routing has not earned its complexity. Failing is a
publishable result, not a disappointment.
"""

from __future__ import annotations

import os
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

import datasets as D
import physics_arms as P

warnings.filterwarnings("ignore")
OUT = P.OUT
DIM = P.DIMLESS


def build_corpus() -> pd.DataFrame:
    """Merged flow corpus with per-row fluid properties and dimensionless groups."""
    frames = []
    for name in ("D1_NRC", "D2_Zhao2020", "D3_KAERI_uniform", "D4_KAERI_nonuniform",
                 "D5_Helical_R123", "D6_Hardik_helical_water",
                 "D7_Hardik_straight_R123", "D8_Pioro_R134a"):
        ds = D.LOADERS[name]()
        f = P.enrich(ds)
        f["dataset"] = name
        f["fluid_name"] = ds.fluid
        frames.append(f)
    keep = (["dataset", "fluid_name", "CHF_kW_m2", "P_kPa", "G_kg_m2s", "X",
             "_D_m", "_L_m", "rho_l", "rho_g", "sigma", "h_fg", "dh_sub",
             "dh_sub_known", "Bo_actual"] + DIM)
    out = pd.concat([f.reindex(columns=keep) for f in frames], ignore_index=True)
    out = out[(out["CHF_kW_m2"] > 0) & np.isfinite(out["CHF_kW_m2"])]
    return out.reset_index(drop=True)


def mahalanobis(train: np.ndarray, test: np.ndarray) -> np.ndarray:
    """Robust Mahalanobis distance of test rows from the training cloud."""
    t = np.log10(np.clip(train, 1e-12, None))
    e = np.log10(np.clip(test, 1e-12, None))
    mu = np.nanmedian(t, axis=0)
    cov = np.cov(np.nan_to_num(t - mu, nan=0.0), rowvar=False)
    cov += np.eye(cov.shape[0]) * 1e-6
    inv = np.linalg.pinv(cov)
    d = np.nan_to_num(e - mu, nan=0.0)
    return np.sqrt(np.clip(np.einsum("ij,jk,ik->i", d, inv, d), 0, None))


def fit_global(tr: pd.DataFrame, feats: list):
    m = HistGradientBoostingRegressor(random_state=0)
    m.fit(tr[feats].to_numpy(float), np.log(tr["CHF_kW_m2"].to_numpy(float)))
    return m


def main():
    df = build_corpus()
    feats = ["P_kPa", "G_kg_m2s", "X", "_D_m", "_L_m"]
    print(f"corpus: {len(df)} rows across {df.dataset.nunique()} datasets\n")

    rows = []
    for held in sorted(df.dataset.unique()):
        tr = df[df.dataset != held]
        te = df[df.dataset == held]
        if len(te) < 5:
            continue
        y = te["CHF_kW_m2"].to_numpy(float)

        # --- competitor 1: one global model, no routing at all
        gm = fit_global(tr, feats)
        p_global = np.exp(gm.predict(te[feats].to_numpy(float)))

        # --- competitor 2: pure physics, zero fitted parameters
        p_phys = np.clip(P.phys_katto(te), 1e-3, None)

        # --- novelty in dimensionless space
        dist = mahalanobis(tr[DIM].to_numpy(float), te[DIM].to_numpy(float))
        dtr = mahalanobis(tr[DIM].to_numpy(float), tr[DIM].to_numpy(float))
        thr = np.percentile(dtr, 95)          # calibrated on TRAINING rows only
        scale = max(np.percentile(dtr, 95) - np.percentile(dtr, 50), 1e-6)
        w_phys = 1.0 / (1.0 + np.exp(-(dist - thr) / scale))   # 1 => trust physics

        # --- the router
        p_router = np.exp(w_phys * np.log(p_phys) + (1 - w_phys) * np.log(p_global))

        # --- gate v2: categorical identity + range, not continuous distance.
        # Diagnosis from v1: an unseen FLUID is invisible to any distance measure
        # in the prediction feature space -- Pioro's R-134a sits inside the water
        # envelope on every continuous axis, and dimensionless space erases fluid
        # identity by construction. Fluid and geometry are known at prediction
        # time, so gating on them is legitimate, not leakage.
        seen_fluids = set(tr["fluid_name"])
        fluid_new = (~te["fluid_name"].isin(seen_fluids)).to_numpy(float)
        lo = tr[DIM].quantile(0.01)
        hi = tr[DIM].quantile(0.99)
        oob = ((te[DIM] < lo) | (te[DIM] > hi)).mean(axis=1).to_numpy(float)
        w2 = np.clip(np.maximum(fluid_new, oob), 0.0, 1.0)
        p_router2 = np.exp(w2 * np.log(p_phys) + (1 - w2) * np.log(p_global))

        # --- oracle upper bound: perfect seen/unseen knowledge
        best = np.where(np.abs(np.log(p_phys / y)) < np.abs(np.log(p_global / y)),
                        p_phys, p_global)

        for label, pred in (("global_model_no_routing", p_global),
                            ("pure_physics_Katto", p_phys),
                            ("ROUTER_v1_dimensionless", p_router),
                            ("ROUTER_v2_identity", p_router2),
                            ("oracle_upper_bound", best)):
            rows.append(dict(held_out=held, arm=label,
                             mean_w_physics=float(w_phys.mean()),
                             frac_flagged_unseen=float((w_phys > 0.5).mean()),
                             **P.metrics(y, pred, len(tr))))
        print(f"{held:<24} n={len(te):>6}  novelty_flagged={100*(w_phys>0.5).mean():5.1f}%  "
              f"global={rows[-5]['R2']:+.3f}  phys={rows[-4]['R2']:+.3f}  "
              f"v1={rows[-3]['R2']:+.3f}  v2={rows[-2]['R2']:+.3f}  "
              f"oracle={rows[-1]['R2']:+.3f}")

    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(OUT, "router_results.csv"), index=False)

    print("\n" + "=" * 90)
    print("VERDICT against the pre-registered bar")
    print("=" * 90)
    piv = res.pivot_table(index="held_out", columns="arm", values="R2")
    w10 = res.pivot_table(index="held_out", columns="arm", values="within_10pct")
    print("\nR2 by held-out dataset:")
    print(piv.round(3).to_string())
    print("\nwithin +/-10% by held-out dataset:")
    print(w10.round(3).to_string())

    beats_g = (piv["ROUTER_v2_identity"] > piv["global_model_no_routing"]).sum()
    beats_p = (piv["ROUTER_v2_identity"] > piv["pure_physics_Katto"]).sum()
    n = len(piv)
    print(f"\nrouter beats the global model on {beats_g}/{n} held-out datasets")
    print(f"router beats pure physics      on {beats_p}/{n} held-out datasets")
    print(f"\nmedian R2   v1 {piv['ROUTER_v1_dimensionless'].median():+.3f} | "
          f"v2 {piv['ROUTER_v2_identity'].median():+.3f} | "
          f"global {piv['global_model_no_routing'].median():+.3f} | "
          f"physics {piv['pure_physics_Katto'].median():+.3f} | "
          f"oracle {piv['oracle_upper_bound'].median():+.3f}")
    verdict = ("PASS" if (beats_g == n and beats_p == n) else
               "FAIL -- routing did not earn its complexity")
    print(f"\nVERDICT: {verdict}")
    with open(os.path.join(OUT, "ANSWER_Q3_router.txt"), "w") as f:
        f.write(piv.round(4).to_string() + "\n\n" + w10.round(4).to_string() +
                f"\n\nrouter beats global on {beats_g}/{n}; beats physics on {beats_p}/{n}"
                f"\nVERDICT: {verdict}\n")
    print("written -> results/prof_tasks/ANSWER_Q3_router.txt")


if __name__ == "__main__":
    main()

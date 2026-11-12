"""Which learner should supply the CORRECTION, inside the anchored architecture?

The project's 961-run grid compared 13 learners STANDALONE -- predicting CHF
directly. That answers a different question. Inside this pipeline the learner
does not predict CHF; it predicts log(CHF / backbone), a small, well-behaved
residual. A learner that wins at one task need not win at the other, and the
choice of ExtraTrees was inherited from the standalone grid, never re-tested here.

Evaluated by leave-one-dataset-out, which is the only protocol in this project
not contaminated by source confounding. 3 seeds for stochastic learners.
"""
from __future__ import annotations
import os, sys, time, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sklearn.ensemble import (ExtraTreesRegressor, RandomForestRegressor,
                              HistGradientBoostingRegressor)
from sklearn.linear_model import Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
import xgboost as xgb, lightgbm as lgb
from catboost import CatBoostRegressor
import datasets as D, physics_arms as P, pipeline as PL

OUT = P.OUT
FLOW = ["D1_NRC", "D2_Zhao2020", "D3_KAERI_uniform", "D4_KAERI_nonuniform",
        "D5_Helical_R123", "D6_Hardik_helical_water", "D7_Hardik_straight_R123",
        "D8_Pioro_R134a"]

MODELS = {
    "ExtraTrees300 (shipped)": lambda s: ExtraTreesRegressor(n_estimators=300, n_jobs=-1, random_state=s),
    "RandomForest300":         lambda s: RandomForestRegressor(n_estimators=300, n_jobs=-1, random_state=s),
    "HistGB":                  lambda s: HistGradientBoostingRegressor(random_state=s),
    "XGBoost":                 lambda s: xgb.XGBRegressor(n_estimators=400, max_depth=6, learning_rate=0.05,
                                                          subsample=0.8, colsample_bytree=0.8,
                                                          n_jobs=-1, random_state=s, verbosity=0),
    "LightGBM":                lambda s: lgb.LGBMRegressor(n_estimators=400, learning_rate=0.05,
                                                           n_jobs=-1, random_state=s, verbose=-1),
    "CatBoost":                lambda s: CatBoostRegressor(iterations=400, depth=6, learning_rate=0.05,
                                                           random_seed=s, verbose=0, allow_writing_files=False),
    "MLP(128,64)":             lambda s: make_pipeline(StandardScaler(),
                                                       MLPRegressor(hidden_layer_sizes=(128, 64), max_iter=600,
                                                                    early_stopping=True, random_state=s)),
    "Ridge":                   lambda s: make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
    "kNN_k5":                  lambda s: make_pipeline(StandardScaler(), KNeighborsRegressor(n_neighbors=5)),
    "SVR_RBF":                 lambda s: make_pipeline(StandardScaler(), SVR(C=10.0, epsilon=0.05)),
}
STOCHASTIC = {"ExtraTrees300 (shipped)", "RandomForest300", "XGBoost", "LightGBM",
              "CatBoost", "MLP(128,64)", "HistGB"}
SLOW_BIG = {"SVR_RBF"}          # O(n^2)+ on 28k rows; subsample the FIT only

def r2(y, p):
    y = np.asarray(y, float); p = np.nan_to_num(np.asarray(p, float), nan=np.nanmedian(y))
    ss = ((y - y.mean()) ** 2).sum()
    return float(1 - ((y - p) ** 2).sum() / ss) if ss > 0 else np.nan
def w10(y, p):
    return float((np.abs(np.asarray(p, float) - y) / np.abs(y) <= 0.10).mean()) * 100

def build():
    fr = []
    for n in FLOW:
        ds = D.LOADERS[n]()
        f = PL.prepare(ds); f["dataset"] = n; f["fluid_name"] = ds.fluid; fr.append(f)
    keep = (["dataset", "fluid_name", "CHF_kW_m2", "phys_katto", "phys_biasi",
             "phys_lut", "corr_spread"] + P.DIMLESS)
    return (pd.concat([f.reindex(columns=keep) for f in fr], ignore_index=True)
              .dropna(subset=["CHF_kW_m2", "phys_katto"]).reset_index(drop=True))

def main():
    df = build()
    print(f"corpus {len(df)} rows, LODO over {len(FLOW)} datasets\n", flush=True)
    rows = []
    for name, fac in MODELS.items():
        seeds = (0, 1, 2) if name in STOCHASTIC else (0,)
        t0 = time.time(); per = {}
        for held in FLOW:
            tr, te = df[df.dataset != held], df[df.dataset == held]
            if name in SLOW_BIG and len(tr) > 6000:
                tr = tr.sample(6000, random_state=0)
            y = te["CHF_kW_m2"].to_numpy(float)
            rs, ws = [], []
            for s in seeds:
                try:
                    m = PL.CHFPipeline(alpha=0.10, seed=s, model=fac).fit(tr)
                    p = m.predict(te)["chf"].to_numpy()
                    rs.append(r2(y, p)); ws.append(w10(y, p))
                except Exception as e:
                    print(f"    {name} on {held}: FAILED {type(e).__name__}: {str(e)[:70]}", flush=True)
            if rs:
                per[held] = (float(np.mean(rs)), float(np.mean(ws)))
        if not per:
            continue
        r_vals = np.array([v[0] for v in per.values()])
        w_vals = np.array([v[1] for v in per.values()])
        rows.append(dict(model=name, n_seeds=len(seeds),
                         R2_median=float(np.median(r_vals)), R2_worst=float(r_vals.min()),
                         R2_mean=float(r_vals.mean()), w10_median=float(np.median(w_vals)),
                         secs=round(time.time() - t0, 1),
                         **{f"R2_{k}": v[0] for k, v in per.items()}))
        print(f"  {name:26s} median R2 {rows[-1]['R2_median']:+.4f}  worst {rows[-1]['R2_worst']:+.4f}"
              f"  median +/-10% {rows[-1]['w10_median']:5.1f}%  ({rows[-1]['secs']:.0f}s)", flush=True)
        pd.DataFrame(rows).to_csv(os.path.join(OUT, "audit_correction_model.csv"), index=False)

    res = pd.DataFrame(rows).sort_values("R2_median", ascending=False)
    print("\n" + "=" * 92)
    print("CORRECTION LEARNER, RANKED BY LEAVE-ONE-DATASET-OUT MEDIAN R2")
    print("=" * 92)
    print(res[["model", "R2_median", "R2_worst", "R2_mean", "w10_median", "secs"]]
          .to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("\nwrote audit_correction_model.csv")

if __name__ == "__main__":
    main()

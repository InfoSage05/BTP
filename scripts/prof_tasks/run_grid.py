"""Phase 2: the model x dataset x formulation x split grid.

Models train on log(CHF); every metric is computed back on the raw kW/m2 scale,
because that is the scale the professor's +/-10% question is asked on.

Splits: 80/20 and 70/30 random over many seeds (mean +/- sd), plus leave-one-
source-out where a group column exists. Single-split numbers are not reported --
this project has already had to retract a single-seed headline.
"""

from __future__ import annotations

import os
import sys
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (ExtraTreesRegressor, HistGradientBoostingRegressor,
                              RandomForestRegressor)
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import GroupKFold, train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures, StandardScaler
from sklearn.svm import SVR

import datasets as D

warnings.filterwarnings("ignore")
OUT = os.path.join(os.path.dirname(__file__), "..", "..", "results", "prof_tasks")
os.makedirs(OUT, exist_ok=True)

BIG = 8000          # above this, subsample for the O(n^3)/O(n^2) learners
SEEDS_SMALL = 20
SEEDS_BIG = 5


# ---------------------------------------------------------------- metrics
def metrics(y, p, n_train):
    y = np.asarray(y, float)
    p = np.clip(np.asarray(p, float), 1e-6, None)
    rel = np.abs(p - y) / np.abs(y)
    ss = ((y - y.mean()) ** 2).sum()
    return dict(
        n_test=len(y), n_train=n_train,
        R2=1.0 - ((y - p) ** 2).sum() / ss if ss > 0 else np.nan,
        within_10pct=float((rel <= 0.10).mean()),
        within_20pct=float((rel <= 0.20).mean()),
        MAPE_pct=float(np.mean(rel) * 100),
        RMSE=float(np.sqrt(np.mean((y - p) ** 2))),
        safe_side_frac=float((p <= y).mean()),
    )


# ---------------------------------------------------------------- models
def build_models():
    """Model zoo. Scale-sensitive learners get a scaler inside the pipeline."""
    m = {
        "Linear": LinearRegression(),
        "Poly2_Ridge": Pipeline([("poly", PolynomialFeatures(2, include_bias=False)),
                                 ("ridge", Ridge(alpha=1.0))]),
        "kNN_k5": KNeighborsRegressor(n_neighbors=5),
        "RandomForest": RandomForestRegressor(n_estimators=300, n_jobs=-1, random_state=0),
        "ExtraTrees": ExtraTreesRegressor(n_estimators=300, n_jobs=-1, random_state=0),
        "HistGB": HistGradientBoostingRegressor(random_state=0),
        "SVR_RBF": SVR(C=10.0, gamma="scale", epsilon=0.05),
        "MLP": MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=2000,
                            early_stopping=True, random_state=0),
    }
    try:
        from xgboost import XGBRegressor
        m["XGBoost"] = XGBRegressor(n_estimators=400, max_depth=6, learning_rate=0.05,
                                    subsample=0.9, colsample_bytree=0.9,
                                    n_jobs=-1, random_state=0, verbosity=0)
    except ImportError:
        pass
    try:
        from lightgbm import LGBMRegressor
        m["LightGBM"] = LGBMRegressor(n_estimators=400, learning_rate=0.05,
                                      n_jobs=-1, random_state=0, verbose=-1)
    except ImportError:
        pass
    try:
        from catboost import CatBoostRegressor
        m["CatBoost"] = CatBoostRegressor(iterations=400, depth=6, learning_rate=0.05,
                                          verbose=0, random_seed=0)
    except ImportError:
        pass
    # GPR is O(n^3); fit_predict subsamples its training set to 2000 rows so it
    # can still be scored on every dataset. That subsampling is reported, because
    # "GPR on 2000 of 24443 rows" is not the same model as GPR on all of a small set.
    m["GPR_Matern"] = GaussianProcessRegressor(
        kernel=ConstantKernel(1.0) * Matern(nu=2.5) + WhiteKernel(1e-3),
        normalize_y=True, random_state=0)
    return m


SCALE_SENSITIVE = {"Linear", "Poly2_Ridge", "kNN_k5", "SVR_RBF", "MLP", "GPR_Matern"}


def make_pipeline(name, model, num_cols, cat_cols):
    pre = ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                          ("sc", StandardScaler() if name in SCALE_SENSITIVE
                           else "passthrough")]), num_cols),
        ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                          ("oh", OneHotEncoder(handle_unknown="ignore",
                                               sparse_output=False))]), cat_cols),
    ], remainder="drop")
    return Pipeline([("pre", pre), ("model", model)])


def fit_predict(name, model, Xtr, ytr, Xte, num_cols, cat_cols, seed):
    """Train on log CHF, return predictions on the raw scale."""
    if name == "GPR_Matern" and len(Xtr) > 2000:
        idx = np.random.RandomState(seed).choice(len(Xtr), 2000, replace=False)
        Xtr, ytr = Xtr.iloc[idx], ytr[idx]
    if name == "SVR_RBF" and len(Xtr) > 6000:
        idx = np.random.RandomState(seed).choice(len(Xtr), 6000, replace=False)
        Xtr, ytr = Xtr.iloc[idx], ytr[idx]
    pipe = make_pipeline(name, model, num_cols, cat_cols)
    pipe.fit(Xtr, np.log(ytr))
    return np.exp(pipe.predict(Xte))


# ---------------------------------------------------------------- runner
def run_dataset(ds, log):
    rows = []
    n_seeds = SEEDS_BIG if ds.n > BIG else SEEDS_SMALL
    for form in ("local", "inlet", "reduced"):
        feats = ds.features(form)
        if not feats:
            continue
        df = ds.df.dropna(subset=["CHF_kW_m2"])
        df = df[df["CHF_kW_m2"] > 0]
        Xall = df[feats]
        yall = df["CHF_kW_m2"].to_numpy(float)
        groups = df[ds.group].to_numpy() if ds.group else None
        num_cols = [c for c in feats if pd.api.types.is_numeric_dtype(Xall[c])]
        cat_cols = [c for c in feats if c not in num_cols]

        for mname, proto in build_models().items():
            # ---- random splits, many seeds
            for label, test_size in (("80/20", 0.20), ("70/30", 0.30)):
                acc = []
                for s in range(n_seeds):
                    tr, te = train_test_split(np.arange(len(df)), test_size=test_size,
                                              random_state=s)
                    try:
                        p = fit_predict(mname, proto, Xall.iloc[tr], yall[tr],
                                        Xall.iloc[te], num_cols, cat_cols, s)
                        acc.append(metrics(yall[te], p, len(tr)))
                    except Exception as e:
                        log(f"      ! {ds.name} {form} {mname} {label} seed{s}: {e}")
                if acc:
                    a = pd.DataFrame(acc)
                    rows.append(dict(dataset=ds.name, n=ds.n, fluid=ds.fluid,
                                     geometry=ds.geometry, form=form, model=mname,
                                     split=label, n_seeds=len(acc),
                                     **{f"{k}_mean": a[k].mean() for k in a.columns},
                                     **{f"{k}_sd": a[k].std() for k in
                                        ("R2", "within_10pct", "MAPE_pct")}))
            # ---- leave-one-source-out
            if groups is not None and len(np.unique(groups)) >= 2:
                gk = GroupKFold(n_splits=min(len(np.unique(groups)), 10))
                acc = []
                for tr, te in gk.split(Xall, yall, groups):
                    try:
                        p = fit_predict(mname, proto, Xall.iloc[tr], yall[tr],
                                        Xall.iloc[te], num_cols, cat_cols, 0)
                        acc.append(metrics(yall[te], p, len(tr)))
                    except Exception as e:
                        log(f"      ! {ds.name} {form} {mname} LOSO: {e}")
                if acc:
                    a = pd.DataFrame(acc)
                    rows.append(dict(dataset=ds.name, n=ds.n, fluid=ds.fluid,
                                     geometry=ds.geometry, form=form, model=mname,
                                     split="LOSO", n_seeds=len(acc),
                                     **{f"{k}_mean": a[k].mean() for k in a.columns},
                                     **{f"{k}_sd": a[k].std() for k in
                                        ("R2", "within_10pct", "MAPE_pct")}))
            log(f"   {ds.name:<24} {form:<8} {mname:<14} done")
    return rows


def main():
    logf = open(os.path.join(OUT, "grid_run.log"), "w", buffering=1)

    def log(msg):
        print(msg)
        logf.write(msg + "\n")

    t0 = time.time()
    all_rows = []
    only = sys.argv[1:] or list(D.LOADERS)
    for name in only:
        ds = D.LOADERS[name]()
        log(f"\n=== {ds.name}  ({ds.n} rows, {ds.fluid}, {ds.geometry}) ===")
        r = run_dataset(ds, log)
        all_rows += r
        pd.DataFrame(all_rows).to_csv(os.path.join(OUT, "grid_results.csv"), index=False)
        log(f"   -> {len(r)} result rows, {time.time()-t0:.0f}s elapsed")

    df = pd.DataFrame(all_rows)
    df.to_csv(os.path.join(OUT, "grid_results.csv"), index=False)
    log(f"\nDONE  {len(df)} rows in {time.time()-t0:.0f}s -> results/prof_tasks/grid_results.csv")
    logf.close()


if __name__ == "__main__":
    main()

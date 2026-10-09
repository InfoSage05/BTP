"""
Real predictions for Random Forest, Gradient Boosting, and XGBoost on the same
Stage-1 interpolation/extrapolation test splits used for the MLP parity plot
(generate_mlp_predictions.py), so all four models are directly comparable.
Every point saved here is a real test-set row and a real model prediction --
nothing is synthetic or invented.

Output: data/processed/stage1/{model}_predictions_{split}.csv
  model in {random_forest, gradient_boosting, xgboost}
  split in {interp, extrap}
"""
import os
import sys
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

sys.path.insert(0, "scripts/chf_pipeline")
from data_prep import FEATURE_COLS, TARGET_COL, OUT_DIR

try:
    from xgboost import XGBRegressor
    HAVE_XGB = True
except ImportError:
    HAVE_XGB = False

RNG_SEED = 42

MODELS = {
    "random_forest": lambda: RandomForestRegressor(
        n_estimators=400, max_depth=None, n_jobs=-1, random_state=RNG_SEED),
    "gradient_boosting": lambda: GradientBoostingRegressor(
        n_estimators=400, max_depth=3, learning_rate=0.05, random_state=RNG_SEED),
}
if HAVE_XGB:
    MODELS["xgboost"] = lambda: XGBRegressor(
        n_estimators=400, max_depth=6, learning_rate=0.05,
        subsample=0.9, colsample_bytree=0.9, random_state=RNG_SEED,
        n_jobs=-1, tree_method="hist")
else:
    print("xgboost not installed -- skipping XGBoost predictions")


def metrics(true, pred):
    rel_err = (pred - true) / true
    ss_res = np.sum((true - pred) ** 2)
    ss_tot = np.sum((true - true.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot
    mape = np.mean(np.abs(rel_err)) * 100
    within_10 = np.mean(np.abs(rel_err) <= 0.10) * 100
    return r2, mape, within_10


def main():
    for split_name in ["interp", "extrap"]:
        train = pd.read_csv(os.path.join(OUT_DIR, f"train_{split_name}.csv"))
        test = pd.read_csv(os.path.join(OUT_DIR, f"test_{split_name}.csv"))

        x_tr = train[FEATURE_COLS].to_numpy(dtype=np.float64)
        y_tr = train[TARGET_COL].to_numpy(dtype=np.float64)
        x_test = test[FEATURE_COLS].to_numpy(dtype=np.float64)
        y_test = test[TARGET_COL].to_numpy(dtype=np.float64)

        for model_name, make_model in MODELS.items():
            print(f"\n=== {model_name} | {split_name} split ===")
            model = make_model()
            model.fit(x_tr, np.log(y_tr))
            pred = np.exp(model.predict(x_test))
            true = y_test

            r2, mape, within_10 = metrics(true, pred)
            print(f"  test: R2={r2:.4f} MAPE={mape:.2f}% within10%={within_10:.1f}%")

            out = pd.DataFrame({"true_CHF_kW_m2": true, "pred_CHF_kW_m2": pred})
            out_path = os.path.join(OUT_DIR, f"{model_name}_predictions_{split_name}.csv")
            out.to_csv(out_path, index=False)
            print(f"  saved {out_path}")


if __name__ == "__main__":
    main()

"""
Stage 1 data preparation: combine the Stage-0 synthetic corpus with the real
core pretraining data (NRC raw 24,579-point database), harmonize schema,
add dimensionless features, and build BOTH evaluation splits:

  - interpolation split: random 80/10/10 train/val/test over the real core
    data (the "easy" split every paper reports, and that misleadingly gets
    R^2 ~ 0.99 -- kept for comparability, not because it's the real test).
  - extrapolation split: train/val on real core data with Pressure <=
    P_CUTOFF_KPA, test on real core data with Pressure > P_CUTOFF_KPA. This
    is the split that actually predicts out-of-distribution behavior
    (mirrors Yang et al. 2025's pressure-cutoff methodology).

The synthetic corpus is used ONLY for pretraining (never appears in any
evaluation split) and is not pressure-cut -- it covers the full LUT range,
which is intentional: the model should see the full physical trend during
pretraining even when fine-tuning data is restricted to low pressure.

Output: data/processed/stage1/{train,val,test}_interp.csv,
        data/processed/stage1/{train,val,test}_extrap.csv,
        data/processed/stage1/synthetic_pretrain.csv
"""
import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, "scripts/chf_pipeline")
from physics_features import add_dimensionless_features

SYNTHETIC_PATH = "data/processed/lut_synthetic/lut_synthetic_pretraining.csv"
NRC_PATH = "data/raw/pretraining/nrc_groeneveld_24579pt_chf_database.csv"
OUT_DIR = "data/processed/stage1"

P_CUTOFF_KPA = 14000.0  # easier cutoff: train pool (18,585) >> test (5,994).
                         # An earlier run used 8000 (harder: train(10,831) < test(13,748));
                         # see data/processed/stage1/README.md for that comparison.
RNG_SEED = 42

FEATURE_COLS = ["P_kPa", "G_kg_m2s", "X", "D_mm",
                 "P_reduced", "rho_l_kg_m3", "rho_g_kg_m3", "density_ratio"]
TARGET_COL = "CHF_kW_m2"


def load_real_core():
    df = pd.read_csv(NRC_PATH)
    df = df.rename(columns={
        "Tube Diameter": "D_m", "Pressure": "P_kPa", "Mass Flux": "G_kg_m2s",
        "Outlet Quality": "X", "CHF": "CHF_kW_m2",
    })
    df["D_mm"] = df["D_m"] * 1000.0
    df = df[["P_kPa", "G_kg_m2s", "X", "D_mm", "CHF_kW_m2"]].dropna()
    # drop physically invalid rows (zero/negative CHF or diameter)
    df = df[(df["CHF_kW_m2"] > 0) & (df["D_mm"] > 0)].reset_index(drop=True)
    df = add_dimensionless_features(df, fluid="water", p_col="P_kPa", g_col="G_kg_m2s")
    df = df.dropna(subset=FEATURE_COLS)  # drop rows where CoolProp lookup failed
    return df


def make_interp_split(df, rng):
    idx = rng.permutation(len(df))
    n = len(df)
    n_train = int(n * 0.8)
    n_val = int(n * 0.1)
    train = df.iloc[idx[:n_train]]
    val = df.iloc[idx[n_train:n_train + n_val]]
    test = df.iloc[idx[n_train + n_val:]]
    return train, val, test


def make_extrap_split(df, rng):
    below = df[df["P_kPa"] <= P_CUTOFF_KPA]
    above = df[df["P_kPa"] > P_CUTOFF_KPA]
    idx = rng.permutation(len(below))
    n_val = int(len(below) * 0.15)
    val = below.iloc[idx[:n_val]]
    train = below.iloc[idx[n_val:]]
    test = above
    return train, val, test


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    rng = np.random.default_rng(RNG_SEED)

    real = load_real_core()
    print(f"Real core data (cleaned): {len(real)} rows")

    synthetic = pd.read_csv(SYNTHETIC_PATH)
    synthetic = synthetic[FEATURE_COLS + [TARGET_COL]].dropna()
    print(f"Synthetic pretraining data: {len(synthetic)} rows")
    synthetic.to_csv(os.path.join(OUT_DIR, "synthetic_pretrain.csv"), index=False)

    tr_i, va_i, te_i = make_interp_split(real, rng)
    print(f"Interpolation split: train={len(tr_i)} val={len(va_i)} test={len(te_i)}")
    tr_i.to_csv(os.path.join(OUT_DIR, "train_interp.csv"), index=False)
    va_i.to_csv(os.path.join(OUT_DIR, "val_interp.csv"), index=False)
    te_i.to_csv(os.path.join(OUT_DIR, "test_interp.csv"), index=False)

    tr_e, va_e, te_e = make_extrap_split(real, rng)
    print(f"Extrapolation split (cutoff {P_CUTOFF_KPA} kPa): "
          f"train={len(tr_e)} val={len(va_e)} test={len(te_e)}")
    tr_e.to_csv(os.path.join(OUT_DIR, "train_extrap.csv"), index=False)
    va_e.to_csv(os.path.join(OUT_DIR, "val_extrap.csv"), index=False)
    te_e.to_csv(os.path.join(OUT_DIR, "test_extrap.csv"), index=False)


if __name__ == "__main__":
    main()

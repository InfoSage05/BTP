"""
Stage 2: fine-tune the Stage-1 pretrained MLP on each small, real, distinct
target-domain dataset in data/raw/fine_tuning/, and compare against a
from-scratch MLP (same architecture, no pretraining) trained on the exact
same per-domain split. The comparison IS the point of Stage 2: does
pretraining on the synthetic LUT corpus actually help each domain, or not?

Domain coverage decisions (see data/processed/stage2/README.md for the full
writeup):
  - hardik2016 (helical coils, R123): included. Coil_No maps to inner
    diameter, taken from the source paper own Table 4 (Hardik and Prabhu,
    Appl. Thermal Eng. 112 (2017) 1223-1239), read earlier in this project:
    Coil_1=6mm, Coil_2=6mm, Coil_3=8mm, Coil_4=8mm, Coil_5=9.7mm, Coil_6=10mm
    -- these are real published values, not guesses; the per-coil row counts
    (47,29,18,22,31,9) match the CSV own Coil_No distribution exactly, which
    is itself a strong cross-check that the mapping is correct.
  - hardik2017 (straight tubes, R123): included, D_mm given directly.
  - kaeri_tr1665_uniform / nonuniform (water): included. Pressure is in Pa
    in the source (converted /1000 to kPa), Diameter in m (converted x1000
    to mm), HeatFlux in W/m^2 (converted /1000 to kW/m^2) and used AS the
    CHF value for that row -- for uniform heating this is the standard
    interpretation (constant flux along the tube equals the CHF trigger
    value). For nonuniform, this treats each row reported HeatFlux as a
    per-test summary CHF metric, not a full axial profile -- a real
    simplification, flagged here and in the README, not modeled as a
    sequence.
  - helical_coil_r123_appendixCD.csv: SKIPPED. No diameter column in this
    file and no reliable source table available in this session to
    reconstruct one (this is a DIFFERENT Mendeley dataset from hardik2016,
    not guaranteed to share the same coil geometries) -- rather than
    fabricate diameters, this domain is left out of Stage 2 entirely.
  - pinfin_chf_water_fc72.csv: SKIPPED. This is pool-boiling data on pin-fin
    surfaces -- it has no pressure, mass-flux, or quality columns at all
    (only subcooling). It is not expressible in the (P,G,X,D) flow-boiling
    feature schema this model uses; fine-tuning on it would require a
    different input space entirely, out of scope for this pass.
  - zhao2020_chf_flowboiling_tubes.csv: included, filtered to
    geometry=="tube" rows only (1439/1865) -- annulus/plate geometries
    do not share a consistent diameter definition with the tube-based
    feature schema, so they are excluded rather than silently mixed in.
"""
import os
import sys
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

sys.path.insert(0, "scripts/chf_pipeline")
from models import SmallMLP
from physics_features import add_dimensionless_features
from data_prep import FEATURE_COLS, TARGET_COL

CKPT_DIR = "data/processed/stage1/checkpoints"
FT_DATA_DIR = "data/raw/fine_tuning"
OUT_DIR = "data/processed/stage2"
RNG_SEED = 42
TEST_FRAC = 0.20
FINETUNE_EPOCHS = 150
SCRATCH_EPOCHS = 150
PATIENCE = 20
LR_FINETUNE = 1e-4
LR_SCRATCH = 2e-3

HARDIK2016_COIL_DIAMETER_MM = {
    "Coil_1": 6.0, "Coil_2": 6.0, "Coil_3": 8.0,
    "Coil_4": 8.0, "Coil_5": 9.7, "Coil_6": 10.0,
}


def load_hardik2016():
    df = pd.read_csv(os.path.join(FT_DATA_DIR, "hardik2016_helical_coils_r123_lowpressure_chf.csv"))
    df["D_mm"] = df["Coil_No"].map(HARDIK2016_COIL_DIAMETER_MM)
    assert df["D_mm"].notna().all(), "unmapped Coil_No in hardik2016"
    df["P_kPa"] = df["P_bar"] * 100.0
    df["X"] = df["xe"]
    return df[["P_kPa", "G_kg_m2s", "X", "D_mm", "CHF_kW_m2"]].copy(), "r123"


def load_hardik2017():
    df = pd.read_csv(os.path.join(FT_DATA_DIR, "hardik2017_straight_tubes_r123_chf.csv"))
    df["P_kPa"] = df["P_bar"] * 100.0
    df["X"] = df["xe"]
    return df[["P_kPa", "G_kg_m2s", "X", "D_mm", "CHF_kW_m2"]].copy(), "r123"


def load_kaeri(fname, quality_col):
    df = pd.read_csv(os.path.join(FT_DATA_DIR, fname))
    df["P_kPa"] = df["Pressure"] / 1000.0
    df["D_mm"] = df["Diameter"] * 1000.0
    df["G_kg_m2s"] = df["MassFlux"]
    df["X"] = df[quality_col]
    df["CHF_kW_m2"] = df["HeatFlux"] / 1000.0
    return df[["P_kPa", "G_kg_m2s", "X", "D_mm", "CHF_kW_m2"]].dropna().copy(), "water"


def load_zhao2020():
    df = pd.read_csv(os.path.join(FT_DATA_DIR, "zhao2020_chf_flowboiling_tubes.csv"), encoding="utf-8-sig")
    df = df[df["geometry"] == "tube"].copy()
    df["P_kPa"] = df["pressure [MPa]"] * 1000.0
    df["G_kg_m2s"] = df["mass_flux [kg/m2-s]"]
    df["X"] = df["x_e_out [-]"]
    df["D_mm"] = df["D_h [mm]"]
    df["CHF_kW_m2"] = df["chf_exp [MW/m2]"] * 1000.0
    return df[["P_kPa", "G_kg_m2s", "X", "D_mm", "CHF_kW_m2"]].dropna().copy(), "water"


DOMAIN_LOADERS = {
    "hardik2016_coils_r123": load_hardik2016,
    "hardik2017_straight_tubes_r123": load_hardik2017,
    "kaeri_uniform_water": lambda: load_kaeri("kaeri_tr1665_uniform_chf.csv", "EquilibriumQuality"),
    "kaeri_nonuniform_water": lambda: load_kaeri("kaeri_tr1665_nonuniform_chf.csv", "Quality"),
    "zhao2020_tubes_water": load_zhao2020,
}

SKIPPED_DOMAINS = {
    "helical_coil_r123_appendixCD": "no diameter column, no reliable source table available this session",
    "pinfin_chf_water_fc72": "pool boiling -- no P/G/X columns, incompatible feature schema",
}


class Standardizer:
    def transform(self, x):
        return (x - self.mean) / self.std


def to_tensor(df, feat_scaler, target_mean, target_std):
    x = feat_scaler.transform(df[FEATURE_COLS].to_numpy(dtype=np.float32))
    y_log = np.log(df[TARGET_COL].to_numpy(dtype=np.float32))
    y_log = (y_log - target_mean) / target_std
    return torch.tensor(x, dtype=torch.float32), torch.tensor(y_log, dtype=torch.float32)


def evaluate(model, x, y_log, target_mean, target_std):
    model.eval()
    with torch.no_grad():
        pred_log = model(x).numpy() * target_std + target_mean
    true_log = y_log.numpy() * target_std + target_mean
    pred, true = np.exp(pred_log), np.exp(true_log)
    rel_err = (pred - true) / true
    rmse = float(np.sqrt(np.mean((pred - true) ** 2)))
    rrmse = float(np.sqrt(np.mean(rel_err ** 2)))
    mape = float(np.mean(np.abs(rel_err)) * 100)
    ss_res = np.sum((true - pred) ** 2)
    ss_tot = np.sum((true - true.mean()) ** 2)
    r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan")
    within_10pct = float(np.mean(np.abs(rel_err) <= 0.10) * 100)
    return {"R2": r2, "rRMSE": rrmse, "MAPE_%": mape, "RMSE_kW_m2": rmse,
            "within_10pct_%": within_10pct, "n": int(len(true))}


def train(model, x_tr, y_tr, x_val, y_val, epochs, lr, label):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    mse = nn.MSELoss()
    best_val, best_state, bad = float("inf"), None, 0
    n = len(x_tr)
    batch_size = min(64, max(8, n // 4))
    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(n)
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            optimizer.zero_grad()
            loss = mse(model(x_tr[idx]), y_tr[idx])
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            val_loss = mse(model(x_val), y_val).item()
        if val_loss < best_val - 1e-6:
            best_val, best_state, bad = val_loss, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            bad += 1
        if bad >= PATIENCE:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    rng = np.random.default_rng(RNG_SEED)

    with open(os.path.join(CKPT_DIR, "scaler.pkl"), "rb") as f:
        scaler_bundle = pickle.load(f)
    feat_scaler = scaler_bundle["feat_scaler"]
    target_mean, target_std = scaler_bundle["target_mean"], scaler_bundle["target_std"]
    pretrained_state = torch.load(os.path.join(CKPT_DIR, "mlp_pretrained.pt"))

    results = []
    for domain_name, loader in DOMAIN_LOADERS.items():
        raw_df, fluid = loader()
        df = raw_df[(raw_df["CHF_kW_m2"] > 0) & (raw_df["D_mm"] > 0)].reset_index(drop=True)
        df = add_dimensionless_features(df, fluid=fluid, p_col="P_kPa", g_col="G_kg_m2s")
        n_before = len(df)
        df = df.dropna(subset=FEATURE_COLS)
        n_dropped = n_before - len(df)

        idx = rng.permutation(len(df))
        n_test = max(1, int(len(df) * TEST_FRAC))
        n_val = max(1, int(len(df) * 0.15))
        test_df = df.iloc[idx[:n_test]]
        val_df = df.iloc[idx[n_test:n_test + n_val]]
        train_df = df.iloc[idx[n_test + n_val:]]

        x_tr, y_tr = to_tensor(train_df, feat_scaler, target_mean, target_std)
        x_val, y_val = to_tensor(val_df, feat_scaler, target_mean, target_std)
        x_test, y_test = to_tensor(test_df, feat_scaler, target_mean, target_std)

        print(f"\n=== {domain_name} (fluid={fluid}) ===", flush=True)
        print(f"  rows: total={n_before} dropped(prop-lookup)={n_dropped} "
              f"train={len(train_df)} val={len(val_df)} test={len(test_df)}", flush=True)

        finetuned = SmallMLP(len(FEATURE_COLS))
        finetuned.load_state_dict(pretrained_state)
        finetuned = train(finetuned, x_tr, y_tr, x_val, y_val, FINETUNE_EPOCHS, LR_FINETUNE,
                           f"{domain_name}-finetuned")
        m_finetuned = evaluate(finetuned, x_test, y_test, target_mean, target_std)
        m_finetuned.update({"domain": domain_name, "fluid": fluid, "model": "pretrained_finetuned"})
        print(f"  pretrained+finetuned: {m_finetuned}", flush=True)

        scratch = SmallMLP(len(FEATURE_COLS))
        scratch = train(scratch, x_tr, y_tr, x_val, y_val, SCRATCH_EPOCHS, LR_SCRATCH,
                         f"{domain_name}-scratch")
        m_scratch = evaluate(scratch, x_test, y_test, target_mean, target_std)
        m_scratch.update({"domain": domain_name, "fluid": fluid, "model": "from_scratch"})
        print(f"  from-scratch:         {m_scratch}", flush=True)

        results.append(m_finetuned)
        results.append(m_scratch)

    results_df = pd.DataFrame(results)[
        ["domain", "fluid", "model", "n", "R2", "rRMSE", "MAPE_%", "within_10pct_%", "RMSE_kW_m2"]]
    results_df.to_csv(os.path.join(OUT_DIR, "stage2_results.csv"), index=False)
    print("\n" + "=" * 90)
    print(results_df.to_string(index=False))


if __name__ == "__main__":
    main()

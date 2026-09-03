"""
Stage 3: deep ensemble of the pretrained+fine-tuned MLP -- N members with
different random seeds (weight init + minibatch shuffling), SAME train/val/
test split per domain (the standard deep-ensemble recipe: vary the model,
not the data, so members disagree because of genuine model uncertainty, not
because they saw different data).

Reuses Stage 2's domain loaders/feature pipeline (scripts/chf_pipeline/
finetune_mlp.py) and Stage 1's interp/extrap splits, so every domain and
every split gets the same ensemble treatment in one place.

Reports, per (domain or split):
  - ensemble-mean R2/rRMSE/MAPE/within-10% (predictions averaged in log
    space across members, then exponentiated -- i.e. the geometric mean of
    each member's CHF prediction)
  - the single-model average +/- std across members, so you can see whether
    ensembling actually beats a typical single run or just matches it
  - 95% interval coverage: fraction of true CHF values falling within
    (ensemble mean +/- 1.96 * ensemble std across members) -- the actual
    calibration check for the uncertainty estimates, not just accuracy
"""
import os
import sys
import pickle
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, "scripts/chf_pipeline")
from models import SmallMLP
from data_prep import FEATURE_COLS, TARGET_COL
from finetune_mlp import (DOMAIN_LOADERS, Standardizer, to_tensor, train,
                           add_dimensionless_features, CKPT_DIR, FINETUNE_EPOCHS, LR_FINETUNE)

N_MEMBERS = 8
OUT_DIR = "data/processed/stage3"
STAGE1_DIR = "data/processed/stage1"
RNG_SEED_DATA = 42  # fixed data split seed, same for every ensemble member


def ensemble_predict(models, x, target_mean, target_std):
    """Return (mean_pred, std_pred) in CHF units (kW/m^2), averaging in log-space."""
    preds_log = []
    for m in models:
        m.eval()
        with torch.no_grad():
            preds_log.append((m(x).numpy() * target_std + target_mean))
    preds_log = np.stack(preds_log, axis=0)  # (N_MEMBERS, n_samples)
    mean_log, std_log = preds_log.mean(axis=0), preds_log.std(axis=0)
    mean_pred = np.exp(mean_log)
    # propagate log-space std to CHF-unit std via delta method: sigma_y ~= y * sigma_logy
    std_pred = mean_pred * std_log
    return mean_pred, std_pred, np.exp(preds_log)  # also return all-members raw preds


def evaluate_ensemble(mean_pred, std_pred, y_log_true, target_mean, target_std):
    true = np.exp(y_log_true.numpy() * target_std + target_mean)
    rel_err = (mean_pred - true) / true
    rmse = float(np.sqrt(np.mean((mean_pred - true) ** 2)))
    rrmse = float(np.sqrt(np.mean(rel_err ** 2)))
    mape = float(np.mean(np.abs(rel_err)) * 100)
    ss_res = np.sum((true - mean_pred) ** 2)
    ss_tot = np.sum((true - true.mean()) ** 2)
    r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan")
    within_10pct = float(np.mean(np.abs(rel_err) <= 0.10) * 100)
    lo, hi = mean_pred - 1.96 * std_pred, mean_pred + 1.96 * std_pred
    coverage_95 = float(np.mean((true >= lo) & (true <= hi)) * 100)
    return {"R2": r2, "rRMSE": rrmse, "MAPE_%": mape, "RMSE_kW_m2": rmse,
            "within_10pct_%": within_10pct, "coverage_95pct_%": coverage_95,
            "n": int(len(true))}


def run_context(name, train_df, val_df, test_df, feat_scaler, target_mean, target_std,
                 pretrained_state):
    x_tr, y_tr = to_tensor(train_df, feat_scaler, target_mean, target_std)
    x_val, y_val = to_tensor(val_df, feat_scaler, target_mean, target_std)
    x_test, y_test = to_tensor(test_df, feat_scaler, target_mean, target_std)

    single_r2 = []
    models = []
    for seed in range(N_MEMBERS):
        torch.manual_seed(1000 + seed)
        model = SmallMLP(len(FEATURE_COLS))
        model.load_state_dict(pretrained_state)
        model = train(model, x_tr, y_tr, x_val, y_val, FINETUNE_EPOCHS, LR_FINETUNE,
                       f"{name}-member{seed}")
        models.append(model)
        m = evaluate_ensemble(*ensemble_predict([model], x_test, target_mean, target_std)[:2],
                               y_test, target_mean, target_std)
        single_r2.append(m["R2"])
        print(f"    [{name}] member {seed}: R2={m['R2']:.4f}", flush=True)

    mean_pred, std_pred, _ = ensemble_predict(models, x_test, target_mean, target_std)
    ens_metrics = evaluate_ensemble(mean_pred, std_pred, y_test, target_mean, target_std)
    ens_metrics.update({
        "context": name,
        "single_model_R2_mean": float(np.mean(single_r2)),
        "single_model_R2_std": float(np.std(single_r2)),
    })
    print(f"  ENSEMBLE [{name}]: {ens_metrics}", flush=True)
    return ens_metrics


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(CKPT_DIR, "scaler.pkl"), "rb") as f:
        scaler_bundle = pickle.load(f)
    feat_scaler = scaler_bundle["feat_scaler"]
    target_mean, target_std = scaler_bundle["target_mean"], scaler_bundle["target_std"]
    pretrained_state = torch.load(os.path.join(CKPT_DIR, "mlp_pretrained.pt"))

    results = []

    # -- Stage 1 core splits (interp, extrap) --
    for split in ["interp", "extrap"]:
        train_df = pd.read_csv(os.path.join(STAGE1_DIR, f"train_{split}.csv"))
        val_df = pd.read_csv(os.path.join(STAGE1_DIR, f"val_{split}.csv"))
        test_df = pd.read_csv(os.path.join(STAGE1_DIR, f"test_{split}.csv"))
        res = run_context(f"core_{split}", train_df, val_df, test_df,
                           feat_scaler, target_mean, target_std, pretrained_state)
        results.append(res)

    # -- Stage 2 fine-tuning domains --
    rng = np.random.default_rng(RNG_SEED_DATA)
    for domain_name, loader in DOMAIN_LOADERS.items():
        raw_df, fluid = loader()
        df = raw_df[(raw_df["CHF_kW_m2"] > 0) & (raw_df["D_mm"] > 0)].reset_index(drop=True)
        df = add_dimensionless_features(df, fluid=fluid, p_col="P_kPa", g_col="G_kg_m2s")
        df = df.dropna(subset=FEATURE_COLS)

        idx = rng.permutation(len(df))
        n_test = max(1, int(len(df) * 0.20))
        n_val = max(1, int(len(df) * 0.15))
        test_df = df.iloc[idx[:n_test]]
        val_df = df.iloc[idx[n_test:n_test + n_val]]
        train_df = df.iloc[idx[n_test + n_val:]]

        print(f"\n=== {domain_name} (fluid={fluid}) ===", flush=True)
        res = run_context(domain_name, train_df, val_df, test_df,
                           feat_scaler, target_mean, target_std, pretrained_state)
        res["fluid"] = fluid
        results.append(res)

    results_df = pd.DataFrame(results)
    cols = ["context", "fluid", "n", "R2", "single_model_R2_mean", "single_model_R2_std",
            "rRMSE", "MAPE_%", "within_10pct_%", "coverage_95pct_%"]
    cols = [c for c in cols if c in results_df.columns]
    results_df = results_df[cols]
    results_df.to_csv(os.path.join(OUT_DIR, "stage3_ensemble_results.csv"), index=False)
    print("\n" + "=" * 100)
    print(results_df.to_string(index=False))


if __name__ == "__main__":
    main()

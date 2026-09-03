"""
Stage 3 for the Transformer -- same deep-ensemble recipe as ensemble_mlp.py
(fixed data split, varied seed, log-space averaging, 95% coverage check),
but N_MEMBERS is reduced to 4 (vs. the MLP's 8): each Transformer fine-tune
costs roughly 30-40x a single MLP fine-tune on this CPU-only setup (see
Stage 1's timing: 1949s vs 65s on the interp split alone), so an 8-member
Transformer ensemble across all 7 contexts would take many hours. 4 members
still gives a genuine mean/std read on ensemble behavior at a tractable
cost -- documented here as a deliberate scope reduction, not a hidden
shortcut.
"""
import os
import sys
import pickle
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, "scripts/chf_pipeline")
from models import FTTransformer
from data_prep import FEATURE_COLS
from finetune_mlp import DOMAIN_LOADERS, to_tensor, add_dimensionless_features, CKPT_DIR
from finetune_transformer import train, FINETUNE_EPOCHS, LR_FINETUNE
from ensemble_mlp import ensemble_predict, evaluate_ensemble


class Standardizer:
    """Unused directly here -- but pickle needs __main__.Standardizer to
    resolve scaler.pkl, which was saved while pretrain.py was __main__."""
    def transform(self, x):
        return (x - self.mean) / self.std

N_MEMBERS = 4
OUT_DIR = "data/processed/stage3"
STAGE1_DIR = "data/processed/stage1"
RNG_SEED_DATA = 42


def run_context(name, train_df, val_df, test_df, feat_scaler, target_mean, target_std,
                 pretrained_state):
    x_tr, y_tr = to_tensor(train_df, feat_scaler, target_mean, target_std)
    x_val, y_val = to_tensor(val_df, feat_scaler, target_mean, target_std)
    x_test, y_test = to_tensor(test_df, feat_scaler, target_mean, target_std)

    single_r2 = []
    models = []
    for seed in range(N_MEMBERS):
        torch.manual_seed(2000 + seed)
        model = FTTransformer(len(FEATURE_COLS))
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
    pretrained_state = torch.load(os.path.join(CKPT_DIR, "transformer_pretrained.pt"))

    results = []

    for split in ["interp", "extrap"]:
        train_df = pd.read_csv(os.path.join(STAGE1_DIR, f"train_{split}.csv"))
        val_df = pd.read_csv(os.path.join(STAGE1_DIR, f"val_{split}.csv"))
        test_df = pd.read_csv(os.path.join(STAGE1_DIR, f"test_{split}.csv"))
        res = run_context(f"core_{split}", train_df, val_df, test_df,
                           feat_scaler, target_mean, target_std, pretrained_state)
        results.append(res)

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
    results_df.to_csv(os.path.join(OUT_DIR, "stage3_transformer_ensemble_results.csv"), index=False)
    print("\n" + "=" * 100)
    print(results_df.to_string(index=False))


if __name__ == "__main__":
    main()

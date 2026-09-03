"""
Stage 2 for the Transformer architecture -- mirrors finetune_mlp.py exactly
(same domains, same splits, same metrics, same from-scratch comparison) but
loads FTTransformer + transformer_pretrained.pt instead of SmallMLP. Reuses
finetune_mlp.py's domain loaders/feature pipeline/eval function directly so
the comparison is genuinely apples-to-apples, not two divergent pipelines.
"""
import os
import sys
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

sys.path.insert(0, "scripts/chf_pipeline")
from models import FTTransformer
from data_prep import FEATURE_COLS, TARGET_COL
from finetune_mlp import (DOMAIN_LOADERS, to_tensor, evaluate, add_dimensionless_features,
                           CKPT_DIR, TEST_FRAC)


class Standardizer:
    """Unused directly here -- but pickle needs __main__.Standardizer to
    resolve scaler.pkl, which was saved while pretrain.py was __main__."""
    def transform(self, x):
        return (x - self.mean) / self.std

OUT_DIR = "data/processed/stage2"
RNG_SEED = 42
FINETUNE_EPOCHS = 100   # transformer converges slower; give it more room than MLP's 150-epoch
                         # budget would suggest per-step, but cap below since patience will
                         # bail out early once it plateaus
SCRATCH_EPOCHS = 100
PATIENCE = 15
LR_FINETUNE = 5e-5      # smaller than MLP's 1e-4 -- transformer pretraining showed more
                         # sensitivity to LR in Stage 1
LR_SCRATCH = 1e-3


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
            print(f"    [{label}] early stop at epoch {epoch+1} (best val={best_val:.4f})", flush=True)
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
    pretrained_state = torch.load(os.path.join(CKPT_DIR, "transformer_pretrained.pt"))

    results = []
    for domain_name, loader in DOMAIN_LOADERS.items():
        raw_df, fluid = loader()
        df = raw_df[(raw_df["CHF_kW_m2"] > 0) & (raw_df["D_mm"] > 0)].reset_index(drop=True)
        df = add_dimensionless_features(df, fluid=fluid, p_col="P_kPa", g_col="G_kg_m2s")
        n_before = len(df)
        df = df.dropna(subset=FEATURE_COLS)

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
        print(f"  rows: total={n_before} train={len(train_df)} val={len(val_df)} test={len(test_df)}", flush=True)

        finetuned = FTTransformer(len(FEATURE_COLS))
        finetuned.load_state_dict(pretrained_state)
        finetuned = train(finetuned, x_tr, y_tr, x_val, y_val, FINETUNE_EPOCHS, LR_FINETUNE,
                           f"{domain_name}-finetuned")
        m_finetuned = evaluate(finetuned, x_test, y_test, target_mean, target_std)
        m_finetuned.update({"domain": domain_name, "fluid": fluid, "model": "pretrained_finetuned"})
        print(f"  pretrained+finetuned: {m_finetuned}", flush=True)

        scratch = FTTransformer(len(FEATURE_COLS))
        scratch = train(scratch, x_tr, y_tr, x_val, y_val, SCRATCH_EPOCHS, LR_SCRATCH,
                         f"{domain_name}-scratch")
        m_scratch = evaluate(scratch, x_test, y_test, target_mean, target_std)
        m_scratch.update({"domain": domain_name, "fluid": fluid, "model": "from_scratch"})
        print(f"  from-scratch:         {m_scratch}", flush=True)

        results.append(m_finetuned)
        results.append(m_scratch)

    results_df = pd.DataFrame(results)[
        ["domain", "fluid", "model", "n", "R2", "rRMSE", "MAPE_%", "within_10pct_%", "RMSE_kW_m2"]]
    results_df.to_csv(os.path.join(OUT_DIR, "stage2_transformer_results.csv"), index=False)
    print("\n" + "=" * 90)
    print(results_df.to_string(index=False))


if __name__ == "__main__":
    main()

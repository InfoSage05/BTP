"""
Regenerates real MLP predictions on the Stage-1 interpolation and extrapolation
test sets, for the parity plot used in the paper's Results section. This runs
the actual SmallMLP architecture (scripts/chf_pipeline/models.py) through the
same pretrain-then-fine-tune procedure as scripts/chf_pipeline/train_stage1.py
(same seed, same data, same hyperparameters), but additionally saves the
per-row true vs. predicted CHF values, which train_stage1.py itself does not
persist. Transformer is intentionally skipped here (its own training run takes
roughly 30x longer per split) -- the MLP is this project's own fast, still
strong architecture and is sufficient to produce a genuine, non-fabricated
parity plot.

Output: data/processed/stage1/mlp_predictions_interp.csv
        data/processed/stage1/mlp_predictions_extrap.csv
"""
import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

sys.path.insert(0, "scripts/chf_pipeline")
from models import SmallMLP
from data_prep import FEATURE_COLS, TARGET_COL, OUT_DIR

torch.manual_seed(42)
DEVICE = "cpu"
MONO_WEIGHT = 0.01
PRETRAIN_EPOCHS = 6
FINETUNE_EPOCHS = 30
PATIENCE = 8
BATCH_SIZE = 512


class Standardizer:
    def fit(self, x):
        self.mean = x.mean(axis=0)
        self.std = x.std(axis=0)
        self.std[self.std == 0] = 1.0
        return self

    def transform(self, x):
        return (x - self.mean) / self.std


def to_tensor(df, feat_scaler, target_mean=None, target_std=None):
    x = feat_scaler.transform(df[FEATURE_COLS].to_numpy(dtype=np.float32))
    y_log = np.log(df[TARGET_COL].to_numpy(dtype=np.float32))
    if target_mean is not None:
        y_log = (y_log - target_mean) / target_std
    return torch.tensor(x, dtype=torch.float32).to(DEVICE), torch.tensor(y_log, dtype=torch.float32).to(DEVICE)


def monotonicity_penalty(model, x_batch, x_idx_of_X):
    x_batch = x_batch.clone().requires_grad_(True)
    pred = model(x_batch)
    grad = torch.autograd.grad(pred.sum(), x_batch, create_graph=True)[0]
    dpred_dX = grad[:, x_idx_of_X]
    return torch.relu(dpred_dX).mean()


def run_epoch(model, x, y, optimizer, x_idx_of_X, train=True, use_mono=True):
    model.train(train)
    n = len(x)
    perm = torch.randperm(n) if train else torch.arange(n)
    total_loss = 0.0
    mse = nn.MSELoss()
    for i in range(0, n, BATCH_SIZE):
        idx = perm[i:i + BATCH_SIZE]
        xb, yb = x[idx], y[idx]
        if train:
            optimizer.zero_grad()
        pred = model(xb)
        loss = mse(pred, yb)
        if train and MONO_WEIGHT > 0 and use_mono:
            loss = loss + MONO_WEIGHT * monotonicity_penalty(model, xb, x_idx_of_X)
        if train:
            loss.backward()
            optimizer.step()
        total_loss += loss.item() * len(idx)
    return total_loss / n


def train_with_early_stopping(model, x_tr, y_tr, x_val, y_val, x_idx_of_X,
                                epochs, lr=1e-3, label="", use_mono=True):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    best_val, best_state, bad_epochs = float("inf"), None, 0
    for epoch in range(epochs):
        train_loss = run_epoch(model, x_tr, y_tr, optimizer, x_idx_of_X, train=True, use_mono=use_mono)
        with torch.no_grad():
            val_loss = nn.functional.mse_loss(model(x_val), y_val).item()
        if val_loss < best_val - 1e-5:
            best_val, best_state, bad_epochs = val_loss, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            bad_epochs += 1
        if bad_epochs >= PATIENCE:
            print(f"    [{label}] early stop at epoch {epoch+1} (best val_loss={best_val:.4f})")
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def main():
    synthetic = pd.read_csv(os.path.join(OUT_DIR, "synthetic_pretrain.csv"))
    x_idx_of_X = FEATURE_COLS.index("X")

    feat_scaler = Standardizer().fit(synthetic[FEATURE_COLS].to_numpy(dtype=np.float32))
    y_log_synth = np.log(synthetic[TARGET_COL].to_numpy(dtype=np.float32))
    target_mean, target_std = float(y_log_synth.mean()), float(y_log_synth.std())
    x_synth, y_synth = to_tensor(synthetic, feat_scaler, target_mean, target_std)

    for split_name in ["interp", "extrap"]:
        train = pd.read_csv(os.path.join(OUT_DIR, f"train_{split_name}.csv"))
        val = pd.read_csv(os.path.join(OUT_DIR, f"val_{split_name}.csv"))
        test = pd.read_csv(os.path.join(OUT_DIR, f"test_{split_name}.csv"))

        x_tr, y_tr = to_tensor(train, feat_scaler, target_mean, target_std)
        x_val, y_val = to_tensor(val, feat_scaler, target_mean, target_std)
        x_test, y_test = to_tensor(test, feat_scaler, target_mean, target_std)

        print(f"\n=== mlp | {split_name} split ===")
        model = SmallMLP(len(FEATURE_COLS)).to(DEVICE)

        print(f"  pretraining on {len(x_synth)} synthetic rows...")
        model = train_with_early_stopping(
            model, x_synth, y_synth, x_val, y_val, x_idx_of_X,
            epochs=PRETRAIN_EPOCHS, lr=2e-3, label=f"mlp-{split_name}-pretrain",
            use_mono=False)

        print(f"  fine-tuning on {len(x_tr)} real rows...")
        model = train_with_early_stopping(
            model, x_tr, y_tr, x_val, y_val, x_idx_of_X,
            epochs=FINETUNE_EPOCHS, lr=5e-4, label=f"mlp-{split_name}-finetune")

        model.eval()
        with torch.no_grad():
            pred_log = model(x_test).numpy() * target_std + target_mean
        true_log = y_test.numpy() * target_std + target_mean
        pred, true = np.exp(pred_log), np.exp(true_log)

        rel_err = (pred - true) / true
        rmse = np.sqrt(np.mean((pred - true) ** 2))
        mape = np.mean(np.abs(rel_err)) * 100
        ss_res = np.sum((true - pred) ** 2)
        ss_tot = np.sum((true - true.mean()) ** 2)
        r2 = 1 - ss_res / ss_tot
        within_10 = np.mean(np.abs(rel_err) <= 0.10) * 100
        print(f"  test: R2={r2:.4f} RMSE={rmse:.1f} MAPE={mape:.2f}% within10%={within_10:.1f}%")

        out = pd.DataFrame({"true_CHF_kW_m2": true, "pred_CHF_kW_m2": pred})
        out_path = os.path.join(OUT_DIR, f"mlp_predictions_{split_name}.csv")
        out.to_csv(out_path, index=False)
        print(f"  saved {out_path}")


if __name__ == "__main__":
    main()

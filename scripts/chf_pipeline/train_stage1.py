"""
Stage 1: pretrain both candidate architectures (SmallMLP, FTTransformer) on
the synthetic LUT corpus, then fine-tune + evaluate each on BOTH the
interpolation split and the harder extrapolation split (pressure > 8 MPa
held out entirely from training). Reports R^2 / rRMSE / MAPE for every
(architecture x split) combination side by side -- this comparison table is
the actual Stage-1 deliverable, not a single chosen "winner" picked a priori.

Target is trained in log-space (log(CHF)) since raw CHF spans ~1-50,000
kW/m^2; metrics are reported after transforming back to the original scale,
matching the definitions used in the literature (Yang et al. 2025, Eq 11-14).

A light monotonicity penalty (CHF should not visibly increase with quality
X in the bulk of the domain) is added to the training loss via autograd,
weighted low enough not to dominate the data-fit loss -- included because
the literature review this project did found it's exactly what curbed
unphysical extrapolation drift in the strongest 2025 results.
"""
import os
import sys
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

sys.path.insert(0, "scripts/chf_pipeline")
from models import SmallMLP, FTTransformer
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
    return torch.tensor(x, dtype=torch.float32), torch.tensor(y_log, dtype=torch.float32)


def monotonicity_penalty(model, x_batch, x_idx_of_X):
    x_batch = x_batch.clone().requires_grad_(True)
    pred = model(x_batch)
    grad = torch.autograd.grad(pred.sum(), x_batch, create_graph=True)[0]
    dpred_dX = grad[:, x_idx_of_X]
    return torch.relu(dpred_dX).mean()  # penalize positive slope only


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
        print(f'    [{label}] epoch {epoch+1}/{epochs} train_loss={train_loss:.4f}', flush=True)
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


def evaluate(model, x, y_log, target_mean, target_std):
    model.eval()
    with torch.no_grad():
        pred_log = model(x).numpy() * target_std + target_mean
    true_log = y_log.numpy() * target_std + target_mean
    pred, true = np.exp(pred_log), np.exp(true_log)

    rel_err = (pred - true) / true
    rmse = np.sqrt(np.mean((pred - true) ** 2))
    rrmse = np.sqrt(np.mean(rel_err ** 2))
    mape = np.mean(np.abs(rel_err)) * 100
    ss_res = np.sum((true - pred) ** 2)
    ss_tot = np.sum((true - true.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot
    within_10pct = np.mean(np.abs(rel_err) <= 0.10) * 100
    return {"R2": r2, "rRMSE": rrmse, "MAPE_%": mape, "RMSE_kW_m2": rmse,
            "within_10pct_%": within_10pct, "n": len(true)}


def build_model(arch: str, n_features: int):
    if arch == "mlp":
        return SmallMLP(n_features)
    if arch == "transformer":
        return FTTransformer(n_features)
    raise ValueError(arch)


def main():
    synthetic = pd.read_csv(os.path.join(OUT_DIR, "synthetic_pretrain.csv"))
    x_idx_of_X = FEATURE_COLS.index("X")

    feat_scaler = Standardizer().fit(synthetic[FEATURE_COLS].to_numpy(dtype=np.float32))
    y_log_synth = np.log(synthetic[TARGET_COL].to_numpy(dtype=np.float32))
    target_mean, target_std = float(y_log_synth.mean()), float(y_log_synth.std())

    x_synth, y_synth = to_tensor(synthetic, feat_scaler, target_mean, target_std)

    results = []
    for split_name in ["interp", "extrap"]:
        train = pd.read_csv(os.path.join(OUT_DIR, f"train_{split_name}.csv"))
        val = pd.read_csv(os.path.join(OUT_DIR, f"val_{split_name}.csv"))
        test = pd.read_csv(os.path.join(OUT_DIR, f"test_{split_name}.csv"))

        x_tr, y_tr = to_tensor(train, feat_scaler, target_mean, target_std)
        x_val, y_val = to_tensor(val, feat_scaler, target_mean, target_std)
        x_test, y_test = to_tensor(test, feat_scaler, target_mean, target_std)

        for arch in ["mlp", "transformer"]:
            t0 = time.time()
            print(f"\n=== {arch} | {split_name} split ===")
            model = build_model(arch, len(FEATURE_COLS))

            print(f"  pretraining on {len(x_synth)} synthetic rows...")
            model = train_with_early_stopping(
                model, x_synth, y_synth, x_val, y_val, x_idx_of_X,
                epochs=PRETRAIN_EPOCHS, lr=2e-3, label=f"{arch}-{split_name}-pretrain",
                use_mono=False)

            print(f"  fine-tuning on {len(x_tr)} real rows...")
            model = train_with_early_stopping(
                model, x_tr, y_tr, x_val, y_val, x_idx_of_X,
                epochs=FINETUNE_EPOCHS, lr=5e-4, label=f"{arch}-{split_name}-finetune")

            metrics = evaluate(model, x_test, y_test, target_mean, target_std)
            metrics.update({"arch": arch, "split": split_name, "seconds": round(time.time() - t0, 1)})
            print(f"  test metrics: {metrics}")
            results.append(metrics)

    results_df = pd.DataFrame(results)[
        ["arch", "split", "n", "R2", "rRMSE", "MAPE_%", "within_10pct_%", "RMSE_kW_m2", "seconds"]]
    os.makedirs(OUT_DIR, exist_ok=True)
    results_df.to_csv(os.path.join(OUT_DIR, "stage1_results.csv"), index=False)
    print("\n" + "=" * 70)
    print(results_df.to_string(index=False))


if __name__ == "__main__":
    main()

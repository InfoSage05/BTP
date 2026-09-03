"""
Pretrain each architecture ONCE on the synthetic corpus, save a checkpoint.
Stage 2 (and the interp/extrap fine-tuning eval) load these checkpoints
instead of re-pretraining from scratch per split -- pretraining is
architecture-specific, not split-specific, so redoing it per split (as the
first Stage 1 pass did) was pure waste.

Epoch budget is now per-architecture, not shared: the Transformer converges
slower than the small MLP (this matches the convergence curves in Yang et
al. 2025's own paper), so giving both the same tiny epoch count structurally
penalized the Transformer. Each gets enough epochs to actually plateau,
judged by early stopping against the same held-out validation slice, not a
fixed guess.
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
CKPT_DIR = os.path.join(OUT_DIR, "checkpoints")

# per-architecture epoch budgets -- Transformer gets far more, since it needs
# it, not because we assume it deserves it (early stopping still applies)
EPOCH_BUDGET = {"mlp": 20, "transformer": 80}
PATIENCE = 10
BATCH_SIZE = 512


class Standardizer:
    def fit(self, x):
        self.mean = x.mean(axis=0)
        self.std = x.std(axis=0)
        self.std[self.std == 0] = 1.0
        return self

    def transform(self, x):
        return (x - self.mean) / self.std


def to_tensor(df, feat_scaler, target_mean, target_std):
    x = feat_scaler.transform(df[FEATURE_COLS].to_numpy(dtype=np.float32))
    y_log = np.log(df[TARGET_COL].to_numpy(dtype=np.float32))
    y_log = (y_log - target_mean) / target_std
    return torch.tensor(x, dtype=torch.float32), torch.tensor(y_log, dtype=torch.float32)


def build_model(arch, n_features):
    if arch == "mlp":
        return SmallMLP(n_features)
    if arch == "transformer":
        return FTTransformer(n_features)
    raise ValueError(arch)


def pretrain_one(arch, x_synth, y_synth, x_val, y_val, n_features):
    model = build_model(arch, n_features)
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-3)
    mse = nn.MSELoss()
    best_val, best_state, bad_epochs = float("inf"), None, 0
    epochs = EPOCH_BUDGET[arch]

    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(len(x_synth))
        total = 0.0
        for i in range(0, len(x_synth), BATCH_SIZE):
            idx = perm[i:i + BATCH_SIZE]
            optimizer.zero_grad()
            pred = model(x_synth[idx])
            loss = mse(pred, y_synth[idx])
            loss.backward()
            optimizer.step()
            total += loss.item() * len(idx)
        train_loss = total / len(x_synth)

        model.eval()
        with torch.no_grad():
            val_loss = mse(model(x_val), y_val).item()
        print(f"    [{arch}-pretrain] epoch {epoch+1}/{epochs} "
              f"train_loss={train_loss:.4f} val_loss={val_loss:.4f}", flush=True)

        if val_loss < best_val - 1e-5:
            best_val = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
        if bad_epochs >= PATIENCE:
            print(f"    [{arch}-pretrain] early stop at epoch {epoch+1} (best val_loss={best_val:.4f})")
            break

    model.load_state_dict(best_state)
    return model, best_val


def main():
    os.makedirs(CKPT_DIR, exist_ok=True)
    synthetic = pd.read_csv(os.path.join(OUT_DIR, "synthetic_pretrain.csv"))
    feat_scaler = Standardizer().fit(synthetic[FEATURE_COLS].to_numpy(dtype=np.float32))
    y_log_synth = np.log(synthetic[TARGET_COL].to_numpy(dtype=np.float32))
    target_mean, target_std = float(y_log_synth.mean()), float(y_log_synth.std())

    # use the interp split's val set as the pretraining validation signal
    # (any real held-out slice works here; pretraining never trains on it)
    val_df = pd.read_csv(os.path.join(OUT_DIR, "val_interp.csv"))
    x_val, y_val = to_tensor(val_df, feat_scaler, target_mean, target_std)
    x_synth, y_synth = to_tensor(synthetic, feat_scaler, target_mean, target_std)

    import pickle
    with open(os.path.join(CKPT_DIR, "scaler.pkl"), "wb") as f:
        pickle.dump({"feat_scaler": feat_scaler, "target_mean": target_mean,
                     "target_std": target_std}, f)

    for arch in sys.argv[1:] or ["mlp", "transformer"]:
        t0 = time.time()
        print(f"=== pretraining {arch} on {len(x_synth)} synthetic rows ===", flush=True)
        model, best_val = pretrain_one(arch, x_synth, y_synth, x_val, y_val, len(FEATURE_COLS))
        ckpt_path = os.path.join(CKPT_DIR, f"{arch}_pretrained.pt")
        torch.save(model.state_dict(), ckpt_path)
        print(f"Saved {ckpt_path} (best_val_loss={best_val:.4f}, {time.time()-t0:.1f}s)\n", flush=True)


if __name__ == "__main__":
    main()

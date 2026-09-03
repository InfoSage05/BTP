"""
Fine-tune on all 5 real fine-tuning domains POOLED into one larger set,
instead of 5 separate per-domain fine-tunes. Reuses Stage 2's domain
loaders/feature pipeline. A 'domain' column is kept through the split so we
can report both the pooled test-set metrics AND a per-domain breakdown of
that same pooled-trained model -- i.e. does training on everything at once
(more data) beat training separately per domain (Stage 2's approach)?

Split is stratified by domain (each domain contributes its own 65/15/20
train/val/test slice before pooling) so a small domain like hardik2017
(55 rows) can't end up entirely in one split by bad luck.

Runs both architectures (each loading its own Stage-1 pretrained
checkpoint), so this is directly comparable to Stage 2's per-domain results.
"""
import os
import sys
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

sys.path.insert(0, "scripts/chf_pipeline")
from models import SmallMLP, FTTransformer
from data_prep import FEATURE_COLS, TARGET_COL
from finetune_mlp import DOMAIN_LOADERS, to_tensor, evaluate, add_dimensionless_features, CKPT_DIR

OUT_DIR = "data/processed/stage2_pooled"
RNG_SEED = 42
EPOCH_BUDGET = {"mlp": 200, "transformer": 120}
LR = {"mlp": 1e-4, "transformer": 5e-5}
PATIENCE = 20


class Standardizer:
    """Unused directly -- pickle needs __main__.Standardizer to resolve scaler.pkl."""
    def transform(self, x):
        return (x - self.mean) / self.std


def build_pooled_dataset(rng):
    frames = []
    for domain_name, loader in DOMAIN_LOADERS.items():
        raw_df, fluid = loader()
        df = raw_df[(raw_df["CHF_kW_m2"] > 0) & (raw_df["D_mm"] > 0)].reset_index(drop=True)
        df = add_dimensionless_features(df, fluid=fluid, p_col="P_kPa", g_col="G_kg_m2s")
        df = df.dropna(subset=FEATURE_COLS)
        df["domain"] = domain_name
        df["fluid"] = fluid

        idx = rng.permutation(len(df))
        n_test = max(1, int(len(df) * 0.20))
        n_val = max(1, int(len(df) * 0.15))
        df = df.assign(_split=np.where(
            np.isin(np.arange(len(df)), idx[:n_test]), "test",
            np.where(np.isin(np.arange(len(df)), idx[n_test:n_test + n_val]), "val", "train")))
        frames.append(df)
        print(f"  {domain_name} ({fluid}): {len(df)} rows -> "
              f"train={sum(df._split=='train')} val={sum(df._split=='val')} test={sum(df._split=='test')}")
    return pd.concat(frames, ignore_index=True)


def build_model(arch, n_features):
    return SmallMLP(n_features) if arch == "mlp" else FTTransformer(n_features)


def train(model, x_tr, y_tr, x_val, y_val, epochs, lr, label):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    mse = nn.MSELoss()
    best_val, best_state, bad = float("inf"), None, 0
    n = len(x_tr)
    batch_size = min(128, max(8, n // 8))
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
        if epoch % 10 == 0 or bad >= PATIENCE:
            print(f"    [{label}] epoch {epoch+1}/{epochs} val_loss={val_loss:.4f}", flush=True)
        if bad >= PATIENCE:
            print(f"    [{label}] early stop at epoch {epoch+1} (best val={best_val:.4f})", flush=True)
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    rng = np.random.default_rng(RNG_SEED)

    print("Building pooled dataset:")
    pooled = build_pooled_dataset(rng)
    print(f"\nTotal pooled: {len(pooled)} rows "
          f"(train={sum(pooled._split=='train')} val={sum(pooled._split=='val')} test={sum(pooled._split=='test')})")

    with open(os.path.join(CKPT_DIR, "scaler.pkl"), "rb") as f:
        scaler_bundle = pickle.load(f)
    feat_scaler = scaler_bundle["feat_scaler"]
    target_mean, target_std = scaler_bundle["target_mean"], scaler_bundle["target_std"]

    train_df = pooled[pooled._split == "train"]
    val_df = pooled[pooled._split == "val"]
    test_df = pooled[pooled._split == "test"]

    x_tr, y_tr = to_tensor(train_df, feat_scaler, target_mean, target_std)
    x_val, y_val = to_tensor(val_df, feat_scaler, target_mean, target_std)
    x_test, y_test = to_tensor(test_df, feat_scaler, target_mean, target_std)

    all_results = []
    for arch in ["mlp", "transformer"]:
        print(f"\n=== pooled fine-tune: {arch} ===", flush=True)
        pretrained_state = torch.load(os.path.join(CKPT_DIR, f"{arch}_pretrained.pt"))
        model = build_model(arch, len(FEATURE_COLS))
        model.load_state_dict(pretrained_state)
        model = train(model, x_tr, y_tr, x_val, y_val, EPOCH_BUDGET[arch], LR[arch], f"pooled-{arch}")

        overall = evaluate(model, x_test, y_test, target_mean, target_std)
        overall.update({"arch": arch, "domain": "ALL_POOLED", "n": len(test_df)})
        print(f"  POOLED TEST: {overall}", flush=True)
        all_results.append(overall)

        for domain_name in DOMAIN_LOADERS:
            sub = test_df[test_df["domain"] == domain_name]
            if len(sub) == 0:
                continue
            x_sub, y_sub = to_tensor(sub, feat_scaler, target_mean, target_std)
            m = evaluate(model, x_sub, y_sub, target_mean, target_std)
            m.update({"arch": arch, "domain": domain_name})
            print(f"    per-domain [{domain_name}]: {m}", flush=True)
            all_results.append(m)

    results_df = pd.DataFrame(all_results)[
        ["arch", "domain", "n", "R2", "rRMSE", "MAPE_%", "within_10pct_%", "RMSE_kW_m2"]]
    results_df.to_csv(os.path.join(OUT_DIR, "pooled_finetune_results.csv"), index=False)
    print("\n" + "=" * 90)
    print(results_df.to_string(index=False))


if __name__ == "__main__":
    main()

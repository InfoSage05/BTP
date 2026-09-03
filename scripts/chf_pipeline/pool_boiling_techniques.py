import os
import sys
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

sys.path.insert(0, "scripts/chf_pipeline")
from models import SmallMLP, FTTransformer, LoRAMLP, LoRAFTTransformer, MoEModel
from data_prep import FEATURE_COLS, TARGET_COL

CKPT_DIR = "data/processed/stage1/checkpoints"
STAGE1_DIR = "data/processed/stage1"
POOL_DATA = "data/processed/pool_boiling/strip_pool_boiling_water.csv"
OUT_DIR = "data/processed/pool_boiling"
RNG_SEED = 42
TEST_FRAC = 0.20
EPOCHS = {"mlp": 200, "transformer": 150}
PATIENCE = 20
LR = {"from_scratch": {"mlp": 2e-3, "transformer": 1e-3},
      "full_finetune": {"mlp": 1e-4, "transformer": 5e-5},
      "lora": {"mlp": 5e-3, "transformer": 2e-3},
      "moe": {"mlp": 3e-3, "transformer": 3e-3}}


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


def train_loop(model, params, x_tr, y_tr, x_val, y_val, epochs, lr, label):
    optimizer = torch.optim.Adam(params, lr=lr)
    mse = nn.MSELoss()
    best_val, best_state, bad = float("inf"), None, 0
    n = len(x_tr)
    batch_size = min(32, max(4, n // 4))
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
            print("    [" + label + "] early stop at epoch " + str(epoch+1) + " (best val=" + str(round(best_val,4)) + ")", flush=True)
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def build_base(arch, n_features):
    return SmallMLP(n_features) if arch == "mlp" else FTTransformer(n_features)


def main():
    rng = np.random.default_rng(RNG_SEED)
    pool_df = pd.read_csv(POOL_DATA)
    idx = rng.permutation(len(pool_df))
    n_test = max(1, int(len(pool_df) * TEST_FRAC))
    n_val = max(1, int(len(pool_df) * 0.15))
    test_df = pool_df.iloc[idx[:n_test]]
    val_df = pool_df.iloc[idx[n_test:n_test + n_val]]
    train_df = pool_df.iloc[idx[n_test + n_val:]]
    print("Pool split: train=" + str(len(train_df)) + " val=" + str(len(val_df)) + " test=" + str(len(test_df)))

    with open(os.path.join(CKPT_DIR, "scaler.pkl"), "rb") as f:
        scaler_bundle = pickle.load(f)
    feat_scaler = scaler_bundle["feat_scaler"]
    target_mean, target_std = scaler_bundle["target_mean"], scaler_bundle["target_std"]

    x_tr, y_tr = to_tensor(train_df, feat_scaler, target_mean, target_std)
    x_val, y_val = to_tensor(val_df, feat_scaler, target_mean, target_std)
    x_test, y_test = to_tensor(test_df, feat_scaler, target_mean, target_std)

    flow_test = pd.read_csv(os.path.join(STAGE1_DIR, "test_interp.csv"))
    flow_sample = flow_test.sample(n=min(len(flow_test), len(test_df)), random_state=RNG_SEED)
    mixed_df = pd.concat([test_df.assign(regime="pool"), flow_sample.assign(regime="flow")], ignore_index=True)
    x_mixed, y_mixed = to_tensor(mixed_df, feat_scaler, target_mean, target_std)
    print("Mixed MoE test set: " + str(len(mixed_df)) + " rows")

    results = []
    for arch in ["mlp", "transformer"]:
        n_feat = len(FEATURE_COLS)
        pretrained_state = torch.load(os.path.join(CKPT_DIR, arch + "_pretrained.pt"))

        print("\n=== " + arch + ": from_scratch ===", flush=True)
        m = build_base(arch, n_feat)
        m = train_loop(m, list(m.parameters()), x_tr, y_tr, x_val, y_val,
                        EPOCHS[arch], LR["from_scratch"][arch], arch + "-scratch")
        r = evaluate(m, x_test, y_test, target_mean, target_std)
        r.update({"arch": arch, "technique": "from_scratch", "test_set": "pool_only"})
        print("  " + str(r), flush=True)
        results.append(r)

        print("\n=== " + arch + ": full_finetune ===", flush=True)
        m = build_base(arch, n_feat)
        m.load_state_dict(pretrained_state)
        m = train_loop(m, list(m.parameters()), x_tr, y_tr, x_val, y_val,
                        EPOCHS[arch], LR["full_finetune"][arch], arch + "-fullft")
        r = evaluate(m, x_test, y_test, target_mean, target_std)
        r.update({"arch": arch, "technique": "full_finetune", "test_set": "pool_only"})
        print("  " + str(r), flush=True)
        results.append(r)

        print("\n=== " + arch + ": lora ===", flush=True)
        base = build_base(arch, n_feat)
        base.load_state_dict(pretrained_state)
        lora_m = LoRAMLP(base, rank=4) if arch == "mlp" else LoRAFTTransformer(base, rank=4)
        n_lora = sum(p.numel() for p in lora_m.lora_parameters())
        n_base = sum(p.numel() for p in base.parameters())
        print("    LoRA trainable params: " + str(n_lora) + " vs base: " + str(n_base), flush=True)
        lora_m = train_loop(lora_m, list(lora_m.lora_parameters()), x_tr, y_tr, x_val, y_val,
                             EPOCHS[arch], LR["lora"][arch], arch + "-lora")
        r = evaluate(lora_m, x_test, y_test, target_mean, target_std)
        r.update({"arch": arch, "technique": "lora", "test_set": "pool_only",
                   "trainable_param_pct": round(100 * n_lora / n_base, 2)})
        print("  " + str(r), flush=True)
        results.append(r)
        print("\n=== " + arch + ": moe ===", flush=True)
        flow_expert = build_base(arch, n_feat)
        flow_expert.load_state_dict(pretrained_state)
        pool_expert = build_base(arch, n_feat)
        moe = MoEModel(flow_expert, pool_expert, n_feat)
        trainable = [p for p in moe.parameters() if p.requires_grad]
        moe = train_loop(moe, trainable, x_tr, y_tr, x_val, y_val,
                          EPOCHS[arch], LR["moe"][arch], arch + "-moe")

        r_pool = evaluate(moe, x_test, y_test, target_mean, target_std)
        r_pool.update({"arch": arch, "technique": "moe", "test_set": "pool_only"})
        print("  MoE pool-only: " + str(r_pool), flush=True)
        results.append(r_pool)

        r_mixed = evaluate(moe, x_mixed, y_mixed, target_mean, target_std)
        r_mixed.update({"arch": arch, "technique": "moe", "test_set": "mixed_pool_flow"})
        print("  MoE MIXED: " + str(r_mixed), flush=True)
        results.append(r_mixed)

        with torch.no_grad():
            gate_weights = moe.gate(x_mixed).numpy()
        mixed_local = mixed_df.reset_index(drop=True)
        pool_gate = gate_weights[mixed_local.regime == "pool", 1].mean()
        flow_gate = gate_weights[mixed_local.regime == "flow", 0].mean()
        print("  MoE gate check: avg P(pool_expert)|pool_row=" + str(round(float(pool_gate),3)) +
              " avg P(flow_expert)|flow_row=" + str(round(float(flow_gate),3)), flush=True)
        results.append({"arch": arch, "technique": "moe_gate_check", "test_set": "mixed_pool_flow",
                         "R2": float("nan"), "avg_pool_gate_on_pool_rows": round(float(pool_gate), 4),
                         "avg_flow_gate_on_flow_rows": round(float(flow_gate), 4), "n": len(mixed_df)})

    results_df = pd.DataFrame(results)
    cols = ["arch", "technique", "test_set", "n", "R2", "rRMSE", "MAPE_%", "within_10pct_%",
            "RMSE_kW_m2", "trainable_param_pct", "avg_pool_gate_on_pool_rows", "avg_flow_gate_on_flow_rows"]
    cols = [c for c in cols if c in results_df.columns]
    results_df = results_df[cols]
    results_df.to_csv(os.path.join(OUT_DIR, "pool_boiling_technique_comparison.csv"), index=False)
    print("\n" + "=" * 100)
    print(results_df.to_string(index=False))


if __name__ == "__main__":
    main()

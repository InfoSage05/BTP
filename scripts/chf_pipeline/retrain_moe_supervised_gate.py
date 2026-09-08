"""
Follow-up to retrain_moe_joint.py: joint-regime training alone only barely
improved gate routing (still far from ideal). Root cause: G_kg_m2s is a
perfectly separable signal (pool rows are exactly 0.0, flow rows are >=8.2),
so this is not a data-diversity problem -- the gate only ever receives
gradient through the indirect regression loss path, which is weak
supervision for what is actually a trivial classification problem.

Fix: add a direct auxiliary cross-entropy loss on the gate output against
the true regime label -- known exactly for every training row.
"""
import os
import sys
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

sys.path.insert(0, "scripts/chf_pipeline")
from models import MoEModel
from data_prep import FEATURE_COLS
from pool_boiling_techniques import to_tensor, evaluate, build_base, Standardizer

CKPT_DIR = "data/processed/stage1/checkpoints"
STAGE1_DIR = "data/processed/stage1"
POOL_DATA = "data/processed/pool_boiling/strip_pool_boiling_water.csv"
OUT_DIR = "data/processed/pool_boiling"
RNG_SEED = 42
TEST_FRAC = 0.20
EPOCHS = {"mlp": 200, "transformer": 150}
LR = {"mlp": 3e-3, "transformer": 3e-3}  # gate LR -- gate is small, trains from scratch, needs speed
POOL_EXPERT_FT_LR = {"mlp": 1e-4, "transformer": 5e-5}  # matches finetune_mlp.py / finetune_transformer.py's
                                                        # LR_FINETUNE convention -- pool_expert is now pretrained,
                                                        # not from-scratch, so it needs the same low fine-tune LR
                                                        # those scripts use, not the gate's LR
PATIENCE = 20
AUX_WEIGHT = 1.0


def train_moe_supervised(moe, x_tr, y_tr, regime_tr, x_val, y_val, regime_val,
                          epochs, gate_lr, pool_expert_lr, label):
    # separate param groups: gate needs to learn fast from scratch; pool_expert
    # is now pretrained (see main()) and needs a low fine-tune LR like every
    # other pretrained-then-finetuned model in this project, or its pretrained
    # weights get destroyed by the gate's much higher LR in a shared optimizer
    optimizer = torch.optim.Adam([
        {"params": moe.gate.parameters(), "lr": gate_lr},
        {"params": moe.pool_expert.parameters(), "lr": pool_expert_lr},
    ])
    mse = nn.MSELoss()
    ce = nn.CrossEntropyLoss()
    best_val, best_state, bad = float("inf"), None, 0
    n = len(x_tr)
    batch_size = min(32, max(4, n // 4))
    for epoch in range(epochs):
        moe.train()
        perm = torch.randperm(n)
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            optimizer.zero_grad()
            pred, gate_w = moe.forward_with_gate(x_tr[idx])
            reg_loss = mse(pred, y_tr[idx])
            gate_loss = ce(gate_w, regime_tr[idx])
            loss = reg_loss + AUX_WEIGHT * gate_loss
            loss.backward()
            optimizer.step()
        moe.eval()
        with torch.no_grad():
            pred_val, gate_val = moe.forward_with_gate(x_val)
            val_loss = (mse(pred_val, y_val) + AUX_WEIGHT * ce(gate_val, regime_val)).item()
        if val_loss < best_val - 1e-6:
            best_val = val_loss
            best_state = {k: v.clone() for k, v in moe.state_dict().items()}
            bad = 0
        else:
            bad += 1
        if bad >= PATIENCE:
            print("    [" + label + "] early stop at epoch " + str(epoch + 1) +
                  " (best val=" + str(round(best_val, 4)) + ")", flush=True)
            break
    if best_state is not None:
        moe.load_state_dict(best_state)
    return moe


def main():
    rng = np.random.default_rng(RNG_SEED)

    pool_df = pd.read_csv(POOL_DATA)
    idx = rng.permutation(len(pool_df))
    n_test = max(1, int(len(pool_df) * TEST_FRAC))
    n_val = max(1, int(len(pool_df) * 0.15))
    pool_test = pool_df.iloc[idx[:n_test]]
    pool_val = pool_df.iloc[idx[n_test:n_test + n_val]]
    pool_train = pool_df.iloc[idx[n_test + n_val:]]

    flow_train_full = pd.read_csv(os.path.join(STAGE1_DIR, "train_interp.csv"))
    flow_val_full = pd.read_csv(os.path.join(STAGE1_DIR, "val_interp.csv"))
    flow_test_full = pd.read_csv(os.path.join(STAGE1_DIR, "test_interp.csv"))
    flow_train = flow_train_full.sample(n=len(pool_train), random_state=RNG_SEED)
    flow_val = flow_val_full.sample(n=len(pool_val), random_state=RNG_SEED)
    flow_test = flow_test_full.sample(n=len(pool_test), random_state=RNG_SEED)

    joint_train = pd.concat([pool_train, flow_train], ignore_index=True)
    joint_val = pd.concat([pool_val, flow_val], ignore_index=True)
    print("Joint set: train=" + str(len(joint_train)) + " (" + str(len(pool_train)) +
          " pool + " + str(len(flow_train)) + " flow), val=" + str(len(joint_val)) +
          " (" + str(len(pool_val)) + " pool + " + str(len(flow_val)) + " flow)")

    with open(os.path.join(CKPT_DIR, "scaler.pkl"), "rb") as f:
        scaler_bundle = pickle.load(f)
    feat_scaler = scaler_bundle["feat_scaler"]
    target_mean, target_std = scaler_bundle["target_mean"], scaler_bundle["target_std"]

    x_tr, y_tr = to_tensor(joint_train, feat_scaler, target_mean, target_std)
    x_val, y_val = to_tensor(joint_val, feat_scaler, target_mean, target_std)
    x_pool_test, y_pool_test = to_tensor(pool_test, feat_scaler, target_mean, target_std)
    x_flow_test, y_flow_test = to_tensor(flow_test, feat_scaler, target_mean, target_std)

    regime_tr = torch.tensor(np.concatenate([np.ones(len(pool_train)), np.zeros(len(flow_train))]), dtype=torch.long)
    regime_val = torch.tensor(np.concatenate([np.ones(len(pool_val)), np.zeros(len(flow_val))]), dtype=torch.long)

    results = []
    for arch in ["mlp", "transformer"]:
        n_feat = len(FEATURE_COLS)
        pretrained_state = torch.load(os.path.join(CKPT_DIR, arch + "_pretrained.pt"))

        print("\n=== " + arch + ": moe (joint-regime + supervised gate) ===", flush=True)
        flow_expert = build_base(arch, n_feat)
        flow_expert.load_state_dict(pretrained_state)
        # pool_expert now ALSO starts from the pretrained checkpoint (previously
        # from-scratch for both archs -- MLP happened to work anyway since MLPs
        # need less data, Transformer failed outright, R2~=0, on only 36 rows).
        # Pretrain-then-finetune is the pattern used everywhere else in this
        # project for exactly this reason; this was the one place it was
        # omitted, not a considered choice to skip it.
        pool_expert = build_base(arch, n_feat)
        pool_expert.load_state_dict(pretrained_state)
        moe = MoEModel(flow_expert, pool_expert, n_feat)
        moe = train_moe_supervised(moe, x_tr, y_tr, regime_tr, x_val, y_val, regime_val,
                                    EPOCHS[arch], LR[arch], POOL_EXPERT_FT_LR[arch],
                                    arch + "-moe-supervised")

        r_pool = evaluate(moe, x_pool_test, y_pool_test, target_mean, target_std)
        r_pool.update({"arch": arch, "technique": "moe_supervised_gate", "test_set": "pool_only_heldout"})
        print("  pool test (n=" + str(len(pool_test)) + "): " + str(r_pool), flush=True)
        results.append(r_pool)

        r_flow = evaluate(moe, x_flow_test, y_flow_test, target_mean, target_std)
        r_flow.update({"arch": arch, "technique": "moe_supervised_gate", "test_set": "flow_only_heldout"})
        print("  flow test (n=" + str(len(flow_test)) + "): " + str(r_flow), flush=True)
        results.append(r_flow)

        with torch.no_grad():
            gate_pool = moe.gate(x_pool_test).numpy()
            gate_flow = moe.gate(x_flow_test).numpy()
        pool_gate_on_pool = float(gate_pool[:, 1].mean())
        flow_gate_on_flow = float(gate_flow[:, 0].mean())
        print("  GATE CHECK: avg P(pool_expert)|pool_row=" + str(round(pool_gate_on_pool, 3)) +
              "  avg P(flow_expert)|flow_row=" + str(round(flow_gate_on_flow, 3)), flush=True)
        results.append({"arch": arch, "technique": "moe_supervised_gate_check", "test_set": "both_heldout",
                         "R2": float("nan"),
                         "avg_pool_gate_on_pool_rows": round(pool_gate_on_pool, 4),
                         "avg_flow_gate_on_flow_rows": round(flow_gate_on_flow, 4),
                         "n": len(pool_test) + len(flow_test)})

    results_df = pd.DataFrame(results)
    cols = ["arch", "technique", "test_set", "n", "R2", "rRMSE", "MAPE_%", "within_10pct_%",
            "RMSE_kW_m2", "avg_pool_gate_on_pool_rows", "avg_flow_gate_on_flow_rows"]
    cols = [c for c in cols if c in results_df.columns]
    results_df = results_df[cols]
    out_path = os.path.join(OUT_DIR, "moe_supervised_gate_results.csv")
    results_df.to_csv(out_path, index=False)
    print("\nWrote " + out_path)
    print(results_df.to_string(index=False))


if __name__ == "__main__":
    main()

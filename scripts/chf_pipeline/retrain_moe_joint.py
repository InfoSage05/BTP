"""
Fixes the MoE gate-training gap flagged in data/processed/stage4/README.md:
the original MoE run (pool_boiling_techniques.py) trained the gate and pool
expert on POOL-ONLY rows, so the gate never saw a genuine flow-boiling (G>0)
example during training and had no data-driven signal to learn real
regime-routing -- it ended up assigning P(pool_expert)~=0.07-0.11 to true
pool rows, essentially inverted.

This script retrains the gate + pool_expert (flow_expert stays frozen, same
as before) on a JOINT training set: the pool-boiling train/val rows plus a
balanced sample of real flow-boiling train/val rows (drawn from Stage 1's
train_interp.csv/val_interp.csv). Evaluation stays regime-separated and
uses ONLY previously-held-out test data:
  - pool regime: the same pool test_df split already carved out in
    prepare_pool_boiling.py / pool_boiling_techniques.py (11 rows, never
    touched here)
  - flow regime: a fresh sample from Stage 1's test_interp.csv (never used
    in any training in this project)
"""
import os
import sys
import pickle
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, "scripts/chf_pipeline")
from device_utils import DEVICE
from models import SmallMLP, FTTransformer, MoEModel
from data_prep import FEATURE_COLS, TARGET_COL
from pool_boiling_techniques import to_tensor, evaluate, train_loop, build_base, Standardizer

CKPT_DIR = "data/processed/stage1/checkpoints"
STAGE1_DIR = "data/processed/stage1"
POOL_DATA = "data/processed/pool_boiling/strip_pool_boiling_water.csv"
OUT_DIR = "data/processed/pool_boiling"
RNG_SEED = 42
TEST_FRAC = 0.20
EPOCHS = {"mlp": 200, "transformer": 150}
LR = {"mlp": 3e-3, "transformer": 3e-3}


def main():
    rng = np.random.default_rng(RNG_SEED)

    pool_df = pd.read_csv(POOL_DATA)
    idx = rng.permutation(len(pool_df))
    n_test = max(1, int(len(pool_df) * TEST_FRAC))
    n_val = max(1, int(len(pool_df) * 0.15))
    pool_test = pool_df.iloc[idx[:n_test]]
    pool_val = pool_df.iloc[idx[n_test:n_test + n_val]]
    pool_train = pool_df.iloc[idx[n_test + n_val:]]
    print(f"Pool split (same as original MoE run): train={len(pool_train)} "
          f"val={len(pool_val)} test={len(pool_test)}")

    flow_train_full = pd.read_csv(os.path.join(STAGE1_DIR, "train_interp.csv"))
    flow_val_full = pd.read_csv(os.path.join(STAGE1_DIR, "val_interp.csv"))
    flow_test_full = pd.read_csv(os.path.join(STAGE1_DIR, "test_interp.csv"))

    # balanced flow sample for JOINT gate training, same order of magnitude as pool
    flow_train = flow_train_full.sample(n=len(pool_train), random_state=RNG_SEED)
    flow_val = flow_val_full.sample(n=len(pool_val), random_state=RNG_SEED)
    flow_test = flow_test_full.sample(n=len(pool_test), random_state=RNG_SEED)

    joint_train = pd.concat([pool_train, flow_train], ignore_index=True)
    joint_val = pd.concat([pool_val, flow_val], ignore_index=True)
    print(f"Joint training set: train={len(joint_train)} "
          f"({len(pool_train)} pool + {len(flow_train)} flow), "
          f"val={len(joint_val)} ({len(pool_val)} pool + {len(flow_val)} flow)")

    with open(os.path.join(CKPT_DIR, "scaler.pkl"), "rb") as f:
        scaler_bundle = pickle.load(f)
    feat_scaler = scaler_bundle["feat_scaler"]
    target_mean, target_std = scaler_bundle["target_mean"], scaler_bundle["target_std"]

    x_tr, y_tr = to_tensor(joint_train, feat_scaler, target_mean, target_std)
    x_val, y_val = to_tensor(joint_val, feat_scaler, target_mean, target_std)
    x_pool_test, y_pool_test = to_tensor(pool_test, feat_scaler, target_mean, target_std)
    x_flow_test, y_flow_test = to_tensor(flow_test, feat_scaler, target_mean, target_std)

    results = []
    for arch in ["mlp", "transformer"]:
        n_feat = len(FEATURE_COLS)
        pretrained_state = torch.load(os.path.join(CKPT_DIR, f"{arch}_pretrained.pt"), map_location=DEVICE)

        print(f"\n=== {arch}: moe (JOINT-regime gate training) ===", flush=True)
        flow_expert = build_base(arch, n_feat)
        flow_expert.load_state_dict(pretrained_state)
        pool_expert = build_base(arch, n_feat)
        moe = MoEModel(flow_expert, pool_expert, n_feat).to(DEVICE)
        trainable = [p for p in moe.parameters() if p.requires_grad]
        moe = train_loop(moe, trainable, x_tr, y_tr, x_val, y_val,
                          EPOCHS[arch], LR[arch], f"{arch}-moe-joint")

        r_pool = evaluate(moe, x_pool_test, y_pool_test, target_mean, target_std)
        r_pool.update({"arch": arch, "technique": "moe_joint", "test_set": "pool_only_heldout"})
        print(f"  pool test (n={len(pool_test)}): {r_pool}", flush=True)
        results.append(r_pool)

        r_flow = evaluate(moe, x_flow_test, y_flow_test, target_mean, target_std)
        r_flow.update({"arch": arch, "technique": "moe_joint", "test_set": "flow_only_heldout"})
        print(f"  flow test (n={len(flow_test)}): {r_flow}", flush=True)
        results.append(r_flow)

        with torch.no_grad():
            gate_pool = moe.gate(x_pool_test).numpy()
            gate_flow = moe.gate(x_flow_test).numpy()
        pool_gate_on_pool = float(gate_pool[:, 1].mean())
        flow_gate_on_flow = float(gate_flow[:, 0].mean())
        print(f"  GATE CHECK: avg P(pool_expert)|pool_row={pool_gate_on_pool:.3f}  "
              f"avg P(flow_expert)|flow_row={flow_gate_on_flow:.3f}", flush=True)
        results.append({"arch": arch, "technique": "moe_joint_gate_check", "test_set": "both_heldout",
                         "R2": float("nan"),
                         "avg_pool_gate_on_pool_rows": round(pool_gate_on_pool, 4),
                         "avg_flow_gate_on_flow_rows": round(flow_gate_on_flow, 4),
                         "n": len(pool_test) + len(flow_test)})

    results_df = pd.DataFrame(results)
    cols = ["arch", "technique", "test_set", "n", "R2", "rRMSE", "MAPE_%", "within_10pct_%",
            "RMSE_kW_m2", "avg_pool_gate_on_pool_rows", "avg_flow_gate_on_flow_rows"]
    cols = [c for c in cols if c in results_df.columns]
    results_df = results_df[cols]
    out_path = os.path.join(OUT_DIR, "moe_joint_regime_results.csv")
    results_df.to_csv(out_path, index=False)
    print(f"\nWrote {out_path}")
    print(results_df.to_string(index=False))


if __name__ == "__main__":
    main()

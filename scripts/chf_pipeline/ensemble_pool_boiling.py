import os
import sys
import pickle
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, "scripts/chf_pipeline")
from models import SmallMLP, FTTransformer
from data_prep import FEATURE_COLS
from pool_boiling_techniques import to_tensor, evaluate, train_loop, CKPT_DIR, POOL_DATA, TEST_FRAC, EPOCHS

N_MEMBERS = 5
OUT_DIR = "data/processed/pool_boiling"
RNG_SEED = 42


class Standardizer:
    def transform(self, x):
        return (x - self.mean) / self.std


def ensemble_predict(models_list, x, target_mean, target_std):
    preds_log = []
    for m in models_list:
        m.eval()
        with torch.no_grad():
            preds_log.append(m(x).numpy() * target_std + target_mean)
    preds_log = np.stack(preds_log, axis=0)
    mean_log, std_log = preds_log.mean(axis=0), preds_log.std(axis=0)
    mean_pred = np.exp(mean_log)
    std_pred = mean_pred * std_log
    return mean_pred, std_pred


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
            "within_10pct_%": within_10pct, "coverage_95pct_%": coverage_95, "n": int(len(true))}


def main():
    rng = np.random.default_rng(RNG_SEED)
    pool_df = pd.read_csv(POOL_DATA)
    idx = rng.permutation(len(pool_df))
    n_test = max(1, int(len(pool_df) * TEST_FRAC))
    n_val = max(1, int(len(pool_df) * 0.15))
    test_df = pool_df.iloc[idx[:n_test]]
    val_df = pool_df.iloc[idx[n_test:n_test + n_val]]
    train_df = pool_df.iloc[idx[n_test + n_val:]]

    with open(os.path.join(CKPT_DIR, "scaler.pkl"), "rb") as f:
        scaler_bundle = pickle.load(f)
    feat_scaler = scaler_bundle["feat_scaler"]
    target_mean, target_std = scaler_bundle["target_mean"], scaler_bundle["target_std"]

    x_tr, y_tr = to_tensor(train_df, feat_scaler, target_mean, target_std)
    x_val, y_val = to_tensor(val_df, feat_scaler, target_mean, target_std)
    x_test, y_test = to_tensor(test_df, feat_scaler, target_mean, target_std)

    results = []
    for arch in ["mlp", "transformer"]:
        n_feat = len(FEATURE_COLS)
        pretrained_state = torch.load(os.path.join(CKPT_DIR, arch + "_pretrained.pt"))
        single_r2 = []
        members = []
        for seed in range(N_MEMBERS):
            torch.manual_seed(3000 + seed)
            m = SmallMLP(n_feat) if arch == "mlp" else FTTransformer(n_feat)
            m.load_state_dict(pretrained_state)
            lr = 1e-4 if arch == "mlp" else 5e-5
            m = train_loop(m, list(m.parameters()), x_tr, y_tr, x_val, y_val,
                            EPOCHS[arch], lr, arch + "-pool-ens" + str(seed))
            members.append(m)
            mean_pred, std_pred = ensemble_predict([m], x_test, target_mean, target_std)
            r2 = evaluate_ensemble(mean_pred, std_pred, y_test, target_mean, target_std)["R2"]
            single_r2.append(r2)
            print("    [" + arch + "] member " + str(seed) + ": R2=" + str(round(r2, 4)), flush=True)

        mean_pred, std_pred = ensemble_predict(members, x_test, target_mean, target_std)
        ens = evaluate_ensemble(mean_pred, std_pred, y_test, target_mean, target_std)
        ens.update({"arch": arch, "single_model_R2_mean": float(np.mean(single_r2)),
                     "single_model_R2_std": float(np.std(single_r2))})
        print("  ENSEMBLE [" + arch + "]: " + str(ens), flush=True)
        results.append(ens)

    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(OUT_DIR, "pool_boiling_ensemble_results.csv"), index=False)
    print(results_df.to_string(index=False))


if __name__ == "__main__":
    main()

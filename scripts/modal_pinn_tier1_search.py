"""
modal_pinn_tier1_search.py
---------------------------
Corrected PINN hyperparameter search on Modal Cloud GPUs (NVIDIA A10G), implementing
the Tier 0 / Tier 1 fixes from .claude/PLAN.md:

  1. No `pos_loss` term (positivity is already structural via exp() at inference; the old
     penalty just inflated predictions in the low-CHF region where Split C lives).
  2. Nested hyperparameter selection: search is ranked on a PROXY split carved out of
     training data only (fit on P<=14000, validate on 15000-16000 kPa). The real Split C
     test set (P>=17000) is touched exactly once, at the end, for the single winning config.
  3. Early stopping uses the proxy/held-out pressure band, not a random 15% split (which
     measures interpolation, not extrapolation).
  4. Mini-batch training (batch=512) instead of one full-batch gradient step per epoch.
  5. Reports log-space R^2, raw-space R^2, and MAPE side by side for every result.
  6. Final stage trains a 10-seed deep ensemble of the winning config on the real training
     set and reports both the per-seed mean R^2 and the ensembled (log-space-averaged) R^2.

Usage:
    python -m modal run scripts/modal_pinn_tier1_search.py
"""
import time
import json
from pathlib import Path
import pandas as pd
import numpy as np
import modal

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch",
        "numpy",
        "pandas",
        "scikit-learn",
        "scipy",
        "CoolProp",
    )
    .add_local_file("data/chf_long_clean.csv", remote_path="/root/data/chf_long_clean.csv")
    .add_local_file("scripts/chf_physics.py", remote_path="/root/chf_physics.py")
)

app = modal.App("chf-pinn-tier1-search")


def _build_model(hidden_layers, in_dim=3):
    import torch.nn as nn
    layers = []
    d = in_dim
    for h in hidden_layers:
        layers.append(nn.Linear(d, h))
        layers.append(nn.Tanh())
        d = h
    layers.append(nn.Linear(d, 1))
    return nn.Sequential(*layers)


def _train_one(Xtr, ytr, Xval, yval, hidden_layers, lam_mono, lam_zuber, lr,
                n_collocation, p_bounds, g_bounds, x_bounds, epochs, patience,
                batch_size, seed, device, weight_decay=1e-5):
    import torch
    import torch.nn as nn
    from sklearn.preprocessing import StandardScaler

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    rng = np.random.RandomState(seed)

    x_scaler = StandardScaler().fit(Xtr)
    Xtr_s = x_scaler.transform(Xtr)
    Xval_s = x_scaler.transform(Xval)

    log_ytr = np.log(ytr)
    y_mean, y_std = log_ytr.mean(), log_ytr.std()
    ytr_norm = (log_ytr - y_mean) / y_std

    Xtr_t = torch.tensor(Xtr_s, dtype=torch.float32, device=device)
    ytr_t = torch.tensor(ytr_norm, dtype=torch.float32, device=device)
    Xval_t = torch.tensor(Xval_s, dtype=torch.float32, device=device)

    model = _build_model(hidden_layers).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=15, min_lr=1e-6
    )

    p_lo, p_hi = p_bounds
    g_lo, g_hi = g_bounds
    x_lo, x_hi = x_bounds

    n = len(Xtr_t)
    best_val = float("inf")
    best_state = None
    no_improve = 0

    for epoch in range(epochs):
        model.train()
        perm = rng.permutation(n)
        for start in range(0, n, batch_size):
            idx = perm[start:start + batch_size]
            optimizer.zero_grad()

            pred_tr = model(Xtr_t[idx]).squeeze(-1)
            data_loss = nn.functional.mse_loss(pred_tr, ytr_t[idx])

            bs = len(idx)
            p_c_raw = rng.uniform(p_lo, p_hi, min(n_collocation, max(bs, 32)))
            g_c_raw = rng.uniform(g_lo, g_hi, len(p_c_raw))
            x_c_raw = rng.uniform(x_lo, x_hi, len(p_c_raw))

            p_c_s = torch.tensor((p_c_raw - x_scaler.mean_[0]) / x_scaler.scale_[0],
                                  dtype=torch.float32, device=device, requires_grad=True)
            g_c_s = torch.tensor((g_c_raw - x_scaler.mean_[1]) / x_scaler.scale_[1],
                                  dtype=torch.float32, device=device)
            x_c_s = torch.tensor((x_c_raw - x_scaler.mean_[2]) / x_scaler.scale_[2],
                                  dtype=torch.float32, device=device, requires_grad=True)

            inp_c = torch.stack([p_c_s, g_c_s, x_c_s], dim=1)
            pred_c = model(inp_c).squeeze(-1)

            dpred_dx = torch.autograd.grad(pred_c.sum(), x_c_s, create_graph=True)[0]
            mono_loss = torch.relu(dpred_dx).pow(2).mean()

            dpred_dp = torch.autograd.grad(pred_c.sum(), p_c_s, create_graph=True)[0]
            zuber_sign = torch.tensor(_zuber_sign_table(p_c_raw), dtype=torch.float32, device=device)
            zuber_loss = torch.relu(-dpred_dp * zuber_sign).pow(2).mean()

            total_loss = data_loss + lam_mono * mono_loss + lam_zuber * zuber_loss
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_pred = model(Xval_t).squeeze(-1)
            val_true = torch.tensor((np.log(yval) - y_mean) / y_std, dtype=torch.float32, device=device)
            val_loss = nn.functional.mse_loss(val_pred, val_true).item()

        scheduler.step(val_loss)

        if val_loss < best_val - 1e-6:
            best_val = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
        if no_improve > patience:
            break

    model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    model.eval()
    with torch.no_grad():
        pred_norm = model(Xval_t).squeeze(-1).cpu().numpy()
    pred_val = np.exp(pred_norm * y_std + y_mean)

    return model, x_scaler, y_mean, y_std, pred_val, epoch + 1


_ZUBER_P = None
_ZUBER_DSIGN = None


def _zuber_sign_table(p_kpa):
    # Real Zuber (1959) pool-boiling correlation via CoolProp/IAPWS steam-table properties
    # (scripts/chf_physics.py, same function the original notebook and Modal scripts use) —
    # not a hand-fit approximation, since the physics penalty's sign directly shapes training.
    global _ZUBER_P, _ZUBER_DSIGN
    if _ZUBER_P is None:
        from chf_physics import zuber_pool_boiling_chf
        p = np.linspace(100.0, 21500.0, 300)
        q_zuber = zuber_pool_boiling_chf(p)
        _ZUBER_P, _ZUBER_DSIGN = p, np.sign(np.gradient(q_zuber, p))
    return np.interp(np.asarray(p_kpa, dtype=float), _ZUBER_P, _ZUBER_DSIGN)


def _metrics(y_true, y_pred):
    from sklearn.metrics import r2_score
    raw_r2 = float(r2_score(y_true, y_pred))
    log_r2 = float(r2_score(np.log(y_true), np.log(np.clip(y_pred, 1e-6, None))))
    mape = float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100)
    return raw_r2, log_r2, mape


# ============================================================================
# 1. PROXY-SPLIT HYPERPARAMETER SEARCH WORKER
# ============================================================================
@app.function(image=image, gpu="A10G", timeout=900)
def proxy_search_worker(config_json: str, seeds: list):
    import torch
    config = json.loads(config_json)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    df = pd.read_csv("/root/data/chf_long_clean.csv")
    df = df[df.X != 1.0].reset_index(drop=True)
    FEATURES = ["P", "G", "X"]
    TARGET = "CHF"

    proxy_train = df[df.P <= 14000].reset_index(drop=True)
    proxy_val = df[(df.P >= 15000) & (df.P <= 16000)].reset_index(drop=True)

    Xtr, ytr = proxy_train[FEATURES].values, proxy_train[TARGET].values
    Xval, yval = proxy_val[FEATURES].values, proxy_val[TARGET].values

    p_bounds = (df.P.min(), df.P.max())
    g_bounds = (df.G.min(), df.G.max())
    x_bounds = (df.X.min(), 0.9)

    raw_r2s, log_r2s, mapes = [], [], []
    for seed in seeds:
        _, _, _, _, pred_val, _ = _train_one(
            Xtr, ytr, Xval, yval,
            hidden_layers=config["hidden_layers"],
            lam_mono=config["lam_mono"], lam_zuber=config["lam_zuber"],
            lr=config["lr"], n_collocation=config["n_collocation"],
            p_bounds=p_bounds, g_bounds=g_bounds, x_bounds=x_bounds,
            epochs=config.get("epochs", 600), patience=config.get("patience", 35),
            batch_size=config.get("batch_size", 2048), seed=seed, device=device,
        )
        raw_r2, log_r2, mape = _metrics(yval, pred_val)
        raw_r2s.append(raw_r2); log_r2s.append(log_r2); mapes.append(mape)

    return {
        "arch": "x".join(str(h) for h in config["hidden_layers"]),
        "lam_mono": config["lam_mono"], "lam_zuber": config["lam_zuber"],
        "lr": config["lr"], "n_collocation": config["n_collocation"],
        "proxy_log_r2_mean": float(np.mean(log_r2s)), "proxy_log_r2_std": float(np.std(log_r2s)),
        "proxy_raw_r2_mean": float(np.mean(raw_r2s)), "proxy_raw_r2_std": float(np.std(raw_r2s)),
        "proxy_mape_mean": float(np.mean(mapes)),
    }


# ============================================================================
# 2. FINAL EVALUATION WORKER (real Split C, touched once, ensemble seeds)
# ============================================================================
@app.function(image=image, gpu="A10G", timeout=900)
def final_eval_worker(config_json: str, seed: int):
    import torch
    config = json.loads(config_json)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    df = pd.read_csv("/root/data/chf_long_clean.csv")
    df = df[df.X != 1.0].reset_index(drop=True)
    FEATURES = ["P", "G", "X"]
    TARGET = "CHF"

    fit_train = df[df.P < 14000].reset_index(drop=True)
    fit_val = df[(df.P >= 14000) & (df.P <= 16000)].reset_index(drop=True)
    real_test = df[df.P >= 17000].reset_index(drop=True)

    Xtr, ytr = fit_train[FEATURES].values, fit_train[TARGET].values
    Xval, yval = fit_val[FEATURES].values, fit_val[TARGET].values
    Xte, yte = real_test[FEATURES].values, real_test[TARGET].values

    p_bounds = (df.P.min(), df.P.max())
    g_bounds = (df.G.min(), df.G.max())
    x_bounds = (df.X.min(), 0.9)

    model, x_scaler, y_mean, y_std, _, epochs_trained = _train_one(
        Xtr, ytr, Xval, yval,
        hidden_layers=config["hidden_layers"],
        lam_mono=config["lam_mono"], lam_zuber=config["lam_zuber"],
        lr=config["lr"], n_collocation=config["n_collocation"],
        p_bounds=p_bounds, g_bounds=g_bounds, x_bounds=x_bounds,
        epochs=config.get("epochs", 1000), patience=config.get("patience", 50),
        batch_size=config.get("batch_size", 2048), seed=seed, device=device,
    )

    Xte_s = x_scaler.transform(Xte)
    with torch.no_grad():
        pred_norm = model(torch.tensor(Xte_s, dtype=torch.float32, device=device)).squeeze(-1).cpu().numpy()
    pred_te = np.exp(pred_norm * y_std + y_mean)
    raw_r2, log_r2, mape = _metrics(yte, pred_te)

    return {
        "seed": seed, "raw_r2": raw_r2, "log_r2": log_r2, "mape": mape,
        "epochs_trained": epochs_trained, "pred_log": (np.log(pred_te)).tolist(),
        "y_true": yte.tolist(),
    }


# ============================================================================
# 3. LOCAL ENTRYPOINT
# ============================================================================
@app.local_entrypoint()
def main():
    print("=" * 80)
    print("TIER 1 PINN SEARCH — proxy-split selection, no pos_loss, mini-batch, honest metrics")
    print("=" * 80)

    # Deliberately small, ablation-shaped grid (not a full cross-product) — informed by the
    # prior full sweep (16x16x8 dominated every 8x8/16x8 config; ncol=512 beat 256; lr=1e-3
    # beat 5e-4 in every top-10 row) plus PLAN.md Tier-1 budget guidance. The (lam_mono,
    # lam_zuber) pairs below directly produce the no-physics -> +mono -> +zuber -> both
    # ablation table the plan asks for, instead of blindly re-sweeping everything.
    archs = [[16, 8], [16, 16, 8]]
    physics_pairs = [
        (0.0, 0.0),   # no physics penalties (pure data-fit baseline)
        (0.3, 0.0),   # monotonicity only
        (0.0, 0.1),   # Zuber pressure-trend only
        (0.3, 0.1),   # both (previous winner's neighborhood)
        (0.5, 0.1),   # previous full-sweep winner
    ]
    lr = 1e-3
    n_collocation = 512
    seeds = [0, 1, 2, 42, 99]

    combos = [
        {"hidden_layers": a, "lam_mono": m, "lam_zuber": z, "lr": lr, "n_collocation": n_collocation}
        for a in archs for (m, z) in physics_pairs
    ]
    work_items = [(json.dumps(c), seeds) for c in combos]
    total_configs = len(work_items)

    print(f"Proxy search: {total_configs} configs x {len(seeds)} seeds = {total_configs * len(seeds)} fits")
    print("Selection metric: log-space R^2 on PROXY validation (P 15000-16000 kPa), "
          "never touching real Split C test set during search.")
    print("Launching parallel GPU containers on Modal Cloud (NVIDIA A10G)...")

    t0 = time.time()
    proxy_results = list(proxy_search_worker.starmap(work_items))
    t_proxy = time.time() - t0
    print(f"\nProxy search complete in {t_proxy:.1f}s ({t_proxy/60:.2f} min)")

    proxy_df = pd.DataFrame(proxy_results).sort_values("proxy_log_r2_mean", ascending=False)
    results_dir = Path("results/pinn")
    results_dir.mkdir(parents=True, exist_ok=True)
    proxy_df.to_csv(results_dir / "tier1_proxy_search_results.csv", index=False)

    print("\n" + "=" * 90)
    print("TOP 10 CONFIGS BY PROXY VALIDATION (log R^2) — selection was NOT based on real Split C")
    print("=" * 90)
    print(proxy_df.head(10).to_string(index=False))

    best_row = proxy_df.iloc[0]
    best_config = {
        "hidden_layers": [int(x) for x in best_row["arch"].split("x")],
        "lam_mono": float(best_row["lam_mono"]),
        "lam_zuber": float(best_row["lam_zuber"]),
        "lr": float(best_row["lr"]),
        "n_collocation": int(best_row["n_collocation"]),
        "epochs": 1000, "patience": 50, "batch_size": 2048,
    }
    print(f"\nWinning config (by proxy val): {best_config}")
    print("\nNow evaluating this ONE config on the real Split C test set (P>=17000 kPa), "
          "10 seeds, touched for the first and only time in this run.")

    final_seeds = list(range(10))
    final_work = [(json.dumps(best_config), s) for s in final_seeds]

    t1 = time.time()
    final_results = list(final_eval_worker.starmap(final_work))
    t_final = time.time() - t1
    print(f"\nFinal Split C evaluation complete in {t_final:.1f}s ({t_final/60:.2f} min)")

    final_df = pd.DataFrame([{k: v for k, v in r.items() if k not in ("pred_log", "y_true")}
                              for r in final_results])
    final_df.to_csv(results_dir / "tier1_final_per_seed.csv", index=False)

    print("\n" + "-" * 80)
    print("PER-SEED HONEST SPLIT C RESULTS (real test set, single touch)")
    print("-" * 80)
    print(final_df.to_string(index=False))
    print(f"\nMean raw R^2:  {final_df.raw_r2.mean():.4f} +/- {final_df.raw_r2.std():.4f}")
    print(f"Mean log R^2:  {final_df.log_r2.mean():.4f} +/- {final_df.log_r2.std():.4f}")
    print(f"Mean MAPE:     {final_df.mape.mean():.2f}% +/- {final_df.mape.std():.2f}%")

    # ---- Ensemble: average predictions in log space across all 10 seeds ----
    y_true = np.array(final_results[0]["y_true"])
    log_preds = np.array([r["pred_log"] for r in final_results])  # (n_seeds, n_test)
    ensemble_log_pred = log_preds.mean(axis=0)
    ensemble_pred = np.exp(ensemble_log_pred)
    ens_raw_r2, ens_log_r2, ens_mape = _metrics(y_true, ensemble_pred)

    print("\n" + "=" * 80)
    print("ENSEMBLE (10-seed average prediction in log space) vs SINGLE-SEED MEAN")
    print("=" * 80)
    print(f"Single-seed mean raw R^2: {final_df.raw_r2.mean():.4f}  (std {final_df.raw_r2.std():.4f})")
    print(f"Ensemble raw R^2:         {ens_raw_r2:.4f}")
    print(f"Ensemble log R^2:         {ens_log_r2:.4f}")
    print(f"Ensemble MAPE:            {ens_mape:.2f}%")
    print(f"\nReference — GridInterp (deterministic) baseline on Split C: raw R^2 = 0.8415")

    summary = {
        "best_config": best_config,
        "proxy_log_r2_mean": float(best_row["proxy_log_r2_mean"]),
        "single_seed_raw_r2_mean": float(final_df.raw_r2.mean()),
        "single_seed_raw_r2_std": float(final_df.raw_r2.std()),
        "ensemble_raw_r2": ens_raw_r2,
        "ensemble_log_r2": ens_log_r2,
        "ensemble_mape": ens_mape,
        "gridinterp_baseline_raw_r2": 0.8415,
        "proxy_search_seconds": t_proxy,
        "final_eval_seconds": t_final,
    }
    with open(results_dir / "tier1_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved: results/pinn/tier1_proxy_search_results.csv, tier1_final_per_seed.csv, tier1_summary.json")

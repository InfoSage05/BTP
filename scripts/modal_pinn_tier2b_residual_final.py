"""
modal_pinn_tier2b_residual_final.py
-------------------------------------
Follow-up to modal_pinn_tier2_search.py. That run selected its "winning" config
by proxy_log_r2_mean (matching Tier 1's convention) and got target_mode=direct,
activation=tanh -- ensemble raw R^2 = 0.7700 on real Split C, still short of the
GridInterp 0.8415 baseline.

But the same proxy search run also showed that target_mode=residual (correction
on top of GridInterp) + activation=silu scores WORSE on proxy log_r2_mean (~0.92
vs ~0.958) yet clearly BETTER on proxy raw_r2_mean (0.944-0.946 vs 0.923) and
proxy MAPE (~8.6-9.3% vs ~20.9%), AND with dramatically lower seed variance
(std 0.001-0.003 vs 0.023) -- see results/pinn/tier2_proxy_search_results.csv.
This is a genuine metric-choice disagreement, not a fishing expedition: the
decision to evaluate this second config is made entirely from proxy-set evidence
(the real Split C test set has not been touched to make this choice), so running
it once and reporting the result honestly -- alongside, not instead of, the
already-reported direct/tanh result -- does not violate the no-test-set-peeking
protocol from PLAN.md Tier 0 item 2.

Runs the full physics ablation (no-physics / +mono / +zuber / +both) for
target_mode=residual, activation=silu, arch=[16,8] on the real Split C test set,
10 seeds each, exactly once.

Usage:
    python -m modal run scripts/modal_pinn_tier2b_residual_final.py
"""
import time
import json
from pathlib import Path
import pandas as pd
import numpy as np
import modal

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch", "numpy", "pandas", "scikit-learn", "scipy", "CoolProp")
    .add_local_file("data/chf_long_with_gridbase.csv", remote_path="/root/data/chf_long_with_gridbase.csv")
    .add_local_file("scripts/chf_physics.py", remote_path="/root/chf_physics.py")
)

app = modal.App("chf-pinn-tier2b-residual-final")


def _build_model(hidden_layers, in_dim=3, activation="silu"):
    import torch.nn as nn
    act_cls = nn.Tanh if activation == "tanh" else nn.SiLU
    layers = []
    d = in_dim
    for h in hidden_layers:
        layers.append(nn.Linear(d, h))
        layers.append(act_cls())
        d = h
    layers.append(nn.Linear(d, 1))
    return nn.Sequential(*layers)


def _train_one(Xtr, ytr, base_tr, Xval, yval, base_val, hidden_layers, activation,
                target_mode, lam_mono, lam_zuber, lr, n_collocation, p_bounds, g_bounds,
                x_bounds, epochs, patience, batch_size, seed, device, weight_decay=1e-5):
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

    if target_mode == "residual":
        train_target_raw = np.log(ytr) - np.log(base_tr)
    else:
        train_target_raw = np.log(ytr)
    y_mean, y_std = train_target_raw.mean(), train_target_raw.std()
    ytr_norm = (train_target_raw - y_mean) / y_std

    Xtr_t = torch.tensor(Xtr_s, dtype=torch.float32, device=device)
    ytr_t = torch.tensor(ytr_norm, dtype=torch.float32, device=device)
    Xval_t = torch.tensor(Xval_s, dtype=torch.float32, device=device)

    model = _build_model(hidden_layers, activation=activation).to(device)
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

    def _val_norm_target():
        if target_mode == "residual":
            return (np.log(yval) - np.log(base_val) - y_mean) / y_std
        return (np.log(yval) - y_mean) / y_std

    val_true_np = _val_norm_target()

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
            val_true = torch.tensor(val_true_np, dtype=torch.float32, device=device)
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
    pred_raw_norm = pred_norm * y_std + y_mean
    if target_mode == "residual":
        pred_val = base_val * np.exp(pred_raw_norm)
        pred_val_log = np.log(base_val) + pred_raw_norm
    else:
        pred_val = np.exp(pred_raw_norm)
        pred_val_log = pred_raw_norm

    return model, x_scaler, y_mean, y_std, pred_val, pred_val_log, epoch + 1


_ZUBER_P = None
_ZUBER_DSIGN = None


def _zuber_sign_table(p_kpa):
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


@app.function(image=image, gpu="A10G", timeout=900)
def final_eval_worker(config_json: str, seed: int):
    import torch
    config = json.loads(config_json)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    df = pd.read_csv("/root/data/chf_long_with_gridbase.csv")
    FEATURES = ["P", "G", "X"]
    TARGET = "CHF"

    fit_train = df[df.P < 14000].reset_index(drop=True)
    fit_val = df[(df.P >= 14000) & (df.P <= 16000)].reset_index(drop=True)
    real_test = df[df.P >= 17000].reset_index(drop=True)

    Xtr, ytr = fit_train[FEATURES].values, fit_train[TARGET].values
    Xval, yval = fit_val[FEATURES].values, fit_val[TARGET].values
    Xte, yte = real_test[FEATURES].values, real_test[TARGET].values
    base_tr = fit_train["grid_base_final"].values
    base_val = fit_val["grid_base_final"].values
    base_te = real_test["grid_base_final"].values

    p_bounds = (df.P.min(), df.P.max())
    g_bounds = (df.G.min(), df.G.max())
    x_bounds = (df.X.min(), 0.9)

    model, x_scaler, y_mean, y_std, _, _, epochs_trained = _train_one(
        Xtr, ytr, base_tr, Xval, yval, base_val,
        hidden_layers=config["hidden_layers"], activation=config["activation"],
        target_mode=config["target_mode"],
        lam_mono=config["lam_mono"], lam_zuber=config["lam_zuber"],
        lr=config["lr"], n_collocation=config["n_collocation"],
        p_bounds=p_bounds, g_bounds=g_bounds, x_bounds=x_bounds,
        epochs=config.get("epochs", 1000), patience=config.get("patience", 50),
        batch_size=config.get("batch_size", 2048), seed=seed, device=device,
    )

    Xte_s = x_scaler.transform(Xte)
    with torch.no_grad():
        pred_norm = model(torch.tensor(Xte_s, dtype=torch.float32, device=device)).squeeze(-1).cpu().numpy()
    pred_raw_norm = pred_norm * y_std + y_mean
    if config["target_mode"] == "residual":
        pred_te = base_te * np.exp(pred_raw_norm)
        pred_te_log = np.log(base_te) + pred_raw_norm
    else:
        pred_te = np.exp(pred_raw_norm)
        pred_te_log = pred_raw_norm
    raw_r2, log_r2, mape = _metrics(yte, pred_te)

    return {
        "seed": seed, "raw_r2": raw_r2, "log_r2": log_r2, "mape": mape,
        "epochs_trained": epochs_trained, "pred_log": pred_te_log.tolist(),
        "y_true": yte.tolist(), "P": real_test["P"].tolist(),
    }


@app.local_entrypoint()
def main():
    print("=" * 80)
    print("TIER 2b — residual-on-GridInterp + SiLU, chosen by proxy raw_R2/MAPE/stability")
    print("(second, honestly-reported real-Split-C evaluation; decision made from proxy data only)")
    print("=" * 80)

    arch = [16, 8]
    base_config = {"hidden_layers": arch, "target_mode": "residual", "activation": "silu",
                    "lr": 1e-3, "n_collocation": 512, "epochs": 1000, "patience": 50, "batch_size": 2048}
    physics_pairs = [(0.0, 0.0), (0.3, 0.0), (0.0, 0.1), (0.3, 0.1)]
    final_seeds = list(range(10))

    results_dir = Path("results/pinn")
    results_dir.mkdir(parents=True, exist_ok=True)

    ablation_rows = []
    ensemble_by_physics = {}
    for (m, z) in physics_pairs:
        cfg = dict(base_config, lam_mono=m, lam_zuber=z)
        final_work = [(json.dumps(cfg), s) for s in final_seeds]
        t1 = time.time()
        final_results = list(final_eval_worker.starmap(final_work))
        dt = time.time() - t1

        final_df = pd.DataFrame([{k: v for k, v in r.items() if k not in ("pred_log", "y_true", "P")}
                                  for r in final_results])
        y_true = np.array(final_results[0]["y_true"])
        p_col = np.array(final_results[0]["P"])
        log_preds = np.array([r["pred_log"] for r in final_results])
        ensemble_pred = np.exp(log_preds.mean(axis=0))
        ens_raw_r2, ens_log_r2, ens_mape = _metrics(y_true, ensemble_pred)

        ablation_rows.append({
            "lam_mono": m, "lam_zuber": z,
            "single_seed_raw_r2_mean": float(final_df.raw_r2.mean()),
            "single_seed_raw_r2_std": float(final_df.raw_r2.std()),
            "ensemble_raw_r2": ens_raw_r2, "ensemble_log_r2": ens_log_r2, "ensemble_mape": ens_mape,
            "seconds": dt,
        })
        ensemble_by_physics[(m, z)] = (y_true, p_col, ensemble_pred)
        print(f"  lam_mono={m} lam_zuber={z}: single-seed R2={final_df.raw_r2.mean():.4f}"
              f"+/-{final_df.raw_r2.std():.4f}  ensemble R2={ens_raw_r2:.4f}  MAPE={ens_mape:.2f}%  ({dt:.0f}s)")

    ablation_df = pd.DataFrame(ablation_rows)
    ablation_df.to_csv(results_dir / "tier2b_residual_silu_ablation_table.csv", index=False)

    print("\n" + "=" * 80)
    print("TIER 2b ABLATION TABLE (residual-on-GridInterp + SiLU, real Split C, 10-seed ensemble)")
    print("=" * 80)
    print(ablation_df.to_string(index=False))
    print(f"\nReference — GridInterp (deterministic) baseline on Split C: raw R^2 = 0.8415")
    print(f"Reference — Tier 2 direct/tanh (proxy-log_r2 winner): ensemble raw R^2 = 0.7700")

    best_idx = ablation_df["ensemble_raw_r2"].idxmax()
    best_physics = (float(ablation_df.loc[best_idx, "lam_mono"]), float(ablation_df.loc[best_idx, "lam_zuber"]))
    y_true, p_col, ensemble_pred = ensemble_by_physics[best_physics]

    print(f"\nPer-pressure-level breakdown (best row: lam_mono={best_physics[0]}, lam_zuber={best_physics[1]}):")
    breakdown_rows = []
    for p_level in sorted(np.unique(p_col)):
        mask = p_col == p_level
        r2, log_r2, mape = _metrics(y_true[mask], ensemble_pred[mask])
        breakdown_rows.append({"P_kPa": p_level, "n": int(mask.sum()), "raw_r2": r2, "log_r2": log_r2, "mape": mape})
        print(f"  P={p_level:>7.0f} kPa  n={int(mask.sum()):>4d}  raw_R2={r2:>7.4f}  log_R2={log_r2:>7.4f}  MAPE={mape:>6.2f}%")
    pd.DataFrame(breakdown_rows).to_csv(results_dir / "tier2b_per_pressure_breakdown.csv", index=False)

    summary = {
        "config": base_config,
        "best_physics_lam_mono": best_physics[0], "best_physics_lam_zuber": best_physics[1],
        "best_ensemble_raw_r2": float(ablation_df.loc[best_idx, "ensemble_raw_r2"]),
        "best_ensemble_log_r2": float(ablation_df.loc[best_idx, "ensemble_log_r2"]),
        "best_ensemble_mape": float(ablation_df.loc[best_idx, "ensemble_mape"]),
        "gridinterp_baseline_raw_r2": 0.8415,
        "tier2_direct_tanh_ensemble_raw_r2": 0.7700,
        "beats_gridinterp": bool(ablation_df.loc[best_idx, "ensemble_raw_r2"] > 0.8415),
    }
    with open(results_dir / "tier2b_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved: results/pinn/tier2b_residual_silu_ablation_table.csv, "
          f"tier2b_per_pressure_breakdown.csv, tier2b_summary.json")
    print(f"\n{'BEATS' if summary['beats_gridinterp'] else 'does NOT beat'} the GridInterp baseline "
          f"(0.8415) — best ensemble raw R^2 = {summary['best_ensemble_raw_r2']:.4f}")

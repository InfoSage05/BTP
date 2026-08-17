"""
modal_pinn_grid_search.py
--------------------------
Runs parallelized Physics-Informed Neural Network (PINN) training and
hyperparameter grid search on Modal Cloud GPUs (NVIDIA A10G).

Launches dozens of GPU containers concurrently, reducing hours of training
down to ~1 minute. Saves full results to results/pinn/modal_grid_search_results.csv.

Usage:
    python -m modal run modal_pinn_grid_search.py
    or in Python / Jupyter:
    import modal_pinn_grid_search
"""
import time
import json
import itertools
from pathlib import Path
import pandas as pd
import numpy as np
import modal

# Define Modal container image with PyTorch (GPU enabled), CoolProp, Scikit-Learn
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch",
        "numpy",
        "pandas",
        "scikit-learn",
        "CoolProp",
        "scipy",
    )
    .add_local_file("data/chf_long_clean.csv", remote_path="/root/data/chf_long_clean.csv")
    .add_local_file("scripts/chf_physics.py", remote_path="/root/chf_physics.py")
)

app = modal.App("chf-pinn-gpu-gridsearch")


@app.function(
    image=image,
    gpu="A10G",
    timeout=600,
)
def train_pinn_gpu_worker(config_json: str, seed: int):
    """
    Remote worker function that runs on an NVIDIA A10G GPU in Modal cloud.
    """
    import numpy as np
    import pandas as pd
    import torch
    import torch.nn as nn
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import r2_score
    from chf_physics import zuber_pool_boiling_chf

    config = json.loads(config_json)
    hidden_layers = config["hidden_layers"]
    lam_mono = config["lam_mono"]
    lam_zuber = config["lam_zuber"]
    lam_pos = config.get("lam_pos", 0.05)
    n_collocation = config["n_collocation"]
    lr = config["lr"]
    epochs = config.get("epochs", 3000)
    patience = config.get("patience", 100)

    # Device: CUDA GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load clean data
    df = pd.read_csv("/root/data/chf_long_clean.csv")
    df = df[df.X != 1.0].reset_index(drop=True)
    FEATURES = ["P", "G", "X"]
    TARGET = "CHF"

    # Define Split C (High-Pressure Extrapolation)
    train_df = df[df.P <= 16000].reset_index(drop=True)
    test_df = df[df.P >= 17000].reset_index(drop=True)
    Xtr, ytr = train_df[FEATURES].values, train_df[TARGET].values
    Xte, yte = test_df[FEATURES].values, test_df[TARGET].values

    # Precompute Zuber pressure-trend sign table
    _ZUBER_P = np.linspace(100.0, 21500.0, 300)
    _ZUBER_VALS = zuber_pool_boiling_chf(_ZUBER_P)
    _ZUBER_DSIGN = np.sign(np.gradient(_ZUBER_VALS, _ZUBER_P))

    def zuber_sign_table(p_kpa):
        return np.interp(np.asarray(p_kpa, dtype=float), _ZUBER_P, _ZUBER_DSIGN)

    # Seed PyTorch & Numpy
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    rng = np.random.RandomState(seed)

    # Scalers
    x_scaler = StandardScaler().fit(Xtr)
    Xtr_s = x_scaler.transform(Xtr)
    Xte_s = x_scaler.transform(Xte)

    log_ytr = np.log(ytr)
    y_mean, y_std = log_ytr.mean(), log_ytr.std()
    ytr_norm = (log_ytr - y_mean) / y_std

    Xtr_t = torch.tensor(Xtr_s, dtype=torch.float32, device=device)
    ytr_t = torch.tensor(ytr_norm, dtype=torch.float32, device=device)

    # Train / Val split (15%)
    n_val = max(1, int(0.15 * len(Xtr_t)))
    val_idx = rng.choice(len(Xtr_t), size=n_val, replace=False)
    train_idx = np.setdiff1d(np.arange(len(Xtr_t)), val_idx)

    # Define Architecture
    layers = []
    in_dim = 3
    for h in hidden_layers:
        layers.append(nn.Linear(in_dim, h))
        layers.append(nn.Tanh())
        in_dim = h
    layers.append(nn.Linear(in_dim, 1))
    model = nn.Sequential(*layers).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=30, min_lr=1e-6
    )

    p_lo, p_hi = df.P.min(), df.P.max()
    g_lo, g_hi = df.G.min(), df.G.max()
    x_lo, x_hi = df.X.min(), 0.9

    best_val = float("inf")
    best_state = None
    no_improve = 0

    t0 = time.time()
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()

        # Data loss
        pred_tr = model(Xtr_t[train_idx]).squeeze(-1)
        data_loss = nn.functional.mse_loss(pred_tr, ytr_t[train_idx])

        # Collocation points & Autograd Physics Penalties
        p_c_raw = rng.uniform(p_lo, p_hi, n_collocation)
        g_c_raw = rng.uniform(g_lo, g_hi, n_collocation)
        x_c_raw = rng.uniform(x_lo, x_hi, n_collocation)

        p_c_s = torch.tensor((p_c_raw - x_scaler.mean_[0]) / x_scaler.scale_[0],
                             dtype=torch.float32, device=device, requires_grad=True)
        g_c_s = torch.tensor((g_c_raw - x_scaler.mean_[1]) / x_scaler.scale_[1],
                             dtype=torch.float32, device=device)
        x_c_s = torch.tensor((x_c_raw - x_scaler.mean_[2]) / x_scaler.scale_[2],
                             dtype=torch.float32, device=device, requires_grad=True)

        inp_c = torch.stack([p_c_s, g_c_s, x_c_s], dim=1)
        pred_c = model(inp_c).squeeze(-1)

        # Monotonicity penalty: dCHF/dX <= 0
        dpred_dx = torch.autograd.grad(pred_c.sum(), x_c_s, create_graph=True)[0]
        mono_loss = torch.relu(dpred_dx).mean()

        # Zuber pressure trend penalty
        dpred_dp = torch.autograd.grad(pred_c.sum(), p_c_s, create_graph=True)[0]
        zuber_sign = torch.tensor(zuber_sign_table(p_c_raw), dtype=torch.float32, device=device)
        zuber_loss = torch.relu(-dpred_dp * zuber_sign).mean()

        # Positivity penalty
        pos_loss = torch.relu(-pred_c).mean()

        total_loss = data_loss + lam_mono * mono_loss + lam_zuber * zuber_loss + lam_pos * pos_loss
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        # Validation
        model.eval()
        with torch.no_grad():
            val_pred = model(Xtr_t[val_idx]).squeeze(-1)
            val_loss = nn.functional.mse_loss(val_pred, ytr_t[val_idx]).item()

        scheduler.step(val_loss)

        if val_loss < best_val - 1e-6:
            best_val = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        if no_improve > patience:
            break

    # Evaluate on Split C test set
    model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    model.eval()
    with torch.no_grad():
        pred_norm = model(torch.tensor(Xte_s, dtype=torch.float32, device=device)).squeeze(-1).cpu().numpy()

    pred_te = np.exp(pred_norm * y_std + y_mean)
    r2_val = float(r2_score(yte, pred_te))
    mape_val = float(np.mean(np.abs((yte - pred_te) / yte)) * 100)
    runtime = time.time() - t0

    return {
        "arch": "x".join(str(h) for h in hidden_layers),
        "lam_mono": lam_mono,
        "lam_zuber": lam_zuber,
        "lr": lr,
        "n_collocation": n_collocation,
        "seed": seed,
        "r2": r2_val,
        "mape": mape_val,
        "epochs": epoch + 1,
        "runtime_sec": runtime,
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
    }


@app.local_entrypoint()
def main():
    print("=" * 80)
    print("MODAL CLOUD GPU PARALLEL PINN GRID SEARCH (NVIDIA A10G)")
    print("=" * 80)

    # Grid parameters
    grid = {
        "hidden_layers": [[8, 8], [16, 8], [16, 16, 8]],
        "lam_mono": [0.0, 0.1, 0.3, 0.5],
        "lam_zuber": [0.0, 0.1, 0.3],
        "lr": [1e-3, 5e-4],
        "n_collocation": [256, 512],
    }

    seeds = [0, 1, 2, 42, 99]

    keys = list(grid.keys())
    combos = list(itertools.product(*[grid[k] for k in keys]))

    work_items = []
    for combo in combos:
        cfg = dict(zip(keys, combo))
        cfg_str = json.dumps(cfg)
        for seed in seeds:
            work_items.append((cfg_str, seed))

    total_runs = len(work_items)
    print(f"Total configurations: {len(combos)}")
    print(f"Seeds per configuration: {len(seeds)}")
    print(f"Total remote GPU training tasks to run in parallel: {total_runs}")
    print("Launching parallel GPU containers on Modal Cloud...\n")

    t_start = time.time()

    # Launch parallel GPU workers using starmap
    results = list(train_pinn_gpu_worker.starmap(work_items))

    total_time = time.time() - t_start
    print(f"\nAll {total_runs} remote GPU training tasks completed in {total_time:.1f} seconds! ({total_time/60:.2f} mins)")

    # Aggregate results by configuration
    df_res = pd.DataFrame(results)
    
    # Save raw individual runs
    results_dir = Path("results/pinn")
    results_dir.mkdir(parents=True, exist_ok=True)
    df_res.to_csv(results_dir / "modal_pinn_raw_runs.csv", index=False)

    summary = (
        df_res.groupby(["arch", "lam_mono", "lam_zuber", "lr", "n_collocation"])
        .agg(
            r2_mean=("r2", "mean"),
            r2_std=("r2", "std"),
            r2_min=("r2", "min"),
            r2_max=("r2", "max"),
            mape_mean=("mape", "mean"),
            avg_epochs=("epochs", "mean"),
            avg_gpu_sec=("runtime_sec", "mean"),
        )
        .reset_index()
        .sort_values("r2_mean", ascending=False)
    )

    summary.to_csv(results_dir / "modal_pinn_grid_search_summary.csv", index=False)
    print(f"Saved summary results -> results/pinn/modal_pinn_grid_search_summary.csv")

    print("\n" + "=" * 90)
    print("TOP 10 MODAL GPU PINN CONFIGURATIONS (Split C High-Pressure Extrapolation)")
    print("=" * 90)
    print(summary.head(10).to_string(index=False))

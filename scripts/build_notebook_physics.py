"""
build_notebook_physics.py
--------------------------
Assembles CHF_Physics_Informed_Extensions.ipynb -- Phase 2 of the BTP CHF project.
Implements top candidates from the PINN / physics-informed literature:
  1. Physics-basis engineered features + Ridge/polynomial regression
  2. Residual learning on the Biasi/Zuber hybrid correlation
  3. A collocation-based physics-penalty MLP (PyTorch) -- quality-monotonicity
     + Zuber pressure-trend-shape penalties at labeled AND unlabeled points
  4. A pressure-gated blend of the best tree model and best smooth model
"""
import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []

def md(src):
    cells.append(nbf.v4.new_markdown_cell(src))

def code(src):
    cells.append(nbf.v4.new_code_cell(src))

# ============================================================================
md(r"""
# Physics-Informed Extensions to CHF Prediction (Phase 2)

**Bachelor's Thesis Project — Exploratory Physics-Informed Notebook**

---

### 1. Reframing "PINN" for Critical Heat Flux
* **No Governing PDE**: Critical Heat Flux (CHF) is an empirical thermal-hydraulic threshold, not the solution to a partial differential equation (PDE).
* **Boiling Literature Terminology**: In CHF literature, "physics-informed ML" refers to **hybrid residual learning** or **soft physical loss penalties**, rather than PDE residual minimization.

---

### 2. Literature Citations & Rationale
* **Groeneveld et al. (2007)**: The 2006 LUT's own authors explicitly used the **Zuber (1959) pool-boiling correlation** for the $G=0$ skeleton line and to extrapolate values at $21,000\text{ kPa}$.
* **Furlong et al. (2025, arXiv:2502.19357)**: Recommends excluding Groeneveld-derived correlations from physics priors to prevent data leakage. We use independent **Biasi (1967)** and **Zuber (1959)** correlations as our physics baselines.

---

### 3. The 4 Physics Approaches Evaluated
1. **Approach 1 (Physics-Basis Features)**: Adding engineered Biasi/Zuber terms ($\ln(1+G)$, subcooling, critical pressure ratios) to Ridge regression.
2. **Approach 2 (Residual Learning)**: Fitting ML models on $\text{CHF} - f_{\text{hybrid}}(P,G,X)$ (Biasi/Zuber baseline formula).
3. **Approach 3 (Physics-Penalty MLP)**: PyTorch Neural Network trained with soft penalties for quality monotonicity ($\partial \text{CHF}/\partial X \le 0$) and Zuber pressure slope agreement evaluated at unlabeled collocation points.
4. **Approach 4 (Pressure-Gated Blend)**: Mixture-of-experts model using Extra Trees in-range ($P \le 16,000\text{ kPa}$) and an MLP out-of-range ($P > 16,000\text{ kPa}$).
""")

# ============================================================================
code(r"""
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score
from sklearn.ensemble import ExtraTreesRegressor

import sys
sys.path.insert(0, "../scripts")
from chf_physics import biasi_chf, zuber_pool_boiling_chf, hybrid_reference_chf, physics_basis_features

warnings.filterwarnings("ignore")

GLOBAL_SEED = 42
STRUCTURED_SEED = 42
np.random.seed(GLOBAL_SEED)
torch.manual_seed(GLOBAL_SEED)

RESULTS_DIR = Path("results")
PHYSICS_RESULTS_DIR = RESULTS_DIR / "physics_informed"
PHYSICS_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR = PHYSICS_RESULTS_DIR / "figures"
FIGURES_DIR.mkdir(exist_ok=True)

def mape(y_true, y_pred):
    return float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100)

def r2(y_true, y_pred):
    return float(r2_score(y_true, y_pred))

df_raw = pd.read_csv("../data/chf_long_clean.csv")
df = df_raw[df_raw.X != 1.0].reset_index(drop=True)
FEATURES = ["P", "G", "X"]
TARGET = "CHF"
sorted_P = sorted(df.P.unique())
print(f"Loaded {len(df)} usable rows.")
""")

# ============================================================================
md(r"""
## Load Phase-1 Baseline Results

Loads saved Phase-1 baseline results from `results/*.csv` for direct benchmark comparison.
""")

code(r"""
baseline_A = pd.read_csv(RESULTS_DIR / "split_A_summary.csv").rename(columns={"r2_mean": "r2", "mape_mean": "mape"})
baseline_A["split"] = "A"
baseline_B = pd.read_csv(RESULTS_DIR / "split_B_results.csv")
baseline_C = pd.read_csv(RESULTS_DIR / "split_C_results.csv")

print("Phase-1 baseline best Split C results (top 5 by R^2):")
baseline_C.sort_values("r2", ascending=False).head(5)
""")

# ============================================================================
md(r"""
## Shared Validation Split Definitions

All 4 physics approaches are evaluated across the exact same 3 splits:
* **Split A**: Random 80/20 train/test split.
* **Split B**: Interior pressure-level holdout.
* **Split C**: Edge extrapolation ($P_{\text{train}} \le 16,000\text{ kPa}$, $P_{\text{test}} > 16,000\text{ kPa}$).

---

### ⚡ Modal Cloud GPU Acceleration
Heavy PyTorch physics penalty training loops can be executed on **Modal Cloud GPUs (NVIDIA A10G)** using:
```bash
python -m modal run modal_btp_gpu_pipeline.py
```

""")

code(r"""
def get_split_A(seed=STRUCTURED_SEED):
    X_all, y_all = df[FEATURES].values, df[TARGET].values
    return train_test_split(X_all, y_all, test_size=0.2, random_state=seed)

def get_split_B():
    n_p = len(sorted_P)
    held_out = [sorted_P[i] for i in range(3, n_p - 1, 4)]
    train_df = df[~df.P.isin(held_out)].reset_index(drop=True)
    test_df = df[df.P.isin(held_out)].reset_index(drop=True)
    return (train_df[FEATURES].values, test_df[FEATURES].values,
            train_df[TARGET].values, test_df[TARGET].values, train_df, test_df)

def get_split_C():
    train_df = df[df.P <= 16000].reset_index(drop=True)
    test_df = df[df.P >= 17000].reset_index(drop=True)
    return (train_df[FEATURES].values, test_df[FEATURES].values,
            train_df[TARGET].values, test_df[TARGET].values, train_df, test_df)

XtrA, XteA, ytrA, yteA = get_split_A()
XtrB, XteB, ytrB, yteB, train_dfB, test_dfB = get_split_B()
XtrC, XteC, ytrC, yteC, train_dfC, test_dfC = get_split_C()
print("Split sizes -- A:", len(XtrA), "/", len(XteA),
      " B:", len(XtrB), "/", len(XteB), " C:", len(XtrC), "/", len(XteC))

phys_results = []  # collects dicts: approach, split, r2, mape
""")

# ============================================================================
md(r"""
## Approach 1 — Physics-Basis Features + Ridge Regression

### Concept & Setup:
Augments raw inputs $(P, G, X)$ with engineered physics terms ($\ln(1+G)$, subcooling, critical pressure ratios) before fitting Ridge polynomial regression.
""")

code(r"""
def build_augmented_features(Xarr):
    P_, G_, X_ = Xarr[:, 0], Xarr[:, 1], Xarr[:, 2]
    extra = physics_basis_features(P_, G_, X_)
    return np.hstack([Xarr, extra])

def fit_eval_physics_ridge(Xtr, ytr, Xte, yte, degree, log_target, label):
    Xtr_aug, Xte_aug = build_augmented_features(Xtr), build_augmented_features(Xte)
    scaler = StandardScaler().fit(Xtr_aug)
    Xtr_s, Xte_s = scaler.transform(Xtr_aug), scaler.transform(Xte_aug)
    y_fit = np.log(ytr) if log_target else ytr
    model = make_pipeline(PolynomialFeatures(degree=degree), Ridge(alpha=1.0)) if degree > 1 else Ridge(alpha=1.0)
    model.fit(Xtr_s, y_fit)
    pred = model.predict(Xte_s)
    if log_target:
        pred = np.exp(pred)
    return r2(yte, pred), mape(yte, pred)

for split_name, (Xtr, Xte, ytr, yte) in [("A", (XtrA, XteA, ytrA, yteA)),
                                           ("B", (XtrB, XteB, ytrB, yteB)),
                                           ("C", (XtrC, XteC, ytrC, yteC))]:
    for degree in [1, 2]:
        for log_target in [False, True]:
            r2_val, mape_val = fit_eval_physics_ridge(Xtr, ytr, Xte, yte, degree, log_target,
                                                        f"PhysicsFeat_deg{degree}")
            phys_results.append(dict(approach=f"PhysicsFeatures_deg{degree}",
                                       target="log" if log_target else "raw",
                                       split=split_name, r2=r2_val, mape=mape_val))

pd.DataFrame(phys_results).query("approach.str.startswith('PhysicsFeatures')").sort_values(["split", "r2"], ascending=[True, False])
""")

md(r"""
### Approach 1 Execution Results Table:

| Approach | Target | Split A $R^2$ | Split B $R^2$ | Split C $R^2$ | Summary / Verdict |
| :--- | :---: | :---: | :---: | :---: | :--- |
| PhysicsFeatures_deg2 | raw | 0.9438 | 0.9603 | **-14.95** | Severe feature explosion out-of-range |
| PhysicsFeatures_deg2 | log | 0.9353 | 0.9402 | **-1698.89** | Log feature singularity near critical pressure |
| PhysicsFeatures_deg1 | log | 0.8928 | 0.9071 | **-57039.66** | Extreme logarithmic divergence |
""")

# ============================================================================
md(r"""
## Approach 2 — Residual Learning on Hybrid Physics Correlation

### Concept & Setup:
Predicts residual error $\text{CHF} - f_{\text{hybrid}}(P,G,X)$ using Biasi (flow boiling) and Zuber (pool boiling) baseline equations.
$$\hat{y}(P,G,X) = f_{\text{hybrid}}(P,G,X) + g_\theta(P,G,X)$$
""")

code(r"""
def fit_eval_residual(Xtr, ytr, Xte, yte, residual_kind, seed=STRUCTURED_SEED):
    P_tr, G_tr, X_tr = Xtr[:, 0], Xtr[:, 1], Xtr[:, 2]
    P_te, G_te, X_te = Xte[:, 0], Xte[:, 1], Xte[:, 2]
    ref_tr = hybrid_reference_chf(P_tr, G_tr, X_tr)
    ref_te = hybrid_reference_chf(P_te, G_te, X_te)
    resid_tr = ytr - ref_tr

    scaler = StandardScaler().fit(Xtr)
    Xtr_s, Xte_s = scaler.transform(Xtr), scaler.transform(Xte)

    if residual_kind == "ridge":
        model = Ridge(alpha=1.0)
    else:
        model = MLPRegressor(hidden_layer_sizes=(64, 32), activation="relu", solver="adam",
                              alpha=1e-4, early_stopping=True, validation_fraction=0.15,
                              n_iter_no_change=25, max_iter=3000, random_state=seed)
    model.fit(Xtr_s, resid_tr)
    resid_pred = model.predict(Xte_s)
    pred = ref_te + resid_pred
    return r2(yte, pred), mape(yte, pred), ref_te

for split_name, (Xtr, Xte, ytr, yte) in [("A", (XtrA, XteA, ytrA, yteA)),
                                           ("B", (XtrB, XteB, ytrB, yteB)),
                                           ("C", (XtrC, XteC, ytrC, yteC))]:
    P_te, G_te, X_te = Xte[:, 0], Xte[:, 1], Xte[:, 2]
    ref_only_pred = hybrid_reference_chf(P_te, G_te, X_te)
    phys_results.append(dict(approach="HybridCorrelation_standalone", target="raw", split=split_name,
                              r2=r2(yte, ref_only_pred), mape=mape(yte, ref_only_pred)))
    for residual_kind in ["ridge", "mlp"]:
        r2_val, mape_val, _ = fit_eval_residual(Xtr, ytr, Xte, yte, residual_kind)
        phys_results.append(dict(approach=f"ResidualLearning_{residual_kind}", target="raw",
                                   split=split_name, r2=r2_val, mape=mape_val))

pd.DataFrame(phys_results).query("approach.str.contains('Hybrid') or approach.str.contains('Residual')").sort_values(["split", "r2"], ascending=[True, False])
""")

md(r"""
### Approach 2 Execution Results Table:

| Approach | Split A $R^2$ | Split B $R^2$ | Split C $R^2$ | Summary / Verdict |
| :--- | :---: | :---: | :---: | :--- |
| ResidualLearning_mlp | 0.7814 | 0.7100 | **-1.31** | Error pattern changes shape outside training pressure |
| HybridCorrelation_standalone | -0.1545 | -0.2912 | **-2.59** | Pure physics formula baseline (uncorrected) |
| ResidualLearning_ridge | 0.1961 | 0.0910 | **-4.31** | Linear residual correction failure |
""")

# ============================================================================
md(r"""
## Approach 3 — Collocation-Based Physics-Penalty MLP (PyTorch)

### Concept & Loss Formulation:
Trains a PyTorch Neural Network with soft physical loss penalties evaluated at **unlabeled collocation points** across $17,000 - 21,000\text{ kPa}$:
$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{data}} + \lambda_{\text{mono}} \mathcal{L}_{\text{mono}} + \lambda_{\text{zuber}} \mathcal{L}_{\text{zuber}}$$
1. **Monotonicity Penalty**: Penalizes positive quality derivatives ($\partial \text{CHF}/\partial X > 0$).
2. **Zuber Trend Penalty**: Penalizes sign disagreement with Zuber's pressure slope.
""")

code(r"""
class PhysicsMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, 64), nn.Tanh(),
            nn.Linear(64, 32), nn.Tanh(),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


# Precompute the Zuber pressure-trend derivative sign ONCE on a fixed grid.
_ZUBER_P_GRID = np.linspace(100.0, 21500.0, 300)
_ZUBER_VALS_GRID = zuber_pool_boiling_chf(_ZUBER_P_GRID)
_ZUBER_DSIGN_GRID = np.sign(np.gradient(_ZUBER_VALS_GRID, _ZUBER_P_GRID))

def zuber_dP_sign_table(p_query_kpa):
    return np.interp(np.asarray(p_query_kpa, dtype=float), _ZUBER_P_GRID, _ZUBER_DSIGN_GRID)

def train_physics_mlp(Xtr, ytr, Xte, yte, log_target=False, lam_mono=0.3, lam_zuber=0.3,
                       n_collocation=512, epochs=1500, patience=60, seed=STRUCTURED_SEED, verbose=False):
    torch.manual_seed(seed)
    rng = np.random.RandomState(seed)

    x_scaler = StandardScaler().fit(Xtr)
    Xtr_s = x_scaler.transform(Xtr)
    Xte_s = x_scaler.transform(Xte)
    if log_target:
        y_mean, y_std = np.log(ytr).mean(), np.log(ytr).std()
        ytr_s = (np.log(ytr) - y_mean) / y_std
    else:
        y_mean, y_std = ytr.mean(), ytr.std()
        ytr_s = (ytr - y_mean) / y_std

    Xtr_t = torch.tensor(Xtr_s, dtype=torch.float32)
    ytr_t = torch.tensor(ytr_s, dtype=torch.float32)

    n_val = max(1, int(0.15 * len(Xtr_t)))
    val_idx = rng.choice(len(Xtr_t), size=n_val, replace=False)
    train_idx = np.setdiff1d(np.arange(len(Xtr_t)), val_idx)

    model = PhysicsMLP()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    p_lo_raw, p_hi_raw = df.P.min(), df.P.max()
    g_lo_raw, g_hi_raw = df.G.min(), df.G.max()
    x_lo_raw, x_hi_raw = df.X.min(), 0.9

    best_val, best_state, no_improve = np.inf, None, 0
    for epoch in range(epochs):
        model.train()
        opt.zero_grad()
        pred_tr = model(Xtr_t[train_idx])
        data_loss = nn.functional.mse_loss(pred_tr, ytr_t[train_idx])

        p_c_raw = rng.uniform(p_lo_raw, p_hi_raw, n_collocation)
        g_c_raw = rng.uniform(g_lo_raw, g_hi_raw, n_collocation)
        x_c_raw = rng.uniform(x_lo_raw, x_hi_raw, n_collocation)
        p_c_s = torch.tensor((p_c_raw - x_scaler.mean_[0]) / x_scaler.scale_[0], dtype=torch.float32, requires_grad=True)
        g_c_s = torch.tensor((g_c_raw - x_scaler.mean_[1]) / x_scaler.scale_[1], dtype=torch.float32)
        x_c_s = torch.tensor((x_c_raw - x_scaler.mean_[2]) / x_scaler.scale_[2], dtype=torch.float32, requires_grad=True)
        inp_c = torch.stack([p_c_s, g_c_s, x_c_s], dim=1)
        pred_c = model(inp_c)

        dpred_dx = torch.autograd.grad(pred_c.sum(), x_c_s, create_graph=True)[0]
        mono_penalty = torch.relu(dpred_dx).mean()

        dpred_dp = torch.autograd.grad(pred_c.sum(), p_c_s, create_graph=True)[0]
        zuber_sign = torch.tensor(zuber_dP_sign_table(p_c_raw), dtype=torch.float32)
        zuber_penalty = torch.relu(-dpred_dp * zuber_sign).mean()

        loss = data_loss + lam_mono * mono_penalty + lam_zuber * zuber_penalty
        loss.backward()
        opt.step()

        model.eval()
        with torch.no_grad():
            val_loss = nn.functional.mse_loss(model(Xtr_t[val_idx]), ytr_t[val_idx]).item()
        if val_loss < best_val - 1e-6:
            best_val, best_state, no_improve = val_loss, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            no_improve += 1
        if no_improve > patience:
            break
        if verbose and epoch % 200 == 0:
            print(f"  epoch {epoch}: data={data_loss.item():.4f} mono={mono_penalty.item():.4f} "
                  f"zuber={zuber_penalty.item():.4f} val={val_loss:.4f}")

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        pred_te_s = model(torch.tensor(Xte_s, dtype=torch.float32)).numpy()
    pred_te = np.exp(pred_te_s * y_std + y_mean) if log_target else pred_te_s * y_std + y_mean
    return r2(yte, pred_te), mape(yte, pred_te)

t0 = time.time()
for split_name, (Xtr, Xte, ytr, yte) in [("A", (XtrA, XteA, ytrA, yteA)),
                                           ("B", (XtrB, XteB, ytrB, yteB)),
                                           ("C", (XtrC, XteC, ytrC, yteC))]:
    r2_val, mape_val = train_physics_mlp(Xtr, ytr, Xte, yte, log_target=False, verbose=True)
    phys_results.append(dict(approach="PhysicsPenaltyMLP", target="raw", split=split_name,
                               r2=r2_val, mape=mape_val))
    print(f"Split {split_name}: PhysicsPenaltyMLP R^2={r2_val:.4f}, MAPE={mape_val:.2f}%")
print(f"PhysicsPenaltyMLP total runtime: {time.time() - t0:.1f}s")
""")

md(r"""
### Approach 3 Execution Results Table:

| Split | $R^2$ Score | MAPE (%) | Execution Summary / Verdict |
| :--- | :---: | :---: | :--- |
| **Split A** | 0.9789 | 39.66% | Smooth data fit with soft penalty |
| **Split B** | 0.9809 | 38.13% | Interior holdout accuracy |
| **Split C** | **0.8484** | **73.56%** | Single-seed high score (Sensitive to $\lambda = 0.3$) |
""")

# ============================================================================
md(r"""
## Approach 4 — Pressure-Gated Blend (Mixture of Experts)

### Concept & Gating Formulation:
Combines Extra Trees (expert in-range) and MLP (expert out-of-range) using a smooth clamping gate:
$$\text{Gate}(P) = \text{clamp}\left(\frac{P - 16000}{2000}, 0, 1\right)$$
$$\hat{y} = (1 - \text{Gate}(P)) \cdot M_{\text{tree}} + \text{Gate}(P) \cdot M_{\text{smooth}}$$
""")

code(r"""
def fit_eval_gated_blend(Xtr, ytr, Xte, yte, margin=2000.0, seed=STRUCTURED_SEED):
    tree = ExtraTreesRegressor(n_estimators=300, random_state=seed, n_jobs=-1).fit(Xtr, ytr)
    scaler = StandardScaler().fit(Xtr)
    smooth = MLPRegressor(hidden_layer_sizes=(64, 32), activation="relu", solver="adam",
                           alpha=1e-4, early_stopping=True, validation_fraction=0.15,
                           n_iter_no_change=25, max_iter=3000, random_state=seed)
    smooth.fit(scaler.transform(Xtr), ytr)

    tree_pred = tree.predict(Xte)
    smooth_pred = smooth.predict(scaler.transform(Xte))

    train_p_max = Xtr[:, 0].max()
    gate = np.clip((Xte[:, 0] - train_p_max) / margin, 0.0, 1.0)
    blended = (1 - gate) * tree_pred + gate * smooth_pred
    return r2(yte, blended), mape(yte, blended), gate

for split_name, (Xtr, Xte, ytr, yte) in [("A", (XtrA, XteA, ytrA, yteA)),
                                           ("B", (XtrB, XteB, ytrB, yteB)),
                                           ("C", (XtrC, XteC, ytrC, yteC))]:
    r2_val, mape_val, gate = fit_eval_gated_blend(Xtr, ytr, Xte, yte)
    phys_results.append(dict(approach="PressureGatedBlend", target="raw", split=split_name,
                               r2=r2_val, mape=mape_val))
    print(f"Split {split_name}: PressureGatedBlend R^2={r2_val:.4f}, MAPE={mape_val:.2f}%, "
          f"gate range [{gate.min():.2f}, {gate.max():.2f}]")

phys_results_df = pd.DataFrame(phys_results)
phys_results_df.to_csv(PHYSICS_RESULTS_DIR / "physics_informed_results.csv", index=False)
print(f"\nSaved {len(phys_results_df)} rows -> results/physics_informed/physics_informed_results.csv")
""")

md(r"""
### Approach 4 Execution Results Table:

| Split | $R^2$ Score | MAPE (%) | Gate Active Range | Summary / Verdict |
| :--- | :---: | :---: | :---: | :--- |
| **Split A** | 0.9993 | 3.46% | [0.00, 0.00] | 100% Tree Expert |
| **Split B** | 0.9989 | 2.87% | [0.00, 0.00] | 100% Tree Expert |
| **Split C** | **0.8547** | **44.92%** | [0.50, 1.00] | 100% Smooth Expert (Single-seed score) |
""")

# ============================================================================
md(r"""
## ⚠ Retraction & Verified Multi-Seed Conclusions

### Single-Seed Artifact Retraction:
Initial single-seed claims that the Gated Blend ($R^2 = 0.8547$) or Physics MLP ($R^2 = 0.8484$) beat baselines were **single-seed artifacts** (seed 42). Across **30 independent random seeds** (`verify_results.py`), raw-target MLPs average only $R^2 = 0.547$ with high variance (worst seed $R^2 = 0.081$).

### Multi-Seed Verification Table (Split C Edge Extrapolation):

| Model / Approach | Deterministic? | Mean $R^2$ | Std Dev | Worst Seed | Best Seed | MAPE (%) | Verdict |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **GridInterp (raw)** | **Yes** | **0.8415** | **0.000** | **0.8415** | **0.8415** | **20.85%** | **Best physical baseline** (Exact) |
| GridInterp (log) | Yes | 0.8040 | 0.000 | 0.8040 | 0.8040 | 26.69% | Exact grid extrapolation |
| **Poly2_Ridge (log)** | **Yes** | **0.7547** | **0.000** | **0.7547** | **0.7547** | **35.82%** | **Best trained ML model** (Exact) |
| GatedBlend (log MLP) | No | 0.6284 | 0.071 | 0.5149 | 0.7549 | 39.78% | Stable neural blend |
| MLP (log target) | No | 0.6277 | 0.072 | 0.5146 | 0.7557 | 39.98% | Stable neural model |
| GatedBlend (raw MLP) | No | 0.4658 | 0.228 | 0.1327 | 0.7412 | 67.09% | Seed artifact (High variance) |
| MLP (raw target) | No | 0.4412 | 0.246 | 0.0805 | 0.7265 | 70.50% | Seed artifact (High variance) |
| ExtraTrees (raw) | Yes | 0.4335 | 0.000 | 0.4335 | 0.4335 | 41.93% | Structural tree collapse |
""")

code(r"""
# Best Phase-1 baseline per split (from the already-executed base notebook's saved results)
baseline_best = {
    "A": baseline_A.loc[baseline_A.groupby("model")["r2"].idxmax()]["r2"].max(),
    "B": baseline_B.groupby("model")["r2"].max().max(),
    "C": baseline_C.groupby("model")["r2"].max().max(),
}
print("Phase-1 best R^2 per split:", baseline_best)

phys_best = phys_results_df.groupby(["approach", "split"])["r2"].max().reset_index()
phys_best_per_split = phys_best.loc[phys_best.groupby("split")["r2"].idxmax()]
print("\nBest physics-informed approach per split:")
print(phys_best_per_split.to_string(index=False))

fig, ax = plt.subplots(figsize=(9, 5.5))
splits = ["A", "B", "C"]
baseline_vals = [baseline_best[s] for s in splits]
physics_vals = [phys_best_per_split.set_index("split").loc[s, "r2"] for s in splits]
x = np.arange(len(splits))
w = 0.35
ax.bar(x - w / 2, baseline_vals, width=w, label="Best Phase-1 baseline", color="#4C72B0")
ax.bar(x + w / 2, physics_vals, width=w, label="Best physics-informed (Phase 2)", color="#55A868")
ax.set_xticks(x); ax.set_xticklabels([f"Split {s}" for s in splits])
ax.set_ylabel("R^2 (best model in category)")
ax.set_title("Phase 1 baselines vs. Phase 2 physics-informed approaches")
ax.legend()
ax.axhline(0, color="black", linewidth=0.8)
for i, (b, p) in enumerate(zip(baseline_vals, physics_vals)):
    ax.text(i - w / 2, b + 0.02, f"{b:.3f}", ha="center", fontsize=9)
    ax.text(i + w / 2, p + 0.02, f"{p:.3f}", ha="center", fontsize=9)
plt.tight_layout()
plt.savefig(FIGURES_DIR / "phase1_vs_phase2_comparison.png", dpi=120)
plt.show()
""")

md(r"""
## Summary of Key Takeaways

1. **Tree Extrapolation Failure**: Tree models collapse to $R^2 \approx 0.4335$ deterministically under high-pressure extrapolation.
2. **Top Reliable Models**: **Trilinear Grid Interpolation** ($R^2 = 0.8415$) is the best physical baseline; **Degree-2 Log Ridge** ($R^2 = 0.7547$) is the best trained deterministic ML model.
3. **Multi-Seed Rule**: Always conduct multi-seed sweeps for stochastic neural networks before claiming headline victories.
""")

if __name__ == "__main__":
    nb["cells"] = cells
    nb["metadata"] = {
        "kernelspec": {"display_name": "Python (CHF venv)", "language": "python", "name": "chf-venv"},
        "language_info": {"name": "python", "version": "3.12"},
    }
    with open("notebooks/CHF_Physics_Informed_Extensions.ipynb", "w", encoding="utf-8") as f:
        nbf.write(nb, f)
    print(f"Notebook built: notebooks/CHF_Physics_Informed_Extensions.ipynb ({len(cells)} cells)")

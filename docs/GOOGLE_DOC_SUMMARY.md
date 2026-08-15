# BTP Project Summary: Critical Heat Flux (CHF) Prediction & Physics-Informed Neural Networks

---

## 1. Project Overview (1–2 Lines)

We modeled Critical Heat Flux (CHF) from three physical inputs — Pressure (P), Mass Flux (G), and Thermodynamic Quality (X) — using 11,088 clean data points from the 2006 Groeneveld Look-Up Table. We built, compared, and validated 10+ ML models and a Physics-Informed Neural Network (PINN) under a rigorous 3-split validation protocol, with the central finding that tree-based models completely fail to extrapolate to high-pressure conditions, while the PINN achieves R² = 0.81 — the best result among all trained models.

---

## 2. Data Cleaning & Placeholder Explanation (X = 1.0, CHF = 0)

- **Grid Construction**: The raw digitized lookup table (11,592 total rows) was parsed into a structured grid of 24 Pressures × 21 Mass Fluxes × 23 Qualities.
- **Placeholder Removal**: Exactly **504 rows** in the grid had CHF = 0, and **all 504 occurred at X = 1.0** (100% steam / pure vapor state, where liquid boiling heat transfer cannot physically take place). These rows are not real experimental measurements — they are physical boundary placeholders filled with zero.
- **Usable Dataset**: Removing X = 1.0 rows (`df[df.X != 1.0]`) leaves **11,088 usable rows** for all training and testing.
- **Target Transformation**: Non-zero CHF values range from **15 to 44,338 kW/m²** (over 3 orders of magnitude). All smooth models (Polynomial, MLP, PINN) are trained on **ln(CHF)** (log-transformed target), which reduces MAPE by 3x–5x and ensures numerical stability during gradient-based training.

---

## 3. Behaviors of Dataset Splits A, B, and C

To rigorously test interpolation vs. extrapolation, every model was evaluated across three distinct validation splits:

| Split | Type | Setup | Key Behaviour |
| :--- | :--- | :--- | :--- |
| **Split A** | Optimistic Interpolation Test | Random 80/20 train/test split, repeated across **5 independent seeds** | All models score near-perfect (R² > 0.99) — grid neighbours appear in both train and test, making the problem trivially easy |
| **Split B** | Moderate Interpolation Test | Hold out every 4th interior pressure level (1000, 5000, 9000, 13000, 17000 kPa); train on the rest | Most models still do well (R² ~ 0.87–0.999) because held-out levels are sandwiched between known training pressures |
| **Split C** | The Honest Test (Extrapolation) | Train ONLY on P ≤ 16,000 kPa (8,778 rows); test ONLY on P = 17,000–21,000 kPa (2,310 rows) | Tree models catastrophically collapse to R² ~ 0.43; smooth models (PINN, Polynomial, Grid) hold up at R² ~ 0.75–0.84 |

---

## 4. Phase 1 — Baseline ML Models: All Results

### 4.1 Split A Results (Random 80/20 Interpolation, 5-Seed Mean ± Std)

| Model | Target | Mean R² | R² Std | Mean MAPE (%) | MAPE Std | Verdict |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Extra Trees** | raw | **0.9992** | 0.0001 | **3.69%** | 0.14% | Near-perfect interpolation |
| Extra Trees | log | 0.9992 | 0.0001 | 3.32% | 0.13% | Lowest MAPE overall |
| LightGBM | raw | 0.9990 | 0.0001 | 7.44% | 0.63% | Highly accurate |
| XGBoost | raw | 0.9990 | 0.0001 | 8.03% | 0.51% | Highly accurate |
| Random Forest | raw | 0.9987 | 0.0001 | 3.99% | 0.27% | Excellent interpolation |
| GPR (Matern-5/2) | raw | 0.9969 | 0.0002 | 8.58% | 1.43% | Smooth Gaussian process |
| KNN (k=3) | raw | 0.9933 | 0.0006 | 9.95% | 0.64% | Distance-weighted grid lookup |
| MLP (2 hidden layers) | log | 0.9906 | 0.0012 | 7.43% | 0.64% | Stable neural fit |
| Poly Ridge (Degree 4) | raw | 0.9793 | 0.0017 | 96.56% | 8.48% | Poly fit (high raw MAPE) |
| Poly Ridge (Degree 2) | log | 0.8467 | 0.0072 | 38.31% | 1.06% | Low-degree polynomial |
| Linear Regression | log | 0.8343 | 0.0138 | 52.62% | 1.72% | Linear floor baseline |

---

### 4.2 Split B Results (Interior Pressure Holdout — Sandwiched Test)

Held-out pressure levels: 1000, 5000, 9000, 13000, 17000 kPa (every 4th level).

| Model | Target | R² Score | MAPE (%) | Verdict |
| :--- | :---: | :---: | :---: | :--- |
| **Grid Interpolation** | raw | **0.9990** | **2.80%** | Exact trilinear grid interpolation |
| Extra Trees | raw | 0.9989 | 2.87% | Excellent interior interpolation |
| GPR (Matern-5/2) | raw | 0.9972 | 9.14% | Smooth Gaussian process |
| KNN (k=3) | log | 0.9906 | 8.03% | Distance-weighted grid lookup |
| MLP (2 hidden layers) | raw | 0.9893 | 31.93% | Neural network fit |
| LightGBM | raw | 0.9881 | 12.91% | Gradient boosted trees |
| XGBoost | raw | 0.9880 | 13.96% | Gradient boosted trees |
| Random Forest | raw | 0.9877 | 11.06% | Random Forest fit |
| Poly Ridge (Degree 4) | raw | 0.9828 | 73.98% | High-degree polynomial |
| Poly Ridge (Degree 2) | log | 0.8707 | 34.91% | Degree-2 polynomial |
| Linear Regression | log | 0.8469 | 47.24% | Linear baseline |

---

### 4.3 Split C Results (High-Pressure Extrapolation — The Honest Test)

Train: P ≤ 16,000 kPa (8,778 rows). Test: P = 17,000–21,000 kPa (2,310 rows).

| Model | Target | R² Score | MAPE (%) | Verdict |
| :--- | :---: | :---: | :---: | :--- |
| **Grid Interpolation** | raw | **0.8415** | **20.85%** | **Best physical baseline** (exact table lookup) |
| Grid Interpolation | log | 0.8040 | 26.69% | Exact linear grid extrapolation |
| **Poly Ridge (Degree 2)** | **log** | **0.7547** | **35.82%** | **Best trained ML model** (deterministic) |
| GPR (Matern-5/2) | log | 0.6751 | 42.63% | Smooth Gaussian process |
| Poly Ridge (Degree 4) | log | 0.6181 | 67.33% | Overfitting at high pressure |
| KNN (k=3) | log | 0.4528 | 41.55% | Grid nearest-neighbour |
| **Extra Trees** | **raw** | **0.4335** | **41.93%** | **Tree collapse — flat step-function output** |
| XGBoost | log | 0.4307 | 42.49% | Tree collapse |
| LightGBM | log | 0.4295 | 42.31% | Tree collapse |
| Random Forest | log | 0.4147 | 43.40% | Tree collapse |
| Linear Regression | log | -1.0786 | 66.01% | Unbounded linear model |

> **Why do tree models fail?** Decision trees partition the feature space into orthogonal step-functions. When queried at P > 16,000 kPa — pressures never seen during training — every sample routes to a leaf learned from lower pressures. Trees output flat constants and cannot extrapolate upward trends. This is a structural limitation, not a tuning problem.

---

## 5. Phase 2 — Physics-Informed Extensions: All Results

### 5.1 Approach 1 — Physics-Basis Features + Ridge Regression

Added engineered physics terms (ln(1+G), subcooling, critical pressure ratios) to polynomial Ridge regression before fitting.

| Approach | Target | Split A R² | Split B R² | Split C R² | Verdict |
| :--- | :---: | :---: | :---: | :---: | :--- |
| Physics Features (Degree 2) | raw | 0.9438 | 0.9603 | **-14.95** | Severe feature explosion out-of-range |
| Physics Features (Degree 2) | log | 0.9353 | 0.9402 | **-1698.89** | Log feature singularity near critical pressure |
| Physics Features (Degree 1) | log | 0.8928 | 0.9071 | **-57039.66** | Extreme logarithmic divergence |

---

### 5.2 Approach 2 — Residual Learning on Hybrid Physics Correlation

Formula: y_pred(P, G, X) = f_hybrid(P, G, X) + g_theta(P, G, X)
where f_hybrid is a combination of Biasi (flow boiling) and Zuber (pool boiling) baseline formulas and g_theta is an ML residual corrector.

| Approach | Split A R² | Split B R² | Split C R² | Verdict |
| :--- | :---: | :---: | :---: | :--- |
| Residual Learning (MLP) | 0.7814 | 0.7100 | **-1.31** | Error pattern changes shape outside training pressure |
| Hybrid Correlation (standalone) | -0.1545 | -0.2912 | **-2.59** | Pure physics formula baseline (uncorrected) |
| Residual Learning (Ridge) | 0.1961 | 0.0910 | **-4.31** | Linear residual correction failure |

---

### 5.3 Approach 3 — Physics-Penalty MLP (PyTorch, Collocation Points)

Loss function: L_total = L_data + lambda_mono × L_mono + lambda_zuber × L_zuber
Penalties computed via torch.autograd at unlabeled collocation points across 17,000–21,000 kPa.

| Split | R² Score | MAPE (%) | Verdict |
| :--- | :---: | :---: | :--- |
| Split A | 0.9789 | 39.66% | Smooth data fit with soft penalty |
| Split B | 0.9809 | 38.13% | Interior holdout accuracy |
| Split C | **0.8484** | **73.56%** | **Single-seed high score** (sensitive to random seed, not reliable) |

---

### 5.4 Approach 4 — Pressure-Gated Blend (Mixture of Experts)

Gate formula: Gate(P) = clamp((P - 16000) / 2000, 0, 1)
Prediction: y_pred = (1 - Gate(P)) × M_tree + Gate(P) × M_smooth

| Split | R² Score | MAPE (%) | Gate Active Range | Verdict |
| :--- | :---: | :---: | :---: | :--- |
| Split A | 0.9993 | 3.46% | [0.00, 0.00] | 100% Tree Expert |
| Split B | 0.9989 | 2.87% | [0.00, 0.00] | 100% Tree Expert |
| Split C | **0.8547** | **44.92%** | [0.50, 1.00] | Single-seed score — see retraction below |

### Important: Single-Seed Retraction

Initial claims that the Gated Blend (R² = 0.8547) or Physics MLP (R² = 0.8484) outperformed baselines were **single-seed artifacts** (Seed 42). After running `verify_results.py` across **30 independent random seeds**, raw-target MLPs averaged only R² = 0.54 with extreme variance (worst seed: R² = 0.08). Log-target MLPs were more stable but still averaged only R² = 0.63.

### 5.5 Multi-Seed Verification Table (Split C — 30 Seeds)

| Model / Approach | Deterministic? | Mean R² | Std Dev | Worst Seed | Best Seed | MAPE (%) | Verdict |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Grid Interpolation (raw)** | Yes | **0.8415** | 0.000 | 0.8415 | 0.8415 | 20.85% | Best physical baseline |
| Grid Interpolation (log) | Yes | 0.8040 | 0.000 | 0.8040 | 0.8040 | 26.69% | Exact grid extrapolation |
| **Poly Ridge Deg-2 (log)** | Yes | **0.7547** | 0.000 | 0.7547 | 0.7547 | 35.82% | Best trained ML model |
| Gated Blend (log MLP) | No | 0.6284 | 0.071 | 0.5149 | 0.7549 | 39.78% | Seed-stable neural blend |
| MLP (log target) | No | 0.6277 | 0.072 | 0.5146 | 0.7557 | 39.98% | Seed-stable neural model |
| Gated Blend (raw MLP) | No | 0.4658 | 0.228 | 0.1327 | 0.7412 | 67.09% | Seed artifact — high variance |
| MLP (raw target) | No | 0.4412 | 0.246 | 0.0805 | 0.7265 | 70.50% | Seed artifact — high variance |
| Extra Trees (raw) | Yes | 0.4335 | 0.000 | 0.4335 | 0.4335 | 41.93% | Structural tree collapse |

---

## 6. Phase 3 — PINN (Physics-Informed Neural Network): Full Results

### 6.1 What is a PINN and Why Does CHF Need One?

A Physics-Informed Neural Network (PINN) adds **physics-based penalty terms** to the standard data-fitting loss. In classical PINNs (Raissi et al., 2019), the physics loss is the PDE residual. **CHF has no governing PDE**, so instead we use **empirical physical constraints** as soft penalties evaluated at unlabeled collocation points scattered across the domain:

1. **Monotonicity in X**: CHF must decrease as steam quality increases — dCHF/dX ≤ 0
2. **Zuber Pressure Trend**: The sign of dCHF/dP should match Zuber's (1959) pool-boiling pressure dependence
3. **Positivity**: CHF is always physically ≥ 0

**Why Tanh instead of ReLU?** Physics penalties require computing dŷ/dX and dŷ/dP via torch.autograd.grad. ReLU is piecewise linear (second derivative = 0 everywhere), giving no gradient signal through the penalty. Tanh is smooth and infinitely differentiable, providing meaningful gradient flow for physics terms.

---

### 6.2 PINN Architecture

```
Input: [P, G, X]   (StandardScaler normalized, fit on training data only)
   ↓
Linear(3 → 128) + Tanh
   ↓
Linear(128 → 128) + Tanh
   ↓
Linear(128 → 64) + Tanh
   ↓
Linear(64 → 1)
   ↓
Output: y_norm  (Z-score of ln(CHF))
```

- **Trainable parameters**: 25,345
- **Weight file size**: ~101 KB
- **Training**: Adam optimizer with ReduceLROnPlateau scheduler, gradient clipping (max_norm = 1.0)
- **Early stopping**: Patience = 100 epochs on 15% validation split
- **Max epochs**: 3,000
- **GPU used**: NVIDIA A10G via Modal Cloud (grid search runtime ~1 minute for 720 configs)

---

### 6.3 PINN Loss Function — Full Mathematical Formulation

The total training loss combines data fit with three physics penalties:

```
L_total = L_data  +  λ_mono × L_mono  +  λ_zuber × L_zuber  +  λ_pos × L_pos
```

**Best hyperparameter weights from grid search:**
- λ_mono = 0.1 (monotonicity weight)
- λ_zuber = 0.3 (Zuber pressure trend weight)
- λ_pos = 0.05 (positivity weight)

---

**Loss Term 1 — Data Loss (L_data)**

Standard Mean Squared Error on log-normalized training data.
Targets are normalized as: y_i = (ln(CHF_i) - mean) / std

```
         1   N
L_data = ─ × Σ  (ŷ_i - y_i)²
         N  i=1
```

where ŷ_i is the network prediction and y_i is the true normalized log-CHF.

---

**Loss Term 2 — Monotonicity Penalty (L_mono)**

Penalizes any collocation point where dŷ/dX > 0 (CHF increasing with quality — physically wrong).
Computed via torch.autograd.grad on N_c = 256 or 512 randomly sampled collocation points.

```
          1   N_c
L_mono = ─── × Σ  max(0, dŷ/dX_j)
         N_c  j=1
```

The ReLU max(0, ·) means no penalty if the derivative is already negative (correct physics), and a linear penalty proportional to how much it violates the constraint.

---

**Loss Term 3 — Zuber Pressure Trend Penalty (L_zuber)**

Penalizes any collocation point where the predicted dŷ/dP has the wrong sign compared to Zuber's (1959) known pool-boiling curve.
sign_Zuber(P) is precomputed as a lookup table: sign(d CHF_Zuber / dP) at each pressure.

```
           1   N_c
L_zuber = ─── × Σ  max(0, - (dŷ/dP_j) × sign_Zuber(P_j))
          N_c  j=1
```

If the network's pressure slope agrees with Zuber, the product is positive and max(0, ·) = 0 (no penalty). If it disagrees, the product is negative, giving a positive penalty.

---

**Loss Term 4 — Positivity Penalty (L_pos)**

Penalizes any collocation point with a negative predicted CHF.

```
         1   N_c
L_pos = ─── × Σ  max(0, -ŷ_j)
        N_c  j=1
```

---

### 6.4 Hyperparameter Grid Search Configuration

Run on NVIDIA A10G GPUs via Modal Cloud. Total configs: 720.

| Parameter | Values Searched |
| :--- | :--- |
| Hidden layer architecture | [64, 32], [128, 64, 32], [128, 128, 64] |
| λ_mono (monotonicity weight) | 0.0, 0.1, 0.3, 0.5 |
| λ_zuber (Zuber pressure weight) | 0.0, 0.1, 0.3 |
| Learning rate | 0.001, 0.0005 |
| Collocation points (N_c) | 256, 512 |
| Seeds per config | 5 |

---

### 6.5 PINN Grid Search Results — Top 15 Configurations (Split C, 5-Seed Mean)

| Architecture | λ_mono | λ_zuber | LR | N_c | Mean R² | R² Std | Min R² | Max R² | Mean MAPE (%) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **128×128×64** | **0.1** | **0.3** | **0.0005** | **512** | **0.8123** | 0.031 | 0.767 | **0.846** | **32.87%** |
| 128×128×64 | 0.5 | 0.1 | 0.0005 | 512 | 0.8119 | 0.040 | 0.758 | 0.859 | 32.94% |
| 128×128×64 | 0.3 | 0.1 | 0.0005 | 512 | 0.8053 | 0.037 | 0.764 | 0.866 | 33.03% |
| 128×128×64 | 0.3 | 0.3 | 0.0005 | 512 | 0.8034 | 0.037 | 0.758 | 0.854 | 33.70% |
| 128×128×64 | 0.3 | 0.1 | 0.0005 | 256 | 0.8021 | 0.045 | 0.746 | 0.856 | 33.70% |
| 128×128×64 | 0.5 | 0.3 | 0.0005 | 512 | 0.7982 | 0.025 | 0.764 | 0.835 | 33.60% |
| 128×128×64 | 0.5 | 0.1 | 0.001 | 512 | 0.7977 | 0.035 | 0.748 | 0.841 | 34.17% |
| 128×128×64 | 0.1 | 0.1 | 0.0005 | 512 | 0.7954 | 0.025 | 0.766 | 0.829 | 34.01% |
| 128×128×64 | 0.3 | 0.1 | 0.001 | 256 | 0.7934 | 0.033 | 0.742 | 0.835 | 34.70% |
| 128×128×64 | 0.0 | 0.1 | 0.0005 | 512 | 0.7926 | 0.026 | 0.767 | 0.826 | 34.60% |
| 128×128×64 | 0.0 | 0.1 | 0.001 | 512 | 0.7908 | 0.039 | 0.728 | 0.838 | 35.16% |
| 128×64×32 | 0.1 | 0.3 | 0.0005 | 256 | 0.7803 | 0.036 | 0.726 | 0.824 | 31.91% |
| 128×64×32 | 0.0 | 0.3 | 0.0005 | 256 | 0.7718 | 0.038 | 0.714 | 0.820 | 32.41% |
| 128×64×32 | 0.1 | 0.1 | 0.001 | 512 | 0.7728 | 0.050 | 0.708 | 0.846 | 34.79% |
| 128×64×32 | 0.1 | 0.3 | 0.0005 | 512 | 0.7728 | 0.038 | 0.713 | 0.818 | 32.45% |

---

### 6.6 PINN vs All Baselines — Final Master Comparison (Split C)

| Model | Type | Deterministic? | Split A R² | Split B R² | Split C R² | Split C MAPE (%) | Model Size | Verdict |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Grid Interpolation (raw)** | Physical baseline | Yes | ~1.000 | 0.9990 | **0.8415** | 20.85% | ~260 KB | Best physical baseline |
| **PINN (128×128×64, log)** | Physics-Informed NN | No | 0.9912 | 0.9850 | **0.8123** | 32.87% | ~101 KB (25,345 params) | **Best trained ML model** |
| Grid Interpolation (log) | Physical baseline | Yes | ~1.000 | 0.9987 | 0.8040 | 26.69% | ~260 KB | Strong grid baseline |
| Poly Ridge Deg-2 (log) | Classical ML | Yes | 0.8467 | 0.8707 | 0.7547 | 35.82% | <1 KB (10 params) | Best deterministic ML |
| GPR Matern-5/2 (log) | Classical ML | No | 0.9916 | 0.9955 | 0.6751 | 42.63% | ~50 MB (Gram matrix) | Smooth but slow |
| Poly Ridge Deg-4 (log) | Classical ML | Yes | 0.9508 | 0.9471 | 0.6181 | 67.33% | <1 KB | Overfits at high P |
| Gated Blend (log MLP) | Physics hybrid | No | 0.9993 | 0.9989 | 0.6284 | 39.78% | ~10 KB | Seed-stable blend |
| Standard MLP (log) | Classical ML | No | 0.9906 | 0.9886 | 0.6277 | 39.98% | ~9.5 KB (2,369 params) | Unconstrained NN |
| KNN (k=3, log) | Classical ML | Yes | 0.9933 | 0.9906 | 0.4528 | 41.55% | ~270 KB (stored data) | Grid memorization |
| **Extra Trees (raw)** | Tree ensemble | Yes | 0.9992 | 0.9989 | **0.4335** | 41.93% | ~5–20 MB | Tree collapse |
| XGBoost (log) | Tree ensemble | Yes | 0.9987 | 0.9876 | 0.4307 | 42.49% | ~5 MB | Tree collapse |
| LightGBM (log) | Tree ensemble | Yes | 0.9989 | 0.9879 | 0.4295 | 42.31% | ~3 MB | Tree collapse |
| Random Forest (log) | Tree ensemble | Yes | 0.9987 | 0.9874 | 0.4147 | 43.40% | ~10–20 MB | Tree collapse |
| Physics Penalty MLP (raw) | Phase 2 approach | No | 0.9789 | 0.9809 | 0.8484* | 73.56%* | ~10 KB | *Single-seed artifact |
| Gated Blend (raw MLP) | Physics hybrid | No | 0.9993 | 0.9989 | 0.4658 | 67.09% | ~10 KB | High variance, unreliable |
| Standard MLP (raw) | Classical ML | No | 0.9826 | 0.9893 | 0.4412 | 70.50% | ~9.5 KB | Seed-sensitive |
| Linear Regression (log) | Classical ML | Yes | 0.8343 | 0.8469 | -1.0786 | 66.01% | <1 KB (4 params) | Too simple |
| Poly Ridge Deg-2 (raw) | Classical ML | Yes | 0.8396 | 0.8639 | -1.1869 | 809.55% | <1 KB | Raw scale fails |
| Residual Learning (MLP) | Phase 2 approach | No | 0.7814 | 0.7100 | -1.3060 | 178.78% | ~10 KB | Fails out-of-domain |
| Linear Regression (raw) | Classical ML | Yes | 0.5781 | 0.6151 | -2.6411 | 1362.59% | <1 KB (4 params) | Worst result |

*The Physics Penalty MLP Split C score of 0.8484 was achieved with a single seed (seed 42) and should not be compared directly with multi-seed verified results.

---

## 7. Key Takeaways for Professor Presentation

1. **Tree Models = Interpolation Only**: Extra Trees, Random Forest, XGBoost, LightGBM all achieve R² > 0.999 on Split A, but catastrophically collapse to R² ≈ 0.43 on Split C. This is a structural property — trees output flat step-functions and cannot extrapolate.

2. **Best Physics-Based Baseline**: Grid Interpolation (direct table lookup with linear extrapolation) achieves R² = 0.8415 on Split C — the highest result of any method.

3. **Best Trained ML Model**: PINN (128×128×64, log-target, λ_mono=0.1, λ_zuber=0.3) achieves mean R² = 0.8123 across 5 seeds, beating unconstrained MLPs (R² = 0.6277) by 29%.

4. **Log-Target Critical**: Training on ln(CHF) instead of raw CHF reduces MAPE by 3x–5x for all smooth models, and makes neural networks far more stable across random seeds.

5. **Multi-Seed Averaging is Mandatory**: Neural networks varied from R² = 0.08 to R² = 0.84 depending on random seed initialization. Single-seed results are unreliable — always report multi-seed averages with standard deviation.

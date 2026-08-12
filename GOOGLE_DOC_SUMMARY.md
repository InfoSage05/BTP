# One-Week BTP Project Summary: Critical Heat Flux (CHF) Prediction & Physics-Informed Neural Networks (PINN)

---

## 1. High-Level Project Overview (Simple 1-2 Lines)
We modeled Critical Heat Flux (CHF) from three physical inputs — Pressure (P), Mass Flux (G), and Thermodynamic Quality (X) — using 11,088 clean data points from the 2006 Groeneveld Look-Up Table, establishing a 3-split validation protocol and developing a Physics-Informed Neural Network (PINN) that successfully generalizes into unknown high-pressure extrapolation regimes.

---

## 2. Data Cleaning & Placeholder Explanation (X = 1.0, CHF = 0)
* **Grid Construction**: The raw digitized lookup table (11,592 total rows) was parsed into a structured grid of 24 Pressures × 21 Mass Fluxes × 23 Qualities.
* **Placeholder Removal**: Exactly 504 rows in the grid had CHF = 0, and **all 504 occurred at X = 1.0** (100% steam / pure vapor state where liquid boiling heat transfer cannot physically take place). These 504 rows are non-trainable placeholder zeros rather than experimental CHF measurements.
* **Usable Dataset**: Filtering out X = 1.0 (`df[df.X != 1.0]`) leaves **11,088 usable rows** for training and testing.
* **Target Transformation**: Non-zero CHF values range from **15 to 44,338 kW/m²** (over 3 orders of magnitude). Training on log-transformed targets ln(CHF) reduces percentage error (MAPE) by 3x to 5x and ensures numerical stability.

---

## 3. Behaviors of Dataset Splits A, B, and C (1-Line Summaries)
To rigorously test interpolation vs. extrapolation performance, models were evaluated across three distinct validation splits:

* **Split A (Random 80/20)**: *Optimistic Interpolation Test.* Randomly splits data; all models score near-perfect (R² > 0.99) because test points sit directly adjacent to training points on the dense grid.
* **Split B (Interior Pressure Holdout)**: *Moderate Interpolation Test.* Holds out every 4th interior pressure level (e.g., 1000, 5000, 9000 kPa); tests filling in missing sandwiched pressure slices (R² ≈ 0.87 - 0.99).
* **Split C (High-Pressure Extrapolation)**: *The Honest Test.* Trains strictly on P <= 16,000 kPa and tests on the 5 highest pressures (17,000 - 21,000 kPa). Reveals that tree models collapse (R² ≈ 0.43) due to flat step predictions, while PINN and smooth models generalize well (R² ≈ 0.75 - 0.81).

---

## 4. Master Model Performance Table
*(Results evaluated on Split C High-Pressure Extrapolation unless specified)*

| Model Name | Target Mode | Split A R² | Split B R² | Split C R² | Split C MAPE (%) | Model Size / Complexity | Determinism & Verdict |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Grid Interpolation** | Raw | 0.9999* | 0.9990 | **0.8415** | **20.8%** | 11,088 table points (~260 KB) | Deterministic — Exact table lookup baseline |
| **PINN (128x128x64 MLP)** | Log | 0.9912 | 0.9850 | **0.8123** | **32.9%** | 25,345 params (~101 KB) | Multi-seed avg — **Best Physics ML model** |
| **Degree-2 Ridge** | Log | 0.8467 | 0.8707 | **0.7547** | **35.8%** | 10 params (< 1 KB) | Deterministic — Smooth polynomial baseline |
| **Standard MLP** | Log | 0.9906 | 0.9886 | **0.6277** | **40.0%** | 2,369 params (~9.5 KB) | Multi-seed avg — Unconstrained neural net |
| **Extra Trees / Random Forest** | Raw | 0.9992 | 0.9989 | **0.4335** | **41.9%** | 100–500 trees (~5–20 MB) | Tree extrapolation failure (flat step functions) |
| **Linear Regression** | Raw | 0.5781 | 0.6151 | **-2.6411** | **1362.6%** | 4 params (< 1 KB) | Linear baseline floor failure |

*\*Note: Grid Interpolation inside Split A training domain is exact (R² ≈ 1.0).*

---

## 5. Physics-Informed Neural Network (PINN) Details

### A. Yesterday's GPU Grid Search Results
Yesterday, we executed parallelized PINN hyperparameter grid searches on Modal Cloud (NVIDIA A10G GPUs).
* **Top Architecture**: 3 Hidden Layers (128 × 128 × 64 neurons with Tanh activation).
* **Model Size**: **25,345 trainable parameters** (≈ 101 KB weight file).
* **Extrapolation Score (Split C)**: Mean R² = **0.8123** (Peak R² = **0.8591**), MAPE = **32.87%**.
* **Key Finding**: Incorporating physical constraints improved high-pressure extrapolation R² from 0.6277 (unconstrained MLP) to **0.8123 (PINN)**, eliminating random seed instability.

### B. PINN Loss Function & Mathematical Formulation
Because Critical Heat Flux has no governing Partial Differential Equation (PDE), the PINN is trained by minimizing a composite loss function combining empirical data fit with physical differential constraints evaluated at collocation points using PyTorch `autograd`:

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{data}} + \lambda_{\text{mono}} \mathcal{L}_{\text{mono}} + \lambda_{\text{zuber}} \mathcal{L}_{\text{zuber}} + \lambda_{\text{pos}} \mathcal{L}_{\text{pos}}$$

*(Hyperparameter Weights: λ_mono = 0.1, λ_zuber = 0.3, λ_pos = 0.05)*

#### 1. Data Loss (L_data)
Measures Mean Squared Error (MSE) on log-normalized CHF targets y = (ln(CHF) - μ) / σ:
$$\mathcal{L}_{\text{data}} = \frac{1}{N} \sum_{i=1}^{N} \left( \hat{y}_i - y_i \right)^2$$

#### 2. Quality Monotonicity Penalty (L_mono)
Enforces the physical law that CHF must monotonically decrease as steam quality X increases (∂CHF / ∂X <= 0):
$$\mathcal{L}_{\text{mono}} = \frac{1}{N_c} \sum_{j=1}^{N_c} \max\left(0, \frac{\partial \hat{y}}{\partial X_j}\right)$$

#### 3. Zuber Hydrodynamic Pressure Trend Penalty (L_zuber)
Enforces that the predicted derivative with respect to pressure matches the slope sign sign(∂CHF_Zuber / ∂P) of Zuber's classic pool boiling curve:
$$\mathcal{L}_{\text{zuber}} = \frac{1}{N_c} \sum_{j=1}^{N_c} \max\left(0, -\frac{\partial \hat{y}}{\partial P_j} \cdot \text{sign}\left(\frac{\partial \text{CHF}_{\text{Zuber}}}{\partial P_j}\right)\right)$$

#### 4. Positivity Penalty (L_pos)
Ensures predictions remain strictly non-negative (CHF >= 0):
$$\mathcal{L}_{\text{pos}} = \frac{1}{N_c} \sum_{j=1}^{N_c} \max\left(0, -\hat{y}_j\right)$$

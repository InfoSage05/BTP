# Critical Heat Flux (CHF) Machine Learning & Physics-Informed Prediction

A comprehensive machine learning and physics-informed framework for predicting **Critical Heat Flux (CHF)** in water-cooled thermal systems using the **2006 Groeneveld Look-Up Table (LUT)** dataset.

## 📌 Repository Overview

This repository contains the complete codebase, data pipelines, Jupyter notebooks, literature references, and verification scripts for the Bachelor's Thesis Project (BTP) on CHF prediction.

### Key Objectives
* **Model Evaluation**: Compare 10 machine learning model families across interpolation and high-pressure edge extrapolation.
* **Extrapolation Benchmark**: Test model performance when extrapolating beyond training pressure boundaries ($P > 16,000\text{ kPa}$).
* **Physics-Informed Extensions**: Evaluate hybrid residual learning, physics-basis feature engineering, PyTorch collocation physics penalties, and mixture-of-expert pressure gating.
* **Rigorous Verification**: Audit single-seed neural network instability using multi-seed verification loops.

---

## 📁 Repository Structure

```
.
├── data/
│   ├── raw/                           # Original supplied/source files
│   │   ├── mentor_master_experiments.xlsx
│   │   ├── external_coil_tube_chf_appendix.pdf
│   │   └── groeneveld_2006_chf_lookup_table.xlsx
│   ├── chf_long_clean.csv             # Canonical cleaned 11,592-row LUT dataset
│   └── chf_long_with_gridbase.csv     # Derived grid-base dataset
│
├── docs/references/                  # Scientific literature and source papers
│
├── notebooks/
│   ├── CHF_ML_Modeling.ipynb                    # Primary Phase-1 Machine Learning Modeling notebook
│   ├── CHF_Physics_Informed_Extensions.ipynb    # Primary Phase-2 Physics-Informed Extensions notebook
│   ├── CHF_PINN_Model.ipynb                     # Phase-3 PINN with grid search & multi-seed evaluation
│   └── model_tests/                             # Focused per-model unit-test & diagnostic notebooks
│       ├── test_extra_trees.ipynb
│       ├── test_gpr.ipynb
│       ├── test_gridinterp.ipynb
│       ├── test_knn.ipynb
│       ├── test_lightgbm.ipynb
│       ├── test_linear_regression.ipynb
│       ├── test_mlp.ipynb
│       ├── test_polynomial_regression.ipynb
│       ├── test_random_forest.ipynb
│       └── test_xgboost.ipynb
│
├── results/                          # Generated numeric results, summary CSVs, and figures
│   ├── figures/
│   ├── physics_informed/
│   ├── plan2/                         # Mentor/PDF audit and external evaluation
│   ├── combined_summary_all_splits.csv
│   ├── split_A_summary.csv
│   ├── split_B_results.csv
│   ├── split_C_results.csv
│   └── split_C_multiseed_verification.csv
│
├── scripts/
│   ├── build_notebook.py             # Rebuilds CHF_ML_Modeling.ipynb
│   ├── build_notebook_physics.py     # Rebuilds CHF_Physics_Informed_Extensions.ipynb
│   ├── build_notebook_pinn.py        # Rebuilds CHF_PINN_Model.ipynb
│   ├── build_model_test_notebooks.py # Rebuilds model test notebooks in notebooks/model_tests/
│   ├── prepare_data.py               # Data extraction & Excel cleaning pipeline
│   ├── verify_results.py             # Senior scientist audit & multi-seed verification script
│   ├── chf_physics.py                # Physical correlation modules (Biasi, Zuber, hybrid models)
│   ├── run_pinn_quick.py             # Quick local PINN sanity run
│   ├── modal_btp_gpu_pipeline.py     # Modal.com GPU pipeline for full training runs
│   └── modal_pinn_grid_search.py     # Modal.com GPU grid search for PINN hyperparameters
│
├── docs/
│   ├── CHF_ML_Context.md             # Background/context on the modeling problem
│   ├── CHF_Physics_Approaches_Explained.md
│   ├── CHF_Project_Simple_Explanation.md
│   ├── GOOGLE_DOC_SUMMARY.md
│   ├── GOOGLE_DOC_SUMMARY.txt
│   ├── SENIOR_REVIEW.md
│   ├── project_status.pdf
│   ├── manuscript/
│   └── references/
│
├── requirements.txt                  # Python dependencies
└── README.md                         # Project documentation
```

---

## 📊 Dataset Facts

The dataset is derived from the **2006 Groeneveld CHF Look-Up Table** for a vertical 8 mm water-cooled tube:
* **Raw Grid Dimensions**: 24 Pressures ($100 - 21,000\text{ kPa}$) $\times$ 21 Mass Fluxes ($0 - 8,000\text{ kg/m}^2/\text{s}$) $\times$ 23 Qualities ($-0.50 - 1.00$) = **11,592 total grid points**.
* **Filtered Usable Dataset**: Exactly 504 rows have $\text{CHF} = 0$, all located at $X = 1.0$ (all-steam boundary condition). Excluding these leaves **11,088 usable rows** for model training and testing.
* **Target Scale**: Non-zero CHF ranges from **15.0 to 44,338.0 kW/m²** (spans over 3.5 orders of magnitude), motivating $\ln(\text{CHF})$ log-target transformations.

---

## 🎯 Validation Protocols

Models are evaluated across three distinct splits:
1. **Split A (Random 80/20 — Interpolation Test)**: 5 random seeds (0–4). Tests local grid point interpolation.
2. **Split B (Interior Pressure Holdout — Sandwiched Test)**: Holds out every 4th interior pressure level (1000, 5000, 9000, 13000, 17000 kPa).
3. **Split C (High Pressure Extrapolation — The Honest Test)**: Trains ONLY on $P \le 16,000\text{ kPa}$ (8,778 rows) and tests ONLY on $17,000 - 21,000\text{ kPa}$ (2,310 rows).

---

## 🏆 Key Findings & Benchmark Performance

### Master Performance Table (Split C Edge Extrapolation)

| Model / Approach | Deterministic? | Mean $R^2$ | Std Dev | Worst Seed | MAPE (%) | Key Takeaway |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **GridInterp (Raw Target)** | **Yes** | **0.8415** | **0.000** | **0.8415** | **20.8%** | **Best Physical Baseline** (Exact) |
| GridInterp (Log Target) | Yes | 0.8040 | 0.000 | 0.8040 | 26.7% | Exact Grid Extrapolation |
| **Poly2_Ridge (Log Target)** | **Yes** | **0.7547** | **0.000** | **0.7547** | **35.8%** | **Best Trained ML Model** (Exact) |
| GatedBlend (Log MLP) | No | 0.6284 | 0.071 | 0.515 | 39.8% | Stable Stochastic Neural Model |
| MLP (Log Target) | No | 0.6277 | 0.072 | 0.515 | 40.0% | Stable Stochastic Neural Model |
| GatedBlend (Raw MLP) | No | 0.4658 | 0.228 | 0.133 | 67.1% | Seed Artifact (High Variance) |
| MLP (Raw Target) | No | 0.4412 | 0.246 | 0.081 | 70.5% | Seed Artifact (High Variance) |
| Tree Ensembles (RF/ET/XGB) | Yes | ~0.4335 | ~0.000 | 0.4335 | ~42.0% | **Structural Tree Extrapolation Collapse** |

### Core Scientific Conclusions
1. **Tree Extrapolation Failure**: Tree models (Random Forest, Extra Trees, XGBoost, LightGBM) score $R^2 > 0.999$ in-domain (Split A), but suffer structural collapse ($R^2 \approx 0.43$) when extrapolating past training pressure boundaries.
2. **Top Extrapolation Performers**: **Trilinear Grid Interpolation** ($R^2 = 0.8415$, exact) is the strongest physical baseline. **Degree-2 Log-Ridge** ($R^2 = 0.7547$, exact) is the most reliable trained ML model.
3. **Log Target Transformation**: Training on $\ln(\text{CHF})$ reduces percentage error (MAPE) by 3–5x and stabilizes neural network variance.
4. **Multi-Seed Verification**: Single-seed neural network benchmarks can be misleading artifacts; multi-seed verification is essential for honest ML evaluation.

---

## 🛠️ Quick Start & Usage

### 1. Installation
Clone the repository and install dependencies:
```bash
git clone https://github.com/your-username/chf-prediction-btp.git
cd chf-prediction-btp
pip install -r requirements.txt
```

### 2. Re-generating Data & Rebuilding Notebooks
Run all commands below from the repository root. To clean raw Excel data:
```bash
python scripts/prepare_data.py
```

To build all Jupyter notebooks:
```bash
python scripts/build_notebook.py
python scripts/build_notebook_physics.py
python scripts/build_notebook_pinn.py
python scripts/build_model_test_notebooks.py
```

To run the verification audit:
```bash
python scripts/verify_results.py
```

To run the leakage-safe mentor audit and PDF extraction:
```bash
python scripts/plan2_pipeline.py
```

---

## 📜 License & Acknowledgments
* Based on the 2006 Groeneveld Critical Heat Flux Look-Up Table (*Nuclear Engineering and Design*, 2007).
* Developed as part of a Bachelor's Thesis Project (BTP).

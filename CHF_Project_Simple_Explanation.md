# Simple Guide to the CHF Machine Learning Project

This guide explains everything in your project folder in simple, easy-to-read English without complicated math symbols or dense formula notation.

---

## 1. What is this Project About?

### The Core Problem: What is Critical Heat Flux (CHF)?
* **Boiling Heat Transfer**: When liquid water cools a very hot metal surface (like inside a nuclear reactor or high-pressure boiler), bubbles form and remove heat.
* **The Danger (CHF)**: If the heat flux gets *too high*, a layer of vapor gas covers the hot surface like an insulating blanket. Because gas conducts heat much worse than liquid water, the surface temperature suddenly spikes to dangerously high levels, which can melt equipment. This tipping point is called **Critical Heat Flux (CHF)**.
* **Goal of the Project**: Build and test machine learning models to accurately predict the exact CHF value (measured in kW/m^2) using 3 input measurements:
  1. **Pressure (P)**: How compressed the liquid/steam system is (measured in kPa).
  2. **Mass Flux (G)**: How fast the liquid is flowing through the pipe (measured in kg/m^2/s; G = 0 means still water / pool boiling).
  3. **Quality (X)**: How much of the water mixture has turned into steam (-0.50 means subcooled cold liquid, 0.0 means boiling water, 1.0 means 100% pure steam).

---

## 2. The Dataset Explained

### Where did the data come from?
* **Original File**: `2006_CHF_Lookup_Table.xlsx` (A large digitized table created by scientists in 2006 from decades of laboratory experiments).
* **Cleaning Script**: `prepare_data.py` (A Python script that cleans merged spreadsheet cells, blank spacer rows, and repeated headers).
* **Final Working File**: `chf_long_clean.csv` (**This is the actual clean dataset file loaded by all Jupyter Notebooks**).

### Dataset Size & Columns
* **Original Grid Size**: 24 Pressure levels x 21 Mass Flux levels x 23 Quality levels = **11,592 total rows**.
* **Columns**: `P` (Pressure), `G` (Mass Flux), `X` (Quality), `CHF` (Critical Heat Flux).
* **Important Cleaning Rule**: 
  * Exactly 504 rows in the grid have `CHF = 0`.
  * **All 504 of these rows occur at Quality X = 1.0** (100% steam condition, where liquid boiling cannot happen).
  * These 504 rows are placeholders, not real measurements.
  * **We filter out all X = 1.0 rows before training**, leaving **11,088 usable rows**.
* **Target Scale**: Non-zero CHF values range from **15 to 44,338 kW/m^2** (a huge 3000-fold difference!). Because of this wide spread, models are trained on both raw CHF values and `log(CHF)` values.

---

## 3. Simple Summary of the 3 Main PDFs

### PDF 1: `CHF Look Up Table.pdf` (14 Pages)
* **Full Title**: *The 2006 CHF look-up table* (Groeneveld et al., 2007).
* **What it contains**: This is the scientific paper introducing the lookup table dataset. The table was built by normalizing over **30,000 real experiment data points**.
* **Key Point**: The authors used a classic physics formula (the Zuber pool boiling formula) to fill in values at zero flow (G = 0) and at extreme top pressures (21,000 kPa).

### PDF 2: `Applications ML.pdf` (8 Pages)
* **Full Title**: *Applications of machine learning methods for boiling modeling and prediction: A comprehensive review* (Rashidi et al., 2022).
* **What it contains**: A review of how researchers around the world use Machine Learning (Neural Networks, Random Forests, Support Vector Machines) to predict boiling and heat transfer.
* **Key Point**: Machine Learning models are great at learning experimental patterns, but they can produce unrealistic predictions if asked to predict outside their training range.

### PDF 3: `Critical heat Flux.pdf` (32 Pages — The paper with >30 pages)
* **Full Title**: *Critical heat flux for water flow in tubes — Compilation and assessment of world CHF data* (Hall & Mudawar, 2000).
* **What it contains**: A landmark study from Purdue University compiling decades of boiling data from around the world.
* **Key Point**: Explains how CHF changes with pressure, flow rate, and tube size, establishing standard error metrics (R-squared and MAPE) used in this project.

*(Note: The 4 other PDFs in the folder are smaller specialized papers studying boiling in helical coils and straight pipes).*

---

## 4. Notebook 1 Explained: `CHF_ML_Modeling.ipynb`

This notebook tests 10 different machine learning models using **3 different testing strategies** to see how well models generalize.

### The 3 Testing Strategies (Splits)
1. **Split A (Random 80/20 Split — The "Easy" Test)**:
   * Randomly pick 80% of rows for training and 20% for testing (repeated across 5 random seeds).
   * *Result*: Almost every model scores near-perfect (R-squared > 0.99) because neighboring grid points are right next to each other.
2. **Split B (Interior Pressure Holdout — The "Medium" Test)**:
   * Remove every 4th interior pressure level (e.g. 1000 kPa, 5000 kPa, 9000 kPa).
   * Tests if models can fill in missing pressure slices when higher and lower pressures are still known.
3. **Split C (High Pressure Extrapolation — The "Honest" Test)**:
   * **Train ONLY on pressures up to 16,000 kPa**.
   * **Test ONLY on the 5 highest pressures (17,000 to 21,000 kPa)**.
   * This tests true extrapolation into unknown conditions.

### Models Tested in Notebook 1
* **Linear & Polynomial Regression** (Degree 2 & Degree 4)
* **KNN** (k-Nearest Neighbors)
* **Tree Ensembles**: Random Forest, Extra Trees, XGBoost, LightGBM
* **Neural Network (MLP)**: A small network with 2 hidden layers (64 and 32 units)
* **Grid Interpolation Baseline**: Standard linear grid interpolation from scientific software (`scipy`)

### Main Findings from Notebook 1
1. **Tree Models Collapse on Split C**:
   * Tree models (Random Forest, Extra Trees, XGBoost, LightGBM) score **R-squared = 0.999** on Split A, but drop to **R-squared = 0.43** on Split C.
   * **Why?** Decision trees produce flat step-like predictions. When queried at high pressures never seen in training, trees cannot extrapolate trends upward — they output flat constant values.
2. **Smooth Models Perform Much Better on Split C**:
   * Smooth models (Grid Interpolation R-squared = 0.84, Degree-2 Polynomial R-squared = 0.75, Neural Network R-squared = 0.63–0.85) continue curves naturally into high pressures.

---

## 5. Notebook 2 Explained: `CHF_Physics_Informed_Extensions.ipynb`

This notebook explores whether adding physical equations or constraints into Machine Learning can improve predictions on Split C.

### The 4 Physics Approaches Tested
1. **Approach 1 (Physics Features)**: Adding engineered physics terms to polynomial regression. *(Failed on Split C because terms exploded out of range).*
2. **Approach 2 (Residual Learning)**: Using ML to predict only the error of a physics formula. *(Failed on Split C because formula errors changed shape out of range).*
3. **Approach 3 (Physics-Penalty Neural Network)**: Training a PyTorch neural network with soft penalties for physical rules (like "CHF must drop as steam quality rises").
4. **Approach 4 (Pressure-Gated Blend)**: A combination model — uses Extra Trees for normal pressures and a Neural Network for high pressure extrapolation.

---

## 6. Important Scientific Correction & Senior Review (`SENIOR_REVIEW.md`)

During deep testing across multiple random seeds, an important discovery was made:

* **The Single-Seed Fluke**: In early tests, one specific random seed (Seed 42) made the Pressure-Gated Blend score a top result of **R-squared = 0.855**.
* **The Multi-Seed Truth**: When re-tested across **30 different random seeds** (`verify_results.py`), raw-target Neural Networks varied wildly (from R-squared = 0.08 to 0.84, averaging 0.54).
* **The Verified Fact**: The initial R-squared = 0.855 claim was a lucky random seed initialization, not a superior model.

---

## 7. Master Model Results Table (Split C High-Pressure Test)

Here are the verified performance scores on **Split C (High Pressure Extrapolation)** in plain numbers:

| Model Name | Is Result Exact? | Average R-squared Score | Error Rate (MAPE %) | Summary / Verdict |
| :--- | :---: | :---: | :---: | :--- |
| **Grid Interpolation (Raw)** | **Yes (Exact)** | **0.8415** | **20.8%** | **Best physical baseline** (Exact table lookup) |
| Grid Interpolation (Log) | Yes (Exact) | 0.8040 | 26.7% | Strong physical baseline |
| **Degree-2 Log Polynomial** | **Yes (Exact)** | **0.7547** | **35.8%** | **Best trained ML model** (Deterministic & reliable) |
| Gated Blend (Log MLP) | No (Random seed) | 0.6284 | 39.8% | Stable neural blend |
| Neural Network (Log MLP) | No (Random seed) | 0.6277 | 40.0% | Stable neural network |
| Gated Blend (Raw MLP) | No (Random seed) | 0.4658 | 67.1% | Unstable (Seed dependent) |
| Neural Network (Raw MLP) | No (Random seed) | 0.4412 | 70.5% | Unstable (Seed dependent) |
| **Tree Models (Random Forest/XGBoost)** | **Yes (Exact)** | **0.4335** | **41.9%** | **Tree extrapolation failure** |

---

## 8. Summary of Main Takeaways for Your Presentation

1. **Tree Models are for Interpolation Only**: Tree-based models (Random Forest, Extra Trees, XGBoost) are brilliant inside known conditions (R-squared > 0.99), but fail badly when extrapolating to higher pressures (R-squared = 0.43).
2. **Best Extrapolation Models**: 
   * The best physical baseline is **Grid Interpolation** (R-squared = 0.84).
   * The best trained ML model for extrapolation is **Degree-2 Log Polynomial Regression** (R-squared = 0.75).
3. **Log Target Transformation**: Training on `log(CHF)` instead of raw `CHF` reduces percentage error by **3x to 5x** and makes neural networks much more stable.
4. **Research Method Lesson**: Never trust a single random seed score for Neural Networks — always test across multiple seeds before making final conclusions.

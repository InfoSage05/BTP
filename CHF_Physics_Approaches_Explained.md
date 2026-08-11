# Detailed Explanation: Why Split C & The 4 Physics Approaches

This guide answers your two questions in clear, simple language with full technical details so you can understand, present, and modify the code.

---

## Part 1: Why split the dataset into Split C in the first place?

### The Core Problem with Standard Testing (Split A)
* In standard Machine Learning, we usually do a random 80/20 train/test split (**Split A**).
* However, our dataset (`chf_long_clean.csv`) is a **structured 3D grid** of Pressure (P), Mass Flux (G), and Quality (X).
* In a random split, test points sit directly next to training points on the grid. 
* **The Problem**: Tree-based models (Random Forest, Extra Trees, XGBoost) get an artificially high score (**R^2 > 0.999**) on Split A because they simply "memorize" neighboring grid points. This is an **interpolation test**.

### The Real-World Engineering Reality
* In real nuclear reactors or steam boilers, engineers often operate at higher pressures or new physical conditions that were **never tested in lab experiments** during training.
* We must know: *"If our system enters a new pressure range (above 16,000 kPa), will our ML model make physically sensible predictions, or will it fail dangerously?"*

### The Purpose of Split C (High-Pressure Extrapolation — "The Honest Test")
* **Split C** intentionally trains ONLY on $P \le 16,000\text{ kPa}$ (8,778 rows) and tests ONLY on $17,000 \text{ to } 21,000\text{ kPa}$ (2,310 rows).
* There are **zero higher-pressure training points** to look up or interpolate between.
* **What Split C Proves**:
  1. **Tree Models Collapse ($R^2 \approx 0.43$)**: Decision trees output flat step-functions. Above 16,000 kPa, every query routes to a leaf learned from lower pressures, causing tree models to fail.
  2. **Smooth Models Hold Up ($R^2 \approx 0.75 - 0.84$)**: Models with continuous mathematical equations (Polynomials, Grid Interpolation, Neural Networks) continue the physical curve upward.
* **Summary**: Split C exists to test **true out-of-domain extrapolation**, which is the ultimate test for real-world engineering safety.

---

## Part 2: The 4 Physics Approaches (Mathematical & Conceptual Breakdown)

---

### Approach 1: Physics-Basis Features + Ridge Regression

#### 1. Concept:
Standard polynomial regression uses raw powers of inputs like $P^2$, $G^2$, $X^2$, or $P \cdot G$. Approach 1 adds **engineered physics terms** inspired by famous boiling equations (Biasi, Zuber, Katto) into the feature matrix before fitting Ridge regression.

#### 2. Mathematical Form:
Instead of input vector $\mathbf{x} = [P, G, X]$, construct an augmented feature vector $\phi(\mathbf{x})$:
$$\phi(\mathbf{x}) = \left[ P, \, G, \, X, \,\, \ln(1+G), \,\, (1-X)^n, \,\, \ln\left(1 - \frac{P}{P_{\text{crit}}}\right), \,\, \left(\frac{P}{P_{\text{crit}}}\right)^m \right]$$

The model predicts:
$$\hat{y} = \mathbf{w}^T \phi(\mathbf{x}) + b$$
fitted by minimizing Ridge L2 error: $\min_{\mathbf{w}} \|y - \hat{y}\|^2 + \alpha \|\mathbf{w}\|^2$.

#### 3. Why it failed on Split C ($R^2 < 0$):
The near-critical term $\ln(1 - P/P_{\text{crit}})$ approaches infinity near critical pressure ($P_{\text{crit}} = 22,064\text{ kPa}$). When trained on $P \le 16,000\text{ kPa}$, weights $\mathbf{w}$ were small. But when tested above $16,000\text{ kPa}$, this logarithmic feature exploded, causing predictions to drop to negative thousands.

#### 4. How you can modify it:
* Remove logarithmic singularities near $P_{\text{crit}}$.
* Use normalized linear ratios like $P/P_{\text{crit}}$ bounded between 0 and 1.
* Apply feature clipping (`np.clip`) to prevent out-of-range feature explosion.

---

### Approach 2: Residual Learning on Biasi/Zuber Hybrid Correlation

#### 1. Concept:
Instead of forcing the ML model to learn CHF from scratch, start with a known textbook physical correlation $f_{\text{hybrid}}(P, G, X)$ (Biasi formula for flow $G>0$, Zuber formula for pool boiling $G=0$). The ML model only needs to learn the **residual error**:
$$\text{Residual } r = \text{True CHF} - f_{\text{hybrid}}(P, G, X)$$

#### 2. Mathematical Form:
1. Calculate physics baseline: $y_{\text{base}} = f_{\text{hybrid}}(P, G, X)$
2. Train ML model $g_\theta(P, G, X)$ on target $r = y_{\text{true}} - y_{\text{base}}$
3. Final prediction:
$$\hat{y}(P, G, X) = f_{\text{hybrid}}(P, G, X) + g_\theta(P, G, X)$$

#### 3. Why it failed on Split C ($R^2 < 0$):
The textbook formula $f_{\text{hybrid}}$ has one shape of error inside the training range ($P \le 16,000\text{ kPa}$). The ML residual model $g_\theta$ learned to correct those specific errors. Outside the training range ($P > 16,000\text{ kPa}$), the formula's error changed shape completely, but $g_\theta$ applied the wrong correction, making total predictions far worse.

#### 4. How you can modify it:
* Damp the residual correction as distance from training data increases: $\hat{y} = f_{\text{hybrid}} + g_\theta \cdot \exp(-\gamma \cdot \text{distance})$.
* Use **multiplicative residual learning** instead of additive: $\hat{y} = f_{\text{hybrid}} \cdot g_\theta$.

---

### Approach 3: Collocation-Based Physics-Penalty Neural Network (PINN Style)

#### 1. Concept:
Train a PyTorch Neural Network $\hat{y} = \text{NN}(P, G, X; \theta)$ on labeled data, but add **soft physical penalty terms** to the loss function evaluated at random **unlabeled collocation points** inside the high-pressure region ($17,000 - 21,000\text{ kPa}$).

#### 2. Mathematical Form:
Total Loss = Data Loss + Monotonicity Penalty + Zuber Trend Penalty:
$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{data}} + \lambda_{\text{mono}} \mathcal{L}_{\text{mono}} + \lambda_{\text{zuber}} \mathcal{L}_{\text{zuber}}$$

* **Data Loss**: Standard MSE on training points:
  $$\mathcal{L}_{\text{data}} = \frac{1}{N} \sum_{i=1}^N (\hat{y}_i - y_i)^2$$
* **Monotonicity Penalty**: Using PyTorch autograd, penalize positive quality derivatives (CHF must drop as quality X rises):
  $$\mathcal{L}_{\text{mono}} = \text{Mean}\left( \max\left(0, \frac{\partial \hat{y}}{\partial X}\right) \right)$$
* **Zuber Trend Penalty**: Penalize sign disagreement with Zuber's pressure derivative $S_P(P) = \text{sign}\left(\frac{d \text{Zuber}}{dP}\right)$:
  $$\mathcal{L}_{\text{zuber}} = \text{Mean}\left( \max\left(0, - \frac{\partial \hat{y}}{\partial P} \cdot S_P(P)\right) \right)$$

#### 3. How it performed & How to modify it:
Reached $R^2 = 0.848$ on a single seed, but penalty weights $\lambda$ were very sensitive.
* **How to modify it**: Use **adaptive loss balancing** (like GradNorm or SoftAdapt) so PyTorch automatically tunes $\lambda_{\text{mono}}$ and $\lambda_{\text{zuber}}$ during training instead of hardcoding $\lambda = 0.3$.

---

### Approach 4: Pressure-Gated Blend (Mixture of Experts)

#### 1. Concept:
Combines two specialized models using a continuous "gating function" $\text{Gate}(P) \in [0, 1]$:
* **Expert 1 ($M_{\text{tree}}$)**: Extra Trees regressor — accurate inside the training pressure range.
* **Expert 2 ($M_{\text{smooth}}$)**: Compact Neural Network — smooth continuous extrapolation outside training.

#### 2. Mathematical Form:
$$\text{Gate}(P) = \text{clamp}\left( \frac{P - P_{\text{train\_max}}}{\text{margin}}, \,\, 0.0, \,\, 1.0 \right)$$
$$\hat{y}(P, G, X) = (1 - \text{Gate}(P)) \cdot M_{\text{tree}}(P, G, X) + \text{Gate}(P) \cdot M_{\text{smooth}}(P, G, X)$$

* For $P \le 16,000\text{ kPa}$: $\text{Gate}(P) = 0 \implies \hat{y} = M_{\text{tree}}$ (100% Tree Model).
* For $P \ge 18,000\text{ kPa}$: $\text{Gate}(P) = 1 \implies \hat{y} = M_{\text{smooth}}$ (100% Smooth Model).
* Between $16,000 < P < 18,000\text{ kPa}$: Smooth weighted blend.

#### 3. How to modify it:
* Replace $M_{\text{smooth}}$ with a **deterministic model** (like Degree-2 Log Ridge, $R^2 = 0.755$) to eliminate random seed variance.
* Make the gating transition threshold dynamic based on local sample density.

"""
build_model_test_notebooks.py
-------------------------------
Generates one focused testing/validation notebook per Phase-1 model family
into model_tests/. Each notebook is NOT a re-run of the full 3-split pipeline
(that's what CHF_ML_Modeling.ipynb already does) -- it's a unit-test-style
deep dive on ONE model: does it behave the way its architecture predicts,
including on edge cases (G=0 pool boiling, exact grid nodes, far
out-of-range queries, seed sensitivity, numerical conditioning)?
"""
import nbformat as nbf

SHARED_SETUP = r"""
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
warnings.filterwarnings("ignore")

def mape(y_true, y_pred):
    return float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100)
def r2(y_true, y_pred):
    return float(r2_score(y_true, y_pred))

df_raw = pd.read_csv("../../data/chf_long_clean.csv")
df = df_raw[df_raw.X != 1.0].reset_index(drop=True)
FEATURES = ["P", "G", "X"]
TARGET = "CHF"
sorted_P = sorted(df.P.unique())

# Split A (random, seed 0) -- quick interpolation check
X_all, y_all = df[FEATURES].values, df[TARGET].values
XtrA, XteA, ytrA, yteA = train_test_split(X_all, y_all, test_size=0.2, random_state=0)

# Split C (edge extrapolation) -- the honest test
train_dfC = df[df.P <= 16000].reset_index(drop=True)
test_dfC = df[df.P >= 17000].reset_index(drop=True)
XtrC, ytrC = train_dfC[FEATURES].values, train_dfC[TARGET].values
XteC, yteC = test_dfC[FEATURES].values, test_dfC[TARGET].values

print(f"Split A: {len(XtrA)} train / {len(XteA)} test")
print(f"Split C: {len(XtrC)} train / {len(XteC)} test")
"""

def build_notebook(model_name, filename, intro_md, imports, fit_code, edge_case_md, edge_case_code, plot_code):
    nb = nbf.v4.new_notebook()
    cells = [
        nbf.v4.new_markdown_cell(f"# Model test & validation: {model_name}\n\n{intro_md}"),
        nbf.v4.new_code_cell(imports + "\n" + SHARED_SETUP),
        nbf.v4.new_markdown_cell("## Fit on Split A (interpolation) and Split C (extrapolation)"),
        nbf.v4.new_code_cell(fit_code),
        nbf.v4.new_markdown_cell(f"## Edge-case tests\n\n{edge_case_md}"),
        nbf.v4.new_code_cell(edge_case_code),
        nbf.v4.new_markdown_cell("## Diagnostic plot"),
        nbf.v4.new_code_cell(plot_code),
    ]
    nb["cells"] = cells
    nb["metadata"] = {
        "kernelspec": {"display_name": "Python (CHF venv)", "language": "python", "name": "chf-venv"},
        "language_info": {"name": "python", "version": "3.12"},
    }
    import os
    os.makedirs("notebooks/model_tests", exist_ok=True)
    with open(f"notebooks/model_tests/{filename}", "w", encoding="utf-8") as f:
        nbf.write(nb, f)
    print(f"Built notebooks/model_tests/{filename}")


# ============================================================================
# 1. Linear Regression
# ============================================================================
build_notebook(
    "Linear Regression",
    "test_linear_regression.ipynb",
    "Floor baseline. Expected to perform poorly overall, and to be numerically "
    "well-behaved (no conditioning issues) since it's the simplest possible model "
    "on only 3 standardized features.",
    "from sklearn.linear_model import LinearRegression",
    r"""
scaler = StandardScaler().fit(XtrA)
model_A = LinearRegression().fit(scaler.transform(XtrA), np.log(ytrA))
predA = np.exp(model_A.predict(scaler.transform(XteA)))
print(f"Split A: R2={r2(yteA, predA):.4f}, MAPE={mape(yteA, predA):.2f}%")

scalerC = StandardScaler().fit(XtrC)
model_C = LinearRegression().fit(scalerC.transform(XtrC), np.log(ytrC))
predC = np.exp(model_C.predict(scalerC.transform(XteC)))
print(f"Split C: R2={r2(yteC, predC):.4f}, MAPE={mape(yteC, predC):.2f}%")
""",
    "Check the design matrix's condition number (multicollinearity risk -- low "
    "expected, since P/G/X are grid-sampled and largely independent) and the "
    "learned coefficients' relative magnitudes (which input dominates the linear fit).",
    r"""
X_design = scaler.transform(XtrA)
cond_number = np.linalg.cond(np.column_stack([X_design, np.ones(len(X_design))]))
print(f"Design matrix condition number: {cond_number:.2f} (low = well-conditioned; "
      f"large [>1e6] would indicate multicollinearity risk)")
print(f"Coefficients (log-CHF per std-dev of P, G, X): {model_A.coef_}")
print(f"Intercept: {model_A.intercept_:.4f}")
assert cond_number < 100, "Unexpectedly ill-conditioned design matrix for only 3 near-independent grid features"
print("PASS: design matrix well-conditioned, as expected for 3 grid-sampled inputs.")
""",
    r"""
residuals = np.log(yteC) - model_C.predict(scalerC.transform(XteC))
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
axes[0].scatter(yteA, predA, s=5, alpha=0.3)
lims = [0, max(yteA.max(), predA.max())]
axes[0].plot(lims, lims, "r--")
axes[0].set_title(f"Split A parity, R2={r2(yteA,predA):.3f}")
axes[0].set_xlabel("True CHF"); axes[0].set_ylabel("Predicted CHF")
axes[1].hist(residuals, bins=40)
axes[1].set_title("Split C residuals (log-CHF scale)")
axes[1].set_xlabel("log(true) - predicted log(CHF)")
plt.tight_layout()
plt.savefig("../results/model_tests_linear.png", dpi=100)
plt.show()
""",
)

# ============================================================================
# 2. Polynomial Regression (degree 2 & 4, Ridge)
# ============================================================================
build_notebook(
    "Polynomial Regression (deg 2 & 4, Ridge)",
    "test_polynomial_regression.ipynb",
    "Tests both polynomial degrees together since they share the same failure "
    "modes. Key question: does higher degree (4) overfit/destabilize more than "
    "degree 2, especially under extrapolation?",
    "from sklearn.preprocessing import PolynomialFeatures\nfrom sklearn.linear_model import Ridge\nfrom sklearn.pipeline import make_pipeline",
    r"""
results = {}
for degree in [2, 4]:
    scaler = StandardScaler().fit(XtrA)
    model_A = make_pipeline(PolynomialFeatures(degree=degree), Ridge(alpha=1.0))
    model_A.fit(scaler.transform(XtrA), np.log(ytrA))
    predA = np.exp(model_A.predict(scaler.transform(XteA)))

    scalerC = StandardScaler().fit(XtrC)
    model_C = make_pipeline(PolynomialFeatures(degree=degree), Ridge(alpha=1.0))
    model_C.fit(scalerC.transform(XtrC), np.log(ytrC))
    predC = np.exp(model_C.predict(scalerC.transform(XteC)))

    results[degree] = dict(model_A=model_A, model_C=model_C, predA=predA, predC=predC, scalerC=scalerC)
    print(f"Degree {degree} -- Split A: R2={r2(yteA,predA):.4f} | Split C: R2={r2(yteC,predC):.4f}")
""",
    "Check how many polynomial terms each degree generates (feature-count blowup), "
    "and whether Ridge coefficient magnitudes grow with degree (a sign of "
    "extrapolation instability -- large coefficients on high-degree terms can "
    "produce wild predictions far from the training distribution's typical scale).",
    r"""
for degree in [2, 4]:
    poly = results[degree]["model_A"].named_steps["polynomialfeatures"]
    ridge = results[degree]["model_A"].named_steps["ridge"]
    n_terms = poly.n_output_features_
    max_coef = np.abs(ridge.coef_).max()
    print(f"Degree {degree}: {n_terms} polynomial terms, max |coefficient|={max_coef:.3f}")

print("\nExpectation: degree 4 has far more terms and larger coefficient magnitudes "
      "than degree 2 -- consistent with the base notebook's finding that Poly4_Ridge "
      "(raw target) had by far the worst Split-A MAPE (96.6%) among all models, "
      "and that Poly4 fared worse than Poly2 specifically on Split C extrapolation.")
assert results[4]["model_A"].named_steps["polynomialfeatures"].n_output_features_ > \
       results[2]["model_A"].named_steps["polynomialfeatures"].n_output_features_
print("PASS: degree 4 generates strictly more terms than degree 2, as expected.")
""",
    r"""
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
for degree, color in [(2, "tab:blue"), (4, "tab:orange")]:
    axes[0].scatter(yteA, results[degree]["predA"], s=5, alpha=0.3, color=color, label=f"deg{degree}")
    axes[1].scatter(yteC, results[degree]["predC"], s=5, alpha=0.3, color=color, label=f"deg{degree}")
for ax, y in [(axes[0], yteA), (axes[1], yteC)]:
    lims = [0, y.max()]
    ax.plot(lims, lims, "r--")
    ax.legend()
axes[0].set_title("Split A parity"); axes[1].set_title("Split C parity (extrapolation)")
for ax in axes:
    ax.set_xlabel("True CHF"); ax.set_ylabel("Predicted CHF")
plt.tight_layout()
plt.savefig("../results/model_tests_polynomial.png", dpi=100)
plt.show()
""",
)

# ============================================================================
# 3. kNN
# ============================================================================
build_notebook(
    "k-Nearest Neighbors (k=3, distance-weighted)",
    "test_knn.ipynb",
    "A 'grid memorization' diagnostic per the project context, not a serious final "
    "model. Key edge case: distance-weighted kNN gives near-infinite weight to an "
    "exact or near-exact match -- verify this behaves sanely rather than dividing "
    "by zero, and confirm kNN cannot extrapolate (its neighbors are always drawn "
    "from the training range, same failure mode as trees).",
    "from sklearn.neighbors import KNeighborsRegressor",
    r"""
scaler = StandardScaler().fit(XtrA)
model_A = KNeighborsRegressor(n_neighbors=3, weights="distance").fit(scaler.transform(XtrA), np.log(ytrA))
predA = np.exp(model_A.predict(scaler.transform(XteA)))
print(f"Split A: R2={r2(yteA, predA):.4f}, MAPE={mape(yteA, predA):.2f}%")

scalerC = StandardScaler().fit(XtrC)
model_C = KNeighborsRegressor(n_neighbors=3, weights="distance").fit(scalerC.transform(XtrC), np.log(ytrC))
predC = np.exp(model_C.predict(scalerC.transform(XteC)))
print(f"Split C: R2={r2(yteC, predC):.4f}, MAPE={mape(yteC, predC):.2f}%")
""",
    "(1) Query EXACTLY at a training point -- distance-weighted kNN with an exact "
    "match should return that point's value with no numerical warning/error. "
    "(2) Query far beyond the training pressure range and confirm the prediction "
    "plateaus (all 3 nearest neighbors are pinned at the training boundary, so the "
    "prediction cannot continue any trend -- the same structural limitation as trees).",
    r"""
# (1) exact-match query
exact_point = XtrA[0:1]
exact_scaled = scaler.transform(exact_point)
with np.errstate(divide="raise"):
    pred_exact = np.exp(model_A.predict(exact_scaled))
print(f"Exact-match query: predicted={pred_exact[0]:.2f}, true={ytrA[0]:.2f} -- "
      f"{'PASS (near-exact recovery)' if abs(pred_exact[0]-ytrA[0]) < 1.0 else 'FAIL'}")

# (2) far-extrapolation plateau check
G_fixed, X_fixed = 2000.0, 0.0
p_probe = np.array([16000, 17000, 19000, 25000, 50000, 100000], dtype=float)
probe_pts = np.column_stack([p_probe, np.full_like(p_probe, G_fixed), np.full_like(p_probe, X_fixed)])
probe_pred = np.exp(model_C.predict(scalerC.transform(probe_pts)))
print("\nPrediction vs. probe pressure (G=2000, X=0.0), training P maxes out at 16000:")
for p, pred in zip(p_probe, probe_pred):
    print(f"  P={p:>7.0f} kPa -> predicted CHF={pred:.1f}")
print(f"\nPredictions at P=25000, 50000, 100000 are identical: "
      f"{'PASS' if np.allclose(probe_pred[-3:], probe_pred[-1]) else 'FAIL'} "
      f"(kNN cannot extrapolate past its 3 nearest -- fixed -- training neighbors)")
""",
    r"""
fig, ax = plt.subplots(figsize=(7, 5))
ax.plot(p_probe, probe_pred, "o-")
ax.axvline(16000, color="gray", linestyle="--", label="max training P")
ax.set_xscale("log")
ax.set_xlabel("Query pressure (kPa, log scale)")
ax.set_ylabel("Predicted CHF")
ax.set_title("kNN prediction plateaus beyond the training pressure range")
ax.legend()
plt.tight_layout()
plt.savefig("../results/model_tests_knn.png", dpi=100)
plt.show()
""",
)

# ============================================================================
# 4/5. Random Forest & Extra Trees (shared template, different estimator)
# ============================================================================
for tree_name, tree_import, tree_class in [
    ("Random Forest", "from sklearn.ensemble import RandomForestRegressor", "RandomForestRegressor"),
    ("Extra Trees", "from sklearn.ensemble import ExtraTreesRegressor", "ExtraTreesRegressor"),
]:
    fname = f"test_{tree_name.lower().replace(' ', '_')}.ipynb"
    build_notebook(
        tree_name,
        fname,
        f"Excellent interpolator (Splits A/B), expected to collapse under extrapolation "
        f"(Split C) because leaves output a constant learned only from in-range training "
        f"data. This notebook numerically demonstrates that specific mechanism, rather "
        f"than just reporting the aggregate R^2 collapse.",
        tree_import,
        f"""
model_A = {tree_class}(n_estimators=300, random_state=0, n_jobs=-1).fit(XtrA, ytrA)
predA = model_A.predict(XteA)
print(f"Split A: R2={{r2(yteA, predA):.4f}}, MAPE={{mape(yteA, predA):.2f}}%")

model_C = {tree_class}(n_estimators=300, random_state=0, n_jobs=-1).fit(XtrC, ytrC)
predC = model_C.predict(XteC)
print(f"Split C: R2={{r2(yteC, predC):.4f}}, MAPE={{mape(yteC, predC):.2f}}%")
""",
        "Query at increasing pressures far beyond the training max (16,000 kPa) at a "
        "fixed (G, X) and confirm the prediction eventually plateaus to a constant -- "
        "the direct mechanism behind the Split C R^2 collapse.",
        r"""
G_fixed, X_fixed = 2000.0, 0.0
p_probe = np.array([15000, 16000, 17000, 19000, 21000, 30000, 60000, 120000], dtype=float)
probe_pts = np.column_stack([p_probe, np.full_like(p_probe, G_fixed), np.full_like(p_probe, X_fixed)])
probe_pred = model_C.predict(probe_pts)
print("Prediction vs. probe pressure (G=2000, X=0.0), training P maxes out at 16000:")
for p, pred in zip(p_probe, probe_pred):
    print(f"  P={p:>7.0f} kPa -> predicted CHF={pred:.1f}")
print(f"\nPredictions beyond ~21000 kPa are (near-)identical: "
      f"{'PASS' if np.allclose(probe_pred[-3:], probe_pred[-1], rtol=0.02) else 'FAIL'} "
      f"-- confirms the leaf-constant extrapolation mechanism directly.")

# Feature importance sanity check
importances = dict(zip(FEATURES, model_A.feature_importances_))
print(f"\nFeature importances (Split A fit): {importances}")
""",
        rf"""
fig, ax = plt.subplots(figsize=(7, 5))
ax.plot(p_probe, probe_pred, "o-")
ax.axvline(16000, color="gray", linestyle="--", label="max training P")
ax.set_xscale("log")
ax.set_xlabel("Query pressure (kPa, log scale)")
ax.set_ylabel("Predicted CHF")
ax.set_title("{tree_name}: prediction plateaus beyond the training pressure range")
ax.legend()
plt.tight_layout()
plt.savefig("../results/model_tests_{tree_name.lower().replace(' ', '_')}.png", dpi=100)
plt.show()
""",
    )

# ============================================================================
# 6/7. XGBoost & LightGBM
# ============================================================================
for name, imp, ctor in [
    ("XGBoost", "import xgboost as xgb", 'xgb.XGBRegressor(n_estimators=400, max_depth=6, learning_rate=0.05, random_state=0, n_jobs=-1, verbosity=0)'),
    ("LightGBM", "import lightgbm as lgb", 'lgb.LGBMRegressor(n_estimators=400, random_state=0, n_jobs=-1, verbosity=-1)'),
]:
    fname = f"test_{name.lower()}.ipynb"
    build_notebook(
        name,
        fname,
        "Gradient-boosted tree ensemble -- same leaf-constant extrapolation limitation "
        "as Random Forest/Extra Trees, but built additively (each tree corrects the "
        "previous ensemble's residual) rather than by independent averaging. Tests "
        "whether this different construction changes the extrapolation failure mode.",
        imp,
        f"""
model_A = {ctor}.fit(XtrA, ytrA)
predA = model_A.predict(XteA)
print(f"Split A: R2={{r2(yteA, predA):.4f}}, MAPE={{mape(yteA, predA):.2f}}%")

model_C = {ctor}.fit(XtrC, ytrC)
predC = model_C.predict(XteC)
print(f"Split C: R2={{r2(yteC, predC):.4f}}, MAPE={{mape(yteC, predC):.2f}}%")
""",
        "Same far-extrapolation plateau probe as the bagged tree ensembles -- confirms "
        "that additive boosting has the identical structural limitation, since its "
        "final prediction is still a sum of leaf constants, each bounded by its own "
        "tree's training-range leaves.",
        r"""
G_fixed, X_fixed = 2000.0, 0.0
p_probe = np.array([15000, 16000, 17000, 19000, 21000, 30000, 60000, 120000], dtype=float)
probe_pts = np.column_stack([p_probe, np.full_like(p_probe, G_fixed), np.full_like(p_probe, X_fixed)])
probe_pred = model_C.predict(probe_pts)
print("Prediction vs. probe pressure (G=2000, X=0.0), training P maxes out at 16000:")
for p, pred in zip(p_probe, probe_pred):
    print(f"  P={p:>7.0f} kPa -> predicted CHF={pred:.1f}")
print(f"\nPredictions beyond ~21000 kPa are (near-)identical: "
      f"{'PASS' if np.allclose(probe_pred[-3:], probe_pred[-1], rtol=0.02) else 'FAIL'} "
      f"-- additive boosting shares the same leaf-constant extrapolation ceiling as "
      f"bagged trees, despite building the ensemble differently.")
""",
        rf"""
fig, ax = plt.subplots(figsize=(7, 5))
ax.plot(p_probe, probe_pred, "o-")
ax.axvline(16000, color="gray", linestyle="--", label="max training P")
ax.set_xscale("log")
ax.set_xlabel("Query pressure (kPa, log scale)")
ax.set_ylabel("Predicted CHF")
ax.set_title(f"{name}: prediction plateaus beyond the training pressure range")
ax.legend()
plt.tight_layout()
plt.savefig("../results/model_tests_{name.lower()}.png", dpi=100)
plt.show()
""",
    )

# ============================================================================
# 8. Gaussian Process Regression
# ============================================================================
build_notebook(
    "Gaussian Process Regression (Matern-5/2, ARD)",
    "test_gpr.ipynb",
    "Full GPR is O(n^3); this model specifically subsamples training data to 2,000 "
    "points. Key tests: (1) is the subsample reproducible given a fixed seed (needed "
    "for the reproducibility requirements in the project context)? (2) what did the "
    "ARD length scales learn about which input dimension matters most?",
    "from sklearn.gaussian_process import GaussianProcessRegressor\n"
    "from sklearn.gaussian_process.kernels import Matern, ConstantKernel, WhiteKernel\n"
    "import time",
    r"""
def make_gpr(seed):
    kernel = (ConstantKernel(1.0, (1e-2, 1e2))
              * Matern(length_scale=[1.0, 1.0, 1.0], length_scale_bounds=(1e-2, 1e2), nu=2.5)
              + WhiteKernel(1e-3, noise_level_bounds=(1e-6, 1e1)))
    return GaussianProcessRegressor(kernel=kernel, normalize_y=True, random_state=seed, n_restarts_optimizer=1)

def fit_gpr_subsampled(Xtr, ytr, seed, n_sub=2000):
    scaler = StandardScaler().fit(Xtr)
    Xtr_s = scaler.transform(Xtr)
    rng = np.random.RandomState(seed)
    idx = rng.choice(len(Xtr_s), size=min(n_sub, len(Xtr_s)), replace=False)
    t0 = time.time()
    model = make_gpr(seed).fit(Xtr_s[idx], np.log(ytr[idx]))
    fit_time = time.time() - t0
    return model, scaler, idx, fit_time

model_A, scalerA, idxA, tA = fit_gpr_subsampled(XtrA, ytrA, seed=42)
predA = np.exp(model_A.predict(scalerA.transform(XteA)))
print(f"Split A: R2={r2(yteA, predA):.4f}, MAPE={mape(yteA, predA):.2f}%, fit time={tA:.1f}s")

model_C, scalerC, idxC, tC = fit_gpr_subsampled(XtrC, ytrC, seed=42)
predC = np.exp(model_C.predict(scalerC.transform(XteC)))
print(f"Split C: R2={r2(yteC, predC):.4f}, MAPE={mape(yteC, predC):.2f}%, fit time={tC:.1f}s")
""",
    "Reproducibility: refit with the SAME seed and confirm the subsample indices and "
    "resulting predictions are identical. Then inspect the learned ARD length scales "
    "(shorter length scale = model is more sensitive to that input; longer = flatter/"
    "less sensitive).",
    r"""
_, _, idxA_repeat, _ = fit_gpr_subsampled(XtrA, ytrA, seed=42)
print(f"Subsample reproducibility: {'PASS' if np.array_equal(idxA, idxA_repeat) else 'FAIL'} "
      f"(same seed -> same 2000-point subsample indices)")

learned_kernel = model_A.kernel_
print(f"\nLearned kernel (Split A fit): {learned_kernel}")
try:
    matern = learned_kernel.k1.k2  # ConstantKernel * Matern + WhiteKernel structure
    print(f"ARD length scales [P, G, X]: {matern.length_scale}")
    print("Shorter length scale = CHF is more sensitive to changes in that input "
          "(after standardization, so scales are directly comparable across P/G/X).")
except AttributeError:
    print("(kernel structure differs from expected -- inspect learned_kernel directly)")
""",
    r"""
fig, ax = plt.subplots(figsize=(6, 5))
ax.scatter(yteA, predA, s=5, alpha=0.3, label="Split A (interpolation)")
ax.scatter(yteC, predC, s=5, alpha=0.3, label="Split C (extrapolation)", color="tab:orange")
lims = [0, max(yteA.max(), yteC.max())]
ax.plot(lims, lims, "r--")
ax.set_xlabel("True CHF"); ax.set_ylabel("Predicted CHF")
ax.set_title(f"GPR parity: A R2={r2(yteA,predA):.3f}, C R2={r2(yteC,predC):.3f}")
ax.legend()
plt.tight_layout()
plt.savefig("../results/model_tests_gpr.png", dpi=100)
plt.show()
""",
)

# ============================================================================
# 9. MLP
# ============================================================================
build_notebook(
    "Compact MLP (2 hidden layers, 64/32 units)",
    "test_mlp.ipynb",
    "The base notebook found a genuine surprise here: the RAW-target MLP "
    "(R^2=0.850 on Split C) clearly outperformed the LOG-target MLP (R^2=0.342) -- "
    "opposite of the general expectation. Since MLP training is stochastic, this "
    "notebook specifically tests seed sensitivity to see whether that raw-vs-log gap "
    "is a robust finding or a lucky/unlucky single run.",
    "from sklearn.neural_network import MLPRegressor",
    r"""
def fit_mlp(Xtr, ytr, Xte, log_target, seed):
    scaler = StandardScaler().fit(Xtr)
    y_fit = np.log(ytr) if log_target else ytr
    model = MLPRegressor(hidden_layer_sizes=(64, 32), activation="relu", solver="adam",
                          alpha=1e-4, early_stopping=True, validation_fraction=0.15,
                          n_iter_no_change=25, max_iter=3000, random_state=seed)
    model.fit(scaler.transform(Xtr), y_fit)
    pred = model.predict(scaler.transform(Xte))
    return (np.exp(pred) if log_target else pred), model

predA_raw, _ = fit_mlp(XtrA, ytrA, XteA, log_target=False, seed=0)
predA_log, _ = fit_mlp(XtrA, ytrA, XteA, log_target=True, seed=0)
print(f"Split A raw: R2={r2(yteA, predA_raw):.4f} | log: R2={r2(yteA, predA_log):.4f}")

predC_raw, modelC_raw = fit_mlp(XtrC, ytrC, XteC, log_target=False, seed=0)
predC_log, modelC_log = fit_mlp(XtrC, ytrC, XteC, log_target=True, seed=0)
print(f"Split C raw: R2={r2(yteC, predC_raw):.4f} | log: R2={r2(yteC, predC_log):.4f}")
""",
    "Seed-sensitivity test: refit the Split C raw- and log-target MLPs across 5 seeds "
    "and report mean +/- std R^2 for each, to check whether the raw-beats-log finding "
    "holds up robustly or was a single lucky draw.",
    r"""
raw_r2s, log_r2s = [], []
for seed in range(5):
    pr, _ = fit_mlp(XtrC, ytrC, XteC, log_target=False, seed=seed)
    pl, _ = fit_mlp(XtrC, ytrC, XteC, log_target=True, seed=seed)
    raw_r2s.append(r2(yteC, pr))
    log_r2s.append(r2(yteC, pl))

print(f"Split C raw-target R^2 across 5 seeds: {np.mean(raw_r2s):.4f} +/- {np.std(raw_r2s):.4f}  {raw_r2s}")
print(f"Split C log-target R^2 across 5 seeds: {np.mean(log_r2s):.4f} +/- {np.std(log_r2s):.4f}  {log_r2s}")
gap = np.mean(raw_r2s) - np.mean(log_r2s)
print(f"\nMean raw-vs-log gap: {gap:+.4f} -- "
      f"{'raw-beats-log finding HOLDS UP across seeds (robust)' if gap > 0.1 else 'gap is NOT robust -- the single-seed result in the base notebook may have been a lucky/unlucky draw, worth flagging'}")
""",
    r"""
fig, ax = plt.subplots(figsize=(7, 5))
ax.boxplot([raw_r2s, log_r2s], tick_labels=["raw-target", "log-target"])
ax.set_ylabel("Split C R^2 across 5 seeds")
ax.set_title("MLP raw-vs-log target: seed-sensitivity check")
plt.tight_layout()
plt.savefig("../results/model_tests_mlp.png", dpi=100)
plt.show()
""",
)

# ============================================================================
# 10. Trilinear Grid Interpolation
# ============================================================================
build_notebook(
    "Trilinear Grid Interpolation",
    "test_gridinterp.ipynb",
    "The 'look-up table interpolating itself' baseline. Key edge cases: (1) exact "
    "recovery at training grid nodes (should be exact, since linear interpolation "
    "at a known node returns that node's own value), (2) correct linear "
    "extrapolation behavior beyond the grid boundary via bounds_error=False, "
    "fill_value=None (should NOT return NaN).",
    "from scipy.interpolate import RegularGridInterpolator",
    r"""
def fit_grid_interpolator(train_df, log_target=False):
    p_u = np.sort(train_df.P.unique()); g_u = np.sort(train_df.G.unique()); x_u = np.sort(train_df.X.unique())
    cube = np.full((len(p_u), len(g_u), len(x_u)), np.nan)
    p_idx = {v: i for i, v in enumerate(p_u)}
    g_idx = {v: i for i, v in enumerate(g_u)}
    x_idx = {v: i for i, v in enumerate(x_u)}
    vals = np.log(train_df.CHF.values) if log_target else train_df.CHF.values
    for (p, g, x), v in zip(train_df[["P", "G", "X"]].values, vals):
        cube[p_idx[p], g_idx[g], x_idx[x]] = v
    interp = RegularGridInterpolator((p_u, g_u, x_u), cube, method="linear", bounds_error=False, fill_value=None)
    return interp

train_dfC_full = train_dfC  # from shared setup
interp_C = fit_grid_interpolator(train_dfC_full, log_target=False)
predC = interp_C(XteC)
print(f"Split C: R2={r2(yteC, predC):.4f}, MAPE={mape(yteC, predC):.2f}%")
""",
    "(1) Query an interior training grid node exactly -- interpolated value should "
    "equal the table value with (near-)zero error. (2) Query at pressures both just "
    "beyond and far beyond the training boundary and confirm finite, non-NaN, "
    "monotonically-changing (not wildly oscillating) extrapolated values.",
    r"""
# (1) exact node recovery
sample_row = train_dfC_full.iloc[100]
exact_query = np.array([[sample_row.P, sample_row.G, sample_row.X]])
exact_pred = interp_C(exact_query)[0]
print(f"Exact grid-node query: predicted={exact_pred:.4f}, true={sample_row.CHF:.4f} -- "
      f"{'PASS' if abs(exact_pred - sample_row.CHF) < 1e-6 else 'FAIL'}")

# (2) extrapolation behavior beyond the grid
G_fixed, X_fixed = 2000.0, 0.0
p_probe = np.array([16000, 17000, 21000, 30000, 60000], dtype=float)
probe_pts = np.column_stack([p_probe, np.full_like(p_probe, G_fixed), np.full_like(p_probe, X_fixed)])
probe_pred = interp_C(probe_pts)
print("\nExtrapolated predictions (G=2000, X=0.0), training P maxes out at 16000:")
for p, pred in zip(p_probe, probe_pred):
    print(f"  P={p:>7.0f} kPa -> predicted CHF={pred:.1f}")
print(f"\nNo NaN in extrapolated region: {'PASS' if not np.any(np.isnan(probe_pred)) else 'FAIL'} "
      f"(bounds_error=False, fill_value=None correctly linearly extrapolates rather than "
      f"returning NaN)")
print(f"Extrapolation continues the pressure-decreasing trend (not flat like trees): "
      f"{'PASS' if probe_pred[-1] < probe_pred[0] else 'CHECK -- trend direction unexpected'}")
""",
    r"""
fig, ax = plt.subplots(figsize=(7, 5))
ax.plot(p_probe, probe_pred, "o-")
ax.axvline(16000, color="gray", linestyle="--", label="max training P")
ax.set_xlabel("Query pressure (kPa)")
ax.set_ylabel("Predicted CHF")
ax.set_title("Grid interpolator: linear extrapolation continues the trend (unlike trees)")
ax.legend()
plt.tight_layout()
plt.savefig("../results/model_tests_gridinterp.png", dpi=100)
plt.show()
""",
)

print("\nAll model-test notebooks built.")

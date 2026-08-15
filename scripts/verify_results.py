"""
verify_results.py
-----------------
Independent verification of the headline claims, run as a senior-reviewer audit
rather than trusting the notebooks' own reported numbers.

Checks:
  1. Determinism: which models give bit-identical results on re-run (safe to
     report as a single number) vs. which are stochastic (need seed averaging).
  2. Multi-seed Split C comparison of the top contenders -- the test
     SENIOR_REVIEW.md flagged as the highest-priority missing piece, and the
     one that decides which approach is actually defensible to present.
  3. Leakage audit: confirm no test-set information reaches any fit.
"""
import warnings
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import r2_score
from scipy.interpolate import RegularGridInterpolator

warnings.filterwarnings("ignore")

from pathlib import Path

data_path = Path("data/chf_long_clean.csv")
if not data_path.exists():
    data_path = Path("../data/chf_long_clean.csv")
if not data_path.exists():
    data_path = Path("chf_long_clean.csv")

df = pd.read_csv(data_path)
df = df[df.X != 1.0].reset_index(drop=True)
FEATURES = ["P", "G", "X"]

train_dfC = df[df.P <= 16000].reset_index(drop=True)
test_dfC = df[df.P >= 17000].reset_index(drop=True)
XtrC, ytrC = train_dfC[FEATURES].values, train_dfC.CHF.values
XteC, yteC = test_dfC[FEATURES].values, test_dfC.CHF.values

def r2(a, b):
    return float(r2_score(a, b))
def mape(a, b):
    return float(np.mean(np.abs((a - b) / a)) * 100)

print("=" * 78)
print("LEAKAGE AUDIT")
print("=" * 78)
overlap = pd.merge(train_dfC[FEATURES], test_dfC[FEATURES], on=FEATURES, how="inner")
print(f"Rows appearing in BOTH Split C train and test: {len(overlap)}  (must be 0)")
print(f"Train P range: {XtrC[:,0].min():.0f}-{XtrC[:,0].max():.0f} kPa")
print(f"Test  P range: {XteC[:,0].min():.0f}-{XteC[:,0].max():.0f} kPa")
print(f"Test pressures strictly above train max: {bool((XteC[:,0] > XtrC[:,0].max()).all())}  (must be True)")
print(f"Train rows {len(XtrC)} + test rows {len(XteC)} = {len(XtrC)+len(XteC)} (dataset total {len(df)})")

# ---------------------------------------------------------------- determinism
print()
print("=" * 78)
print("DETERMINISM CHECK  (does re-running change the answer?)")
print("=" * 78)

def grid_interp(train_df, log_target, Xq):
    p_u = np.sort(train_df.P.unique()); g_u = np.sort(train_df.G.unique()); x_u = np.sort(train_df.X.unique())
    cube = np.full((len(p_u), len(g_u), len(x_u)), np.nan)
    pi = {v: i for i, v in enumerate(p_u)}; gi = {v: i for i, v in enumerate(g_u)}; xi = {v: i for i, v in enumerate(x_u)}
    vals = np.log(train_df.CHF.values) if log_target else train_df.CHF.values
    for (p, g, x), v in zip(train_df[FEATURES].values, vals):
        cube[pi[p], gi[g], xi[x]] = v
    assert not np.isnan(cube).any()
    f = RegularGridInterpolator((p_u, g_u, x_u), cube, method="linear", bounds_error=False, fill_value=None)
    out = f(Xq)
    return np.exp(out) if log_target else out

def poly2_log(Xtr, ytr, Xte, seed=None):
    sc = StandardScaler().fit(Xtr)
    m = make_pipeline(PolynomialFeatures(2), Ridge(alpha=1.0)).fit(sc.transform(Xtr), np.log(ytr))
    return np.exp(m.predict(sc.transform(Xte)))

det_runs = {
    "GridInterp(raw)": lambda: grid_interp(train_dfC, False, XteC),
    "GridInterp(log)": lambda: grid_interp(train_dfC, True, XteC),
    "Poly2_Ridge(log)": lambda: poly2_log(XtrC, ytrC, XteC),
}
for nm, fn in det_runs.items():
    r1, r2_ = fn(), fn()
    identical = np.array_equal(r1, r2_)
    print(f"{nm:<20} R2={r2(yteC, r1):.4f}  bit-identical on re-run: {identical}")

# ------------------------------------------------------- multi-seed contenders
print()
print("=" * 78)
print("MULTI-SEED SPLIT C  (10 seeds; the decisive comparison)")
print("=" * 78)

SEEDS = list(range(10))

def mlp(Xtr, ytr, Xte, log_target, seed):
    sc = StandardScaler().fit(Xtr)
    y = np.log(ytr) if log_target else ytr
    m = MLPRegressor(hidden_layer_sizes=(64, 32), activation="relu", solver="adam",
                     alpha=1e-4, early_stopping=True, validation_fraction=0.15,
                     n_iter_no_change=25, max_iter=3000, random_state=seed)
    m.fit(sc.transform(Xtr), y)
    p = m.predict(sc.transform(Xte))
    return np.exp(p) if log_target else p

def gated(Xtr, ytr, Xte, log_smooth, seed, margin=2000.0):
    tree = ExtraTreesRegressor(n_estimators=300, random_state=seed, n_jobs=-1).fit(Xtr, ytr)
    sm = mlp(Xtr, ytr, Xte, log_smooth, seed)
    tp = tree.predict(Xte)
    g = np.clip((Xte[:, 0] - Xtr[:, 0].max()) / margin, 0.0, 1.0)
    return (1 - g) * tp + g * sm

contenders = {
    "MLP(raw)":            lambda s: mlp(XtrC, ytrC, XteC, False, s),
    "MLP(log)":            lambda s: mlp(XtrC, ytrC, XteC, True, s),
    "GatedBlend(raw MLP)": lambda s: gated(XtrC, ytrC, XteC, False, s),
    "GatedBlend(log MLP)": lambda s: gated(XtrC, ytrC, XteC, True, s),
    "ExtraTrees(raw)":     lambda s: ExtraTreesRegressor(n_estimators=300, random_state=s, n_jobs=-1).fit(XtrC, ytrC).predict(XteC),
}

rows = []
for nm, fn in contenders.items():
    scores = [r2(yteC, fn(s)) for s in SEEDS]
    mapes = [mape(yteC, fn(s)) for s in SEEDS[:3]]
    rows.append(dict(model=nm, mean=np.mean(scores), std=np.std(scores),
                     min=np.min(scores), max=np.max(scores), mape=np.mean(mapes)))
    print(f"{nm:<22} R2 mean={np.mean(scores):.4f} std={np.std(scores):.4f} "
          f"min={np.min(scores):.4f} max={np.max(scores):.4f}")

# deterministic baselines for the same table
for nm, fn in [("GridInterp(raw)", lambda: grid_interp(train_dfC, False, XteC)),
               ("GridInterp(log)", lambda: grid_interp(train_dfC, True, XteC)),
               ("Poly2_Ridge(log)", lambda: poly2_log(XtrC, ytrC, XteC))]:
    p = fn()
    rows.append(dict(model=nm + " [deterministic]", mean=r2(yteC, p), std=0.0,
                     min=r2(yteC, p), max=r2(yteC, p), mape=mape(yteC, p)))
    print(f"{nm + ' [det]':<22} R2 = {r2(yteC, p):.4f}   (std 0 by construction)   MAPE={mape(yteC,p):.1f}%")

out = pd.DataFrame(rows).sort_values("mean", ascending=False)
out.to_csv("results/split_C_multiseed_verification.csv", index=False)
print()
print("=" * 78)
print("RANKED BY MEAN (worst-case column is what matters for a safety-critical claim)")
print("=" * 78)
print(out.to_string(index=False))
print()
print("Saved -> results/split_C_multiseed_verification.csv")

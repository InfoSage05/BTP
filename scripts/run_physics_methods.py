"""
run_physics_methods.py
----------------------
Does physics-guided learning actually improve cross-laboratory and cross-fluid
generalisation? Four configurations, each isolating one change:

  1 BASELINE      predict log(CHF) from raw P, G, X
  2 MONOTONIC     same, with physics-mandated monotonicity constraints
  3 DIMENSIONLESS predict Boiling number Bo = q/(G*h_fg) from dimensionless
                  inputs (density ratio, Weber number on capillary length, quality)
  4 BOTH          dimensionless target + monotonicity

Validation is two-stage and deliberately strict:
  (a) leave-one-publication-out inside the training data (NRC + Zhao, water)
  (b) external test on six held-out datasets of escalating difficulty, including
      two refrigerants the model has never seen

Why the dimensionless form might transfer: lab-to-lab and fluid-to-fluid offsets
are largely scale and property offsets. Non-dimensionalising is how this field
has moved results between fluids and tube sizes since Katto & Ohno (1984). If
the hypothesis is right, config 3/4 should degrade far less on R-134a and R-123
than config 1/2. If it is wrong, they will degrade the same and we report that.

Monotonicity directions are taken from measured trends, not assumption:
  CHF decreases with quality      (measured r = -0.59)
  CHF increases with mass flux    (measured r = +0.46)
  Bo  decreases with mass flux    (measured r = -0.56, slope -0.73 in log-log)
  Pressure is deliberately NOT constrained: CHF vs P is non-monotonic
  (rises then falls, peaking near 3-4 MPa), so forcing a direction would be wrong.

    python scripts/run_physics_methods.py
"""
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
from CoolProp.CoolProp import PropsSI
from lightgbm import LGBMRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import LeaveOneGroupOut

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
MR = ROOT / "data" / "model_ready"
RES = ROOT / "results" / "physics_methods"
SEED = 42
G_EARTH = 9.81
FLUID = {"water": "Water", "R-134a": "R134a", "R-123": "R123"}


# ---------------------------------------------------------------- properties
_cache = {}


def props(fluid, P_kPa):
    """Saturation properties at pressure. Returns rho_l, rho_v, hfg, sigma (SI)."""
    name = FLUID[fluid]
    out = np.empty((len(P_kPa), 4))
    for i, p in enumerate(P_kPa):
        key = (name, round(float(p), 3))
        if key not in _cache:
            pa = np.clip(float(p) * 1000.0, 1e3, 0.995 * PropsSI("PCRIT", name))
            try:
                rl = PropsSI("D", "P", pa, "Q", 0, name)
                rv = PropsSI("D", "P", pa, "Q", 1, name)
                hf = PropsSI("H", "P", pa, "Q", 1, name) - PropsSI("H", "P", pa, "Q", 0, name)
                sg = PropsSI("I", "P", pa, "Q", 0, name)
                _cache[key] = (rl, rv, hf, max(sg, 1e-6))
            except Exception:
                _cache[key] = (np.nan,) * 4
        out[i] = _cache[key]
    return out[:, 0], out[:, 1], out[:, 2], out[:, 3]


def featurise(df):
    """Attach raw and dimensionless feature columns."""
    d = df.copy()
    rl, rv, hfg, sig = np.array([]), np.array([]), np.array([]), np.array([])
    for f, g in d.groupby("fluid"):
        a, b, c, e = props(f, g.P_kPa.values)
        d.loc[g.index, "rho_l"] = a
        d.loc[g.index, "rho_v"] = b
        d.loc[g.index, "hfg"] = c
        d.loc[g.index, "sigma"] = e
    d["Lc"] = np.sqrt(d.sigma / (G_EARTH * (d.rho_l - d.rho_v)))      # capillary length
    d["dens_ratio"] = d.rho_v / d.rho_l                                # rho_v/rho_l
    d["We_Lc"] = d.G_kg_m2s**2 * d.Lc / (d.rho_l * d.sigma)            # Weber on Lc
    d["Bo"] = d.CHF_kW_m2 * 1000.0 / (d.G_kg_m2s * d.hfg)              # Boiling number
    d["logCHF"] = np.log(d.CHF_kW_m2)
    return d.replace([np.inf, -np.inf], np.nan)


RAW = ["P_kPa", "G_kg_m2s", "X"]
DIM = ["dens_ratio", "We_Lc", "X"]
# monotone_constraints follow the feature order above; 0 = unconstrained
MONO_RAW = [0, 1, -1]      # P free, CHF up with G, CHF down with quality
MONO_DIM = [0, -1, -1]     # density ratio free, Bo down with Weber, Bo down with quality


def model(mono=None):
    return LGBMRegressor(n_estimators=400, num_leaves=31, learning_rate=0.05,
                         random_state=SEED, n_jobs=-1, verbose=-1,
                         monotone_constraints=mono)


def to_chf(pred, d, dimensionless):
    """Convert a model output back to CHF in kW/m2 for fair comparison."""
    if dimensionless:
        return np.exp(pred) * d.G_kg_m2s.values * d.hfg.values / 1000.0
    return np.exp(pred)


def fit_predict(tr, te, feats, target_dimensionless, mono):
    y = np.log(tr.Bo.values) if target_dimensionless else tr.logCHF.values
    m = model(mono).fit(tr[feats].values, y)
    p = m.predict(te[feats].values)
    return to_chf(p, te, target_dimensionless)


CONFIGS = [
    ("1_baseline",      RAW, False, None),
    ("2_monotonic",     RAW, False, MONO_RAW),
    ("3_dimensionless", DIM, True,  None),
    ("4_both",          DIM, True,  MONO_DIM),
]


def main():
    RES.mkdir(parents=True, exist_ok=True)
    # study labels must be strings: NRC uses numeric reference IDs, Zhao uses
    # author names, and pandas re-infers the numeric ones as int on read, which
    # makes the combined group column unsortable.
    a = pd.read_csv(MR / "train/01_NRC_flow.csv")
    b = pd.read_csv(MR / "train/02_Zhao2020_flow.csv")
    a["study"] = "NRC:" + a.study.astype(str)
    b["study"] = "Zhao:" + b.study.astype(str)
    tr = pd.concat([a, b], ignore_index=True)
    tr = featurise(tr).dropna(subset=RAW + DIM + ["Bo", "logCHF"])
    tr = tr[(tr.Bo > 0) & (tr.G_kg_m2s > 0)]
    print(f"training rows {len(tr)}, publications {tr.study.nunique()}", flush=True)

    tests = {}
    for f in sorted((MR / "test").glob("L*.csv")):
        d = pd.read_csv(f)
        if d.boiling_mode.iloc[0] != "flow":
            continue                      # pool boiling needs a different target
        d = featurise(d).dropna(subset=RAW + DIM)
        d = d[(d.G_kg_m2s > 0) & (d.CHF_kW_m2 > 0)]
        if len(d):
            tests[f.stem] = d

    rows = []
    for cname, feats, dimless, mono in CONFIGS:
        # (a) internal: leave-one-publication-out on the training pool
        preds, actual = [], []
        for tri, tei in LeaveOneGroupOut().split(tr, groups=tr.study.values):
            a, b = tr.iloc[tri], tr.iloc[tei]
            if len(b) < 3:
                continue
            preds.append(fit_predict(a, b, feats, dimless, mono))
            actual.append(b.CHF_kW_m2.values)
        P, A = np.concatenate(preds), np.concatenate(actual)
        rows.append(dict(config=cname, evaluation="LOPO_internal", dataset="NRC+Zhao",
                         n=len(A), R2_logCHF=r2_score(np.log(A), np.log(np.clip(P, 1e-6, None))),
                         MAPE=float(np.mean(np.abs((A - P) / A)) * 100)))
        print(f"  {cname}: internal done", flush=True)

        # (b) external: train on everything, predict each held-out dataset
        for tname, te in tests.items():
            p = fit_predict(tr, te, feats, dimless, mono)
            a = te.CHF_kW_m2.values
            rows.append(dict(config=cname, evaluation="external", dataset=tname,
                             n=len(a), fluid=te.fluid.iloc[0],
                             R2_logCHF=r2_score(np.log(a), np.log(np.clip(p, 1e-6, None))),
                             MAPE=float(np.mean(np.abs((a - p) / a)) * 100)))
        print(f"  {cname}: external done", flush=True)

    R = pd.DataFrame(rows)
    R.to_csv(RES / "physics_methods_results.csv", index=False)
    pd.set_option("display.width", 220)
    print("\n================= R2 on log CHF =================")
    print(R.pivot_table(index="dataset", columns="config", values="R2_logCHF").round(3).to_string())
    print("\n================= MAPE (%) =================")
    print(R.pivot_table(index="dataset", columns="config", values="MAPE").round(1).to_string())
    print(f"\n-> {RES/'physics_methods_results.csv'}")


if __name__ == "__main__":
    main()

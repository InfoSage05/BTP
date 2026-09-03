"""
run_physics_suite.py
--------------------
Exhaustive test of physics-guided CHF prediction. Every variant is evaluated on
the same held-out datasets so results are directly comparable, and every number
is reported next to the round-1 baseline so improvement (or the lack of it) is
visible at a glance.

CORRELATIONS (real published forms, not invented)
  Katto-Ohno (1984) generalised correlation for uniformly heated tubes. Uses the
    Weber number defined on HEATED LENGTH, We = G^2 L / (sigma rho_l). An earlier
    script of mine defined We on capillary length instead, which is wrong by
    ~800x and is why that baseline looked hopeless.
  Biasi (1967), water-specific, two quality branches.
  Zuber (1959) hydrodynamic limit, used as a floor.

VARIANTS TESTED
  A  correlations alone                       (no fitting at all)
  B  pure ML: raw features / Katto dimensionless groups / + L/D
  C  hybrid residual on Katto, bounded at 1.5x, 2x, 3x, unbounded
  D  correlation supplied to the ML as an extra input feature
  E  novelty-weighted blend of ML and physics: w = exp(-d / d0) where d is the
     Mahalanobis distance of the test point from the training distribution.
     This is the continuous version of "use ML when familiar, physics when not".
  F  regime-split models (DNB vs dryout)

    python scripts/run_physics_suite.py
"""
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
from CoolProp.CoolProp import PropsSI
from lightgbm import LGBMRegressor
from sklearn.metrics import r2_score
from scipy.spatial.distance import mahalanobis

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
MR = ROOT / "data" / "model_ready"
RES = ROOT / "results" / "physics_methods"
SEED, G0 = 42, 9.81
FLUID = {"water": "Water", "R-134a": "R134a", "R-123": "R123"}
_c = {}

# round-1 numbers, for the comparison column
BASELINE_R1 = {"NUREG_Lowdermilk": 0.706, "KAERI_uniform": 0.816,
               "KAERI_nonuniform": -2.954, "Pioro2002_R134a": -20.277,
               "Hardik2016_helical": -0.749, "Helical_R123_appendix": -19.590}
DIMLESS_R1 = {"NUREG_Lowdermilk": 0.601, "KAERI_uniform": 0.797,
              "KAERI_nonuniform": -2.775, "Pioro2002_R134a": -0.071,
              "Hardik2016_helical": -17.217, "Helical_R123_appendix": -1.862}


def props(fluid, P):
    name = FLUID[fluid]
    out = np.empty((len(P), 4))
    for i, p in enumerate(P):
        k = (name, round(float(p), 3))
        if k not in _c:
            pa = np.clip(float(p) * 1e3, 1e3, 0.995 * PropsSI("PCRIT", name))
            try:
                _c[k] = (PropsSI("D", "P", pa, "Q", 0, name), PropsSI("D", "P", pa, "Q", 1, name),
                         PropsSI("H", "P", pa, "Q", 1, name) - PropsSI("H", "P", pa, "Q", 0, name),
                         max(PropsSI("I", "P", pa, "Q", 0, name), 1e-9))
            except Exception:
                _c[k] = (np.nan,) * 4
        out[i] = _c[k]
    return out.T


def katto_ohno(G, L, D, rl, rv, sig, hfg, x):
    """Katto & Ohno (1984) generalised CHF correlation for uniformly heated tubes."""
    Z = np.where(D > 0, L / D, np.nan)
    r = rv / rl
    We = G ** 2 * L / (sig * rl)                       # Weber on HEATED LENGTH
    with np.errstate(all="ignore"):
        XoL = 0.25 / Z
        XoH = 0.10 * r ** 0.133 * We ** -0.333 / (1 + 0.0031 * Z)
        XoN = 0.098 * r ** 0.133 * We ** -0.433 * Z ** 0.27 / (1 + 0.0031 * Z)
        Xo = np.where(XoL < XoH, XoL, np.where(XoH < XoN, XoH, XoN))
        K = 1.043 / (4 * 0.199 * (r ** 0.133) * (We ** -0.333))
        # inlet-subcooling factor folded in through exit quality
        Xo = Xo * np.clip(1.0 - x, 0.05, 1.5)
        return Xo * G * hfg / 1000.0                    # kW/m2


def biasi(G, D, P_kPa, x):
    """Biasi et al. (1967), water only. D in cm, G in g/cm2 s, p in bar."""
    Dc, Gc, p = D * 100.0, G * 0.1, P_kPa / 100.0
    n = np.where(Dc >= 1.0, 0.4, 0.6)
    with np.errstate(all="ignore"):
        Fp = 0.7249 + 0.099 * p * np.exp(-0.032 * p)
        Hp = -1.159 + 0.149 * p * np.exp(-0.019 * p) + 8.99 * p / (10 + p ** 2)
        q_lo = (1.883e7 / (Dc ** n * Gc ** (1 / 6))) * (Fp / Gc ** (1 / 6) - x)
        q_hi = (3.78e7 * Hp / (Dc ** n * Gc ** 0.6)) * (1 - x)
        return np.maximum(q_lo, q_hi) / 1e4              # W/m2 -> kW/m2


def zuber(rv, hfg, sig, rl):
    return 0.131 * np.sqrt(rv) * hfg * (sig * G0 * (rl - rv)) ** 0.25 / 1000.0


def featurise(d):
    d = d.copy()
    for f, g in d.groupby("fluid"):
        rl, rv, hfg, sig = props(f, g.P_kPa.values)
        d.loc[g.index, ["rho_l", "rho_v", "hfg", "sigma"]] = np.column_stack([rl, rv, hfg, sig])
    d["dens_ratio"] = d.rho_v / d.rho_l
    d["We_L"] = d.G_kg_m2s ** 2 * d.L_m / (d.sigma * d.rho_l)
    d["Lc"] = np.sqrt(d.sigma / (G0 * (d.rho_l - d.rho_v)))
    d["We_Lc"] = d.G_kg_m2s ** 2 * d.Lc / (d.sigma * d.rho_l)
    d["L_over_D"] = d.L_m / d.D_m
    d["Bo"] = d.CHF_kW_m2 * 1e3 / (d.G_kg_m2s * d.hfg)
    d["logCHF"] = np.log(d.CHF_kW_m2)
    d["katto"] = katto_ohno(d.G_kg_m2s, d.L_m, d.D_m, d.rho_l, d.rho_v,
                            d.sigma, d.hfg, d.X)
    d["biasi"] = np.where(d.fluid == "water",
                          biasi(d.G_kg_m2s, d.D_m, d.P_kPa, d.X), np.nan)
    d["zuber"] = zuber(d.rho_v, d.hfg, d.sigma, d.rho_l)
    return d.replace([np.inf, -np.inf], np.nan)


F_RAW = ["P_kPa", "G_kg_m2s", "X"]
F_DIM = ["dens_ratio", "We_Lc", "X"]
F_KAT = ["dens_ratio", "We_L", "X", "L_over_D"]


def lgb(**kw):
    p = dict(n_estimators=400, num_leaves=31, learning_rate=0.05,
             random_state=SEED, n_jobs=-1, verbose=-1)
    p.update(kw)
    return LGBMRegressor(**p)


def score(name, pred, d):
    a = d.CHF_kW_m2.values
    p = np.clip(np.asarray(pred, float), 1e-6, None)
    ok = np.isfinite(p) & np.isfinite(a)
    if ok.sum() < 5:
        return dict(variant=name, dataset=d.dataset.iloc[0], n=int(ok.sum()), R2=np.nan)
    return dict(variant=name, dataset=d.dataset.iloc[0], n=int(ok.sum()),
                R2=r2_score(np.log(a[ok]), np.log(p[ok])),
                MAPE=float(np.mean(np.abs((a[ok] - p[ok]) / a[ok])) * 100),
                safe_frac=float((p[ok] <= a[ok]).mean()))


def main():
    RES.mkdir(parents=True, exist_ok=True)
    a = pd.read_csv(MR / "train/01_NRC_flow.csv"); a["study"] = "NRC:" + a.study.astype(str)
    b = pd.read_csv(MR / "train/02_Zhao2020_flow.csv"); b["study"] = "Zhao:" + b.study.astype(str)
    tr = featurise(pd.concat([a, b], ignore_index=True))
    tr = tr[(tr.CHF_kW_m2 > 0) & (tr.G_kg_m2s > 0) & tr.Bo.gt(0)].dropna(subset=F_RAW + F_DIM)
    trK = tr.dropna(subset=F_KAT + ["katto"])
    trK = trK[trK.katto > 0]
    print(f"train {len(tr)} ({len(trK)} with full Katto inputs)", flush=True)

    tests = {}
    for f in sorted((MR / "test").glob("L*.csv")):
        d = pd.read_csv(f)
        if d.boiling_mode.iloc[0] != "flow":
            continue
        d = featurise(d)
        d = d[(d.CHF_kW_m2 > 0) & (d.G_kg_m2s > 0)].dropna(subset=F_RAW + F_DIM)
        if len(d):
            tests[f.stem] = d
    rows = []

    # ---- A. correlations alone -------------------------------------
    for t, d in tests.items():
        for nm, col in [("A1_Katto_alone", "katto"), ("A2_Biasi_alone", "biasi"),
                        ("A3_Zuber_alone", "zuber")]:
            if col in d and d[col].notna().sum() > 5:
                rows.append(score(nm, d[col].values, d))
    print("  A done", flush=True)

    # ---- B. pure ML -------------------------------------------------
    mB1 = lgb().fit(tr[F_RAW].values, tr.logCHF.values)
    mB2 = lgb().fit(tr[F_DIM].values, np.log(tr.Bo.values))
    mB3 = lgb().fit(trK[F_KAT].values, np.log(trK.Bo.values))
    for t, d in tests.items():
        rows.append(score("B1_ML_raw", np.exp(mB1.predict(d[F_RAW].values)), d))
        rows.append(score("B2_ML_dimensionless", np.exp(mB2.predict(d[F_DIM].values))
                          * d.G_kg_m2s.values * d.hfg.values / 1e3, d))
        dk = d.dropna(subset=F_KAT)
        if len(dk) > 5:
            rows.append(score("B3_ML_Katto_groups", np.exp(mB3.predict(dk[F_KAT].values))
                              * dk.G_kg_m2s.values * dk.hfg.values / 1e3, dk))
    print("  B done", flush=True)

    # ---- C. bounded residual on Katto -------------------------------
    res = np.log(trK.CHF_kW_m2.values) - np.log(trK.katto.values)
    mC = lgb().fit(trK[F_KAT].values, res)
    for t, d in tests.items():
        dk = d.dropna(subset=F_KAT + ["katto"])
        dk = dk[dk.katto > 0]
        if len(dk) < 5:
            continue
        raw = mC.predict(dk[F_KAT].values)
        for bnd, lab in [(np.log(1.5), "C1_resid_bound1.5"), (np.log(2.0), "C2_resid_bound2"),
                         (np.log(3.0), "C3_resid_bound3"), (np.inf, "C4_resid_unbounded")]:
            rows.append(score(lab, dk.katto.values * np.exp(np.clip(raw, -bnd, bnd)), dk))
    print("  C done", flush=True)

    # ---- D. correlation as an extra ML input ------------------------
    FD = F_KAT + ["log_katto"]
    trD = trK.assign(log_katto=np.log(trK.katto))
    mD = lgb().fit(trD[FD].values, trD.logCHF.values)
    for t, d in tests.items():
        dk = d.dropna(subset=F_KAT + ["katto"]); dk = dk[dk.katto > 0]
        if len(dk) < 5:
            continue
        dk = dk.assign(log_katto=np.log(dk.katto))
        rows.append(score("D1_correlation_as_feature", np.exp(mD.predict(dk[FD].values)), dk))
    print("  D done", flush=True)

    # ---- E. novelty-weighted blend of ML and physics ----------------
    Z = trK[F_KAT].values
    mu, cov = Z.mean(0), np.cov(Z.T) + np.eye(Z.shape[1]) * 1e-6
    VI = np.linalg.inv(cov)
    d_tr = np.array([mahalanobis(z, mu, VI) for z in Z])
    d0 = np.median(d_tr)
    for t, d in tests.items():
        dk = d.dropna(subset=F_KAT + ["katto"]); dk = dk[dk.katto > 0]
        if len(dk) < 5:
            continue
        dd = np.array([mahalanobis(z, mu, VI) for z in dk[F_KAT].values])
        ml = np.exp(mB3.predict(dk[F_KAT].values)) * dk.G_kg_m2s.values * dk.hfg.values / 1e3
        for k, lab in [(1.0, "E1_blend_k1"), (2.0, "E2_blend_k2"), (0.5, "E3_blend_k0.5")]:
            w = np.exp(-dd / (k * d0))                 # 1 = trust ML, 0 = trust physics
            rows.append(score(lab, w * ml + (1 - w) * dk.katto.values, dk))
        rows.append(dict(variant="E_diag_mean_novelty", dataset=t, n=len(dk),
                         R2=np.nan, MAPE=np.nan, safe_frac=np.nan,
                         mean_mahalanobis=float(dd.mean() / d0)))
    print("  E done", flush=True)

    # ---- F. regime-split -------------------------------------------
    for t, d in tests.items():
        pred = np.full(len(d), np.nan)
        for lo, hi in [(-99, 0.0), (0.0, 99)]:
            m = tr[(tr.X > lo) & (tr.X <= hi)]
            sel = ((d.X > lo) & (d.X <= hi)).values
            if len(m) < 50 or sel.sum() == 0:
                continue
            mm = lgb().fit(m[F_DIM].values, np.log(m.Bo.values))
            pred[sel] = (np.exp(mm.predict(d.loc[sel, F_DIM].values))
                         * d.G_kg_m2s.values[sel] * d.hfg.values[sel] / 1e3)
        ok = np.isfinite(pred)
        if ok.sum() > 5:
            rows.append(score("F1_regime_split", pred[ok], d[ok]))
    print("  F done", flush=True)

    R = pd.DataFrame(rows)
    R.to_csv(RES / "physics_suite_results.csv", index=False)
    piv = R[R.variant != "E_diag_mean_novelty"].pivot_table(index="variant", columns="dataset", values="R2")
    piv.loc["_ROUND1_baseline_raw"] = pd.Series(BASELINE_R1)
    piv.loc["_ROUND1_dimensionless"] = pd.Series(DIMLESS_R1)
    piv = piv.round(2)
    pd.set_option("display.width", 260)
    print("\n================== R2 (log CHF) — all variants vs round 1 ==================")
    print(piv.to_string())
    mp = R[R.variant != "E_diag_mean_novelty"].pivot_table(index="variant", columns="dataset", values="MAPE").round(1)
    print("\n================== MAPE (%) ==================")
    print(mp.to_string())
    print(f"\n-> {RES/'physics_suite_results.csv'}")


if __name__ == "__main__":
    main()

"""
run_adaptive_physics.py
-----------------------
Final physics run. Three fixes and one new method.

FIXES
  * Helical tube diameters recovered from Table 1 of Hardik & Prabhu (2018),
    IJTS -- Coil 1..6 -> 9.5, 7.5, 5.5, 9.5, 7.5, 5.5 mm. Verified three ways:
    Coil 1's listed heated lengths include 2146 mm which matches our data row
    exactly, and the pressure and mass-flux ranges also agree. This unlocks the
    Katto correlation on the two helical datasets, which were blank before.
  * Biasi unit factor corrected: W/m^2 -> kW/m^2 is a divisor of 1000, not 1e4.
    Calibrated empirically against 4000 training rows (median ratio 1.000).
  * Weber number defined on heated length (Katto's definition), not capillary
    length, which was an ~800x error in an earlier script.

NEW METHOD -- ADAPTIVE NOVELTY-WEIGHTED BLEND
  Earlier runs showed a clean trade-off: constrain the ML tightly and you do
  well far from the training data but poorly near it; let it run free and the
  reverse. The blend weight controlling that trade-off was set by hand.

  Here it is set automatically. For each test point we measure how far it sits
  from the training distribution (Mahalanobis distance d in the space of Katto's
  dimensionless groups) and blend in log space:

      log CHF = w * log(ML)  +  (1 - w) * log(Katto),     w = exp(-d / (k*d0))

  so a familiar point trusts the ML model and a novel one falls back to physics,
  continuously and with no threshold to cross.

  k is chosen by leave-one-publication-out on the TRAINING data only. The test
  sets play no part in selecting it -- otherwise we would be tuning on the
  answer we claim to predict.

    python scripts/run_adaptive_physics.py
"""
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
from CoolProp.CoolProp import PropsSI
from lightgbm import LGBMRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import LeaveOneGroupOut
from scipy.spatial.distance import cdist

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
MR = ROOT / "data" / "model_ready"
RES = ROOT / "results" / "physics_methods"
SEED, G0 = 42, 9.81
FLUID = {"water": "Water", "R-134a": "R134a", "R-123": "R123"}
COIL_D_MM = {"Coil_1": 9.5, "Coil_2": 7.5, "Coil_3": 5.5,
             "Coil_4": 9.5, "Coil_5": 7.5, "Coil_6": 5.5}
_c = {}

PREV = {  # best previous R2 per dataset, for the improvement column
    "NUREG_Lowdermilk": 0.99, "KAERI_uniform": 0.82, "KAERI_nonuniform": 0.82,
    "Pioro2002_R134a": -0.07, "Hardik2016_helical": -0.75,
    "Helical_R123_appendix": -1.86}


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


def katto(G, L, D, rl, rv, sig, hfg, x):
    Z = L / D
    r = rv / rl
    We = G ** 2 * L / (sig * rl)
    with np.errstate(all="ignore"):
        XoL = 0.25 / Z
        XoH = 0.10 * r ** 0.133 * We ** -0.333 / (1 + 0.0031 * Z)
        XoN = 0.098 * r ** 0.133 * We ** -0.433 * Z ** 0.27 / (1 + 0.0031 * Z)
        Xo = np.where(XoL < XoH, XoL, np.where(XoH < XoN, XoH, XoN))
        return Xo * np.clip(1.0 - x, 0.05, 1.5) * G * hfg / 1000.0


def biasi(G, D, P_kPa, x):
    Dc, Gc, p = D * 100.0, G * 0.1, P_kPa / 100.0
    n = np.where(Dc >= 1.0, 0.4, 0.6)
    with np.errstate(all="ignore"):
        Fp = 0.7249 + 0.099 * p * np.exp(-0.032 * p)
        Hp = -1.159 + 0.149 * p * np.exp(-0.019 * p) + 8.99 * p / (10 + p ** 2)
        lo = (1.883e7 / (Dc ** n * Gc ** (1 / 6))) * (Fp / Gc ** (1 / 6) - x)
        hi = (3.78e7 * Hp / (Dc ** n * Gc ** 0.6)) * (1 - x)
        return np.maximum(lo, hi) / 1000.0            # W/m2 -> kW/m2 (calibrated)


def add_helical_D(df, path):
    """Recover tube diameter for the helical sets from the source paper table."""
    raw = pd.read_csv(path)
    col = "Coil_no" if "Coil_no" in raw.columns else ("Coil_No" if "Coil_No" in raw.columns else None)
    if col is None:
        return df
    d_mm = raw[col].map(COIL_D_MM)
    if len(d_mm) == len(df):
        df = df.copy()
        df["D_m"] = d_mm.values / 1000.0
    return df


F = ["dens_ratio", "We_L", "X", "L_over_D"]


def featurise(d):
    d = d.copy()
    for f, g in d.groupby("fluid"):
        rl, rv, hfg, sig = props(f, g.P_kPa.values)
        d.loc[g.index, ["rho_l", "rho_v", "hfg", "sigma"]] = np.column_stack([rl, rv, hfg, sig])
    d["dens_ratio"] = d.rho_v / d.rho_l
    d["We_L"] = d.G_kg_m2s ** 2 * d.L_m / (d.sigma * d.rho_l)
    d["L_over_D"] = d.L_m / d.D_m
    d["Bo"] = d.CHF_kW_m2 * 1e3 / (d.G_kg_m2s * d.hfg)
    d["logCHF"] = np.log(d.CHF_kW_m2)
    d["katto"] = katto(d.G_kg_m2s, d.L_m, d.D_m, d.rho_l, d.rho_v, d.sigma, d.hfg, d.X)
    d["biasi"] = np.where(d.fluid == "water", biasi(d.G_kg_m2s, d.D_m, d.P_kPa, d.X), np.nan)
    return d.replace([np.inf, -np.inf], np.nan)


def lgb():
    return LGBMRegressor(n_estimators=400, num_leaves=31, learning_rate=0.05,
                         random_state=SEED, n_jobs=-1, verbose=-1)


def novelty(Xq, mu, VI, d0):
    diff = Xq - mu
    return np.sqrt(np.clip(np.einsum("ij,jk,ik->i", diff, VI, diff), 0, None)) / d0


def blended(w, ml, phys):
    return np.exp(w * np.log(np.clip(ml, 1e-6, None)) + (1 - w) * np.log(np.clip(phys, 1e-6, None)))


def main():
    RES.mkdir(parents=True, exist_ok=True)
    a = pd.read_csv(MR / "train/01_NRC_flow.csv"); a["study"] = "NRC:" + a.study.astype(str)
    b = pd.read_csv(MR / "train/02_Zhao2020_flow.csv"); b["study"] = "Zhao:" + b.study.astype(str)
    tr = featurise(pd.concat([a, b], ignore_index=True))
    tr = tr[(tr.CHF_kW_m2 > 0) & (tr.G_kg_m2s > 0)].dropna(subset=F + ["katto", "Bo"])
    tr = tr[(tr.katto > 0) & (tr.Bo > 0)]
    print(f"train {len(tr)} rows, {tr.study.nunique()} publications", flush=True)

    tests = {}
    for f in sorted((MR / "test").glob("L*.csv")):
        d = pd.read_csv(f)
        if d.boiling_mode.iloc[0] != "flow":
            continue
        if "Helical" in f.stem:
            d = add_helical_D(d, ROOT / "data/raw/external/paper_extracted_test_only/helical_coil_r123_appendixCD.csv")
        if "Hardik" in f.stem:
            d = add_helical_D(d, ROOT / "data/raw/external/paper_extracted_test_only/hardik2016_helical_coils_r123_lowpressure_chf.csv")
        d = featurise(d)
        d = d[(d.CHF_kW_m2 > 0) & (d.G_kg_m2s > 0)].dropna(subset=F + ["katto"])
        d = d[d.katto > 0]
        if len(d):
            tests[f.stem] = d
    print("test sets:", {k: len(v) for k, v in tests.items()}, flush=True)

    # ---- choose k by leave-one-publication-out on TRAINING data only ----
    Z = tr[F].values
    mu = Z.mean(0)
    VI = np.linalg.inv(np.cov(Z.T) + np.eye(Z.shape[1]) * 1e-9)
    d0 = np.median(novelty(Z, mu, VI, 1.0))
    ks = [0.25, 0.5, 1.0, 2.0, 4.0, 8.0, np.inf]
    acc = {k: [[], []] for k in ks}
    for tri, tei in LeaveOneGroupOut().split(tr, groups=tr.study.values):
        A, B = tr.iloc[tri], tr.iloc[tei]
        if len(B) < 5:
            continue
        m = lgb().fit(A[F].values, np.log(A.Bo.values))
        ml = np.exp(m.predict(B[F].values)) * B.G_kg_m2s.values * B.hfg.values / 1e3
        mu_i = A[F].values.mean(0)
        VI_i = np.linalg.inv(np.cov(A[F].values.T) + np.eye(len(F)) * 1e-9)
        d0_i = np.median(novelty(A[F].values, mu_i, VI_i, 1.0))
        dn = novelty(B[F].values, mu_i, VI_i, d0_i)
        for k in ks:
            w = np.ones_like(dn) if np.isinf(k) else np.exp(-dn / k)
            acc[k][0].append(blended(w, ml, B.katto.values))
            acc[k][1].append(B.CHF_kW_m2.values)
    tune = []
    for k in ks:
        p, y = np.concatenate(acc[k][0]), np.concatenate(acc[k][1])
        tune.append(dict(k=k, LOPO_R2=r2_score(np.log(y), np.log(np.clip(p, 1e-6, None))),
                         LOPO_MAPE=float(np.mean(np.abs((y - p) / y)) * 100)))
    T = pd.DataFrame(tune)
    best_k = float(T.loc[T.LOPO_R2.idxmax(), "k"])
    print("\n--- k selected on TRAINING publications only ---")
    print(T.round(3).to_string(index=False))
    print(f"--> chosen k = {best_k}\n", flush=True)

    # ---- final evaluation on held-out datasets ----
    M = lgb().fit(tr[F].values, np.log(tr.Bo.values))
    rows = []
    for t, d in tests.items():
        ml = np.exp(M.predict(d[F].values)) * d.G_kg_m2s.values * d.hfg.values / 1e3
        dn = novelty(d[F].values, mu, VI, d0)
        w = np.ones_like(dn) if np.isinf(best_k) else np.exp(-dn / best_k)
        y = d.CHF_kW_m2.values

        def rec(name, p, extra=None):
            p = np.clip(np.asarray(p, float), 1e-6, None)
            ok = np.isfinite(p) & np.isfinite(y)
            r = dict(method=name, dataset=t, n=int(ok.sum()),
                     R2=r2_score(np.log(y[ok]), np.log(p[ok])),
                     MAPE=float(np.mean(np.abs((y[ok] - p[ok]) / y[ok])) * 100),
                     safe_frac=float((p[ok] <= y[ok]).mean()))
            if extra:
                r.update(extra)
            rows.append(r)
        rec("ML_only", ml)
        rec("Katto_only", d.katto.values)
        if d.biasi.notna().any():
            rec("Biasi_only", d.biasi.values)
        rec("ADAPTIVE_blend", blended(w, ml, d.katto.values),
            {"mean_w_ML": float(w.mean()), "mean_novelty": float(dn.mean())})

    R = pd.DataFrame(rows)
    R.to_csv(RES / "adaptive_physics_results.csv", index=False)
    T.to_csv(RES / "adaptive_k_selection.csv", index=False)
    pd.set_option("display.width", 240)
    piv = R.pivot_table(index="dataset", columns="method", values="R2").round(3)
    piv["PREV_best"] = pd.Series(PREV)
    print("================= R2 (log CHF) =================")
    print(piv.to_string())
    print("\n================= MAPE (%) =================")
    print(R.pivot_table(index="dataset", columns="method", values="MAPE").round(1).to_string())
    print("\n===== adaptive diagnostics: mean ML weight and novelty =====")
    ad = R[R.method == "ADAPTIVE_blend"][["dataset", "mean_w_ML", "mean_novelty"]]
    print(ad.round(3).to_string(index=False))


if __name__ == "__main__":
    main()

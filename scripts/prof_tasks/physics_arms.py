"""Phase 2b: physics baselines, physics-informed ML, and the iterative closure.

Four kinds of arm:

  PHYS_*    closed-form correlations, ZERO fitted parameters. The reference every
            learned model has to beat.
  PIML_*    ML trained in dimensionless space (Katto groups -> boiling number),
            which is the space where different fluids map onto the same point.
  BOUND_*   ML residual correction on a physics baseline, clipped to a maximum
            multiplicative factor so it degrades into the correlation off-manifold.
  ITER_*    the iterative closure: a local-form model solved jointly with the
            energy balance from INLET conditions only, the way reactor codes
            actually use the look-up table. This makes a local-form model
            legitimately predictive instead of circular.

Correlation implementations are reused from the existing pipeline rather than
rewritten, because those have already had real bugs found and fixed in them
(Weber number on heated length, Biasi's dimensional divisor). Katto's C_Kc
constant is tagged [S] UNVERIFIED upstream; that carries through to these results.
"""

from __future__ import annotations

import os
import sys
import warnings

import numpy as np
import pandas as pd
from CoolProp.CoolProp import PropsSI
from scipy.interpolate import RegularGridInterpolator
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold, train_test_split

import datasets as D

warnings.filterwarnings("ignore")
ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
sys.path.insert(0, os.path.join(ROOT, "physics_pipeline", "scripts"))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from physics.correlations import katto_ohno_chf  # noqa: E402
from chf_physics import biasi_chf                # noqa: E402

OUT = os.path.join(ROOT, "results", "prof_tasks")
CP = {"water": "Water", "R-123": "R123", "R-134a": "R134a"}


# ------------------------------------------------------------------ properties
_cache: dict = {}


def props(fluid: str, P_kPa: np.ndarray) -> pd.DataFrame:
    """Saturation properties at each row's pressure. SI units."""
    name = CP[fluid]
    pc = PropsSI("Pcrit", name)
    rows = []
    for p in np.asarray(P_kPa, float):
        key = (name, round(p, 3))
        if key not in _cache:
            pa = min(max(p * 1e3, 1e3), pc * 0.999)
            try:
                hf = PropsSI("H", "P", pa, "Q", 0, name)
                hg = PropsSI("H", "P", pa, "Q", 1, name)
                _cache[key] = dict(
                    rho_l=PropsSI("D", "P", pa, "Q", 0, name),
                    rho_g=PropsSI("D", "P", pa, "Q", 1, name),
                    sigma=PropsSI("I", "P", pa, "Q", 0, name),
                    h_fg=hg - hf, h_f=hf,
                    Tsat=PropsSI("T", "P", pa, "Q", 0, name), P_crit=pc)
            except Exception:
                _cache[key] = dict(rho_l=np.nan, rho_g=np.nan, sigma=np.nan,
                                   h_fg=np.nan, h_f=np.nan, Tsat=np.nan, P_crit=pc)
        rows.append(_cache[key])
    return pd.DataFrame(rows)


def enrich(ds) -> pd.DataFrame:
    """Attach fluid properties, inlet subcooling and dimensionless groups."""
    df = ds.df.copy()
    fluid = ds.fluid if ds.fluid in CP else "water"
    pr = props(fluid, df["P_kPa"].to_numpy())
    for c in pr.columns:
        df[c] = pr[c].to_numpy()

    # inlet subcooling enthalpy [J/kg], >= 0. Only where genuinely reported.
    if "dHin_sub_kJkg" in df:
        df["dh_sub"] = np.clip(df["dHin_sub_kJkg"] * 1e3, 0, None)
        df["dh_sub_known"] = True
    elif "Tin_C" in df:
        hin = []
        for t, p in zip(df["Tin_C"], df["P_kPa"]):
            try:
                hin.append(PropsSI("H", "T", t + 273.15, "P", p * 1e3, CP[fluid]))
            except Exception:
                hin.append(np.nan)
        df["dh_sub"] = np.clip(df["h_f"] - np.array(hin), 0, None)
        df["dh_sub_known"] = True
    else:
        df["dh_sub"] = 0.0            # saturated-inlet limit
        df["dh_sub_known"] = False

    d_m = df["D_mm"] / 1e3 if "D_mm" in df else pd.Series(0.008, index=df.index)
    l_m = df["L_mm"] / 1e3 if "L_mm" in df else pd.Series(1.0, index=df.index)
    df["_D_m"], df["_L_m"] = d_m, l_m

    # dimensionless groups -- the transferable feature space
    G = df["G_kg_m2s"].clip(lower=1e-6)
    df["Katto_We_inv"] = df["sigma"] * df["rho_l"] / (G ** 2 * l_m.clip(lower=1e-9))
    df["rho_ratio_gl"] = df["rho_g"] / df["rho_l"]
    df["LtoD_dim"] = l_m / d_m.clip(lower=1e-9)
    df["dh_sub_star"] = df["dh_sub"] / df["h_fg"]
    df["P_red"] = df["P_kPa"] * 1e3 / df["P_crit"]
    df["Bo_actual"] = df["CHF_kW_m2"] * 1e3 / (G * df["h_fg"])
    return df


# ------------------------------------------------------------------ baselines
def phys_katto(df) -> np.ndarray:
    q = katto_ohno_chf(df["G_kg_m2s"].to_numpy(), df["_D_m"].to_numpy(),
                       df["_L_m"].to_numpy(), df["rho_l"].to_numpy(),
                       df["rho_g"].to_numpy(), df["sigma"].to_numpy(),
                       df["h_fg"].to_numpy(), df["dh_sub"].to_numpy())
    return np.asarray(q, float) / 1e3


def phys_biasi(df) -> np.ndarray:
    """Biasi with each row's OWN diameter.

    biasi_chf picks its exponent with a scalar branch (alpha = 0.6 below 1 cm,
    else 0.4), so it must be called per distinct diameter rather than with an
    array. Passing the 8 mm default for every dataset would have penalised the
    correlation for a mistake in the caller, not in the correlation.
    """
    d_m = df["_D_m"].to_numpy(float)
    P = df["P_kPa"].to_numpy(float)
    G = df["G_kg_m2s"].to_numpy(float)
    X = df["X"].to_numpy(float)
    out = np.empty(len(df), float)
    for d in np.unique(np.round(d_m, 6)):
        m = np.round(d_m, 6) == d
        out[m] = np.asarray(biasi_chf(P[m], G[m], X[m], float(d)), float)
    return out


_lut = None


def phys_lut(df) -> np.ndarray:
    """Trilinear interpolation of the 2006 look-up table at (P, G, X)."""
    global _lut
    if _lut is None:
        t = pd.read_csv(os.path.join(D.SHARE, "05_lookup_table_reference",
                                     "groeneveld_2006_lut_long_format.csv"))
        P, G, X = [np.sort(t[c].unique()) for c in ("P", "G", "X")]
        grid = (t.set_index(["P", "G", "X"])["CHF"]
                 .reindex(pd.MultiIndex.from_product([P, G, X])).to_numpy()
                 .reshape(len(P), len(G), len(X)))
        _lut = RegularGridInterpolator((P, G, X), grid, bounds_error=False,
                                       fill_value=None)
    pts = np.column_stack([df["P_kPa"], df["G_kg_m2s"], df["X"]])
    return np.clip(_lut(pts), 1e-3, None)


PHYSICS = {"PHYS_KattoOhno": phys_katto, "PHYS_Biasi": phys_biasi,
           "PHYS_LUT2006": phys_lut}

DIMLESS = ["Katto_We_inv", "rho_ratio_gl", "LtoD_dim", "dh_sub_star", "P_red"]


# ------------------------------------------------------------------ metrics
def metrics(y, p, n_train):
    y = np.asarray(y, float)
    p = np.clip(np.nan_to_num(np.asarray(p, float), nan=np.nanmedian(y)), 1e-6, None)
    rel = np.abs(p - y) / np.abs(y)
    ss = ((y - y.mean()) ** 2).sum()
    return dict(n_test=len(y), n_train=n_train,
                R2=1.0 - ((y - p) ** 2).sum() / ss if ss > 0 else np.nan,
                within_10pct=float((rel <= .10).mean()),
                within_20pct=float((rel <= .20).mean()),
                MAPE_pct=float(rel.mean() * 100),
                RMSE=float(np.sqrt(np.mean((y - p) ** 2))),
                safe_side_frac=float((p <= y).mean()))


def _gb():
    return HistGradientBoostingRegressor(random_state=0)


def learned_arms(df, tr, te, has_X):
    """PIML (dimensionless), BOUND (clipped residual), ITER (energy-balance closure)."""
    out = {}
    y = df["CHF_kW_m2"].to_numpy(float)
    G = df["G_kg_m2s"].clip(lower=1e-6).to_numpy()
    hfg = df["h_fg"].to_numpy()

    # --- PIML: predict the boiling number from dimensionless groups
    feats = DIMLESS + (["X"] if has_X else [])
    A = df[feats].to_numpy(float)
    ok = np.isfinite(A).all(1) & np.isfinite(df["Bo_actual"]) & (df["Bo_actual"] > 0)
    tr_ok = tr[ok[tr]]
    if len(tr_ok) > 10:
        m = _gb().fit(A[tr_ok], np.log(df["Bo_actual"].to_numpy()[tr_ok]))
        out["PIML_dimensionless"] = np.exp(m.predict(A[te])) * G[te] * hfg[te] / 1e3

    # --- BOUND: clipped multiplicative correction on Katto
    base = np.clip(phys_katto(df), 1e-3, None)
    ratio = np.log(np.clip(y / base, 1e-6, None))
    B = df[DIMLESS].to_numpy(float)
    okb = np.isfinite(B).all(1) & np.isfinite(ratio)
    trb = tr[okb[tr]]
    if len(trb) > 10:
        m = _gb().fit(B[trb], ratio[trb])
        corr = m.predict(B[te])
        for b in (1.5, 2.0, 3.0):
            out[f"BOUND_Katto_x{b}"] = base[te] * np.exp(np.clip(corr, -np.log(b), np.log(b)))
        out["BOUND_Katto_unbounded"] = base[te] * np.exp(corr)

    # --- ITER: local-form model closed against the energy balance from inlet only
    if has_X and df["dh_sub_known"].iloc[0]:
        loc = ["P_kPa", "G_kg_m2s", "X"] + (["_D_m", "_L_m"] if "D_mm" in df else [])
        L = df[loc].to_numpy(float)
        okl = np.isfinite(L).all(1)
        trl = tr[okl[tr]]
        if len(trl) > 10:
            m = _gb().fit(L[trl], np.log(y[trl]))
            sub = df.iloc[te]
            x = np.zeros(len(te))                       # start from saturated
            for _ in range(40):                         # fixed-point iteration
                Lt = sub[loc].copy()
                Lt["X"] = x
                q = np.exp(m.predict(Lt.to_numpy(float)))
                # energy balance: x_out = -dh_sub/h_fg + 4 q L / (G D h_fg)
                x_new = (-sub["dh_sub"].to_numpy() / sub["h_fg"].to_numpy()
                         + 4 * q * 1e3 * sub["_L_m"].to_numpy()
                         / (sub["G_kg_m2s"].to_numpy() * sub["_D_m"].to_numpy()
                            * sub["h_fg"].to_numpy()))
                x_new = np.clip(x_new, -3, 1.5)
                if np.max(np.abs(x_new - x)) < 1e-4:
                    x = x_new
                    break
                x = 0.5 * x + 0.5 * x_new               # damped for stability
            Lt = sub[loc].copy()
            Lt["X"] = x
            out["ITER_closure_GB"] = np.exp(m.predict(Lt.to_numpy(float)))
    return out


# ------------------------------------------------------------------ runner
def main():
    rows = []
    names = sys.argv[1:] or list(D.LOADERS)
    for name in names:
        ds = D.LOADERS[name]()
        if ds.fluid not in CP:
            print(f"skip {name}: mixed fluid")
            continue
        df = enrich(ds).reset_index(drop=True)
        df = df[df["CHF_kW_m2"] > 0].reset_index(drop=True)
        y = df["CHF_kW_m2"].to_numpy(float)
        has_X = "X" in df.columns
        idx = np.arange(len(df))
        print(f"\n=== {name} ({len(df)} rows, {ds.fluid}) "
              f"inlet_subcooling_known={bool(df['dh_sub_known'].iloc[0])} ===")

        # closed-form arms: no fitting, so scored on every row at once
        for pname, fn in PHYSICS.items():
            if pname in ("PHYS_Biasi", "PHYS_LUT2006") and ds.fluid != "water":
                continue
            if not has_X and pname in ("PHYS_Biasi", "PHYS_LUT2006"):
                continue
            try:
                p = fn(df)
                rows.append(dict(dataset=name, n=len(df), fluid=ds.fluid, model=pname,
                                 split="all_rows_no_fitting", **metrics(y, p, 0)))
                print(f"   {pname:<22} R2={rows[-1]['R2']:+.3f} "
                      f"within10={rows[-1]['within_10pct']*100:.1f}%")
            except Exception as e:
                print(f"   {pname}: {e}")

        # learned arms under both random splits and LOSO
        splits = [("80/20", 0.2), ("70/30", 0.3)]
        for label, ts in splits:
            acc: dict = {}
            for s in range(10 if len(df) < 8000 else 3):
                tr, te = train_test_split(idx, test_size=ts, random_state=s)
                for k, p in learned_arms(df, tr, te, has_X).items():
                    acc.setdefault(k, []).append(metrics(y[te], p, len(tr)))
            for k, v in acc.items():
                a = pd.DataFrame(v)
                rows.append(dict(dataset=name, n=len(df), fluid=ds.fluid, model=k,
                                 split=label, n_seeds=len(v),
                                 **{c: a[c].mean() for c in a.columns},
                                 R2_sd=a["R2"].std()))
                print(f"   {k:<22} {label} R2={a['R2'].mean():+.3f} "
                      f"within10={a['within_10pct'].mean()*100:.1f}%")
        if ds.group and df[ds.group].nunique() >= 2:
            g = df[ds.group].to_numpy()
            gk = GroupKFold(n_splits=min(df[ds.group].nunique(), 10))
            acc = {}
            for tr, te in gk.split(idx, y, g):
                for k, p in learned_arms(df, tr, te, has_X).items():
                    acc.setdefault(k, []).append(metrics(y[te], p, len(tr)))
            for k, v in acc.items():
                a = pd.DataFrame(v)
                rows.append(dict(dataset=name, n=len(df), fluid=ds.fluid, model=k,
                                 split="LOSO", n_seeds=len(v),
                                 **{c: a[c].mean() for c in a.columns},
                                 R2_sd=a["R2"].std()))
                print(f"   {k:<22} LOSO  R2={a['R2'].mean():+.3f} "
                      f"within10={a['within_10pct'].mean()*100:.1f}%")
        pd.DataFrame(rows).to_csv(os.path.join(OUT, "physics_results.csv"), index=False)
    print(f"\nDONE -> results/prof_tasks/physics_results.csv ({len(rows)} rows)")


if __name__ == "__main__":
    main()

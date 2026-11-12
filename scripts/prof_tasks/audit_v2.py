"""Independent adversarial audit + feature-space reconstruction.

Three questions the existing work never answered:

 Q-A  DECOMPOSITION. The published comparison is "pipeline vs pure_physics",
      but pure_physics was HARDCODED to Katto while the pipeline SELECTS its
      backbone. So part of the reported advantage may be backbone selection
      rather than the machine-learning correction. Decompose it.

 Q-B  FEATURE SUFFICIENCY. The model sees 5 dimensionless groups built from
      density, surface tension and latent heat. It sees NO viscosity, NO
      thermal conductivity, NO specific heat and NO gravity -- so no Reynolds
      number, no Prandtl number, no Bond number. For a convective boiling
      problem that is a conspicuous omission. Test whether adding them helps.

 Q-C  IS LtoD LEAKING? Deferred earlier. L and D both appear in the energy
      balance, so L/D may carry a weak trace of it. Drop it and measure.

Everything is evaluated by leave-one-dataset-out, which is the only protocol
in this project that is not contaminated by source confounding.
"""
from __future__ import annotations
import os, sys, json, warnings
import numpy as np, pandas as pd
from CoolProp.CoolProp import PropsSI
from sklearn.ensemble import ExtraTreesRegressor

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import datasets as D, physics_arms as P, pipeline as PL

OUT = P.OUT
FLOW = ["D1_NRC", "D2_Zhao2020", "D3_KAERI_uniform", "D4_KAERI_nonuniform",
        "D5_Helical_R123", "D6_Hardik_helical_water", "D7_Hardik_straight_R123",
        "D8_Pioro_R134a"]
G_ACC = 9.80665  # standard gravity, m/s^2

# ---------------------------------------------------------------- properties
_tcache: dict = {}

def transport(fluid: str, P_kPa: np.ndarray) -> pd.DataFrame:
    """Viscosity, thermal conductivity and specific heat at saturation.

    These are the properties the existing feature set omits entirely. Without
    them there is no Reynolds number and no Prandtl number, i.e. no measure of
    turbulence and no measure of the momentum/thermal boundary-layer ratio.
    """
    name = P.cp_name(fluid)
    pc = PropsSI("Pcrit", name)
    rows = []
    for p in np.asarray(P_kPa, float):
        key = (name, round(p, 3))
        if key not in _tcache:
            pa = min(max(p * 1e3, 1e3), pc * 0.999)
            try:
                _tcache[key] = dict(
                    mu_l=PropsSI("V", "P", pa, "Q", 0, name),
                    mu_g=PropsSI("V", "P", pa, "Q", 1, name),
                    k_l=PropsSI("L", "P", pa, "Q", 0, name),
                    cp_l=PropsSI("C", "P", pa, "Q", 0, name))
            except Exception:
                _tcache[key] = dict(mu_l=np.nan, mu_g=np.nan, k_l=np.nan, cp_l=np.nan)
        rows.append(_tcache[key])
    return pd.DataFrame(rows)


EXTRA = ["Re_l", "Pr_l", "We_D", "Bond", "Fr", "mu_ratio"]

def add_extended(df: pd.DataFrame, fluid: str) -> pd.DataFrame:
    tr = transport(fluid, df["P_kPa"].to_numpy())
    for c in tr.columns:
        df[c] = tr[c].to_numpy()
    G = df["G_kg_m2s"].clip(lower=1e-6)
    d = df["_D_m"].clip(lower=1e-9)
    drho = (df["rho_l"] - df["rho_g"]).clip(lower=1e-9)
    # Reynolds number, all-liquid basis: inertia vs viscous force
    df["Re_l"] = G * d / df["mu_l"].clip(lower=1e-12)
    # Prandtl number: momentum diffusivity vs thermal diffusivity
    df["Pr_l"] = df["mu_l"] * df["cp_l"] / df["k_l"].clip(lower=1e-12)
    # Weber number on DIAMETER (the existing one is on heated LENGTH)
    df["We_D"] = G ** 2 * d / (df["rho_l"] * df["sigma"].clip(lower=1e-12))
    # Bond number: buoyancy vs surface tension -- the confinement scale
    df["Bond"] = G_ACC * drho * d ** 2 / df["sigma"].clip(lower=1e-12)
    # Froude number: inertia vs gravity
    df["Fr"] = G ** 2 / (df["rho_l"] ** 2 * G_ACC * d)
    df["mu_ratio"] = df["mu_g"] / df["mu_l"].clip(lower=1e-12)
    return df


def build_corpus() -> pd.DataFrame:
    frames = []
    for name in FLOW:
        ds = D.LOADERS[name]()
        f = PL.prepare(ds)
        f = add_extended(f, ds.fluid)
        f["dataset"] = name
        f["fluid_name"] = ds.fluid
        frames.append(f)
    keep = (["dataset", "fluid_name", "CHF_kW_m2", "phys_katto", "phys_biasi",
             "phys_lut", "corr_spread"] + P.DIMLESS + EXTRA)
    df = pd.concat([f.reindex(columns=keep) for f in frames], ignore_index=True)
    return df.dropna(subset=["CHF_kW_m2", "phys_katto"]).reset_index(drop=True)


# ---------------------------------------------------------------- feature sets
BASE = list(P.DIMLESS)
FEATURE_SETS = {
    "F0_base5":            BASE,
    "F1_base+Re_Pr":       BASE + ["Re_l", "Pr_l"],
    "F2_base+WeD_Bond_Fr": BASE + ["We_D", "Bond", "Fr"],
    "F3_all11":            BASE + EXTRA,
    "F4_base_minus_LtoD":  [f for f in BASE if f != "LtoD_dim"],
    "F5_no_geometry":      ["rho_ratio_gl", "dh_sub_star", "P_red"],
    "F6_base+Re":          BASE + ["Re_l"],
    "F7_base+Bond":        BASE + ["Bond"],
}


def r2(y, p):
    y = np.asarray(y, float); p = np.asarray(p, float)
    ss = ((y - y.mean()) ** 2).sum()
    return float(1 - ((y - p) ** 2).sum() / ss) if ss > 0 else np.nan

def w10(y, p):
    return float((np.abs(np.asarray(p, float) - y) / np.abs(y) <= 0.10).mean())


def run_lodo(df: pd.DataFrame, feats: list, seeds=(0, 1, 2)) -> pd.DataFrame:
    """LODO with a given feature set. Returns per-dataset arm results."""
    old = PL.DIM
    PL.DIM = feats                       # pipeline reads DIM at call time
    rows = []
    try:
        for held in FLOW:
            tr = df[df.dataset != held]
            te = df[df.dataset == held]
            if len(te) < 5:
                continue
            y = te["CHF_kW_m2"].to_numpy(float)
            accs = {k: [] for k in ("PIPE", "PIPE_unbounded", "backbone_alone",
                                    "katto_alone", "pooled_ML")}
            bb_used = None
            for s in seeds:
                pipe = PL.CHFPipeline(alpha=0.10, seed=s).fit(tr)
                bb_used = pipe.backbone_
                accs["PIPE"].append(pipe.predict(te)["chf"].to_numpy())
                # --- unbounded: same model, cap removed
                base = np.clip(te[pipe.backbone_].to_numpy(float), 1e-3, None)
                X = np.nan_to_num(te[feats].to_numpy(float), nan=0.0)
                accs["PIPE_unbounded"].append(base * np.exp(pipe.model_.predict(X)))
                accs["backbone_alone"].append(base)
                accs["katto_alone"].append(te["phys_katto"].to_numpy(float))
                m = ExtraTreesRegressor(n_estimators=300, n_jobs=-1, random_state=s)
                m.fit(np.nan_to_num(tr[feats].to_numpy(float), nan=0.0),
                      np.log(tr["CHF_kW_m2"].to_numpy(float)))
                accs["pooled_ML"].append(np.exp(m.predict(X)))
            for arm, preds in accs.items():
                rr = [r2(y, p) for p in preds]; ww = [w10(y, p) for p in preds]
                rows.append(dict(features="|".join(feats), n_feat=len(feats),
                                 held_out=held, arm=arm, backbone=bb_used,
                                 R2=float(np.mean(rr)), R2_sd=float(np.std(rr)),
                                 w10=float(np.mean(ww)) * 100, n_test=len(te)))
    finally:
        PL.DIM = old
    return pd.DataFrame(rows)


def main():
    print("building corpus with extended properties ...", flush=True)
    df = build_corpus()
    print(f"corpus {len(df)} rows, {df.dataset.nunique()} datasets, "
          f"fluids {sorted(df.fluid_name.unique())}", flush=True)
    for c in EXTRA:
        v = df[c].replace([np.inf, -np.inf], np.nan)
        print(f"  {c:10s} finite {v.notna().mean()*100:5.1f}%  "
              f"range [{v.min():.3g}, {v.max():.3g}]", flush=True)

    all_res = []
    for tag, feats in FEATURE_SETS.items():
        print(f"\n=== {tag}  ({len(feats)} features) ===", flush=True)
        r = run_lodo(df, feats)
        r["set"] = tag
        all_res.append(r)
        piv = r.pivot_table(index="held_out", columns="arm", values="R2")
        for arm in ("katto_alone", "backbone_alone", "pooled_ML",
                    "PIPE_unbounded", "PIPE"):
            if arm in piv:
                print(f"  {arm:16s} median R2 {piv[arm].median():+.4f}  "
                      f"worst {piv[arm].min():+.4f}", flush=True)
        out = pd.concat(all_res, ignore_index=True)
        out.to_csv(os.path.join(OUT, "audit_v2_features.csv"), index=False)
    print("\nwrote audit_v2_features.csv", flush=True)


if __name__ == "__main__":
    main()

"""
run_physics_methods2.py
-----------------------
Round 2. Adds the agreed methods that round 1 did not cover, plus the
missing-variable test that round 1's anomaly pointed to.

Configurations
  5 DIM+LENGTH      dimensionless target, heated length added as L/Lc.
                    Round 1 showed L5 and L6 (same fluid, same lab, overlapping
                    P and G) differ ~7x in CHF purely through heated length,
                    which the P,G,X feature set cannot see. This tests that.
  6 REGIME_MONO     monotonicity applied PER REGIME instead of globally.
                    Round 1's global constraints hurt everywhere; the likely
                    cause is that "CHF rises with mass flux" is false near
                    dryout. Separate models for DNB (X <= 0) and dryout (X > 0).
  7 QUANTILE_10     conservative prediction: LightGBM pinball loss at the 10th
                    percentile. Under-predicting CHF is safe, over-predicting is
                    hazardous, so the design value should be a low quantile, not
                    the mean. Reported as coverage (fraction of true CHF above
                    the prediction) -- for a safe bound this should be ~90%.
  8 BOUNDED_RESID   ML corrects the Katto-Ohno style physical baseline, with the
                    correction hard-clipped to a factor of 2. Far from training
                    data the correction saturates and the answer degrades into
                    the correlation rather than diverging.
  9 MIXED_EFFECTS   GPBoost: tree ensemble for universal physics plus a random
                    intercept per publication. At prediction time on an unseen
                    publication the random effect is 0, i.e. the population mean.

    python scripts/run_physics_methods2.py
"""
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
from CoolProp.CoolProp import PropsSI
from lightgbm import LGBMRegressor
from sklearn.metrics import r2_score

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
MR = ROOT / "data" / "model_ready"
RES = ROOT / "results" / "physics_methods"
SEED, G_EARTH = 42, 9.81
FLUID = {"water": "Water", "R-134a": "R134a", "R-123": "R123"}
_cache = {}


def props(fluid, P_kPa):
    name = FLUID[fluid]
    out = np.empty((len(P_kPa), 4))
    for i, p in enumerate(P_kPa):
        key = (name, round(float(p), 3))
        if key not in _cache:
            pa = np.clip(float(p) * 1000.0, 1e3, 0.995 * PropsSI("PCRIT", name))
            try:
                _cache[key] = (PropsSI("D", "P", pa, "Q", 0, name),
                               PropsSI("D", "P", pa, "Q", 1, name),
                               PropsSI("H", "P", pa, "Q", 1, name) - PropsSI("H", "P", pa, "Q", 0, name),
                               max(PropsSI("I", "P", pa, "Q", 0, name), 1e-6))
            except Exception:
                _cache[key] = (np.nan,) * 4
        out[i] = _cache[key]
    return out[:, 0], out[:, 1], out[:, 2], out[:, 3]


def featurise(d):
    d = d.copy()
    for f, g in d.groupby("fluid"):
        a, b, c, e = props(f, g.P_kPa.values)
        d.loc[g.index, ["rho_l", "rho_v", "hfg", "sigma"]] = np.column_stack([a, b, c, e])
    d["Lc"] = np.sqrt(d.sigma / (G_EARTH * (d.rho_l - d.rho_v)))
    d["dens_ratio"] = d.rho_v / d.rho_l
    d["We_Lc"] = d.G_kg_m2s ** 2 * d.Lc / (d.rho_l * d.sigma)
    d["L_over_Lc"] = d.L_m / d.Lc                      # dimensionless heated length
    d["Bo"] = d.CHF_kW_m2 * 1000.0 / (d.G_kg_m2s * d.hfg)
    d["logCHF"] = np.log(d.CHF_kW_m2)
    # Katto-Ohno style baseline: Bo ~ C * We^-0.3 * (rho_v/rho_l)^0.133
    d["Bo_katto"] = 0.10 * d.We_Lc ** -0.3 * d.dens_ratio ** 0.133
    d["CHF_katto"] = d.Bo_katto * d.G_kg_m2s * d.hfg / 1000.0
    return d.replace([np.inf, -np.inf], np.nan)


DIM = ["dens_ratio", "We_Lc", "X"]
DIML = ["dens_ratio", "We_Lc", "X", "L_over_Lc"]


def lgb(mono=None, obj=None, alpha=None):
    kw = dict(n_estimators=400, num_leaves=31, learning_rate=0.05,
              random_state=SEED, n_jobs=-1, verbose=-1)
    if mono is not None:
        kw["monotone_constraints"] = mono
    if obj:
        kw["objective"] = obj
        kw["alpha"] = alpha
    return LGBMRegressor(**kw)


def bo_to_chf(logbo, d):
    return np.exp(logbo) * d.G_kg_m2s.values * d.hfg.values / 1000.0


def evaluate(name, pred, d, extra=None):
    a = d.CHF_kW_m2.values
    p = np.clip(pred, 1e-6, None)
    r = dict(config=name, dataset=d.dataset.iloc[0], n=len(a), fluid=d.fluid.iloc[0],
             R2_logCHF=r2_score(np.log(a), np.log(p)),
             MAPE=float(np.mean(np.abs((a - p) / a)) * 100),
             frac_conservative=float((p <= a).mean()))
    if extra:
        r.update(extra)
    return r


def main():
    RES.mkdir(parents=True, exist_ok=True)
    a = pd.read_csv(MR / "train/01_NRC_flow.csv"); a["study"] = "NRC:" + a.study.astype(str)
    b = pd.read_csv(MR / "train/02_Zhao2020_flow.csv"); b["study"] = "Zhao:" + b.study.astype(str)
    tr = featurise(pd.concat([a, b], ignore_index=True))
    tr = tr.dropna(subset=DIM + ["Bo", "logCHF"])
    tr = tr[(tr.Bo > 0) & (tr.G_kg_m2s > 0)]
    trL = tr.dropna(subset=["L_over_Lc"])
    print(f"train {len(tr)} rows ({len(trL)} with heated length), {tr.study.nunique()} publications", flush=True)

    tests = {}
    for f in sorted((MR / "test").glob("L*.csv")):
        d = pd.read_csv(f)
        if d.boiling_mode.iloc[0] != "flow":
            continue
        d = featurise(d).dropna(subset=DIM)
        d = d[(d.G_kg_m2s > 0) & (d.CHF_kW_m2 > 0)]
        if len(d):
            tests[f.stem] = d

    rows = []

    # --- 5. dimensionless + heated length -------------------------------
    m5 = lgb().fit(trL[DIML].values, np.log(trL.Bo.values))
    for t, d in tests.items():
        dd = d.dropna(subset=["L_over_Lc"])
        if len(dd) < 5:
            rows.append(dict(config="5_dim_plus_length", dataset=t, n=0,
                             note="no heated length reported")); continue
        rows.append(evaluate("5_dim_plus_length", bo_to_chf(m5.predict(dd[DIML].values), dd), dd))
    print("  5 done", flush=True)

    # --- 6. regime-specific monotonicity --------------------------------
    # DNB (X<=0) and dryout (X>0) modelled separately; constrain only within regime
    for t, d in tests.items():
        pred = np.empty(len(d)); pred[:] = np.nan
        for lo, hi, mono in [(-99, 0.0, [0, 1, -1]), (0.0, 99, [0, -1, -1])]:
            mtr = tr[(tr.X > lo) & (tr.X <= hi)]
            mask = (d.X > lo) & (d.X <= hi)
            if len(mtr) < 50 or mask.sum() == 0:
                continue
            mm = lgb(mono=mono).fit(mtr[DIM].values, np.log(mtr.Bo.values))
            pred[mask.values] = bo_to_chf(mm.predict(d.loc[mask, DIM].values), d[mask])
        ok = ~np.isnan(pred)
        if ok.sum() > 3:
            rows.append(evaluate("6_regime_monotonic", pred[ok], d[ok]))
    print("  6 done", flush=True)

    # --- 7. conservative 10th-percentile quantile -----------------------
    m7 = lgb(obj="quantile", alpha=0.10).fit(tr[DIM].values, np.log(tr.Bo.values))
    for t, d in tests.items():
        rows.append(evaluate("7_quantile_p10", bo_to_chf(m7.predict(d[DIM].values), d), d))
    print("  7 done", flush=True)

    # --- 8. bounded physics residual ------------------------------------
    resid = np.log(tr.Bo.values) - np.log(tr.Bo_katto.values)
    m8 = lgb().fit(tr[DIM].values, resid)
    BOUND = np.log(2.0)
    for t, d in tests.items():
        corr = np.clip(m8.predict(d[DIM].values), -BOUND, BOUND)
        rows.append(evaluate("8_bounded_residual", np.exp(np.log(d.Bo_katto.values) + corr)
                             * d.G_kg_m2s.values * d.hfg.values / 1000.0, d))
    # physics alone, for reference
    for t, d in tests.items():
        rows.append(evaluate("8b_physics_only", d.CHF_katto.values, d))
    print("  8 done", flush=True)

    # --- 9. mixed-effects (GPBoost): random intercept per publication ---
    try:
        import gpboost as gpb
        grp = tr.study.astype("category").cat.codes.values.reshape(-1, 1)
        ds = gpb.Dataset(tr[DIM].values, label=np.log(tr.Bo.values))
        gp = gpb.GPModel(group_data=grp, likelihood="gaussian")
        bst = gpb.train(params={"objective": "regression_l2", "learning_rate": 0.05,
                                "num_leaves": 31, "verbose": -1},
                        train_set=ds, gp_model=gp, num_boost_round=300)
        for t, d in tests.items():
            # unseen publication -> random effect is zero, predict population mean
            new_grp = np.full((len(d), 1), -1)
            pr = bst.predict(data=d[DIM].values, group_data_pred=new_grp,
                             predict_var=False, pred_latent=False)
            fixed = pr["response_mean"] if isinstance(pr, dict) else pr
            rows.append(evaluate("9_mixed_effects", bo_to_chf(np.asarray(fixed), d), d))
        print("  9 done", flush=True)
    except Exception as e:
        print("  9 FAILED:", str(e)[:150], flush=True)

    R = pd.DataFrame(rows)
    R.to_csv(RES / "physics_methods2_results.csv", index=False)
    pd.set_option("display.width", 240)
    print("\n============ R2 on log CHF ============")
    print(R.pivot_table(index="dataset", columns="config", values="R2_logCHF").round(2).to_string())
    print("\n============ MAPE (%) ============")
    print(R.pivot_table(index="dataset", columns="config", values="MAPE").round(1).to_string())
    print("\n============ fraction where prediction <= true CHF (safe side) ============")
    print(R.pivot_table(index="dataset", columns="config", values="frac_conservative").round(2).to_string())


if __name__ == "__main__":
    main()

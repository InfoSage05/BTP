"""
run_full_grid.py
----------------
Every model x every split strategy x every dataset, in one table.

This is the complete version of manuscript outline sections 10-14. Earlier runs
covered subsets; this one closes the gaps:

  * ALL useful input columns are used. Earlier runs silently dropped some
    (pin-fin fin width and mean-bubble-length; KAERI outlet quality; Zhao's
    second diameter). Those are restored here. Columns are still dropped where
    there is a reason, and every drop is recorded in FEATURE_NOTES below.

  * ALL four split strategies from outline section 13, on every dataset.

  * ELEVEN models: the five the outline names, plus the tree/kernel families
    already used elsewhere in this project, plus CatBoost and a stacking
    ensemble (both reported as strong performers in recent CHF-ML papers).

  * A MERGED flow dataset. The three flow-boiling families share a real common
    feature set (D, L, P, G, quality) once units are harmonised, so they CAN be
    pooled and the merge is evaluated as its own dataset. The surface families
    cannot join it -- see the missingness report the script prints.

Outputs
    results/confounding/FULL_GRID_results.csv    one row per dataset x model x split
    results/confounding/FULL_GRID_summary.csv    pivot: R2 by model x split
    results/confounding/merge_feasibility.csv    why surface+flow cannot merge

    python scripts/run_full_grid.py
"""
from pathlib import Path
import warnings, time

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import (RandomForestRegressor, ExtraTreesRegressor,
                              StackingRegressor)
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import GroupKFold, KFold, LeaveOneGroupOut
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.svm import SVR
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results" / "confounding"
SEED = 42
GPR_MAX = 3000          # exact GP is O(n^3)
HEAVY_MAX = 30000       # above this, skip the slowest learners

FEATURE_NOTES = """
Columns deliberately excluded, with reasons:
  KAERI  Power, Area, Perimeter  -> answer-side quantities (CHF is a heat flux);
                                    including them risks leakage.
  KAERI  MassFlow, InletTemperature -> redundant with MassFlux / InletEnthalpy.
  KAERI  TestID, Shape, Fluid, Continuous, CHFLocation, QualityPosition,
         WallPower, WallMesh       -> identifiers or constant/text metadata.
  Zhao   id, geometry              -> identifier / text label.
  ATF    material, geometry, fluid -> text; encoded numerically instead
                                    (effusivity proxy, geometry size, is_water).
All other numeric columns are now used as inputs.
"""


# ----------------------------------------------------------------------
def models(n_rows):
    m = {
        "Linear": make_pipeline(StandardScaler(), LinearRegression()),
        "Poly2_Ridge": make_pipeline(StandardScaler(),
                                     PolynomialFeatures(2, include_bias=False),
                                     Ridge(alpha=1.0, random_state=SEED)),
        "KNN_k5": make_pipeline(StandardScaler(), KNeighborsRegressor(n_neighbors=5)),
        "SVR_RBF": make_pipeline(StandardScaler(), SVR(C=10.0, epsilon=0.05)),
        "ANN_MLP": make_pipeline(StandardScaler(), MLPRegressor(
            hidden_layer_sizes=(64, 32), max_iter=3000, random_state=SEED)),
        "RandomForest": RandomForestRegressor(n_estimators=200, min_samples_leaf=2,
                                              random_state=SEED, n_jobs=-1),
        "ExtraTrees": ExtraTreesRegressor(n_estimators=200, min_samples_leaf=2,
                                          random_state=SEED, n_jobs=-1),
        "XGBoost": XGBRegressor(n_estimators=250, max_depth=5, learning_rate=0.07,
                                subsample=0.9, colsample_bytree=0.9,
                                random_state=SEED, n_jobs=-1, verbosity=0),
        "LightGBM": LGBMRegressor(n_estimators=250, num_leaves=31, learning_rate=0.07,
                                  random_state=SEED, n_jobs=-1, verbose=-1),
        "CatBoost": CatBoostRegressor(iterations=300, depth=6, learning_rate=0.08,
                                      random_seed=SEED, verbose=0, allow_writing_files=False),
    }
    if n_rows <= GPR_MAX:
        m["GPR_Matern"] = make_pipeline(StandardScaler(), GaussianProcessRegressor(
            kernel=ConstantKernel(1.0) * Matern(nu=2.5) + WhiteKernel(1e-2),
            normalize_y=True, random_state=SEED))
    if n_rows <= HEAVY_MAX:
        m["Stacking_RF_XGB_SVR"] = StackingRegressor(
            estimators=[("rf", RandomForestRegressor(n_estimators=150, random_state=SEED, n_jobs=-1)),
                        ("xgb", XGBRegressor(n_estimators=150, max_depth=4, random_state=SEED,
                                             n_jobs=-1, verbosity=0)),
                        ("svr", make_pipeline(StandardScaler(), SVR(C=10.0)))],
            final_estimator=Ridge(alpha=1.0), n_jobs=1)
    return m


# ----------------------------------------------------------------------
def load_all():
    F = {}

    # ---------- ATF surface ----------
    a = pd.read_csv(ROOT / "data/processed/surface_master.csv")
    a = a.dropna(subset=["CHF_kW_m2"]).copy()
    a["logCHF"] = np.log(a.CHF_kW_m2)
    size = {"flat_plate": 0.010, "horizontal_tube": 0.009525,
            "vertical_tube": 0.0102, "rodlet": 0.020}
    a["heater_size_m"] = a.geometry.map(size).fillna(0.010)
    # Thermal effusivity of the BOILING SURFACE. Order matters: for a coated
    # sample the coating is the boiling surface, so coating keywords are tested
    # before substrate keywords. An earlier dict-order version assigned
    # "Cold spray Cr coated Zirlo" the Zirlo value and "Fe13Cr4Al" the chromium
    # value, and mis-assigned "SS 304" (with a space) to the fallback -- errors
    # that tracked study membership and so fed the confound being measured.
    def effus(m):
        s = " ".join(str(m).lower().replace("-", " ").split())
        for kw, v in [("sic", 12000), ("zrsi", 11000), ("sapphire", 9000),
                      ("al2o3", 9000), ("ito", 8500), ("cuo", 9200), ("zno", 8800),
                      ("sio2", 1600)]:                      # coatings / ceramics
            if kw in s:
                return v
        if "coated" in s or "coating" in s:                  # metal coatings
            if "cral" in s or "fecral" in s:
                return 7300
            if "cr" in s:
                return 9500
        for kw, v in [("fecral", 7300), ("fe13cr4al", 7300), ("fe12cr5al", 7300),
                      ("inconel", 8200), ("titanium", 6000),
                      ("ss30", 7600), ("ss 30", 7600), ("ss31", 7600), ("ss 31", 7600),
                      ("ss32", 7600), ("ss 32", 7600), ("ss34", 7600), ("ss 34", 7600),
                      ("stainless", 7600), ("zirlo", 6800), ("zirc", 6800),
                      ("zr 705", 6500), ("zr705", 6500), ("chromium", 9500)]:
            if kw in s:
                return v
        return np.nan          # unknown -> honest missing, not a fabricated 7500
    a["effusivity"] = a.material.map(effus)
    a["is_water"] = 1.0
    a["surface_id"] = a.index.astype(str)
    a["cond_id"] = a.P_kPa.round(0).astype(str)
    F["ATF_surface"] = dict(df=a, feats=["Ra_um", "CA_deg", "orient_deg", "P_kPa",
                                         "heater_size_m", "effusivity"])

    # ---------- pin-fin (ALL geometry columns restored) ----------
    p = pd.read_csv(ROOT / "data/processed/pinfin_master.csv")
    p = p.dropna(subset=["CHF_kW_m2"]).copy()
    p["logCHF"] = np.log(p.CHF_kW_m2.clip(lower=1))
    p["is_water"] = (p.fluid == "Water").astype(float)
    p["surface_id"] = (p.material.astype(str) + "|" + p["Width(um)"].astype(str) + "|"
                       + p["Height(um)"].astype(str) + "|" + p["Spacing(um)"].astype(str))
    p["cond_id"] = p["Subcooling(K)"].astype(str)
    pf = ["Ra_um", "Porosity", "Coverage", "Subcooling(K)", "Width(um)",
          "Height(um)", "Spacing(um)", "MBL, Lateral (um)", "MBL, Total (um)", "is_water"]
    p[pf] = p[pf].fillna(p[pf].median(numeric_only=True))
    F["pin_fin"] = dict(df=p, feats=pf)

    # ---------- Zhao (both diameters kept) ----------
    z = pd.read_csv(ROOT / "data/raw/external/zhao2020_chf_flowboiling_tubes.csv")
    z = z.rename(columns={"author": "study", "pressure [MPa]": "P_MPa",
                          "mass_flux [kg/m2-s]": "G", "x_e_out [-]": "X",
                          "D_e [mm]": "D_e_mm", "D_h [mm]": "D_h_mm",
                          "length [mm]": "L_mm", "chf_exp [MW/m2]": "CHF_MW"})
    z = z.dropna(subset=["CHF_MW", "study"]).copy()
    z["CHF_kW_m2"] = z.CHF_MW * 1000.0
    z = z[z.CHF_kW_m2 > 0]
    z["logCHF"] = np.log(z.CHF_kW_m2)
    zf = ["P_MPa", "G", "X", "D_e_mm", "D_h_mm", "L_mm"]
    z[zf] = z[zf].apply(pd.to_numeric, errors="coerce")
    z[zf] = z[zf].fillna(z[zf].median())
    z["surface_id"] = z.D_e_mm.round(2).astype(str)
    z["cond_id"] = pd.cut(z.P_MPa, 6, labels=False).astype(str)
    F["Zhao_flow"] = dict(df=z, feats=zf)

    # ---------- KAERI (Quality restored; leakage columns excluded) ----------
    # UNITS: this file is pure SI. Pressure is in Pa (median 1.0e7 = 10 MPa) and
    # HeatFlux is in W/m^2 (verified: HeatFlux == Power/(Perimeter*Length) to
    # within 0.05%). An earlier version of this script assumed kPa and kW/m^2,
    # which inflated KAERI pressures and CHF by 1000x and corrupted the merged
    # dataset. Units are now stated explicitly, never inferred by heuristic.
    k = pd.read_csv(ROOT / "data/raw/external/kaeri_tr1665_nonuniform_chf.csv")
    k = k.rename(columns={"Source": "study"})
    num = ["Diameter", "Length", "Pressure", "MassFlux", "InletEnthalpy",
           "Quality", "HeatFlux"]
    for c in num:
        k[c] = pd.to_numeric(k[c], errors="coerce")
    k = k.dropna(subset=["HeatFlux", "study"])
    k = k[k.HeatFlux > 0].copy()
    k["CHF_kW_m2"] = k.HeatFlux / 1000.0          # W/m^2 -> kW/m^2
    k["Pressure_kPa"] = k.Pressure / 1000.0       # Pa    -> kPa
    k["logCHF"] = np.log(k.CHF_kW_m2)
    kf = ["Diameter", "Length", "Pressure_kPa", "MassFlux", "InletEnthalpy", "Quality"]
    k[kf] = k[kf].fillna(k[kf].median())
    k["surface_id"] = k.Diameter.round(4).astype(str)
    k["cond_id"] = pd.cut(k.Pressure_kPa, 6, labels=False).astype(str)
    F["KAERI_flow"] = dict(df=k, feats=kf)

    # ---------- NRC ----------
    n = pd.read_csv(ROOT / "data/nrc_chf_clean.csv").rename(columns={"ref_id": "study"})
    n = n.dropna(subset=["CHF_kW_m2"]).copy()
    n["logCHF"] = np.log(n.CHF_kW_m2.clip(lower=1))
    n["surface_id"] = n.D_m.round(4).astype(str)
    n["cond_id"] = pd.cut(n.P_kPa, 8, labels=False).astype(str)
    F["NRC_flow"] = dict(df=n, feats=["D_m", "L_m", "P_kPa", "G_kg_m2s", "X",
                                      "dHin_sub_kJkg", "Tin_C"])

    # ---------- MERGED flow: units harmonised to m, kPa, kg/m2s ----------
    # Every conversion factor below is stated explicitly from the source file's
    # documented units. No heuristic unit sniffing -- that is what produced the
    # 1000x KAERI error previously. A sanity assert guards the result.
    def part(df, D, L, P, G, X, scaleD, scaleL, scaleP, src):
        o = pd.DataFrame({
            "D_m": df[D] * scaleD, "L_m": df[L] * scaleL, "P_kPa": df[P] * scaleP,
            "G_kg_m2s": df[G], "X": df[X], "logCHF": df.logCHF,
            "CHF_kW_m2": df.CHF_kW_m2, "study": src + ":" + df.study.astype(str)})
        o["origin"] = src
        return o
    kk = F["KAERI_flow"]["df"]
    merged = pd.concat([
        # Zhao: D_e in mm, length in mm, pressure in MPa
        part(F["Zhao_flow"]["df"], "D_e_mm", "L_mm", "P_MPa", "G", "X", .001, .001, 1000., "Zhao"),
        # KAERI: D and L already in m; Pressure_kPa already converted from Pa above
        part(kk, "Diameter", "Length", "Pressure_kPa", "MassFlux", "Quality", 1., 1., 1., "KAERI"),
        # NRC: already m, m, kPa
        part(F["NRC_flow"]["df"], "D_m", "L_m", "P_kPa", "G_kg_m2s", "X", 1., 1., 1., "NRC"),
    ], ignore_index=True).dropna()
    # physical plausibility guard: water CHF rigs run below ~25 MPa and 100 MW/m^2
    assert merged.P_kPa.max() < 25000, f"merged pressure {merged.P_kPa.max():.3g} kPa is unphysical -- unit bug"
    assert merged.CHF_kW_m2.max() < 100000, f"merged CHF {merged.CHF_kW_m2.max():.3g} kW/m2 is unphysical -- unit bug"
    merged["surface_id"] = merged.D_m.round(4).astype(str)
    merged["cond_id"] = pd.cut(merged.P_kPa, 8, labels=False).astype(str)
    F["MERGED_flow"] = dict(df=merged, feats=["D_m", "L_m", "P_kPa", "G_kg_m2s", "X"])

    # ---------- ABLATIONS demanded by the independent audit ----------
    # (a) NRC without X. In this compilation X is computed FROM the measured CHF
    #     by heat balance, so CHF = G*D*(dHin + X*hfg)/(4L) is an exact identity
    #     in the feature set (R^2 = 0.9998 with no fitting). Reporting NRC with X
    #     as evidence that "models learn physics" is not defensible.
    n_noX = n.copy()
    F["NRC_flow_noX"] = dict(df=n_noX, feats=["D_m", "L_m", "P_kPa", "G_kg_m2s",
                                             "dHin_sub_kJkg", "Tin_C"])
    # (b) pin-fin without is_water. Fluid is perfectly nested inside study
    #     (each study is entirely FC-72 or entirely water), so is_water doubles
    #     as a study-block indicator and carries most of the apparent skill.
    F["pin_fin_noFluid"] = dict(df=p, feats=[c for c in pf if c != "is_water"])
    return F


# ----------------------------------------------------------------------
def evaluate(name, d, log):
    df, feats = d["df"], d["feats"]
    df = df.dropna(subset=feats + ["logCHF"])
    X, y = df[feats].values, df.logCHF.values
    n = len(df)
    strategies = {
        "I_random": None,
        "II_condition_wise": df.cond_id.astype(str).values,
        "III_surface_wise": df.surface_id.astype(str).values,
        "IV_leave_one_publication_out": df.study.astype(str).values,
    }
    # A grouped split only means something if the groups are neither all-singleton
    # (then it IS a random split) nor near-perfectly aligned with the study labels
    # (then it is a study split wearing a different name). Both happened on the
    # ATF set and were reported as independent protocols. Detect and label.
    degenerate = {}
    studies = df.study.astype(str).values
    for sname, grp in strategies.items():
        if grp is None:
            continue
        ng = len(np.unique(grp))
        if ng >= 0.95 * len(df):
            degenerate[sname] = f"DEGENERATE: {ng} groups for {len(df)} rows -> equivalent to random split"
        elif ng < 2:
            degenerate[sname] = "DEGENERATE: fewer than 2 groups"
        else:
            # does this grouping just reproduce the study grouping?
            ct = pd.crosstab(pd.Series(grp), pd.Series(studies))
            purity = (ct.max(axis=1) / ct.sum(axis=1)).mean()
            if purity > 0.95:
                degenerate[sname] = (f"DEGENERATE: groups are {purity:.0%} study-pure "
                                     f"-> this is a study split, not a genuine {sname}")
    rows = []
    for mname, mk in models(n).items():
        for sname, grp in strategies.items():
            t0 = time.time()
            if grp is None:
                cv, gsplit = KFold(5, shuffle=True, random_state=SEED), False
            else:
                ng = len(np.unique(grp))
                if ng < 2:
                    continue
                # true leave-one-group-out when the group count is manageable,
                # otherwise 5-fold grouped CV. Previously this was always
                # GroupKFold(5) yet reported as leave-one-out.
                if sname.startswith("IV") and ng <= 20:
                    cv, gsplit = LeaveOneGroupOut(), True
                else:
                    cv, gsplit = GroupKFold(n_splits=min(5, ng)), True
            try:
                P, A, folds = [], [], []
                it = cv.split(X, y, grp) if gsplit else cv.split(X)
                for tr, te in it:
                    m = clone(mk).fit(X[tr], y[tr])
                    pr = m.predict(X[te])
                    P.append(pr); A.append(y[te])
                    if len(te) > 2:
                        folds.append(r2_score(y[te], pr))
                pr, ac = np.concatenate(P), np.concatenate(A)
                rows.append(dict(
                    dataset=name, n=n, model=mname, split=sname,
                    split_warning=degenerate.get(sname, ""),
                    R2=r2_score(ac, pr),
                    median_fold_R2=float(np.nanmedian(folds)) if folds else np.nan,
                    worst_fold_R2=float(np.nanmin(folds)) if folds else np.nan,
                    RMSE_log=float(np.sqrt(mean_squared_error(ac, pr))),
                    MAE_log=mean_absolute_error(ac, pr),
                    MAPE_pct=float(np.mean(np.abs((np.exp(ac)-np.exp(pr))/np.exp(ac)))*100),
                    n_groups=int(len(np.unique(grp))) if grp is not None else np.nan,
                    seconds=round(time.time()-t0, 1)))
            except Exception as e:
                rows.append(dict(dataset=name, n=n, model=mname, split=sname,
                                 R2=np.nan, note=str(e)[:70]))
        print(f"    {name:14s} {mname:20s} done", flush=True)
    return pd.DataFrame(rows)


def merge_feasibility(F):
    """Quantify why the surface families cannot join the flow merge."""
    sets = {k: set(v["feats"]) for k, v in F.items() if k != "MERGED_flow"}
    rows = []
    keys = list(sets)
    for i, a in enumerate(keys):
        for b in keys[i+1:]:
            common = sets[a] & sets[b]
            rows.append(dict(dataset_A=a, dataset_B=b,
                             n_features_A=len(sets[a]), n_features_B=len(sets[b]),
                             n_shared=len(common), shared=", ".join(sorted(common)) or "NONE"))
    return pd.DataFrame(rows)


def main():
    RES.mkdir(parents=True, exist_ok=True)
    print(FEATURE_NOTES, flush=True)
    F = load_all()
    print("Datasets:", flush=True)
    for k, v in F.items():
        d = v["df"].dropna(subset=v["feats"] + ["logCHF"])
        print(f"  {k:14s} n={len(d):6d} studies={d.study.nunique():3d} features={len(v['feats'])}", flush=True)
    mf = merge_feasibility(F)
    mf.to_csv(RES / "merge_feasibility.csv", index=False)
    print("\nFeature overlap between families:\n", mf.to_string(index=False), flush=True)

    out = []
    for name, d in F.items():
        print(f"\n>>> {name}", flush=True)
        out.append(evaluate(name, d, None))
        pd.concat(out, ignore_index=True).to_csv(RES / "FULL_GRID_results.csv", index=False)
    R = pd.concat(out, ignore_index=True)
    R.to_csv(RES / "FULL_GRID_results.csv", index=False)
    piv = R.pivot_table(index=["dataset", "model"], columns="split", values="R2").round(3)
    piv.to_csv(RES / "FULL_GRID_summary.csv")
    print("\n\n================= FULL GRID: R2 =================")
    pd.set_option("display.width", 250)
    print(piv.to_string())
    print(f"\nRows: {len(R)}  ->  {RES/'FULL_GRID_results.csv'}")


if __name__ == "__main__":
    main()

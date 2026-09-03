"""
run_confounding_analysis.py
---------------------------
Core evidence for the paper: quantify how much of the apparent
surface-characteristic -> CHF signal in consolidated databases is actually
source-level (inter-laboratory) confounding, and how badly random-split
validation overstates model skill as a result.

Runs the identical protocol on every data family so the numbers are
comparable across three orders of magnitude in sample size:

    ATF_surface   55 modellable rows /  7 studies
    pin_fin      175 rows            / 16 studies
    NRC_flow     24,443 rows         / 60 studies

For each family it reports
    1. ICC  -- share of log-CHF variance attributable to study identity alone
    2. pooled vs within-study (study-centred) correlation for each feature,
       which is where the sign reversals show up
    3. variance explained by  features / study dummies / both
    4. 5 ML models under RANDOM CV vs GROUPED (leave-studies-out) CV
    5. per-study leave-one-study-out detail for the surface families

Everything is deterministic (fixed seeds) and writes CSVs to
results/confounding/ so the manuscript numbers are traceable to a rerun.

    python scripts/run_confounding_analysis.py
"""
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import GroupKFold, KFold, LeaveOneGroupOut
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results" / "confounding"
SEED = 42
MIN_TEST = 3          # a held-out study smaller than this cannot support an R2
N_SPLITS = 5


GPR_MAX_N = 3000      # GPR is O(n^3): above this it is fitted on a fixed subsample
SUBSAMPLE_SEED = 7


def models(n_rows=None):
    """The five algorithm families named in the manuscript outline.

    GPR is dropped when the training set is far past what an exact GP can
    factorise -- reporting a silently subsampled GP next to full-data models
    would not be a like-for-like comparison.
    """
    m = {
        "ANN_MLP": make_pipeline(StandardScaler(), MLPRegressor(
            hidden_layer_sizes=(64, 32), max_iter=4000, random_state=SEED)),
        "RandomForest": RandomForestRegressor(
            n_estimators=300, min_samples_leaf=2, random_state=SEED, n_jobs=-1),
        "XGBoost": XGBRegressor(
            n_estimators=300, max_depth=4, learning_rate=0.07,
            subsample=0.9, colsample_bytree=0.9, random_state=SEED, n_jobs=-1),
        "SVR_RBF": make_pipeline(StandardScaler(), SVR(C=10.0, epsilon=0.05)),
        "GPR_Matern": make_pipeline(StandardScaler(), GaussianProcessRegressor(
            kernel=ConstantKernel(1.0) * Matern(nu=2.5) + WhiteKernel(1e-2),
            normalize_y=True, random_state=SEED)),
    }
    if n_rows is not None and n_rows > GPR_MAX_N:
        m.pop("GPR_Matern")
    return m


def icc(y, groups):
    """Share of variance between groups (study identity) vs total."""
    y = np.asarray(y, float)
    grand = y.mean()
    sst = ((y - grand) ** 2).sum()
    ssb = sum(len(y[groups == g]) * (y[groups == g].mean() - grand) ** 2
              for g in np.unique(groups))
    return ssb / sst if sst > 0 else np.nan


def correlation_table(df, feats, target, group):
    """Pooled vs study-centred correlation -- where Simpson reversals appear."""
    rows = []
    for f in feats:
        d = df.dropna(subset=[f, target])
        if len(d) < 5 or d[f].nunique() < 3:
            continue
        pooled = d[f].corr(d[target])
        c = d.copy()
        c["_f"] = c[f] - c.groupby(group)[f].transform("mean")
        c["_t"] = c[target] - c.groupby(group)[target].transform("mean")
        within = c["_f"].corr(c["_t"])
        # per-study direction agreement
        signs = []
        for _, g in d.groupby(group):
            if len(g) >= 4 and g[f].nunique() > 2:
                signs.append(np.sign(g[f].corr(g[target])))
        rows.append(dict(feature=f, n=len(d), pooled_r=pooled, within_study_r=within,
                         sign_reversal=bool(np.sign(pooled) != np.sign(within)),
                         n_studies_testable=len(signs),
                         n_studies_agreeing_with_within=int(sum(s == np.sign(within) for s in signs))))
    return pd.DataFrame(rows)


def variance_partition(df, feats, target, group):
    d = df.dropna(subset=feats + [target])
    y = d[target].values
    X = d[feats].values
    D = pd.get_dummies(d[group], drop_first=True).astype(float).values
    def r2(M):
        return r2_score(y, LinearRegression().fit(M, y).predict(M))
    return dict(n=len(d),
                r2_features_only=r2(X),
                r2_study_only=r2(D),
                r2_features_plus_study=r2(np.column_stack([X, D])),
                features_gain_over_study=r2(np.column_stack([X, D])) - r2(D))


def cv_compare(df, feats, target, group, name):
    """Random CV vs grouped CV for each model -- the optimism gap."""
    d = df.dropna(subset=feats + [target])
    X, y, g = d[feats].values, d[target].values, d[group].values
    n_groups = len(np.unique(g))
    k = min(N_SPLITS, n_groups)
    rows = []
    for mname, mk in models(n_rows=len(d)).items():
        print(f"    [{name}] {mname} ...", flush=True)
        for split, cv in (("random", KFold(k, shuffle=True, random_state=SEED)),
                          ("grouped", GroupKFold(n_splits=k))):
            preds, actual = [], []
            for tr, te in (cv.split(X, y, g) if split == "grouped" else cv.split(X)):
                m = mk if not hasattr(mk, "fit") else __import__("sklearn").base.clone(mk)
                m.fit(X[tr], y[tr])
                preds.append(m.predict(X[te])); actual.append(y[te])
            p, a = np.concatenate(preds), np.concatenate(actual)
            rows.append(dict(dataset=name, model=mname, split=split,
                             r2=r2_score(a, p), mae=mean_absolute_error(a, p),
                             rmse=float(np.sqrt(mean_squared_error(a, p))),
                             mape=float(np.mean(np.abs((np.exp(a) - np.exp(p)) / np.exp(a))) * 100)))
    out = pd.DataFrame(rows)
    piv = out.pivot_table(index=["dataset", "model"], columns="split", values="r2")
    piv["optimism_gap"] = piv["random"] - piv["grouped"]
    return out, piv.reset_index()


def loso_detail(df, feats, target, group, name):
    d = df.dropna(subset=feats + [target])
    X, y, g = d[feats].values, d[target].values, d[group].values
    rows = []
    for tr, te in LeaveOneGroupOut().split(X, y, g):
        held = g[te][0]
        if len(te) < MIN_TEST:
            rows.append(dict(dataset=name, held_out_study=held, n_test=len(te),
                             r2=np.nan, note="test fold too small for R2"))
            continue
        m = RandomForestRegressor(n_estimators=300, min_samples_leaf=2,
                                  random_state=SEED, n_jobs=-1).fit(X[tr], y[tr])
        rows.append(dict(dataset=name, held_out_study=held, n_test=len(te),
                         r2=r2_score(y[te], m.predict(X[te])), note=""))
    return pd.DataFrame(rows)


def load_families():
    fams = {}

    atf = pd.read_csv(ROOT / "data/processed/surface_master.csv")
    atf = atf.dropna(subset=["CHF_kW_m2"]).copy()
    atf["logCHF"] = np.log(atf.CHF_kW_m2.clip(lower=1))
    fams["ATF_surface"] = (atf, ["Ra_um", "CA_deg"])

    pin = pd.read_csv(ROOT / "data/processed/pinfin_master.csv")
    pin["logCHF"] = np.log(pin.CHF_kW_m2.clip(lower=1))
    pin["is_water"] = (pin.fluid == "Water").astype(float)
    fams["pin_fin"] = (pin, ["Ra_um", "Porosity", "Coverage", "Subcooling(K)",
                             "Height(um)", "Spacing(um)", "is_water"])

    # Zhao 2020: compiled multi-source flow-boiling database, 10 named authors.
    # Sits in the 10^3 gap between pin-fin (175) and NRC (24k) on the
    # rows-per-study axis, which is the axis the whole paper is organised around.
    z = pd.read_csv(ROOT / "data/raw/external/zhao2020_chf_flowboiling_tubes.csv")
    z = z.rename(columns={"author": "study", "pressure [MPa]": "P_MPa",
                          "mass_flux [kg/m2-s]": "G", "x_e_out [-]": "X",
                          "D_e [mm]": "D_e_mm", "D_h [mm]": "D_h_mm",
                          "length [mm]": "L_mm", "chf_exp [MW/m2]": "CHF_MW"})
    z = z.dropna(subset=["CHF_MW", "study"]).copy()
    z["CHF_kW_m2"] = z.CHF_MW * 1000.0
    z = z[z.CHF_kW_m2 > 0]
    z["logCHF"] = np.log(z.CHF_kW_m2)
    fams["Zhao_flow"] = (z, ["P_MPa", "G", "X", "D_e_mm", "L_mm"])

    # KAERI TR-1665 non-uniform axial power, 11 named source studies.
    k = pd.read_csv(ROOT / "data/raw/external/kaeri_tr1665_nonuniform_chf.csv")
    k = k.rename(columns={"Source": "study"})
    k = k.dropna(subset=["HeatFlux", "study"]).copy()
    k["CHF_kW_m2"] = pd.to_numeric(k.HeatFlux, errors="coerce")
    k = k[k.CHF_kW_m2 > 0].dropna(subset=["CHF_kW_m2"])
    k["logCHF"] = np.log(k.CHF_kW_m2)
    for c in ["Diameter", "Length", "Pressure", "MassFlux", "InletEnthalpy"]:
        k[c] = pd.to_numeric(k[c], errors="coerce")
    k = k.dropna(subset=["Diameter", "Length", "Pressure", "MassFlux", "InletEnthalpy"])
    fams["KAERI_flow"] = (k, ["Diameter", "Length", "Pressure", "MassFlux", "InletEnthalpy"])

    nrc = pd.read_csv(ROOT / "data/nrc_chf_clean.csv")
    nrc = nrc.rename(columns={"ref_id": "study"})
    nrc = nrc.dropna(subset=["CHF_kW_m2"]).copy()
    nrc["logCHF"] = np.log(nrc.CHF_kW_m2.clip(lower=1))
    fams["NRC_flow"] = (nrc, ["D_m", "L_m", "P_kPa", "G_kg_m2s", "X",
                              "dHin_sub_kJkg", "Tin_C"])
    return fams


def main():
    RES.mkdir(parents=True, exist_ok=True)
    fams = load_families()

    icc_rows, corr_all, var_all, cv_all, gap_all, loso_all = [], [], [], [], [], []

    for name, (df, feats) in fams.items():
        d = df.dropna(subset=feats + ["logCHF"])
        icc_rows.append(dict(dataset=name, n=len(d), studies=d.study.nunique(),
                             icc_study=icc(d.logCHF.values, d.study.values),
                             rows_per_study=len(d) / d.study.nunique()))

        c = correlation_table(df, feats, "logCHF", "study"); c.insert(0, "dataset", name)
        corr_all.append(c)

        v = variance_partition(df, feats, "logCHF", "study"); v["dataset"] = name
        var_all.append(v)

        cv, gap = cv_compare(df, feats, "logCHF", "study", name)
        cv_all.append(cv); gap_all.append(gap)

        if name != "NRC_flow":       # per-study detail only meaningful on the small families
            loso_all.append(loso_detail(df, feats, "logCHF", "study", name))
        print(f"done: {name}", flush=True)

    pd.DataFrame(icc_rows).to_csv(RES / "icc_by_dataset.csv", index=False)
    pd.concat(corr_all).to_csv(RES / "pooled_vs_within_correlations.csv", index=False)
    pd.DataFrame(var_all).to_csv(RES / "variance_partition.csv", index=False)
    pd.concat(cv_all).to_csv(RES / "cv_random_vs_grouped.csv", index=False)
    pd.concat(gap_all).to_csv(RES / "optimism_gap.csv", index=False)
    pd.concat(loso_all).to_csv(RES / "loso_per_study.csv", index=False)

    print("\n================ HEADLINE NUMBERS ================\n")
    print("ICC -- share of log-CHF variance explained by study identity alone:")
    print(pd.DataFrame(icc_rows).to_string(index=False))
    print("\nPooled vs within-study correlations (sign_reversal = Simpson's paradox):")
    print(pd.concat(corr_all).to_string(index=False))
    print("\nVariance partition:")
    print(pd.DataFrame(var_all)[["dataset", "n", "r2_features_only", "r2_study_only",
                                 "r2_features_plus_study", "features_gain_over_study"]]
          .to_string(index=False))
    print("\nOptimism gap (random-split R2 minus grouped-split R2):")
    print(pd.concat(gap_all).to_string(index=False))
    print(f"\nAll tables written to {RES}")


if __name__ == "__main__":
    main()

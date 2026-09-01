"""
run_phase1_analysis.py
----------------------
Phase 1: complete the statistical case begun in run_confounding_analysis.py.

Five deliverables, each writing a CSV under results/confounding/:

  1. FOUR SPLIT STRATEGIES (outline section 13)
     Random / condition-wise / surface-wise / leave-one-study-out, per dataset.
     Not every strategy is meaningful on every dataset -- where it degenerates
     (e.g. ATF, where every row is already its own unique surface) that is
     recorded explicitly rather than reported as if it were a real protocol.

  2. MIXED-EFFECTS MODELS
     Random intercept per study. This is the statistically correct way to
     estimate a surface-feature effect from multi-laboratory data, and it is
     what the paper recommends in place of pooled OLS. Reported next to the
     pooled OLS coefficient so the shrinkage is visible.

  3. BOOTSTRAP CONFIDENCE INTERVALS
     On pooled and within-study correlations. With n=55 a point estimate alone
     is not defensible; the CI is what makes the sign-reversal claim survive review.

  4. SHAP IMPORTANCE UNDER BOTH PROTOCOLS
     Feature rankings computed from models trained under random vs grouped
     splits. If the ranking reorders, the published importance ordering in this
     literature is protocol-dependent -- which is the outline's Gap 4.

  5. SENSITIVITY OF THE ROUGHNESS REVERSAL
     With and without the sandblasted (Ra > 10 um) arm. The direction is robust;
     the magnitude is not. Both are reported.

    python scripts/run_phase1_analysis.py
"""
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from sklearn.base import clone
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import GroupKFold, KFold

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results" / "confounding"
SEED = 42
N_BOOT = 2000
rng = np.random.default_rng(SEED)


# --------------------------------------------------------------------------
# data
# --------------------------------------------------------------------------
def load():
    fams = {}

    atf = pd.read_csv(ROOT / "data/processed/surface_master.csv")
    atf = atf.dropna(subset=["CHF_kW_m2", "Ra_um", "CA_deg"]).copy()
    atf["logCHF"] = np.log(atf.CHF_kW_m2)
    # every ATF row is a distinct tested surface -> surface id == row id
    atf["surface_id"] = atf.index.astype(str)
    atf["cond_id"] = atf.P_kPa.round(0).astype(str)
    fams["ATF_surface"] = dict(df=atf, feats=["Ra_um", "CA_deg"],
                               formula="logCHF ~ Ra_um + CA_deg")

    pin = pd.read_csv(ROOT / "data/processed/pinfin_master.csv")
    pin = pin.dropna(subset=["CHF_kW_m2"]).copy()
    pin["logCHF"] = np.log(pin.CHF_kW_m2.clip(lower=1))
    pin["is_water"] = (pin.fluid == "Water").astype(float)
    pin["surface_id"] = (pin.material.astype(str) + "|" + pin["Width(um)"].astype(str)
                         + "|" + pin["Height(um)"].astype(str) + "|"
                         + pin["Spacing(um)"].astype(str))
    pin["cond_id"] = pin["Subcooling(K)"].astype(str)
    pin[["Porosity", "Coverage", "Height(um)", "Spacing(um)", "Ra_um"]] = \
        pin[["Porosity", "Coverage", "Height(um)", "Spacing(um)", "Ra_um"]].fillna(0)
    fams["pin_fin"] = dict(df=pin,
                           feats=["Ra_um", "Porosity", "Coverage", "Subcooling(K)",
                                  "Height(um)", "Spacing(um)", "is_water"],
                           formula="logCHF ~ Ra_um + Porosity + Coverage + Q('Subcooling(K)') + is_water")

    nrc = pd.read_csv(ROOT / "data/nrc_chf_clean.csv").rename(columns={"ref_id": "study"})
    nrc = nrc.dropna(subset=["CHF_kW_m2"]).copy()
    nrc["logCHF"] = np.log(nrc.CHF_kW_m2.clip(lower=1))
    nrc["surface_id"] = nrc.D_m.round(4).astype(str)          # tube geometry stands in for "surface"
    nrc["cond_id"] = pd.cut(nrc.P_kPa, 8, labels=False).astype(str)
    fams["NRC_flow"] = dict(df=nrc,
                            feats=["D_m", "L_m", "P_kPa", "G_kg_m2s", "X",
                                   "dHin_sub_kJkg", "Tin_C"],
                            formula="logCHF ~ D_m + L_m + P_kPa + G_kg_m2s + X + dHin_sub_kJkg")
    return fams


def rf():
    return RandomForestRegressor(n_estimators=300, min_samples_leaf=2,
                                 random_state=SEED, n_jobs=-1)


# --------------------------------------------------------------------------
# 1. four split strategies
# --------------------------------------------------------------------------
def grouped_cv_r2(X, y, groups, k):
    preds, actual, folds = [], [], []
    cv = GroupKFold(n_splits=k)
    for tr, te in cv.split(X, y, groups):
        m = clone(rf()).fit(X[tr], y[tr])
        p = m.predict(X[te])
        preds.append(p); actual.append(y[te])
        folds.append(r2_score(y[te], p) if len(te) > 2 else np.nan)
    return r2_score(np.concatenate(actual), np.concatenate(preds)), np.array(folds)


def four_splits(name, d):
    df, feats = d["df"], d["feats"]
    X, y = df[feats].values, df.logCHF.values
    rows = []

    # I. random
    preds, actual, folds = [], [], []
    for tr, te in KFold(5, shuffle=True, random_state=SEED).split(X):
        m = clone(rf()).fit(X[tr], y[tr])
        p = m.predict(X[te]); preds.append(p); actual.append(y[te])
        folds.append(r2_score(y[te], p))
    rows.append(dict(dataset=name, strategy="I_random", n_groups=np.nan,
                     pooled_r2=r2_score(np.concatenate(actual), np.concatenate(preds)),
                     median_fold_r2=float(np.nanmedian(folds)),
                     worst_fold_r2=float(np.nanmin(folds)), note=""))

    # II/III/IV. condition-wise, surface-wise, study-wise
    for label, col, note_if_degenerate in [
            ("II_condition_wise", "cond_id", "only one condition present"),
            ("III_surface_wise", "surface_id", "every row is a unique surface -> equivalent to random"),
            ("IV_leave_one_study_out", "study", "")]:
        g = df[col].astype(str).values
        ng = len(np.unique(g))
        note = ""
        if ng < 2:
            rows.append(dict(dataset=name, strategy=label, n_groups=ng, pooled_r2=np.nan,
                             median_fold_r2=np.nan, worst_fold_r2=np.nan,
                             note=note_if_degenerate or "too few groups"))
            continue
        if ng >= len(df) * 0.95:
            note = note_if_degenerate
        k = min(5, ng)
        pooled, folds = grouped_cv_r2(X, y, g, k)
        rows.append(dict(dataset=name, strategy=label, n_groups=ng, pooled_r2=pooled,
                         median_fold_r2=float(np.nanmedian(folds)),
                         worst_fold_r2=float(np.nanmin(folds)), note=note))
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# 2. mixed effects vs pooled OLS
# --------------------------------------------------------------------------
def mixed_effects(name, d):
    df = d["df"].copy()
    out = []
    try:
        pooled = smf.ols(d["formula"], data=df).fit()
        mixed = smf.mixedlm(d["formula"], data=df, groups=df["study"]).fit(method="lbfgs")
    except Exception as e:
        return pd.DataFrame([dict(dataset=name, term="FIT_FAILED", note=str(e)[:120])])
    for term in pooled.params.index:
        if term == "Intercept":
            continue
        pc, mc = pooled.params.get(term, np.nan), mixed.params.get(term, np.nan)
        out.append(dict(dataset=name, term=term,
                        pooled_coef=pc, pooled_p=pooled.pvalues.get(term, np.nan),
                        mixed_coef=mc, mixed_p=mixed.pvalues.get(term, np.nan),
                        sign_flip=bool(np.sign(pc) != np.sign(mc)) if np.isfinite(pc*mc) else None,
                        shrinkage_pct=float((1 - abs(mc) / abs(pc)) * 100) if pc else np.nan))
    return pd.DataFrame(out)


# --------------------------------------------------------------------------
# 3. bootstrap CIs on pooled vs within-study correlation
# --------------------------------------------------------------------------
def boot_corr(name, d):
    df, feats = d["df"], d["feats"]
    rows = []
    for f in feats:
        sub = df.dropna(subset=[f, "logCHF"])
        if sub[f].nunique() < 3:
            continue
        studies = sub.study.unique()

        def one(sample):
            pooled = sample[f].corr(sample.logCHF)
            c = sample.copy()
            c["_f"] = c[f] - c.groupby("study")[f].transform("mean")
            c["_t"] = c.logCHF - c.groupby("study")["logCHF"].transform("mean")
            return pooled, c["_f"].corr(c["_t"])

        p0, w0 = one(sub)
        bp, bw = [], []
        for _ in range(N_BOOT):
            # cluster bootstrap: resample STUDIES, not rows -- rows within a study
            # are not independent, which is the whole point of the paper
            pick = rng.choice(studies, size=len(studies), replace=True)
            samp = pd.concat([sub[sub.study == s] for s in pick], ignore_index=True)
            if samp.study.nunique() < 2:
                continue
            a, b = one(samp)
            if np.isfinite(a): bp.append(a)
            if np.isfinite(b): bw.append(b)
        rows.append(dict(dataset=name, feature=f, n=len(sub),
                         pooled_r=p0, pooled_lo=np.percentile(bp, 2.5), pooled_hi=np.percentile(bp, 97.5),
                         within_r=w0, within_lo=np.percentile(bw, 2.5), within_hi=np.percentile(bw, 97.5),
                         reversal=bool(np.sign(p0) != np.sign(w0)),
                         within_ci_excludes_zero=bool(np.percentile(bw, 2.5) > 0 or np.percentile(bw, 97.5) < 0)))
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# 4. SHAP importance under both protocols
# --------------------------------------------------------------------------
def shap_by_protocol(name, d):
    import shap
    df, feats = d["df"], d["feats"]
    X, y, g = df[feats].values, df.logCHF.values, df.study.astype(str).values
    rows = []
    for proto, splitter in [("random", KFold(5, shuffle=True, random_state=SEED)),
                            ("grouped", GroupKFold(n_splits=min(5, len(np.unique(g)))))]:
        imps = []
        it = splitter.split(X, y, g) if proto == "grouped" else splitter.split(X)
        for tr, te in it:
            m = clone(rf()).fit(X[tr], y[tr])
            sv = shap.TreeExplainer(m).shap_values(X[te], check_additivity=False)
            imps.append(np.abs(sv).mean(axis=0))
        mean_imp = np.mean(imps, axis=0)
        order = np.argsort(-mean_imp)
        rank = {feats[i]: r + 1 for r, i in enumerate(order)}
        for i, f in enumerate(feats):
            rows.append(dict(dataset=name, protocol=proto, feature=f,
                             mean_abs_shap=float(mean_imp[i]), rank=rank[f]))
    out = pd.DataFrame(rows)
    piv = out.pivot_table(index=["dataset", "feature"], columns="protocol",
                          values="rank").reset_index()
    piv["rank_change"] = piv["random"] - piv["grouped"]
    return out, piv


# --------------------------------------------------------------------------
# 5. sensitivity of the roughness reversal
# --------------------------------------------------------------------------
def roughness_sensitivity():
    atf = pd.read_csv(ROOT / "data/processed/surface_master.csv")
    atf = atf.dropna(subset=["CHF_kW_m2", "Ra_um"]).copy()
    atf["logCHF"] = np.log(atf.CHF_kW_m2)
    rows = []
    for label, sub in [("all rows", atf),
                       ("exclude sandblasted (Ra>10um)", atf[atf.Ra_um <= 10]),
                       ("exclude Ra>5um", atf[atf.Ra_um <= 5])]:
        if len(sub) < 8:
            continue
        c = sub.copy()
        c["_f"] = c.Ra_um - c.groupby("study")["Ra_um"].transform("mean")
        c["_t"] = c.logCHF - c.groupby("study")["logCHF"].transform("mean")
        rows.append(dict(subset=label, n=len(sub), studies=sub.study.nunique(),
                         pooled_r=sub.Ra_um.corr(sub.logCHF),
                         within_r=c["_f"].corr(c["_t"]),
                         reversal=bool(np.sign(sub.Ra_um.corr(sub.logCHF)) != np.sign(c["_f"].corr(c["_t"])))))
    return pd.DataFrame(rows)


def main():
    RES.mkdir(parents=True, exist_ok=True)
    fams = load()

    splits, mixed, boots, shaps, shap_ranks = [], [], [], [], []
    for name, d in fams.items():
        print(f"[{name}] four splits ...", flush=True)
        splits.append(four_splits(name, d))
        print(f"[{name}] mixed effects ...", flush=True)
        mixed.append(mixed_effects(name, d))
        print(f"[{name}] bootstrap ({N_BOOT}) ...", flush=True)
        boots.append(boot_corr(name, d))
        print(f"[{name}] SHAP ...", flush=True)
        s, r = shap_by_protocol(name, d)
        shaps.append(s); shap_ranks.append(r)

    S = pd.concat(splits, ignore_index=True); S.to_csv(RES / "four_split_strategies.csv", index=False)
    M = pd.concat(mixed, ignore_index=True); M.to_csv(RES / "mixed_effects_vs_pooled.csv", index=False)
    B = pd.concat(boots, ignore_index=True); B.to_csv(RES / "bootstrap_correlations.csv", index=False)
    pd.concat(shaps, ignore_index=True).to_csv(RES / "shap_importance_by_protocol.csv", index=False)
    R = pd.concat(shap_ranks, ignore_index=True); R.to_csv(RES / "shap_rank_change.csv", index=False)
    SN = roughness_sensitivity(); SN.to_csv(RES / "roughness_sensitivity.csv", index=False)

    pd.set_option("display.width", 200)
    print("\n\n============ 1. FOUR SPLIT STRATEGIES (RandomForest, log CHF) ============")
    print(S.to_string(index=False))
    print("\n============ 2. MIXED EFFECTS vs POOLED OLS ============")
    print(M.to_string(index=False))
    print("\n============ 3. BOOTSTRAP CIs (cluster bootstrap over studies) ============")
    print(B.to_string(index=False))
    print("\n============ 4. SHAP RANK CHANGE (random -> grouped) ============")
    print(R.to_string(index=False))
    print("\n============ 5. ROUGHNESS REVERSAL SENSITIVITY ============")
    print(SN.to_string(index=False))
    print(f"\nAll tables -> {RES}")


if __name__ == "__main__":
    main()

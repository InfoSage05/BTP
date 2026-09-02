"""
run_improvements.py
-------------------
Three follow-up experiments, each aimed at a specific failure the first
ablation diagnosed. Evaluated on strategies 3 and 4 ONLY -- the splits where a
whole source is held out. Strategies 1 and 2 keep the same rig on both sides
and score 0.96-0.98 for a dozen different models, so they cannot discriminate.

E1  BOUND x BASELINE factorial.
    A2 (Katto-Ohno, unbounded) collapsed to -0.797 pooled LOSO, and on the NRC
    fold physics alone scored 0.926 while the learned correction dragged it to
    -1.286. Bounding is precisely the fix for that failure -- but in the first
    ablation the bound was only ever tested combined with the gated baseline
    AND the pi feature space (arm A5). The pairing "good baseline + bounded
    correction" was never run. This runs it.

E2  TRUST-DECAY SWEEP.
    A5's trust decay is effectively binary: 100% of strategy-3 test rows get a
    trust weight below 0.01, so the learned correction switches off entirely
    and A5 reduces exactly to the physics. That is safe but wastes the ML.
    Sweeping trust_gamma asks whether a graded decay keeps the safety while
    recovering some of the correction.

E3  CORRELATION STACKING.
    Dividing by ONE correlation forces a choice the data does not support: no
    single correlation wins across all seven rigs (Katto-Ohno is 0.926 on NRC
    and -0.362 on zhao2020). Instead of choosing, hand the model every
    correlation's predicted log(Bo) as a feature and let it learn where each is
    trustworthy. Target stays log(Bo) with the latent baseline, so "just copy
    Katto-Ohno" is representable and the model has to beat it.

Run:  python physics_pipeline/scripts/run_improvements.py
"""
import warnings

import numpy as np
import pandas as pd

import features_v2 as F
import paths
from metrics_utils import compute_metrics
from models_v2 import PhysicsCorrectedModel
from physics import baseline as phys_baseline
from physics import repair as phys_repair

warnings.filterwarnings("ignore")

paths.check_inputs()
RESULTS = paths.ensure_output_dirs()

BOUND = 3.0

# name -> (baseline_mode, space, bound, monotone, trust_decay, trust_gamma)
EXPERIMENTS = {
    # --- reference points carried over from the first ablation -------------
    "REF_A1_latent_unbounded":      ("latent", "raw", None, False, False, 1.0),
    "REF_A2_katto_unbounded":       ("katto",  "raw", None, False, False, 1.0),
    "REF_A5_gated_pi_constrained":  ("gated",  "pi",  BOUND, True, True, 1.0),

    # --- E1: bound x baseline factorial (the untested pairings) ------------
    "E1_latent_raw_bounded":        ("latent", "raw", BOUND, True, True, 1.0),
    "E1_katto_raw_bounded":         ("katto",  "raw", BOUND, True, True, 1.0),
    "E1_katto_pi_bounded":          ("katto",  "pi",  BOUND, True, True, 1.0),
    "E1_katto_raw_bound_notrust":   ("katto",  "raw", BOUND, True, False, 1.0),

    # --- E2: trust-decay sweep on the E1 winner's configuration ------------
    "E2_katto_raw_gamma0.25":       ("katto",  "raw", BOUND, True, True, 0.25),
    "E2_katto_raw_gamma0.5":        ("katto",  "raw", BOUND, True, True, 0.5),
    "E2_katto_raw_gamma2.0":        ("katto",  "raw", BOUND, True, True, 2.0),

    # --- E3: correlation stacking ------------------------------------------
    "E3_stack_raw_unbounded":       ("latent", "raw_corr", None, False, False, 1.0),
    "E3_stack_pi_unbounded":        ("latent", "pi_corr",  None, False, False, 1.0),
    "E3_stack_raw_bounded":         ("latent", "raw_corr", BOUND, True, True, 1.0),
    "E3_stack_pi_bounded":          ("latent", "pi_corr",  BOUND, True, True, 1.0),
}

PHYSICS_ONLY = {"PHYS_katto": "katto", "PHYS_gated": "gated"}


def main():
    raw = pd.read_csv(paths.MASTER_CSV, low_memory=False)
    master = phys_repair.repair(raw)

    modes = {cfg[0] for cfg in EXPERIMENTS.values()} | set(PHYSICS_ONLY.values())
    prepared = {}
    for mode in sorted(modes):
        print(f"preparing baseline mode {mode!r} ...")
        prepared[mode] = F.prepare(master, baseline_mode=mode).set_index("row_id", drop=False)
    ref = prepared["katto"]

    rows = []

    def fit_predict(cfg, tr, te):
        mode, space, bound, mono, trust, gamma = cfg
        prep = prepared[mode]
        X = F.build_matrix(prep, space)
        y, b = prep["CHF_kW_m2"], prep["physics_baseline_kW_m2"]
        w = F.sample_weights_by_source(prep.loc[tr, "source_dataset"])
        m = PhysicsCorrectedModel(base="histgb", space=space, bound=bound,
                                  monotone=mono, trust_decay=trust, trust_gamma=gamma)
        m.fit(X.loc[tr], y.loc[tr], b.loc[tr], sample_weight=w)
        pred = m.predict(X.loc[te], b.loc[te])
        tw = m._trust_weight(X.loc[te])
        tw_med = float(np.median(tw)) if np.ndim(tw) else float(tw)
        return pred, tw_med

    # ---- strategy 3 --------------------------------------------------------
    s3 = pd.read_csv(paths.SPLITS_DIR / paths.STRATEGY_FILES["strategy3"])
    tr3 = pd.Index(s3.loc[s3["split"] == "train", "row_id"])
    te3 = pd.Index(s3.loc[s3["split"] == "test", "row_id"])
    print(f"\n=== strategy3: train={len(tr3)} test={len(te3)} ===")

    for name, mode in PHYSICS_ONLY.items():
        p = prepared[mode].loc[te3, "physics_baseline_kW_m2"].values
        m = compute_metrics(ref.loc[te3, "CHF_kW_m2"], p)
        m.update({"arm": name, "split": "strategy3", "trust_weight_median": np.nan})
        rows.append(m)
    for name, cfg in EXPERIMENTS.items():
        pred, tw = fit_predict(cfg, tr3, te3)
        m = compute_metrics(ref.loc[te3, "CHF_kW_m2"], pred)
        m.update({"arm": name, "split": "strategy3", "trust_weight_median": tw})
        rows.append(m)
        print(f"  {name:32s} R2={m['R2']:9.4f}  MAPE%={m['MAPE_pct']:7.2f}  trust={tw:.3f}")

    # ---- strategy 4 (leave-one-source-out) --------------------------------
    s4 = pd.read_csv(paths.SPLITS_DIR / paths.LOSO_FILE)
    sources = sorted(s4["fold"].unique())
    names = list(PHYSICS_ONLY) + list(EXPERIMENTS)
    oof = {n: pd.Series(index=ref.index, dtype=float) for n in names}

    for held in sources:
        te = pd.Index(s4.loc[s4["fold"] == held, "row_id"])
        tr = pd.Index(s4.loc[s4["fold"] != held, "row_id"])
        print(f"\n=== LOSO fold {held}: train={len(tr)} test={len(te)} ===")
        for name, mode in PHYSICS_ONLY.items():
            oof[name].loc[te] = prepared[mode].loc[te, "physics_baseline_kW_m2"].values
        for name, cfg in EXPERIMENTS.items():
            pred, _ = fit_predict(cfg, tr, te)
            oof[name].loc[te] = pred
            r2 = compute_metrics(ref.loc[te, "CHF_kW_m2"], pred)["R2"]
            print(f"  {name:32s} R2={r2:10.4f}")

    for name in names:
        m = compute_metrics(ref["CHF_kW_m2"], oof[name].values)
        m.update({"arm": name, "split": "strategy4_pooled_oof", "trust_weight_median": np.nan})
        rows.append(m)

    summary = pd.DataFrame(rows)
    summary.to_csv(RESULTS / "improvement_metrics.csv", index=False)

    oof_df = pd.DataFrame({"row_id": ref.index, "source_dataset": ref["source_dataset"],
                           "chf_regime": ref["chf_regime"], "y_true": ref["CHF_kW_m2"].values})
    for name in names:
        oof_df[f"y_pred_{name}"] = oof[name].values
    oof_df.to_csv(paths.PREDICTIONS_DIR / "improvement_loso_oof.csv", index=False)

    # ---- report ------------------------------------------------------------
    piv = summary.pivot_table(index="arm", columns="split", values="R2").reindex(names)
    from sklearn.metrics import r2_score
    per_fold = {}
    for name in names:
        col = oof_df[f"y_pred_{name}"]
        per_fold[name] = {src: round(float(r2_score(g["y_true"], g[f"y_pred_{name}"])), 3)
                          for src, g in oof_df.groupby("source_dataset")}
    fold_tbl = pd.DataFrame(per_fold).T.reindex(names)

    lines = ["# Improvement experiments", "",
             "Evaluated on strategies 3 and 4 only -- the splits where a whole source is",
             "held out. See the module docstring in `run_improvements.py` for what each",
             "experiment is testing and why.", "",
             "## R2", "", piv.round(4).to_markdown(), "",
             "## Leave-one-source-out R2 per fold", "", fold_tbl.to_markdown(), "",
             "## Median trust weight on the strategy-3 test set", "",
             "A value near 0 means the learned correction is switched off and the arm has",
             "reduced to pure physics; near 1 means the correction is fully active.", "",
             summary[summary["split"] == "strategy3"][["arm", "trust_weight_median"]]
             .set_index("arm").round(4).to_markdown(), ""]
    (RESULTS / "improvement_report.md").write_text("\n".join(lines))

    print("\n" + "=" * 78)
    print(piv.round(4).to_string())
    print(f"\nWrote {RESULTS/'improvement_report.md'}")


if __name__ == "__main__":
    main()

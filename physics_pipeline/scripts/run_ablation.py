"""
run_ablation.py
---------------
The physics ablation ladder. Each rung adds ONE idea, so the contribution of
each can be read off in isolation rather than inferred from a single
end-to-end number.

    A0  no physics                raw features, target log(CHF)
    A1  latent-heat baseline      raw features  <- what the pipeline does TODAY
    A2  + Katto-Ohno baseline     raw features  <- idea 1
    A3  + dimensionless features  pi space      <- idea 2
    A4  + mechanism gating        pi space      <- idea 4
    A5  + bounded/monotone/trust  pi space      <- idea 5

    PHYS_katto / PHYS_gated  -- the closed-form physics ALONE, no learning at
    all. The reference every learned arm has to beat to justify its existence.

    A1_ANN / A5_ANN -- the same ladder ends run with an MLP instead of HistGB,
    to test whether the structural guarantees fix the ANN instability that
    made strategy 3 return R^2 = -4132 in the original pipeline.

Idea 3 (the physics-consistency scorecard) is not a rung: it is applied to
EVERY arm, as extra columns beside R2/RMSE/MAE/MAPE.

Run:  python physics_pipeline/scripts/run_ablation.py
"""
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

import features_v2 as F
from metrics_utils import compute_metrics
from models_v2 import DEFAULT_CORRECTION_BOUND, PhysicsCorrectedModel
from physics import baseline as phys_baseline
from physics import constraints as phys_constraints
from physics import repair as phys_repair

warnings.filterwarnings("ignore")

import paths

paths.check_inputs()
RESULTS = paths.ensure_output_dirs()
DATA, SPLITS = paths.DATA_DIR, paths.SPLITS_DIR

# name -> (baseline_mode, space, base, bound, monotone, trust_decay)
ARMS = {
    "A0_no_physics":      ("none",   "raw", "histgb", None, False, False),
    "A1_latent_baseline": ("latent", "raw", "histgb", None, False, False),
    "A2_katto_baseline":  ("katto",  "raw", "histgb", None, False, False),
    "A3_pi_features":     ("katto",  "pi",  "histgb", None, False, False),
    "A4_mechanism_gated": ("gated",  "pi",  "histgb", None, False, False),
    "A5_constrained":     ("gated",  "pi",  "histgb", DEFAULT_CORRECTION_BOUND, True, True),
    "A1_ANN":             ("latent", "raw", "ann",    None, False, False),
    "A5_ANN":             ("gated",  "pi",  "ann",    DEFAULT_CORRECTION_BOUND, False, True),
}

PHYSICS_ONLY_ARMS = {"PHYS_katto": "katto", "PHYS_gated": "gated"}


def load_prepared():
    """Load, repair, and prepare the master table once per baseline mode."""
    raw = pd.read_csv(paths.MASTER_CSV, low_memory=False)
    repaired = phys_repair.repair(raw)
    print(phys_repair.repair_report(raw, repaired))
    prepared = {}
    for mode in phys_baseline.BASELINE_MODES:
        print(f"  preparing features for baseline mode {mode!r} ...")
        p = F.prepare(repaired, baseline_mode=mode)
        prepared[mode] = p.set_index("row_id", drop=False)
    return repaired, prepared


def make_predict_fn(model, mode, space, template_cols):
    """Closure for the constraint probes: raw-schema frame -> CHF kW/m^2."""
    def predict_fn(df_probe):
        df_probe = df_probe.reindex(columns=template_cols)
        prep = F.prepare(phys_repair.repair(df_probe), baseline_mode=mode)
        X = F.build_matrix(prep, space)
        return model.predict(X, prep["physics_baseline_kW_m2"])
    return predict_fn


def evaluate(y_true, y_pred, df_test, scorecard_extra=None):
    m = compute_metrics(y_true, y_pred)
    m.update(phys_constraints.score_predictions(y_pred, df_test))
    if scorecard_extra:
        m.update(scorecard_extra)
    return m


def run_split(arm, cfg, prepared, train_ids, test_ids, tag, template, rows, do_probe):
    mode, space, base, bound, monotone, trust = cfg
    prep = prepared[mode]
    X = F.build_matrix(prep, space)
    y = prep["CHF_kW_m2"]
    b = prep["physics_baseline_kW_m2"]

    w = F.sample_weights_by_source(prep.loc[train_ids, "source_dataset"])
    model = PhysicsCorrectedModel(base=base, bound=bound, monotone=monotone,
                                  trust_decay=trust, space=space)
    model.fit(X.loc[train_ids], y.loc[train_ids], b.loc[train_ids], sample_weight=w)
    y_pred = model.predict(X.loc[test_ids], b.loc[test_ids])

    extra = {}
    if do_probe:
        try:
            fn = make_predict_fn(model, mode, space, list(template.columns))
            extra = phys_constraints.probe_model(fn, template)
        except Exception as exc:
            extra = {"probe_errors": f"{type(exc).__name__}: {exc}"}

    m = evaluate(y.loc[test_ids], y_pred, prep.loc[test_ids], extra)
    m.update({"arm": arm, "split_tag": tag, "train_seconds": model.train_seconds_,
              "monotone_applied": model._monotone_applied})
    rows.append(m)
    return pd.Series(y_pred, index=test_ids)


def run_physics_only(name, mode, prepared, test_ids, tag, rows):
    prep = prepared[mode]
    y_pred = prep.loc[test_ids, "physics_baseline_kW_m2"].values
    m = evaluate(prep.loc[test_ids, "CHF_kW_m2"], y_pred, prep.loc[test_ids])
    m.update({"arm": name, "split_tag": tag, "train_seconds": 0.0})
    rows.append(m)
    return pd.Series(y_pred, index=test_ids)


def main():
    repaired, prepared = load_prepared()
    template = repaired.copy()
    rows = []

    strategies = [(i, fname) for i, fname in
                  enumerate(paths.STRATEGY_FILES.values(), start=1)]

    for num, fname in strategies:
        split = pd.read_csv(SPLITS / fname)
        train_ids = pd.Index(split.loc[split["split"] == "train", "row_id"])
        test_ids = pd.Index(split.loc[split["split"] == "test", "row_id"])
        tag = f"strategy{num}"
        print(f"\n=== {tag}: train={len(train_ids)} test={len(test_ids)} ===")

        for name, mode in PHYSICS_ONLY_ARMS.items():
            run_physics_only(name, mode, prepared, test_ids, tag, rows)
        for arm, cfg in ARMS.items():
            run_split(arm, cfg, prepared, train_ids, test_ids, tag, template, rows,
                      do_probe=True)
            last = rows[-1]
            print(f"  {arm:22s} R2={last['R2']:9.4f}  MAPE%={last['MAPE_pct']:8.2f}  "
                  f"C3viol={last.get('C3_quality_violation_frac', np.nan):.2f}")

    # ---- strategy 4: leave-one-source-out ---------------------------------
    s4 = pd.read_csv(SPLITS / paths.LOSO_FILE)
    sources = sorted(s4["fold"].unique())
    all_names = list(PHYSICS_ONLY_ARMS) + list(ARMS)
    oof = {n: pd.Series(index=prepared["katto"].index, dtype=float) for n in all_names}
    fold_rows = []

    for held in sources:
        test_ids = pd.Index(s4.loc[s4["fold"] == held, "row_id"])
        train_ids = pd.Index(s4.loc[s4["fold"] != held, "row_id"])
        tag = f"strategy4_fold={held}"
        print(f"\n=== {tag}: train={len(train_ids)} test={len(test_ids)} ===")
        for name, mode in PHYSICS_ONLY_ARMS.items():
            oof[name].loc[test_ids] = run_physics_only(
                name, mode, prepared, test_ids, tag, fold_rows).values
        for arm, cfg in ARMS.items():
            oof[arm].loc[test_ids] = run_split(
                arm, cfg, prepared, train_ids, test_ids, tag, template, fold_rows,
                do_probe=False).values
            last = fold_rows[-1]
            print(f"  {arm:22s} R2={last['R2']:12.4f}  MAPE%={last['MAPE_pct']:9.2f}")

    for r in fold_rows:
        r["held_out_source"] = r["split_tag"].split("=", 1)[1]
    rows.extend(fold_rows)

    prep_ref = prepared["katto"]
    for name in all_names:
        s = oof[name]
        m = evaluate(prep_ref["CHF_kW_m2"], s.values, prep_ref)
        m.update({"arm": name, "split_tag": "strategy4_pooled_oof"})
        rows.append(m)

    summary = pd.DataFrame(rows)
    lead = ["split_tag", "arm", "n_test", "R2", "RMSE", "MAE", "MAPE_pct", "train_seconds"]
    summary = summary[lead + [c for c in summary.columns if c not in lead]]
    summary.to_csv(RESULTS / "ablation_metrics.csv", index=False)

    oof_df = pd.DataFrame({"row_id": prep_ref.index,
                            "source_dataset": prep_ref["source_dataset"],
                            "chf_regime": prep_ref["chf_regime"],
                            "y_true": prep_ref["CHF_kW_m2"].values})
    for name in all_names:
        oof_df[f"y_pred_{name}"] = oof[name].values
    oof_df.to_csv(RESULTS / "predictions" / "ablation_loso_oof.csv", index=False)

    print(f"\nWrote {RESULTS/'ablation_metrics.csv'} ({len(summary)} rows)")


if __name__ == "__main__":
    main()

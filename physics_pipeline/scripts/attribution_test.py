"""
attribution_test.py
-------------------
Isolates WHY strategy 3 (surface-wise, the hardest split) improved, using the
ORIGINAL features.py / models.py so the comparison is against the published
number rather than against a rewritten pipeline.

The ablation ladder in run_ablation.py compares physics ideas against arm A1.
But A1 is not literally today's pipeline -- it also carries the Stage-0 data
repair, FC-72 saturation properties, and a different subcooling column. This
script separates those data fixes from the modelling ideas, so neither gets
credit for the other's work.

Run:  python physics_pipeline/scripts/attribution_test.py
"""
import sys
import warnings

import pandas as pd

import paths

# The ORIGINAL bake-off modules live in the shared pipeline. They are imported
# from there rather than copied, so this comparison always runs against the
# code that produced the published numbers. This is the only place in
# physics_pipeline that reaches into unified_chf_pipeline for CODE (everywhere
# else it only reads data), and it is deliberate: the whole point of this
# script is to compare against the unmodified original.
sys.path.insert(0, str(paths.SHARED_SCRIPTS))
import features as F_old          # noqa: E402  -- unified_chf_pipeline/scripts
import models as M_old            # noqa: E402  -- unified_chf_pipeline/scripts

import features_v2 as F_new       # noqa: E402  -- this folder
from metrics_utils import compute_metrics  # noqa: E402
from models_v2 import PhysicsCorrectedModel  # noqa: E402
from physics import repair as phys_repair  # noqa: E402

warnings.filterwarnings("ignore")

DATA, SPLITS = paths.DATA_DIR, paths.SPLITS_DIR


def _split():
    s = pd.read_csv(SPLITS / "strategy3_surface_wise.csv")
    return (pd.Index(s.loc[s["split"] == "train", "row_id"]),
            pd.Index(s.loc[s["split"] == "test", "row_id"]))


def run_old(df, label, subcool_col=None, tr=None, te=None):
    """Original pipeline, optionally with one column swapped."""
    full = F_old.add_fluid_properties(df)
    full = F_old.add_categorical_onehot(full)
    full["physics_baseline_kW_m2"] = F_old.compute_physics_baseline_kw_m2(full)
    full = full.set_index("row_id", drop=False)
    cols = [subcool_col if (subcool_col and c == "subcooling_K") else c
            for c in F_old.FEATURE_COLUMNS]
    X, y, b = full[cols], full["CHF_kW_m2"], full["physics_baseline_kW_m2"]
    w = F_old.sample_weights_by_source(full.loc[tr, "source_dataset"])
    y_pred, _ = M_old.fit_predict_sklearn_model(
        M_old.build_hist_gb, X.loc[tr], y.loc[tr], b.loc[tr],
        X.loc[te], b.loc[te], sample_weight=w)
    m = compute_metrics(y.loc[te], y_pred)
    print(f"  {label:56s} R2={m['R2']:8.4f}  MAPE%={m['MAPE_pct']:7.2f}")
    return m["R2"]


def main():
    raw = pd.read_csv(paths.MASTER_CSV, low_memory=False)
    tr, te = _split()
    repaired = phys_repair.repair(raw)

    print("Strategy 3 (surface-wise), original features.py + HistGB")
    print("-" * 78)
    run_old(raw, "as published", None, tr, te)
    run_old(repaired, "+ Stage-0 repair (helical diameters)", None, tr, te)
    run_old(raw, "+ subcooling_kJkg (86.3% cov) not subcooling_K (0.8%)",
            "subcooling_kJkg", tr, te)
    run_old(repaired, "+ repair + subcooling_kJkg", "subcooling_kJkg", tr, te)

    print()
    print("Same split, new feature module (adds FC-72 saturation properties)")
    print("-" * 78)
    prep = F_new.prepare(repaired, baseline_mode="latent").set_index("row_id", drop=False)
    X = F_new.build_matrix(prep, "raw")
    y, b = prep["CHF_kW_m2"], prep["physics_baseline_kW_m2"]
    w = F_new.sample_weights_by_source(prep.loc[tr, "source_dataset"])

    for label, baseline in (
            ("A1 (FC-72 properties available)", b),
            ("A1 with FC-72 baseline forced back to 1.0", None)):
        bb = b.copy()
        if baseline is None:
            bb.loc[prep["fluid"] == "fc-72"] = 1.0
        model = PhysicsCorrectedModel(base="histgb", space="raw")
        model.fit(X.loc[tr], y.loc[tr], bb.loc[tr], sample_weight=w)
        m = compute_metrics(y.loc[te], model.predict(X.loc[te], bb.loc[te]))
        print(f"  {label:56s} R2={m['R2']:8.4f}  MAPE%={m['MAPE_pct']:7.2f}")

    n_fc72 = int((prep.loc[te, "fluid"] == "fc-72").sum())
    print()
    print(f"  {n_fc72} of {len(te)} strategy-3 test rows are FC-72. Before this fix they")
    print("  had NaN properties, so the physics baseline fell back to 1.0 and the")
    print("  model was predicting raw log(CHF) for them.")


if __name__ == "__main__":
    main()

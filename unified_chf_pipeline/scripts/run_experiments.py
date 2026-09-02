"""
run_experiments.py
--------------------
Trains every model in the bake-off (ANN, Random Forest, HistGB, GPR, and the
proposed Tier-0+Tier-1 hierarchy) on each of the 4 split strategies, tests
them on the held-out rows, and stores R2 / RMSE / MAE / MAPE / training time
for all of them.

Run (after merge_datasets.py and build_splits.py have produced their outputs):
    python unified_chf_pipeline/scripts/run_experiments.py

Writes, into unified_chf_pipeline/results/:
    metrics_summary.csv         -- one row per (model, strategy[, fold])
    loso_fold_details.csv       -- strategy-4 per-fold breakdown
    predictions_strategy1.csv, predictions_strategy2.csv, predictions_strategy3.csv
    predictions_strategy4_oof.csv   -- out-of-fold predictions for every row
    results_report.md           -- human-readable summary tables
"""
import time
from pathlib import Path

import pandas as pd

from features import (
    FEATURE_COLUMNS, add_categorical_onehot, add_fluid_properties,
    compute_physics_baseline_kw_m2, sample_weights_by_source,
)
from metrics_utils import compute_metrics
from models import (
    build_ann, build_hist_gb, build_random_forest, fit_predict_gpr,
    fit_predict_sklearn_model, fit_tier0, fit_tier1_models, predict_hierarchy,
    predict_tier0,
)
from pinn_model import build_pinn

SCRIPTS_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPTS_DIR.parent / "data"
SPLITS_DIR = SCRIPTS_DIR.parent / "splits"
RESULTS_DIR = SCRIPTS_DIR.parent / "results"
(RESULTS_DIR / "predictions").mkdir(parents=True, exist_ok=True)

FLAT_MODEL_BUILDERS = {
    "ANN": build_ann,
    "RandomForest": build_random_forest,
    "HistGB": build_hist_gb,
    "PINN": build_pinn,
}


def load_full_table() -> pd.DataFrame:
    master = pd.read_csv(DATA_DIR / "master_chf_dataset.csv", low_memory=False)
    full = add_fluid_properties(master)
    full = add_categorical_onehot(full)
    full["physics_baseline_kW_m2"] = compute_physics_baseline_kw_m2(full)
    full = full.set_index("row_id", drop=False)
    return full


def run_flat_models_on_split(full, X, y, baseline, train_ids, test_ids, tag: str,
                              metrics_rows: list, preds: dict):
    X_train, y_train = X.loc[train_ids], y.loc[train_ids]
    X_test, y_test = X.loc[test_ids], y.loc[test_ids]
    baseline_train, baseline_test = baseline.loc[train_ids], baseline.loc[test_ids]
    weights_train = sample_weights_by_source(full.loc[train_ids, "source_dataset"])

    for name, builder in FLAT_MODEL_BUILDERS.items():
        y_pred, train_seconds = fit_predict_sklearn_model(
            builder, X_train, y_train, baseline_train, X_test, baseline_test,
            sample_weight=weights_train)
        m = compute_metrics(y_test, y_pred, train_seconds)
        m.update({"model": name, "split_tag": tag})
        metrics_rows.append(m)
        preds[name] = pd.Series(y_pred, index=test_ids)

    y_pred, train_seconds, n_used = fit_predict_gpr(X_train, y_train, baseline_train, X_test, baseline_test)
    m = compute_metrics(y_test, y_pred, train_seconds)
    m.update({"model": "GPR", "split_tag": tag, "gpr_train_rows_used": n_used})
    metrics_rows.append(m)
    preds["GPR"] = pd.Series(y_pred, index=test_ids)

    return preds


def run_hierarchy_on_split(full, X, y, baseline, train_ids, test_ids, tag: str,
                            metrics_rows: list, preds: dict):
    df_train = full.loc[train_ids]
    df_test = full.loc[test_ids]
    X_train, y_train = X.loc[train_ids], y.loc[train_ids]
    X_test, y_test = X.loc[test_ids], y.loc[test_ids]
    baseline_train, baseline_test = baseline.loc[train_ids], baseline.loc[test_ids]
    weights_train = sample_weights_by_source(df_train["source_dataset"])

    t0 = time.time()
    tier0_model, tier0_seconds = fit_tier0(X_train, y_train, baseline_train, sample_weight=weights_train)
    tier0_pred_train = predict_tier0(tier0_model, X_train, baseline_train)
    tier1_models = fit_tier1_models(df_train, y_train, tier0_pred_train)
    train_seconds = time.time() - t0

    y_pred, used_tier1 = predict_hierarchy(tier0_model, tier1_models, df_test, X_test, baseline_test)
    m = compute_metrics(y_test, y_pred, train_seconds)
    m.update({
        "model": "Proposed_Hierarchy", "split_tag": tag,
        "n_test_with_tier1_correction": int(used_tier1.sum()),
        "tier1_families_fitted": ",".join(f for f, mdl in tier1_models.items() if mdl is not None) or "(none)",
    })
    metrics_rows.append(m)
    preds["Proposed_Hierarchy"] = pd.Series(y_pred, index=test_ids)

    # Breakdown: accuracy on the subset that actually got a Tier-1 correction
    # vs the subset that fell back to Tier-0 alone (only meaningful if both
    # subsets are non-empty).
    if 0 < used_tier1.sum() < len(used_tier1):
        with_t1_ids = test_ids[used_tier1]
        without_t1_ids = test_ids[~used_tier1]
        m_with = compute_metrics(y_test.loc[with_t1_ids], pd.Series(y_pred, index=test_ids).loc[with_t1_ids])
        m_with.update({"model": "Proposed_Hierarchy__WITH_tier1", "split_tag": tag})
        metrics_rows.append(m_with)
        m_without = compute_metrics(y_test.loc[without_t1_ids], pd.Series(y_pred, index=test_ids).loc[without_t1_ids])
        m_without.update({"model": "Proposed_Hierarchy__TIER0_ONLY", "split_tag": tag})
        metrics_rows.append(m_without)

    return preds


def main():
    full = load_full_table()
    X = full[FEATURE_COLUMNS]
    y = full["CHF_kW_m2"]
    baseline = full["physics_baseline_kW_m2"]

    metrics_rows = []

    # ---- Strategies 1-3: single train/test split each -------------------
    for strategy_num, fname in [(1, "strategy1_random_stratified.csv"),
                                 (2, "strategy2_condition_wise.csv"),
                                 (3, "strategy3_surface_wise.csv")]:
        split = pd.read_csv(SPLITS_DIR / fname)
        train_ids = pd.Index(split.loc[split["split"] == "train", "row_id"])
        test_ids = pd.Index(split.loc[split["split"] == "test", "row_id"])
        tag = f"strategy{strategy_num}"
        print(f"\n=== {tag} ({fname}): train={len(train_ids)} test={len(test_ids)} ===")

        preds = {}
        run_flat_models_on_split(full, X, y, baseline, train_ids, test_ids, tag, metrics_rows, preds)
        run_hierarchy_on_split(full, X, y, baseline, train_ids, test_ids, tag, metrics_rows, preds)

        pred_df = pd.DataFrame({"row_id": test_ids, "y_true": y.loc[test_ids].values})
        for model_name, s in preds.items():
            pred_df[f"y_pred_{model_name}"] = s.loc[test_ids].values
        pred_df.to_csv(RESULTS_DIR / "predictions" / f"predictions_{tag}.csv", index=False)

        for row in metrics_rows:
            if row["split_tag"] == tag:
                print(f"  {row['model']:<28} R2={row.get('R2'):.4f}  RMSE={row.get('RMSE'):.1f}  "
                      f"MAE={row.get('MAE'):.1f}  MAPE%={row.get('MAPE_pct'):.2f}")

    # ---- Strategy 4: leave-one-source-out --------------------------------
    s4 = pd.read_csv(SPLITS_DIR / "strategy4_leave_one_source_out.csv")
    sources = sorted(s4["fold"].unique())
    loso_rows = []
    oof_preds = {name: pd.Series(index=full.index, dtype=float) for name in
                 list(FLAT_MODEL_BUILDERS) + ["GPR", "Proposed_Hierarchy"]}

    for held_out in sources:
        test_ids = pd.Index(s4.loc[s4["fold"] == held_out, "row_id"])
        train_ids = pd.Index(s4.loc[s4["fold"] != held_out, "row_id"])
        tag = f"strategy4_fold={held_out}"
        print(f"\n=== {tag}: train={len(train_ids)} test={len(test_ids)} ===")

        fold_metrics = []
        preds = {}
        run_flat_models_on_split(full, X, y, baseline, train_ids, test_ids, tag, fold_metrics, preds)
        run_hierarchy_on_split(full, X, y, baseline, train_ids, test_ids, tag, fold_metrics, preds)

        for name, s in preds.items():
            if name in oof_preds:
                oof_preds[name].loc[test_ids] = s.loc[test_ids].values

        for row in fold_metrics:
            row["held_out_source"] = held_out
            loso_rows.append(row)
            metrics_rows.append(row)
        for row in fold_metrics:
            if "__" not in row["model"]:
                print(f"  {row['model']:<28} R2={row.get('R2'):.4f}  RMSE={row.get('RMSE'):.1f}  "
                      f"MAE={row.get('MAE'):.1f}  MAPE%={row.get('MAPE_pct'):.2f}")

    # Pooled leave-one-source-out metric (every row tested exactly once, out-of-fold)
    for name, s in oof_preds.items():
        m = compute_metrics(y.loc[s.index], s.values)
        m.update({"model": name, "split_tag": "strategy4_pooled_oof"})
        metrics_rows.append(m)

    oof_df = pd.DataFrame({"row_id": full.index, "source_dataset": full["source_dataset"],
                            "y_true": y.values})
    for name, s in oof_preds.items():
        oof_df[f"y_pred_{name}"] = s.values
    oof_df.to_csv(RESULTS_DIR / "predictions" / "predictions_strategy4_oof.csv", index=False)

    pd.DataFrame(loso_rows).to_csv(RESULTS_DIR / "loso_fold_details.csv", index=False)

    # ---- Save everything --------------------------------------------------
    summary = pd.DataFrame(metrics_rows)
    col_order = ["split_tag", "model", "n_test", "R2", "RMSE", "MAE", "MAPE_pct",
                 "train_seconds"] + [c for c in summary.columns if c not in
                 {"split_tag", "model", "n_test", "R2", "RMSE", "MAE", "MAPE_pct", "train_seconds"}]
    summary = summary[col_order]
    summary.to_csv(RESULTS_DIR / "metrics_summary.csv", index=False)

    report_lines = ["# Model Comparison Results\n"]
    for tag in ["strategy1", "strategy2", "strategy3", "strategy4_pooled_oof"]:
        sub = summary[summary["split_tag"] == tag]
        if len(sub) == 0:
            continue
        report_lines.append(f"## {tag}\n")
        report_lines.append(sub[["model", "n_test", "R2", "RMSE", "MAE", "MAPE_pct", "train_seconds"]]
                             .to_markdown(index=False))
        report_lines.append("")
    (RESULTS_DIR / "results_report.md").write_text("\n".join(report_lines))

    print(f"\nWrote {RESULTS_DIR / 'metrics_summary.csv'} ({len(summary)} rows)")
    print(f"Wrote {RESULTS_DIR / 'loso_fold_details.csv'}")
    print(f"Wrote {RESULTS_DIR / 'results_report.md'}")


if __name__ == "__main__":
    main()

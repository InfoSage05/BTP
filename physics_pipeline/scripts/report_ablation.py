"""
report_ablation.py
------------------
Turns `results/ablation_metrics.csv` into the comparison report.

The report deliberately puts accuracy and physics-consistency side by side.
Foundation doc section 9.5: a model at R^2 = 0.95 that violates the
monotonicity constraint and misses the pressure maximum is worse than one at
R^2 = 0.85 that satisfies both, because only the latter can be trusted at a
condition nobody has tested.

Run:  python physics_pipeline/scripts/report_ablation.py
"""
from pathlib import Path

import numpy as np
import pandas as pd

import paths

RESULTS = paths.RESULTS_DIR

ARM_ORDER = [
    "PHYS_katto", "PHYS_gated",
    "A0_no_physics", "A1_latent_baseline", "A2_katto_baseline",
    "A3_pi_features", "A4_mechanism_gated", "A5_constrained",
    "A1_ANN", "A5_ANN",
]

ARM_LABEL = {
    "PHYS_katto": "Physics only (Katto-Ohno), no learning",
    "PHYS_gated": "Physics only (gated), no learning",
    "A0_no_physics": "A0  no physics, raw features",
    "A1_latent_baseline": "A1  latent-heat baseline  [today's pipeline]",
    "A2_katto_baseline": "A2  + Katto-Ohno baseline  [idea 1]",
    "A3_pi_features": "A3  + dimensionless features  [idea 2]",
    "A4_mechanism_gated": "A4  + mechanism gating  [idea 4]",
    "A5_constrained": "A5  + bounded/monotone/trust  [idea 5]",
    "A1_ANN": "A1-ANN  latent baseline, MLP",
    "A5_ANN": "A5-ANN  constrained, MLP",
}

SPLIT_LABEL = {
    "strategy1": "Strategy 1 - random (optimistic)",
    "strategy2": "Strategy 2 - condition-wise (pressure extrapolation)",
    "strategy3": "Strategy 3 - surface-wise (held-out surface types)",
    "strategy4_pooled_oof": "Strategy 4 - leave-one-source-out (pooled)",
}

ACC_COLS = ["R2", "RMSE", "MAE", "MAPE_pct", "train_seconds"]
PHYS_COLS = [
    "C1_nonpositive_frac", "C3_quality_violation_frac", "C4_massflux_violation_frac",
    "C6_satisfied", "S1_peak_reduced_pressure", "S6_sign_reversal_captured",
    "S7_crossover_captured", "S2_pool_K_median",
]


def _fmt(v, nd=4):
    if isinstance(v, (bool, np.bool_)):
        return "yes" if v else "no"
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "-"
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    return f"{v:.{nd}f}"


def table(df, cols, nd=4):
    out = df.copy()
    for c in cols:
        if c in out.columns:
            out[c] = out[c].map(lambda v: _fmt(v, nd))
    keep = ["arm"] + [c for c in cols if c in out.columns]
    return out[keep].to_markdown(index=False)


def main():
    m = pd.read_csv(RESULTS / "ablation_metrics.csv")
    m["arm"] = pd.Categorical(m["arm"], categories=ARM_ORDER, ordered=True)

    lines = ["# Physics Ablation Results", ""]
    lines.append("## Read this first")
    lines.append("")
    lines.append("**Arm A1 is NOT literally today's pipeline.** It shares the latent-heat")
    lines.append("baseline, but it also carries the Stage-0 data repair, FC-72 saturation")
    lines.append("properties, and `subcooling_kJkg` in place of the 0.8%-coverage")
    lines.append("`subcooling_K`. `attribution_test.py` separates those data fixes from the")
    lines.append("modelling ideas; on strategy 3 they account for 0.173 -> 0.711 on their")
    lines.append("own, before any physics idea is applied. Credit the data, not the models.")
    lines.append("")
    lines.append("**A5's out-of-domain numbers are the physics, not the model.** Trust decay")
    lines.append("drives the learned correction to zero for 100% of strategy-3 test rows, so")
    lines.append("A5 there reduces exactly to `PHYS_gated`. That is the designed behaviour")
    lines.append("(fall back to physics off-manifold) but it means A5 can only MATCH pure")
    lines.append("physics out of domain, never beat it. The decay is currently binary rather")
    lines.append("than graded -- softening `trust_gamma` is the obvious next experiment.")
    lines.append("")
    lines.append("Each rung adds one idea to the rung above it, so the contribution of")
    lines.append("each is isolated. `PHYS_*` arms are the closed-form physics with no")
    lines.append("learning at all -- the reference every learned arm must beat.")
    lines.append("")
    lines.append("Constraint columns come from `physics/constraints.py`; C3/C4/C6/S1/S6/S7")
    lines.append("are measured by probing each trained model on synthetic sweeps it never")
    lines.append("saw in training.")
    lines.append("")

    # ---- headline accuracy table -----------------------------------------
    lines.append("## Accuracy by split")
    lines.append("")
    for tag in ["strategy1", "strategy2", "strategy3", "strategy4_pooled_oof"]:
        sub = m[m["split_tag"] == tag].sort_values("arm")
        if sub.empty:
            continue
        lines.append(f"### {SPLIT_LABEL.get(tag, tag)}")
        lines.append("")
        disp = sub.copy()
        disp["arm"] = disp["arm"].map(ARM_LABEL).fillna(disp["arm"].astype(str))
        lines.append(table(disp, ACC_COLS, nd=4))
        lines.append("")

    # ---- R2 pivot --------------------------------------------------------
    lines.append("## R2 across all splits")
    lines.append("")
    piv = m[m["split_tag"].isin(SPLIT_LABEL)].pivot_table(
        index="arm", columns="split_tag", values="R2", observed=True).reindex(ARM_ORDER)
    piv.index = [ARM_LABEL.get(i, i) for i in piv.index]
    lines.append(piv.round(4).to_markdown())
    lines.append("")

    lines.append("## MAPE % across all splits")
    lines.append("")
    piv2 = m[m["split_tag"].isin(SPLIT_LABEL)].pivot_table(
        index="arm", columns="split_tag", values="MAPE_pct", observed=True).reindex(ARM_ORDER)
    piv2.index = [ARM_LABEL.get(i, i) for i in piv2.index]
    lines.append(piv2.round(2).to_markdown())
    lines.append("")

    # ---- physics scorecard ----------------------------------------------
    lines.append("## Physics-consistency scorecard")
    lines.append("")
    lines.append("Measured on strategy 3 (the hardest split). Lower is better for the")
    lines.append("violation fractions; `S1_peak_reduced_pressure` should be near 0.35;")
    lines.append("`S2_pool_K_median` should sit in the Zuber band 0.119-0.157.")
    lines.append("")
    sub = m[m["split_tag"] == "strategy3"].sort_values("arm")
    disp = sub.copy()
    disp["arm"] = disp["arm"].map(ARM_LABEL).fillna(disp["arm"].astype(str))
    lines.append(table(disp, PHYS_COLS, nd=3))
    lines.append("")

    # ---- per-source LOSO -------------------------------------------------
    fold = m[m["split_tag"].str.startswith("strategy4_fold=")].copy()
    if not fold.empty:
        lines.append("## Leave-one-source-out, per fold (R2)")
        lines.append("")
        fold["held_out_source"] = fold["split_tag"].str.split("=", n=1).str[1]
        pf = fold.pivot_table(index="arm", columns="held_out_source", values="R2",
                               observed=True).reindex(ARM_ORDER)
        pf.index = [ARM_LABEL.get(i, i) for i in pf.index]
        lines.append(pf.round(3).to_markdown())
        lines.append("")

    # ---- LOSO accuracy by physical regime --------------------------------
    oof_path = RESULTS / "predictions" / "ablation_loso_oof.csv"
    if oof_path.exists():
        from sklearn.metrics import r2_score
        oof = pd.read_csv(oof_path)
        lines.append("## Leave-one-source-out R2 by physical regime")
        lines.append("")
        lines.append("DNB and dryout are physically distinct crises (foundation doc 1.2);")
        lines.append("a single pooled number hides which mechanism a model actually handles.")
        lines.append("")
        recs = []
        for arm in ARM_ORDER:
            col = f"y_pred_{arm}"
            if col not in oof.columns:
                continue
            rec = {"arm": ARM_LABEL.get(arm, arm)}
            for regime, grp in oof.groupby("chf_regime"):
                ok = np.isfinite(grp[col]) & np.isfinite(grp["y_true"])
                rec[regime] = (round(float(r2_score(grp.loc[ok, "y_true"], grp.loc[ok, col])), 3)
                               if ok.sum() > 2 else np.nan)
            recs.append(rec)
        lines.append(pd.DataFrame(recs).to_markdown(index=False))
        lines.append("")

    (RESULTS / "ablation_report.md").write_text("\n".join(lines))
    print(f"Wrote {RESULTS/'ablation_report.md'}")
    print()
    print(piv.round(4).to_string())


if __name__ == "__main__":
    main()

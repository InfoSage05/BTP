"""Phase 3: turn the raw grid into answers to Q1 and Q2.

Q1  Are there models that work for specific dataset types?
Q2  Is there one model that works well across all of them?

Everything is reported per formulation, because the local form (which includes
outlet quality) is not a prediction -- see Exhibit A. Headline answers use the
reduced form, which is available for every dataset and carries no circularity.
"""

from __future__ import annotations

import os
import numpy as np
import pandas as pd

OUT = os.path.join(os.path.dirname(__file__), "..", "..", "results", "prof_tasks")
pd.set_option("display.width", 200)


# A model scoring R2 = -3.8e9 and one scoring -5 are both "worse than useless".
# Aggregating raw R2 lets a single blown-up fold dominate any mean, so every
# cross-dataset summary uses R2 floored at -1. Per-cell tables keep the raw value.
R2_FLOOR = -1.0


def load():
    g = pd.read_csv(os.path.join(OUT, "grid_results.csv"))
    p = os.path.join(OUT, "physics_results.csv")
    if os.path.exists(p):
        ph = pd.read_csv(p)
        ph = ph.rename(columns={c: f"{c}_mean" for c in
                                ("R2", "within_10pct", "within_20pct", "MAPE_pct",
                                 "RMSE", "safe_side_frac", "n_test", "n_train")
                                if c in ph.columns})
        ph["form"] = np.where(ph["model"].str.startswith(("PHYS_", "ITER_")),
                              "physics_or_closure", "physics_or_closure")
        g = pd.concat([g, ph], ignore_index=True)
    g["R2_floored"] = g["R2_mean"].clip(lower=R2_FLOOR)
    return g


def fmt(df, cols, nd=3):
    return df[cols].to_string(index=False,
                              float_format=lambda v: f"{v:.{nd}f}")


def q1(df, form, split):
    """Best model per dataset."""
    d = df[(df.form == form) & (df.split == split)].copy()
    if d.empty:
        return None
    best = (d.sort_values("R2_mean", ascending=False)
              .groupby("dataset", as_index=False).head(1)
              .sort_values("R2_mean", ascending=False))
    return best


# The LUT's only inputs are (P, G, X). Its "reduced" form is therefore (P, G):
# predicting CHF with no quality information at all. Every model scores ~0 there,
# so it becomes every model's nominal worst case and tells us nothing about the
# models. It is an ill-posed problem, not a failure, and is excluded from the
# consistency ranking (reported separately instead).
DEGENERATE = {("D9_LUT2006", "reduced")}


def q2(df, form, split, min_datasets=None, drop_degenerate=True):
    """Models ranked by consistency across datasets, not by any single win."""
    d = df[(df.form == form) & (df.split == split)].copy()
    if drop_degenerate:
        d = d[~d.apply(lambda r: (r["dataset"], r["form"]) in DEGENERATE, axis=1)]
    if d.empty:
        return None
    d["rank"] = d.groupby("dataset")["R2_mean"].rank(ascending=False)
    n_ds = d.dataset.nunique()
    min_datasets = min_datasets or n_ds
    agg = (d.groupby("model")
             .agg(datasets=("dataset", "nunique"),
                  mean_rank=("rank", "mean"),
                  median_R2=("R2_mean", "median"),
                  min_R2=("R2_mean", "min"),
                  mean_R2_floored=("R2_floored", "mean"),
                  mean_within10=("within_10pct_mean", "mean"),
                  worst_dataset=("R2_mean", "idxmin"))
             .reset_index())
    agg["worst_dataset"] = d.loc[agg["worst_dataset"], "dataset"].to_numpy()
    agg = agg[agg.datasets >= min_datasets].sort_values("mean_rank")
    return agg


def main():
    df = load()
    lines = []

    def say(s=""):
        print(s)
        lines.append(s)

    say("=" * 100)
    say("FLOW-BOILING MODEL COMPARISON  --  results by formulation and split")
    say("=" * 100)
    say(f"\nresult rows: {len(df)}   datasets: {df.dataset.nunique()}   "
        f"models: {df.model.nunique()}")
    say(f"datasets present: {', '.join(sorted(df.dataset.unique()))}")

    # ---- formulation gap -------------------------------------------------
    say("\n\n" + "-" * 100)
    say("THE FORMULATION GAP  (mean of best-model R2 over datasets)")
    say("-" * 100)
    rows = []
    for split in ("80/20", "70/30", "LOSO"):
        for form in ("local", "inlet", "reduced"):
            d = df[(df.form == form) & (df.split == split)]
            if d.empty:
                continue
            b = d.sort_values("R2_mean", ascending=False).groupby("dataset").head(1)
            rows.append(dict(split=split, form=form, datasets=len(b),
                             mean_best_R2=b.R2_mean.mean(),
                             mean_best_within10=b.within_10pct_mean.mean()))
    if rows:
        say(fmt(pd.DataFrame(rows), ["split", "form", "datasets",
                                     "mean_best_R2", "mean_best_within10"]))

    # ---- Q1 --------------------------------------------------------------
    for split in ("80/20", "70/30", "LOSO"):
        for form in ("reduced", "local", "inlet"):
            b = q1(df, form, split)
            if b is None or b.empty:
                continue
            say("\n\n" + "-" * 100)
            say(f"Q1  BEST MODEL PER DATASET   form={form}   split={split}")
            say("-" * 100)
            say(fmt(b, ["dataset", "n", "fluid", "geometry", "model", "R2_mean",
                        "R2_sd", "within_10pct_mean", "within_20pct_mean",
                        "MAPE_pct_mean", "n_test_mean"]))

    # ---- Q2 --------------------------------------------------------------
    for split in ("80/20", "70/30", "LOSO"):
        for form in ("reduced", "local"):
            a = q2(df, form, split)
            if a is None or a.empty:
                continue
            say("\n\n" + "-" * 100)
            say(f"Q2  MOST CONSISTENT MODEL ACROSS DATASETS   form={form}   split={split}")
            say("   (ranked by mean rank; min_R2 is the worst dataset it faces)")
            if form == "reduced":
                say("   D9_LUT2006 excluded here: its reduced form is (P, G) with no "
                    "quality, an ill-posed problem where every model scores ~0.")
            say("-" * 100)
            say(fmt(a, ["model", "datasets", "mean_rank", "median_R2",
                        "mean_R2_floored", "min_R2", "mean_within10",
                        "worst_dataset"]))

    # ---- random vs grouped ----------------------------------------------
    say("\n\n" + "-" * 100)
    say("OPTIMISM: random 80/20 vs leave-one-source-out, same model and dataset")
    say("-" * 100)
    a = df[(df.split == "80/20")][["dataset", "form", "model", "R2_floored",
                                   "within_10pct_mean"]]
    b = df[(df.split == "LOSO")][["dataset", "form", "model", "R2_floored",
                                  "within_10pct_mean"]]
    m = a.merge(b, on=["dataset", "form", "model"], suffixes=("_rand", "_loso"))
    if not m.empty:
        m["optimism_gap"] = m.R2_floored_rand - m.R2_floored_loso
        top = (m.sort_values("optimism_gap", ascending=False)
                 .groupby("dataset").head(1).sort_values("optimism_gap", ascending=False))
        say(fmt(top, ["dataset", "form", "model", "R2_floored_rand",
                      "R2_floored_loso", "optimism_gap"]))
        say(f"\n(R2 floored at {R2_FLOOR} before differencing)")
        say(f"median optimism gap over all model x dataset pairs: "
            f"{m.optimism_gap.median():.3f}")
        say(f"mean   optimism gap over all model x dataset pairs: "
            f"{m.optimism_gap.mean():.3f}")
        m.to_csv(os.path.join(OUT, "optimism_gap.csv"), index=False)

    with open(os.path.join(OUT, "ANSWERS_Q1_Q2.txt"), "w") as f:
        f.write("\n".join(lines))
    say(f"\n\nwritten -> results/prof_tasks/ANSWERS_Q1_Q2.txt")


if __name__ == "__main__":
    main()

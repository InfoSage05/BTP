"""
build_splits.py
----------------
Builds the 4 train/test split strategies discussed for the master CHF
dataset, as index files keyed by row_id (never duplicating the actual data).

Run (after merge_datasets.py has produced data/master_chf_dataset.csv):
    python unified_chf_pipeline/scripts/build_splits.py

Writes, into unified_chf_pipeline/splits/:
    strategy1_random_stratified.csv   -- row_id, split (train/test)
    strategy2_condition_wise.csv      -- row_id, split (train/test)
    strategy3_surface_wise.csv        -- row_id, split (train/test)
    strategy4_leave_one_source_out.csv-- row_id, fold  (fold = held-out source)
    splits_report.md                  -- train/test counts per strategy per source
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = Path(__file__).resolve().parents[1] / "data"
SPLITS_DIR = Path(__file__).resolve().parents[1] / "splits"
SPLITS_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42

# Strategy 3: which whole sources are held out as a "never seen before" test
# set. Chosen to be structurally distinct from the tube-dominated bulk of the
# data (pool boiling / helical coil, not just another flow-boiling tube set)
# so this strategy actually tests cross-surface generalization.
STRATEGY3_HELDOUT_SOURCES = ["pinfin_chf_water_fc72", "helical_coil_r123"]

# Strategy 2: per-source condition-wise holdout. Sources with fewer than
# MIN_ROWS_FOR_CONDITION_SPLIT rows, or without pressure data at all, are
# left entirely in train (too little data to define a meaningful high-
# pressure extrapolation region).
MIN_ROWS_FOR_CONDITION_SPLIT = 30
CONDITION_HOLDOUT_FRAC = 0.2


def load_master() -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / "master_chf_dataset.csv", low_memory=False)


def strategy1_random_stratified(master: pd.DataFrame, test_frac: float = 0.2) -> pd.DataFrame:
    """Random 80/20 split, done independently within each source_dataset so
    every source keeps its proportional share of the test set."""
    rng = np.random.default_rng(SEED)
    rows = []
    for source, grp in master.groupby("source_dataset"):
        idx = grp["row_id"].to_numpy().copy()
        rng.shuffle(idx)
        n_test = int(round(len(idx) * test_frac))
        test_ids = set(idx[:n_test])
        for rid in idx:
            rows.append({"row_id": rid, "split": "test" if rid in test_ids else "train"})
    return pd.DataFrame(rows)


def strategy2_condition_wise(master: pd.DataFrame) -> pd.DataFrame:
    """Per-source: hold out the top CONDITION_HOLDOUT_FRAC of pressure_kPa as
    test (an extrapolation test), rest as train. Sources too small or without
    pressure data go entirely to train."""
    rows = []
    for source, grp in master.groupby("source_dataset"):
        has_pressure = grp["pressure_kPa"].notna()
        if has_pressure.sum() < MIN_ROWS_FOR_CONDITION_SPLIT:
            for rid in grp["row_id"]:
                rows.append({"row_id": rid, "split": "train"})
            continue
        threshold = grp.loc[has_pressure, "pressure_kPa"].quantile(1 - CONDITION_HOLDOUT_FRAC)
        for rid, p in zip(grp["row_id"], grp["pressure_kPa"]):
            is_test = pd.notna(p) and p >= threshold
            rows.append({"row_id": rid, "split": "test" if is_test else "train"})
    return pd.DataFrame(rows)


def strategy3_surface_wise(master: pd.DataFrame) -> pd.DataFrame:
    """Hold out whole sources (STRATEGY3_HELDOUT_SOURCES) as test."""
    split = np.where(master["source_dataset"].isin(STRATEGY3_HELDOUT_SOURCES), "test", "train")
    return pd.DataFrame({"row_id": master["row_id"], "split": split})


def strategy4_leave_one_source_out(master: pd.DataFrame) -> pd.DataFrame:
    """Not a single train/test split -- one fold per source_dataset. For fold
    F, all rows with source_dataset == F are the test set and everything else
    is train. This file just records the fold assignment (== source_dataset);
    the train/test membership for a given fold is derived at use-time via
    `fold == F` (test) vs `fold != F` (train), e.g. with sklearn's
    LeaveOneGroupOut(groups=master['source_dataset'])."""
    return master[["row_id", "source_dataset"]].rename(columns={"source_dataset": "fold"})


def summarize(name: str, master: pd.DataFrame, split_df: pd.DataFrame, split_col: str) -> str:
    merged = master[["row_id", "source_dataset"]].merge(split_df, on="row_id")
    table = merged.pivot_table(index="source_dataset", columns=split_col,
                                values="row_id", aggfunc="count", fill_value=0)
    return f"## {name}\n\n```\n{table.to_string()}\n```\n"


if __name__ == "__main__":
    master = load_master()

    s1 = strategy1_random_stratified(master)
    s2 = strategy2_condition_wise(master)
    s3 = strategy3_surface_wise(master)
    s4 = strategy4_leave_one_source_out(master)

    s1.to_csv(SPLITS_DIR / "strategy1_random_stratified.csv", index=False)
    s2.to_csv(SPLITS_DIR / "strategy2_condition_wise.csv", index=False)
    s3.to_csv(SPLITS_DIR / "strategy3_surface_wise.csv", index=False)
    s4.to_csv(SPLITS_DIR / "strategy4_leave_one_source_out.csv", index=False)

    report = "# Split Strategy Report\n\n"
    report += summarize("Strategy 1: Random split (stratified by source)", master, s1, "split")
    report += summarize("Strategy 2: Condition-wise split (per-source top-pressure holdout)", master, s2, "split")
    report += summarize(f"Strategy 3: Surface-wise split (held-out sources: {STRATEGY3_HELDOUT_SOURCES})",
                         master, s3, "split")
    report += summarize("Strategy 4: Leave-one-source-out (fold == held-out source; "
                         "counts below are rows per fold, i.e. rows per source)",
                         master, s4, "fold")

    (SPLITS_DIR / "splits_report.md").write_text(report)

    print(f"Wrote strategy1_random_stratified.csv ({len(s1)} rows)")
    print(f"Wrote strategy2_condition_wise.csv ({len(s2)} rows)")
    print(f"Wrote strategy3_surface_wise.csv ({len(s3)} rows)")
    print(f"Wrote strategy4_leave_one_source_out.csv ({len(s4)} rows)")
    print(f"Wrote splits_report.md")
    print("\n" + report)

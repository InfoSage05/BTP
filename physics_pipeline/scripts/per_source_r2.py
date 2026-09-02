"""
per_source_r2.py
----------------
R2 for every model on every source dataset, for all four split strategies.

Covers BOTH model families so they can be compared on identical rows:
  * the original bake-off (ANN, RandomForest, HistGB, PINN, GPR,
    Proposed_Hierarchy) -- read from the saved predictions in
    results/predictions/predictions_strategy*.csv
  * the physics ablation arms (PHYS_katto ... A5_ANN) -- strategies 1-3 are
    refit here because run_ablation.py only persisted the LOSO out-of-fold
    predictions; strategy 4 is read from results/predictions/ablation_loso_oof.csv

A blank cell means that source had no test rows in that split (e.g. the
surface-wise split tests only helical_coil_r123 and pinfin_chf_water_fc72;
the condition-wise split sends mentor_master and pinfin entirely to train).

Run:  python physics_pipeline/scripts/per_source_r2.py
"""
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

import features_v2 as F
from models_v2 import PhysicsCorrectedModel
from physics import repair as phys_repair
from run_ablation import ARMS, PHYSICS_ONLY_ARMS

warnings.filterwarnings("ignore")

import paths

paths.check_inputs()
RESULTS = paths.ensure_output_dirs()
DATA, SPLITS = paths.DATA_DIR, paths.SPLITS_DIR
PRED = paths.PREDICTIONS_DIR
SHARED_PRED = paths.SHARED_PREDICTIONS   # original bake-off predictions (read-only)

STRATEGY_FILES = paths.STRATEGY_FILES

MIN_ROWS = 3  # R2 on fewer points than this is not meaningful


def r2_by_source(y_true, y_pred, source, min_rows=MIN_ROWS):
    out = {}
    df = pd.DataFrame({"y": y_true, "p": y_pred, "s": source})
    for src, g in df.groupby("s"):
        ok = np.isfinite(g["y"]) & np.isfinite(g["p"])
        out[src] = float(r2_score(g.loc[ok, "y"], g.loc[ok, "p"])) if ok.sum() >= min_rows else np.nan
    return out


def ablation_predictions_1_to_3(master):
    """Refit each ablation arm on strategies 1-3 and return per-row predictions."""
    prepared = {}
    for mode in {cfg[0] for cfg in ARMS.values()} | set(PHYSICS_ONLY_ARMS.values()):
        prepared[mode] = F.prepare(master, baseline_mode=mode).set_index("row_id", drop=False)

    preds = {}
    for tag, fname in STRATEGY_FILES.items():
        s = pd.read_csv(SPLITS / fname)
        tr = pd.Index(s.loc[s["split"] == "train", "row_id"])
        te = pd.Index(s.loc[s["split"] == "test", "row_id"])
        print(f"  {tag}: train={len(tr)} test={len(te)}")
        frame = {}
        for name, mode in PHYSICS_ONLY_ARMS.items():
            frame[name] = prepared[mode].loc[te, "physics_baseline_kW_m2"].values
        for arm, (mode, space, base, bound, mono, trust) in ARMS.items():
            prep = prepared[mode]
            X = F.build_matrix(prep, space)
            y, b = prep["CHF_kW_m2"], prep["physics_baseline_kW_m2"]
            w = F.sample_weights_by_source(prep.loc[tr, "source_dataset"])
            m = PhysicsCorrectedModel(base=base, bound=bound, monotone=mono,
                                      trust_decay=trust, space=space)
            m.fit(X.loc[tr], y.loc[tr], b.loc[tr], sample_weight=w)
            frame[arm] = m.predict(X.loc[te], b.loc[te])
        out = pd.DataFrame(frame, index=te)
        out.insert(0, "y_true", prepared["katto"].loc[te, "CHF_kW_m2"].values)
        out.insert(1, "source_dataset", prepared["katto"].loc[te, "source_dataset"].values)
        out.to_csv(PRED / f"ablation_{tag}_predictions.csv", index_label="row_id")
        preds[tag] = out
    return preds


def main():
    master = phys_repair.repair(pd.read_csv(paths.MASTER_CSV, low_memory=False))
    src_by_id = master.set_index("row_id")["source_dataset"]

    print("Refitting ablation arms for strategies 1-3 ...")
    abl = ablation_predictions_1_to_3(master)

    tables = {}
    for tag in list(STRATEGY_FILES) + ["strategy4"]:
        rows = {}

        # --- original bake-off models ---
        old_path = SHARED_PRED / (f"predictions_{tag}.csv" if tag != "strategy4"
                                  else "predictions_strategy4_oof.csv")
        if old_path.exists():
            old = pd.read_csv(old_path)
            src = (old["source_dataset"] if "source_dataset" in old.columns
                   else old["row_id"].map(src_by_id))
            for c in [c for c in old.columns if c.startswith("y_pred_")]:
                rows[c.replace("y_pred_", "")] = r2_by_source(old["y_true"], old[c], src)

        # --- ablation arms ---
        if tag == "strategy4":
            new = pd.read_csv(PRED / "ablation_loso_oof.csv")
            src = new["source_dataset"]
            for c in [c for c in new.columns if c.startswith("y_pred_")]:
                rows[c.replace("y_pred_", "")] = r2_by_source(new["y_true"], new[c], src)
        else:
            new = abl[tag]
            for c in [c for c in new.columns if c not in ("y_true", "source_dataset")]:
                rows[c] = r2_by_source(new["y_true"], new[c], new["source_dataset"])

        tables[tag] = pd.DataFrame(rows).T.sort_index()

    lines = ["# R2 by source dataset, every model, every split", "",
             "Blank = that source had no test rows in that split.",
             "`strategy4` is leave-one-source-out: each column is scored by a model",
             "that never saw that source in training.", ""]
    for tag, t in tables.items():
        lines.append(f"## {tag}")
        lines.append("")
        lines.append(t.round(3).to_markdown())
        lines.append("")
    (RESULTS / "per_source_r2.md").write_text("\n".join(lines))

    for tag, t in tables.items():
        print(f"\n===== {tag} =====")
        print(t.round(3).to_string())
    print(f"\nWrote {RESULTS/'per_source_r2.md'}")


if __name__ == "__main__":
    main()

"""
paths.py
--------
Single place where this pipeline decides what it reads and what it writes.

READS the merged dataset and the four split definitions from
`unified_chf_pipeline/`. They are deliberately NOT copied here: duplicating a
28,470-row table and its splits is how two copies quietly drift apart, and
every claim in this folder depends on being scored on exactly the same rows as
the original bake-off. One source of truth, referenced.

WRITES everything -- metrics, reports, per-row predictions -- into
`physics_pipeline/results/`. Nothing in this folder writes outside it.

Override either input location with an environment variable if the shared
pipeline ever moves:

    CHF_DATA_DIR=/path/to/data  CHF_SPLITS_DIR=/path/to/splits  python run_ablation.py
"""
import os
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent
PIPELINE_DIR = SCRIPTS_DIR.parent
REPO_ROOT = PIPELINE_DIR.parent

#: The shared pipeline that owns the merged dataset and the splits.
SHARED_PIPELINE = REPO_ROOT / "unified_chf_pipeline"
SHARED_SCRIPTS = SHARED_PIPELINE / "scripts"

DATA_DIR = Path(os.environ.get("CHF_DATA_DIR", SHARED_PIPELINE / "data"))
SPLITS_DIR = Path(os.environ.get("CHF_SPLITS_DIR", SHARED_PIPELINE / "splits"))

MASTER_CSV = DATA_DIR / "master_chf_dataset.csv"

#: Predictions from the ORIGINAL bake-off (ANN / RF / HistGB / GPR / PINN /
#: Proposed_Hierarchy). Read-only, used by per_source_r2.py so both model
#: families can be compared on identical rows.
SHARED_PREDICTIONS = SHARED_PIPELINE / "results" / "predictions"

# --- outputs, all inside this folder ---------------------------------------
RESULTS_DIR = PIPELINE_DIR / "results"
PREDICTIONS_DIR = RESULTS_DIR / "predictions"

STRATEGY_FILES = {
    "strategy1": "strategy1_random_stratified.csv",
    "strategy2": "strategy2_condition_wise.csv",
    "strategy3": "strategy3_surface_wise.csv",
}
LOSO_FILE = "strategy4_leave_one_source_out.csv"


def ensure_output_dirs():
    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    return RESULTS_DIR


def check_inputs():
    """Fail early and clearly if the shared dataset is not where we expect."""
    missing = [p for p in (MASTER_CSV, SPLITS_DIR) if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "physics_pipeline reads its data from the shared unified_chf_pipeline.\n"
            "Missing: " + ", ".join(str(p) for p in missing) +
            "\nSet CHF_DATA_DIR / CHF_SPLITS_DIR if that folder has moved.")

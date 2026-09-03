"""
metrics_utils.py
-----------------
Shared metric computation, matching what the paper outline asks for in its
"Performance metrics comparison of all ML models" section: R2, RMSE, MAE,
MAPE, training time.
"""
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def compute_metrics(y_true, y_pred, train_seconds: float | None = None) -> dict:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true, y_pred = y_true[mask], y_pred[mask]

    if len(y_true) == 0:
        return {"n_test": 0, "R2": np.nan, "RMSE": np.nan, "MAE": np.nan,
                "MAPE_pct": np.nan, "train_seconds": train_seconds}

    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    # MAPE: guard against division by ~0 CHF values (none expected physically,
    # but keep it safe rather than emitting inf and poisoning averages).
    nonzero = np.abs(y_true) > 1e-6
    if nonzero.sum() > 0:
        mape = float(np.mean(np.abs((y_true[nonzero] - y_pred[nonzero]) / y_true[nonzero])) * 100)
    else:
        mape = np.nan
    r2 = float(r2_score(y_true, y_pred)) if len(y_true) > 1 else np.nan

    return {
        "n_test": int(len(y_true)),
        "R2": r2,
        "RMSE": rmse,
        "MAE": mae,
        "MAPE_pct": mape,
        "train_seconds": train_seconds,
    }

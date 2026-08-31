"""
generate_synthetic_chf_gp.py
-----------------------------
Synthetic CHF data augmentation via Gaussian Process posterior sampling.

Why a GP and not a CVAE/diffusion model: two recent papers (arXiv:2409.05790,
CVAE augmentation of this exact Groeneveld LUT; arXiv:2511.16207, physics-
consistent conditional diffusion for CHF) both use deep generative models for
this purpose, but their own closest-comparator group (Furlong, Zhao, Salko &
Wu -- the ORNL/NCSU CHF-UQ line of work, see docs/references) uses a deep
Gaussian process as one of three interchangeable UQ/generative techniques
alongside DNN ensembles and Bayesian NNs. A GP is generative in exactly the
same sense a CVAE is (sample the posterior predictive at a query point), needs
no new heavy dependency (no torch/GPU), fits in seconds on this dataset size,
and comes with calibrated uncertainty for free -- each synthetic point carries
its own posterior std, so downstream models can be told how much to trust it.

Method:
  1. Fit a GP (Matern-5/2, ARD length scales -- same kernel family as the
     project's existing GPR baseline in CHF_ML_Modeling.ipynb) on log(CHF) vs.
     standardized (P, G, X), using only training-fold data (never Split C).
  2. Sample new query points (P, G, X) *off* the discrete 24x21x23 grid --
     uniformly at random within the training envelope's convex-ish bounding
     box -- so the augmented data densifies the interior instead of just
     resampling existing grid nodes. This directly targets the "tree models
     memorize discrete grid cells" failure mode documented in SENIOR_REVIEW.md.
  3. Draw one synthetic log(CHF) sample per query point from the GP posterior
     (not just the posterior mean -- that would just be smoothing, not
     generation) and exponentiate back to CHF.
  4. Physical sanity filter before accepting a synthetic point:
       - CHF must be positive and finite,
       - CHF must be within a tolerance band of the deterministic trilinear
         grid-interpolation baseline at that (P, G, X) (a synthetic point that
         disagrees wildly with the table itself is more likely a GP
         extrapolation artifact than a physically plausible new state).
     This mirrors -- in a much simpler form -- the "physical consistency"
     check the diffusion-model paper reports evaluating against.
  5. Everything is seeded and logged (config + acceptance stats) next to the
     output CSV for reproducibility, matching results/seed_and_version_log.json.

This is training-data *augmentation*, not new ground truth: every synthetic
row must be labeled as such downstream and never mixed into Split C's test
set. See data/synthetic/README.md.
"""
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.interpolate import RegularGridInterpolator
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, ConstantKernel, WhiteKernel
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = REPO_ROOT / "data" / "chf_long_clean.csv"
OUT_DIR = REPO_ROOT / "data" / "synthetic"
OUT_CSV = OUT_DIR / "chf_synthetic_gp_augmented.csv"
OUT_LOG = OUT_DIR / "generation_log.json"

SEED = 42
GP_FIT_SUBSAMPLE = 2000  # matches the project's existing GPR baseline convention
N_QUERY_POINTS = 8000  # candidates sampled off-grid; fewer will survive the filter
CONSISTENCY_TOL_LOG = 0.35  # accept if |log(GP_sample) - log(GridInterp)| <= this
TRAIN_PRESSURE_MAX_KPA = 16000.0  # never generate points in the Split C extrapolation
# region -- augmentation must not manufacture "extra" high-pressure evidence.


def load_training_frame():
    df = pd.read_csv(DATA_PATH)
    df = df[df["X"] != 1.0].reset_index(drop=True)
    df = df[df["P"] <= TRAIN_PRESSURE_MAX_KPA].reset_index(drop=True)
    return df


def fit_gp(df, rng):
    idx = rng.choice(len(df), size=min(GP_FIT_SUBSAMPLE, len(df)), replace=False)
    sub = df.iloc[idx]
    X_raw = sub[["P", "G", "X"]].to_numpy(dtype=float)
    y_log = np.log(sub["CHF"].to_numpy(dtype=float))

    scaler = StandardScaler().fit(X_raw)
    X_scaled = scaler.transform(X_raw)

    kernel = ConstantKernel(1.0, (1e-2, 1e3)) * Matern(
        length_scale=[1.0, 1.0, 1.0], length_scale_bounds=(1e-2, 1e2), nu=2.5
    ) + WhiteKernel(noise_level=1e-3, noise_level_bounds=(1e-6, 1e0))

    gp = GaussianProcessRegressor(
        kernel=kernel, normalize_y=True, random_state=SEED, n_restarts_optimizer=2
    )
    gp.fit(X_scaled, y_log)
    return gp, scaler, len(sub)


def build_grid_interpolator(df_full):
    piv_shape_check = df_full.groupby(["P", "G"]).size()
    pressures = np.sort(df_full["P"].unique())
    fluxes = np.sort(df_full["G"].unique())
    qualities = np.sort(df_full["X"].unique())
    grid = np.full((len(pressures), len(fluxes), len(qualities)), np.nan)
    p_idx = {v: i for i, v in enumerate(pressures)}
    g_idx = {v: i for i, v in enumerate(fluxes)}
    x_idx = {v: i for i, v in enumerate(qualities)}
    for _, row in df_full.iterrows():
        grid[p_idx[row["P"]], g_idx[row["G"]], x_idx[row["X"]]] = row["CHF"]
    # Fill remaining NaN (the X==1.0 placeholder cells excluded upstream) with
    # nearest-neighbor so the interpolator has a complete rectilinear grid.
    if np.isnan(grid).any():
        from scipy.ndimage import distance_transform_edt

        nan_mask = np.isnan(grid)
        idx_map = distance_transform_edt(nan_mask, return_distances=False, return_indices=True)
        grid = grid[tuple(idx_map)]
    return RegularGridInterpolator(
        (pressures, fluxes, qualities), grid, bounds_error=False, fill_value=None
    ), pressures, fluxes, qualities


def sample_query_points(df, rng, n):
    p_lo, p_hi = df["P"].min(), df["P"].max()
    g_lo, g_hi = df["G"].min(), df["G"].max()
    x_lo, x_hi = df["X"].min(), df["X"].max()
    P = rng.uniform(p_lo, p_hi, size=n)
    G = rng.uniform(g_lo, g_hi, size=n)
    X = rng.uniform(x_lo, x_hi, size=n)
    return np.column_stack([P, G, X])


def main():
    t0 = time.time()
    rng = np.random.default_rng(SEED)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    df_full = pd.read_csv(DATA_PATH)
    df_full = df_full[df_full["X"] != 1.0].reset_index(drop=True)
    df_train = load_training_frame()

    gp, scaler, n_fit = fit_gp(df_train, rng)
    grid_interp, pressures, fluxes, qualities = build_grid_interpolator(df_full)

    queries = sample_query_points(df_train, rng, N_QUERY_POINTS)
    queries_scaled = scaler.transform(queries)

    mean, std = gp.predict(queries_scaled, return_std=True)
    # One posterior *sample* per point (not the mean) -- this is what makes it
    # generation rather than smoothing.
    log_chf_sample = rng.normal(loc=mean, scale=std)
    chf_sample = np.exp(log_chf_sample)

    grid_ref = grid_interp(queries)
    grid_ref = np.clip(grid_ref, 1e-6, None)
    log_diff = np.abs(log_chf_sample - np.log(grid_ref))

    finite_mask = np.isfinite(chf_sample) & (chf_sample > 0)
    consistency_mask = log_diff <= CONSISTENCY_TOL_LOG
    accept_mask = finite_mask & consistency_mask

    accepted = pd.DataFrame(
        {
            "P": queries[accept_mask, 0],
            "G": queries[accept_mask, 1],
            "X": queries[accept_mask, 2],
            "CHF": chf_sample[accept_mask],
            "gp_posterior_std_log": std[accept_mask],
            "gridinterp_reference_kWm2": grid_ref[accept_mask],
            "source": "synthetic_gp_augmentation",
        }
    )
    accepted.to_csv(OUT_CSV, index=False)

    log = {
        "seed": SEED,
        "gp_fit_points": n_fit,
        "n_query_points": N_QUERY_POINTS,
        "n_accepted": int(accept_mask.sum()),
        "acceptance_rate": float(accept_mask.mean()),
        "consistency_tolerance_log_scale": CONSISTENCY_TOL_LOG,
        "train_pressure_max_kpa": TRAIN_PRESSURE_MAX_KPA,
        "kernel": str(gp.kernel_),
        "runtime_seconds": round(time.time() - t0, 2),
        "sklearn_version": __import__("sklearn").__version__,
    }
    with open(OUT_LOG, "w") as f:
        json.dump(log, f, indent=2)

    print(f"Generated {log['n_accepted']} synthetic points "
          f"({log['acceptance_rate']:.1%} of {N_QUERY_POINTS} candidates accepted)")
    print(f"Wrote {OUT_CSV}")
    print(f"Wrote {OUT_LOG}")


if __name__ == "__main__":
    main()

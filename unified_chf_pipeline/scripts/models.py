"""
models.py
----------
Model definitions for the bake-off:
  - ANN            (MLPRegressor, needs imputation -- can't take NaN)
  - Random Forest   (native NaN handling, sklearn >= 1.4)
  - HistGB          (HistGradientBoostingRegressor, native NaN handling --
                      this is also the Tier-0 algorithm in the proposed
                      hierarchy)
  - GPR             (GaussianProcessRegressor, needs imputation; doesn't
                      scale past a few thousand rows, so flat-baseline GPR is
                      fit on a capped random subsample of the training data)

Every model trains on log(CHF / physics_baseline) rather than raw CHF, where
physics_baseline is a per-row closed-form physical CHF scale computed by
features.compute_physics_baseline_kw_m2 (Zuber correlation for pool boiling,
G*h_fg for flow boiling -- see that function's docstring for why). This
replaced a simpler log1p(CHF) transform: log1p alone helped in-distribution
accuracy (CHF spans ~770x, 37.7-28,800 kW/m^2) but did nothing for cross-fluid
extrapolation, since the model still had to learn each fluid's absolute CHF
scale from scratch. Dividing by a physics-computed scale *before* log-space
training removes that scale mismatch up front -- e.g. water's h_fg is ~9x
R123's, which is almost exactly the magnitude of the water-trained model's
CHF overprediction on R123 test rows seen before this change.

RandomForest and HistGB also accept `sample_weight` at fit time (see
features.sample_weights_by_source) -- MLPRegressor and GaussianProcessRegressor
don't support sample_weight at all, so weighting is skipped for those two
(handled transparently: fit falls back to unweighted .fit() if the model
rejects the sample_weight kwarg).

Plus the proposed Tier-0 + Tier-1 hierarchy:
  - Tier 0: HistGradientBoostingRegressor (physics-normalized log target,
            sample-weighted) on the core feature set (features.py), trained
            on ALL training rows regardless of geometry family.
  - Tier 1: one small GaussianProcessRegressor per geometry family that has
            real optional surface/geometry columns, trained to predict the
            *residual* (true CHF - Tier-0 prediction, real kW/m^2 units) from
            that family's own optional features. Families with no Tier-1
            training data in a given split (e.g. held out entirely by the
            surface-wise split) simply fall back to the Tier-0 prediction
            alone.
"""
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
from sklearn.impute import SimpleImputer
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

RANDOM_STATE = 42
GPR_TRAIN_CAP = 800  # GaussianProcessRegressor fit cost grows as O(n^3)

# Physically-plausible CHF bounds (kW/m^2). Observed data spans 37.7-28,800;
# this clip is deliberately just above that -- it exists only to catch
# extrapolation blow-ups, not to constrain normal predictions. It matters a
# lot in log-space training: MLPRegressor (and to a lesser extent GPR) can
# produce an unbounded raw log-space output when extrapolating far outside
# the training distribution (unlike tree models, which can't predict outside
# the range of their training leaves), and exp() turns even a moderately
# wrong log-space value into an astronomically wrong real-space one. Without
# this clip, a single such blow-up can turn a fold's R2 into something like
# -1,600,000 -- not a meaningful signal of "how wrong", just numerical
# instability drowning out the real comparison.
CHF_CLIP_MIN = 0.0
CHF_CLIP_MAX = 30_000.0  # observed data max is 28,800 kW/m^2


def _clip_chf(y_pred: np.ndarray) -> np.ndarray:
    return np.clip(y_pred, CHF_CLIP_MIN, CHF_CLIP_MAX)


def _to_log_ratio(y: pd.Series, baseline: pd.Series) -> np.ndarray:
    return np.log(np.asarray(y, dtype=float) / np.asarray(baseline, dtype=float))


def _from_log_ratio(log_ratio_pred: np.ndarray, baseline: pd.Series) -> np.ndarray:
    return _clip_chf(np.exp(log_ratio_pred) * np.asarray(baseline, dtype=float))


# Family -> the optional columns that family actually reports, used as
# Tier-1 correction-model inputs. Families not listed here have no
# meaningful optional surface data (e.g. the two KAERI tube sets only carry
# a mesh-quality metadata flag, not a physical surface property) and always
# fall back to Tier-0 alone.
TIER1_FAMILY_FEATURES = {
    "pin_fin_pool_boiling": [
        "fin_width_um", "fin_height_um", "fin_spacing_um", "coverage",
        "porosity", "roughness_factor",
    ],
    "helical_coil": ["rho_l_over_rho_g"],
    "flat_heater_pool_boiling": [
        "angle_deg", "surface_tension_N_m", "rho_l_kg_m3", "cp_l_J_kgK",
        "kl_W_mK", "mu_l_Pa_s", "alpha_m2_s", "ja", "r_bubble_m",
    ],
}


def build_ann():
    return make_pipeline(
        SimpleImputer(strategy="median", add_indicator=True),
        StandardScaler(),
        MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=500,
                     early_stopping=True, random_state=RANDOM_STATE),
    )


def build_random_forest():
    return RandomForestRegressor(n_estimators=500, max_depth=None,
                                  min_samples_leaf=2, n_jobs=-1, random_state=RANDOM_STATE)


def build_hist_gb():
    return HistGradientBoostingRegressor(max_iter=500, learning_rate=0.05,
                                          max_leaf_nodes=63, random_state=RANDOM_STATE)


def _gpr_pipeline():
    kernel = ConstantKernel(1.0) * RBF(length_scale=1.0) + WhiteKernel(noise_level=1.0)
    return make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        GaussianProcessRegressor(kernel=kernel, normalize_y=True,
                                  random_state=RANDOM_STATE),
    )


def fit_predict_gpr(X_train: pd.DataFrame, y_train: pd.Series, baseline_train: pd.Series,
                     X_test: pd.DataFrame, baseline_test: pd.Series, cap: int = GPR_TRAIN_CAP):
    """Fits GPR on at most `cap` training rows (random subsample if larger),
    predicts on X_test. Returns (y_pred, train_seconds, n_train_used)."""
    n = len(X_train)
    if n > cap:
        rng = np.random.default_rng(RANDOM_STATE)
        idx = rng.choice(n, size=cap, replace=False)
        X_fit = X_train.iloc[idx]
        target_fit = _to_log_ratio(y_train.iloc[idx], baseline_train.iloc[idx])
    else:
        X_fit = X_train
        target_fit = _to_log_ratio(y_train, baseline_train)

    pipe = _gpr_pipeline()
    t0 = time.time()
    pipe.fit(X_fit, target_fit)
    train_seconds = time.time() - t0
    y_pred = _from_log_ratio(pipe.predict(X_test), baseline_test)
    return y_pred, train_seconds, len(X_fit)


def fit_predict_sklearn_model(builder, X_train, y_train, baseline_train, X_test, baseline_test,
                               sample_weight=None):
    """Generic fit/predict for RF, HistGB, ANN. Returns (y_pred, train_seconds).
    Tries to pass sample_weight through; silently falls back to an unweighted
    fit if the underlying estimator doesn't support it (ANN)."""
    model = builder()
    target_train = _to_log_ratio(y_train, baseline_train)
    t0 = time.time()
    if sample_weight is not None:
        try:
            model.fit(X_train, target_train, sample_weight=sample_weight)
        except (TypeError, ValueError):
            model.fit(X_train, target_train)
    else:
        model.fit(X_train, target_train)
    train_seconds = time.time() - t0
    y_pred = _from_log_ratio(model.predict(X_test), baseline_test)
    return y_pred, train_seconds


# --------------------------------------------------------------------------
# Proposed hierarchy: Tier 0 (HistGB on core features) + Tier 1 (per-family
# GPR residual correction).
# --------------------------------------------------------------------------

def fit_tier0(X_train_core: pd.DataFrame, y_train: pd.Series, baseline_train: pd.Series,
              sample_weight=None):
    """Returns the fitted (log-ratio-space) model; use predict_tier0() to get
    real-kW/m^2 predictions out of it."""
    model = build_hist_gb()
    target_train = _to_log_ratio(y_train, baseline_train)
    t0 = time.time()
    if sample_weight is not None:
        try:
            model.fit(X_train_core, target_train, sample_weight=sample_weight)
        except (TypeError, ValueError):
            model.fit(X_train_core, target_train)
    else:
        model.fit(X_train_core, target_train)
    return model, time.time() - t0


def predict_tier0(tier0_model, X_core: pd.DataFrame, baseline: pd.Series) -> np.ndarray:
    return _from_log_ratio(tier0_model.predict(X_core), baseline)


def fit_tier1_models(df_train: pd.DataFrame, y_train: pd.Series, tier0_pred_train: np.ndarray):
    """Trains one GPR correction model per eligible family, using only that
    family's rows present in this training split. tier0_pred_train must
    already be in real kW/m^2 units (i.e. the output of predict_tier0).
    Returns a dict {family: fitted_pipeline_or_None}. None means that family
    had no (or too little) training data in this split -- Tier-1 will be
    skipped for it."""
    residual_train = y_train.values - tier0_pred_train
    tier1_models = {}
    for family, cols in TIER1_FAMILY_FEATURES.items():
        mask = (df_train["geometry_family"] == family).values
        n = int(mask.sum())
        if n < 10:  # not enough data in this split to fit a meaningful correction
            tier1_models[family] = None
            continue
        X_fam = df_train.loc[mask, cols]
        y_fam = residual_train[mask]
        pipe = make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            GaussianProcessRegressor(
                kernel=ConstantKernel(1.0) * RBF(length_scale=1.0) + WhiteKernel(noise_level=1.0),
                normalize_y=True, random_state=RANDOM_STATE),
        )
        pipe.fit(X_fam, y_fam)
        tier1_models[family] = pipe
    return tier1_models


def predict_hierarchy(tier0_model, tier1_models: dict, df_test: pd.DataFrame,
                       X_test_core: pd.DataFrame, baseline_test: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    """Returns (y_pred, used_tier1_mask) -- used_tier1_mask[i] is True where
    a Tier-1 correction was actually applied for that test row."""
    y_pred = predict_tier0(tier0_model, X_test_core, baseline_test)
    used_tier1 = np.zeros(len(df_test), dtype=bool)

    for family, cols in TIER1_FAMILY_FEATURES.items():
        model = tier1_models.get(family)
        if model is None:
            continue
        mask = (df_test["geometry_family"] == family).values
        if mask.sum() == 0:
            continue
        correction = model.predict(df_test.loc[mask, cols])
        y_pred[mask] = y_pred[mask] + correction
        used_tier1[mask] = True

    return _clip_chf(y_pred), used_tier1

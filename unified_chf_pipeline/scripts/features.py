"""
features.py
------------
Builds the "core" feature matrix used by every model in this pipeline
(the Tier-0 baseline and all flat comparison models): the near-universal
mandatory operating/geometry columns, plus fluid thermophysical properties
computed from (fluid, pressure) via CoolProp -- this replaces the categorical
`fluid` label with continuous physics, which is what lets a model generalize
to conditions/fluids it didn't see much of in training.

CoolProp does not know "fc-72" (Fluorinert, not in its fluid database) --
those rows simply get NaN fluid properties, which every model used in this
pipeline (RandomForest, HistGradientBoosting, GPR w/ imputation, MLP w/
imputation) already handles.
"""
from functools import lru_cache

import numpy as np
import pandas as pd
from CoolProp.CoolProp import PropsSI

# Canonical fluid name -> CoolProp fluid name. Only fluids CoolProp actually
# supports get an entry; anything else (fc-72) is left out on purpose.
COOLPROP_FLUID_MAP = {
    "water": "Water",
    "r123": "R123",
}

CORE_NUMERIC_COLUMNS = [
    "pressure_kPa", "mass_flux_kg_m2s", "quality", "subcooling_K",
    "diameter_mm", "heated_length_mm", "reduced_pressure",
]

FLUID_PROP_COLUMNS = [
    "rho_l_coolprop", "rho_g_coolprop", "mu_l_coolprop", "k_l_coolprop",
    "cp_l_coolprop", "sigma_coolprop", "h_fg_coolprop",
]

GRAVITY = 9.80665  # m/s^2

# Critical pressure per fluid (Pa), used for reduced_pressure = P / P_crit.
# This puts water's 100-20,000 kPa range and R123's 200-1,500 kPa range on
# the same 0-1 scale instead of two numerically non-overlapping raw ranges a
# tree model has no way to relate to each other.
_PCRIT_PA = {name: PropsSI("Pcrit", cp_name) for name, cp_name in COOLPROP_FLUID_MAP.items()}

# Fixed, global category lists (not derived per-split) so the one-hot columns
# always have the same shape/meaning regardless of which split/fold is being
# trained -- including folds where a category is entirely absent from train.
GEOMETRY_FAMILIES = [
    "tube", "annulus", "plate", "helical_coil",
    "pin_fin_pool_boiling", "flat_heater_pool_boiling",
]
FLUIDS = ["water", "r123", "fc-72"]

# Geometry families where CHF is dominated by pool-boiling physics (no bulk
# flow) vs. flow-boiling physics (forced convection through a channel).
POOL_BOILING_FAMILIES = {"pin_fin_pool_boiling", "flat_heater_pool_boiling"}

# Neither pinfin_chf_water_fc72 nor mentor_master report a system pressure at
# all (0/175 and 0/55 rows respectively) -- both are open pool-boiling rigs,
# where atmospheric is the standard implicit condition unless stated
# otherwise. This assumed value is used ONLY to look up fluid properties for
# the physics baseline below; the real pressure_kPa feature is left NaN, not
# silently overwritten.
ASSUMED_POOL_BOILING_PRESSURE_KPA = 101.325

GEOMETRY_ONEHOT_COLUMNS = [f"geometry_family__{g}" for g in GEOMETRY_FAMILIES]
FLUID_ONEHOT_COLUMNS = [f"fluid__{f}" for f in FLUIDS]

# The model previously had NO explicit signal for "which physical regime is
# this row" -- it had to infer tube vs. pin-fin-pool-boiling vs. helical-coil
# purely from which numeric columns happened to be NaN and from CoolProp
# fluid properties. That's weak, especially for small minority sources (e.g.
# helical_coil_r123 is <1% of rows). Adding these one-hot flags directly was
# the single biggest lever for improving accuracy on those minority regimes.
FEATURE_COLUMNS = (
    CORE_NUMERIC_COLUMNS + FLUID_PROP_COLUMNS
    + GEOMETRY_ONEHOT_COLUMNS + FLUID_ONEHOT_COLUMNS
)


@lru_cache(maxsize=None)
def _saturation_properties(fluid_cp_name: str, pressure_pa: float):
    """Cached per (fluid, rounded pressure) -- there are only ~1500 distinct
    pressures in the whole merged table, so caching turns ~28k CoolProp calls
    into a few thousand."""
    try:
        rho_l = PropsSI("Dmass", "P", pressure_pa, "Q", 0, fluid_cp_name)
        rho_g = PropsSI("Dmass", "P", pressure_pa, "Q", 1, fluid_cp_name)
        mu_l = PropsSI("viscosity", "P", pressure_pa, "Q", 0, fluid_cp_name)
        k_l = PropsSI("conductivity", "P", pressure_pa, "Q", 0, fluid_cp_name)
        cp_l = PropsSI("Cpmass", "P", pressure_pa, "Q", 0, fluid_cp_name)
        sigma = PropsSI("surface_tension", "P", pressure_pa, "Q", 0, fluid_cp_name)
        h_l = PropsSI("Hmass", "P", pressure_pa, "Q", 0, fluid_cp_name)
        h_g = PropsSI("Hmass", "P", pressure_pa, "Q", 1, fluid_cp_name)
        return rho_l, rho_g, mu_l, k_l, cp_l, sigma, h_g - h_l
    except Exception:
        return (np.nan,) * 7


def add_fluid_properties(df: pd.DataFrame) -> pd.DataFrame:
    """Adds FLUID_PROP_COLUMNS + reduced_pressure to a copy of df, using each
    row's (fluid, pressure_kPa). Rows with an unsupported fluid get NaN
    properties. Pool-boiling rows with no reported pressure (pinfin,
    mentor -- 0/175 and 0/55 respectively) use ASSUMED_POOL_BOILING_PRESSURE_KPA
    for this lookup only; pressure_kPa itself is left untouched/NaN."""
    df = df.copy()
    props = {c: np.full(len(df), np.nan) for c in FLUID_PROP_COLUMNS}
    reduced_pressure = np.full(len(df), np.nan)

    for i, (fluid, p_kpa, family) in enumerate(
            zip(df["fluid"], df["pressure_kPa"], df["geometry_family"])):
        fluid_key = str(fluid).lower()
        cp_name = COOLPROP_FLUID_MAP.get(fluid_key)
        if cp_name is None:
            continue
        if pd.isna(p_kpa) or p_kpa <= 0:
            if family in POOL_BOILING_FAMILIES:
                p_kpa = ASSUMED_POOL_BOILING_PRESSURE_KPA
            else:
                continue
        # round pressure to the nearest 0.5 kPa for cache efficiency; CHF is
        # insensitive to sub-kPa pressure differences for property lookup purposes
        p_pa = round(float(p_kpa) * 1000 / 500) * 500
        vals = _saturation_properties(cp_name, p_pa)
        for c, v in zip(FLUID_PROP_COLUMNS, vals):
            props[c][i] = v
        reduced_pressure[i] = p_pa / _PCRIT_PA[fluid_key]

    for c in FLUID_PROP_COLUMNS:
        df[c] = props[c]
    df["reduced_pressure"] = reduced_pressure
    return df


def compute_physics_baseline_kw_m2(df: pd.DataFrame) -> pd.Series:
    """A closed-form, per-row physical CHF scale, used to non-dimensionalize
    the modeling target (see models.fit_predict_with_baseline). This is the
    single biggest lever for cross-fluid/cross-regime generalization: raw CHF
    magnitude differs by ~9x between water and R123 purely because their
    latent heats differ by ~9x (h_fg water ~1500-2000 kJ/kg vs R123
    ~170 kJ/kg) -- a model predicting raw kW/m^2 has to relearn that whole
    scale factor from scratch for every fluid it sees, which is exactly why
    a water-trained model predicts water-scale CHF for R123 test rows.
    Dividing by a physics-computed scale first removes that scale mismatch
    before the model ever sees the target.

    - Pool boiling rows: the classical Zuber hydrodynamic CHF correlation,
      q'' = C * h_fg * sqrt(rho_g) * (sigma * g * (rho_l - rho_g))^0.25,
      C = pi/24 ~= 0.131 (the commonly cited constant; the literature range
      is roughly 0.13-0.16 depending on the source). Computed purely from
      CoolProp properties -- needs zero training data for the target fluid.
    - Flow boiling rows: G * h_fg (a boiling-number-style flux scale --
      CHF / (G*h_fg) is the dimensionless Boiling number, typically
      0.0005-0.05 across a very wide range of fluids/conditions).
    - Rows without usable fluid properties (fc-72 -- not in CoolProp; or
      missing pressure/mass flux) fall back to a baseline of 1.0, which
      makes the transform in models.py reduce to plain log1p(CHF) for those
      rows -- i.e. exactly the previous behavior, not a regression.
    """
    rho_l, rho_g = df["rho_l_coolprop"], df["rho_g_coolprop"]
    sigma, h_fg = df["sigma_coolprop"], df["h_fg_coolprop"]

    zuber_w_m2 = 0.131 * h_fg * np.sqrt(rho_g) * (sigma * GRAVITY * (rho_l - rho_g)) ** 0.25
    flow_w_m2 = df["mass_flux_kg_m2s"] * h_fg

    is_pool = df["geometry_family"].isin(POOL_BOILING_FAMILIES)
    baseline_w_m2 = flow_w_m2.where(~is_pool, zuber_w_m2)
    baseline_kw_m2 = baseline_w_m2 / 1000.0

    baseline_kw_m2 = baseline_kw_m2.where(np.isfinite(baseline_kw_m2) & (baseline_kw_m2 > 0), 1.0)
    return baseline_kw_m2


def add_categorical_onehot(df: pd.DataFrame) -> pd.DataFrame:
    """Adds fixed-shape one-hot columns for geometry_family and fluid (see
    GEOMETRY_FAMILIES / FLUIDS above for why these are hard-coded rather than
    derived from whatever categories happen to be in df)."""
    df = df.copy()
    for g, col in zip(GEOMETRY_FAMILIES, GEOMETRY_ONEHOT_COLUMNS):
        df[col] = (df["geometry_family"] == g).astype(float)
    for f, col in zip(FLUIDS, FLUID_ONEHOT_COLUMNS):
        df[col] = (df["fluid"].str.lower() == f).astype(float)
    return df


def sample_weights_by_source(source_dataset: pd.Series) -> np.ndarray:
    """Inverse-sqrt-frequency weights so the ~86%-share nrc_groeneveld source
    doesn't drown out minority regimes (helical coil, pin-fin, mentor pool
    boiling) during training. sqrt (not full inverse) keeps the weighting
    gentle -- full inverse-frequency would let the 55-row mentor source
    outweigh the 24,579-row Groeneveld source by ~450x per point, which is
    too aggressive and just trades one kind of dominance for another."""
    counts = source_dataset.value_counts()
    w = source_dataset.map(lambda s: 1.0 / np.sqrt(counts[s]))
    return (w / w.mean()).values


def build_feature_matrix(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Returns (X, y) using FEATURE_COLUMNS as X and CHF_kW_m2 as y. Does NOT
    impute or scale -- that's each model's own job (tree models want raw
    NaN, ANN/GPR want imputed+scaled)."""
    df = add_fluid_properties(df)
    df = add_categorical_onehot(df)
    X = df[FEATURE_COLUMNS].copy()
    y = df["CHF_kW_m2"].copy()
    return X, y

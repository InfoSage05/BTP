"""
features_v2.py
--------------
Feature construction for the physics ablation. Two feature spaces, selectable
so the ablation can isolate what the change of basis is worth:

  "raw"  -- what the pipeline uses today: pressure_kPa, mass_flux, quality,
            subcooling, diameter_mm, heated_length_mm, reduced_pressure, plus
            CoolProp saturation properties and geometry/fluid one-hots.

  "pi"   -- the dimensionless map of foundation doc section 2.2/9.3: Weber,
            Katto, density ratio, reduced pressure, quality, L/D, D/D_ref,
            Jakob, Bond, K1, Tanase n, baseline boiling number, confinement,
            plus the sparse surface groups and the same one-hots.

Why the one-hots stay in BOTH spaces: they were the single biggest accuracy
lever in the original pipeline (helical_coil_r123 went from R^2 = -10.3 to
0.93-0.96 on the random split when geometry_family was added). Dropping them
in the "pi" arm would confound the change of basis with the loss of the
regime indicator, and the ablation would measure the wrong thing.
"""
import numpy as np
import pandas as pd

from physics import baseline as phys_baseline
from physics import groups as phys_groups
from physics import properties as phys_properties

POOL_BOILING_FAMILIES = phys_baseline.POOL_BOILING_FAMILIES

GEOMETRY_FAMILIES = [
    "tube", "annulus", "plate", "helical_coil",
    "pin_fin_pool_boiling", "flat_heater_pool_boiling",
]
FLUIDS = ["water", "r123", "fc-72"]

GEOMETRY_ONEHOT_COLUMNS = [f"geometry_family__{g}" for g in GEOMETRY_FAMILIES]
FLUID_ONEHOT_COLUMNS = [f"fluid__{f}" for f in FLUIDS]
ONEHOT_COLUMNS = GEOMETRY_ONEHOT_COLUMNS + FLUID_ONEHOT_COLUMNS

RAW_NUMERIC_COLUMNS = [
    "pressure_kPa", "mass_flux_kg_m2s", "quality", "subcooling_kJkg",
    "diameter_mm", "heated_length_mm", "reduced_pressure",
]
RAW_PROPERTY_COLUMNS = [f"{n}_sat" for n in phys_properties.PROPERTY_NAMES]

PI_COLUMNS = phys_groups.DIMENSIONLESS_COLUMNS + phys_groups.SURFACE_COLUMNS
CORR_COLUMNS = phys_groups.CORRELATION_COLUMNS

FEATURE_SPACES = ("raw", "pi", "pi_corr", "raw_corr")


def feature_columns(space: str) -> list:
    if space == "raw":
        return RAW_NUMERIC_COLUMNS + RAW_PROPERTY_COLUMNS + ONEHOT_COLUMNS
    if space == "pi":
        return PI_COLUMNS + ONEHOT_COLUMNS
    if space == "pi_corr":
        return PI_COLUMNS + CORR_COLUMNS + ONEHOT_COLUMNS
    if space == "raw_corr":
        return RAW_NUMERIC_COLUMNS + RAW_PROPERTY_COLUMNS + CORR_COLUMNS + ONEHOT_COLUMNS
    raise ValueError(f"space must be one of {FEATURE_SPACES}, got {space!r}")


def add_categorical_onehot(df: pd.DataFrame) -> pd.DataFrame:
    """Fixed-shape one-hots, hard-coded rather than derived from whatever
    categories happen to be in `df` -- so a fold that holds out an entire
    family still produces a matrix of the same shape and meaning."""
    df = df.copy()
    for g, col in zip(GEOMETRY_FAMILIES, GEOMETRY_ONEHOT_COLUMNS):
        df[col] = (df["geometry_family"] == g).astype(float)
    fluid_lower = df["fluid"].astype(str).str.lower()
    for f, col in zip(FLUIDS, FLUID_ONEHOT_COLUMNS):
        df[col] = (fluid_lower == f).astype(float)
    return df


def prepare(df: pd.DataFrame, baseline_mode: str = "gated") -> pd.DataFrame:
    """Full feature preparation: properties -> baseline -> groups -> one-hots.

    Ordering matters. The dimensionless group `pi_Bo_baseline` is derived from
    the baseline, so the baseline has to exist first; the baseline needs
    saturation properties, so those come first of all.
    """
    out = phys_properties.add_saturation_properties(df, POOL_BOILING_FAMILIES)
    out = phys_baseline.add_physics_baseline(out, mode=baseline_mode)
    out = phys_groups.add_dimensionless_groups(out)
    out = add_categorical_onehot(out)
    return out


def build_matrix(df_prepared: pd.DataFrame, space: str) -> pd.DataFrame:
    cols = feature_columns(space)
    missing = [c for c in cols if c not in df_prepared.columns]
    if missing:
        raise KeyError(f"prepare() did not produce required columns: {missing}")
    X = df_prepared[cols].copy()
    return X.apply(pd.to_numeric, errors="coerce")


def sample_weights_by_source(source_dataset: pd.Series) -> np.ndarray:
    """Inverse-sqrt-frequency weights, unchanged from the original pipeline:
    keeps the 86%-share nrc_groeneveld source from drowning out the minority
    regimes without letting the 55-row mentor source dominate instead."""
    counts = source_dataset.value_counts()
    w = source_dataset.map(lambda s: 1.0 / np.sqrt(counts[s]))
    return (w / w.mean()).values

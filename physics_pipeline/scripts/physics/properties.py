"""
physics/properties.py
---------------------
Saturation thermophysical properties for every fluid in the merged dataset.

This replaces `features.add_fluid_properties`, which had two gaps that
mattered a lot for the 230 pool-boiling rows:

  1. CoolProp does not know FC-72 (Fluorinert FC-72 / perfluorohexane is not
     in its fluid database), so 149 of the 175 pin-fin rows -- 65% of all
     pool-boiling data -- got NaN properties, which made
     `compute_physics_baseline_kw_m2` fall back to a baseline of 1.0. The
     entire pool-boiling physics apparatus therefore reached only 81 rows.
  2. Several sources report their own measured properties per row
     (`mentor_master` reports sigma, rho_l, cp_l, k_l, mu_l for all 55 of its
     rows). Those are better than a generic lookup at an assumed pressure,
     and they were being ignored.

Resolution order, per row:
    row-reported property  >  CoolProp lookup  >  constant table  >  NaN

FC-72 property source
---------------------
3M Fluorinert Electronic Liquid FC-72 product datasheet, values at the
1 atm normal boiling point (56 degC):

    T_sat   = 56 degC          rho_l = 1680 kg/m^3     h_fg  = 88 kJ/kg
    sigma   = 10 dyne/cm       cp_l  = 1100 J/kg-K     k_l   = 0.057 W/m-K
    nu_l    = 0.38 cSt

Two honest caveats, both flagged in the code below:
  - rho_l = 1680 kg/m^3 is the datasheet's 25 degC value, not the 56 degC
    saturation value. It enters Zuber only as (rho_l - rho_g)^0.25, so a 5%
    density error moves the CHF scale by ~1%.
  - The datasheet gives no vapour density. rho_g is computed from the ideal
    gas law using the C6F14 molar mass (338.04 g/mol) at T_sat and 1 atm.
    This is an ESTIMATE, marked as such; FC-72 vapour at 1 atm is far from
    critical so ideal-gas is a reasonable approximation, but it is not a
    measured value.

FC-72 is only ever used at pool-boiling conditions here (all 149 rows are
open pin-fin rigs at atmospheric pressure), so a single-point property set is
adequate. If FC-72 flow-boiling data is ever added, this must become a
pressure-dependent table.
"""
from functools import lru_cache

import numpy as np
import pandas as pd
from CoolProp.CoolProp import PropsSI

GRAVITY = 9.80665  # m/s^2
R_UNIVERSAL = 8.314462618  # J/(mol K)

# Canonical fluid name -> CoolProp fluid name.
COOLPROP_FLUID_MAP = {
    "water": "Water",
    "r123": "R123",
}

# Open pool-boiling rigs report no system pressure; atmospheric is the
# standard implicit condition. Used ONLY for property lookup -- the real
# `pressure_kPa` feature is left NaN, never overwritten.
ASSUMED_POOL_BOILING_PRESSURE_KPA = 101.325

_FC72_MOLAR_MASS_KG_MOL = 0.33804  # C6F14
_FC72_TSAT_K = 56.0 + 273.15

#: FC-72 saturation properties at 1 atm. See module docstring for provenance.
FC72_PROPERTIES = {
    "rho_l": 1680.0,        # kg/m^3   [3M datasheet, 25 degC value]
    "h_fg": 88.0e3,         # J/kg     [3M datasheet, at normal boiling point]
    "sigma": 0.010,         # N/m      [3M datasheet, 10 dyne/cm]
    "cp_l": 1100.0,         # J/(kg K) [3M datasheet]
    "k_l": 0.057,           # W/(m K)  [3M datasheet]
    # mu_l = nu_l * rho_l = 0.38 cSt * 1680 kg/m^3
    "mu_l": 0.38e-6 * 1680.0,   # Pa s  [3M datasheet, derived from kinematic viscosity]
    # ESTIMATE, not a datasheet value -- ideal gas at T_sat, 1 atm.
    "rho_g": 101325.0 * _FC72_MOLAR_MASS_KG_MOL / (R_UNIVERSAL * _FC72_TSAT_K),
    "p_crit": 1.83e6,       # Pa -- see note below
}

# FC-72 critical pressure is quoted as ~18.3 bar in the boiling-heat-transfer
# literature. It is used only for the reduced-pressure feature, where all 149
# FC-72 rows sit at the same P/P_c anyway, so the exact value cannot change
# any model comparison -- it only sets a constant offset for those rows.

# Property columns the merged table may already carry per row, mapped to the
# canonical name used internally. `mentor_master` fills all of these.
ROW_REPORTED_COLUMNS = {
    "rho_l": "rho_l_kg_m3",
    "sigma": "surface_tension_N_m",
    "cp_l": "cp_l_J_kgK",
    "k_l": "kl_W_mK",
    "mu_l": "mu_l_Pa_s",
}

PROPERTY_NAMES = ["rho_l", "rho_g", "mu_l", "k_l", "cp_l", "sigma", "h_fg"]
PROPERTY_COLUMNS = [f"{name}_sat" for name in PROPERTY_NAMES]

_PCRIT_PA = {name: PropsSI("Pcrit", cp) for name, cp in COOLPROP_FLUID_MAP.items()}
_PCRIT_PA["fc-72"] = FC72_PROPERTIES["p_crit"]


@lru_cache(maxsize=None)
def _coolprop_saturation(fluid_cp_name: str, pressure_pa: float):
    """Cached saturation lookup. There are only ~1,500 distinct pressures in
    the merged table, so caching turns ~28k CoolProp calls into a few thousand."""
    try:
        rho_l = PropsSI("Dmass", "P", pressure_pa, "Q", 0, fluid_cp_name)
        rho_g = PropsSI("Dmass", "P", pressure_pa, "Q", 1, fluid_cp_name)
        mu_l = PropsSI("viscosity", "P", pressure_pa, "Q", 0, fluid_cp_name)
        k_l = PropsSI("conductivity", "P", pressure_pa, "Q", 0, fluid_cp_name)
        cp_l = PropsSI("Cpmass", "P", pressure_pa, "Q", 0, fluid_cp_name)
        sigma = PropsSI("surface_tension", "P", pressure_pa, "Q", 0, fluid_cp_name)
        h_l = PropsSI("Hmass", "P", pressure_pa, "Q", 0, fluid_cp_name)
        h_g = PropsSI("Hmass", "P", pressure_pa, "Q", 1, fluid_cp_name)
        return dict(rho_l=rho_l, rho_g=rho_g, mu_l=mu_l, k_l=k_l,
                    cp_l=cp_l, sigma=sigma, h_fg=h_g - h_l)
    except Exception:
        return {name: np.nan for name in PROPERTY_NAMES}


def _lookup_row(fluid: str, pressure_kpa, is_pool: bool) -> dict:
    fluid_key = str(fluid).lower().strip()

    if fluid_key == "fc-72":
        return {name: FC72_PROPERTIES[name] for name in PROPERTY_NAMES}

    cp_name = COOLPROP_FLUID_MAP.get(fluid_key)
    if cp_name is None:
        return {name: np.nan for name in PROPERTY_NAMES}

    if pd.isna(pressure_kpa) or pressure_kpa <= 0:
        if not is_pool:
            return {name: np.nan for name in PROPERTY_NAMES}
        pressure_kpa = ASSUMED_POOL_BOILING_PRESSURE_KPA

    # Round to 0.5 kPa for cache efficiency; CHF is insensitive to sub-kPa
    # pressure differences at the property-lookup level.
    p_pa = round(float(pressure_kpa) * 1000 / 500) * 500
    p_pa = float(np.clip(p_pa, 1000.0, 0.999 * _PCRIT_PA[fluid_key]))
    return _coolprop_saturation(cp_name, p_pa)


def add_saturation_properties(df: pd.DataFrame, pool_families: set) -> pd.DataFrame:
    """Adds `<name>_sat` columns plus `reduced_pressure` and `p_crit_Pa`.

    Row-reported measured properties take precedence over the looked-up
    values -- if a source measured sigma at its own conditions, that beats a
    generic saturation lookup at an assumed pressure.
    """
    df = df.copy()
    n = len(df)
    out = {name: np.full(n, np.nan) for name in PROPERTY_NAMES}
    reduced_pressure = np.full(n, np.nan)
    p_crit = np.full(n, np.nan)

    is_pool_series = df["geometry_family"].isin(pool_families).values
    for i, (fluid, p_kpa, is_pool) in enumerate(
            zip(df["fluid"].values, df["pressure_kPa"].values, is_pool_series)):
        vals = _lookup_row(fluid, p_kpa, bool(is_pool))
        for name in PROPERTY_NAMES:
            out[name][i] = vals[name]

        fluid_key = str(fluid).lower().strip()
        pc = _PCRIT_PA.get(fluid_key, np.nan)
        p_crit[i] = pc
        p_eff = p_kpa
        if (pd.isna(p_eff) or p_eff <= 0) and is_pool:
            p_eff = ASSUMED_POOL_BOILING_PRESSURE_KPA
        if not pd.isna(p_eff) and p_eff > 0 and np.isfinite(pc):
            reduced_pressure[i] = (p_eff * 1000.0) / pc

    for name in PROPERTY_NAMES:
        df[f"{name}_sat"] = out[name]

    # Row-reported measurements override the lookup where present.
    for name, col in ROW_REPORTED_COLUMNS.items():
        if col in df.columns:
            reported = pd.to_numeric(df[col], errors="coerce")
            usable = reported.notna() & (reported > 0)
            df.loc[usable, f"{name}_sat"] = reported[usable]

    df["reduced_pressure"] = reduced_pressure
    df["p_crit_Pa"] = p_crit
    return df

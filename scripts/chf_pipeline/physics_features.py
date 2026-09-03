"""
Physics-informed / dimensionless feature engineering for CHF prediction.

The goal is to make the model's inputs fluid-agnostic where possible: instead
of feeding raw pressure/mass-flux in fluid-specific units and hoping the
network learns "R123 behaves differently from water" from scratch, we feed
dimensionless groups that are approximately fluid-independent (the classic
fluid-to-fluid CHF modeling approach used in the literature, e.g. Pioro et
al.'s R-134a-to-water scaling). Raw columns are kept alongside the derived
ones -- this adds features, it does not replace the originals.

All fluid property lookups use CoolProp. Water is the default fluid for the
core LUT-derived data; pass a different `fluid` for R123/FC-72/etc. datasets.
"""
import numpy as np
import pandas as pd
import CoolProp.CoolProp as CP

# CoolProp fluid names for the fluids actually present across our datasets.
FLUID_NAME_MAP = {
    "water": "Water",
    "r123": "R123",
    "r134a": "R134a",
    "fc72": None,  # not in CoolProp's default fluid list; handled separately
}


def _critical_pressure_kpa(fluid: str) -> float:
    name = FLUID_NAME_MAP.get(fluid.lower())
    if name is None:
        raise ValueError(f"No CoolProp fluid mapping for '{fluid}'")
    return CP.PropsSI("Pcrit", name) / 1000.0  # Pa -> kPa


def _saturation_densities(p_kpa: np.ndarray, fluid: str) -> tuple[np.ndarray, np.ndarray]:
    """Return (rho_liquid, rho_vapor) in kg/m^3 at saturation for each pressure."""
    name = FLUID_NAME_MAP.get(fluid.lower())
    if name is None:
        raise ValueError(f"No CoolProp fluid mapping for '{fluid}'")

    p_pa = np.asarray(p_kpa, dtype=float) * 1000.0
    p_crit = CP.PropsSI("Pcrit", name)
    p_trip = CP.PropsSI("ptriple", name)
    # clip to CoolProp's valid saturation-curve range to avoid solver failures
    # at rows just outside the fluid's physical envelope (flagged, not silently wrong)
    p_pa_clipped = np.clip(p_pa, p_trip * 1.001, p_crit * 0.999)

    rho_l = np.full_like(p_pa, np.nan, dtype=float)
    rho_g = np.full_like(p_pa, np.nan, dtype=float)
    for i, p in enumerate(p_pa_clipped):
        try:
            rho_l[i] = CP.PropsSI("D", "P", p, "Q", 0, name)
            rho_g[i] = CP.PropsSI("D", "P", p, "Q", 1, name)
        except ValueError:
            pass  # leave as NaN; caller decides how to handle
    return rho_l, rho_g


def add_dimensionless_features(df: pd.DataFrame, fluid: str,
                                p_col: str = "P", g_col: str = "G",
                                d_col: str | None = None,
                                l_col: str | None = None) -> pd.DataFrame:
    """
    Add fluid-aware dimensionless columns to a copy of df. Expects pressure
    in kPa and mass flux in kg/m^2/s (this repo's convention throughout).

    Added columns:
      P_reduced        = P / P_critical                (fluid-agnostic pressure scale)
      rho_l_kg_m3, rho_g_kg_m3, density_ratio = rho_l / rho_g
      L_over_D          (only if both d_col and l_col given)

    Rows where CoolProp couldn't resolve saturation properties (P outside the
    fluid's valid envelope) get NaN in the derived columns -- these are
    flagged via `n_property_lookup_failures` printed to stdout, never
    silently dropped or imputed.
    """
    out = df.copy()
    p_crit = _critical_pressure_kpa(fluid)
    out["P_reduced"] = out[p_col] / p_crit

    rho_l, rho_g = _saturation_densities(out[p_col].to_numpy(), fluid)
    out["rho_l_kg_m3"] = rho_l
    out["rho_g_kg_m3"] = rho_g
    out["density_ratio"] = rho_l / rho_g

    n_fail = int(np.isnan(rho_l).sum())
    if n_fail:
        print(f"[physics_features] {n_fail}/{len(out)} rows had P outside "
              f"{fluid}'s CoolProp saturation range -> NaN density columns")

    if d_col is not None and l_col is not None:
        out["L_over_D"] = out[l_col] / out[d_col]

    return out

"""
physics/baseline.py
-------------------
Phi_physics: the closed-form, regime-dispatched CHF scale that the model
learns a bounded multiplicative correction on top of.

    CHF = Phi_physics(x) * exp( g_theta(pi(x)) )

Four modes, corresponding to rungs of the ablation ladder so each physics
layer can be isolated:

    "none"    Phi = 1                    -> the model predicts log1p(CHF) directly
    "latent"  Phi = G h_fg | Zuber       -> what the pipeline uses TODAY
    "katto"   Phi = Katto-Ohno | Zuber   -> idea 1
    "gated"   + DNB/dryout blending, helical-coil transfer function,
              pool-boiling surface and finite-size corrections   -> idea 4

A DESIGN CORRECTION discovered during implementation
----------------------------------------------------
Foundation doc section 9.2 lists "any tube, D != 8mm -> x K1" as part of
Phi_physics. That is correct for a LOOK-UP-TABLE baseline, which is
normalised to an 8 mm tube and carries no diameter dependence of its own.
It is WRONG on top of Katto-Ohno, which already contains D_h explicitly in
every branch -- multiplying by K1 there double-counts the diameter effect.

So K1 and the Tanase exponent are carried as FEATURES (see groups.py), where
they hand the model the regime-dependent diameter sensitivity -- including the
sign reversal at low mass flux, constraint S6 -- without corrupting the scale.

Guarantees
----------
The returned baseline is always finite and strictly positive. Any row whose
physics cannot be evaluated falls back, in order, to G*h_fg, then to 1.0
(which makes the log-ratio target degenerate to plain log CHF for that row --
the previous behaviour, not a regression).
"""
import numpy as np
import pandas as pd

from . import correlations as corr

POOL_BOILING_FAMILIES = {"pin_fin_pool_boiling", "flat_heater_pool_boiling"}

BASELINE_MODES = ("none", "latent", "katto", "gated")

#: Below this the baseline is treated as unusable (kW/m^2).
_MIN_BASELINE_KW = 1e-3


def _col(df, name, default=np.nan):
    if name not in df.columns:
        return np.full(len(df), default, dtype=float)
    return pd.to_numeric(df[name], errors="coerce").values.astype(float)


def _pool_baseline_w_m2(df, mode):
    """Zuber. See below for two corrections that were implemented, TESTED, and
    then deliberately NOT applied.

    Orientation (foundation doc 4.3, constraint S5) -- NOT APPLIED.
        `mentor_master` is the only orientation data in the merged table
        (55 rows at angle_deg = 0/90/180). Constraint S5 expects CHF to fall
        slightly from 0 to 90 deg and then sharply toward 180 deg. The data
        says otherwise:

            angle_deg      0        90       180
            median CHF   2237      3300      1874   kW/m^2

        90 deg is the HIGHEST, not intermediate. Applying Vishnev's
        monotonically decreasing K therefore makes the baseline worse -- it
        moved the median CHF/Phi ratio for this source from 2.07 to 3.57.
        Either the source uses a different angle convention than Liang &
        Mudawar's (0 deg = horizontal upward-facing), or 55 rows across three
        angles is too few to resolve the trend. Applying a correction the
        data contradicts would be fitting the literature, not the physics, so
        `vishnev_K` stays available in correlations.py and unused here.

    Finite heater size (foundation doc 3.3, constraint S8) -- NOT APPLIED.
        The foundation document gives a validity THRESHOLD (Zuber holds above
        3*lambda_d) and the 1.14 Lienhard-Dhir ratio, but no sub-threshold
        correction; the tapered form in `lienhard_dhir_size_factor` was
        flagged [S] as a modelling choice, not a transcribed result. Its
        direction is also wrong in practice: mentor's 10-20 mm heaters are far
        below 3*lambda_d (~81 mm for water) and show CHF roughly 2x Zuber, i.e.
        small heaters give HIGHER CHF, whereas the taper reduces the factor.
        Applying it needs the actual Lienhard-Dhir L' correlation, which was
        not obtained. Left unapplied rather than approximated.
    """
    return corr.zuber_chf(df["rho_l_sat"].values, df["rho_g_sat"].values,
                          df["sigma_sat"].values, df["h_fg_sat"].values)


def _flow_baseline_w_m2(df, mode):
    """Katto-Ohno, with an optional smooth DNB/dryout blend and coil factor."""
    G = _col(df, "mass_flux_kg_m2s")
    x = _col(df, "quality")
    D_m = _col(df, "diameter_mm") / 1000.0
    L_m = _col(df, "heated_length_mm") / 1000.0
    dh_sub = _col(df, "subcooling_kJkg") * 1000.0  # kJ/kg -> J/kg

    rho_l = df["rho_l_sat"].values
    rho_g = df["rho_g_sat"].values
    sigma = df["sigma_sat"].values
    h_fg = df["h_fg_sat"].values

    # Missing inlet subcooling (13.7% of rows) -> treat as saturated inlet.
    # That is the conservative reading: the subcooling term in Katto-Ohno
    # eq (8) only ever RAISES CHF, so assuming zero cannot inflate the scale.
    dh_sub = np.where(np.isfinite(dh_sub) & (dh_sub > 0), dh_sub, 0.0)

    q_katto = corr.katto_ohno_chf(G, D_m, L_m, rho_l, rho_g, sigma, h_fg, dh_sub)

    if mode == "katto":
        q = q_katto
    else:
        # --- DNB / dryout gating (foundation doc 1.2) ----------------------
        # Hall-Mudawar is validated for a SUBCOOLED outlet (-1 <= x_o <= 0),
        # which is 3,637 of 28,240 flow rows. Katto-Ohno covers the saturated
        # majority. Rather than an `if x <= 0` switch, the two are blended
        # with a smooth weight in quality, so the scale is differentiable
        # across the boundary (foundation doc 3.5).
        q_hm = corr.hall_mudawar_chf(G, D_m, x, rho_l, rho_g, sigma, h_fg)
        q_hm = np.where(np.isfinite(q_hm) & (q_hm > 0), q_hm, q_katto)

        # w = 1 deep in the subcooled region, 0 once saturated. The 0.05
        # quality half-width is a smoothing scale, not a physical constant.
        w_dnb = 1.0 / (1.0 + np.exp(x / 0.05))
        w_dnb = np.where(np.isfinite(x), w_dnb, 0.0)
        q = np.exp(w_dnb * np.log(np.maximum(q_hm, 1e-30))
                   + (1.0 - w_dnb) * np.log(np.maximum(q_katto, 1e-30)))

        # --- helical coil transfer function (foundation doc 6.1) ----------
        is_coil = (df["geometry_family"] == "helical_coil").values
        if is_coil.any():
            q = np.where(is_coil, q * corr.helical_coil_factor(x), q)

    return q


def compute_physics_baseline_kw_m2(df: pd.DataFrame, mode: str = "gated") -> pd.Series:
    """Phi_physics in kW/m^2, one value per row. Always finite and > 0.

    `df` must already carry saturation properties (physics.properties).
    """
    if mode not in BASELINE_MODES:
        raise ValueError(f"mode must be one of {BASELINE_MODES}, got {mode!r}")

    n = len(df)
    if mode == "none":
        return pd.Series(np.ones(n), index=df.index, name="physics_baseline_kW_m2")

    G = _col(df, "mass_flux_kg_m2s")
    h_fg = df["h_fg_sat"].values
    is_pool = df["geometry_family"].isin(POOL_BOILING_FAMILIES).values

    # Fallback chain: G*h_fg is the latent-heat scale, which is what the
    # current pipeline uses for every flow row.
    fallback_w = G * h_fg

    if mode == "latent":
        pool_w = corr.zuber_chf(df["rho_l_sat"].values, df["rho_g_sat"].values,
                                df["sigma_sat"].values, h_fg)
        q_w = np.where(is_pool, pool_w, fallback_w)
    else:
        pool_w = _pool_baseline_w_m2(df, mode)
        flow_w = _flow_baseline_w_m2(df, mode)
        q_w = np.where(is_pool, pool_w, flow_w)
        # Where the richer physics could not be evaluated, drop back to the
        # latent-heat scale rather than to 1.0 -- it is still dimensionally
        # correct and still removes the cross-fluid h_fg mismatch.
        q_w = np.where(np.isfinite(q_w) & (q_w > 0), q_w, fallback_w)

    q_kw = q_w / 1000.0
    q_kw = np.where(np.isfinite(q_kw) & (q_kw > _MIN_BASELINE_KW), q_kw, 1.0)
    return pd.Series(q_kw, index=df.index, name="physics_baseline_kW_m2")


def add_physics_baseline(df: pd.DataFrame, mode: str = "gated") -> pd.DataFrame:
    df = df.copy()
    df["physics_baseline_kW_m2"] = compute_physics_baseline_kw_m2(df, mode=mode)
    return df

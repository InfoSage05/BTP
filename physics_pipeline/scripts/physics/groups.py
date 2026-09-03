"""
physics/groups.py
-----------------
The dimensionless feature map pi(x).

Foundation doc section 2.2 quotes NUREG/KM-0011's own general statement of a
tube CHF correlation:

    CHF / (h_fg G) = f( rho_f/rho_g , D^0.5/(sigma rho_f)^0.5 , x , D/D_ref )

Katto-Ohno, Hall-Mudawar and Merilo are all written in exactly this space.
A model in this space can map water at 15 MPa and R123 at 1 MPa onto the same
point; a model in raw kPa / kg m^-2 s^-1 / mm cannot. That is constraint C7
(dimensional homogeneity) and the whole argument for idea 2.

ONE RULE, enforced by construction: no group here may contain q''_CHF.
The boiling number Bo = q''/(G h_fg) is the natural target, not a feature --
including the measured Bo would leak the label. `Bo_baseline` below is the
boiling number OF THE PHYSICS BASELINE, a pure function of the inputs, and is
therefore legitimate.
"""
import numpy as np
import pandas as pd

from . import correlations as corr

GRAVITY = corr.GRAVITY

#: The dimensionless columns produced by `add_dimensionless_groups`.
DIMENSIONLESS_COLUMNS = [
    "pi_We_D",            # G^2 D / (rho_l sigma)      -- inertia vs surface tension
    "pi_katto",           # sigma rho_l / (G^2 L)      -- inverse Weber on heated length
    "pi_density_ratio",   # rho_l / rho_g              -- regime / reduced-pressure proxy
    "pi_reduced_pressure",# P / P_crit                 -- property-collapse coordinate
    "pi_quality",         # x                          -- already dimensionless
    "pi_L_over_D",        # L / D_h                    -- heated-length / history effect
    "pi_D_over_Dref",     # D / 8 mm                   -- geometry ratio
    "pi_jakob",           # rho_l dh_sub / (rho_g h_fg) -- subcooling
    "pi_bond",            # (rho_l - rho_g) g D^2 / sigma -- buoyancy vs surface tension
    "pi_K1_diameter",     # (8/D)^n, n from Tanase     -- regime-dependent diameter effect
    "pi_tanase_n",        # the exponent itself        -- carries the sign reversal (S6)
    "pi_Bo_baseline",     # Phi_physics / (G h_fg)     -- physics-only boiling number
    "pi_confinement",     # capillary length / D       -- micro/mini-channel indicator
]

#: Correlation-stacking features. Each is the boiling number that ONE
#: correlation predicts, in logs:  log( q''_corr / (G h_fg) ).
#:
#: This is a different architecture from dividing by a single baseline. The
#: ablation showed that committing to one correlation as Phi_physics is
#: fragile -- Katto-Ohno is excellent on the NRC fold (R2 = 0.926) and poor on
#: zhao2020 (-0.362), and no single correlation spans all seven rigs. Handing
#: the model every correlation's prediction as a FEATURE lets it learn where
#: each one is trustworthy, instead of the modeller having to choose in
#: advance. The target stays log(Bo), so these features and the target live in
#: the same space and a "just copy Katto-Ohno" solution is representable.
CORRELATION_COLUMNS = [
    "phi_zuber_logBo",
    "phi_katto_logBo",
    "phi_hallmudawar_logBo",
    "phi_biasi_logBo",
    "phi_corr_spread",   # max - min across correlations = disagreement
]

#: Sparse surface-characteristic groups. Coverage in the merged table is
#: 0.6-0.8% (constraint D2), so these are almost entirely NaN. Tree models
#: handle that natively; they are carried so the surface sub-study has a
#: place to plug in, not because they can support a headline claim.
SURFACE_COLUMNS = [
    "pi_roughness_wenzel",   # Wenzel area ratio r (roughness_factor), 1.0 = smooth
    "pi_porosity",
    "pi_orientation_deg",
    "pi_fin_aspect",         # fin height / fin spacing -- wicking geometry proxy
]


def _col(df, name):
    """Numeric view of a column that may be absent or of mixed dtype."""
    if name not in df.columns:
        return pd.Series(np.nan, index=df.index)
    return pd.to_numeric(df[name], errors="coerce")


def add_dimensionless_groups(df: pd.DataFrame) -> pd.DataFrame:
    """Adds DIMENSIONLESS_COLUMNS + SURFACE_COLUMNS.

    Requires `add_saturation_properties` and `add_physics_baseline` to have
    run first (needs `<name>_sat`, `reduced_pressure`, `physics_baseline_kW_m2`).
    """
    df = df.copy()

    G = _col(df, "mass_flux_kg_m2s").values
    x = _col(df, "quality").values
    D_m = _col(df, "diameter_mm").values / 1000.0
    L_m = _col(df, "heated_length_mm").values / 1000.0
    dh_sub = _col(df, "subcooling_kJkg").values * 1000.0  # kJ/kg -> J/kg

    rho_l = df["rho_l_sat"].values
    rho_g = df["rho_g_sat"].values
    sigma = df["sigma_sat"].values
    h_fg = df["h_fg_sat"].values

    with np.errstate(divide="ignore", invalid="ignore"):
        df["pi_We_D"] = G ** 2 * D_m / (rho_l * sigma)
        df["pi_katto"] = sigma * rho_l / (G ** 2 * L_m)
        df["pi_density_ratio"] = rho_l / rho_g
        df["pi_reduced_pressure"] = df["reduced_pressure"].values
        df["pi_quality"] = x
        df["pi_L_over_D"] = L_m / D_m
        df["pi_D_over_Dref"] = (D_m * 1000.0) / corr.LUT_REFERENCE_DIAMETER_MM
        df["pi_jakob"] = rho_l * dh_sub / (rho_g * h_fg)
        df["pi_bond"] = (rho_l - rho_g) * GRAVITY * D_m ** 2 / sigma
        df["pi_confinement"] = corr.capillary_length(rho_l, rho_g, sigma) / D_m

        n = corr.tanase_diameter_exponent(
            _col(df, "pressure_kPa").values, G, x)
        df["pi_tanase_n"] = n
        df["pi_K1_diameter"] = corr.lut_diameter_factor_K1(
            _col(df, "diameter_mm").values, n)

        if "physics_baseline_kW_m2" in df.columns:
            df["pi_Bo_baseline"] = (
                df["physics_baseline_kW_m2"].values * 1000.0) / (G * h_fg)
        else:
            df["pi_Bo_baseline"] = np.nan

    # --- sparse surface groups ---------------------------------------------
    df["pi_roughness_wenzel"] = _col(df, "roughness_factor").values
    df["pi_porosity"] = _col(df, "porosity").values
    df["pi_orientation_deg"] = _col(df, "angle_deg").values
    fin_h = _col(df, "fin_height_um").values
    fin_s = _col(df, "fin_spacing_um").values
    with np.errstate(divide="ignore", invalid="ignore"):
        df["pi_fin_aspect"] = np.where(fin_s > 0, fin_h / fin_s, np.nan)

    # --- correlation-stacking features -------------------------------------
    # Each correlation is MASKED to its own stated validity domain and emitted
    # as NaN elsewhere. Without this the features are actively harmful:
    # Hall-Mudawar's bracket [1 - C4 (rho_f/rho_g)^C5 x] turns negative once
    # the outlet is saturated, so on the dryout majority it produced log-Bo
    # values down to -92, and the "disagreement" feature just measured that
    # blow-up rather than any real disagreement. A correlation used outside
    # its range is not weak evidence, it is noise, and NaN says so honestly --
    # the tree models read NaN natively.
    Ghfg = G * h_fg
    dh_sub_pos = np.where(np.isfinite(dh_sub) & (dh_sub > 0), dh_sub, 0.0)
    has_flow = np.isfinite(G) & (G > 1.0)   # log-Bo is meaningless as G -> 0

    with np.errstate(divide="ignore", invalid="ignore"):
        candidates = {
            # Zuber is a pool-boiling scale; expressed as a boiling number it
            # is only meaningful where there is actually a flow to compare to.
            "phi_zuber_logBo": (
                corr.zuber_chf(rho_l, rho_g, sigma, h_fg), has_flow),
            # Katto-Ohno: flow boiling, needs heated length and diameter.
            "phi_katto_logBo": (
                corr.katto_ohno_chf(G, D_m, L_m, rho_l, rho_g, sigma, h_fg, dh_sub_pos),
                has_flow & np.isfinite(D_m) & np.isfinite(L_m)),
            # Hall-Mudawar: validated for a SUBCOOLED outlet, -1.0 <= x <= 0.
            "phi_hallmudawar_logBo": (
                corr.hall_mudawar_chf(G, D_m, x, rho_l, rho_g, sigma, h_fg),
                has_flow & np.isfinite(x) & (x <= 0.0) & (x >= -1.0)),
            # Biasi: water tubes; its G^(1/6) denominators diverge as G -> 0.
            "phi_biasi_logBo": (
                corr.biasi_chf(_col(df, "pressure_kPa").values, G, D_m, x),
                has_flow & np.isfinite(D_m)
                & (df["fluid"].astype(str).str.lower() == "water").values),
        }
        stack = []
        for name, (q_w_m2, valid) in candidates.items():
            log_bo = np.log(np.maximum(q_w_m2, 1e-30) / np.maximum(Ghfg, 1e-30))
            log_bo = np.where(valid & np.isfinite(log_bo), log_bo, np.nan)
            # Physically, Bo spans roughly 1e-5 to 1 -- anything outside that
            # is a correlation failing, not a prediction.
            log_bo = np.where((log_bo > -12.0) & (log_bo < 0.5), log_bo, np.nan)
            df[name] = log_bo
            stack.append(log_bo)
        stack = np.vstack(stack)
        n_valid = np.sum(np.isfinite(stack), axis=0)
        # Disagreement between correlations is itself informative: it is high
        # exactly where the physics is least settled, which is where a learned
        # correction is most needed and least trustworthy. Undefined unless at
        # least two correlations are actually in range.
        spread = np.nanmax(stack, axis=0) - np.nanmin(stack, axis=0)
        df["phi_corr_spread"] = np.where(n_valid >= 2, spread, np.nan)

    # Any non-finite group becomes NaN rather than +/-inf: inf poisons
    # scalers and silently becomes a huge finite number after imputation.
    for c in DIMENSIONLESS_COLUMNS + CORRELATION_COLUMNS + SURFACE_COLUMNS:
        v = pd.to_numeric(df[c], errors="coerce").values.astype(float)
        df[c] = np.where(np.isfinite(v), v, np.nan)

    return df

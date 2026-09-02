"""
physics/constraints.py
----------------------
The physics-consistency scorecard: foundation doc section 8, turned into
executable tests.

Two kinds of check:

  * PREDICTION-LEVEL -- evaluated on the actual test-set predictions.
    Cheap, and applicable retroactively to models already trained.
        C1  CHF > 0
        S2  dimensionless K in [0.119, 0.157] for saturated pool boiling

  * PROBE-LEVEL -- the model is queried on synthetic sweeps it was never
    trained on, to see whether it obeys the physics OUT of sample. This is
    the part that a leaderboard R^2 cannot tell you.
        C3  dCHF/dx <= 0        (quality)
        C4  dCHF/dG >= 0        (mass flux, flow boiling)
        C6  CHF -> 0 as P -> P_crit
        S1  pool-boiling CHF vs pressure peaks near P/P_c ~ 0.35
        S6  diameter effect reverses sign at low mass flux
        S7  helical-coil CHF crosses straight-tube CHF as quality rises

The argument this scorecard exists to support (foundation doc 9.5): a model
at R^2 = 0.95 that violates C3 and misses S1 is worse than one at R^2 = 0.85
that satisfies both, because only the latter can be trusted at a condition
nobody has tested.
"""
import numpy as np
import pandas as pd

from . import correlations as corr

ZUBER_K_LOW, ZUBER_K_HIGH = 0.119, 0.157  # foundation doc 3.2 / constraint S2


# ---------------------------------------------------------------------------
# Prediction-level checks
# ---------------------------------------------------------------------------

def score_predictions(y_pred, df_test) -> dict:
    """C1 and S2, evaluated directly on a model's test-set output."""
    y = np.asarray(y_pred, dtype=float)
    finite = np.isfinite(y)
    out = {
        "C1_nonpositive_frac": float(np.mean(~(y > 0) | ~finite)) if len(y) else np.nan,
        "C1_nonfinite_frac": float(np.mean(~finite)) if len(y) else np.nan,
    }

    # S2: is the implied dimensionless K physically plausible for pool rows?
    is_pool = df_test["geometry_family"].isin(
        ["pin_fin_pool_boiling", "flat_heater_pool_boiling"]).values
    if is_pool.any() and {"rho_l_sat", "rho_g_sat", "sigma_sat", "h_fg_sat"} <= set(df_test.columns):
        scale = corr.kutateladze_scale(
            df_test["rho_l_sat"].values[is_pool], df_test["rho_g_sat"].values[is_pool],
            df_test["sigma_sat"].values[is_pool], df_test["h_fg_sat"].values[is_pool])
        with np.errstate(divide="ignore", invalid="ignore"):
            K = y[is_pool] * 1000.0 / scale  # kW/m^2 -> W/m^2
        ok = np.isfinite(K)
        out["S2_pool_K_median"] = float(np.median(K[ok])) if ok.any() else np.nan
        out["S2_pool_K_in_band_frac"] = (
            float(np.mean((K[ok] >= ZUBER_K_LOW) & (K[ok] <= ZUBER_K_HIGH))) if ok.any() else np.nan)
    else:
        out["S2_pool_K_median"] = np.nan
        out["S2_pool_K_in_band_frac"] = np.nan

    return out


# ---------------------------------------------------------------------------
# Probe construction
# ---------------------------------------------------------------------------

def _base_row(template: pd.DataFrame, geometry_family="tube", fluid="water") -> dict:
    """A single physically-typical row, used as the stem for every sweep.

    Values are taken as medians of the real tube data so the probe sits inside
    the region the model was actually trained on -- the point is to test the
    SHAPE of the response, not to extrapolate wildly.
    """
    tube = template[(template["geometry_family"] == geometry_family)
                    & (template["fluid"] == fluid)]
    if len(tube) == 0:
        tube = template
    med = lambda c, d: (float(pd.to_numeric(tube[c], errors="coerce").median())
                        if c in tube.columns and pd.to_numeric(tube[c], errors="coerce").notna().any()
                        else d)
    row = {c: np.nan for c in template.columns}
    row.update({
        "geometry_family": geometry_family,
        "fluid": fluid,
        "data_type": "probe",
        "source_dataset": "probe",
        "pressure_kPa": med("pressure_kPa", 7000.0),
        "mass_flux_kg_m2s": med("mass_flux_kg_m2s", 2000.0),
        "quality": 0.2,
        "diameter_mm": med("diameter_mm", 8.0),
        "heated_length_mm": med("heated_length_mm", 1000.0),
        "subcooling_kJkg": med("subcooling_kJkg", 200.0),
    })
    return row


def make_sweep(template: pd.DataFrame, varying: str, values, **overrides) -> pd.DataFrame:
    """Build a synthetic sweep frame varying one parameter."""
    base = _base_row(template, overrides.pop("geometry_family", "tube"),
                     overrides.pop("fluid", "water"))
    base.update(overrides)
    rows = []
    for v in values:
        r = dict(base)
        r[varying] = v
        rows.append(r)
    out = pd.DataFrame(rows, columns=template.columns)
    out["row_id"] = [f"probe_{varying}_{i}" for i in range(len(out))]
    return out


def _monotone_violation(y, increasing: bool, tol=1e-9):
    """Fraction of adjacent steps going the wrong way."""
    y = np.asarray(y, dtype=float)
    ok = np.isfinite(y)
    if ok.sum() < 2:
        return np.nan
    d = np.diff(y[ok])
    bad = (d < -tol) if increasing else (d > tol)
    return float(np.mean(bad))


# ---------------------------------------------------------------------------
# Probe-level checks
# ---------------------------------------------------------------------------

def probe_model(predict_fn, template: pd.DataFrame) -> dict:
    """Run every out-of-sample physics probe against `predict_fn`.

    `predict_fn(df) -> np.ndarray` must accept a frame with the master schema
    and return CHF in kW/m^2.
    """
    out = {}

    def safe(fn, *keys):
        try:
            return fn()
        except Exception as exc:  # a probe must never abort a whole run
            for k in keys:
                out[k] = np.nan
            out.setdefault("probe_errors", "")
            out["probe_errors"] += f"{keys[0]}:{type(exc).__name__} "
            return None

    # --- C3: dCHF/dx <= 0 -------------------------------------------------
    def c3():
        sweep = make_sweep(template, "quality", np.linspace(-0.3, 0.9, 25))
        out["C3_quality_violation_frac"] = _monotone_violation(predict_fn(sweep), increasing=False)
    safe(c3, "C3_quality_violation_frac")

    # --- C4: dCHF/dG >= 0 -------------------------------------------------
    def c4():
        sweep = make_sweep(template, "mass_flux_kg_m2s", np.linspace(200, 6000, 25))
        out["C4_massflux_violation_frac"] = _monotone_violation(predict_fn(sweep), increasing=True)
    safe(c4, "C4_massflux_violation_frac")

    # --- C6: CHF -> 0 as P -> P_crit --------------------------------------
    # Water P_crit = 22,064 kPa. A model obeying C6 must predict a LOWER CHF
    # at 21,500 kPa than at 10,000 kPa.
    def c6():
        sweep = make_sweep(template, "pressure_kPa", [10000.0, 21500.0])
        y = predict_fn(sweep)
        out["C6_nearcrit_ratio"] = float(y[1] / y[0]) if np.isfinite(y).all() and y[0] > 0 else np.nan
        out["C6_satisfied"] = bool(out["C6_nearcrit_ratio"] < 1.0) if np.isfinite(
            out["C6_nearcrit_ratio"]) else False
    safe(c6, "C6_nearcrit_ratio", "C6_satisfied")

    # --- S1: pool-boiling pressure maximum near P/P_c ~ 0.35 --------------
    def s1():
        p_crit_kpa = 22064.0
        pr = np.linspace(0.02, 0.9, 40)
        sweep = make_sweep(template, "pressure_kPa", pr * p_crit_kpa,
                           geometry_family="flat_heater_pool_boiling", fluid="water")
        sweep["mass_flux_kg_m2s"] = 0.0
        y = np.asarray(predict_fn(sweep), dtype=float)
        if np.isfinite(y).sum() < 5:
            out["S1_peak_reduced_pressure"] = np.nan
        else:
            out["S1_peak_reduced_pressure"] = float(pr[int(np.nanargmax(y))])
    safe(s1, "S1_peak_reduced_pressure")

    # --- S6: diameter effect reverses sign with mass flux -----------------
    # The expected direction is taken FROM the Tanase table rather than
    # asserted: the LUT correction is K1 = (8/D)^n, so
    #
    #       d ln CHF / d ln D  =  -n
    #
    # Tanase gives n < 0 at G < 250 (=> slope POSITIVE, CHF rises with
    # diameter) and n > 0 at higher flux (=> slope NEGATIVE). A model that has
    # learned a single global diameter trend shows the same sign at both.
    def s6():
        diam = np.array([4.0, 8.0, 16.0])
        for tag, g in (("lowG", 150.0), ("highG", 3000.0)):
            sweep = make_sweep(template, "diameter_mm", diam)
            sweep["mass_flux_kg_m2s"] = g
            y = np.asarray(predict_fn(sweep), dtype=float)
            slope = np.nan
            if np.isfinite(y).all() and (y > 0).all():
                slope = float(np.polyfit(np.log(diam), np.log(y), 1)[0])
            out[f"S6_dlnCHF_dlnD_{tag}"] = slope
            # expected slope = -n at these conditions
            n_exp = corr.tanase_diameter_exponent(
                np.full(3, float(sweep["pressure_kPa"].iloc[0])),
                np.full(3, g), np.full(3, float(sweep["quality"].iloc[0])))
            out[f"S6_expected_slope_{tag}"] = float(-np.median(n_exp))
        lo, hi = out.get("S6_dlnCHF_dlnD_lowG"), out.get("S6_dlnCHF_dlnD_highG")
        elo, ehi = out.get("S6_expected_slope_lowG"), out.get("S6_expected_slope_highG")
        out["S6_sign_reversal_captured"] = bool(
            np.isfinite(lo) and np.isfinite(hi) and np.isfinite(elo) and np.isfinite(ehi)
            and np.sign(lo) == np.sign(elo) and np.sign(hi) == np.sign(ehi))
    safe(s6, "S6_dlnCHF_dlnD_lowG", "S6_dlnCHF_dlnD_highG", "S6_sign_reversal_captured")

    # --- S7: helical-coil / straight-tube crossover -----------------------
    # Hardik & Prabhu: coil CHF is BELOW a straight tube at low quality and
    # ABOVE it at high quality, so the coil/tube ratio must RISE with x.
    #
    # Both arms must sit at the SAME operating point or the ratio is
    # meaningless -- so one base row is built and only `geometry_family` is
    # switched. (An earlier version built each arm from its own family/fluid
    # medians, and since the merged table has no r123 tubes, the tube arm
    # silently fell back to global medians and the comparison was invalid.)
    def s7():
        qual = np.array([0.05, 0.5])
        coil = make_sweep(template, "quality", qual,
                          geometry_family="helical_coil", fluid="r123")
        tube = coil.copy()
        tube["geometry_family"] = "tube"
        tube["row_id"] = [f"probe_tube_{i}" for i in range(len(tube))]
        yc = np.asarray(predict_fn(coil), dtype=float)
        yt = np.asarray(predict_fn(tube), dtype=float)
        if np.isfinite(yc).all() and np.isfinite(yt).all() and (yt > 0).all():
            r_lo, r_hi = yc[0] / yt[0], yc[1] / yt[1]
            out["S7_coil_ratio_lowx"] = float(r_lo)
            out["S7_coil_ratio_highx"] = float(r_hi)
            out["S7_crossover_captured"] = bool(r_hi > r_lo)
        else:
            out["S7_coil_ratio_lowx"] = out["S7_coil_ratio_highx"] = np.nan
            out["S7_crossover_captured"] = False
    safe(s7, "S7_coil_ratio_lowx", "S7_coil_ratio_highx", "S7_crossover_captured")

    return out


def scorecard(y_pred, df_test, predict_fn=None, template=None) -> dict:
    """Full scorecard. Probes are skipped when no callable model is supplied."""
    out = score_predictions(y_pred, df_test)
    if predict_fn is not None and template is not None:
        out.update(probe_model(predict_fn, template))
    return out

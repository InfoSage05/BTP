"""
physics/correlations.py
-----------------------
Closed-form CHF correlations, transcribed from the sources catalogued in
`physics_foundation/CHF_Physics_Foundation.md`. Every function here has ZERO
fitted parameters -- all constants come from the literature. That is the
property that makes cross-fluid and cross-surface generalisation possible:
none of these need training data for the target fluid.

Provenance tags follow the foundation document:
    [V] transcribed verbatim from a primary source read during the research
    [S] structure verified; a constant needs checking against the original
    [X] not implemented -- no clean primary source obtained

All functions are vectorised over numpy arrays and take/return SI unless
stated. CHF is returned in W/m^2 throughout; conversion to the dataset's
kW/m^2 happens once, in baseline.py.
"""
import numpy as np

GRAVITY = 9.80665


def _safe(a, lo=1e-30):
    """Clamp to strictly positive so fractional powers stay real and finite."""
    return np.maximum(np.asarray(a, dtype=float), lo)


# ---------------------------------------------------------------------------
# Pool boiling
# ---------------------------------------------------------------------------

#: Zuber's recommended dimensionless CHF. The derivation brackets K between
#: 0.119 and 0.157 (from lambda_c < lambda_T < lambda_d); 0.131 is the
#: intermediate value Zuber recommended. Foundation doc section 3.2. [V]
ZUBER_K = 0.131

#: Kutateladze's dimensional-analysis value for a large horizontal flat
#: surface. Foundation doc section 2.1. [V]
KUTATELADZE_K = 0.16

#: Lienhard & Dhir: setting lambda_H = lambda_d instead of Zuber's pi*D_j
#: raises CHF by 14%. Foundation doc section 3.3. [V]
LIENHARD_DHIR_RATIO = 1.14


def kutateladze_scale(rho_l, rho_g, sigma, h_fg):
    """The Kutateladze-Zuber hydrodynamic CHF scale with K factored out:

        q''_CHF = K * rho_g * h_fg * [sigma * g * (rho_l - rho_g) / rho_g^2]^(1/4)

    Returning the scale separately from K is deliberate -- every pool-boiling
    model in the literature (Zuber, Lienhard-Dhir, Kandlikar, Kim, Haramura-
    Katto, Theofanous-Dinh) can be written as this same scale times a
    different K. Foundation doc section 2.1, eq (4). [V]
    """
    rho_g = _safe(rho_g)
    drho = _safe(np.asarray(rho_l, dtype=float) - np.asarray(rho_g, dtype=float))
    return rho_g * np.asarray(h_fg, dtype=float) * (
        _safe(sigma) * GRAVITY * drho / rho_g ** 2) ** 0.25


def zuber_chf(rho_l, rho_g, sigma, h_fg, K=ZUBER_K):
    """Zuber (1959) hydrodynamic instability CHF, eq (15). [V]"""
    return K * kutateladze_scale(rho_l, rho_g, sigma, h_fg)


def capillary_length(rho_l, rho_g, sigma):
    """delta* = [sigma / (g (rho_l - rho_g))]^(1/2). Foundation doc eq (3). [V]"""
    return np.sqrt(_safe(sigma) / (GRAVITY * _safe(np.asarray(rho_l) - np.asarray(rho_g))))


def taylor_wavelengths(rho_l, rho_g, sigma):
    """Rayleigh-Taylor critical and most-dangerous wavelengths, eqs (7a)/(7b). [V]

        lambda_c = 2 pi sqrt(sigma / (g (rho_l - rho_g)))
        lambda_d = sqrt(3) * lambda_c
    """
    lam_c = 2.0 * np.pi * capillary_length(rho_l, rho_g, sigma)
    return lam_c, np.sqrt(3.0) * lam_c


def kandlikar_K(contact_angle_deg, orientation_deg):
    """Kandlikar (2001) force-balance dimensionless CHF, valid theta = 0-90 deg.
    Foundation doc section 4.3, Liang & Mudawar Table 2. [V]

        K = [(1 + cos a)/16] * [2/pi + (pi/4)(1 + cos a) cos(theta)]^(1/2)

    `a` is the receding contact angle, `theta` the surface orientation
    (0 deg = horizontal upward-facing, 90 deg = vertical).

    The bracket can go negative for theta > 90 deg, where this model is not
    valid anyway; those rows return NaN rather than a complex number, and the
    caller is expected to fall back to an orientation correlation with a wider
    range (see `vishnev_K`).
    """
    a = np.deg2rad(np.asarray(contact_angle_deg, dtype=float))
    th = np.deg2rad(np.asarray(orientation_deg, dtype=float))
    inner = 2.0 / np.pi + (np.pi / 4.0) * (1.0 + np.cos(a)) * np.cos(th)
    K = ((1.0 + np.cos(a)) / 16.0) * np.sqrt(np.where(inner > 0, inner, np.nan))
    return K


def vishnev_K(orientation_deg):
    """Vishnev orientation correlation, valid theta = 0-180 deg. [V]

        K = 0.0125 (190 - theta)^(1/2)

    Used as the fallback where Kandlikar's 0-90 deg range does not reach --
    notably the 180 deg (downward-facing) rows in `mentor_master`.
    """
    th = np.asarray(orientation_deg, dtype=float)
    return 0.0125 * np.sqrt(_safe(190.0 - th))


def kim_roughness_K(contact_angle_deg, Ra, Sm):
    """Kim et al. combined wettability + roughness dimensionless CHF, eq (21). [V]

        K = 0.811 { (1+cos a)/16 [ 2/pi + (pi/4)(1+cos a)
                                   + 351.2 (cos a/(1+cos a)) (Ra/Sm) ] }^(1/2)

    Needs BOTH roughness amplitude Ra and mean peak spacing Sm. The merged
    dataset carries neither (it has a Wenzel-type `roughness_factor` ratio
    instead), so this is implemented for completeness and for the
    surface sub-study, not used in the main ablation. See foundation doc
    section 4.2 / constraint D2.
    """
    a = np.deg2rad(np.asarray(contact_angle_deg, dtype=float))
    cos_a = np.cos(a)
    ratio = np.asarray(Ra, dtype=float) / _safe(Sm)
    inner = (2.0 / np.pi + (np.pi / 4.0) * (1.0 + cos_a)
             + 351.2 * cos_a / _safe(1.0 + cos_a) * ratio)
    return 0.811 * np.sqrt(_safe((1.0 + cos_a) / 16.0 * inner))


def lienhard_dhir_size_factor(heater_width_m, rho_l, rho_g, sigma):
    """Finite-heater correction. Foundation doc section 3.3 / constraint S8. [V]

    The infinite-flat-plate assumption behind Zuber holds for heater widths
    greater than 3*lambda_d. Below that, Zuber is the wrong normaliser.

    Sources disagree on the exact threshold (Lienhard: 3 lambda_d; Zhang:
    12 capillary lengths; Gogonin: 2 capillary lengths) and none of them give
    a continuous sub-threshold correction. Rather than invent one, this
    returns the Lienhard-Dhir 1.14 enhancement for heaters comfortably above
    threshold and tapers smoothly toward 1.0 below it, which is a modelling
    CHOICE, not a transcribed result -- flagged [S]. Rows with no heater
    width (the pin-fin source reports none) return 1.0, i.e. plain Zuber.
    """
    w = np.asarray(heater_width_m, dtype=float)
    _, lam_d = taylor_wavelengths(rho_l, rho_g, sigma)
    ratio = w / _safe(3.0 * lam_d)
    factor = 1.0 + (LIENHARD_DHIR_RATIO - 1.0) * np.clip(ratio, 0.0, 1.0)
    return np.where(np.isfinite(ratio), factor, 1.0)


# ---------------------------------------------------------------------------
# Flow boiling
# ---------------------------------------------------------------------------

# --- Katto & Ohno (1984) ----------------------------------------------------
#
# Foundation doc section 5.5, transcribed from NASA NTRS 20230009827. [V]
# for every constant except C_Kc.
#
# C_Kc is the Katto-Ohno length-dependent constant appearing in q_co,2 and
# K6. It did not extract cleanly from any source obtained during the
# research, and the primary paper (Katto & Ohno 1984, IJHMT 27(9):1641) was
# not accessible. The piecewise form below is the one commonly reproduced in
# secondary literature and is marked [S] -- UNVERIFIED.
#
# Because q_co is a min() over three branches and K a max() over two or three,
# C_Kc only matters where its branch is actually selected. `katto_ohno_chf`
# can report branch-activation counts so that sensitivity is measured rather
# than assumed -- see scripts/validate_physics.py.

KATTO_C_SMALL = 0.25   # [S] UNVERIFIED -- L/D < 50
KATTO_C_LARGE = 0.34   # [S] UNVERIFIED -- L/D > 150


def katto_C(L_over_D):
    """[S] UNVERIFIED. Piecewise L/D-dependent Katto-Ohno constant."""
    r = np.asarray(L_over_D, dtype=float)
    c = np.where(
        r < 50.0, KATTO_C_SMALL,
        np.where(r > 150.0, KATTO_C_LARGE,
                 KATTO_C_SMALL + (KATTO_C_LARGE - KATTO_C_SMALL) * (r - 50.0) / 100.0))
    return c


def katto_ohno_chf(G, D_h, L, rho_l, rho_g, sigma, h_fg, delta_h_sub,
                   return_branches=False):
    """Katto & Ohno (1984) generalised flow-boiling CHF. Foundation doc 5.5. [V]

    Parameters are SI: G [kg/m^2 s], D_h and L [m], densities [kg/m^3],
    sigma [N/m], h_fg [J/kg], delta_h_sub = (h_f,sat - h_in) [J/kg] >= 0.

    Returns q''_CHF in W/m^2.

        q''_CHF = q''_co * [1 + K * delta_h_sub / h_fg]                    (8)

    with the regime gate at rho_g/rho_f = 0.15 (eqs 17, 18) -- a physically
    motivated boundary where the controlling mechanism changes, not a tuned
    threshold.
    """
    G = _safe(np.asarray(G, dtype=float), 1e-6)
    D_h = _safe(np.asarray(D_h, dtype=float), 1e-9)
    L = _safe(np.asarray(L, dtype=float), 1e-9)
    rho_l = _safe(rho_l)
    rho_g = _safe(rho_g)
    h_fg = _safe(h_fg)
    sigma = _safe(sigma)

    # We_inv = sigma rho_f / (G^2 L) -- the "Katto number" of foundation doc 2.2
    We_inv = sigma * rho_l / (G ** 2 * L)
    dr = rho_g / rho_l          # density ratio
    LD = L / D_h                # heated length over hydraulic diameter
    Ghfg = G * h_fg
    C = katto_C(LD)

    q2 = C * We_inv ** 0.043 * (D_h / L) * Ghfg                                     # (9)
    q3 = 0.1 * dr ** 0.133 * We_inv ** 0.333 * Ghfg / (1.0 + 0.0031 * LD)           # (10)
    q4 = (0.098 * dr ** 0.133 * We_inv ** 0.433 * LD ** 0.27 * Ghfg
          / (1.0 + 0.0031 * LD))                                                     # (11)
    q5 = (0.0384 * dr ** 0.6 * We_inv ** 0.173 * Ghfg
          / (1.0 + 0.28 * We_inv ** 0.233 * LD))                                     # (12)
    q13 = (0.234 * dr ** 0.513 * We_inv ** 0.433 * LD ** 0.27 * Ghfg
           / (1.0 + 0.0031 * LD))                                                    # (13)

    K6 = 1.043 / (4.0 * C * We_inv ** 0.043)                                         # (14)
    K7 = (5.0 / 6.0) * (0.0124 + D_h / L) / (dr ** 0.133 * We_inv ** 0.333)          # (15)
    K9 = 1.12 * (1.52 * We_inv ** 0.233 + D_h / L) / (dr ** 0.6 * We_inv ** 0.173)   # (16)

    low = dr < 0.15
    q_co = np.where(low, np.minimum(np.minimum(q2, q3), q4),
                          np.minimum(np.minimum(q2, q5), q13))                       # (17)
    K = np.where(low, np.maximum(K6, K7),
                       np.maximum(np.maximum(K6, K7), K9))                           # (18)

    chf = q_co * (1.0 + K * np.asarray(delta_h_sub, dtype=float) / h_fg)             # (8)

    if not return_branches:
        return chf

    # Which branch supplied the min? Used to measure C_Kc sensitivity.
    stack_low = np.stack([q2, q3, q4])
    stack_high = np.stack([q2, q5, q13])
    branch = np.where(low, np.argmin(stack_low, axis=0), np.argmin(stack_high, axis=0))
    # branch == 0 means q_co,2 was selected, i.e. C_Kc mattered for q_co.
    return chf, branch, low


# --- Hall & Mudawar (2000) --------------------------------------------------
#
# Foundation doc section 5.7. Fully dimensionless, five constants, functional
# form derived from observed parametric trends. [V]
#
# Validity is SUBCOOLED OUTLET (-1.0 <= x_o <= 0.0). Only ~13% of the merged
# dataset satisfies that; it is the DNB-branch scale, not the general one.

HM_C1, HM_C2, HM_C3, HM_C4, HM_C5 = 0.0332, -0.235, -0.681, 0.684, 0.832


def hall_mudawar_chf(G, D, x_out, rho_l, rho_g, sigma, h_fg):
    """Hall & Mudawar outlet-conditions correlation, eq (7). [V]

        Bo = C1 We_D^C2 (rho_f/rho_g)^C3 [1 - C4 (rho_f/rho_g)^C5 x_o]

    with Bo = q''/(G h_fg) and We_D = G^2 D / (rho_f sigma).
    Returns q''_CHF in W/m^2.
    """
    G = _safe(np.asarray(G, dtype=float), 1e-6)
    rho_l = _safe(rho_l)
    rho_g = _safe(rho_g)
    We_D = G ** 2 * _safe(D, 1e-9) / (rho_l * _safe(sigma))
    dr = rho_l / rho_g
    Bo = HM_C1 * We_D ** HM_C2 * dr ** HM_C3 * (
        1.0 - HM_C4 * dr ** HM_C5 * np.asarray(x_out, dtype=float))
    return Bo * G * _safe(h_fg)


# --- Biasi et al. (1967) ----------------------------------------------------

def biasi_chf(P_kpa, G, D, x):
    """Biasi correlation as implemented in scripts/chf_physics.py. [V]

    Units are the original paper's: D in cm, G in g/(cm^2 s), P in atm.
    Inputs here are SI (P in kPa, G in kg/m^2 s, D in m); conversion is
    internal. Returns W/m^2.

    Retained as a comparison baseline only. Note the G^(1/6) denominators
    DIVERGE as G -> 0, violating constraint C5 (the pool-boiling limit) --
    which is exactly why baseline.py does not use it.
    """
    P_atm = np.asarray(P_kpa, dtype=float) / 101.325
    G_gcm2s = _safe(np.asarray(G, dtype=float) / 10.0, 1e-6)
    D_cm = _safe(np.asarray(D, dtype=float) * 100.0, 1e-6)
    x = np.asarray(x, dtype=float)

    alpha = np.where(D_cm < 1.0, 0.6, 0.4)
    A = 0.7249 + 0.099 * P_atm * np.exp(-0.032 * P_atm)
    B = -1.159 + 0.149 * P_atm * np.exp(-0.019 * P_atm) + 8.99 * P_atm / (10.0 + P_atm ** 2)

    q_low = (1.883e4 / (D_cm ** alpha * G_gcm2s ** (1.0 / 6.0))) * (A / G_gcm2s ** (1.0 / 6.0) - x)
    q_high = (3.78e4 * B / (D_cm ** alpha * G_gcm2s ** 0.6)) * (1.0 - x)
    return np.maximum(np.maximum(q_low, q_high), 1.0) * 1000.0  # kW/m^2 -> W/m^2


# ---------------------------------------------------------------------------
# Geometry corrections
# ---------------------------------------------------------------------------

LUT_REFERENCE_DIAMETER_MM = 8.0

# Tanase et al. (2009) Table 3: the LUT diameter exponent n by (P, G, x)
# region. Foundation doc section 5.3. [V]
# Note the NEGATIVE values at low mass flux -- the diameter trend reverses,
# which is constraint S6. Entries are (P_kPa_max, G_max, [n for the four
# quality bands: x<-0.25, -0.25..0, 0..0.5, >0.5]).
_TANASE_N_TABLE = [
    (14000.0, 250.0, [-0.2, -0.2, -0.2, -0.3]),
    (14000.0, 3000.0, [0.4, 0.4, 0.5, 0.6]),
    (14000.0, np.inf, [0.3, 0.3, 0.4, 0.4]),
    (np.inf, 250.0, [-0.2, -0.2, -0.2, -0.3]),
    (np.inf, 3000.0, [0.4, 0.2, 0.4, 0.4]),
    (np.inf, np.inf, [0.3, 0.2, 0.2, 0.2]),
]

TANASE_N_DEFAULT = 0.5  # Groeneveld's original, used where P/G/x are missing


def tanase_diameter_exponent(P_kpa, G, x):
    """Regime-dependent LUT diameter exponent n. Foundation doc 5.3, Table 3. [V]

    A tree model cannot discover a sign reversal it never observes in a
    held-out geometry. This table hands it over directly.
    """
    P = np.asarray(P_kpa, dtype=float)
    G = np.asarray(G, dtype=float)
    x = np.asarray(x, dtype=float)

    band = np.select(
        [x < -0.25, x < 0.0, x < 0.5], [0, 1, 2], default=3)
    n = np.full(P.shape, TANASE_N_DEFAULT, dtype=float)

    filled = np.zeros(P.shape, dtype=bool)
    for p_max, g_max, n_by_band in _TANASE_N_TABLE:
        sel = (~filled) & (P <= p_max) & (G <= g_max) & np.isfinite(P) & np.isfinite(G)
        if not sel.any():
            continue
        vals = np.asarray(n_by_band)[band]
        n = np.where(sel, vals, n)
        filled |= sel

    return np.where(np.isfinite(P) & np.isfinite(G) & np.isfinite(x), n, TANASE_N_DEFAULT)


def lut_diameter_factor_K1(D_mm, n):
    """K1 = (8/D_h)^n for 2 <= D_h <= 25 mm, 0.57 above. Foundation doc 5.3, eq (9). [V]

    Below 2 mm the source gives no guidance; the formula is applied but the
    diameter is floored at 2 mm so a sub-millimetre tube cannot produce an
    unbounded factor. That floor is a modelling choice, flagged [S].
    """
    D = np.asarray(D_mm, dtype=float)
    D_clamped = np.clip(D, 2.0, 25.0)
    K1 = (LUT_REFERENCE_DIAMETER_MM / D_clamped) ** np.asarray(n, dtype=float)
    K1 = np.where(D > 25.0, 0.57, K1)
    return np.where(np.isfinite(D), K1, 1.0)


def helical_coil_factor(x):
    """Hardik & Prabhu (2017) LUT-normalised helical coil transfer function. [V]

        CHF_coil / CHF_LUT = 1.637 x + 0.568                                (11)

    Physically explained: at low quality gravity dominates and coil CHF is
    below a vertical tube; at high quality, secondary-flow liquid redeposition
    dominates and coil CHF exceeds it. That crossover is constraint S7.

    Reported fit quality: 90% of data within 25%, RMS 17%, mean -1.6%.
    Floored at a small positive value so the factor can never drive CHF
    negative at strongly subcooled conditions (constraint C1); the source
    only covers x = 0 to 1.
    """
    return np.maximum(1.637 * np.asarray(x, dtype=float) + 0.568, 0.05)


# ---------------------------------------------------------------------------
# Regime blending
# ---------------------------------------------------------------------------

def yagov_blend(q_a, q_b, exponent=3.0):
    """Smooth asymptotic blend of two regime scales. Foundation doc 3.5, eq (36). [V]

        q = (q_a^-p + q_b^-p)^(-1/p)

    Yagov used p = 3 to join his low- and high-reduced-pressure dry-spot
    branches. Reproduced here in harmonic form so the blend approaches
    min(q_a, q_b) smoothly: differentiable everywhere, no discontinuity at the
    regime boundary, unlike the hard `if` switches inside Biasi and
    Katto-Ohno.
    """
    a = _safe(q_a)
    b = _safe(q_b)
    return (a ** (-exponent) + b ** (-exponent)) ** (-1.0 / exponent)

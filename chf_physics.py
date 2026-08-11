"""
chf_physics.py
---------------
Closed-form empirical/semi-analytical CHF correlations used as physics priors
for the physics-informed extensions notebook (CHF_Physics_Informed_Extensions.ipynb).

Deliberately does NOT include the Groeneveld correlation family, since the
Groeneveld 2006 look-up table (our training data) was itself partly built
from Groeneveld correlations -- using it as a "physics prior" here would leak
information about the very data we're trying to generalize beyond (see the
literature caveat in the notebook's markdown). Biasi, Bowring and Zuber are
independently-fit correlations and are safe to use as priors.

All functions take numpy arrays and work in the SAME units as chf_long_clean.csv:
    P   pressure, kPa
    G   mass flux, kg / (m^2 s)
    X   thermodynamic quality, dimensionless
    CHF kW / m^2
Internally each correlation converts to whatever units its original paper used.
"""
import numpy as np

# Water critical pressure, kPa (used for reduced-pressure terms)
P_CRIT_KPA = 22060.0


def biasi_chf(P_kpa, G_kgm2s, X, D_m=0.008):
    """
    Biasi et al. (1967) CHF correlation for water, tubes:
        CHF = max[ 1.883e4 / (D^alpha * G^(1/6)) * (A/G^(1/6) - X),
                   3.78e4 * B / (D^alpha * G^0.6) * (1 - X) ]   [kW/m^2]
    with D in cm, G in g/(cm^2 s), P in atm, alpha=0.6 for D<1cm else 0.4
    (Todreas & Kazimi, "Nuclear Systems I"; as used e.g. in Furlong et al. 2025,
    arXiv:2502.19357, as a base correlation for hybrid CHF ML models).

    D_m: hydraulic diameter in meters (default 8 mm, a typical rod-bundle-scale
         value; the Groeneveld LUT itself is diameter-normalized to 8 mm).
    Returns CHF in kW/m^2. Vectorized over numpy arrays.
    """
    P_atm = np.asarray(P_kpa, dtype=float) / 101.325
    G_gcm2s = np.asarray(G_kgm2s, dtype=float) / 10.0  # kg/m^2/s -> g/cm^2/s
    D_cm = D_m * 100.0
    X = np.asarray(X, dtype=float)

    alpha = 0.6 if D_cm < 1.0 else 0.4
    A = 0.7249 + 0.099 * P_atm * np.exp(-0.032 * P_atm)
    B = -1.159 + 0.149 * P_atm * np.exp(-0.019 * P_atm) + 8.99 * P_atm / (10.0 + P_atm ** 2)

    G_safe = np.maximum(np.abs(G_gcm2s), 1e-6)
    q_low = (1.883e4 / (D_cm ** alpha * G_safe ** (1.0 / 6.0))) * (A / G_safe ** (1.0 / 6.0) - X)
    q_high = (3.78e4 * B / (D_cm ** alpha * G_safe ** 0.6)) * (1.0 - X)

    q_kwm2 = np.maximum(q_low, q_high)
    return np.maximum(q_kwm2, 1.0)  # CHF is physically positive; floor tiny/negative values


_ZUBER_CACHE = {}


def zuber_pool_boiling_chf(P_kpa):
    """
    Zuber (1959) pool-boiling CHF correlation. This is a PRESSURE-ONLY
    correlation (no G, X dependence) -- used here purely as a pressure-trend
    reference, exactly as the Groeneveld 2006 LUT's own authors used it to
    build the table's G=0 skeleton and to extrapolate to the 21 MPa pressure
    level (see the notebook's physics-literature markdown cell).

    Saturation properties (rho_l, rho_g, h_fg, sigma) come from CoolProp's
    IAPWS-based water equation of state -- real steam-table values, not a
    hand-fit approximation -- so the resulting pressure-trend peak (~6-7 MPa
    for water) matches the literature's reported range without any tuning.
    Returns CHF in kW/m^2. Results are cached per unique pressure value since
    CoolProp calls are comparatively expensive and P takes few unique values
    on this grid.
    """
    import CoolProp.CoolProp as CP

    P = np.atleast_1d(np.asarray(P_kpa, dtype=float))
    q_kwm2 = np.empty_like(P)
    K = 0.131  # Zuber constant (Ivey-Morris low-velocity limit)
    g = 9.81
    for i, p in enumerate(P):
        p_clamped = float(np.clip(p, 10.0, 21900.0))  # stay just below P_crit=22064 kPa
        if p_clamped not in _ZUBER_CACHE:
            p_pa = p_clamped * 1000.0
            rho_l = CP.PropsSI("D", "P", p_pa, "Q", 0, "Water")
            rho_g = CP.PropsSI("D", "P", p_pa, "Q", 1, "Water")
            h_fg = CP.PropsSI("H", "P", p_pa, "Q", 1, "Water") - CP.PropsSI("H", "P", p_pa, "Q", 0, "Water")
            sigma = CP.PropsSI("I", "P", p_pa, "Q", 0, "Water")
            q = K * h_fg * rho_g ** 0.5 * (sigma * g * max(rho_l - rho_g, 1e-6)) ** 0.25  # W/m^2
            _ZUBER_CACHE[p_clamped] = q / 1000.0  # -> kW/m^2
        q_kwm2[i] = _ZUBER_CACHE[p_clamped]
    return q_kwm2 if np.ndim(P_kpa) else q_kwm2[0]


def hybrid_reference_chf(P_kpa, G_kgm2s, X, D_m=0.008):
    """
    Regime-dispatching physics-prior reference: Zuber pool-boiling correlation
    at G=0, Biasi flow-boiling correlation at G>0. Mirrors exactly how the
    Groeneveld 2006 LUT's own skeleton table was constructed (Zuber+Ivey-Morris
    for G=0, a flow-boiling correlation elsewhere) -- see the notebook's
    literature markdown cell. Biasi's G^(1/6)-in-denominator terms diverge as
    G -> 0, so a G=0 special case is a correctness requirement, not a nicety.
    Returns CHF in kW/m^2.
    """
    P = np.atleast_1d(np.asarray(P_kpa, dtype=float))
    G = np.atleast_1d(np.asarray(G_kgm2s, dtype=float))
    X = np.atleast_1d(np.asarray(X, dtype=float))
    out = np.empty_like(P, dtype=float)
    pool_mask = G == 0
    if pool_mask.any():
        out[pool_mask] = zuber_pool_boiling_chf(P[pool_mask])
    if (~pool_mask).any():
        out[~pool_mask] = biasi_chf(P[~pool_mask], G[~pool_mask], X[~pool_mask], D_m=D_m)
    return out


def physics_basis_features(P_kpa, G_kgm2s, X):
    """
    Engineered feature set inspired by the functional forms appearing in the
    Zuber / Biasi / Katto correlations (power laws in G, (1-X), and a
    near-critical-pressure decay term). Meant to be concatenated with
    [P, G, X] before fitting a linear/ridge/polynomial model -- a cheap way
    to inject physically-motivated basis functions without a full residual-
    learning pipeline. Returns an (n, k) array of extra features.
    """
    P = np.asarray(P_kpa, dtype=float)
    G = np.asarray(G_kgm2s, dtype=float)
    X = np.asarray(X, dtype=float)
    Pr = np.clip(P / P_CRIT_KPA, 1e-4, 0.9999)

    G_safe = np.maximum(G, 1e-3)
    one_minus_x = 1.0 - X

    feats = np.column_stack([
        np.log1p(G_safe),                 # ~ log(G), Biasi/Katto power-law-in-G proxy
        np.sign(one_minus_x) * np.abs(one_minus_x) ** 0.6,   # Biasi high-quality branch shape, X<=1
        np.log(np.clip(1.0 - Pr, 1e-4, None)),   # near-critical decay term (Zuber h_fg, sigma -> 0 as P->Pc)
        Pr,                                 # reduced pressure (captures the non-monotonic P trend's location)
        Pr ** 2,                            # quadratic in Pr, to allow the pressure peak's curvature
        G_safe ** 0.5 * one_minus_x,        # a Biasi/Katto-style cross term
    ])
    return feats


FEATURE_NAMES = [
    "log1p_G", "sign_1mX_pow0p6", "log_1mPr", "Pr", "Pr_sq", "sqrtG_times_1mX",
]


if __name__ == "__main__":
    # Quick smoke test
    from pathlib import Path
    data_path = Path("data/chf_long_clean.csv")
    if not data_path.exists():
        data_path = Path("../data/chf_long_clean.csv")
    if not data_path.exists():
        data_path = Path("chf_long_clean.csv")
    df = pd.read_csv(data_path)
    df = df[df.X != 1.0]
    P, G, X, CHF = df.P.values, df.G.values, df.X.values, df.CHF.values

    flow_mask = G > 0
    biasi_pred = biasi_chf(P[flow_mask], G[flow_mask], X[flow_mask])
    print("Biasi CHF range (flow boiling only, G>0):", biasi_pred.min(), biasi_pred.max())
    print("Actual LUT CHF range (flow boiling only, G>0):", CHF[flow_mask].min(), CHF[flow_mask].max())
    print("Biasi vs LUT R^2 (flow boiling rows, no fitting -- raw correlation vs table):",
          r2_score(CHF[flow_mask], biasi_pred))

    hybrid_pred = hybrid_reference_chf(P, G, X)
    print("\nHybrid (Zuber@G=0 + Biasi@G>0) CHF range (all rows):", hybrid_pred.min(), hybrid_pred.max())
    print("Hybrid vs LUT R^2 (all rows, no fitting):", r2_score(CHF, hybrid_pred))

    p_scan = np.linspace(100, 21000, 500)
    z_scan = zuber_pool_boiling_chf(p_scan)
    print("\nZuber trend peak at P ~", p_scan[np.argmax(z_scan)], "kPa (expect a few MPa, per literature)")

    feats = physics_basis_features(P, G, X)
    print("physics_basis_features shape:", feats.shape, "names:", FEATURE_NAMES)

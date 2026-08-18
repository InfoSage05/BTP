"""
prepare_gridinterp_bases.py
----------------------------
Precomputes honest, leakage-free GridInterp (trilinear RegularGridInterpolator,
raw-target -- the variant that scores R^2=0.8415 on Split C, vs 0.8040 for the
log-target variant) base predictions for every row, to be used as the base
correlate for Tier 3's residual/multiplicative PINN correction
(PLAN.md Tier 3: "learn the PINN as a correction on top of GridInterp").

Why this needs care, not just "call fit_grid_interpolator and predict everywhere":
If GridInterp is fit on a P-range and then evaluated AT ITS OWN FITTING ROWS, the
prediction is (near-)exact by construction (those rows are grid nodes), so a
residual target log(CHF) - log(GridInterp_pred) would be trivially ~0 for every
training row -- the PINN would have nothing to learn. Base predictions used for
TRAINING/EARLY-STOPPING residual targets must therefore be genuinely out-of-sample,
via leave-one-pressure-level-out (LOPO): for each training P-level, fit GridInterp
on every OTHER training P-level and predict at the held-out level. This is cheap
(the grid interpolator is deterministic, CPU-only, ~15-20 refits) and gives every
training/validation row an honest "what would GridInterp predict here, had it not
seen this exact pressure slice" value.

Two independent phases, matching the proxy/final split protocol already
established in scripts/modal_pinn_tier1_search.py (PLAN.md Tier 1 fix):

  grid_base_proxy  -- for the PROXY phase (proxy_train = P<=14000, proxy_val = P in [15000,16000]):
                        * P<=14000 rows:            LOPO base within the P<=14000 subset.
                        * P in [15000,16000] rows:  GridInterp fit on ALL of P<=14000, extrapolated
                                                     forward (genuine out-of-sample, no leakage).
                        * elsewhere: NaN (unused by the proxy phase).

  grid_base_final  -- for the FINAL phase (fit_train = P<14000, fit_val = P in [14000,16000],
                       real_test = P>=17000):
                        * P<=16000 rows (covers both fit_train and fit_val): LOPO base within
                                                     the full P<=16000 subset.
                        * P>=17000 rows (real_test): GridInterp fit on ALL of P<=16000, extrapolated
                                                     forward -- this is exactly how the established
                                                     GridInterp(raw) R^2=0.8415 Split C baseline is
                                                     computed, so the residual model's final-phase
                                                     base matches that reference exactly.
                        * elsewhere: NaN (unused by the final phase).

Both base columns are clipped to a small positive floor (1.0 kW/m^2) before any
downstream log() is taken, since raw-target linear extrapolation can occasionally
undershoot below zero far from the fitted P-range.

Output: data/chf_long_with_gridbase.csv (P, G, X, CHF, grid_base_proxy, grid_base_final).
"""
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.interpolate import RegularGridInterpolator

DATA_PATH = Path("data/chf_long_clean.csv")
OUT_PATH = Path("data/chf_long_with_gridbase.csv")
FLOOR = 1.0  # kW/m^2, matches chf_physics.py's positivity floor convention


def fit_grid_interpolator(sub_df):
    """Raw-target trilinear RegularGridInterpolator, linear-extrapolating outside
    its fitted P/G/X range (bounds_error=False, fill_value=None -- see build_notebook.py's
    identical fit_grid_interpolator, which this mirrors so results stay comparable)."""
    p_u = np.sort(sub_df.P.unique())
    g_u = np.sort(sub_df.G.unique())
    x_u = np.sort(sub_df.X.unique())
    cube = np.full((len(p_u), len(g_u), len(x_u)), np.nan)
    p_idx = {v: i for i, v in enumerate(p_u)}
    g_idx = {v: i for i, v in enumerate(g_u)}
    x_idx = {v: i for i, v in enumerate(x_u)}
    for (p, g, x), v in zip(sub_df[["P", "G", "X"]].values, sub_df.CHF.values):
        cube[p_idx[p], g_idx[g], x_idx[x]] = v
    n_missing = int(np.isnan(cube).sum())
    if n_missing:
        raise ValueError(f"Grid has {n_missing} missing cells for P-levels {list(p_u)} -- "
                          f"not a complete rectilinear grid; LOPO requires every remaining "
                          f"P-level to still have full G,X coverage.")
    interp = RegularGridInterpolator((p_u, g_u, x_u), cube, method="linear",
                                      bounds_error=False, fill_value=None)
    return interp


def lopo_base(sub_df):
    """Leave-one-pressure-level-out base predictions for every row of sub_df."""
    out = np.full(len(sub_df), np.nan)
    p_levels = sorted(sub_df.P.unique())
    for p_level in p_levels:
        held_mask = (sub_df.P.values == p_level)
        fit_mask = ~held_mask
        fit_subset = sub_df[fit_mask]
        interp = fit_grid_interpolator(fit_subset)
        held_rows = sub_df.loc[held_mask, ["P", "G", "X"]].values
        out[held_mask] = interp(held_rows)
    return out


def extrapolated_base(fit_df, query_df):
    """GridInterp fit on fit_df, evaluated (extrapolated) at query_df's rows."""
    interp = fit_grid_interpolator(fit_df)
    return interp(query_df[["P", "G", "X"]].values)


def main():
    df = pd.read_csv(DATA_PATH)
    df = df[df.X != 1.0].reset_index(drop=True)

    grid_base_proxy = np.full(len(df), np.nan)
    grid_base_final = np.full(len(df), np.nan)

    # ---- Proxy phase ----
    le14k_mask = (df.P <= 14000).values
    proxyval_mask = ((df.P >= 15000) & (df.P <= 16000)).values

    print(f"LOPO base for proxy_train (P<=14000, n={le14k_mask.sum()}) "
          f"across {df.loc[le14k_mask, 'P'].nunique()} P-levels...")
    grid_base_proxy[le14k_mask] = lopo_base(df[le14k_mask])

    print(f"Extrapolated base for proxy_val (P in [15000,16000], n={proxyval_mask.sum()})...")
    grid_base_proxy[proxyval_mask] = extrapolated_base(df[le14k_mask], df[proxyval_mask])

    # ---- Final phase ----
    le16k_mask = (df.P <= 16000).values
    real_test_mask = (df.P >= 17000).values

    print(f"LOPO base for fit_train+fit_val (P<=16000, n={le16k_mask.sum()}) "
          f"across {df.loc[le16k_mask, 'P'].nunique()} P-levels...")
    grid_base_final[le16k_mask] = lopo_base(df[le16k_mask])

    print(f"Extrapolated base for real_test (P>=17000, n={real_test_mask.sum()}) "
          f"-- matches the established GridInterp(raw) R^2=0.8415 Split C baseline...")
    grid_base_final[real_test_mask] = extrapolated_base(df[le16k_mask], df[real_test_mask])

    df["grid_base_proxy"] = np.clip(grid_base_proxy, FLOOR, None)
    df["grid_base_final"] = np.clip(grid_base_final, FLOOR, None)

    # Sanity check: reproduce the established 0.8415 raw-R^2 baseline exactly.
    from sklearn.metrics import r2_score
    real_test_r2 = r2_score(df.loc[real_test_mask, "CHF"], df.loc[real_test_mask, "grid_base_final"])
    print(f"\nSanity check -- GridInterp(raw) R^2 on real Split C test set "
          f"(should match established 0.8415): {real_test_r2:.4f}")
    if abs(real_test_r2 - 0.8415) > 0.005:
        print("WARNING: does not match the established baseline within tolerance -- check the fit.")

    n_negative_clipped = int((grid_base_proxy[~np.isnan(grid_base_proxy)] < FLOOR).sum()
                              + (grid_base_final[~np.isnan(grid_base_final)] < FLOOR).sum())
    print(f"Rows clipped to positivity floor ({FLOOR} kW/m^2): {n_negative_clipped}")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_PATH, index=False)
    print(f"\nSaved {OUT_PATH} ({len(df)} rows, columns: {list(df.columns)})")


if __name__ == "__main__":
    main()

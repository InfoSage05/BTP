"""
Stage 0: generate a large, clean synthetic (P, G, X, D) -> CHF pretraining
corpus from the 2006 Groeneveld CHF lookup table (data/chf_long_clean.csv).

Method
------
1. Load the 11,592-row LUT (a 24 x 21 x 23 grid over P x G x X, CHF @ D=8mm).
   504 of those cells are placeholder zeros (no experimental coverage) --
   these are treated as INVALID, not as real CHF=0 measurements.
2. Build a trilinear interpolator over the full grid. To keep the
   interpolator itself smooth (no zero-cliffs), invalid cells are first
   filled by nearest-valid-neighbor imputation -- but every synthetic sample
   is separately checked against the ORIGINAL (non-imputed) grid: it is only
   kept if all 8 surrounding grid corners were real, valid LUT cells. This
   means the interpolator can be queried anywhere, but only genuinely
   well-supported synthetic points make it into the output file.
3. Diameter is NOT an axis of the 2006 LUT (it's normalized to D=8mm).
   We extend into diameter-space using the published diameter-correction
   exponent from Tanase et al. (2009), "Diameter effect on critical heat
   flux" (Nucl. Eng. Design 239):
        CHF_D = CHF_(D=8mm) * (8 / D)^n
   where n depends on which (P, G, X) bin the point falls in (their Table 3,
   parsed into data/raw/testing/tanase2009_diameter_correction_exponent_grid.csv).
   This is a published physics FORMULA applied deterministically, not
   training rows copied from that (test-only) file -- no experimental data
   point from data/raw/testing/ enters the output.
4. Heated length (L) is NOT modeled here -- the base LUT has no clean,
   well-documented L-dependence correction available in this repo. Output
   rows do not include an L column; that axis of variation is left to the
   real fine-tuning data, which does vary L. This is a known, deliberate
   limitation, not an oversight -- see the README written alongside the
   output for the full caveat.

Output: data/processed/lut_synthetic/lut_synthetic_pretraining.csv
"""
import sys
import re
import numpy as np
import pandas as pd
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import distance_transform_edt

sys.path.insert(0, "scripts/chf_pipeline")
from physics_features import add_dimensionless_features

LUT_PATH = "data/chf_long_clean.csv"
TANASE_PATH = "data/raw/testing/tanase2009_diameter_correction_exponent_grid.csv"
OUT_DIR = "data/processed/lut_synthetic"

N_SAMPLES = 300_000          # CPU-appropriate corpus size (see conversation: scaled
                              # down from a GPU-scale "millions" figure)
D_MIN_MM, D_MAX_MM = 3.0, 20.0   # matches the diameter range actually present
                                   # across this repo's real datasets
RNG_SEED = 42


def load_lut_grid():
    df = pd.read_csv(LUT_PATH)
    p_vals = np.sort(df["P"].unique())
    g_vals = np.sort(df["G"].unique())
    x_vals = np.sort(df["X"].unique())

    grid = np.full((len(p_vals), len(g_vals), len(x_vals)), np.nan)
    pi = {v: i for i, v in enumerate(p_vals)}
    gi = {v: i for i, v in enumerate(g_vals)}
    xi = {v: i for i, v in enumerate(x_vals)}
    for row in df.itertuples(index=False):
        grid[pi[row.P], gi[row.G], xi[row.X]] = row.CHF

    valid_mask = grid > 0  # the 504 zero-cells are invalid placeholders
    return p_vals, g_vals, x_vals, grid, valid_mask


def build_interpolator(p_vals, g_vals, x_vals, grid, valid_mask):
    # Fill invalid cells via nearest-valid-neighbor so the interpolator has
    # no hard zero-cliffs; validity is tracked separately and checked later.
    filled = grid.copy()
    if (~valid_mask).any():
        idx = distance_transform_edt(~valid_mask, return_distances=False,
                                      return_indices=True)
        filled = grid[tuple(idx)]

    interp = RegularGridInterpolator((p_vals, g_vals, x_vals), filled,
                                      method="linear", bounds_error=False,
                                      fill_value=None)
    # a second interpolator on the 0/1 validity mask: a queried point is
    # "well-supported" only if trilinear-interpolated validity == 1.0
    validity_interp = RegularGridInterpolator(
        (p_vals, g_vals, x_vals), valid_mask.astype(float),
        method="linear", bounds_error=False, fill_value=0.0,
    )
    return interp, validity_interp


def load_tanase_lookup():
    df = pd.read_csv(TANASE_PATH)

    def parse_range(s):
        # "-0.5 to -0.25" style (negative bounds, unambiguous separator)
        if " to " in s:
            lo, hi = s.split(" to ")
            return float(lo), float(hi)
        # "100-14000" / "0-250" style (no negative bounds in this repo's data,
        # so a bare split on the single separating hyphen is safe)
        lo, hi = s.split("-")
        return float(lo), float(hi)

    rows = []
    for r in df.itertuples(index=False):
        p_lo, p_hi = parse_range(r.Pressure_kPa_range)
        g_lo, g_hi = parse_range(r.MassFlux_kg_m2s_range)
        x_lo, x_hi = parse_range(r.Quality_range)
        rows.append((p_lo, p_hi, g_lo, g_hi, x_lo, x_hi, r.exponent_n))
    return rows


def lookup_exponent(p, g, x, table):
    for p_lo, p_hi, g_lo, g_hi, x_lo, x_hi, n in table:
        if p_lo <= p <= p_hi and g_lo <= g <= g_hi and x_lo <= x <= x_hi:
            return n
    return np.nan  # outside Tanase's covered ranges


def main():
    p_vals, g_vals, x_vals, grid, valid_mask = load_lut_grid()
    print(f"LUT grid: {grid.shape}, invalid cells: {(~valid_mask).sum()}/{grid.size}")

    interp, validity_interp = build_interpolator(p_vals, g_vals, x_vals, grid, valid_mask)
    tanase_table = load_tanase_lookup()

    rng = np.random.default_rng(RNG_SEED)
    p_lo, p_hi = p_vals.min(), p_vals.max()
    g_lo, g_hi = g_vals.min(), g_vals.max()
    x_lo, x_hi = x_vals.min(), x_vals.max()

    # oversample, then filter to well-supported points
    n_draw = int(N_SAMPLES * 2.5)
    P = rng.uniform(p_lo, p_hi, n_draw)
    G = rng.uniform(g_lo, g_hi, n_draw)
    X = rng.uniform(x_lo, x_hi, n_draw)
    D = np.exp(rng.uniform(np.log(D_MIN_MM), np.log(D_MAX_MM), n_draw))  # log-uniform

    pts = np.column_stack([P, G, X])
    chf_8mm = interp(pts)
    validity = validity_interp(pts)
    well_supported = validity > 0.999  # all 8 surrounding corners were valid

    P, G, X, D, chf_8mm = P[well_supported], G[well_supported], X[well_supported], \
        D[well_supported], chf_8mm[well_supported]
    print(f"Well-supported samples after filtering: {len(P)} / {n_draw}")

    n = np.array([lookup_exponent(p, g, x, tanase_table) for p, g, x in zip(P, G, X)])
    has_n = ~np.isnan(n)
    print(f"Samples with a Tanase diameter-exponent match: {has_n.sum()} / {len(n)}")

    P, G, X, D, chf_8mm, n = P[has_n], G[has_n], X[has_n], D[has_n], chf_8mm[has_n], n[has_n]
    chf_d = chf_8mm * (8.0 / D) ** n

    if len(P) > N_SAMPLES:
        keep = rng.choice(len(P), N_SAMPLES, replace=False)
        P, G, X, D, chf_d, n = P[keep], G[keep], X[keep], D[keep], chf_d[keep], n[keep]

    out = pd.DataFrame({
        "P_kPa": P, "G_kg_m2s": G, "X": X, "D_mm": D, "CHF_kW_m2": chf_d,
        "diameter_correction_exponent_n": n,
    })

    out = add_dimensionless_features(out, fluid="water", p_col="P_kPa", g_col="G_kg_m2s")

    import os
    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, "lut_synthetic_pretraining.csv")
    out.to_csv(out_path, index=False)
    print(f"\nWrote {len(out)} rows -> {out_path}")
    print(out.describe())


if __name__ == "__main__":
    main()

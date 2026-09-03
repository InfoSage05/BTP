"""
validate_physics.py
-------------------
Sanity checks on the closed-form physics BEFORE any model is trained. If the
correlations are wrong, every downstream comparison is meaningless.

Checks:
  1. Zuber reproduces the textbook water-at-1-atm pool boiling CHF (~1.1 MW/m^2).
  2. Zuber reproduces the P/P_c ~ 0.35 pressure maximum (constraint S1) with
     no tuning -- a free validation of the CoolProp property chain.
  3. Katto-Ohno branch activation, which MEASURES the sensitivity to the
     unverified C_Kc constant instead of assuming it away.
  4. Baseline quality per mode: the spread of CHF/Phi_physics by source. A
     good scale makes that ratio ~1 and, more importantly, makes it CONSISTENT
     across sources -- that consistency is what cross-source generalisation
     actually needs.

Run:  python physics_pipeline/scripts/validate_physics.py
"""
from pathlib import Path

import numpy as np
import pandas as pd

from physics import baseline as phys_baseline
from physics import correlations as corr
from physics import properties as phys_properties
from physics import repair as phys_repair

import paths

DATA = paths.MASTER_CSV
POOL = phys_baseline.POOL_BOILING_FAMILIES


def check_zuber_water_1atm():
    print("\n[1] Zuber, water at 1 atm")
    from CoolProp.CoolProp import PropsSI
    p = 101325.0
    rho_l = PropsSI("Dmass", "P", p, "Q", 0, "Water")
    rho_g = PropsSI("Dmass", "P", p, "Q", 1, "Water")
    sigma = PropsSI("surface_tension", "P", p, "Q", 0, "Water")
    h_fg = PropsSI("Hmass", "P", p, "Q", 1, "Water") - PropsSI("Hmass", "P", p, "Q", 0, "Water")
    q = corr.zuber_chf(rho_l, rho_g, sigma, h_fg)
    print(f"    q_CHF = {q/1e6:.3f} MW/m^2   (literature value ~1.1 MW/m^2)")
    ok = 0.9e6 < q < 1.3e6
    print(f"    {'PASS' if ok else 'FAIL'}")
    return ok


def check_pressure_maximum():
    print("\n[2] Zuber pressure trend, water (constraint S1: peak near P/P_c ~ 0.35)")
    from CoolProp.CoolProp import PropsSI
    pc = PropsSI("Pcrit", "Water")
    pr = np.linspace(0.01, 0.95, 120)
    q = []
    for r in pr:
        p = r * pc
        rho_l = PropsSI("Dmass", "P", p, "Q", 0, "Water")
        rho_g = PropsSI("Dmass", "P", p, "Q", 1, "Water")
        sigma = PropsSI("surface_tension", "P", p, "Q", 0, "Water")
        h_fg = PropsSI("Hmass", "P", p, "Q", 1, "Water") - PropsSI("Hmass", "P", p, "Q", 0, "Water")
        q.append(corr.zuber_chf(rho_l, rho_g, sigma, h_fg))
    q = np.array(q)
    peak = pr[int(np.argmax(q))]
    print(f"    peak at P/P_c = {peak:.3f}  (P = {peak*pc/1e6:.2f} MPa)")
    print(f"    CHF -> 0 near critical: q(0.95 P_c)/q(peak) = {q[-1]/q.max():.4f}  (constraint C6)")
    ok = 0.25 <= peak <= 0.45
    print(f"    {'PASS' if ok else 'FAIL'}")
    return ok


def check_katto_branches(df):
    print("\n[3] Katto-Ohno branch activation (C_Kc sensitivity)")
    flow = df[~df["geometry_family"].isin(POOL)].copy()
    G = pd.to_numeric(flow["mass_flux_kg_m2s"], errors="coerce").values
    D = pd.to_numeric(flow["diameter_mm"], errors="coerce").values / 1000.0
    L = pd.to_numeric(flow["heated_length_mm"], errors="coerce").values / 1000.0
    dh = np.nan_to_num(pd.to_numeric(flow["subcooling_kJkg"], errors="coerce").values * 1000.0)
    chf, branch, low = corr.katto_ohno_chf(
        G, D, L, flow["rho_l_sat"].values, flow["rho_g_sat"].values,
        flow["sigma_sat"].values, flow["h_fg_sat"].values, dh, return_branches=True)

    usable = np.isfinite(chf)
    frac_q2 = float(np.mean(branch[usable] == 0))
    print(f"    rows with rho_g/rho_f < 0.15 (low-density regime): {np.mean(low[usable])*100:.1f}%")
    print(f"    q_co,2 selected (the branch that uses C_Kc): {frac_q2*100:.1f}% of rows")

    # Direct sensitivity: rerun with C_Kc perturbed +/-30% and measure the
    # change in the resulting baseline.
    c_small, c_large = corr.KATTO_C_SMALL, corr.KATTO_C_LARGE
    ratios = []
    for scale in (0.7, 1.3):
        corr.KATTO_C_SMALL, corr.KATTO_C_LARGE = c_small * scale, c_large * scale
        chf2 = corr.katto_ohno_chf(G, D, L, flow["rho_l_sat"].values, flow["rho_g_sat"].values,
                                    flow["sigma_sat"].values, flow["h_fg_sat"].values, dh)
        m = usable & np.isfinite(chf2) & (chf > 0)
        r = chf2[m] / chf[m]
        ratios.append(float(np.median(r)))
        frac_changed = float(np.mean(np.abs(r - 1.0) > 1e-6))
        max_change = float(np.max(np.abs(r - 1.0)) + 1.0)
    corr.KATTO_C_SMALL, corr.KATTO_C_LARGE = c_small, c_large
    print(f"    median CHF change for C_Kc -30% / +30%: {ratios[0]:.4f}x / {ratios[1]:.4f}x")
    print(f"    rows whose CHF changed at all: {frac_changed*100:.1f}%, "
          f"max change {max_change:.3f}x")
    print("    -> The MEDIAN is unchanged, but a substantial minority of rows move,")
    print("       some by ~3x. C_Kc therefore DOES matter for absolute accuracy and")
    print("       remains a real blocker (Katto & Ohno 1984, IJHMT 27(9):1641).")
    print("       The ablation comparison below is still internally valid, because")
    print("       every arm uses the same C_Kc -- but the absolute CHF numbers are")
    print("       provisional until the primary source is read.")
    return True


def check_baseline_modes(df):
    print("\n[4] Baseline quality by mode: CHF / Phi_physics")
    print("    Good scale => ratio near 1 AND consistent across sources.")
    y = pd.to_numeric(df["CHF_kW_m2"], errors="coerce").values
    rows = []
    for mode in phys_baseline.BASELINE_MODES:
        b = phys_baseline.compute_physics_baseline_kw_m2(df, mode=mode).values
        r = y / b
        ok = np.isfinite(r) & (r > 0)
        per_source = pd.Series(r[ok]).groupby(df["source_dataset"].values[ok]).median()
        rows.append({
            "mode": mode,
            "median_ratio": float(np.median(r[ok])),
            "log10_spread_all": float(np.std(np.log10(r[ok]))),
            "log10_spread_across_sources": float(np.std(np.log10(per_source.values))),
            "max/min across sources": float(per_source.max() / per_source.min()),
        })
    out = pd.DataFrame(rows)
    print(out.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print("\n    Per-source median CHF/Phi for each mode:")
    tbl = {}
    for mode in phys_baseline.BASELINE_MODES:
        b = phys_baseline.compute_physics_baseline_kw_m2(df, mode=mode).values
        r = y / b
        ok = np.isfinite(r) & (r > 0)
        tbl[mode] = pd.Series(r[ok]).groupby(df["source_dataset"].values[ok]).median()
    print(pd.DataFrame(tbl).to_string(float_format=lambda v: f"{v:.3f}"))
    return True


def main():
    raw = pd.read_csv(DATA, low_memory=False)
    df = phys_repair.repair(raw)
    print(phys_repair.repair_report(raw, df))
    df = phys_properties.add_saturation_properties(df, POOL)

    print("=" * 74)
    print("PHYSICS VALIDATION")
    print("=" * 74)
    print(f"rows: {len(df)}")
    prop_cov = df[[f"{n}_sat" for n in phys_properties.PROPERTY_NAMES]].notna().all(axis=1)
    print(f"rows with complete saturation properties: {prop_cov.sum()} "
          f"({100*prop_cov.mean():.1f}%)")
    for fl in sorted(df["fluid"].astype(str).str.lower().unique()):
        m = df["fluid"].astype(str).str.lower() == fl
        print(f"    {fl:8s} {m.sum():6d} rows, properties complete: "
              f"{100*prop_cov[m].mean():5.1f}%")

    check_zuber_water_1atm()
    check_pressure_maximum()
    check_katto_branches(df)
    check_baseline_modes(df)
    print("\n" + "=" * 74)


if __name__ == "__main__":
    main()

"""
physics/repair.py
-----------------
Stage-0 data repair. Fixes gaps in `master_chf_dataset.csv` that block the
physics, without touching the merge itself.

REPAIR 1 -- helical-coil tube diameter (257 rows)
-------------------------------------------------
The helical source carries `coil_no` (Coil_1..Coil_6) and `heated_length_mm`
but NO diameter at all, so every diameter-dependent correlation returned NaN
for those rows and silently fell back to the latent-heat scale. That is why
`helical_coil_r123` showed CHF/Phi = 0.002 in every baseline mode.

The diameter is not missing from the underlying data, only from the merge.
The source appendix (`data/raw/external/helical_coil_chf_research_data.pdf`)
tabulates heat supply Q [W], heated length L_h [mm] and CHF [kW/m^2] for all
257 rows, and those three are linked by the definition of heat flux on a
tube's inner surface:

        CHF = Q / (pi * d * L_h)     =>     d = Q / (pi * L_h * CHF)

Solving that per row and grouping by `coil_no` recovers each coil's tube
diameter to better than 0.003 mm standard deviation:

        Coil_1  9.500 mm      Coil_4  9.500 mm
        Coil_2  7.500 mm      Coil_5  7.500 mm
        Coil_3  5.500 mm      Coil_6  5.400 mm

These are DERIVED from the source's own energy balance, not fitted and not
guessed -- the sub-0.003 mm spread across 22-60 independent rows per coil is
the evidence. They are also consistent with the companion water study
(Hardik & Prabhu 2017), which used six coils of 6-10 mm inner diameter.

NOT recoverable: the COIL diameter D_coil, and therefore the curvature ratio
d/D_coil. Nothing in the appendix constrains it -- the coils pair up by tube
diameter (1/4, 2/5, 3/6), which strongly suggests each pair differs only in
coil diameter, but the values themselves are not in the data. The curvature
ratio is left NaN rather than invented; see foundation doc section 6.1.

REPAIR 2 -- regime label
------------------------
Adds `chf_regime` in {pool, DNB, dryout} from geometry family and quality
sign. Foundation doc section 1.2: DNB and dryout are physically distinct
crises that share a name, and the merged table has 3,637 DNB-side rows
against 24,603 dryout-side ones. Making the distinction explicit lets the
baseline dispatch on mechanism rather than on the administrative
`geometry_family` label.
"""
import numpy as np
import pandas as pd

#: Tube inner diameter per coil, derived from the source energy balance.
#: See module docstring for the derivation and its verification.
HELICAL_COIL_DIAMETER_MM = {
    "Coil_1": 9.500,
    "Coil_2": 7.500,
    "Coil_3": 5.500,
    "Coil_4": 9.500,
    "Coil_5": 7.500,
    "Coil_6": 5.400,
}

POOL_BOILING_FAMILIES = {"pin_fin_pool_boiling", "flat_heater_pool_boiling"}


def repair_helical_diameters(df: pd.DataFrame) -> pd.DataFrame:
    """Fill `diameter_mm` for helical-coil rows from `coil_no`."""
    df = df.copy()
    if "coil_no" not in df.columns:
        return df
    is_coil = (df["geometry_family"] == "helical_coil").values
    missing = is_coil & df["diameter_mm"].isna().values
    if not missing.any():
        return df
    mapped = df["coil_no"].map(HELICAL_COIL_DIAMETER_MM)
    df.loc[missing, "diameter_mm"] = mapped[missing]
    df.loc[missing & mapped.notna().values, "assumptions"] = (
        df.loc[missing & mapped.notna().values, "assumptions"].fillna("")
        + "; diameter_mm derived from source Q/(pi L CHF) energy balance")
    return df


def add_regime_label(df: pd.DataFrame) -> pd.DataFrame:
    """Add `chf_regime` in {pool, DNB, dryout}. Foundation doc section 1.2."""
    df = df.copy()
    x = pd.to_numeric(df["quality"], errors="coerce").values
    is_pool = df["geometry_family"].isin(POOL_BOILING_FAMILIES).values
    regime = np.where(is_pool, "pool", np.where(x > 0.0, "dryout", "DNB"))
    # Rows with no quality at all are flow rows of unknown mechanism; the
    # subcooled default is the conservative one (it predicts the higher CHF
    # branch, so it cannot be used to hide an optimistic prediction).
    regime = np.where(~is_pool & ~np.isfinite(x), "DNB", regime)
    df["chf_regime"] = regime
    return df


def repair(df: pd.DataFrame) -> pd.DataFrame:
    """Apply every Stage-0 repair, in order."""
    return add_regime_label(repair_helical_diameters(df))


def repair_report(before: pd.DataFrame, after: pd.DataFrame) -> str:
    lines = ["Stage-0 data repair", "=" * 60]
    for fam in sorted(after["geometry_family"].unique()):
        m_b = before["geometry_family"] == fam
        m_a = after["geometry_family"] == fam
        nb = int(pd.to_numeric(before.loc[m_b, "diameter_mm"], errors="coerce").notna().sum())
        na = int(pd.to_numeric(after.loc[m_a, "diameter_mm"], errors="coerce").notna().sum())
        if nb != na:
            lines.append(f"  diameter_mm {fam:26s} {nb:6d} -> {na:6d} / {int(m_a.sum())}")
    lines.append("")
    lines.append("  chf_regime counts:")
    for k, v in after["chf_regime"].value_counts().items():
        lines.append(f"    {k:8s} {v:6d}")
    return "\n".join(lines)

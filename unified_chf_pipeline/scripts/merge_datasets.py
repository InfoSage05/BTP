"""
merge_datasets.py
------------------
Builds the single master CHF dataset by running every loader in recipes.py,
standardizing each result onto one canonical column list/unit system, and
concatenating them into one long table with full provenance.

Run:
    python unified_chf_pipeline/scripts/merge_datasets.py

Writes:
    unified_chf_pipeline/data/master_chf_dataset.csv
    unified_chf_pipeline/data/merge_report.md
"""
from pathlib import Path

import pandas as pd

from recipes import SOURCE_LOADERS

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = Path(__file__).resolve().parents[1] / "data"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Canonical column order. Core (always-relevant) columns first, then the
# sparse, family-specific optional columns. Any column a given loader does
# not fill is left as NaN -- that sparsity is expected, not an error.
CANONICAL_COLUMNS = [
    # provenance
    "row_id", "source_dataset", "source_row_id", "geometry_family", "data_type",
    # core operating / geometry (mandatory-ish, but see per-family notes)
    "fluid", "pressure_kPa", "mass_flux_kg_m2s", "quality",
    "subcooling_K", "subcooling_kJkg", "diameter_mm", "heated_length_mm",
    # target
    "CHF_kW_m2",
    # provenance / traceability extras
    "reference_id", "author", "source_citation", "inlet_temperature_C",
    "assumptions",
    # tube/coil specific
    "D_e_mm", "wallpower", "wallmesh", "chf_location", "shape", "continuous",
    "coil_no", "appendix", "rho_l_over_rho_g",
    # pin-fin / pool-boiling surface characteristics
    "fin_shape", "fin_array", "fin_width_um", "fin_height_um", "fin_spacing_um",
    "coverage", "porosity", "mbl_lateral_um", "mbl_total_um", "roughness_factor",
    "surface_material",
    # flat-heater pool-boiling (mentor) specific
    "heater_width_mm", "orientation", "angle_deg", "tsat_minus_tpool_K",
    "surface_tension_N_m", "rho_l_kg_m3", "cp_l_J_kgK", "kl_W_mK", "mu_l_Pa_s",
    "alpha_m2_s", "ja", "r_bubble_m",
]


def merge_all() -> pd.DataFrame:
    frames = []
    for source_name, loader in SOURCE_LOADERS.items():
        df = loader()
        df = df.copy()
        df["source_dataset"] = source_name
        df["row_id"] = [f"{source_name}__{i}" for i in range(len(df))]
        frames.append(df)

    master = pd.concat(frames, ignore_index=True, sort=False)
    master = master.reindex(columns=CANONICAL_COLUMNS)
    # pinfin's source file spells the fluid column "Water"/"FC-72" while every
    # other source uses lowercase "water" -- normalize so they aren't treated
    # as distinct categories downstream.
    master["fluid"] = master["fluid"].str.lower()
    return master


def validate(master: pd.DataFrame) -> str:
    lines = []
    lines.append(f"Total rows: {len(master)}")
    lines.append(f"Duplicate row_id count: {master['row_id'].duplicated().sum()}")

    dup_cols = ["pressure_kPa", "mass_flux_kg_m2s", "quality", "CHF_kW_m2"]
    dup_mask = master.dropna(subset=dup_cols).duplicated(subset=dup_cols, keep=False)
    n_cross_source_dupes = 0
    if dup_mask.any():
        dup_rows = master.dropna(subset=dup_cols)[dup_mask]
        n_cross_source_dupes = dup_rows.groupby(dup_cols)["source_dataset"].nunique().gt(1).sum()
    lines.append(f"Exact-value groups shared across >1 source_dataset "
                 f"(possible unflagged duplicates): {n_cross_source_dupes}")

    lines.append("\nRows per source_dataset:")
    lines.append(master["source_dataset"].value_counts().to_string())

    lines.append("\nRows per geometry_family:")
    lines.append(master["geometry_family"].value_counts(dropna=False).to_string())

    lines.append("\nColumn coverage (% non-null):")
    coverage = (master.notna().mean() * 100).round(1).sort_values(ascending=False)
    lines.append(coverage.to_string())

    return "\n".join(lines)


if __name__ == "__main__":
    master = merge_all()
    report = validate(master)

    master_path = OUT_DIR / "master_chf_dataset.csv"
    master.to_csv(master_path, index=False)

    report_path = OUT_DIR / "merge_report.md"
    report_path.write_text(
        "# Merge Report\n\n```\n" + report + "\n```\n"
    )

    print(f"Wrote {master_path} ({len(master)} rows, {len(master.columns)} columns)")
    print(f"Wrote {report_path}")
    print("\n" + report)

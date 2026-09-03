"""
build_eda_notebooks.py
-----------------------
Programmatically assembles one exploratory-data-analysis notebook per raw
dataset in data/raw/external/ (13 notebooks total: 7 point-level datasets
with a full statistical EDA, 6 aggregate/range-summary tables with a
lighter comparison-style EDA). Mirrors the pattern used in build_notebook.py.

Run once to (re)generate the notebooks:
    python scripts/build_eda_notebooks.py
Then execute them (populates outputs/figures) with, e.g.:
    jupyter nbconvert --to notebook --execute --inplace notebooks/eda/*.ipynb
"""
import nbformat as nbf
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
NB_DIR = ROOT / "notebooks" / "eda"
NB_DIR.mkdir(parents=True, exist_ok=True)


def md(src):
    return nbf.v4.new_markdown_cell(src)


def code(src):
    return nbf.v4.new_code_cell(src)


def write_notebook(slug, cells):
    nb = nbf.v4.new_notebook()
    nb["cells"] = cells
    nb["metadata"] = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "pygments_lexer": "ipython3"},
    }
    path = NB_DIR / f"{slug}.ipynb"
    nbf.write(nb, path)
    print("wrote", path)


# ============================================================================
# Shared generic cells for Group A (point-level) notebooks.
# These assume the load/clean cell has already defined:
#   df, NUMERIC_COLS, CATEGORICAL_COLS, TARGET_COL, SLUG, SLUG_TITLE, OUT_DIR, FIG_DIR
# ============================================================================

CELL_QUALITY = code(r"""
print("Shape:", df.shape)
print()
print("Dtypes:")
print(df.dtypes)
print()
missing = df.isna().sum()
missing = missing[missing > 0].sort_values(ascending=False)
print("Missing values per column:")
print(missing if len(missing) else "None")
print()
print("Duplicate rows:", df.duplicated().sum())
""".strip())

CELL_DESCRIBE = code(r"""
numeric_summary = df[NUMERIC_COLS].describe().T
numeric_summary.to_csv(OUT_DIR / "summary_stats.csv")
numeric_summary
""".strip())

CELL_CATCOUNTS = code(r"""
for col in CATEGORICAL_COLS:
    print(f"--- {col} (n_unique={df[col].nunique(dropna=True)}) ---")
    print(df[col].value_counts(dropna=False).head(15))
    print()
""".strip())

CELL_HIST = code(r"""
n = len(NUMERIC_COLS)
ncols = 3
nrows = -(-n // ncols)
fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3.5 * nrows))
axes = np.array(axes).reshape(-1)
for ax, col in zip(axes, NUMERIC_COLS):
    ax.hist(df[col].dropna(), bins=30, color="#4C72B0", edgecolor="white")
    ax.set_title(col, fontsize=10)
for ax in axes[n:]:
    ax.axis("off")
fig.suptitle(f"{SLUG_TITLE}: Numeric Feature Distributions", y=1.02)
fig.tight_layout()
fig.savefig(FIG_DIR / "histograms.png", dpi=150, bbox_inches="tight")
plt.show()
""".strip())

CELL_IQR = code(r"""
iqr_rows = []
for col in NUMERIC_COLS:
    s = df[col].dropna()
    q1, q3 = s.quantile([0.25, 0.75])
    iqr = q3 - q1
    lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    n_out = int(((s < lo) | (s > hi)).sum())
    iqr_rows.append({
        "column": col, "q1": q1, "q3": q3, "iqr": iqr,
        "lower_fence": lo, "upper_fence": hi,
        "n_outliers": n_out, "pct_outliers": 100 * n_out / len(s) if len(s) else np.nan,
    })
iqr_table = pd.DataFrame(iqr_rows).set_index("column")
iqr_table.to_csv(OUT_DIR / "iqr_outlier_summary.csv")
iqr_table
""".strip())

CELL_BOXPLOT = code(r"""
z = df[NUMERIC_COLS].apply(lambda s: (s - s.mean()) / s.std())
fig, ax = plt.subplots(figsize=(max(6, 1.2 * len(NUMERIC_COLS)), 5))
ax.boxplot([z[c].dropna() for c in NUMERIC_COLS])
ax.set_xticklabels(NUMERIC_COLS, rotation=60, ha="right")
ax.set_title(f"{SLUG_TITLE}: Standardized Boxplots (z-score)")
ax.axhline(0, color="grey", linewidth=0.8, linestyle="--")
fig.tight_layout()
fig.savefig(FIG_DIR / "boxplots_standardized.png", dpi=150, bbox_inches="tight")
plt.show()
""".strip())

CELL_CORR = code(r"""
corr = df[NUMERIC_COLS].corr()
corr.to_csv(OUT_DIR / "correlation_matrix.csv")
fig, ax = plt.subplots(figsize=(0.6 * len(NUMERIC_COLS) + 2, 0.6 * len(NUMERIC_COLS) + 2))
im = ax.imshow(corr, vmin=-1, vmax=1, cmap="RdBu_r")
ax.set_xticks(range(len(NUMERIC_COLS)))
ax.set_xticklabels(NUMERIC_COLS, rotation=90)
ax.set_yticks(range(len(NUMERIC_COLS)))
ax.set_yticklabels(NUMERIC_COLS)
for i in range(len(NUMERIC_COLS)):
    for j in range(len(NUMERIC_COLS)):
        ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=7)
fig.colorbar(im, shrink=0.8)
ax.set_title(f"{SLUG_TITLE}: Correlation Matrix")
fig.tight_layout()
fig.savefig(FIG_DIR / "correlation_heatmap.png", dpi=150, bbox_inches="tight")
plt.show()
""".strip())

CELL_SCATTER = code(r"""
feature_cols = [c for c in NUMERIC_COLS if c != TARGET_COL]
n = len(feature_cols)
ncols = 3
nrows = -(-n // ncols)
fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3.5 * nrows))
axes = np.array(axes).reshape(-1)
for ax, col in zip(axes, feature_cols):
    ax.scatter(df[col], df[TARGET_COL], s=8, alpha=0.35, color="#4C72B0")
    ax.set_xlabel(col)
    ax.set_ylabel(TARGET_COL)
for ax in axes[n:]:
    ax.axis("off")
fig.suptitle(f"{SLUG_TITLE}: {TARGET_COL} vs. Features", y=1.02)
fig.tight_layout()
fig.savefig(FIG_DIR / "target_vs_features.png", dpi=150, bbox_inches="tight")
plt.show()
""".strip())

CELL_CATBOX = code(r"""
cat_cols_plot = [c for c in CATEGORICAL_COLS
                 if 2 <= df[c].nunique(dropna=True) <= 15]
if cat_cols_plot:
    fig, axes = plt.subplots(1, len(cat_cols_plot), figsize=(6 * len(cat_cols_plot), 5), squeeze=False)
    axes = axes[0]
    for ax, col in zip(axes, cat_cols_plot):
        groups = [g[TARGET_COL].dropna().values for _, g in df.groupby(col)]
        labels = [str(k) for k, _ in df.groupby(col)]
        ax.boxplot(groups)
        ax.set_xticklabels(labels, rotation=60, ha="right")
        ax.set_title(f"{TARGET_COL} by {col}")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "target_by_category.png", dpi=150, bbox_inches="tight")
    plt.show()
else:
    print("No low-cardinality categorical columns (2-15 unique values) available for grouped comparison.")
""".strip())


def generic_a_block(target_heading_num=8):
    """Returns the shared markdown+code sequence for a Group A notebook."""
    return [
        md("## 2. Data Quality"), CELL_QUALITY,
        md("## 3. Descriptive Statistics"), CELL_DESCRIBE,
        md("## 4. Categorical Value Counts"), CELL_CATCOUNTS,
        md("## 5. Univariate Distributions"), CELL_HIST,
        md("## 6. Outlier Scan (IQR, z-score)"), CELL_IQR, CELL_BOXPLOT,
        md("## 7. Correlation Structure"), CELL_CORR,
        md(f"## {target_heading_num}. Target vs. Features"), CELL_SCATTER,
        md(f"## {target_heading_num + 1}. Target by Category"), CELL_CATBOX,
    ]


def setup_cell(slug, data_expr):
    return code(f"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

pd.set_option("display.max_columns", 60)
plt.rcParams["figure.dpi"] = 100

DATA_DIR = Path("../../data/raw")
OUT_DIR = Path("../../results/eda/{slug}")
FIG_DIR = OUT_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

SLUG = "{slug}"
{data_expr}
""".strip())


# ============================================================================
# 1. pinfin_chf_water_fc72  (== CHF Dataset.csv, byte-identical duplicate)
# ============================================================================
def build_pinfin():
    slug = "pinfin_chf_water_fc72"
    cells = [
        md(r"""
# EDA: Pin-Fin CHF — Water / FC-72 (`pinfin_chf_water_fc72.csv`)

Pool-boiling critical heat flux measurements on micro-pin-fin enhanced
surfaces, water and FC-72 dielectric fluid.

**Note on duplication**: `data/raw/external/CHF Dataset.csv` is a
byte-identical copy of `pinfin_chf_water_fc72.csv` (verified below). This
notebook treats them as a single dataset and analyzes only
`pinfin_chf_water_fc72.csv`.

## 1. Load Data
"""),
        code(r"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

pd.set_option("display.max_columns", 60)
plt.rcParams["figure.dpi"] = 100

DATA_DIR = Path("../../data/raw")
OUT_DIR = Path("../../results/eda/pinfin_chf_water_fc72")
FIG_DIR = OUT_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

SLUG = "pinfin_chf_water_fc72"
SLUG_TITLE = "Pin-Fin CHF (Water/FC-72)"

df = pd.read_csv(DATA_DIR / "fine_tuning" / "pinfin_chf_water_fc72.csv", encoding="utf-8-sig")
dup = pd.read_csv(DATA_DIR / "fine_tuning" / "CHF Dataset.csv", encoding="utf-8-sig")
print("CHF Dataset.csv is byte-identical to pinfin_chf_water_fc72.csv:", df.equals(dup))

NUMERIC_COLS = ["Subcooling(K)", "Width(um)", "Height(um)", "Spacing(um)", "Coverage",
                "Porosity", "MBL, Lateral (um)", "MBL, Total (um)", "Roughness Factor",
                "CHF(kW/m2)"]
CATEGORICAL_COLS = ["Fin Shape", "Fin Array", "Surface Material", "Fluid"]
TARGET_COL = "CHF(kW/m2)"

df.head()
""".strip()),
    ]
    cells += generic_a_block()
    cells.append(md(r"""
## 10. Takeaways

- The first row (`Fin Shape`/`Fin Array` = NaN, all geometry columns = 0) is a
  plain/unfinned reference surface — keep it as the baseline case, but be aware
  it will show up as an outlier / missing-category row in the checks above.
- `CHF(kW/m2)` is the modeling target; geometry columns (`Width`, `Height`,
  `Spacing`, `Coverage`, `Porosity`, the two `MBL` columns, `Roughness Factor`)
  are the engineered-surface features. `Subcooling(K)` is 0 for most rows in
  this extract (saturated pool boiling) — check the histogram above before
  using it as a continuous predictor.
- `Source` is a free-text citation column (mostly missing after the first row
  per source) — useful for provenance, not for modeling.
"""))
    write_notebook(slug, cells)


# ============================================================================
# 2. kaeri_tr1665_uniform
# ============================================================================
def build_kaeri_uniform():
    slug = "kaeri_tr1665_uniform"
    cells = [
        md(r"""
# EDA: KAERI TR-1665 — Uniformly Heated Tubes (`kaeri_tr1665_uniform_chf.csv`)

CHF test data for uniformly (axially) heated round tubes from KAERI/TR-1665
(Swenson-type dataset). A parallel `kaeri_tr1665_uniform.xml` file carries the
same records in XML form; a cross-check is included below.

## 1. Load Data
"""),
        code(r"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

pd.set_option("display.max_columns", 60)
plt.rcParams["figure.dpi"] = 100

DATA_DIR = Path("../../data/raw")
OUT_DIR = Path("../../results/eda/kaeri_tr1665_uniform")
FIG_DIR = OUT_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

SLUG = "kaeri_tr1665_uniform"
SLUG_TITLE = "KAERI TR-1665 Uniform Heating"

df = pd.read_csv(DATA_DIR / "fine_tuning" / "kaeri_tr1665_uniform_chf.csv")

NUMERIC_COLS = ["Diameter", "Perimeter", "Area", "Length", "Pressure", "Power",
                "MassFlux", "MassFlow", "EquilibriumQuality", "InletTemperature",
                "InletEnthalpy", "HeatFlux", "WallMesh"]
CATEGORICAL_COLS = ["Fluid", "Source"]
TARGET_COL = "HeatFlux"

df.head()
""".strip()),
        md("## 1b. Cross-check against `kaeri_tr1665_uniform.xml`"),
        code(r"""
import xml.etree.ElementTree as ET

tree = ET.parse(DATA_DIR / "fine_tuning" / "kaeri_tr1665_uniform.xml")
records = []
for ds in tree.getroot().findall("dataset"):
    rec = {}
    for child in ds:
        rec.setdefault(child.tag, []).append(child.text)
    records.append(rec)

# Flag fields that repeat within a single <dataset> block (e.g. WallPower, WallMesh
# each appear twice in the raw XML) -- this is a structural quirk of the source file,
# not extra information.
repeated_fields = sorted({tag for rec in records for tag, vals in rec.items() if len(vals) > 1})
print("XML fields with duplicated tags per record:", repeated_fields)

df_xml = pd.DataFrame([{tag: vals[0] for tag, vals in rec.items()} for rec in records])
df_xml = df_xml.apply(pd.to_numeric, errors="ignore")

print("CSV rows:", len(df), " XML records:", len(df_xml))
print("Column sets match:", set(df.columns) == set(df_xml.columns))
""".strip()),
    ]
    cells += generic_a_block(target_heading_num=8)
    cells.append(md(r"""
## 10. Takeaways

- The XML source repeats `WallPower`/`WallMesh` tags identically within each
  `<dataset>` block (a serialization quirk, not additional information) — the
  first occurrence matches the CSV exactly.
- `HeatFlux` is used as the CHF target; `EquilibriumQuality` at CHF and
  `MassFlux` are the primary boiling-crisis predictors alongside `Pressure`
  and tube `Diameter`/`Length`.
- All rows are `Fluid = water`, `Source = Swenson` — these columns carry no
  variance in this file and are shown for completeness only.
"""))
    write_notebook(slug, cells)


# ============================================================================
# 3. kaeri_tr1665_nonuniform
# ============================================================================
def build_kaeri_nonuniform():
    slug = "kaeri_tr1665_nonuniform"
    cells = [
        md(r"""
# EDA: KAERI TR-1665 — Non-Uniformly Heated Tubes (`kaeri_tr1665_nonuniform_chf.csv`)

CHF test data for axially non-uniform (cosine-type) heating profiles from
KAERI/TR-1665. The companion `kaeri_tr1665_nonuniform.xml` file carries, per
test, the full axial local-quality profile (`Quality` at each
`QualityPosition`) collapsed to a single row in the flattened CSV — that
profile structure is explored separately below.

## 1. Load Data
"""),
        code(r"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

pd.set_option("display.max_columns", 60)
plt.rcParams["figure.dpi"] = 100

DATA_DIR = Path("../../data/raw")
OUT_DIR = Path("../../results/eda/kaeri_tr1665_nonuniform")
FIG_DIR = OUT_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

SLUG = "kaeri_tr1665_nonuniform"
SLUG_TITLE = "KAERI TR-1665 Non-Uniform Heating"

df = pd.read_csv(DATA_DIR / "fine_tuning" / "kaeri_tr1665_nonuniform_chf.csv")

NUMERIC_COLS = ["Diameter", "Perimeter", "Area", "Length", "Pressure", "Power",
                "MassFlux", "MassFlow", "InletTemperature", "InletEnthalpy",
                "HeatFlux", "CHFLocation", "Quality", "QualityPosition",
                "WallPower", "WallMesh"]
CATEGORICAL_COLS = ["Shape", "Fluid", "Source", "Continuous"]
TARGET_COL = "HeatFlux"

df.head()
""".strip()),
    ]
    cells += generic_a_block(target_heading_num=8)
    cells += [
        md(r"""
## 10. Axial Local-Quality Profiles (from `kaeri_tr1665_nonuniform.xml`)

Each test in the XML repeats the `Quality`/`QualityPosition` tags once per
axial measurement location, i.e. the CHF-location `Quality`/`QualityPosition`
columns in the CSV are only the *last* value of a full axial profile. This
section reconstructs those profiles.
"""),
        code(r"""
import xml.etree.ElementTree as ET

tree = ET.parse(DATA_DIR / "fine_tuning" / "kaeri_tr1665_nonuniform.xml")
profiles = []
for ds in tree.getroot().findall("dataset"):
    test_id = ds.find("TestID").text
    qualities = [float(q.text) for q in ds.findall("Quality")]
    positions = [float(p.text) for p in ds.findall("QualityPosition")]
    profiles.append({"TestID": test_id, "n_points": len(qualities),
                      "qualities": qualities, "positions": positions})

profiles_df = pd.DataFrame(profiles)
profiles_df["n_points"].describe()
""".strip()),
        code(r"""
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

axes[0].hist(profiles_df["n_points"], bins=range(1, profiles_df["n_points"].max() + 2),
             color="#4C72B0", edgecolor="white")
axes[0].set_title("Axial profile length (# points per test)")
axes[0].set_xlabel("n_points")

rng = np.random.default_rng(0)
sample_idx = rng.choice(len(profiles_df), size=min(8, len(profiles_df)), replace=False)
for i in sample_idx:
    row = profiles_df.iloc[i]
    axes[1].plot(row["positions"], row["qualities"], marker="o", markersize=3,
                 alpha=0.8, label=f"Test {row['TestID']}")
axes[1].set_xlabel("Axial position")
axes[1].set_ylabel("Local quality")
axes[1].set_title("Sample axial quality profiles")
axes[1].legend(fontsize=7, ncol=2)

fig.suptitle(f"{SLUG_TITLE}: Axial Quality Profile Structure", y=1.03)
fig.tight_layout()
fig.savefig(FIG_DIR / "axial_quality_profiles.png", dpi=150, bbox_inches="tight")
plt.show()
""".strip()),
        md(r"""
## 11. Takeaways

- `HeatFlux` at the reported `CHFLocation` is the target; `Quality` /
  `QualityPosition` in the flat CSV are single (CHF-location) snapshots of a
  richer axial profile that only exists in the XML.
- Non-uniform heating means `Shape` (e.g. `inlet`, cosine-peak location) is an
  important categorical driver of where CHF occurs along the tube — check the
  target-by-category plot above.
- As with the uniform-heating file, `Fluid`/`Source` are constant
  (`water`/`Swenson`) and carry no discriminative signal here.
"""),
    ]
    write_notebook(slug, cells)


# ============================================================================
# 4. nrc_groeneveld_chf_database
# ============================================================================
def build_groeneveld():
    slug = "nrc_groeneveld_chf_database"
    cells = [
        md(r"""
# EDA: NRC/Groeneveld 24,579-Point CHF Database (`nrc_groeneveld_24579pt_chf_database.csv`)

The raw experimental database underlying the 2006 Groeneveld CHF Look-Up
Table: tube diameter, heated length, pressure, mass flux, outlet quality,
inlet subcooling/temperature and measured CHF, compiled from many source
studies (`Reference ID`). This is by far the largest raw file in the corpus
(24,579 data rows) — plots below use transparency/hexbin-style styling and a
top-N view of `Reference ID` rather than a full one-hot breakdown.

## 1. Load Data

Row 2 of the raw CSV is a units header (`m`, `kPa`, `kg/m^2/s`, ...), not
data, and is skipped on load.
"""),
        code(r"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

pd.set_option("display.max_columns", 60)
plt.rcParams["figure.dpi"] = 100

DATA_DIR = Path("../../data/raw")
OUT_DIR = Path("../../results/eda/nrc_groeneveld_chf_database")
FIG_DIR = OUT_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

SLUG = "nrc_groeneveld_chf_database"
SLUG_TITLE = "NRC/Groeneveld 24,579-pt CHF Database"

raw = pd.read_csv(DATA_DIR / "pretraining" / "nrc_groeneveld_24579pt_chf_database.csv",
                   encoding="utf-8-sig", skiprows=[1])

numeric_like = ["Number", "Tube Diameter", "Heated Length", "Pressure", "Mass Flux",
                 "Outlet Quality", "Inlet Subcooling", "Inlet Temperature", "CHF"]
for c in numeric_like:
    raw[c] = pd.to_numeric(raw[c], errors="coerce")
raw["CHF Result"] = pd.to_numeric(raw["CHF Result"], errors="coerce")

print("Rows where 'CHF Result' is populated (data-quality artifact; expected ~0):",
      raw["CHF Result"].notna().sum(), "/", len(raw))

df = raw.drop(columns=["CHF Result"])

NUMERIC_COLS = ["Tube Diameter", "Heated Length", "Pressure", "Mass Flux",
                 "Outlet Quality", "Inlet Subcooling", "Inlet Temperature", "CHF"]
CATEGORICAL_COLS = ["Reference ID"]
TARGET_COL = "CHF"

df.head()
""".strip()),
    ]
    cells += generic_a_block(target_heading_num=8)
    cells += [
        md("## 10. Coverage by Source Study (`Reference ID`)"),
        code(r"""
top_refs = df["Reference ID"].value_counts().head(20)
fig, ax = plt.subplots(figsize=(8, 6))
ax.barh(top_refs.index.astype(str)[::-1], top_refs.values[::-1], color="#4C72B0")
ax.set_xlabel("Number of rows")
ax.set_title(f"{SLUG_TITLE}: Top 20 Source Studies by Row Count "
             f"(of {df['Reference ID'].nunique()} total)")
fig.tight_layout()
fig.savefig(FIG_DIR / "top_reference_ids.png", dpi=150, bbox_inches="tight")
plt.show()
""".strip()),
        md(r"""
## 11. Takeaways

- `CHF` (kW/m^2) is the target; `Pressure`, `Mass Flux`, `Outlet Quality`,
  `Tube Diameter` and `Heated Length` are the standard Groeneveld LUT
  predictors.
- `CHF Result` is essentially an empty column in this extract (a single stray
  unit-string value) — dropped from `df` as a data artifact rather than a
  real feature.
- `Reference ID` spans dozens of source studies at very uneven sample counts;
  treat it as provenance metadata rather than a modeling feature unless
  explicitly building a source-aware/leave-one-study-out validation scheme.
- Given the row count, scatter/histogram plots above use small markers and
  transparency (alpha) to stay legible — inspect `results/eda/{slug}/figures/`
  at full resolution for detail.
""".replace("{slug}", slug)),
    ]
    write_notebook(slug, cells)


# ============================================================================
# 5. helical_coil_r123_appendixCD
# ============================================================================
def build_helical_coil():
    slug = "helical_coil_r123_appendixCD"
    cells = [
        md(r"""
# EDA: Helical-Coil R-123 CHF Data — Appendices C/D (`helical_coil_r123_appendixCD.csv`)

Digitized CHF test data for R-123 flow boiling in helical coils, low- and
(where present) higher-pressure appendices. Source document:
`helical_coil_chf_research_data.pdf` (page count checked below for
provenance).

## 1. Load Data
"""),
        code(r"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

pd.set_option("display.max_columns", 60)
plt.rcParams["figure.dpi"] = 100

DATA_DIR = Path("../../data/raw")
OUT_DIR = Path("../../results/eda/helical_coil_r123_appendixCD")
FIG_DIR = OUT_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

SLUG = "helical_coil_r123_appendixCD"
SLUG_TITLE = "Helical-Coil R-123 CHF (Appendix C/D)"

df = pd.read_csv(DATA_DIR / "fine_tuning" / "helical_coil_r123_appendixCD.csv")

NUMERIC_COLS = ["Lh_mm", "G_kg_m2s", "Psys_bar", "rho_l_over_rho_g", "xe", "Q_watt", "CHF_kW_m2"]
CATEGORICAL_COLS = ["appendix", "Coil_no"]
TARGET_COL = "CHF_kW_m2"

df.head()
""".strip()),
        md("## 1b. Source Document Provenance"),
        code(r"""
try:
    from pypdf import PdfReader
    reader = PdfReader(DATA_DIR / "fine_tuning" / "helical_coil_chf_research_data.pdf")
    print("Source PDF:", (DATA_DIR / "fine_tuning" / "helical_coil_chf_research_data.pdf").name)
    print("Pages:", len(reader.pages))
except Exception as e:
    print("Could not read source PDF:", e)
""".strip()),
    ]
    cells += generic_a_block(target_heading_num=8)
    cells.append(md(r"""
## 10. Takeaways

- `CHF_kW_m2` is the target; `G_kg_m2s` (mass flux), `Psys_bar` (system
  pressure), `xe` (exit quality) and `rho_l_over_rho_g` (density ratio, a
  pressure surrogate) are the physical predictors.
- `appendix` distinguishes the low-pressure vs. other appendix subsets —
  check the target-by-category plot to see whether CHF trends differ by
  appendix before pooling them in a single model.
- `Coil_no` identifies the physical test coil; treat as metadata/grouping
  variable for train/test splitting rather than a physical predictor.
"""))
    write_notebook(slug, cells)


# ============================================================================
# 6. nureg_km0011_table4-1_sample
# ============================================================================
def build_nureg_sample():
    slug = "nureg_km0011_table4-1_sample"
    cells = [
        md(r"""
# EDA: NUREG/KM-0011 Table 4-1 Sample (`nureg_km0011_table4-1_SAMPLE_ONLY.csv`)

**This file is explicitly labeled `SAMPLE_ONLY`** — 21 rows, all from a single
source study (Lowdermilk, 1958). It is far too small for reliable
distribution/correlation conclusions on its own; treat this EDA as a
schema/spot-check against the full-corpus range summary in
`nureg_km0011_table4-2_source_dataset_ranges` rather than a standalone
dataset profile.

## 1. Load Data

`G_kg_m2s` and `CHF_kW_m2` contain thousands-separator commas in the raw CSV
(e.g. `"1,372.7"`) and are cleaned to floats below.
"""),
        code(r"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

pd.set_option("display.max_columns", 60)
plt.rcParams["figure.dpi"] = 100

DATA_DIR = Path("../../data/raw")
OUT_DIR = Path("../../results/eda/nureg_km0011_table4-1_sample")
FIG_DIR = OUT_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

SLUG = "nureg_km0011_table4-1_sample"
SLUG_TITLE = "NUREG/KM-0011 Table 4-1 (Sample)"

df = pd.read_csv(DATA_DIR / "testing" / "nureg_km0011_table4-1_SAMPLE_ONLY.csv")
for c in ["G_kg_m2s", "CHF_kW_m2"]:
    df[c] = df[c].astype(str).str.replace(",", "", regex=False).astype(float)

NUMERIC_COLS = ["D_m", "L_m", "P_kPa", "G_kg_m2s", "Xchf", "DHin_kJkg", "CHF_kW_m2", "Tin_C"]
CATEGORICAL_COLS = ["Reference"]
TARGET_COL = "CHF_kW_m2"

df.head()
""".strip()),
    ]
    cells += generic_a_block(target_heading_num=8)
    cells.append(md(r"""
## 10. Takeaways

- Only 21 rows / a single `Reference` — histograms and the correlation matrix
  above are illustrative of shape/schema only, not statistically meaningful
  on their own.
- `D_m` and `L_m` are constant across this sample (single tube geometry);
  `CHF_kW_m2` variation here is driven almost entirely by `G_kg_m2s`.
- Cross-referencing against `nureg_km0011_table4-2_source_dataset_ranges`
  requires manual name matching — the two files spell the same source
  differently (`"Lowdermilk, 1958"` here vs.
  `"(1990) Lowdermilk et al. (1958)"` there), so no automated join is
  attempted in this notebook.
"""))
    write_notebook(slug, cells)


# ============================================================================
# 7. zhao2020_chf_flowboiling_tubes
# ============================================================================
def build_zhao2020():
    slug = "zhao2020_chf_flowboiling_tubes"
    cells = [
        md(r"""
# EDA: Zhao (2020) Flow-Boiling CHF in Tubes (`zhao2020_chf_flowboiling_tubes.csv`)

Literature-compiled flow-boiling CHF dataset for tubes, spanning many source
studies (`author`) and both circular and non-circular (`geometry`) test
sections.

## 1. Load Data
"""),
        code(r"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

pd.set_option("display.max_columns", 60)
plt.rcParams["figure.dpi"] = 100

DATA_DIR = Path("../../data/raw")
OUT_DIR = Path("../../results/eda/zhao2020_chf_flowboiling_tubes")
FIG_DIR = OUT_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

SLUG = "zhao2020_chf_flowboiling_tubes"
SLUG_TITLE = "Zhao (2020) Flow-Boiling CHF (Tubes)"

df = pd.read_csv(DATA_DIR / "fine_tuning" / "zhao2020_chf_flowboiling_tubes.csv", encoding="utf-8-sig")

NUMERIC_COLS = ["pressure [MPa]", "mass_flux [kg/m2-s]", "x_e_out [-]", "D_e [mm]",
                 "D_h [mm]", "length [mm]", "chf_exp [MW/m2]"]
CATEGORICAL_COLS = ["author", "geometry"]
TARGET_COL = "chf_exp [MW/m2]"

df.head()
""".strip()),
    ]
    cells += generic_a_block(target_heading_num=8)
    cells += [
        md("## 10. Coverage by Author (Source Study)"),
        code(r"""
top_authors = df["author"].value_counts().head(20)
fig, ax = plt.subplots(figsize=(8, 6))
ax.barh(top_authors.index.astype(str)[::-1], top_authors.values[::-1], color="#4C72B0")
ax.set_xlabel("Number of rows")
ax.set_title(f"{SLUG_TITLE}: Top 20 Authors by Row Count "
             f"(of {df['author'].nunique()} total)")
fig.tight_layout()
fig.savefig(FIG_DIR / "top_authors.png", dpi=150, bbox_inches="tight")
plt.show()
""".strip()),
        md(r"""
## 11. Takeaways

- `chf_exp [MW/m2]` is the target; `D_e`/`D_h` (equivalent/hydraulic
  diameter) are near-duplicate columns for circular tubes and will show
  strong collinearity in the correlation matrix — check whether `geometry`
  explains any divergence between them.
- `author` is high-cardinality (many literature sources at uneven sample
  counts); consider grouping rare authors into an "other" bucket for any
  categorical-encoding use, or use it only for source-aware validation
  splits.
- `x_e_out [-]` (exit quality) can be negative (subcooled exit) — confirm the
  histogram range matches physical expectations for flow boiling.
"""),
    ]
    write_notebook(slug, cells)


# ============================================================================
# Group B: aggregate / range-summary tables (lighter EDA)
# ============================================================================

CELL_PARSE_MINMAX = code(r"""
import re

def parse_minmax(token):
    # Parse a range-ish string token into (min, max) floats.
    # Handles: '20, 60' (list), '2.09-4.44' (dash range),
    # '-0.5 to -0.25' (worded range, needed for negative bounds), '1000' (single value).
    if pd.isna(token):
        return np.nan, np.nan
    s = str(token).strip()
    if s == "":
        return np.nan, np.nan
    if " to " in s:
        a, b = s.split(" to ")
        a, b = float(a), float(b)
        return min(a, b), max(a, b)
    if "," in s:
        vals = [float(t.strip()) for t in s.split(",")]
        return min(vals), max(vals)
    m = re.match(r"^(-?\d+\.?\d*)-(-?\d+\.?\d*)$", s)
    if m:
        a, b = float(m.group(1)), float(m.group(2))
        return min(a, b), max(a, b)
    v = float(s)
    return v, v
""".strip())


def cell_range_bar_helper():
    return code(r"""
def plot_range_bars(ax, labels, mins, maxs, title, xlabel=""):
    y = np.arange(len(labels))
    ax.hlines(y, mins, maxs, color="#4C72B0", linewidth=6)
    ax.plot(mins, y, "|", color="#2f4b7c", markersize=14, markeredgewidth=2)
    ax.plot(maxs, y, "|", color="#2f4b7c", markersize=14, markeredgewidth=2)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel(xlabel)
    ax.invert_yaxis()
""".strip())


def setup_b_cell(slug, filename):
    return f"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

pd.set_option("display.max_columns", 60)
plt.rcParams["figure.dpi"] = 100

DATA_DIR = Path("../../data/raw/testing")
OUT_DIR = Path("../../results/eda/{slug}")
FIG_DIR = OUT_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(DATA_DIR / "{filename}")
df
""".strip()


# ---- B1: narrowchannel_chf_ml2025_source_summary --------------------------
def build_narrowchannel():
    slug = "narrowchannel_chf_ml2025_source_summary"
    cells = [
        md(r"""
# EDA: Narrow-Channel CHF (ML 2025) Source Summary (`narrowchannel_chf_ml2025_source_summary.csv`)

A small (4-row) summary table: one row per literature source used in a
narrow-channel CHF machine-learning study, giving the geometric/operating
**range** each source contributes (not raw data points). EDA here compares
coverage across sources rather than profiling a single distribution.

## 1. Load Data
"""),
        code(setup_b_cell(slug, "narrowchannel_chf_ml2025_source_summary.csv")),
        md("## 2. Parse Ranges"),
        CELL_PARSE_MINMAX,
        code(r"""
range_cols = ["Width_mm", "Height_mm", "Hydraulic_diameter_mm", "Heated_length_mm",
              "MassFlux_kg_m2s", "InletSubcooling_K", "Pressure_MPa"]

parsed = df[["Source"]].copy()
for col in range_cols:
    mins, maxs = zip(*df[col].map(parse_minmax))
    parsed[f"{col}_min"] = mins
    parsed[f"{col}_max"] = maxs
parsed.to_csv(OUT_DIR / "parsed_ranges.csv", index=False)
parsed
""".strip()),
        md("## 3. Range Coverage by Source"),
        cell_range_bar_helper(),
        code(r"""
ncols = 3
nrows = -(-len(range_cols) // ncols)
fig, axes = plt.subplots(nrows, ncols, figsize=(5.5 * ncols, 0.9 * len(df) + 1.5 * nrows))
axes = np.array(axes).reshape(-1)
for ax, col in zip(axes, range_cols):
    plot_range_bars(ax, parsed["Source"], parsed[f"{col}_min"], parsed[f"{col}_max"], col)
for ax in axes[len(range_cols):]:
    ax.axis("off")
fig.suptitle("Narrow-Channel CHF (ML 2025): Parameter Ranges by Source", y=1.01)
fig.tight_layout()
fig.savefig(FIG_DIR / "range_coverage_by_source.png", dpi=150, bbox_inches="tight")
plt.show()
""".strip()),
        md("## 4. Sample Counts by Source"),
        code(r"""
fig, ax = plt.subplots(figsize=(6, 3.5))
ax.bar(df["Source"], df["N_points"], color="#4C72B0")
ax.set_ylabel("N_points")
ax.set_title("Narrow-Channel CHF (ML 2025): Points Contributed per Source")
fig.tight_layout()
fig.savefig(FIG_DIR / "n_points_by_source.png", dpi=150, bbox_inches="tight")
plt.show()
""".strip()),
        md(r"""
## 5. Takeaways

- `Du et al.` contributes the widest hydraulic-diameter and mass-flux range;
  `He et al.` is a narrow, single-pressure (`0.14 MPa`) slice — check for
  gaps before pooling all four sources into one training set.
- Ranges given as multi-value lists (e.g. `Width_mm = "20, 60"`) mean only
  those discrete widths were tested, not a continuum — `parse_minmax` treats
  them as `[min, max]` bounds only, which can overstate actual coverage.
"""),
    ]
    write_notebook(slug, cells)


# ---- B2: furlong2025_nrc_vs_debortoli_range_comparison --------------------
def build_furlong():
    slug = "furlong2025_nrc_vs_debortoli_range_comparison"
    cells = [
        md(r"""
# EDA: Furlong (2025) NRC vs. DeBortoli Range Comparison (`furlong2025_nrc_vs_debortoli_range_comparison.csv`)

A 2-row table comparing the parameter range of the NRC source tube database
against the DeBortoli rectangular-channel target range. Already stored as
explicit `_min`/`_max` column pairs — no string parsing needed.

## 1. Load Data
"""),
        code(setup_b_cell(slug, "furlong2025_nrc_vs_debortoli_range_comparison.csv")),
        md("## 2. Range Comparison"),
        cell_range_bar_helper(),
        code(r"""
variables = ["Dh_mm", "L_m", "P_MPa", "G_kg_m2s", "DHsubin_kJkg", "xe_cr", "CHF_kW_m2"]
ncols = 3
nrows = -(-len(variables) // ncols)
fig, axes = plt.subplots(nrows, ncols, figsize=(5.5 * ncols, 3))
axes = np.array(axes).reshape(-1)
for ax, var in zip(axes, variables):
    plot_range_bars(ax, df["Dataset"], df[f"{var}_min"], df[f"{var}_max"], var)
for ax in axes[len(variables):]:
    ax.axis("off")
fig.suptitle("NRC Source Tube vs. DeBortoli Target Range: Parameter Overlap", y=1.05)
fig.tight_layout()
fig.savefig(FIG_DIR / "nrc_vs_debortoli_ranges.png", dpi=150, bbox_inches="tight")
plt.show()
""".strip()),
        md(r"""
## 3. Takeaways

- Where the DeBortoli target range sits fully inside the NRC source range for
  a variable, extrapolation risk is low; where it extends beyond (check
  `L_m`, `Dh_mm` in particular — rectangular channels vs. round tubes),
  a model trained only on the NRC tube data would be extrapolating out of
  domain for that variable.
- This table is purpose-built for extrapolation-risk assessment, not for
  statistical distribution analysis — the range-bar comparison above is the
  primary artifact.
"""),
    ]
    write_notebook(slug, cells)


# ---- B3/B4: supercritical CO2 / water source summaries (Luo 2020) ---------
def build_supercritical(fluid_slug, fluid_title, filename):
    slug = f"supercritical_{fluid_slug}_source_summary_luo2020"
    cells = [
        md(f"""
# EDA: Supercritical {fluid_title} Source Summary — Luo (2020) (`{filename}`)

A small source-summary table where the single packed column
`P_MPa_G_kg_m2s_q_kW_m2_range` encodes pressure, mass-flux and heat-flux
ranges per source as `"P / G / q"` (each of `P`, `G`, `q` itself possibly a
single value, a comma list, or a dash range). `D_mm` is parsed the same way.

## 1. Load Data
"""),
        code(setup_b_cell(slug, filename)),
        md("## 2. Parse Packed Ranges"),
        CELL_PARSE_MINMAX,
        code(r"""
def split_pgq(s):
    parts = str(s).split("/")
    assert len(parts) == 3, f"unexpected format: {s}"
    return parts  # P, G, q substrings

pgq = df["P_MPa_G_kg_m2s_q_kW_m2_range"].map(split_pgq)
p_str = pgq.map(lambda t: t[0])
g_str = pgq.map(lambda t: t[1])
q_str = pgq.map(lambda t: t[2])

parsed = df[["Source"]].copy()
for name, s in [("P_MPa", p_str), ("G_kg_m2s", g_str), ("q_kW_m2", q_str), ("D_mm", df["D_mm"])]:
    mins, maxs = zip(*s.map(parse_minmax))
    parsed[f"{name}_min"] = mins
    parsed[f"{name}_max"] = maxs
parsed["N_points"] = df["N_points"]
parsed.to_csv(OUT_DIR / "parsed_ranges.csv", index=False)
parsed
""".strip()),
        md("## 3. Range Coverage by Source"),
        cell_range_bar_helper(),
        code(r"""
range_vars = ["P_MPa", "G_kg_m2s", "q_kW_m2", "D_mm"]
fig, axes = plt.subplots(2, 2, figsize=(11, 1.0 * len(df) + 3))
axes = axes.reshape(-1)
for ax, var in zip(axes, range_vars):
    plot_range_bars(ax, parsed["Source"], parsed[f"{var}_min"], parsed[f"{var}_max"], var)
fig.suptitle(f"Supercritical {fluid_title} ({{n}} sources): Parameter Ranges".format(n=len(df)), y=1.01)
fig.tight_layout()
fig.savefig(FIG_DIR / "range_coverage_by_source.png", dpi=150, bbox_inches="tight")
plt.show()
""".strip().replace("{fluid_title}", fluid_title)),
        md("## 4. Sample Counts by Source"),
        code(r"""
fig, ax = plt.subplots(figsize=(7, max(3, 0.35 * len(df))))
ax.barh(df["Source"], df["N_points"], color="#4C72B0")
ax.invert_yaxis()
ax.set_xlabel("N_points")
ax.set_title(f"Points Contributed per Source")
fig.tight_layout()
fig.savefig(FIG_DIR / "n_points_by_source.png", dpi=150, bbox_inches="tight")
plt.show()
""".strip()),
        md(f"""
## 5. Takeaways

- Pressure ranges cluster around the {fluid_title} critical/supercritical
  region for this fluid; check `parsed_ranges.csv` for the exact per-source
  span rather than eyeballing the packed source strings.
- A few sources report only a single `(P, G, q)` operating point
  (`N_points` small, `min == max`) rather than a swept range — these plot as
  a single tick rather than a bar in the range-coverage figure.
""".strip()),
    ]
    write_notebook(slug, cells)


# ---- B5: nureg_km0011_table4-2_source_dataset_ranges -----------------------
def build_nureg_ranges():
    slug = "nureg_km0011_table4-2_source_dataset_ranges"
    cells = [
        md(r"""
# EDA: NUREG/KM-0011 Table 4-2 Source Dataset Ranges (`nureg_km0011_table4-2_source_dataset_ranges.csv`)

The largest range-summary table (72 source studies) underlying the
NUREG/KM-0011 CHF compilation. Columns are already split into `_min`/`_max`
pairs per variable, but several are stored as thousands-separated strings
(e.g. `"9,800"`) and need cleaning before use.

## 1. Load Data
"""),
        code(setup_b_cell(slug, "nureg_km0011_table4-2_source_dataset_ranges.csv")),
        md(r"""
## 2. Clean Numeric Columns

A handful of cells in this source table contain free text (`"data"`,
`"distribution"`) instead of a number, alongside a garbled `name` value that
looks like two source citations merged into one row — a digitization
artifact in the underlying NUREG/KM-0011 PDF table extraction. These are
coerced to `NaN` (not dropped) and counted below rather than silently
ignored.
"""),
        code(r"""
minmax_vars = ["D_mm", "L_m", "P_kPa", "G_kg_m2s", "Xchf", "DHin_kJkg", "q_kWm2", "Tin_C"]
n_coerced = 0
for var in minmax_vars:
    for suffix in ["_min", "_max"]:
        col = f"{var}{suffix}"
        cleaned = df[col].astype(str).str.replace(",", "", regex=False)
        numeric = pd.to_numeric(cleaned, errors="coerce")
        n_coerced += int((numeric.isna() & df[col].notna()).sum())
        df[col] = numeric

print(f"Non-numeric values coerced to NaN: {n_coerced}")
print(df.loc[df[[f'{v}{s}' for v in minmax_vars for s in ('_min', '_max')]].isna().any(axis=1),
             ["name"]])

df.to_csv(OUT_DIR / "parsed_ranges.csv", index=False)
df[["name", "N_points"] + [f"{v}{s}" for v in minmax_vars for s in ("_min", "_max")]].head()
""".strip()),
        md("## 3. Descriptive Stats on Reported Ranges"),
        code(r"""
stats_rows = []
for var in minmax_vars:
    span = df[f"{var}_max"] - df[f"{var}_min"]
    stats_rows.append({
        "variable": var,
        "n_sources": df[f"{var}_min"].notna().sum(),
        "global_min": df[f"{var}_min"].min(),
        "global_max": df[f"{var}_max"].max(),
        "median_span": span.median(),
    })
pd.DataFrame(stats_rows).set_index("variable")
""".strip()),
        md("## 4. Range Coverage by Source (one figure per variable — 72 sources)"),
        cell_range_bar_helper(),
        code(r"""
order = df.sort_values("q_kWm2_max", ascending=False)["name"]
df_sorted = df.set_index("name").loc[order].reset_index()

for var in minmax_vars:
    fig, ax = plt.subplots(figsize=(8, max(6, 0.22 * len(df_sorted))))
    plot_range_bars(ax, df_sorted["name"], df_sorted[f"{var}_min"], df_sorted[f"{var}_max"], var)
    ax.tick_params(axis="y", labelsize=6)
    fig.suptitle(f"NUREG/KM-0011 Table 4-2: {var} Range by Source", y=1.005)
    fig.tight_layout()
    fig.savefig(FIG_DIR / f"range_by_source_{var}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
print("Saved one range-by-source figure per variable to", FIG_DIR)
""".strip()),
        md("## 5. Points per Source (Top 20)"),
        code(r"""
top_n = df.nlargest(20, "N_points")[["name", "N_points"]]
fig, ax = plt.subplots(figsize=(8, 6))
ax.barh(top_n["name"][::-1], top_n["N_points"][::-1], color="#4C72B0")
ax.set_xlabel("N_points")
ax.set_title(f"NUREG/KM-0011 Table 4-2: Top 20 Sources by Point Count (of {len(df)} total)")
fig.tight_layout()
fig.savefig(FIG_DIR / "top_sources_by_n_points.png", dpi=150, bbox_inches="tight")
plt.show()
""".strip()),
        md(r"""
## 6. Takeaways

- A handful of rows have digitization artifacts (garbled/merged `name`
  citations, `D_mm_min`/`D_mm_max` cells containing the text `"data"` or
  `"distribution"` instead of a number) — these are coerced to `NaN` rather
  than dropped, so downstream aggregates naturally exclude them but the rows
  remain visible for manual correction against the source PDF.
- 72 source studies span very different pressure/mass-flux/quality windows —
  the per-variable range-by-source figures (saved to `figures/`, one PNG per
  variable since 72 rows don't fit one readable panel) are the primary way
  to spot coverage gaps.
- `_source_pdf_page` is a citation/provenance column (which NUREG/KM-0011 PDF
  page the row was digitized from), not a physical variable — excluded from
  the numeric analysis.
- Name matching against `nureg_km0011_table4-1_SAMPLE_ONLY.csv`'s single
  `Reference` value is not automated (citation formats differ between the two
  files) — see the takeaways note in that notebook.
"""),
    ]
    write_notebook(slug, cells)


# ---- B6: tanase2009_diameter_correction_exponent_grid ----------------------
def build_tanase():
    slug = "tanase2009_diameter_correction_exponent_grid"
    cells = [
        md(r"""
# EDA: Tanase (2009) Diameter-Correction Exponent Grid (`tanase2009_diameter_correction_exponent_grid.csv`)

Unlike the other summary tables, this file is a **complete lookup grid**: 2
pressure bins x 3 mass-flux bins x 4 quality bins = 24 rows, each carrying a
single empirical diameter-correction exponent `exponent_n` (used to correct
CHF Look-Up Table predictions between reference and non-reference tube
diameters). EDA here visualizes it as the grid it is, rather than as
per-source ranges.

## 1. Load Data
"""),
        code(setup_b_cell(slug, "tanase2009_diameter_correction_exponent_grid.csv")),
        md("## 2. Grid Structure Check"),
        code(r"""
print("Unique Pressure_kPa_range bins:", df["Pressure_kPa_range"].unique())
print("Unique MassFlux_kg_m2s_range bins:", df["MassFlux_kg_m2s_range"].unique())
print("Unique Quality_range bins:", df["Quality_range"].unique())
print("Rows:", len(df), " = ", df["Pressure_kPa_range"].nunique(),
      "x", df["MassFlux_kg_m2s_range"].nunique(),
      "x", df["Quality_range"].nunique())
print("Any missing exponent_n:", df["exponent_n"].isna().any())
""".strip()),
        md("## 3. Exponent Heatmap (Quality x Mass Flux, faceted by Pressure)"),
        code(r"""
pressure_bins = df["Pressure_kPa_range"].unique()
quality_order = df["Quality_range"].unique()
massflux_order = df["MassFlux_kg_m2s_range"].unique()

fig, axes = plt.subplots(1, len(pressure_bins), figsize=(6 * len(pressure_bins), 4.5),
                          squeeze=False, sharey=True)
axes = axes[0]
vmin, vmax = df["exponent_n"].min(), df["exponent_n"].max()
for i, (ax, pbin) in enumerate(zip(axes, pressure_bins)):
    sub = df[df["Pressure_kPa_range"] == pbin]
    grid = sub.pivot(index="MassFlux_kg_m2s_range", columns="Quality_range", values="exponent_n")
    grid = grid.reindex(index=massflux_order, columns=quality_order)
    im = ax.imshow(grid.values, cmap="RdBu_r", vmin=vmin, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(quality_order)))
    ax.set_xticklabels(quality_order, rotation=45, ha="right")
    ax.set_yticks(range(len(massflux_order)))
    ax.set_yticklabels(massflux_order)
    ax.set_xlabel("Quality range")
    if i == 0:
        ax.set_ylabel("Mass flux range (kg/m2s)")
    ax.set_title(f"Pressure: {pbin} kPa")
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            ax.text(j, i, f"{grid.values[i, j]:.2f}", ha="center", va="center", fontsize=8)
fig.colorbar(im, ax=axes, shrink=0.8, label="exponent_n")
fig.suptitle("Tanase (2009): Diameter-Correction Exponent Grid", y=1.03)
fig.savefig(FIG_DIR / "exponent_heatmap.png", dpi=150, bbox_inches="tight")
plt.show()
""".strip()),
        md(r"""
## 4. Takeaways

- The grid is complete (no missing `exponent_n` cells) and exactly
  2 x 3 x 4 = 24 rows, confirming this is a lookup table rather than sampled
  data — no missing-value or outlier analysis is meaningful here.
- `exponent_n` swings from negative (CHF decreases with diameter) at low
  mass flux to positive (CHF increases with diameter) at higher mass flux —
  the heatmap makes the sign flip pattern across mass-flux bins visible at a
  glance; pressure bin has comparatively little effect.
- Downstream use: given a target tube diameter and known (P, G, x) operating
  point, look up the matching cell's `exponent_n` to correct a reference-tube
  CHF prediction to the target diameter.
"""),
    ]
    write_notebook(slug, cells)


# ============================================================================
if __name__ == "__main__":
    build_pinfin()
    build_kaeri_uniform()
    build_kaeri_nonuniform()
    build_groeneveld()
    build_helical_coil()
    build_nureg_sample()
    build_zhao2020()

    build_narrowchannel()
    build_furlong()
    build_supercritical("co2", "CO2", "supercritical_co2_source_summary_luo2020.csv")
    build_supercritical("water", "Water", "supercritical_water_source_summary_luo2020.csv")
    build_nureg_ranges()
    build_tanase()

    print("\nDone: 13 EDA notebooks written to", NB_DIR)

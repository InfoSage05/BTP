"""Flow-boiling dataset loaders with a hard leakage whitelist.

One loader per dataset. Each returns a Dataset carrying:

  df            harmonised frame (kPa, kg/m2s, mm, kW/m2)
  local         feature list INCLUDING outlet/critical quality
  inlet         feature list EXCLUDING it (None where the dataset cannot support it)
  target        always 'CHF_kW_m2'
  group         column identifying the source study/campaign, or None
  banned        columns proven to reconstruct the target; never fed to a model

`banned` is not advisory. verify_leakage() asserts each banned column really does
recover the target, so the exclusion is evidenced rather than assumed. Pool-boiling
datasets are deliberately absent -- this module is flow boiling only.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

SHARE = os.path.join(os.path.dirname(__file__), "..", "..", "share")
FLOW = os.path.join(SHARE, "02_flow_boiling")


@dataclass
class Dataset:
    name: str
    df: pd.DataFrame
    local: list
    inlet: list | None
    group: str | None
    fluid: str
    geometry: str
    banned: list = field(default_factory=list)
    note: str = ""

    @property
    def n(self) -> int:
        return len(self.df)

    @property
    def reduced(self) -> list:
        """Local features minus outlet quality, with no inlet substitute.

        Available for every dataset, unlike the inlet form, which needs a
        published inlet temperature or subcooling. Answers: how well can CHF be
        predicted from geometry and flow conditions alone?
        """
        return [f for f in self.local if f != "X"]

    def features(self, form: str) -> list | None:
        return {"local": self.local, "inlet": self.inlet, "reduced": self.reduced}[form]


def _read(path: str) -> pd.DataFrame:
    return pd.read_csv(path, encoding="utf-8-sig")


# --------------------------------------------------------------------------
# D1  NRC / Groeneveld, deduplicated
# --------------------------------------------------------------------------
def load_nrc() -> Dataset:
    d = _read(os.path.join(FLOW, "nrc_groeneveld_deduplicated_24443.csv"))
    out = pd.DataFrame({
        "D_mm": d["D_m"] * 1000.0,
        "L_mm": d["L_m"] * 1000.0,
        "P_kPa": d["P_kPa"],
        "G_kg_m2s": d["G_kg_m2s"],
        "X": d["X"],
        "dHin_sub_kJkg": d["dHin_sub_kJkg"],
        "Tin_C": d["Tin_C"],
        "CHF_kW_m2": d["CHF_kW_m2"],
        "group": "ref_" + d["ref_id"].astype(str),
    })
    out["LtoD"] = out["L_mm"] / out["D_mm"]
    return Dataset(
        name="D1_NRC",
        df=out,
        local=["D_mm", "L_mm", "LtoD", "P_kPa", "G_kg_m2s", "X"],
        inlet=["D_mm", "L_mm", "LtoD", "P_kPa", "G_kg_m2s", "dHin_sub_kJkg", "Tin_C"],
        group="group",
        fluid="water",
        geometry="vertical_tube",
        note="X is back-computed from CHF via the energy balance; see Exhibit A.",
    )


# --------------------------------------------------------------------------
# D2  Zhao 2020 compilation
# --------------------------------------------------------------------------
def load_zhao() -> Dataset:
    d = _read(os.path.join(FLOW, "zhao2020_chf_flowboiling_tubes.csv"))
    out = pd.DataFrame({
        "P_kPa": d["pressure [MPa]"] * 1000.0,
        "G_kg_m2s": d["mass_flux [kg/m2-s]"],
        "X": d["x_e_out [-]"],
        # D_mm must exist or enrich() silently substitutes a hard-coded 8 mm for
        # every row. Zhao's true diameters span 1-37.5 mm, so that default made the
        # physics backbone meaningless on all 1,865 rows. D_e (heated equivalent) is
        # used: measured against D_h it gives a better Katto baseline (R2 -0.350 vs
        # -0.362) and a better held-out pipeline score (-0.415 vs -0.507).
        "D_mm": d["D_e [mm]"],
        "De_mm": d["D_e [mm]"],
        "Dh_mm": d["D_h [mm]"],
        "L_mm": d["length [mm]"],
        "CHF_kW_m2": d["chf_exp [MW/m2]"] * 1000.0,
        "geometry_cat": d["geometry"],
        "group": "zhao_" + d["author"].astype(str),
    })
    out["LtoD"] = out["L_mm"] / out["De_mm"]
    # One row is a plate at G = 0, i.e. pool boiling inside a flow-boiling corpus.
    # Katto-Ohno and Biasi both divide by G, so it has no defined physics baseline.
    n_before = len(out)
    out = out[out["G_kg_m2s"] > 0].reset_index(drop=True)
    assert n_before - len(out) == 1, f"expected to drop exactly 1 G=0 row, dropped {n_before-len(out)}"
    return Dataset(
        name="D2_Zhao2020",
        df=out,
        local=["P_kPa", "G_kg_m2s", "X", "De_mm", "Dh_mm", "L_mm", "LtoD", "geometry_cat"],
        inlet=None,  # no inlet temperature or subcooling is reported
        group="group",
        fluid="water",
        geometry="tube_annulus_plate",
        note="No inlet enthalpy reported, so the inlet form cannot be built.",
    )


# --------------------------------------------------------------------------
# D3 / D4  KAERI TR-1665
#   Power/(Perimeter*Length) reconstructs HeatFlux exactly -> banned.
# --------------------------------------------------------------------------
_KAERI_BANNED = ["Power", "Perimeter", "Area", "MassFlow", "InletEnthalpy"]


def load_kaeri_uniform() -> Dataset:
    d = _read(os.path.join(FLOW, "kaeri_tr1665_uniform_chf.csv"))
    out = pd.DataFrame({
        "D_mm": d["Diameter"] * 1000.0,
        "L_mm": d["Length"] * 1000.0,
        "P_kPa": d["Pressure"] / 1000.0,
        "G_kg_m2s": d["MassFlux"],
        "X": d["EquilibriumQuality"],
        "Tin_C": d["InletTemperature"],
        "WallMesh": d["WallMesh"],
        "CHF_kW_m2": d["HeatFlux"] / 1000.0,
        "group": "kaeri_" + d["Source"].astype(str),
    })
    out["LtoD"] = out["L_mm"] / out["D_mm"]
    return Dataset(
        name="D3_KAERI_uniform",
        df=out,
        local=["D_mm", "L_mm", "LtoD", "P_kPa", "G_kg_m2s", "X", "WallMesh"],
        inlet=["D_mm", "L_mm", "LtoD", "P_kPa", "G_kg_m2s", "Tin_C", "WallMesh"],
        group="group",
        fluid="water",
        geometry="tube",
        banned=_KAERI_BANNED,
    )


def load_kaeri_nonuniform() -> Dataset:
    d = _read(os.path.join(FLOW, "kaeri_tr1665_nonuniform_chf.csv"))
    out = pd.DataFrame({
        "D_mm": d["Diameter"] * 1000.0,
        "L_mm": d["Length"] * 1000.0,
        "P_kPa": d["Pressure"] / 1000.0,
        "G_kg_m2s": d["MassFlux"],
        # VERIFIED inlet quality, not outlet: (h_in - h_f)/h_fg reproduces this
        # column on all 888 rows to within 0.001 (r = 0.999999). It is an
        # independently measured inlet condition, so it is NOT circular and it
        # belongs in the inlet and reduced feature sets, unlike every other
        # dataset's quality column.
        "X_inlet": d["Quality"],
        "Tin_C": d["InletTemperature"],
        "WallPower": d["WallPower"],
        "WallMesh": d["WallMesh"],
        "Shape": d["Shape"],
        "Continuous": d["Continuous"],
        "CHF_kW_m2": d["HeatFlux"] / 1000.0,
        "group": "kaeri_" + d["Source"].astype(str),
    })
    out["LtoD"] = out["L_mm"] / out["D_mm"]
    base = ["D_mm", "L_mm", "LtoD", "P_kPa", "G_kg_m2s", "WallPower", "WallMesh",
            "Shape", "Continuous", "X_inlet"]
    return Dataset(
        name="D4_KAERI_nonuniform",
        df=out,
        # no outlet quality is published for this set, so there is no circular
        # "local" form to build -- all three forms carry the inlet quality.
        local=base,
        inlet=base + ["Tin_C"],
        group="group",
        fluid="water",
        geometry="tube",
        # CHFLocation / QualityPosition are measured at burnout, i.e. outcomes.
        banned=_KAERI_BANNED + ["CHFLocation", "QualityPosition"],
    )


# --------------------------------------------------------------------------
# D5  Helical coil R-123, appendices C + D.  Q_watt reconstructs CHF -> banned.
#     Coil geometry for THIS paper is not verified, so Coil_no stays categorical.
# --------------------------------------------------------------------------
# Recovered by inverting Q_watt/(pi*d*L) = CHF per coil. Each coil returns a clean
# constant (sd ~0.002 mm), i.e. the true design diameter. Recovering a fixed geometric
# parameter is not target leakage -- Q_watt itself stays banned.
D5_COIL_D_MM = {"Coil_1": 9.5, "Coil_2": 7.5, "Coil_3": 5.5,
                "Coil_4": 9.5, "Coil_5": 7.5, "Coil_6": 5.4}


def load_helical_r123() -> Dataset:
    d = _read(os.path.join(FLOW, "helical_coil_r123_appendixCD.csv"))
    dia = d["Coil_no"].map(D5_COIL_D_MM)
    assert dia.notna().all(), "unmapped Coil_no"
    out = pd.DataFrame({
        "D_mm": dia,
        "L_mm": d["Lh_mm"],
        "G_kg_m2s": d["G_kg_m2s"],
        "P_kPa": d["Psys_bar"] * 100.0,
        "rho_ratio": d["rho_l_over_rho_g"],
        "X": d["xe"],
        "CHF_kW_m2": d["CHF_kW_m2"],
        "group": d["appendix"],
    })
    out["LtoD"] = out["L_mm"] / out["D_mm"]
    return Dataset(
        name="D5_Helical_R123",
        df=out,
        local=["D_mm", "L_mm", "LtoD", "G_kg_m2s", "P_kPa", "rho_ratio", "X"],
        inlet=None,  # no inlet temperature reported
        group="group",
        fluid="R-123",
        geometry="helical_coil",
        banned=["Q_watt"],
        note=("Tube diameter recovered per coil by inverting the heated-area relation. "
              "28 of 257 rows report exit quality above 1.0 (max 1.169), i.e. nominally "
              "superheated vapour, where dryout normally occurs at x <= 1. Retained as "
              "published and flagged rather than silently dropped."),
    )


# --------------------------------------------------------------------------
# D6  Hardik helical coils, water.  Q_W reconstructs CHF; CHF_ratio is CHF/LUT.
#     Coil -> inner diameter comes from the paper's own Table 4 (verified).
# --------------------------------------------------------------------------
COIL_D_MM = {"Coil_1": 6.0, "Coil_2": 6.0, "Coil_3": 8.0,
             "Coil_4": 8.0, "Coil_5": 9.7, "Coil_6": 10.0}


def load_hardik_helical_water() -> Dataset:
    d = _read(os.path.join(FLOW, "hardik2017_helical_coils_water_lowpressure_chf.csv"))
    dia = d["Coil_No"].map(COIL_D_MM)
    assert dia.notna().all(), "unmapped Coil_No"
    out = pd.DataFrame({
        "D_mm": dia,
        "L_mm": d["L_mm"],
        "P_kPa": d["P_bar"] * 100.0,
        "G_kg_m2s": d["G_kg_m2s"],
        "X": d["xe"],
        "CHF_kW_m2": d["CHF_kW_m2"],
        "group": d["Coil_No"],
    })
    out["LtoD"] = out["L_mm"] / out["D_mm"]
    return Dataset(
        name="D6_Hardik_helical_water",
        df=out,
        local=["D_mm", "L_mm", "LtoD", "P_kPa", "G_kg_m2s", "X"],
        inlet=None,  # no inlet temperature reported
        group="group",
        fluid="water",
        geometry="helical_coil",
        banned=["Q_W", "CHF_ratio"],
        note="CHF_ratio is CHF normalised by the 2006 LUT, i.e. a function of the target.",
    )


# --------------------------------------------------------------------------
# D7  Hardik straight tubes, R-123.  No leakage columns.
# --------------------------------------------------------------------------
def load_hardik_straight_r123() -> Dataset:
    d = _read(os.path.join(FLOW, "hardik2017_straight_tubes_r123_chf.csv"))
    out = pd.DataFrame({
        "D_mm": d["D_mm"],
        "L_mm": d["L_mm"],
        "P_kPa": d["P_bar"] * 100.0,
        "Tin_C": d["Tin_C"],
        "G_kg_m2s": d["G_kg_m2s"],
        "X": d["xe"],
        "CHF_kW_m2": d["CHF_kW_m2"],
    })
    out["LtoD"] = out["L_mm"] / out["D_mm"]
    return Dataset(
        name="D7_Hardik_straight_R123",
        df=out,
        local=["D_mm", "L_mm", "LtoD", "P_kPa", "G_kg_m2s", "X"],
        inlet=["D_mm", "L_mm", "LtoD", "P_kPa", "G_kg_m2s", "Tin_C"],
        group=None,
        fluid="R-123",
        geometry="horizontal_tube",
        note="55 rows: an 80/20 test set is 11 rows. Indicative only.",
    )


# --------------------------------------------------------------------------
# D8  Pioro R-134a, digitised.  Diameter is constant -> carries no information.
# --------------------------------------------------------------------------
def load_pioro() -> Dataset:
    d = _read(os.path.join(SHARE, "04_holdout_digitized",
                           "pioro2002_r134a_horizontal_vertical_chf_DIGITIZED.csv"))
    out = pd.DataFrame({
        # constant within this dataset, so absent from its own feature list, but
        # carried so the merged corpus has a diameter for these rows
        "D_mm": d["D_mm"],
        "P_kPa": d["P_MPa"] * 1000.0,
        "G_kg_m2s": d["G_kg_m2s"],
        "X": d["x_cr"],
        "orientation": d["orientation"],
        "CHF_kW_m2": d["CHF_kW_m2"],
        "group": d["source_figure"],
    })
    return Dataset(
        name="D8_Pioro_R134a",
        df=out,
        local=["P_kPa", "G_kg_m2s", "X", "orientation"],
        inlet=None,
        group="group",
        fluid="R-134a",
        geometry="horiz_and_vert_tube",
        note="Eye-digitised, +/-5-10%. Constant D=6.92mm dropped (no variance).",
    )


# --------------------------------------------------------------------------
# D9  2006 look-up table.  Tabulated, not measured. Local form only.
# --------------------------------------------------------------------------
def load_lut() -> Dataset:
    d = _read(os.path.join(SHARE, "05_lookup_table_reference",
                           "groeneveld_2006_lut_long_format.csv"))
    d = d[d["CHF"] > 0].reset_index(drop=True)  # drop the X=1.0 all-steam placeholder
    out = pd.DataFrame({
        "P_kPa": d["P"], "G_kg_m2s": d["G"], "X": d["X"], "CHF_kW_m2": d["CHF"],
    })
    return Dataset(
        name="D9_LUT2006",
        df=out,
        local=["P_kPa", "G_kg_m2s", "X"],
        inlet=None,
        group=None,
        fluid="water",
        geometry="normalised_8mm_tube",
        note="Smoothed table, not raw measurement. 504 zero rows at X=1.0 removed.",
    )


# --------------------------------------------------------------------------
# M  Merged flow-only corpus, rebuilt from the deduplicated NRC.
# --------------------------------------------------------------------------
def load_merged_flow() -> Dataset:
    parts = []
    for ds in (load_nrc(), load_zhao(), load_kaeri_uniform(), load_kaeri_nonuniform(),
               load_helical_r123(), load_hardik_helical_water(),
               load_hardik_straight_r123(), load_pioro()):
        f = ds.df.copy()
        if "De_mm" in f.columns and "D_mm" not in f.columns:
            f["D_mm"] = f["De_mm"]
        keep = ["D_mm", "L_mm", "LtoD", "P_kPa", "G_kg_m2s", "X", "CHF_kW_m2"]
        for c in keep:
            if c not in f.columns:
                f[c] = np.nan
        f = f[keep].copy()
        f["dataset"] = ds.name
        f["fluid"] = ds.fluid
        f["geometry"] = ds.geometry
        f["group"] = ds.name  # leave-one-dataset-out
        parts.append(f)
    out = pd.concat(parts, ignore_index=True)
    return Dataset(
        name="M_merged_flow",
        df=out,
        local=["D_mm", "L_mm", "LtoD", "P_kPa", "G_kg_m2s", "X", "fluid", "geometry"],
        inlet=None,
        group="group",
        fluid="mixed",
        geometry="mixed",
        note="Common schema across all eight flow sources; grouping is leave-one-dataset-out.",
    )


LOADERS = {
    "D1_NRC": load_nrc,
    "D2_Zhao2020": load_zhao,
    "D3_KAERI_uniform": load_kaeri_uniform,
    "D4_KAERI_nonuniform": load_kaeri_nonuniform,
    "D5_Helical_R123": load_helical_r123,
    "D6_Hardik_helical_water": load_hardik_helical_water,
    "D7_Hardik_straight_R123": load_hardik_straight_r123,
    "D8_Pioro_R134a": load_pioro,
    "D9_LUT2006": load_lut,
    "M_merged_flow": load_merged_flow,
}


def load_all() -> dict:
    return {k: fn() for k, fn in LOADERS.items()}

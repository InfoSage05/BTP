"""
Preprocess the raw "Short Helical Minichannel Evaporators" CHF test-rig logs.

Each raw file is a high-frequency (roughly 0.5 s cadence) time-series log of a
single burnout run: ~30 sensor channels (mass flow, pressures, several
thermocouples, applied electrical power). There is no pre-labelled CHF point,
no heated-area value, and no documentation file. This script:

1. Parses every raw log (fixing encoding / delimiter / decimal-comma issues)
   into a clean, uniform time-series CSV per run.
2. Parses the run metadata (material, geometry variant, run date) out of the
   folder/file names.
3. Detects a *candidate* CHF (burnout) onset point per run: the first
   sustained, abnormal upward excursion in any wall-thermocouple channel
   while the heater power is still being ramped. This is a heuristic, not a
   validated physical measurement -- every run gets a diagnostic plot so it
   can be checked by eye before being trusted for training.
4. Writes one master summary CSV: one row per run with the operating
   conditions and heater power at the detected CHF point.

Outputs:
    data/processed/helical_minichannel/timeseries/<run>.csv   (cleaned full logs)
    data/processed/helical_minichannel/figures/<run>.png      (QC plot)
    data/processed/helical_minichannel/minichannel_chf_summary.csv (one row/run)
"""
import re
import glob
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RAW_ROOT = "data/raw/helical_minichannel/CHF_Experiments_Raw_Data"
OUT_ROOT = "data/processed/helical_minichannel"
TS_DIR = os.path.join(OUT_ROOT, "timeseries")
FIG_DIR = os.path.join(OUT_ROOT, "figures")

RAW_COLUMNS = [
    "t_s", "mass_flow", "density_kg_l", "T_after_subcooling_C",
    "suction_pressure_bar", "high_pressure_bar", "T_before_subcooling_C",
    "T_after_superheat_C", "T_high_pressure_line_C", "T_suction_line_C",
    "thermostat_outlet_C", "unknown1", "glycol_inlet_T_C", "glycol_outlet_T_C",
    "pressure_after_compression_bar", "mass_flow_avg_kg_s",
    "te1_T_C", "te2_T_C", "te3_T_C", "te4_T_C", "thermostat_inlet_C",
    "T_after_compressor_C", "irrelevant_y", "power_actual_W",
    "control_value_W", "irrelevant_x", "power_setpoint_W", "mean_temp",
    "mass_flow_PT1", "te5_T_evaporator_front_C", "room_temp_C",
]
# wall/evaporator-surface thermocouples used for CHF (dryout) detection
WALL_TE_COLS = ["te1_T_C", "te2_T_C", "te3_T_C", "te4_T_C", "te5_T_evaporator_front_C"]


def parse_metadata(path):
    folder = os.path.basename(os.path.dirname(path))
    fname = os.path.basename(path)
    run_num = int(re.search(r"Test Run (\d+)", folder).group(1))

    material = "aluminium" if re.search(r"alu", fname, re.I) else (
        "copper" if re.search(r"kupfer", fname, re.I) else "unknown")

    variant_match = re.search(r"St(\d+)mm", fname, re.I)
    pitch_mm = float(variant_match.group(1)) if variant_match else np.nan

    insert = "none"
    for tag, label in [
        ("swirlomat", "swirl_insert"),
        ("plastikoptimiert", "plastic_insert_optimized"),
        ("plastiknormal", "plastic_insert_normal"),
        ("leerlauf", "no_insert_baseline"),
        ("lang", "long_tube"),
        ("kurz", "short_tube"),
    ]:
        if tag in fname.lower():
            insert = label
            break

    version_match = re.search(r"V(\d+(?:\.\d+)?)", fname)
    version = version_match.group(1) if version_match else ""

    dates = re.findall(r"(\d{2}\.\d{2}\.\d{4})", fname)
    test_date = dates[0] if dates else ""

    return {
        "run_id": f"Test Run {run_num}",
        "run_num": run_num,
        "source_file": fname,
        "material": material,
        "channel_pitch_mm": pitch_mm,
        "insert_type": insert,
        "version_tag": version,
        "test_date": test_date,
    }


def detect_delimiter(path):
    with open(path, "rb") as f:
        first = f.readline().decode("cp1252", errors="replace")
    return ";" if first.count(";") > first.count(",") // 2 else ","


def load_raw(path):
    delim = detect_delimiter(path)
    df = pd.read_csv(
        path, sep=delim, decimal=",", encoding="cp1252",
        header=0, engine="python", on_bad_lines="skip",
    )
    df = df.iloc[:, :len(RAW_COLUMNS)]
    df.columns = RAW_COLUMNS[: df.shape[1]]
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=["t_s"]).reset_index(drop=True)
    return df


def detect_chf_point(df, power_col="power_actual_W", rate_thresh_sigma=4.0, min_run=3):
    """
    Heuristic burnout detector: flag the first sample where a wall
    thermocouple's rate of change exceeds `rate_thresh_sigma` standard
    deviations above its own baseline noise, sustained for `min_run`
    consecutive samples, while heater power is not decreasing.
    Returns (index, confidence) where confidence in {"high", "low", "none"}.
    """
    available_te = [c for c in WALL_TE_COLS if c in df.columns and df[c].notna().sum() > 20]
    if not available_te or power_col not in df.columns:
        return None, "none"

    power = df[power_col].rolling(5, min_periods=1, center=True).median()
    candidates = []
    for col in available_te:
        temp = df[col].rolling(3, min_periods=1, center=True).median()
        d = temp.diff()
        baseline = d.iloc[: max(20, len(d) // 10)].std()
        if not baseline or np.isnan(baseline) or baseline == 0:
            continue
        spike = d > rate_thresh_sigma * baseline
        run = 0
        for i in range(1, len(spike)):
            if spike.iloc[i]:
                run += 1
            else:
                run = 0
            if run >= min_run and power.iloc[i] >= power.iloc[max(0, i - 10)] - 1e-6:
                candidates.append(i - min_run + 1)
                break

    if not candidates:
        return None, "none"
    idx = min(candidates)
    confidence = "high" if len(candidates) >= 2 else "low"
    return idx, confidence


def make_plot(df, meta, chf_idx, confidence, out_path):
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(9, 6))
    ax1.plot(df["t_s"], df["power_actual_W"], color="tab:red", label="power_actual_W")
    ax1.set_ylabel("Power (W)")
    ax1.set_title(f"{meta['run_id']} | {meta['material']} | pitch={meta['channel_pitch_mm']}mm "
                   f"| {meta['insert_type']} | CHF confidence={confidence}")
    ax1.legend(loc="upper left", fontsize=8)

    for col in WALL_TE_COLS:
        if col in df.columns:
            ax2.plot(df["t_s"], df[col], label=col, linewidth=0.8)
    ax2.set_ylabel("Wall thermocouples (C)")
    ax2.set_xlabel("time (s)")
    ax2.legend(loc="upper left", fontsize=7, ncol=2)

    if chf_idx is not None:
        t_chf = df["t_s"].iloc[chf_idx]
        ax1.axvline(t_chf, color="k", linestyle="--", linewidth=1)
        ax2.axvline(t_chf, color="k", linestyle="--", linewidth=1)

    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


def main():
    os.makedirs(TS_DIR, exist_ok=True)
    os.makedirs(FIG_DIR, exist_ok=True)

    files = sorted(glob.glob(os.path.join(RAW_ROOT, "*", "*.csv")))
    summary_rows = []

    for path in files:
        meta = parse_metadata(path)
        try:
            df = load_raw(path)
        except Exception as e:
            print(f"FAILED to parse {path}: {e}")
            continue

        slug = re.sub(r"[^A-Za-z0-9]+", "_", meta["source_file"]).strip("_")[:60]
        ts_out = os.path.join(TS_DIR, f"{meta['run_id'].replace(' ', '_')}_{slug}.csv")
        df.to_csv(ts_out, index=False)

        chf_idx, confidence = detect_chf_point(df)

        fig_out = os.path.join(FIG_DIR, f"{meta['run_id'].replace(' ', '_')}.png")
        make_plot(df, meta, chf_idx, confidence, fig_out)

        row = dict(meta)
        row["n_samples"] = len(df)
        row["duration_s"] = float(df["t_s"].iloc[-1] - df["t_s"].iloc[0]) if len(df) else np.nan
        row["chf_detection_confidence"] = confidence

        if chf_idx is not None:
            point = df.iloc[chf_idx]
            row["t_chf_s"] = point["t_s"]
            row["mass_flow_avg_kg_s_at_chf"] = point.get("mass_flow_avg_kg_s")
            row["suction_pressure_bar_at_chf"] = point.get("suction_pressure_bar")
            row["high_pressure_bar_at_chf"] = point.get("high_pressure_bar")
            row["subcooling_T_C_at_chf"] = point.get("T_after_subcooling_C")
            row["power_actual_W_at_chf"] = point.get("power_actual_W")
            row["max_wall_te_C_at_chf"] = point[[c for c in WALL_TE_COLS if c in df.columns]].max()
        else:
            for k in ["t_chf_s", "mass_flow_avg_kg_s_at_chf", "suction_pressure_bar_at_chf",
                      "high_pressure_bar_at_chf", "subcooling_T_C_at_chf",
                      "power_actual_W_at_chf", "max_wall_te_C_at_chf"]:
                row[k] = np.nan

        summary_rows.append(row)
        print(f"{meta['run_id']:12s} rows={len(df):6d}  chf_idx={chf_idx}  confidence={confidence}")

    summary = pd.DataFrame(summary_rows).sort_values("run_num")
    summary_path = os.path.join(OUT_ROOT, "minichannel_chf_summary.csv")
    summary.to_csv(summary_path, index=False)
    print(f"\nWrote summary: {summary_path}  ({len(summary)} runs)")
    print(summary["chf_detection_confidence"].value_counts())


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
export_blockometry_charts.py
============================
Reproduit les deux figures de blockométrie VARENNE en Excel :
  Sheet 1 — "Shape Distribution"  : bar chart groupé (6 formes × 5 bins de volume)
  Sheet 2 — "Shape Diagram"       : scatter α vs β, coloré par bin de volume

Entrée  : outputs/VARENNE/06_blockometry_plots/block_shape_data_VARENNE.csv
Sortie  : outputs/VARENNE/06_blockometry_plots/Blockometry_charts.xlsx
"""

import os
import math
import numpy as np
import pandas as pd
import xlsxwriter

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_CSV = os.path.join(SCRIPT_DIR, "outputs", "VARENNE",
                        "06_blockometry_plots", "block_shape_data_VARENNE.csv")
OUT_FILE = os.path.join(SCRIPT_DIR, "outputs", "VARENNE",
                        "06_blockometry_plots", "Blockometry_charts.xlsx")
VTK_BLOCKS = os.path.join(SCRIPT_DIR, "outputs", "VARENNE",
                           "04_blocks_vtk", "VIZ_calibrated_VARENNE_Blocks_clean.vtk")


def _build_csv_from_vtk():
    """Generate DATA_CSV from the block VTK file when the CSV is missing."""
    import sys
    sys.path.insert(0, SCRIPT_DIR)
    from utils.vtk_io import read_legacy_vtk_unstructured_grid
    _points, _cells, _cell_types, cell_data, _point_data = read_legacy_vtk_unstructured_grid(VTK_BLOCKS)
    vols  = cell_data["volume"]
    alpha = cell_data["alpha"]
    beta  = cell_data["beta"]
    os.makedirs(os.path.dirname(DATA_CSV), exist_ok=True)
    pd.DataFrame({"volume": vols, "alpha": alpha, "beta": beta}).to_csv(DATA_CSV, index=False)
    print(f"✅ Generated {DATA_CSV} from VTK ({len(vols)} blocks)")


# ── Shape zone rules (same as plotTools.BlockShapeDistribution) ──────────
def classify_shape(alpha, beta):
    if beta >= 7:                           return "E"
    if alpha < 2 and beta < 4:             return "C"
    if alpha < 3 and 4 <= beta < 7:        return "CE"
    if alpha >= 3 and 4 <= beta < 7:       return "EP"
    if alpha >= 5 and beta < 4:            return "P"
    if 2 <= alpha < 5 and beta < 4:        return "PC"
    return "C"   # fallback

SHAPES       = ["C", "CE", "E", "EP", "P", "PC"]
SHAPE_LABELS = {
    "C":  "C – Cubic",
    "CE": "CE – Transitional Shape",
    "E":  "E – Elongated",
    "EP": "EP – Transitional Shape",
    "P":  "P – Platy",
    "PC": "PC – Transitional Shape",
}
SHAPE_COLORS = {
    "C":  "#1f77b4",   # blue
    "CE": "#ff7f0e",   # orange
    "E":  "#2ca02c",   # green
    "EP": "#d62728",   # red
    "P":  "#9467bd",   # purple
    "PC": "#8c564b",   # brown
}

BIN_LABELS  = ["< D20", "D20 to D40", "D40 to D60", "D60 to D80", "> D80"]
BIN_COLORS  = {
    "< D20":       "#000000",   # black (tiny dots)
    "D20 to D40":  "#17becf",   # cyan
    "D40 to D60":  "#bcbd22",   # yellow-green
    "D60 to D80":  "#e377c2",   # magenta
    "> D80":       "#d62728",   # red
}

V_MIN = 1e-15


def volume_weighted_percentiles(vols):
    """Return D20/D40/D60/D80 using volume-weighted CDF (same as plotTools)."""
    v = np.sort(vols)
    cum = np.cumsum(v)
    total = cum[-1]
    pct = 100.0 * cum / total
    D20 = v[np.searchsorted(pct, 20, side="right") - 1] if np.any(pct <= 20) else v[0]
    D40 = v[np.searchsorted(pct, 40, side="right") - 1] if np.any(pct <= 40) else v[0]
    D60 = v[np.searchsorted(pct, 60, side="right") - 1] if np.any(pct <= 60) else v[0]
    D80 = v[np.searchsorted(pct, 80, side="right") - 1] if np.any(pct <= 80) else v[0]
    return D20, D40, D60, D80


def assign_bin(vol, D20, D40, D60, D80):
    if vol < D20:                   return "< D20"
    if D20 <= vol < D40:            return "D20 to D40"
    if D40 <= vol < D60:            return "D40 to D60"
    if D60 <= vol < D80:            return "D60 to D80"
    return "> D80"


def alpha_to_plot(alpha, beta):
    """Same transform as plotTools.blockShapeDiagram."""
    a = math.log10(max(alpha, 1e-9)) * 9 + 1
    a = (1 - (1.0 / 9.0) * (beta - 1)) * a + (5.5 / 9.0) * (beta - 1)
    return a


def main():
    if not os.path.exists(DATA_CSV):
        _build_csv_from_vtk()
    df = pd.read_csv(DATA_CSV)
    df = df[np.isfinite(df["volume"]) & np.isfinite(df["alpha"]) & np.isfinite(df["beta"])
            & (df["volume"] > V_MIN)].copy()
    print(f"Loaded {len(df)} blocks")

    vols = df["volume"].to_numpy(float)
    D20, D40, D60, D80 = volume_weighted_percentiles(vols)
    print(f"D20={D20:.4f}  D40={D40:.4f}  D60={D60:.4f}  D80={D80:.4f}")

    total_volume = float(vols.sum())

    df["bin"]   = df["volume"].apply(lambda v: assign_bin(v, D20, D40, D60, D80))
    df["shape"] = df.apply(lambda r: classify_shape(r["alpha"], r["beta"]), axis=1)

    # ── Bar chart data: % of total volume per (bin, shape) ───────────────
    bar_data = {}
    for b in BIN_LABELS:
        row = {}
        sub_b = df[df["bin"] == b]
        for s in SHAPES:
            vol_sum = float(sub_b[sub_b["shape"] == s]["volume"].sum())
            row[s] = 100.0 * vol_sum / total_volume
        bar_data[b] = row

    # ── Scatter data: alpha_plot vs beta per bin ──────────────────────────
    # Apply the log transform to alpha
    df["alpha_plot"] = df.apply(
        lambda r: alpha_to_plot(r["alpha"], r["beta"]), axis=1)

    # ═══════════════════════════════════════════════════════════
    wb   = xlsxwriter.Workbook(OUT_FILE)
    bold = wb.add_format({"bold": True})

    # ────────────────────────────────────────────────────────────
    # SHEET 1 — Shape Distribution bar chart
    # ────────────────────────────────────────────────────────────
    ws1 = wb.add_worksheet("bar_data")

    # Write bar chart data table
    ws1.write(0, 0, "Bin", bold)
    for j, s in enumerate(SHAPES):
        ws1.write(0, j + 1, SHAPE_LABELS[s], bold)

    for i, b in enumerate(BIN_LABELS):
        ws1.write(i + 1, 0, b)
        for j, s in enumerate(SHAPES):
            ws1.write(i + 1, j + 1, round(bar_data[b][s], 4))

    ws_bar = wb.add_worksheet("Shape Distribution")
    bar_chart = wb.add_chart({"type": "column"})
    bar_chart.set_title({"name": "Shape Distribution"})
    bar_chart.set_x_axis({
        "name": "Block Volume Bin",
        "major_gridlines": {"visible": False},
    })
    bar_chart.set_y_axis({
        "name": "% of Total Volume",
        "min":  0,
        "major_gridlines": {"visible": True,
                            "line": {"color": "#BFBFBF", "width": 0.5}},
    })
    bar_chart.set_legend({"position": "top_right"})
    bar_chart.set_size({"width": 760, "height": 530})
    bar_chart.set_chartarea({"border": {"none": True}})

    for j, s in enumerate(SHAPES):
        col = j + 1
        bar_chart.add_series({
            "name":       ["bar_data", 0, col],
            "categories": ["bar_data", 1, 0, 5, 0],
            "values":     ["bar_data", 1, col, 5, col],
            "fill":       {"color": SHAPE_COLORS[s]},
            "border":     {"color": SHAPE_COLORS[s]},
            "gap":        50,
        })

    ws_bar.insert_chart("A1", bar_chart)

    # ────────────────────────────────────────────────────────────
    # SHEET 2 — Shape Diagram (α vs β scatter)
    # ────────────────────────────────────────────────────────────
    # Apply the log transform to alpha
    df["alpha_plot"] = df.apply(
        lambda r: alpha_to_plot(r["alpha"], r["beta"]), axis=1)

    ws_diag = wb.add_worksheet("Shape Diagram")
    MAX_PER_BIN = 800
    rng = np.random.default_rng(42)

    # Subsample per bin to avoid overplotting

    # Write scatter data — one sheet per bin
    bin_counts = {}
    for b in BIN_LABELS:
        sub = df[df["bin"] == b][["alpha_plot", "beta"]].reset_index(drop=True)
        if len(sub) > MAX_PER_BIN:
            idx = rng.choice(len(sub), MAX_PER_BIN, replace=False)
            idx.sort()
            sub = sub.iloc[idx].reset_index(drop=True)
        bin_counts[b] = len(sub)
        sname = f"sc_{BIN_LABELS.index(b)}"
        ws_sc = wb.add_worksheet(sname)
        ws_sc.write(0, 0, "alpha_plot", bold)
        ws_sc.write(0, 1, "beta", bold)
        for i, row in sub.iterrows():
            ws_sc.write(i + 1, 0, float(row["alpha_plot"]))
            ws_sc.write(i + 1, 1, float(row["beta"]))

    sc_chart = wb.add_chart({"type": "scatter"})
    sc_chart.set_title({"name": "Block Shape Diagram (\u03b1 vs \u03b2)"})
    sc_chart.set_x_axis({
        "name":    "\u03b1 (transformed)",
        "min":     1, "max": 11,
        "crossing": "min",
        "major_gridlines": {"visible": True,
                            "line": {"color": "#BFBFBF", "width": 0.5}},
    })
    sc_chart.set_y_axis({
        "name":    "\u03b2",
        "min":     1, "max": 11,
        "crossing": "min",
        "major_gridlines": {"visible": True,
                            "line": {"color": "#BFBFBF", "width": 0.5}},
    })
    sc_chart.set_legend({"position": "top_right"})
    sc_chart.set_size({"width": 700, "height": 650})
    sc_chart.set_chartarea({"border": {"none": True}})

    # marker sizes: small for <D20 (background), larger for higher bins (foreground)
    marker_sizes = {"< D20": 3, "D20 to D40": 5, "D40 to D60": 7,
                    "D60 to D80": 9, "> D80": 12}

    for b in BIN_LABELS:
        n     = bin_counts[b]
        sname = f"sc_{BIN_LABELS.index(b)}"
        color = BIN_COLORS[b]
        sz    = marker_sizes[b]

        sc_chart.add_series({
            "name":       b,
            "categories": [sname, 1, 0, n, 0],
            "values":     [sname, 1, 1, n, 1],
            "marker": {
                "type":   "circle",
                "size":   sz,
                "fill":   {"color": color},
                "border": {"color": color},
            },
            "line": {"none": True},
        })

    ws_diag.insert_chart("A1", sc_chart)

    wb.close()
    print(f"\n✅ Saved: {OUT_FILE}")


if __name__ == "__main__":
    main()

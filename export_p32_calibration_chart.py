#!/usr/bin/env python3
"""
export_p32_calibration_chart.py
================================
Reproduit la figure "BC TOTAL — P32 calibration (all families)"
dans un classeur Excel avec graphique natif xlsxwriter.

Sortie : outputs/BCTOTAL/02_calibration/P32_calibration_chart.xlsx

Éléments reproduits :
  • Points (scatter) : résultats DFN par famille
  • Droite de régression linéaire : P21 = a·P32 + b
  • Ligne horizontale en tirets : P21_target par famille
  • Marqueur croix (×) : P32_calibrated × P21_target
"""

import os
import numpy as np
import pandas as pd
import xlsxwriter

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

SWEEP_CSV   = os.path.join(SCRIPT_DIR, "outputs", "BCTOTAL", "01_sweep",
                           "P32_to_P21_sweep_results_BCTOTAL.csv")
CALIB_CSV   = os.path.join(SCRIPT_DIR, "outputs", "BCTOTAL", "02_calibration",
                           "P32_calibrated_summary_BCTOTAL.csv")
OUT_FILE    = os.path.join(SCRIPT_DIR, "outputs", "BCTOTAL", "02_calibration",
                           "P32_calibration_chart.xlsx")

# Colours matching matplotlib tab10
FAM_COLORS = {
    "fam1": "#1f77b4",   # bleu
    "fam2": "#ff7f0e",   # orange
    "fam3": "#2ca02c",   # vert
    "fam4": "#d62728",   # rouge
}

N_FIT_PTS = 200   # points on each regression line


def main():
    # ── Load data ────────────────────────────────────────────
    sweep = pd.read_csv(SWEEP_CSV)
    calib = pd.read_csv(CALIB_CSV)

    families = sorted(sweep["fam_name"].unique().tolist())
    print(f"Families: {families}")

    wb = xlsxwriter.Workbook(OUT_FILE)
    bold = wb.add_format({"bold": True})

    # ══════════════════════════════════════════════════════════
    # Build one data sheet per family, then one combined chart
    # sheet.  Excel scatter charts can reference multiple data
    # sheets, so we keep the data for each family on its own
    # sheet named "data_famX".
    # ══════════════════════════════════════════════════════════

    # Global y range for target lines (P32 on Y axis now)
    y_all = sweep["P32_obt"].to_numpy(float)
    y_global_min = float(np.min(y_all)) * 0.85
    y_global_max = float(np.max(y_all)) * 1.05

    chart_series_meta = []   # filled per family, used to build the chart

    for fam_name in families:
        sub   = sweep[sweep["fam_name"] == fam_name].sort_values("P32_obt")
        crow  = calib[calib["fam_name"] == fam_name].iloc[0]

        x_pts = sub["P21_obt"].to_numpy(float)   # P21 on X
        y_pts = sub["P32_obt"].to_numpy(float)   # P32 on Y

        a        = float(crow["fit_a"])
        b        = float(crow["fit_b"])
        r2       = float(crow["fit_R2"])
        p32_cal  = float(crow["P32_calibrated"])
        p21_tgt  = float(crow["P21_target"])
        flag     = str(crow["flag"])
        color    = FAM_COLORS.get(fam_name, "#808080")

        # Regression was P21=a*P32+b; with P21 on X: P32=(P21-b)/a
        x_fit = np.linspace(float(np.min(x_pts)), float(np.max(x_pts)), N_FIT_PTS)
        y_fit = (x_fit - b) / a

        # Vertical target line at P21_target (2 points, spans full Y range)
        x_tgt = np.array([p21_tgt, p21_tgt])
        y_tgt = np.array([y_global_min, y_global_max])

        # Cross marker: single point at (p21_tgt, p32_cal)
        x_cross = np.array([p21_tgt])
        y_cross = np.array([p32_cal])

        # Horizontal target line at p32_cal (spans full x range)
        x_htgt = np.array([0.0, float(np.max(x_pts)) * 1.1])
        y_htgt = np.array([p32_cal, p32_cal])

        # ── Write data sheet ──────────────────────────────────
        sname = f"data_{fam_name}"
        ws    = wb.add_worksheet(sname)

        # Column headers
        headers = [
            "scatter_x", "scatter_y",
            "fit_x",     "fit_y",
            "tgt_x",     "tgt_y",
            "cross_x",   "cross_y",
            "htgt_x",    "htgt_y",
        ]
        for j, h in enumerate(headers):
            ws.write(0, j, h, bold)

        # Scatter points
        for i, (xv, yv) in enumerate(zip(x_pts, y_pts)):
            ws.write(i + 1, 0, float(xv))
            ws.write(i + 1, 1, float(yv))

        # Regression line
        for i, (xv, yv) in enumerate(zip(x_fit, y_fit)):
            ws.write(i + 1, 2, float(xv))
            ws.write(i + 1, 3, float(yv))

        # Vertical target line (2 points)
        for i, (xv, yv) in enumerate(zip(x_tgt, y_tgt)):
            ws.write(i + 1, 4, float(xv))
            ws.write(i + 1, 5, float(yv))

        # Cross marker (1 point)
        ws.write(1, 6, float(x_cross[0]))
        ws.write(1, 7, float(y_cross[0]))

        # Horizontal target line (2 points)
        ws.write(1, 8, float(x_htgt[0]))
        ws.write(1, 9, float(y_htgt[0]))
        ws.write(2, 8, float(x_htgt[1]))
        ws.write(2, 9, float(y_htgt[1]))

        n_scatter = len(x_pts)
        fam_short    = fam_name.replace("fam", "F")
        legend_label = f"{fam_short}  fit (R\u00b2={r2:.4f})"

        chart_series_meta.append({
            "fam_name":    fam_name,
            "sname":       sname,
            "color":       color,
            "n_scatter":   n_scatter,
            "legend_label": legend_label,
            "p32_cal":     p32_cal,
            "p21_tgt":     p21_tgt,
        })

        print(f"  {fam_name}: {n_scatter} scatter pts, "
              f"P32_cal={p32_cal:.5f}, R²={r2:.4f}")

    # ══════════════════════════════════════════════════════════
    # Chart sheet
    # ══════════════════════════════════════════════════════════
    ws_chart = wb.add_worksheet("P32 Calibration")
    chart = wb.add_chart({"type": "scatter"})
    chart.set_title({"name": "BC TOTAL \u2014 P32 calibration (all families)"})
    chart.set_x_axis({
        "name":           "P21 (1/m)",
        "num_format":     "0.00",
        "crossing":       "min",
        "major_gridlines": {"visible": True,
                            "line": {"color": "#BFBFBF", "width": 0.5}},
    })
    chart.set_y_axis({
        "name":           "P32 (1/m)",
        "num_format":     "0.00",
        "min":            0,
        "crossing":       "min",
        "major_gridlines": {"visible": True,
                            "line": {"color": "#BFBFBF", "width": 0.5}},
    })
    # Each family has 5 series: scatter(show), fit, vtgt, cross, htgt(hide)
    # Global indices: F1:0-4, F2:5-9, F3:10-14, F4:15-19
    chart.set_legend({
        "position":      "top_right",
        "delete_series": [1,2,3,4, 6,7,8,9, 11,12,13,14, 16,17,18,19],
    })
    chart.set_size({"width": 760, "height": 530})
    chart.set_chartarea({"border": {"none": True}})

    for meta in chart_series_meta:
        sname     = meta["sname"]
        color     = meta["color"]
        n_sc      = meta["n_scatter"]
        fam_label = meta["legend_label"]

        # 1 — Scatter points (subtype "straight" but no line, just markers)
        chart.add_series({
            "name":         fam_label,
            "categories":   [sname, 1, 0, n_sc, 0],  # scatter_x
            "values":       [sname, 1, 1, n_sc, 1],  # scatter_y
            "marker": {
                "type":   "circle",
                "size":   6,
                "fill":   {"color": color},
                "border": {"color": color},
            },
            "line": {"none": True},
        })

        # 2 — Regression line (solid, no markers)
        chart.add_series({
            "name":       f"_nolegend_{fam_label}_fit",
            "categories": [sname, 1, 2, N_FIT_PTS, 2],  # fit_x
            "values":     [sname, 1, 3, N_FIT_PTS, 3],  # fit_y
            "line":  {"color": color, "width": 2.0, "dash_type": "solid"},
            "marker": {"type": "none"},
        })

        # 3 — Horizontal target line (dashed)
        chart.add_series({
            "name":       f"_nolegend_{fam_label}_tgt",
            "categories": [sname, 1, 4, 2, 4],  # tgt_x (2 pts)
            "values":     [sname, 1, 5, 2, 5],  # tgt_y
            "line":  {"color": color, "width": 1.5, "dash_type": "dash"},
            "marker": {"type": "none"},
        })

        # 4 — Cross marker at (p21_tgt, p32_cal)
        chart.add_series({
            "name":       f"_nolegend_{fam_label}_cross",
            "categories": [sname, 1, 6, 1, 6],  # cross_x (1 pt)
            "values":     [sname, 1, 7, 1, 7],  # cross_y
            "marker": {
                "type":   "x",
                "size":   12,
                "fill":   {"none": True},
                "border": {"color": color, "width": 2.5},
            },
            "line": {"none": True},
        })

        # 5 — Horizontal target line at p32_cal (dashed)
        chart.add_series({
            "name":       f"_nolegend_{fam_label}_htgt",
            "categories": [sname, 1, 8, 2, 8],  # htgt_x (2 pts)
            "values":     [sname, 1, 9, 2, 9],  # htgt_y
            "line":  {"color": color, "width": 1.5, "dash_type": "dash"},
            "marker": {"type": "none"},
        })

    ws_chart.insert_chart("A1", chart)
    wb.close()
    print(f"\n✅ Saved: {OUT_FILE}")
    print("Pour Word : cliquez sur le graphique → Ctrl+C → Coller spécial → "
          "Graphique Microsoft Excel Object")


if __name__ == "__main__":
    main()

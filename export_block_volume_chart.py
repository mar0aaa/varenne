#!/usr/bin/env python3
"""
export_block_volume_chart.py
============================
Reproduit la figure "Block Volume Distributions Overlay — BC sites"
dans un classeur Excel avec graphique natif xlsxwriter.

Sortie : outputs/BCTOTAL/06_blockometry_plots/Block_Volume_Overlay.xlsx

Éléments reproduits :
  • CDF empirique compte-basée pour chaque site (ligne pleine)
  • BCTOTAL en tirets noirs
  • Fuseau P10–P90 (série de remplissage gris)
  • Axe X logarithmique [1e-4 … 1e3]
"""

import os
import numpy as np
import xlsxwriter
from scipy.interpolate import interp1d

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

SITES = [
    ("BC1LEFT",  "BC-1 Left",  "#1f77b4"),
    ("BC1RIGHT", "BC-1 Right", "#ff7f0e"),
    ("BC2",      "BC-2",       "#2ca02c"),
    ("BC3LEFT",  "BC-3 Left",  "#d62728"),
    ("BC3RIGHT", "BC-3 Right", "#9467bd"),
    ("BCTOTAL",  "BC-Total",   "#000000"),
]

OUT_FILE = os.path.join(SCRIPT_DIR, "outputs", "BCTOTAL",
                        "06_blockometry_plots", "Block_Volume_Overlay.xlsx")
os.makedirs(os.path.dirname(OUT_FILE), exist_ok=True)

# CDF grid — log10 uniform, 300 points from 1e-4 to 1e3
XMIN_LOG = -4.0   # 1e-4
XMAX_LOG =  3.0   # 1e3
N_PTS    = 300

Q_LOW  = 0.10
Q_HIGH = 0.90
MIN_POSITIVE = 1e-15


def load_volumes(site):
    f = os.path.join(SCRIPT_DIR, "outputs", site, "05_block_volumes",
                     f"VIZ_calibrated_{site}_BlockVolumes_clean.txt")
    v = np.loadtxt(f)
    v = v[np.isfinite(v) & (v > MIN_POSITIVE)]
    return v


def empirical_cdf(v, x_grid_log):
    """Count-based empirical CDF interpolated onto log10 x_grid."""
    x_sorted = np.sort(v)
    n = len(x_sorted)
    y = 100.0 * np.arange(1, n + 1) / n
    x_log = np.log10(x_sorted)
    # interpolate; clamp below data min to 0, above to 100
    y_grid = np.interp(x_grid_log, x_log, y, left=0.0, right=100.0)
    return y_grid


def main():
    x_grid_log = np.linspace(XMIN_LOG, XMAX_LOG, N_PTS)
    x_grid     = 10.0 ** x_grid_log

    # ── Compute CDFs ─────────────────────────────────────────
    cdfs = {}
    for site, label, _ in SITES:
        v = load_volumes(site)
        cdfs[site] = empirical_cdf(v, x_grid_log)
        n = len(v)
        print(f"  {site}: n={n}")

    # ── Fuseau P10–P90 (exclude BCTOTAL) ─────────────────────
    fus_sites = [s for s, _, _ in SITES if s != "BCTOTAL"]
    y_stack   = np.array([cdfs[s] for s in fus_sites])
    y_low     = np.quantile(y_stack, Q_LOW,  axis=0)
    y_high    = np.quantile(y_stack, Q_HIGH, axis=0)
    # small margin
    y_low  = np.maximum(y_low  - 2.0, 0.0)
    y_high = np.minimum(y_high + 2.0, 100.0)

    # ── Excel workbook ────────────────────────────────────────
    wb = xlsxwriter.Workbook(OUT_FILE)
    bold = wb.add_format({"bold": True})

    # ── Data sheet ───────────────────────────────────────────
    ws = wb.add_worksheet("data")

    # Headers
    headers = ["x_vol"]
    for site, label, _ in SITES:
        headers.append(f"y_{site}")
    headers += ["y_fus_low", "y_fus_high"]

    for j, h in enumerate(headers):
        ws.write(0, j, h, bold)

    # Values
    for i in range(N_PTS):
        ws.write(i + 1, 0, float(x_grid[i]))
        for j, (site, label, _) in enumerate(SITES):
            ws.write(i + 1, j + 1, float(cdfs[site][i]))
        ws.write(i + 1, len(SITES) + 1, float(y_low[i]))
        ws.write(i + 1, len(SITES) + 2, float(y_high[i]))

    # ── Chart sheet ──────────────────────────────────────────
    ws_chart = wb.add_worksheet("Block Volume Overlay")
    chart = wb.add_chart({"type": "scatter", "subtype": "straight"})

    chart.set_title({"name": "Block Volume Distributions Overlay \u2014 BC sites"})
    chart.set_x_axis({
        "name":        "Block Volume (m\u00b3)",
        "log_base":    10,
        "min":         0.0001,
        "max":         1000,
        "num_format":  "0.0E+0",
        "crossing":    "min",
        "major_gridlines": {"visible": True,
                            "line": {"color": "#BFBFBF", "width": 0.5}},
        "minor_gridlines": {"visible": True,
                            "line": {"color": "#E5E5E5", "width": 0.25}},
    })
    chart.set_y_axis({
        "name":        "Cumulative Probability [% of blocks \u2264 x]",
        "min":         0,
        "max":         100,
        "crossing":    "min",
        "major_gridlines": {"visible": True,
                            "line": {"color": "#BFBFBF", "width": 0.5}},
    })
    chart.set_size({"width": 820, "height": 560})
    chart.set_chartarea({"border": {"none": True}})

    # ── Individual site CDF lines (added first → appear first in legend) ────
    for j, (site, _, _) in enumerate(SITES):
        col_idx = j + 1
        is_bctotal = (site == "BCTOTAL")
        n_blocks = len(load_volumes(site))
        label = f"{next(l for s,l,_ in SITES if s==site)} (n={n_blocks:,})"

        chart.add_series({
            "name":       label,
            "categories": ["data", 1, 0, N_PTS, 0],
            "values":     ["data", 1, col_idx, N_PTS, col_idx],
            "line": {
                "color":     next(c for s,_,c in SITES if s==site),
                "width":     2.5 if is_bctotal else 2.0,
                "dash_type": "dash" if is_bctotal else "solid",
            },
            "marker": {"type": "none"},
        })

    # ── Fuseau P10 boundary (shown in legend as band label) ──
    chart.add_series({
        "name":       "Envelope (P10\u2013P90)",
        "categories": ["data", 1, 0, N_PTS, 0],
        "values":     ["data", 1, len(SITES) + 1, N_PTS, len(SITES) + 1],
        "line":  {"color": "#AAAAAA", "width": 1.25, "dash_type": "dash"},
        "marker": {"type": "none"},
    })
    # ── Fuseau P90 boundary (hidden from legend) ─────────────
    chart.add_series({
        "name":       "_fus_high",
        "categories": ["data", 1, 0, N_PTS, 0],
        "values":     ["data", 1, len(SITES) + 2, N_PTS, len(SITES) + 2],
        "line":  {"color": "#AAAAAA", "width": 1.25, "dash_type": "dash"},
        "marker": {"type": "none"},
    })

    # Hide only the P90 duplicate from legend (index 7)
    chart.set_legend({
        "position":      "right",
        "delete_series": [7],
    })

    ws_chart.insert_chart("A1", chart)
    wb.close()
    print(f"\n✅ Saved: {OUT_FILE}")


if __name__ == "__main__":
    main()

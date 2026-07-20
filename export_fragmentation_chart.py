"""
Export the DFN-predicted pre-blast vs WipFrag post-blast fragmentation comparison
as an editable Excel chart.
Output: outputs/BCTOTAL/08_fragmentation_comparison/fragmentation_comparison_chart.xlsx
"""

import os
import sys
import numpy as np
import xlsxwriter

# ── reuse existing readers from the project ──────────────────────────────────
sys.path.insert(0, os.path.dirname(__file__))
from plot_blockometry_vs_fragmentation import (
    read_blockometry_curve,
    read_wipfrag_curves,
    volumes_to_equivalent_diameter_mm,
)

BLOCK_VOLUMES_TXT = (
    "outputs/BCTOTAL/05_block_volumes/"
    "VIZ_calibrated_BCTOTAL_BlockVolumes_clean.txt"
)
WIPFRAG_XLSX = "assets/Fragmentation wipfrag results.xlsx"
OUT_FILE     = (
    "outputs/BCTOTAL/08_fragmentation_comparison/"
    "fragmentation_comparison_chart.xlsx"
)
N_POINTS = 300   # number of interpolation points for the DFN smooth curve

# ── Load data ─────────────────────────────────────────────────────────────────
vols_sorted, passing_blocko = read_blockometry_curve(BLOCK_VOLUMES_TXT)
x_blocko = volumes_to_equivalent_diameter_mm(vols_sorted)

# Downsample DFN curve to N_POINTS evenly spaced in log space
log_min = np.log10(x_blocko[0])
log_max = np.log10(x_blocko[-1])
x_grid  = np.logspace(log_min, log_max, N_POINTS)
y_grid  = np.interp(x_grid, x_blocko, passing_blocko)

wipfrag_curves = read_wipfrag_curves(WIPFRAG_XLSX)


def densify_wipfrag(size_mm, passing_pct, gap_lo=20.0, gap_hi=400.0, n_extra=20):
    """
    Insert spline-interpolated points inside [gap_lo, gap_hi] mm so the
    curve looks smooth in the fines region.  A cubic spline is fitted through
    all existing WipFrag points in log-x space, then evaluated at n_extra
    log-spaced diameters within the gap.
    """
    from scipy.interpolate import CubicSpline
    log_x = np.log10(size_mm)
    cs = CubicSpline(log_x, passing_pct, extrapolate=False)
    x_extra = np.logspace(np.log10(gap_lo), np.log10(gap_hi), n_extra + 2)[1:-1]
    # Keep only extras strictly within the measured range
    x_extra = x_extra[(x_extra > size_mm.min()) & (x_extra < size_mm.max())]
    if len(x_extra) == 0:
        return size_mm, passing_pct
    y_extra = cs(np.log10(x_extra))
    y_extra = np.clip(y_extra, 0, 100)
    x_new = np.concatenate([size_mm, x_extra])
    y_new = np.concatenate([passing_pct, y_extra])
    order = np.argsort(x_new)
    return x_new[order], y_new[order]


wipfrag_curves = [
    (name, *densify_wipfrag(sizes, passing))
    for name, sizes, passing in wipfrag_curves
]

os.makedirs(os.path.dirname(OUT_FILE), exist_ok=True)

# ── Build Excel workbook ──────────────────────────────────────────────────────
wb   = xlsxwriter.Workbook(OUT_FILE)
bold = wb.add_format({"bold": True, "bg_color": "#D9E1F2", "border": 1})
num  = wb.add_format({"num_format": "0.000", "border": 1})
txt  = wb.add_format({"border": 1})

# ── Sheet: DFN data ───────────────────────────────────────────────────────────
ws_dfn = wb.add_worksheet("DFN")
ws_dfn.write(0, 0, "Diameter (mm)",  bold)
ws_dfn.write(0, 1, "Passing (%)",    bold)
ws_dfn.set_column(0, 1, 16)
for i, (x, y) in enumerate(zip(x_grid, y_grid), start=1):
    ws_dfn.write(i, 0, float(x))
    ws_dfn.write(i, 1, float(y))

# ── Sheets: WipFrag zones ─────────────────────────────────────────────────────
wf_sheet_names = []
for zi, (zone_name, size_mm, passing_pct) in enumerate(wipfrag_curves):
    sname = f"WF_{zi}"
    wf_sheet_names.append(sname)
    ws_wf = wb.add_worksheet(sname)
    ws_wf.write(0, 0, f"{zone_name} – size (mm)", bold)
    ws_wf.write(0, 1, f"{zone_name} – passing (%)", bold)
    ws_wf.set_column(0, 1, 22)
    for i, (x, y) in enumerate(zip(size_mm, passing_pct), start=1):
        ws_wf.write(i, 0, float(x))
        ws_wf.write(i, 1, float(y))

# ── Chart ─────────────────────────────────────────────────────────────────────
ws_chart = wb.add_worksheet("Chart")
chart = wb.add_chart({"type": "scatter", "subtype": "straight"})

chart.set_title({"name": "DFN-predicted pre-blast vs WipFrag post-blast"})
chart.set_x_axis({
    "name":       "Particle size (equivalent diameter, mm)",
    "log_base":   10,
    "min":        1,
    "max":        10000,
    "major_gridlines": {"visible": True,
                        "line": {"color": "#D9D9D9", "width": 0.5}},
    "minor_gridlines": {"visible": True,
                        "line": {"color": "#EFEFEF", "width": 0.25}},
})
chart.set_y_axis({
    "name": "Percent passing (%)",
    "min":  0,
    "max":  100,
    "major_gridlines": {"visible": True,
                        "line": {"color": "#D9D9D9", "width": 0.5}},
})
chart.set_legend({"position": "top"})
chart.set_size({"width": 820, "height": 560})
chart.set_chartarea({"border": {"none": True}})
chart.set_plotarea({
    "border": {"color": "#BFBFBF", "width": 0.75},
})

# DFN series — smooth red line, no markers
chart.add_series({
    "name":       "DFN-predicted pre-blast (model volumes)",
    "categories": ["DFN", 1, 0, N_POINTS, 0],
    "values":     ["DFN", 1, 1, N_POINTS, 1],
    "line":       {"color": "#FF0000", "width": 2.25},
    "marker":     {"type": "none"},
})

# WipFrag series — one per zone, with circle markers
WIPFRAG_COLORS = ["#4472C4", "#ED7D31", "#70AD47", "#FFC000"]
for zi, (zone_name, size_mm, passing_pct) in enumerate(wipfrag_curves):
    n  = len(size_mm)
    sn = wf_sheet_names[zi]
    c  = WIPFRAG_COLORS[zi % len(WIPFRAG_COLORS)]
    chart.add_series({
        "name":       f"WipFrag post-blast – {zone_name}",
        "categories": [sn, 1, 0, n, 0],
        "values":     [sn, 1, 1, n, 1],
        "line":       {"color": c, "width": 1.75},
        "marker":     {"type": "circle", "size": 5,
                       "fill": {"color": c}, "border": {"color": c}},
    })

ws_chart.insert_chart("A1", chart)

wb.close()
print(f"\u2705 Saved: {OUT_FILE}")

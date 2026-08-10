"""
Export a grouped bar chart of absolute relative error (%) in trace length
between CC field measurements and DFN model, grouped by site (5 groups),
with one bar per family (F1–F4) per group.

Formula: |CC - DFN| / CC * 100

Output: outputs/VARENNE/02_calibration/CC_vs_DFN_error_chart.xlsx
"""

import xlsxwriter
import math

OUT_FILE = "outputs/VARENNE/02_calibration/CC_vs_DFN_error_chart.xlsx"

# ── Raw data (site, family, cc, dfn) ────────────────────────────────────────
ROWS = [
    ("BC1LEFT",  "fam1", 1.604, 1.620),
    ("BC1LEFT",  "fam3", 1.598, 1.597),
    ("BC1LEFT",  "fam4", 2.056, 2.056),
    ("BC1RIGHT", "fam1", 1.698, 1.689),
    ("BC1RIGHT", "fam2", 1.616, 1.661),
    ("BC1RIGHT", "fam3", 1.303, 1.326),
    ("BC1RIGHT", "fam4", 1.694, 1.739),
    ("BC2",      "fam2", 4.099, 4.498),
    ("BC2",      "fam3", 2.851, 2.981),
    ("BC2",      "fam4", 2.557, 2.662),
    ("BC3LEFT",  "fam1", 4.848, 5.127),
    ("BC3LEFT",  "fam3", 5.060, 5.289),
    ("BC3LEFT",  "fam4", 4.686, 4.876),
    ("BC3RIGHT", "fam2", 4.142, 4.275),
    ("BC3RIGHT", "fam3", 4.230, 4.359),
    ("BC3RIGHT", "fam4", 3.920, 4.201),
]

SITE_LABEL = {
    "BC1LEFT":  "BC-1 Left",
    "BC1RIGHT": "BC-1 Right",
    "BC2":      "BC-2",
    "BC3LEFT":  "BC-3 Left",
    "BC3RIGHT": "BC-3 Right",
}

SITES   = ["BC1LEFT", "BC1RIGHT", "BC2", "BC3LEFT", "BC3RIGHT"]
FAMILIES = ["fam1", "fam2", "fam3", "fam4"]
FAM_COLORS = {
    "fam1": "#4472C4",
    "fam2": "#ED7D31",
    "fam3": "#A9D18E",
    "fam4": "#FF0000",
}

# Build lookup: (site, fam) -> abs relative error %
err = {}
for site, fam, cc, dfn in ROWS:
    err[(site, fam)] = abs(cc - dfn) / cc * 100.0

# ── Build pivot: rows = sites, cols = families ───────────────────────────────
# Layout in worksheet:
#   col 0: site label
#   col 1: F1 error
#   col 2: F2 error
#   col 3: F3 error
#   col 4: F4 error

wb  = xlsxwriter.Workbook(OUT_FILE)
hdr = wb.add_format({"bold": True, "align": "center",
                     "bg_color": "#D9E1F2", "border": 1})
num = wb.add_format({"num_format": "0.0", "border": 1})
txt = wb.add_format({"border": 1})
na  = wb.add_format({"border": 1, "align": "center", "italic": True,
                     "font_color": "#AAAAAA"})

ws = wb.add_worksheet("Data")
ws.set_column(0, 0, 14)
ws.set_column(1, 4, 10)

# Header row
ws.write(0, 0, "Site", hdr)
for j, fam in enumerate(FAMILIES, start=1):
    ws.write(0, j, fam.replace("fam", "F"), hdr)

# Data rows
for i, site in enumerate(SITES, start=1):
    ws.write(i, 0, SITE_LABEL[site], txt)
    for j, fam in enumerate(FAMILIES, start=1):
        val = err.get((site, fam), None)
        if val is None:
            ws.write(i, j, "—", na)
        else:
            ws.write(i, j, round(val, 2), num)

n_sites = len(SITES)

# ── Chart ────────────────────────────────────────────────────────────────────
chart = wb.add_chart({"type": "column"})

chart.set_title({
    "name": "Figure X. Absolute relative error in mean trace length between field measurements and DFN model (all sites and families)."
})
chart.set_x_axis({
    "name":      "Site",
    "text_axis": True,
})
chart.set_y_axis({
    "name":       "Absolute relative error (%)",
    "min":        0,
    "num_format": "0",
    "major_gridlines": {"visible": True,
                        "line": {"color": "#D9D9D9", "width": 0.5}},
})
chart.set_legend({"position": "right"})
chart.set_size({"width": 700, "height": 480})
chart.set_chartarea({"border": {"none": True}})
chart.set_plotarea({"border": {"color": "#BFBFBF", "width": 0.75}})

for j, fam in enumerate(FAMILIES, start=1):
    fam_short = fam.replace("fam", "F")
    chart.add_series({
        "name":       fam_short,
        "categories": ["Data", 1, 0, n_sites, 0],
        "values":     ["Data", 1, j, n_sites, j],
        "fill":       {"color": FAM_COLORS[fam]},
        "gap":        80,
    })

ws.insert_chart("G1", chart)

wb.close()
print(f"✅ Saved: {OUT_FILE}")

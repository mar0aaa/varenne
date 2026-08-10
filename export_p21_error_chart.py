"""
Export a grouped bar chart of absolute relative error (%) in P21
between field measurements (terrain) and DFN model, grouped by site,
with one bar per family (F1–F4) per group.

Formula: |P21_terrain - P21_model| / P21_terrain * 100

Output: outputs/VARENNE/02_calibration/P21_error_chart.xlsx

NOTE: This file currently contains hardcoded BC site data (lines 17-48).
      Update with VARENNE-specific data if needed.
"""

import xlsxwriter

OUT_FILE = "outputs/VARENNE/02_calibration/P21_error_chart.xlsx"

# ── Raw data (BC SITE DATA - UPDATE FOR VARENNE) ─────────────────────────────
ROWS = [
    ("BC1LEFT",  "fam1", 0.02832, 0.02839),
    ("BC1LEFT",  "fam3", 0.08833, 0.07927),
    ("BC1LEFT",  "fam4", 0.05760, 0.05179),
    ("BC1RIGHT", "fam1", 0.02438, 0.02631),
    ("BC1RIGHT", "fam2", 0.06104, 0.06749),
    ("BC1RIGHT", "fam3", 0.02079, 0.02131),
    ("BC1RIGHT", "fam4", 0.03515, 0.03247),
    ("BC2",      "fam2", 0.14106, 0.18744),
    ("BC2",      "fam3", 0.10703, 0.12158),
    ("BC2",      "fam4", 0.04799, 0.05683),
    ("BC3LEFT",  "fam1", 0.15701, 0.18150),
    ("BC3LEFT",  "fam3", 0.08885, 0.08690),
    ("BC3LEFT",  "fam4", 0.03658, 0.05668),
    ("BC3RIGHT", "fam2", 0.16888, 0.18683),
    ("BC3RIGHT", "fam3", 0.13204, 0.11853),
    ("BC3RIGHT", "fam4", 0.02248, 0.01622),
    ("BCTOTAL",  "fam1", 0.05450, 0.05245),
    ("BCTOTAL",  "fam2", 0.05379, 0.05872),
    ("BCTOTAL",  "fam3", 0.08424, 0.10884),
    ("BCTOTAL",  "fam4", 0.04085, 0.04146),
]

SITE_LABEL = {
    "BC1LEFT":  "BC-1 Left",
    "BC1RIGHT": "BC-1 Right",
    "BC2":      "BC-2",
    "BC3LEFT":  "BC-3 Left",
    "BC3RIGHT": "BC-3 Right",
    "BCTOTAL":  "BC-Total",
}

SITES    = ["BC1LEFT", "BC1RIGHT", "BC2", "BC3LEFT", "BC3RIGHT", "BCTOTAL"]
FAMILIES = ["fam1", "fam2", "fam3", "fam4"]
FAM_COLORS = {
    "fam1": "#4472C4",
    "fam2": "#ED7D31",
    "fam3": "#A9D18E",
    "fam4": "#FF0000",
}

# Build lookup: (site, fam) -> abs relative error %
err = {}
for site, fam, terrain, model in ROWS:
    err[(site, fam)] = abs(terrain - model) / terrain * 100.0

# ── Workbook ──────────────────────────────────────────────────────────────────
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

# ── Chart ─────────────────────────────────────────────────────────────────────
chart = wb.add_chart({"type": "column"})

chart.set_title({
    "name": "Figure X. Absolute relative error between field-measured and simulated P\u2082\u2081 fracture intensities per site and fracture family."
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

"""
Export a grouped column chart (P21 terrain / P21 model / P32 cal) per site/family into Excel.
Output: outputs/VARENNE/02_calibration/P21_calibration_chart.xlsx

NOTE: This file currently contains hardcoded BC site data.
      Update with VARENNE-specific data if needed.
"""

import xlsxwriter

OUT_FILE = "outputs/VARENNE/02_calibration/P21_calibration_chart.xlsx"

# ── Data (BC SITE DATA - UPDATE FOR VARENNE) ─────────────────────────────────
ROWS = [
    ("BC1LEFT",  "fam1", 0.02832, 0.02839, 0.02513, 0.9959),
    ("BC1LEFT",  "fam3", 0.08833, 0.07927, 0.09100, 0.9940),
    ("BC1LEFT",  "fam4", 0.05760, 0.05179, 0.06507, 0.9952),
    ("BC1RIGHT", "fam1", 0.02438, 0.02631, 0.02756, 0.9975),
    ("BC1RIGHT", "fam2", 0.06104, 0.06749, 0.06662, 0.9987),
    ("BC1RIGHT", "fam3", 0.02079, 0.02131, 0.02915, 0.9848),
    ("BC1RIGHT", "fam4", 0.03515, 0.03247, 0.03376, 0.9443),
    ("BC2",      "fam2", 0.14106, 0.18744, 0.13959, 0.9998),
    ("BC2",      "fam3", 0.10703, 0.12158, 0.11673, 0.9987),
    ("BC2",      "fam4", 0.04799, 0.05683, 0.05390, 0.9938),
    ("BC3LEFT",  "fam1", 0.15701, 0.18150, 0.18951, 0.9996),
    ("BC3LEFT",  "fam3", 0.08885, 0.08690, 0.11179, 0.9919),
    ("BC3LEFT",  "fam4", 0.03658, 0.05668, 0.04538, 0.9407),
    ("BC3RIGHT", "fam2", 0.16888, 0.18683, 0.16800, 0.9990),
    ("BC3RIGHT", "fam3", 0.13204, 0.11853, 0.15004, 0.9968),
    ("BC3RIGHT", "fam4", 0.02248, 0.01622, 0.03973, 0.9633),
    ("BCTOTAL",  "fam1", 0.05450, 0.05245, 0.06302, 0.9983),
    ("BCTOTAL",  "fam2", 0.05379, 0.05872, 0.05239, 0.9843),
    ("BCTOTAL",  "fam3", 0.08424, 0.10884, 0.09717, 0.9779),
    ("BCTOTAL",  "fam4", 0.04085, 0.04146, 0.05126, 0.9754),
]

# ── Workbook ──────────────────────────────────────────────────────────────────
wb   = xlsxwriter.Workbook(OUT_FILE)
bold = wb.add_format({"bold": True})
hdr  = wb.add_format({"bold": True, "align": "center",
                       "bg_color": "#D9E1F2", "border": 1})
num4 = wb.add_format({"num_format": "0.00000", "border": 1})
num2 = wb.add_format({"num_format": "0.0000",  "border": 1})
txt  = wb.add_format({"border": 1})

ws = wb.add_worksheet("Data")

headers = ["Site", "Family", "P21 terrain", "P21 model", "P32 cal", "R²", "Label"]
col_widths = [12, 10, 13, 13, 13, 8, 20]
for c, (h, w) in enumerate(zip(headers, col_widths)):
    ws.write(0, c, h, hdr)
    ws.set_column(c, c, w)

n = len(ROWS)
SITE_LABEL = {
    "BC1LEFT":  "BC-1 Left",
    "BC1RIGHT": "BC-1 Right",
    "BC2":      "BC-2",
    "BC3LEFT":  "BC-3 Left",
    "BC3RIGHT": "BC-3 Right",
    "BCTOTAL":  "BC-Total",
}

for i, (site, fam, p21t, p21m, p32c, r2) in enumerate(ROWS, start=1):
    fam_short = fam.replace("fam", "F")
    label = f"{SITE_LABEL.get(site, site)} – {fam_short}"
    ws.write(i, 0, site,   txt)
    ws.write(i, 1, fam,    txt)
    ws.write(i, 2, p21t,   num4)
    ws.write(i, 3, p21m,   num4)
    ws.write(i, 4, p32c,   num4)
    ws.write(i, 5, r2,     num2)
    ws.write(i, 6, label,  txt)

# ── Chart ─────────────────────────────────────────────────────────────────────
chart = wb.add_chart({"type": "column"})

chart.set_title({"name": "Figure 16. Comparison of P\u2082\u2081 targets and simulated P\u2082\u2081 values (all sites and families)."})
chart.set_x_axis({
    "name":      "Site – Family",
    "text_axis": True,
})
chart.set_y_axis({
    "name":       "P21 (m⁻¹)",
    "min":        0,
    "num_format": "0.##",
    "major_gridlines": {"visible": True,
                        "line": {"color": "#D9D9D9", "width": 0.5}},
})
chart.set_legend({"position": "bottom"})
chart.set_size({"width": 820, "height": 500})
chart.set_chartarea({"border": {"none": True}})
chart.set_plotarea({
    "border": {"color": "#BFBFBF", "width": 0.75},
})

chart.add_series({
    "name":       "P21 terrain",
    "categories": ["Data", 1, 6, n, 6],
    "values":     ["Data", 1, 2, n, 2],
    "fill":       {"color": "#4472C4"},
    "gap":        60,
})

chart.add_series({
    "name":       "P21 model",
    "categories": ["Data", 1, 6, n, 6],
    "values":     ["Data", 1, 3, n, 3],
    "fill":       {"color": "#ED7D31"},
    "gap":        60,
})

ws.insert_chart("I1", chart)

wb.close()
print(f"\u2705 Saved: {OUT_FILE}")

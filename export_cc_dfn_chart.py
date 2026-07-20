"""
Export a grouped bar chart (Avg CC vs Avg DFN) per site/family into Excel.
Output: outputs/BCTOTAL/02_calibration/CC_vs_DFN_chart.xlsx
"""

import xlsxwriter

OUT_FILE = "outputs/BCTOTAL/02_calibration/CC_vs_DFN_chart.xlsx"

# ── Data ─────────────────────────────────────────────────────────────────────
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

# ── Workbook ──────────────────────────────────────────────────────────────────
wb   = xlsxwriter.Workbook(OUT_FILE)
bold = wb.add_format({"bold": True})
hdr  = wb.add_format({"bold": True, "align": "center",
                       "bg_color": "#D9E1F2", "border": 1})
num  = wb.add_format({"num_format": "0.000", "border": 1})
txt  = wb.add_format({"border": 1})

ws = wb.add_worksheet("Data")

# Headers
headers = ["Site", "Family", "Avg CC trace length (m)", "Avg DFN trace length (m)"]
for c, h in enumerate(headers):
    ws.write(0, c, h, hdr)

ws.set_column(0, 0, 12)
ws.set_column(1, 1, 10)
ws.set_column(2, 3, 14)

# Labels (x-axis categories): "BC1LEFT – fam1", etc.
labels = [f"{r[0]} – {r[1]}" for r in ROWS]
cc_vals  = [r[2] for r in ROWS]
dfn_vals = [r[3] for r in ROWS]

for i, (lbl, cc, dfn) in enumerate(zip(labels, cc_vals, dfn_vals), start=1):
    site, fam = lbl.split(" – ")
    ws.write(i, 0, site, txt)
    ws.write(i, 1, fam,  txt)
    ws.write(i, 2, cc,   num)
    ws.write(i, 3, dfn,  num)

n = len(ROWS)

# ── Chart ─────────────────────────────────────────────────────────────────────
chart = wb.add_chart({"type": "column"})

chart.set_title({"name": "Average Trace Length: CloudCompare (CC) vs DFN"})
chart.set_x_axis({
    "name":         "Site – Family",
    "text_axis":    True,
})
chart.set_y_axis({
    "name": "Average Trace Length (m)",
    "min":  0,
    "major_gridlines": {"visible": True,
                        "line": {"color": "#D9D9D9", "width": 0.5}},
})
chart.set_legend({"position": "bottom"})
chart.set_size({"width": 700, "height": 560})
chart.set_chartarea({"border": {"none": True}})
chart.set_plotarea({
    "border": {"color": "#BFBFBF", "width": 0.75},
})

# Series: category labels from col A+B joined — write a helper label column
ws.write(0, 4, "Label", hdr)
ws.set_column(4, 4, 20)
for i, lbl in enumerate(labels, start=1):
    ws.write(i, 4, lbl, txt)

chart.add_series({
    "name":       "Avg CC trace length (m)",
    "categories": ["Data", 1, 4, n, 4],
    "values":     ["Data", 1, 2, n, 2],
    "fill":       {"color": "#4472C4"},
    "gap":        60,
})

chart.add_series({
    "name":       "Avg DFN trace length (m)",
    "categories": ["Data", 1, 4, n, 4],
    "values":     ["Data", 1, 3, n, 3],
    "fill":       {"color": "#ED7D31"},
    "gap":        60,
})

ws.insert_chart("G1", chart)

wb.close()
print(f"\u2705 Saved: {OUT_FILE}")

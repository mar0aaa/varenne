"""
Export a summary table (Joint Set / Dip / Dip Direction / Avg Spacing / Avg Persistence)
for the BCTOTAL DFN into Excel.
Output: outputs/BCTOTAL/09_report/Joint_set_summary_BCTOTAL.xlsx
"""

import os
import xlsxwriter
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_FILE   = os.path.join(SCRIPT_DIR, "outputs", "BCTOTAL", "09_report",
                          "Joint_set_summary_BCTOTAL.xlsx")
os.makedirs(os.path.dirname(OUT_FILE), exist_ok=True)

# ── Load mean spacing and persistence from existing summaries ─────────────────
sp = pd.read_excel(os.path.join(SCRIPT_DIR, "outputs", "SPACING",
                                "spacing_fit_summary_LOGN.xlsx"))
pe = pd.read_excel(os.path.join(SCRIPT_DIR, "outputs", "PERSISTENCE",
                                "persistence_fit_summary_EXP_LOGN.xlsx"))

sp_bc = sp[sp["region"] == "BCTOTAL"].set_index("family")["mean_fit"]
pe_bc = pe[pe["region"] == "TOTAL"].set_index("family")["mean_fit"]

# ── Mean orientations (computed from all-site pooled measurements) ────────────
ORIENTATIONS = {
    "fam1": (40, 210),
    "fam2": (35,  45),
    "fam3": (76, 203),
    "fam4": (71, 132),
}

ROWS = [
    ("F1", "fam1"),
    ("F2", "fam2"),
    ("F3", "fam3"),
    ("F4", "fam4"),
]

# ── Workbook ──────────────────────────────────────────────────────────────────
wb = xlsxwriter.Workbook(OUT_FILE)

title_fmt = wb.add_format({
    "bold": True, "font_size": 13, "align": "center", "valign": "vcenter",
    "bg_color": "#1F3864", "font_color": "#FFFFFF", "border": 1,
})
hdr_fmt = wb.add_format({
    "bold": True, "align": "center", "valign": "vcenter",
    "bg_color": "#2F5496", "font_color": "#FFFFFF", "border": 1,
    "text_wrap": True,
})
fam_fmt = wb.add_format({
    "bold": True, "align": "center", "valign": "vcenter",
    "bg_color": "#D6E4F0", "border": 1,
})
num_fmt = wb.add_format({
    "num_format": "0.00", "align": "center", "valign": "vcenter",
    "border": 1,
})
int_fmt = wb.add_format({
    "num_format": "0", "align": "center", "valign": "vcenter",
    "border": 1,
})

ws = wb.add_worksheet("Summary")

ws.set_column(0, 0, 12)  # Joint Set
ws.set_column(1, 1, 12)  # Dip
ws.set_column(2, 2, 18)  # Dip Direction
ws.set_column(3, 3, 18)  # Avg Spacing
ws.set_column(4, 4, 20)  # Avg Persistence
ws.set_row(0, 22)
ws.set_row(1, 30)

# Title row
ws.merge_range(0, 0, 0, 4,
               "BCTOTAL — Fracture Family Characteristics (DFN)", title_fmt)

# Header row
headers = ["Joint Set", "Dip (°)", "Dip Direction (°)",
           "Avg. Spacing (m)", "Avg. Persistence (m)"]
for c, h in enumerate(headers):
    ws.write(1, c, h, hdr_fmt)

# Data rows
for row_idx, (label, fam) in enumerate(ROWS, start=2):
    dip, dipdir = ORIENTATIONS[fam]
    spacing     = float(sp_bc.get(fam, float("nan")))
    persistence = float(pe_bc.get(fam, float("nan")))

    # alternating row background
    bg = "#EBF3FB" if row_idx % 2 == 0 else "#FFFFFF"
    alt_num = wb.add_format({"num_format": "0.00", "align": "center",
                             "valign": "vcenter", "border": 1, "bg_color": bg})
    alt_int = wb.add_format({"num_format": "0",    "align": "center",
                             "valign": "vcenter", "border": 1, "bg_color": bg})
    alt_fam = wb.add_format({"bold": True, "align": "center",
                             "valign": "vcenter", "border": 1,
                             "bg_color": "#D6E4F0"})

    ws.write(row_idx, 0, label,       alt_fam)
    ws.write(row_idx, 1, dip,         alt_int)
    ws.write(row_idx, 2, dipdir,      alt_int)
    ws.write(row_idx, 3, spacing,     alt_num)
    ws.write(row_idx, 4, persistence, alt_num)

wb.close()
print(f"\u2705 Saved: {OUT_FILE}")

"""
report_whiteboard.py
====================
Professional scientific paper–style report generator.
Format: GeoQuébec / journal two-column layout using ReportLab Platypus.

Outputs  →  outputs/VARENNE/09_report/scientific_report.pdf
"""

import os
import sys
from typing import List, Optional

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")

from run_site import SITE_CONFIGS


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SITES = ["VARENNE"]

# RGB tuples (0–1) for site row colouring
_SITE_RGB = {
    "VARENNE":  (0.871, 0.918, 0.945),
}


def _first_existing(paths):
    for p in paths:
        if os.path.exists(p):
            return p
    return None


def _fam_sort(name):
    txt = str(name).lower().replace("fam", "")
    try:
        return (0, int(txt))
    except ValueError:
        return (1, name)


# ── data builders ─────────────────────────────────────────────────────────────

def _build_blockometry_df():
    rows = []
    for site in SITES:
        path = os.path.join(SCRIPT_DIR, "outputs", site, "05_block_volumes",
                            f"VIZ_calibrated_{site}_BlockVolumes_clean.txt")
        if not os.path.exists(path):
            continue
        v = np.loadtxt(path, ndmin=1).ravel()
        v = v[np.isfinite(v) & (v > 0)]
        if not v.size:
            continue
        d50 = float(np.percentile(v, 50))
        rows.append({
            "Site":       site,
            "n blocks":   int(v.size),
            "D20 (m³)":  f"{np.percentile(v,20):.4f}",
            "D50 (m³)":  f"{d50:.4f}",
            "D80 (m³)":  f"{np.percentile(v,80):.4f}",
            "D90 (m³)":  f"{np.percentile(v,90):.4f}",
            "Mean (m³)": f"{np.mean(v):.4f}",
            "L_D50 (m)": f"{d50**(1/3):.4f}",
        })
    return pd.DataFrame(rows)
def _build_trace_df():
    assets = os.path.join(SCRIPT_DIR, "assets")
    site_info = [
        ("VARENNE",  SITE_CONFIGS["VARENNE"]["trace_name"],  False, SITE_CONFIGS["VARENNE"]["region_filter"]),
    ]
    cc = {}
    for site, fname, is_xl, rfilt in site_info:
        p = os.path.join(assets, fname)
        if not os.path.exists(p):
            continue
        df = pd.read_excel(p) if is_xl else pd.read_csv(p)
        if rfilt and "REGION" in df.columns:
            df = df[df["REGION"] == rfilt]
        fc = next((c for c in df.columns if c.lower() in ("fam", "family")), None)
        lc = next((c for c in df.columns if c.lower() in ("length", "corrected length")), None)
        if not fc or not lc:
            continue
        df[lc] = pd.to_numeric(df[lc], errors="coerce")
        df = df.dropna(subset=[fc, lc])
        df = df[df[lc] > 0]
        for fam, grp in df.groupby(fc):
            cc[(site, str(fam).strip())] = grp[lc].astype(float).tolist()
    dfn = {}
    for site in SITES:
        pd_ = os.path.join(SCRIPT_DIR, "outputs", site, "07_persistence")
        if not os.path.isdir(pd_):
            continue
        for fn in sorted(os.listdir(pd_)):
            if not (fn.startswith("persistence_radii_") and fn.endswith(".csv")):
                continue
            fam = fn.replace("persistence_radii_", "").replace(".csv", "")
            d = pd.read_csv(os.path.join(pd_, fn))
            vals = pd.to_numeric(d.get("diameter_m"), errors="coerce").dropna()
            dfn[(site, fam)] = [float(x) for x in vals if float(x) > 0]
    keys = sorted(set(cc) | set(dfn), key=lambda x: (x[0], _fam_sort(x[1])))
    rows = []
    for site, fam in keys:
        c = cc.get((site, fam), [])
        d = dfn.get((site, fam), [])
        rows.append({
            "Site":         site,
            "Family":       fam,
            "CC mean (m)":  f"{np.mean(c):.3f}" if c else "—",
            "DFN mean (m)": f"{np.mean(d):.3f}" if d else "—",
        })
    return pd.DataFrame(rows)


def _build_calib_df():
    csv_map = {
        "VARENNE":  "outputs/VARENNE/02_calibration/P32_calibrated_summary.csv",
    }
    rows = []
    for site, rel in csv_map.items():
        full = os.path.join(SCRIPT_DIR, rel)
        if not os.path.exists(full):
            continue
        df = pd.read_csv(full)
        for _, r in df.iterrows():
            fam = r.get("fam_name")
            if pd.isna(fam):
                fam = f"fam{int(r['fam'])}"
            p21m = r.get("P21_model", np.nan)
            rows.append({
                "Site":        site,
                "Family":      str(fam),
                "P21 terrain": f"{float(r.get('P21_target', np.nan)):.5f}",
                "P21 model":   f"{float(p21m):.5f}" if pd.notna(p21m) else "—",
                "P32 cal":     f"{float(r.get('P32_calibrated', np.nan)):.5f}",
            })
    return pd.DataFrame(rows)


def _find_persistence_survival_plot() -> Optional[str]:
    """
    Locate the validated combined persistence survival plot produced by
    PERSISTENCE.py (real field trace-length survival, all families).
    Does NOT regenerate it — PERSISTENCE.py is the single source of truth.
    """
    persist_out = os.path.join(SCRIPT_DIR, "outputs", "PERSISTENCE")
    return _first_existing([
        os.path.join(persist_out, "persistence_survival_VARENNE_all_families.png"),
        os.path.join(persist_out, "persistence_survival_BC_TOTAL_all_families.png"),
    ])


def _find_spacing_survival_plot() -> Optional[str]:
    """
    Locate the validated combined spacing survival plot produced by
    SPACING.py (real perpendicular-spacing survival, all families).
    Does NOT regenerate it — SPACING.py is the single source of truth.
    """
    spacing_out = os.path.join(SCRIPT_DIR, "outputs", "SPACING")
    return _first_existing([
        os.path.join(spacing_out, "spacing_survival_VARENNE_all_families.png"),
        os.path.join(spacing_out, "spacing_survival_BC_TOTAL_all_families.png"),
    ])


def _generate_blockometry_curve(out_path: str) -> bool:
    """
    Generate logarithmic block volume distribution curve for VARENNE site.
    
    Creates a cumulative probability plot on a logarithmic x-axis.
    
    Args:
        out_path: Path to save the figure (PNG format)
    
    Returns:
        True if successful, False otherwise
    """
    import matplotlib.pyplot as plt
    
    # Load block volumes
    vol_path = os.path.join(SCRIPT_DIR, "outputs", "VARENNE", "05_block_volumes",
                           "VIZ_calibrated_VARENNE_BlockVolumes_clean.txt")
    
    if not os.path.exists(vol_path):
        print(f"⚠️  Block volumes not found: {vol_path}")
        return False
    
    vols = np.loadtxt(vol_path, ndmin=1).ravel()
    vols = vols[np.isfinite(vols) & (vols > 0)]
    
    if vols.size == 0:
        print(f"⚠️  No valid block volumes found")
        return False
    
    # Sort volumes and compute cumulative probability (volume-weighted)
    vols_sorted = np.sort(vols)
    cumulative_volumes = np.cumsum(vols_sorted)
    cumulative_probability = 100.0 * cumulative_volumes / cumulative_volumes[-1]
    
    # Create figure — aspect ratio matched to the report's figure box
    # (COL_W x FIG_H = COL_W x 0.60*COL_W) so it renders at full column width,
    # same as the persistence/spacing figures.
    fig, ax = plt.subplots(figsize=(8, 4.8))
    
    # Plot VARENNE curve
    ax.plot(vols_sorted, cumulative_probability, 
           color='#FF6B35', linewidth=2.5, 
           label=f'VARENNE (n={len(vols)})')
    
    ax.set_xscale('log')
    ax.set_xlim(1e-4, 1e3)
    ax.set_ylim(0, 100)
    ax.set_xlabel('Block Volume (m³)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Cumulative Probability (%)', fontsize=11, fontweight='bold')
    ax.set_title('Block Volume Distribution — VARENNE', fontsize=12, fontweight='bold')
    ax.grid(True, which='both', linestyle='--', alpha=0.4)
    ax.legend(loc='lower right', fontsize=10)
    
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return True


# ── main report generator ─────────────────────────────────────────────────────

def generate_scientific_report(project_name: str = "VARENNE") -> str:
    """Generate a clean results-only scientific paper PDF (GeoQuébec style)."""

    from reportlab.lib.pagesizes import A4
    from reportlab.lib.units import cm
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_CENTER, TA_LEFT
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.platypus import (
        BaseDocTemplate, Frame, PageTemplate,
        Paragraph, Spacer, Table, TableStyle,
        Image, PageBreak, HRFlowable, KeepTogether,
        NextPageTemplate,
    )

    report_dir = os.path.join(SCRIPT_DIR, "outputs", "VARENNE", "09_report")
    os.makedirs(report_dir, exist_ok=True)
    out_pdf = os.path.join(report_dir, "scientific_report.pdf")

    # ── geometry ──────────────────────────────────────────────────────────────
    PW, PH = A4                       # 595.28 × 841.89 pt
    ML = MR = 1.0 * cm
    MT = MB = 1.0 * cm
    BODY_W = PW - ML - MR
    GAP = 0.35 * cm
    COL_W = (BODY_W - GAP) / 2

    # ── palette ───────────────────────────────────────────────────────────────
    C_TITLE   = colors.HexColor("#1A3A5C")
    C_RULE    = colors.HexColor("#2E75B6")
    C_HEADBG  = colors.HexColor("#1F4E79")
    C_HEADTXT = colors.white
    C_ALT     = colors.HexColor("#EEF3F8")
    C_BORDER  = colors.HexColor("#BBBBBB")
    C_SEC     = colors.HexColor("#2E75B6")

    SITE_CLR = {k: colors.Color(*v, 1.0) for k, v in _SITE_RGB.items()}

    # ── styles ────────────────────────────────────────────────────────────────
    def S(name, **kw):
        return ParagraphStyle(name, **kw)

    sTitle   = S("Title",   fontName="Helvetica-Bold",    fontSize=12, leading=13,
                             alignment=TA_CENTER, textColor=C_TITLE, spaceAfter=2)
    sSec     = S("Sec",     fontName="Helvetica-Bold",    fontSize=10, leading=11,
                             alignment=TA_LEFT,  textColor=C_SEC,   spaceBefore=4, spaceAfter=2)
    sSubsec  = S("Subsec",  fontName="Helvetica-Bold",    fontSize=8.5, leading=11,
                             alignment=TA_LEFT,  textColor=C_TITLE, spaceBefore=6,  spaceAfter=2)
    _cap_kw  = dict(fontName="Helvetica-Oblique", fontSize=7.5, leading=8.5,
                    alignment=TA_CENTER, textColor=colors.HexColor("#333333"),
                    spaceBefore=1.5, spaceAfter=2)
    sTblCap  = S("TblCap", **_cap_kw)
    sFigCap  = S("FigCap", **_cap_kw)
    sNA      = S("NA",      fontName="Helvetica-Oblique", fontSize=8, leading=11,
                             alignment=TA_CENTER, textColor=colors.grey)
    sBody    = S("Body",    fontName="Helvetica", fontSize=8.5, leading=11,
                             alignment=TA_LEFT,  textColor=colors.black,
                             spaceBefore=0, spaceAfter=3)

    # ── page template (single full-width column only) ─────────────────────────
    def _footer(canvas, doc):
        canvas.saveState()
        canvas.setFont("Helvetica", 7)
        canvas.setFillColor(colors.HexColor("#777777"))
        canvas.drawCentredString(PW / 2, MB * 0.5, str(doc.page))
        canvas.restoreState()

    frame_l = Frame(ML, MB, COL_W, PH - MT - MB, id="left_col",
                    leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0)
    frame_r = Frame(ML + COL_W + GAP, MB, COL_W, PH - MT - MB, id="right_col",
                    leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0)
    frame_main = Frame(ML, MB, BODY_W, PH - MT - MB, id="main",
                       leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0)
    doc = BaseDocTemplate(
        out_pdf, pagesize=A4,
        pageTemplates=[
            PageTemplate(id="full_width",    frames=[frame_main], onPage=_footer),
            PageTemplate(id="results_2col",  frames=[frame_l, frame_r], onPage=_footer),
            PageTemplate(id="appendix_1col", frames=[frame_main], onPage=_footer),
        ],
        title="DFN Calibration and Blockometry Results for the VARENNE Model",
        leftMargin=ML, rightMargin=MR, topMargin=MT, bottomMargin=MB,
    )

    def _auto_col_widths(df, total_w, min_w=28, max_w=70, target_ratio=0.82):
        lens = []
        for col in df.columns:
            vals = [str(col)] + [str(v) for v in df[col].head(50).tolist()]
            max_len = max((len(v) for v in vals), default=4)
            lens.append(max_len)

        raw = [max(min_w, min(max_w, 3.0 * l + 4.0)) for l in lens]

        target_w = total_w * target_ratio

        if sum(raw) < target_w:
            scale = target_w / sum(raw)
            raw = [w * scale for w in raw]

        if sum(raw) > total_w:
            scale = total_w / sum(raw)
            raw = [w * scale for w in raw]

        return raw

    # ── table helpers ─────────────────────────────────────────────────────────
    def _tbl_style(df, site_col=None, numeric_cols=None):
        n = len(df)
        numeric_cols = numeric_cols or []
        cmds = [
            ("BACKGROUND",    (0, 0), (-1, 0), C_HEADBG),
            ("TEXTCOLOR",     (0, 0), (-1, 0), C_HEADTXT),
            ("FONTNAME",      (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE",      (0, 0), (-1, 0), 8),
            ("ALIGN",         (0, 0), (-1, 0), "CENTER"),
            ("FONTNAME",      (0, 1), (-1,-1), "Helvetica"),
            ("FONTSIZE",      (0, 1), (-1,-1), 7.5),
            ("ALIGN",         (0, 0), (-1,-1), "CENTER"),
            ("VALIGN",        (0, 0), (-1,-1), "MIDDLE"),
            ("LINEBELOW",     (0, 0), (-1, 0), 0.5, C_HEADBG),
            ("LINEABOVE",     (0, 0), (-1, 0), 0.5, C_HEADBG),
            ("INNERGRID",     (0, 1), (-1,-1), 0.2, C_BORDER),
            ("BOX",           (0, 0), (-1,-1), 0.4, colors.HexColor("#888888")),
            ("TOPPADDING",    (0, 0), (-1,-1), 5.0),
            ("BOTTOMPADDING", (0, 0), (-1,-1), 5.0),
            ("LEFTPADDING",   (0, 0), (-1,-1), 0.6),
            ("RIGHTPADDING",  (0, 0), (-1,-1), 0.6),
        ]
        for col_idx in numeric_cols:
            cmds.append(("ALIGN", (col_idx, 1), (col_idx, -1), "CENTER"))
        if site_col is not None:
            for i, val in enumerate(df[site_col].values):
                bg = SITE_CLR.get(str(val), C_ALT)
                cmds.append(("BACKGROUND", (0, i+1), (-1, i+1), bg))
        else:
            for i in range(1, n + 1):
                bg = C_ALT if i % 2 == 0 else colors.white
                cmds.append(("BACKGROUND", (0, i), (-1, i), bg))
        return TableStyle(cmds)

    def _df_table(df, widths, site_col=None):
        hdr = [Paragraph(f"<b>{c}</b>",
                             S("h", fontName="Helvetica-Bold", fontSize=8, leading=8.5,
                 alignment=TA_CENTER, textColor=C_HEADTXT))
               for c in df.columns]
        numeric_cols = []
        for idx, col in enumerate(df.columns):
            if site_col is not None and col == site_col:
                continue
            s = pd.to_numeric(df[col], errors="coerce")
            if s.notna().mean() >= 0.8:
                numeric_cols.append(idx)
        data = [hdr] + [[str(v) for v in r] for r in df.itertuples(index=False)]
        row_heights = [12] + [10.0] * len(df)
        t = Table(
            data,
            colWidths=widths,
            rowHeights=row_heights,
            repeatRows=1,
            hAlign="CENTER",
        )
        t.setStyle(_tbl_style(df, site_col, numeric_cols=numeric_cols))
        return t

    # ── image helper ──────────────────────────────────────────────────────────
    def _img(path, width, max_height=None):
        if path and os.path.exists(path):
            if max_height:
                return Image(path, width=width, height=max_height, kind="proportional")
            return Image(path, width=width, kind="proportional")
        return Paragraph("<i>[Figure not available]</i>", sNA)

    # ── load data ─────────────────────────────────────────────────────────────
    blocko_df = _build_blockometry_df()
    trace_df  = _build_trace_df()
    calib_df  = _build_calib_df()

    annex_csv = os.path.join(SCRIPT_DIR, "outputs", "combined",
                             "DFN_fracture_characteristics_VARENNE.csv")
    annex_df  = pd.read_csv(annex_csv) if os.path.exists(annex_csv) else pd.DataFrame()

    # ── figure paths ──────────────────────────────────────────────────────────
    FIGSITE = _first_existing([
        os.path.join(SCRIPT_DIR, "assets", "varenne_pic.png"),
    ])
    
    # Reuse the validated combined persistence/spacing survival plots produced
    # by PERSISTENCE.py / SPACING.py (do not regenerate with different logic).
    FIG_PERSIST = _find_persistence_survival_plot()
    FIG_SPACING = _find_spacing_survival_plot()
    FIG_BLOCKVOL = os.path.join(report_dir, "blockometry_curve.png")
    _generate_blockometry_curve(FIG_BLOCKVOL)
    
    FIG4 = FIG_BLOCKVOL  # Use the generated blockometry curve

    # ── story ─────────────────────────────────────────────────────────────────
    S_ = []

    def add(*items):
        S_.extend(items)

    HR = lambda: HRFlowable(width="100%", thickness=0.5, color=C_RULE, spaceAfter=3, spaceBefore=3)

    # ── TITLE (full-width template) ──────────────────────────────────────────
    add(
        Spacer(1, 0.06 * cm),
        Paragraph("DFN Calibration and Blockometry Results for the VARENNE Model", sTitle),
        HR(),
    )

    # ── SECTION 1 — Site Description (text left, image right) ────────────────
    add(Paragraph("1. Site Description", sSec))

    site_desc_txt = (
        "The VARENNE site is located in the Charlevoix region of Québec, "
        "Canada. The site was investigated for fracture network characterization "
        "and blockometry analysis. Four fracture families (fam1, fam2, fam3, fam4) "
        "were identified from field mapping data. The DFN calibration was performed "
        "using P21 targets derived from trace length measurements, followed by "
        "block volume distribution and shape analysis."
    )
    # Site description: compact text left, small image right
    TEXT_W = BODY_W * 0.52
    IMG_W  = BODY_W * 0.46
    site_img_h = IMG_W * 0.36          # deliberately short to leave room for Results
    site_side = Table(
        [[Paragraph(site_desc_txt, sBody),
          [_img(FIGSITE, IMG_W, site_img_h),
           Paragraph("Figure 1. Location of the VARENNE study site.", sFigCap)]
         ]],
        colWidths=[TEXT_W, IMG_W],
        style=TableStyle([
            ("VALIGN",        (0, 0), (-1, -1), "MIDDLE"),
            ("VALIGN",        (1, 0), (1, 0), "MIDDLE"),
            ("ALIGN",         (1, 0), (1, 0), "CENTER"),
            ("LEFTPADDING",   (0, 0), (-1, -1), 0),
            ("RIGHTPADDING",  (0, 0), (-1, -1), 4),
            ("TOPPADDING",    (0, 0), (-1, -1), 0),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
        ]),
    )
    add(site_side)
    add(HR())

    # ── SECTION 2 — Results ───────────────────────────────────────────────────
    # Layout: sequence of independent 2-col band Tables so each band can
    # split across pages naturally (a single 1-row Table cannot be split).
    add(Paragraph("2. Results", sSec))

    FIG_H = COL_W * 0.60

    def _cap(txt):
        return Paragraph(txt, sFigCap)

    def _sp():
        return Spacer(1, 0.08 * cm)

    _2col_style = TableStyle([
        ("VALIGN",        (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING",   (0, 0), (-1, -1), 0),
        ("RIGHTPADDING",  (0, 0), (-1, -1), 4),
        ("TOPPADDING",    (0, 0), (-1, -1), 0),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ])

    def _band(left, right):
        return Table([[left, right]], colWidths=[COL_W, COL_W],
                     style=_2col_style)

    def _center_tbl(tbl):
        return Table(
            [[tbl]],
            colWidths=[COL_W],
            style=TableStyle([
                ("ALIGN",         (0, 0), (-1, -1), "CENTER"),
                ("VALIGN",        (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING",   (0, 0), (-1, -1), 0),
                ("RIGHTPADDING",  (0, 0), (-1, -1), 0),
                ("TOPPADDING",    (0, 0), (-1, -1), 0),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
            ]),
        )

    # ── compute table data ────────────────────────────────────────────────────
    t1_cols = ["Site", "D20 (m³)", "D50 (m³)", "D80 (m³)", "D90 (m³)"]
    t1_df = blocko_df[t1_cols] if not blocko_df.empty else blocko_df
    cw1 = _auto_col_widths(t1_df, COL_W) if not t1_df.empty else None
    cw2 = _auto_col_widths(trace_df, COL_W) if not trace_df.empty else None

    if not calib_df.empty:
        keep = ["Site", "Family", "P21 terrain", "P21 model", "P32 cal"]
        cd = calib_df[[c for c in keep if c in calib_df.columns]]
        cw3 = _auto_col_widths(cd, COL_W)
    else:
        cd = calib_df
        cw3 = None

    # Band row 1: Fig 2 (left) | Fig 4 (right)
    add(_band(
        [_img(FIG_PERSIST, COL_W, FIG_H),
         _cap("Figure 2. Persistence survival curves — all families.")],
        [_img(FIG4, COL_W, FIG_H),
         _cap("Figure 4. Block volume overlay distribution.")],
    ))
    add(_sp())

    # Band row 2: Table 1 (left) | Table 2 (right)
    t1_block = ([_cap("Table 1. Blockometry percentile summary — VARENNE."),
                 _center_tbl(_df_table(t1_df, cw1, site_col="Site"))]
                if not t1_df.empty else [Paragraph("<i>No data.</i>", sNA)])
    t2_block = ([_cap("Table 2. Mean trace lengths comparison (CloudCompare vs DFN)."),
                 _center_tbl(_df_table(trace_df, cw2, site_col="Site"))]
                if not trace_df.empty else [Paragraph("<i>No data.</i>", sNA)])
    add(_band(t1_block, t2_block))
    add(_sp())

    # Band row 3: Fig 3 (left) | Table 3 (right)
    fig3_block = [_img(FIG_SPACING, COL_W, FIG_H),
                  _cap("Figure 3. Spacing survival curves with lognormal fits — all families.")]
    t3_block = ([_cap("Table 3. DFN calibration summary — VARENNE."),
                 _center_tbl(_df_table(cd, cw3, site_col="Site"))]
                if not cd.empty else [Paragraph("<i>No data.</i>", sNA)])
    add(_band(fig3_block, t3_block))
    add(_sp())

    # ── APPENDIX — Table A1 ───────────────────────────────────────────────────
    add(NextPageTemplate("appendix_1col"))
    add(PageBreak())
    add(
        Paragraph("Appendix", sSec),
        Paragraph("Table A1. DFN fracture characteristics — VARENNE", sTblCap),
        Spacer(1, 0.2 * cm),
    )

    if not annex_df.empty:
        RPP = 38
        ann = annex_df.copy()
        for col in ["dip_deg", "dipdir_deg"]:
            if col in ann.columns:
                ann[col] = ann[col].round(1)
        for col in ["area_m2", "radius_m", "diameter_m"]:
            if col in ann.columns:
                ann[col] = ann[col].round(4)
        cols = list(ann.columns)
        cwa  = [BODY_W / len(cols)] * len(cols)
        chunks = [ann.iloc[i:i+RPP] for i in range(0, len(ann), RPP)]
        for k, chunk in enumerate(chunks):
            hdr = [Paragraph(f"<b>{c}</b>",
                   S(f"ah{k}{j}", fontName="Helvetica-Bold", fontSize=8, leading=8.5,
                     alignment=TA_CENTER, textColor=C_HEADTXT))
                   for j, c in enumerate(cols)]
            data = [hdr] + [[str(v) for v in r] for r in chunk.itertuples(index=False)]
            tbl  = Table(data, colWidths=cwa, repeatRows=1)
            cmds = [
                ("BACKGROUND",    (0,0), (-1,0), C_HEADBG),
                ("TEXTCOLOR",     (0,0), (-1,0), C_HEADTXT),
                ("FONTNAME",      (0,1), (-1,-1), "Helvetica"),
                ("FONTSIZE",      (0,0), (-1,0), 8),
                ("FONTSIZE",      (0,1), (-1,-1), 7.5),
                ("ALIGN",         (0,0), (-1,-1), "CENTER"),
                ("VALIGN",        (0,0), (-1,-1), "MIDDLE"),
                ("INNERGRID",     (0,1), (-1,-1), 0.25, C_BORDER),
                ("BOX",           (0,0), (-1,-1), 0.5, colors.HexColor("#888888")),
                ("LINEBELOW",     (0,0), (-1,0),  0.6, C_HEADBG),
                ("TOPPADDING",    (0,0), (-1,-1), 1.5),
                ("BOTTOMPADDING", (0,0), (-1,-1), 1.5),
                ("LEFTPADDING",   (0,0), (-1,-1), 2),
                ("RIGHTPADDING",  (0,0), (-1,-1), 2),
            ]
            for j, sv in enumerate(chunk["site"].values if "site" in chunk.columns else []):
                cmds.append(("BACKGROUND", (0,j+1), (-1,j+1), SITE_CLR.get(str(sv), C_ALT)))
            tbl.setStyle(TableStyle(cmds))
            add(tbl)
            if k < len(chunks) - 1:
                add(PageBreak())
    else:
        add(Paragraph("<i>No DFN fracture characteristics data found. Run varenne.py first.</i>", sNA))

    doc.build(S_)
    return out_pdf


# backward-compatible alias
def generate_whiteboard_report(project_name: str = "VARENNE") -> str:
    return generate_scientific_report(project_name)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate scientific report for VARENNE DFN analysis.")
    parser.add_argument("--project", default="VARENNE", help="Project name (default: VARENNE)")
    args = parser.parse_args()
    
    print(f"Generating report for {args.project}...")
    output_path = generate_scientific_report(args.project)
    print(f"✅ Report saved: {output_path}")

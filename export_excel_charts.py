#!/usr/bin/env python3
"""
export_excel_charts.py
======================
Génère deux classeurs Excel avec graphiques intégrés (espacement et
persistance), directement éditables dans Word via copier-coller.

Un classeur par analyse :
  outputs/SPACING/spacing_survival_charts.xlsx
  outputs/PERSISTENCE/persistence_survival_charts.xlsx

Chaque classeur contient :
  • Une feuille par famille de discontinuités (fam1, fam2, fam3, fam4)
  • Sur chaque feuille : tableau de données + graphique linéaire intégré
    (courbe empirique en pointillés, courbe ajustée en trait plein)
  • Une feuille "BC_TOTAL" combinant toutes les familles (courbes ajustées uniquement)

Données d'entrée :
  assets/all bc corrected fam.xlsx  (même fichier que SPACING.py et PERSISTENCE.py)

Utilise xlsxwriter (pas openpyxl) pour la génération des graphiques,
afin d'éviter les erreurs de réparation XML dans Excel.
"""

import os
import math
import numpy as np
import pandas as pd
import xlsxwriter

# ============================================================
# CHEMINS
# ============================================================
SCRIPT_DIR      = os.path.dirname(os.path.abspath(__file__))
INPUT_FILE      = os.path.join(SCRIPT_DIR, "assets", "all bc corrected fam.xlsx")
OUT_SPACING     = os.path.join(SCRIPT_DIR, "outputs", "SPACING",     "spacing_survival_charts.xlsx")
OUT_PERSISTENCE = os.path.join(SCRIPT_DIR, "outputs", "PERSISTENCE", "persistence_survival_charts.xlsx")

N_GRID = 150   # points sur la grille x commune
S_MIN  = 0.01  # espacement minimal retenu (m)  — identique à SPACING.py
CAP_Q  = 0.95  # quantile de troncature pour la persistance — identique à PERSISTENCE.py
MIN_N  = 3     # minimum de valeurs pour tracer une courbe

# Couleurs par région (format ARGB sans '#')
REGION_COLORS = {
    "BC1-LEFT":  "FF4472C4",  # bleu
    "BC1-RIGHT": "FFED7D31",  # orange
    "BC2":       "FFA9D18E",  # vert clair
    "BC3-LEFT":  "FFFFC000",  # jaune/or
    "BC3-RIGHT": "FF7030A0",  # violet
    "BCTOTAL":   "FF000000",  # noir (espacement poolé)
    "TOTAL":     "FF000000",  # noir (persistance poolée)
}
DEFAULT_COLOR = "FF808080"  # gris pour les régions non listées


# ============================================================
# FONCTIONS MATHÉMATIQUES
# ============================================================
def _normalize(v):
    v = np.asarray(v, dtype=float)
    n = np.linalg.norm(v)
    return v / n if n > 1e-10 else v


def _norm_cdf(z):
    z = np.asarray(z, dtype=float)
    return 0.5 * (1.0 + np.vectorize(math.erf)(z / math.sqrt(2.0)))


def _fam_key(x):
    try:
        return int(float(str(x).strip().lower().replace("fam", "")))
    except Exception:
        return 9999


def _empirical_on_grid(values, x_grid):
    """Survie empirique P(X ≥ x) évaluée sur la grille x_grid."""
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v) & (v > 0)]
    if len(v) < 2:
        return None
    return np.array([100.0 * float(np.mean(v >= xi)) for xi in x_grid])


def _lognormal_on_grid(values, x_grid):
    """Ajustement lognormal : survie S(x) = 100 × (1 – Φ((ln x – μ) / σ))."""
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v) & (v > 0)]
    if len(v) < 2:
        return None
    lv    = np.log(v)
    mu    = float(np.mean(lv))
    sigma = max(float(np.std(lv, ddof=0)), 1e-12)
    xg    = np.where(x_grid <= 0, 1e-12, x_grid.astype(float))
    return 100.0 * (1.0 - _norm_cdf((np.log(xg) - mu) / sigma))


def _exp_on_grid(values, x_grid):
    """Ajustement exponentiel : survie S(x) = 100 × e^(−λx)."""
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v) & (v > 0)]
    if len(v) < 2:
        return None
    lam = 1.0 / float(np.mean(v))
    return 100.0 * np.exp(-lam * x_grid)


def _best_fit_on_grid(values, x_grid):
    """Sélectionne le meilleur modèle (exp ou lognormal) par RMSE empirique."""
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v) & (v > 0)]
    if len(v) < MIN_N:
        return None, "—"
    y_ln = _lognormal_on_grid(v, x_grid)
    y_ex = _exp_on_grid(v, x_grid)
    y_emp = _empirical_on_grid(v, x_grid)
    if y_ln is None and y_ex is None:
        return None, "—"
    if y_ln is None:
        return y_ex, "EXP"
    if y_ex is None:
        return y_ln, "LOGN"
    if y_emp is None:
        return y_ln, "LOGN"
    rmse_ln = float(np.sqrt(np.mean((y_ln - y_emp) ** 2)))
    rmse_ex = float(np.sqrt(np.mean((y_ex - y_emp) ** 2)))
    if rmse_ln <= rmse_ex:
        return y_ln, "LOGN"
    return y_ex, "EXP"


# ============================================================
# CALCUL ESPACEMENT PERPENDICULAIRE
# ============================================================
def _compute_spacing(df_group):
    """Calcule les espacements perpendiculaires entre traces (vue 2D plan)."""
    if len(df_group) < 3:
        return np.array([])
    sx = df_group["Sx"].to_numpy(dtype=float)
    sy = df_group["Sy"].to_numpy(dtype=float)
    ex = df_group["Ex"].to_numpy(dtype=float)
    ey = df_group["Ey"].to_numpy(dtype=float)
    dx, dy = ex - sx, ey - sy
    dirs = []
    for i in range(len(df_group)):
        d = np.array([dx[i], dy[i]])
        if np.linalg.norm(d) > 1e-10:
            dirs.append(_normalize(d))
    if len(dirs) < 2:
        return np.array([])
    mean_dir = _normalize(np.mean(dirs, axis=0))
    perp = np.array([-mean_dir[1], mean_dir[0]])
    centers = np.column_stack([(sx + ex) / 2.0, (sy + ey) / 2.0])
    proj = np.dot(centers - centers.mean(axis=0), perp)
    spacings = np.diff(np.sort(np.abs(proj)))
    return spacings[spacings > S_MIN]


# ============================================================
# ÉCRITURE FEUILLE + GRAPHIQUE EXCEL  (xlsxwriter)
# ============================================================

# Couleurs xlsxwriter : format '#RRGGBB' (sans canal alpha)
_XL_COLORS = {
    "BC1-LEFT":  "#4472C4",
    "BC1-RIGHT": "#ED7D31",
    "BC2":       "#70AD47",
    "BC3-LEFT":  "#FFC000",
    "BC3-RIGHT": "#7030A0",
    "BCTOTAL":   "#000000",
    "TOTAL":     "#000000",
}
_FAM_COLORS = ["#4472C4", "#ED7D31", "#70AD47", "#C00000", "#7030A0"]
_XL_DEFAULT = "#808080"


def _write_chart_sheet(wb, sheet_name, x_grid, series_list,
                       x_label, y_label, chart_title,
                       x_log=False):
    """
    Crée une feuille xlsxwriter avec tableau de données et graphique linéaire.

    series_list : list of dict avec clés
        'label'  : str
        'y'      : np.ndarray | None
        'color'  : str  région (clé de _XL_COLORS)
        'dashed' : bool — True = pointillés
    """
    ws = wb.add_worksheet(sheet_name[:31])

    # ---- En-têtes ----
    hdr_fmt = wb.add_format({"bold": True})
    ws.write(0, 0, x_label, hdr_fmt)
    for j, s in enumerate(series_list):
        ws.write(0, j + 1, s["label"], hdr_fmt)

    # ---- Données ----
    n_rows = len(x_grid)
    for i, xv in enumerate(x_grid):
        ws.write(i + 1, 0, round(float(xv), 5))
        for j, s in enumerate(series_list):
            y = s.get("y")
            if y is not None and i < len(y) and np.isfinite(y[i]):
                ws.write(i + 1, j + 1, round(float(y[i]), 3))

    # ---- Graphique ----
    chart = wb.add_chart({"type": "scatter", "subtype": "straight"}) if x_log else wb.add_chart({"type": "line"})
    chart.set_title({"name": chart_title})
    x_axis = {"name": x_label}
    if x_log:
        x_pos = np.asarray(x_grid, dtype=float)
        x_pos = x_pos[np.isfinite(x_pos) & (x_pos > 0)]
        if len(x_pos):
            x_axis["log_base"] = 10
            x_axis["min"] = float(np.min(x_pos))
            x_axis["max"] = float(np.max(x_pos))
            x_axis["crossing"] = float(np.min(x_pos))
            x_axis["num_format"] = "0.###"
            x_axis["major_gridlines"] = {"visible": True,
                                           "line": {"color": "#D9D9D9", "width": 0.5}}
            x_axis["minor_gridlines"] = {"visible": True,
                                           "line": {"color": "#ECECEC", "width": 0.35}}
    chart.set_x_axis(x_axis)
    chart.set_y_axis({"name": y_label, "min": 0, "max": 100})
    chart.set_legend({"position": "bottom"})
    chart.set_size({"width": 700, "height": 420})
    chart.set_chartarea({"border": {"none": True}})

    sname = sheet_name[:31]
    for j, s in enumerate(series_list):
        region_key = s.get("color", "")
        color = _XL_COLORS.get(region_key, _XL_DEFAULT)
        dash  = "dash" if s.get("dashed", False) else "solid"
        width = 1.25 if s.get("dashed", False) else 2.25

        series_opts = {
            "name":       [sname, 0, j + 1],
            "values":     [sname, 1, j + 1, n_rows, j + 1],
            "line": {
                "color": color,
                "dash_type": dash,
                "width": width,
            },
            "marker": {"type": "none"},
        }
        if x_log:
            series_opts["categories"] = [sname, 1, 0, n_rows, 0]
        else:
            series_opts["categories"] = [sname, 1, 0, n_rows, 0]
        chart.add_series(series_opts)

    # Placer le graphique sous les données
    anchor_row = n_rows + 3
    ws.insert_chart(anchor_row, 0, chart)


# ============================================================
# FEUILLE COMBINÉE (toutes familles, région TOTAL uniquement)
# ============================================================
def _write_combined_sheet(wb, sheet_name, family_curves,
                          x_label, y_label, chart_title,
                          x_log=False, fit_label_suffix="fit",
                          x_axis_min=None, x_axis_max=None):
    """
    family_curves : list of (fam_label, x_grid, y_fit, y_emp)
    Uses scatter chart for a continuous numeric x-axis.
    y_emp may be None (omitted for spacing).
    """
    if not family_curves:
        return

    sname = sheet_name[:31]
    ws = wb.add_worksheet(sname)
    hdr_fmt = wb.add_format({"bold": True})

    all_x = np.concatenate([np.asarray(entry[1], dtype=float) for entry in family_curves])
    all_x = all_x[np.isfinite(all_x)]
    all_x_pos = all_x[all_x > 0]
    x_min = float(np.min(all_x_pos)) if len(all_x_pos) else 0.0
    x_max = float(np.max(all_x_pos)) if len(all_x_pos) else 1.0
    if x_axis_min is not None:
        x_min = float(x_axis_min)
    if x_axis_max is not None:
        x_max = float(x_axis_max)

    chart = wb.add_chart({"type": "scatter", "subtype": "straight"})
    chart.set_title({"name": chart_title})
    x_axis = {
        "name": x_label,
        "major_gridlines": {"visible": True,
                            "line": {"color": "#D9D9D9", "width": 0.5}},
    }
    if x_log:
        x_axis["log_base"] = 10
        x_axis["min"] = x_min
        x_axis["max"] = x_max
        x_axis["crossing"] = x_min
        x_axis["num_format"] = "0.###"
        x_axis["minor_gridlines"] = {
            "visible": True,
            "line": {"color": "#ECECEC", "width": 0.35},
        }
    else:
        x_axis["min"] = 0
        x_axis["max"] = x_max
        x_axis["crossing"] = "min"
        x_axis["minor_gridlines"] = {
            "visible": False,
        }
    chart.set_x_axis(x_axis)
    chart.set_y_axis({
        "name": y_label,
        "min":  0, "max": 100,
        "major_gridlines": {"visible": True,
                            "line": {"color": "#D9D9D9", "width": 0.5}},
        "minor_gridlines": {"visible": False},
    })
    chart.set_legend({"position": "bottom"})
    chart.set_size({"width": 700, "height": 480})
    chart.set_chartarea({"border": {"none": True}})

    col = 0
    for j, entry in enumerate(family_curves):
        fam_label, xg = entry[0], entry[1]
        y_fit = entry[2] if len(entry) > 2 else None
        y_emp = entry[3] if len(entry) > 3 else None
        fit_name = entry[4] if len(entry) > 4 else fit_label_suffix

        color = _FAM_COLORS[j % len(_FAM_COLORS)]
        n = len(xg)
        fam_short = str(fam_label).replace("fam", "F")

        ws.write(0, col,     f"{fam_short} x",         hdr_fmt)
        ws.write(0, col + 1, f"{fam_short} empirical", hdr_fmt)
        ws.write(0, col + 2, f"{fam_short} {fit_name}", hdr_fmt)
        ws.set_column(col, col + 2, 14)

        for i in range(n):
            ws.write(i + 1, col, float(xg[i]))
            if y_emp is not None and i < len(y_emp) and np.isfinite(y_emp[i]):
                ws.write(i + 1, col + 1, float(y_emp[i]))
            if y_fit is not None and i < len(y_fit) and np.isfinite(y_fit[i]):
                ws.write(i + 1, col + 2, float(y_fit[i]))

        col += 3

    # Legend order to match the figures: all fits first, then all empirical curves.
    col = 0
    for j, entry in enumerate(family_curves):
        fam_label, xg = entry[0], entry[1]
        y_fit = entry[2] if len(entry) > 2 else None
        fit_name = entry[4] if len(entry) > 4 else fit_label_suffix
        color = _FAM_COLORS[j % len(_FAM_COLORS)]
        n = len(xg)
        fam_short = str(fam_label).replace("fam", "F")

        if y_fit is not None:
            chart.add_series({
                "name":       f"{fam_short} {fit_name}",
                "categories": [sname, 1, col,     n, col],
                "values":     [sname, 1, col + 2, n, col + 2],
                "line":       {"color": color, "width": 2.25},
                "marker":     {"type": "none"},
            })
        col += 3

    col = 0
    for j, entry in enumerate(family_curves):
        fam_label, xg = entry[0], entry[1]
        y_emp = entry[3] if len(entry) > 3 else None
        color = _FAM_COLORS[j % len(_FAM_COLORS)]
        n = len(xg)
        fam_short = str(fam_label).replace("fam", "F")

        if y_emp is not None:
            chart.add_series({
                "name":       f"{fam_short} empirical",
                "categories": [sname, 1, col,     n, col],
                "values":     [sname, 1, col + 1, n, col + 1],
                "line":       {"color": color, "dash_type": "dash", "width": 1.25},
                "marker":     {"type": "none"},
            })
        col += 3

    n_rows = len(family_curves[0][1])
    ws.insert_chart(n_rows + 3, 0, chart)


# ============================================================
# MAIN
# ============================================================
def main():
    print(f"Chargement : {INPUT_FILE}")
    if not os.path.exists(INPUT_FILE):
        raise FileNotFoundError(f"Fichier introuvable : {INPUT_FILE}")

    df = pd.read_excel(INPUT_FILE)
    df.columns = df.columns.str.strip()
    df["fam"]    = df["fam"].astype(str).str.strip()
    df["REGION"] = df["REGION"].astype(str).str.strip().str.upper()

    families = sorted(df["fam"].dropna().unique().tolist(), key=_fam_key)
    regions  = sorted(df["REGION"].dropna().unique().tolist())

    print(f"Familles : {families}")
    print(f"Régions  : {regions}")

    # ==========================================================
    # CLASSEUR ESPACEMENT
    # ==========================================================
    print("\n--- Espacement ---")
    os.makedirs(os.path.dirname(OUT_SPACING), exist_ok=True)
    wb_sp = xlsxwriter.Workbook(OUT_SPACING)

    combined_spacing = []  # pour la feuille BC_TOTAL

    for fam in families:
        df_f = df[df["fam"] == fam]
        spacing_data = {}

        for region in regions:
            s = _compute_spacing(df_f[df_f["REGION"] == region])
            if len(s) >= MIN_N:
                spacing_data[region] = s

        s_total = _compute_spacing(df_f)
        if len(s_total) >= MIN_N:
            spacing_data["BCTOTAL"] = s_total

        if not spacing_data:
            print(f"  {fam}: pas assez de données d'espacement")
            continue

        # Grille x commune (log-espace car espacement souvent log-normal)
        all_s = np.concatenate(list(spacing_data.values()))
        all_s = all_s[np.isfinite(all_s) & (all_s > 0)]
        x_min = float(np.min(all_s))
        x_max = float(np.max(all_s))
        x_grid = np.logspace(np.log10(max(x_min, S_MIN * 0.9)), np.log10(x_max), N_GRID)

        # Construction des séries
        series_list = []
        for key in regions + ["BCTOTAL"]:
            if key not in spacing_data:
                continue
            s = spacing_data[key]

            emp = _empirical_on_grid(s, x_grid)
            fit = _lognormal_on_grid(s, x_grid)

            if emp is not None:
                series_list.append({
                    "label":  f"{key} empirique",
                    "y":      emp,
                    "color":  key,
                    "dashed": True,
                })
            if fit is not None:
                series_list.append({
                    "label":  f"{key} ajusté (LOGN)",
                    "y":      fit,
                    "color":  key,
                    "dashed": False,
                })
                if key == "BCTOTAL":
                    combined_spacing.append((fam, x_grid, fit, emp))

        if not series_list:
            print(f"  {fam}: aucune série valide")
            continue

        _write_chart_sheet(
            wb_sp,
            sheet_name=fam,
            x_grid=x_grid,
            series_list=series_list,
            x_label="Espacement perpendiculaire (m)",
            y_label="Occurrence (%) ≥ x",
            chart_title=f"Courbe de survie — Espacement — {fam}",
        )
        print(f"  {fam} : {len(series_list)} séries écrites")

    # Feuille combinée toutes familles
    if combined_spacing:
        _write_combined_sheet(
            wb_sp,
            sheet_name="BC_TOTAL (toutes familles)",
            family_curves=combined_spacing,
            x_label="Perpendicular spacing S (m)",
            y_label="Occurrence (%) ≥ x",
            chart_title="Occurrence (%) vs Spacing — BC_TOTAL (all families)\nEmpirical (dashed) + Lognormal fit (solid)",
            x_log=True,
            fit_label_suffix="LN-fit",
            x_axis_min=0.01,
            x_axis_max=100.0,
        )

    wb_sp.close()
    print(f"\n✅ Sauvegardé : {OUT_SPACING}")

    # ==========================================================
    # CLASSEUR PERSISTANCE
    # ==========================================================
    print("\n--- Persistance ---")
    os.makedirs(os.path.dirname(OUT_PERSISTENCE), exist_ok=True)
    wb_pe = xlsxwriter.Workbook(OUT_PERSISTENCE)

    combined_persistence = []

    for fam in families:
        df_f = df[df["fam"] == fam]
        persist_data = {}

        for region in regions:
            df_fr = df_f[df_f["REGION"] == region]
            if "Length" in df_fr.columns:
                L = pd.to_numeric(df_fr["Length"], errors="coerce").dropna().to_numpy()
                L = L[L > 0]
                if len(L) >= MIN_N:
                    persist_data[region] = L

        # Poolé total
        if "Length" in df_f.columns:
            L_tot = pd.to_numeric(df_f["Length"], errors="coerce").dropna().to_numpy()
            L_tot = L_tot[L_tot > 0]
            if len(L_tot) >= MIN_N:
                # Troncature au quantile CAP_Q (identique à PERSISTENCE.py)
                if len(L_tot) >= 30:
                    cap = np.quantile(L_tot, CAP_Q)
                    L_tot = L_tot[L_tot <= cap]
                persist_data["TOTAL"] = L_tot

        if not persist_data:
            print(f"  {fam}: pas assez de données de persistance")
            continue

        # Grille x commune — limitée au max TOTAL (post-cap CAP_Q)
        if "TOTAL" in persist_data:
            x_max = float(np.max(persist_data["TOTAL"]))
        else:
            all_L = np.concatenate(list(persist_data.values()))
            all_L = all_L[np.isfinite(all_L) & (all_L > 0)]
            x_max = float(np.max(all_L))
        x_min = float(np.min(persist_data["TOTAL"])) if "TOTAL" in persist_data else float(np.min(all_L))
        x_grid = np.logspace(np.log10(max(x_min * 0.9, 1e-6)), np.log10(x_max), N_GRID)

        series_list = []
        for key in regions + ["TOTAL"]:
            if key not in persist_data:
                continue
            L = persist_data[key]

            emp = _empirical_on_grid(L, x_grid)
            if emp is not None:
                emp[0] = 100.0  # ancrer à 100 % en x=0

            y_fit, model_name = _best_fit_on_grid(L, x_grid)
            if y_fit is not None:
                y_fit[0] = 100.0

            if emp is not None:
                series_list.append({
                    "label":  f"{key} empirique",
                    "y":      emp,
                    "color":  key,
                    "dashed": True,
                })
            if y_fit is not None:
                series_list.append({
                    "label":  f"{key} ajusté ({model_name})",
                    "y":      y_fit,
                    "color":  key,
                    "dashed": False,
                })
                if key == "TOTAL":
                    fit_name = "EXP-fit" if model_name == "EXP" else "LN-fit"
                    combined_persistence.append((fam, x_grid, y_fit, emp, fit_name))

        if not series_list:
            print(f"  {fam}: aucune série valide")
            continue

        _write_chart_sheet(
            wb_pe,
            sheet_name=fam,
            x_grid=x_grid,
            series_list=series_list,
            x_label="Persistance / longueur de trace (m)",
            y_label="Occurrence (%) ≥ x",
            chart_title=f"Courbe de survie — Persistance — {fam}",
            x_log=True,
        )
        print(f"  {fam} : {len(series_list)} séries écrites")

    # Feuille combinée toutes familles
    if combined_persistence:
        _write_combined_sheet(
            wb_pe,
            sheet_name="BC_TOTAL (toutes familles)",
            family_curves=combined_persistence,
            x_label="Persistance / longueur de trace (m)",
            y_label="Occurrence (%) ≥ x",
            chart_title="Occurrence (%) vs Persistence — BC_TOTAL\nEmpirical (dashed) + Best fit (solid: Exp or Lognormal)",
            x_log=True,
            fit_label_suffix="LN-fit",
            x_axis_min=0.1,
            x_axis_max=10.0,
        )

    wb_pe.close()
    print(f"✅ Sauvegardé : {OUT_PERSISTENCE}")

    print("\n=== TERMINÉ ===")
    print(f"  Espacement  : {OUT_SPACING}")
    print(f"  Persistance : {OUT_PERSISTENCE}")
    print("\nPour Word : ouvrez le fichier Excel, cliquez sur un graphique,")
    print("  faites Ctrl+C, puis dans Word Collage spécial → Graphique Microsoft Excel.")


if __name__ == "__main__":
    main()

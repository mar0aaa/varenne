import os
import sys
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from run_site import SITE_CONFIGS, run_site
from extract_dfn_diameters import export_dfn_fracture_characteristics
from kco_model import BlastDesign, predict_kco

MENU_ITEMS = {
    # ──── DFN + Blockometry ────
    "1": ("VARENNE",      "VARENNE     — DFN calibration + blockometry (fam1, fam2, fam3, fam4)"),

    # ──── Persistence / Spacing ────
    "2": ("PERSISTENCE",  "PERSISTENCE — Trace length survival fits"),
    "3": ("SPACING",      "SPACING     — Perpendicular spacing survival fits"),

    # ──── Exports ────
    "4": ("DFNCHAR",      "Export DFN fracture characteristics (dip, dipdir, area, diameter)"),
    "5": ("BLOCKOMETRY",  "Blockometry percentile summary"),
    "6": ("TRACELENGTHS", "Export trace length Excel: DFN vs CloudCompare"),
    "7": ("PLOTVOLUMES",  "Plot block volume distribution"),

    # ──── Post-blast fragmentation ────
    "8": ("KCO",          "KCO         — Post-blast fragmentation prediction (Kuz-Ram/KCO)"),
}


def _run_persistence():
    import importlib.util
    spec = importlib.util.spec_from_file_location("PERSISTENCE", os.path.join(SCRIPT_DIR, "PERSISTENCE.py"))
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.main()


def _run_spacing():
    import importlib.util
    spec = importlib.util.spec_from_file_location("SPACING", os.path.join(SCRIPT_DIR, "SPACING.py"))
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)


def _run_blockometry():
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "export_blockometry_charts",
        os.path.join(SCRIPT_DIR, "export_blockometry_charts.py")
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.main()


def _run_plotvolumes():
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "PLOTVARENNE",
        os.path.join(SCRIPT_DIR, "PLOTVARENNE.py")
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)


def _prompt_float(label, allow_blank=False, default=None):
    while True:
        suffix = f" [{default}]" if default is not None else ""
        raw = input(f"  {label}{suffix}: ").strip()
        if raw == "":
            if allow_blank:
                return None
            if default is not None:
                return default
            print("    -> required, please enter a number.")
            continue
        try:
            return float(raw)
        except ValueError:
            print("    -> not a valid number, try again.")


def _prompt_str(label, choices=None, default=None):
    while True:
        suffix = f" [{default}]" if default is not None else ""
        raw = input(f"  {label}{suffix}: ").strip()
        if raw == "" and default is not None:
            return default
        if choices and raw not in choices:
            print(f"    -> must be one of {choices}")
            continue
        if raw == "":
            print("    -> required, please enter a value.")
            continue
        return raw


def _load_mean_joint_spacing_varenne():
    """
    Load the mean joint spacing Sj (m) for VARENNE from the project's own
    SPACING output (outputs/SPACING/raw_spacing_values.xlsx), averaged
    over all families. Returns None if the file is not available.
    """
    import pandas as pd
    path = os.path.join(SCRIPT_DIR, "outputs", "SPACING", "raw_spacing_values.xlsx")
    if not os.path.exists(path):
        return None
    df = pd.read_excel(path)
    df = df[df["region"] == "VARENNE"]
    if df.empty:
        return None
    return float(df["spacing"].mean())


def _load_block_volumes_varenne():
    """
    Load DFN block volumes (m3) for VARENNE from the blockometry export
    (outputs/VARENNE/06_blockometry_plots/block_shape_data_VARENNE.csv).
    Returns None if the file is not available.
    """
    import pandas as pd
    path = os.path.join(SCRIPT_DIR, "outputs", "VARENNE",
                        "06_blockometry_plots", "block_shape_data_VARENNE.csv")
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    if "volume" not in df.columns:
        return None
    vols = df["volume"].to_numpy(dtype=float)
    vols = vols[np.isfinite(vols) & (vols > 0.0)]
    if vols.size == 0:
        return None
    return vols


def _run_kco():
    """
    Interactive KCO (Kuz-Ram) post-blast fragmentation prediction.

    Auto-loads from the project's own outputs where available:
      - mean_joint_spacing_m: from outputs/SPACING/raw_spacing_values.xlsx
        (VARENNE, mean over all families).
      - block volumes: from outputs/VARENNE/06_blockometry_plots/
        block_shape_data_VARENNE.csv (only used if the "dfn" block-size
        method is selected).

    Every other blast-design parameter (drill pattern, explosive charge,
    rock mechanical properties, rock mass/joint condition, timing
    scatter) is NOT available anywhere in this project and must be
    entered by hand; the function prompts for each one explicitly so no
    value is silently assumed.
    """
    print("\n" + "=" * 60)
    print("  KCO — Post-blast fragmentation prediction")
    print("=" * 60)

    sj = _load_mean_joint_spacing_varenne()
    if sj is not None:
        print(f"  Auto-loaded mean_joint_spacing_m (Sj) from "
              f"outputs/SPACING: {sj:.4f} m")
    else:
        print("  outputs/SPACING/raw_spacing_values.xlsx not found; "
              "Sj must be entered manually.")
        sj = _prompt_float("Mean joint spacing Sj (m)")

    print("\n  -- Drill & blast geometry (not in the project, required) --")
    hole_diameter_mm = _prompt_float("Hole diameter D (mm)")
    burden_m = _prompt_float("Burden B (m)")
    spacing_m = _prompt_float("Hole spacing S (m)")
    bench_height_m = _prompt_float("Bench height H (m)")
    subdrill_m = _prompt_float("Subdrill (m)", default=0.0)
    drill_accuracy_sd_m = _prompt_float("Drilling accuracy std dev W (m)")

    print("\n  -- Charge (not in the project, required) --")
    bottom_charge_m = _prompt_float("Bottom charge length Lb (m)")
    column_charge_m = _prompt_float("Column charge length Lc (m)", default=0.0)
    total_charge_m = _prompt_float("Total charge length above grade Ltot (m)")
    charge_per_hole_kg = _prompt_float("Charge per hole Q (kg)")
    explosive_name = _prompt_str("Explosive name", default="ANFO")
    s_anfo_pct = _prompt_float("Weight strength vs ANFO s_ANFO (%)", default=100.0)

    print("\n  -- Powder factor (not in the project, required) --")
    powder_factor_mode = _prompt_str(
        "Powder factor mode ('reported' or 'calculated')",
        choices=["reported", "calculated"], default="reported")
    powder_factor_reported_kg_m3 = None
    if powder_factor_mode == "reported":
        powder_factor_reported_kg_m3 = _prompt_float("Reported powder factor q (kg/m3)")

    print("\n  -- Rock mechanical properties (not in the project, required) --")
    rock_density_kg_m3 = _prompt_float("Rock density rho (kg/m3)")
    ucs_mpa = _prompt_float("UCS sigma_c (MPa)")
    youngs_modulus_gpa = _prompt_float("Young's modulus E (GPa)")

    print("\n  -- Rock mass / joint factor (not in the project, required) --")
    rock_mass_case = _prompt_str(
        "Rock mass case ('powdery_friable', 'jointed', 'massive')",
        choices=["powdery_friable", "jointed", "massive"], default="jointed")
    jpa_case = _prompt_str(
        "JPA case ('dip_out_of_face', 'strike_perpendicular_to_face', "
        "'dip_into_face')",
        choices=["dip_out_of_face", "strike_perpendicular_to_face",
                "dip_into_face"])

    print("\n  -- In-situ block size (required) --")
    block_size_method = _prompt_str(
        "Block size method ('user_defined' or 'dfn')",
        choices=["user_defined", "dfn"], default="user_defined")
    in_situ_block_size_m = None
    block_statistic = None
    block_volumes = None
    if block_size_method == "user_defined":
        in_situ_block_size_m = _prompt_float("In-situ block size (m)")
    else:
        block_volumes = _load_block_volumes_varenne()
        if block_volumes is None:
            print("  block_shape_data_VARENNE.csv not found or empty; "
                  "falling back to 'user_defined'.")
            block_size_method = "user_defined"
            in_situ_block_size_m = _prompt_float("In-situ block size (m)")
        else:
            print(f"  Auto-loaded {block_volumes.size} block volumes from "
                  "blockometry export.")
            block_size_method = _prompt_str(
                "Volume-to-length conversion "
                "('equivalent_cube' or 'equivalent_sphere')",
                choices=["equivalent_cube", "equivalent_sphere"],
                default="equivalent_cube")
            block_statistic = _prompt_str(
                "Statistic on the block distribution "
                "('median', 'p80', 'p95', 'maximum')",
                choices=["median", "p80", "p95", "maximum"], default="median")

    print("\n  -- Timing (not in the project, required) --")
    timing_scatter_factor_ns = _prompt_float(
        "Timing-scatter factor n_s (Cunningham 2005)", default=1.0)

    print("\n  -- g(n) shift factor mode --")
    shift_factor_mode = _prompt_str(
        "Shift factor mode ('no_shift' or 'mean_to_median_shift')",
        choices=["no_shift", "mean_to_median_shift"], default="no_shift")

    design = BlastDesign(
        name="VARENNE",
        hole_diameter_mm=hole_diameter_mm,
        burden_m=burden_m,
        spacing_m=spacing_m,
        bench_height_m=bench_height_m,
        total_charge_m=total_charge_m,
        bottom_charge_m=bottom_charge_m,
        column_charge_m=column_charge_m,
        subdrill_m=subdrill_m,
        drill_accuracy_sd_m=drill_accuracy_sd_m,
        charge_per_hole_kg=charge_per_hole_kg,
        powder_factor_reported_kg_m3=powder_factor_reported_kg_m3,
        powder_factor_mode=powder_factor_mode,
        s_anfo_pct=s_anfo_pct,
        explosive_name=explosive_name,
        rock_density_kg_m3=rock_density_kg_m3,
        ucs_mpa=ucs_mpa,
        youngs_modulus_gpa=youngs_modulus_gpa,
        rock_mass_case=rock_mass_case,
        jpa_case=jpa_case,
        mean_joint_spacing_m=sj,
        block_size_method=block_size_method,
        block_statistic=block_statistic,
        in_situ_block_size_m=in_situ_block_size_m,
        timing_scatter_factor_ns=timing_scatter_factor_ns,
        shift_factor_mode=shift_factor_mode,
    )

    try:
        result = predict_kco(design, block_volumes_m3=block_volumes)
    except ValueError as exc:
        print(f"\n❌ KCO prediction failed: {exc}")
        return

    print("\n" + result.audit_table())


def _export_trace_length_excel():
    import pandas as pd
    import openpyxl
    from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
    from openpyxl.utils import get_column_letter

    assets = os.path.join(SCRIPT_DIR, "assets")
    trace_file = SITE_CONFIGS["VARENNE"]["trace_name"]
    region_filter = SITE_CONFIGS["VARENNE"]["region_filter"]
    persist_dir = os.path.join(SCRIPT_DIR, "outputs", "VARENNE", "07_persistence")

    thin   = Side(style="thin", color="AAAAAA")
    border = Border(left=thin, right=thin, top=thin, bottom=thin)
    center = Alignment(horizontal="center", vertical="center")
    hdr_fill = PatternFill("solid", fgColor="1F4E79")
    hdr_font = Font(bold=True, color="FFFFFF", size=11)
    row_fill = PatternFill("solid", fgColor="DEEAF1")

    def _style_header(ws, n):
        for c in range(1, n + 1):
            cell = ws.cell(row=1, column=c)
            cell.fill = hdr_fill; cell.font = hdr_font
            cell.alignment = center; cell.border = border
            ws.column_dimensions[get_column_letter(c)].width = 18

    def _style_row(ws, r, n):
        for c in range(1, n + 1):
            cell = ws.cell(row=r, column=c)
            cell.fill = row_fill; cell.alignment = center; cell.border = border

    wb = openpyxl.Workbook()

    # Sheet 1 — CC trace lengths
    ws1 = wb.active; ws1.title = "CC Trace Lengths"
    ws1.append(["Site", "Family", "Length (m)"])
    _style_header(ws1, 3)
    path = os.path.join(assets, trace_file)
    if os.path.exists(path):
        df = pd.read_csv(path)
        if region_filter and "REGION" in df.columns:
            df = df[df["REGION"] == region_filter]
        df["Length"] = pd.to_numeric(df["Length"], errors="coerce")
        df = df.dropna(subset=["fam", "Length"])
        df = df[df["Length"] > 0]
        for _, row in df.iterrows():
            ws1.append(["VARENNE", str(row["fam"]), round(float(row["Length"]), 4)])
            _style_row(ws1, ws1.max_row, 3)

    # Sheet 2 — DFN trace lengths
    ws2 = wb.create_sheet("DFN Trace Lengths")
    ws2.append(["Site", "Family", "Radius (m)", "Diameter (m)"])
    _style_header(ws2, 4)
    if os.path.isdir(persist_dir):
        for fname in sorted(os.listdir(persist_dir)):
            if fname.startswith("persistence_radii_") and fname.endswith(".csv"):
                fam = fname.replace("persistence_radii_", "").replace(".csv", "")
                ddf = pd.read_csv(os.path.join(persist_dir, fname))
                for _, row in ddf.iterrows():
                    r = round(float(row.get("radius_m", float("nan"))), 4)
                    d = round(float(row.get("diameter_m", float("nan"))), 4)
                    ws2.append(["VARENNE", fam, r, d])
                    _style_row(ws2, ws2.max_row, 4)

    out = os.path.join(SCRIPT_DIR, "outputs", "trace_length_comparison.xlsx")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    wb.save(out)
    print(f"\n✅ Saved: {out}")


def main():
    print("\n" + "=" * 60)
    print("  VARENNE — DFN Analysis Menu")
    print("=" * 60)
    for key in sorted(MENU_ITEMS, key=lambda x: int(x)):
        print(f"  [{key}] {MENU_ITEMS[key][1]}")
    print("  [0] Exit")
    print("=" * 60)

    choice = input("Enter choice: ").strip()

    if choice == "0":
        print("Goodbye.")
        return

    if choice not in MENU_ITEMS:
        print(f"Unknown option: {choice}")
        return

    action = MENU_ITEMS[choice][0]

    if action == "VARENNE":
        run_site(SITE_CONFIGS["VARENNE"])

    elif action == "PERSISTENCE":
        _run_persistence()

    elif action == "SPACING":
        _run_spacing()

    elif action == "DFNCHAR":
        out_dir = os.path.join(SCRIPT_DIR, "outputs", "VARENNE", "07_persistence")
        export_dfn_fracture_characteristics(
            dfn_vtk_dir=os.path.join(SCRIPT_DIR, "outputs", "VARENNE", "03_dfn_vtk"),
            output_dir=out_dir,
            site_name="VARENNE",
        )

    elif action == "BLOCKOMETRY":
        _run_blockometry()

    elif action == "TRACELENGTHS":
        _export_trace_length_excel()

    elif action == "PLOTVOLUMES":
        _run_plotvolumes()

    elif action == "KCO":
        _run_kco()

    else:
        print(f"Action '{action}' not yet implemented.")


if __name__ == "__main__":
    main()

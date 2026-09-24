# ============================================================
# run_kco_varenne_demo.py
#
# End-to-end run of classical KCO and KCO* for Varenne, producing the
# single comparison plot (in-situ DFN, classical KCO, KCO*[, WipFrag]).
#
# INPUT PROVENANCE (read this before trusting any number below):
#
#   REAL, user-provided Varenne values (used as-is):
#     - hole_diameter_mm = 114.0        (D)
#     - burden_m         = 3.4          (B)
#     - spacing_m        = 4.1          (S)
#     - bench_height_m   = 14.0         (H)
#     - subdrill_m       = 0.0
#     - total_charge_m   = 14.0         (no subdrill -> full column)
#     - charge_per_hole_kg = 165.0      (Q)
#     - powder_factor_reported_kg_m3 = 0.8 (q, "reported")
#     - explosive_name   = "Dyno Nobel XL900 emulsion"
#
#   REAL, project-derived values (loaded from existing outputs):
#     - mean_joint_spacing_m: outputs/SPACING/raw_spacing_values.xlsx
#       (mean over families) -- used by CLASSICAL KCO only.
#     - Xmax block-volume population (for BOTH classical KCO and KCO*):
#       the SAME pooled S x B x H blast-cell population used for Sj*
#       (professor's request: Xmax = min(P95(Sj*), B, S) with
#       Sj* = V^(1/3) of the pooled blast-cell blocks). The 50x50x50 m
#       DFN (06_blockometry_plots/block_shape_data_VARENNE.csv) is still
#       loaded for the printed comparison only and is NO LONGER used in
#       any calculation. xmax_block_percentile=95,
#       xmax_block_size_method="equivalent_cube".
#     - Sj* block-volume population (KCO* only): generated HERE, for
#       real, using blast_cell_dfn.generate_blast_cell_realizations with
#       the REAL B=3.4, S=4.1, H=14.0 above (NOT placeholder dimensions).
#       This reuses the calibrated Varenne fracture parameters unchanged
#       (see blast_cell_dfn.py). 30 realizations (seeds 3001-3030) are
#       pooled -- a PROVISIONAL count: the convergence criterion
#       proposed in blast_cell_dfn_diagnostics.py was only tested at
#       placeholder dimensions (3x4x12 m), NOT at these real 3.4x4.1x14 m
#       dimensions. Re-run the convergence analysis at the real
#       dimensions before treating this as final.
#
#   ESTIMATED / NOT CONFIRMED (no source in the project or from you;
#   flagged loudly, must be confirmed before this is a final result):
#     - drill_accuracy_sd_m   = 0.3   (placeholder; must satisfy 0<=W<B)
#     - s_anfo_pct            = 100.0 (ANFO-equivalent placeholder; a
#                                       web search found Dyno Nobel Titan
#                                       product RWS values of 0.77-0.94,
#                                       but none specifically for
#                                       "XL900" -- NOT used, to avoid
#                                       guessing)
#     - rock_mass_case         = "jointed" (reasonably justified: this
#                                 whole project characterises Varenne as
#                                 a jointed rock mass via DFN)
#     - jpa_case               = "strike_perpendicular_to_face"
#     - timing_scatter_factor_ns = 1.0 (Cunningham 2005 neutral baseline)
#
#   INTACT-ROCK PROPERTIES (professor-provided, corrected values):
#     - rock_density_kg_m3    = 2700.0
#     - ucs_mpa                = 200.0
#     - youngs_modulus_gpa     = 60.0
#
#   KCO* CLASS WEIGHTING (professor's request): w_i = sum(V_j in class) /
#   sum(V_j total), i.e. block-VOLUME weighting, not block-count
#   weighting. Both weights are still reported in the output tables.
#
#   MEASURED (used as the WipFrag curve in the comparison plot):
#     - "Résultats WIPFRAG.xlsx" in assets/ -- full-blast WipFrag
#       gradation table (size mm vs % passing).
#     - Current KCO/KCO* run: Varenne production blast (B=3.4 m, S=4.1 m,
#       H=14 m, Q=165 kg, blast date 2026-07-23) -> sheet "2026-07-23",
#       "travaillée" gradation column is used.
#     - 2026-05-11 full-blast data is NOT used, even though its B/S/H/Q are
#       numerically close, because the current production run corresponds
#       to the July 2026 blast.
#
# Does not modify kco_model.py, kco_star_model.py, blast_cell_dfn.py,
# run_site.py or main.py.
# ============================================================

import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.stats import gaussian_kde
import xlsxwriter

from run_site import SCRIPT_DIR
from kco_model import BlastDesign, predict_kco
from kco_star_model import (predict_kco_star, plot_main_comparison,
                            in_situ_size_at_passing,
                            in_situ_passing_from_sj_star)
from blast_cell_dfn import generate_blast_cell_realizations
from main import _load_mean_joint_spacing_varenne, _load_block_volumes_varenne


# ---- WipFrag measured fragmentation curve ----
# Current Varenne production blast: 2026-07-23, "travaillée" gradation.
WIPFRAG_XLSX = os.path.join(SCRIPT_DIR, "assets", "Résultats WIPFRAG.xlsx")
WIPFRAG_SHEET = "2026-07-23"
WIPFRAG_CURVE = "travaillée"


def load_wipfrag_curve(xlsx_path: str, sheet_name: str, curve_name: str):
    """Return (size_mm, passing_pct) for the selected WipFrag gradation."""
    df = pd.read_excel(xlsx_path, sheet_name=sheet_name, header=None)
    # Locate the column header matching curve_name (e.g. "travaillée")
    pct_col, size_col = -1, -1
    for i, row in df.iterrows():
        for j, val in enumerate(row):
            if (isinstance(val, str)
                    and val.strip().lower() == curve_name.lower()):
                # percentiles are in this column, sizes in the next column
                pct_col = j
                size_col = j + 1
                start_row = i + 2  # skip the "Granulo" / "D" sub-header rows
                break
        if pct_col >= 0:
            break
    if pct_col < 0 or size_col < 0 or size_col >= df.shape[1]:
        raise ValueError(f"WipFrag curve {curve_name!r} not found in "
                         f"{xlsx_path} sheet {sheet_name!r}")
    sizes, passing = [], []
    for i in range(start_row, df.shape[0]):
        pct = pd.to_numeric(df.iat[i, pct_col], errors="coerce")
        mm = pd.to_numeric(df.iat[i, size_col], errors="coerce")
        if pd.notna(pct) and pd.notna(mm):
            passing.append(float(pct))
            sizes.append(float(mm))
    if not sizes:
        raise ValueError(f"No numeric WipFrag rows found for {curve_name!r}")
    arr = np.array(sorted(zip(sizes, passing), key=lambda x: x[0]))
    return arr[:, 0], arr[:, 1]


# ---- REAL Varenne blast geometry / explosive (user-provided) ----
B_REAL = 3.4
S_REAL = 4.1
H_REAL = 14.0

design = BlastDesign(
    name="VARENNE (real geometry + corrected intact-rock properties)",
    hole_diameter_mm=114.0,
    burden_m=B_REAL,
    spacing_m=S_REAL,
    bench_height_m=H_REAL,
    subdrill_m=0.0,
    total_charge_m=14.0,
    bottom_charge_m=0.0,
    column_charge_m=0.0,
    drill_accuracy_sd_m=0.3,                 # ESTIMATE
    charge_per_hole_kg=165.0,
    powder_factor_reported_kg_m3=0.8,
    powder_factor_mode="reported",
    s_anfo_pct=100.0,                        # ESTIMATE
    explosive_name="Dyno Nobel XL900 emulsion",
    rock_density_kg_m3=2700.0,               # professor-provided
    ucs_mpa=200.0,                           # professor-provided
    youngs_modulus_gpa=60.0,                 # professor-provided
    rock_mass_case="jointed",
    jpa_case="strike_perpendicular_to_face", # ESTIMATE
    timing_scatter_factor_ns=1.0,
    shift_factor_mode="no_shift",
)

print("=" * 70)
print("INPUT PROVENANCE -- see run_kco_varenne_demo.py header for full list")
print("=" * 70)
print("REAL: B=3.4 m, S=4.1 m, H=14.0 m, D=114 mm, Q=165 kg, q=0.8 kg/m3")
print("INTACT ROCK (professor-provided): rho=2700 kg/m3, UCS=200 MPa, E=60 GPa")
print("ESTIMATED (unconfirmed): drill_accuracy_sd_m, s_anfo_pct, jpa_case")
print("Observed but NOT usable as a curve: 3 oversize boulders ~1 m each")
print()

# ---- Classical KCO: real mean joint spacing + blast-cell (S x B x H) Xmax ----
sj_classical = _load_mean_joint_spacing_varenne()
if sj_classical is None:
    raise RuntimeError("outputs/SPACING/raw_spacing_values.xlsx not found; "
                       "cannot get a real mean_joint_spacing_m for classical KCO")

# Professor-prescribed reference value for the classical KCO baseline.
# The SPACING analysis gives a measured mean of ~0.7532 m, but Prof. Aubertin
# instructed the classical KCO to use S_j = 1.5 m as the representative spacing.
# KCO* is NOT affected: it continues to use the 12 DFN-derived S_{j,i}^* values.
SJ_CLASSICAL_PROFESSOR_M = 1.5
print(f"SPACING-derived mean joint spacing: {sj_classical:.4f} m")
print(f"Classical KCO mean_joint_spacing_m (professor-prescribed): "
     f"{SJ_CLASSICAL_PROFESSOR_M:.4f} m")

large_domain_vols = _load_block_volumes_varenne()
if large_domain_vols is not None:
    print(f"Large-domain (50x50x50 m) block volumes loaded for reference "
          f"only (NOT used for Xmax): {large_domain_vols.size} blocks, "
          f"P95 cube = {np.percentile(large_domain_vols, 95) ** (1/3):.4f} m")

# ---- Sj* / Xmax population: REAL S x B x H DFN block volumes ----
print(f"\nGenerating REAL S x B x H DFN blast-cell realizations "
     f"(B={B_REAL}, S={S_REAL}, H={H_REAL} m, 30 seeds, PROVISIONAL count "
     "-- convergence not yet re-validated at these exact dimensions)...")
REUSE_POOLED_CSV = True  # reuse the existing pooled CSV (same seeds) if present
_pooled_csv = os.path.join(SCRIPT_DIR, "outputs", "VARENNE",
                           "08_blastcell_SxBxH",
                           "block_volumes_blastcell_pooled.csv")
if REUSE_POOLED_CSV and os.path.isfile(_pooled_csv):
    sj_star_vols = pd.read_csv(_pooled_csv)["volume"].dropna().values.astype(float)
    sj_star_vols = sj_star_vols[np.isfinite(sj_star_vols) & (sj_star_vols > 0.0)]
    print(f"Reusing pooled S x B x H block volumes from {_pooled_csv}: "
          f"{sj_star_vols.size} blocks")
else:
    blastcell = generate_blast_cell_realizations(
        burden_m=B_REAL, spacing_m=S_REAL, bench_height_m=H_REAL,
        seeds=list(range(3001, 3031)),
        site_key="VARENNE",
    )
    sj_star_vols = blastcell["pooled_volumes_m3"]
    print(f"Pooled S x B x H block volumes (real, 30 realizations): "
         f"{sj_star_vols.size} blocks")
    print(f"Saved: {blastcell['pooled_csv_path']}")

# Xmax population = the SAME pooled blast-cell blocks as Sj*:
#   Xmax = min(P95(Sj*), B, S), Sj* = V^(1/3)
xmax_vols = sj_star_vols
print(f"Xmax population = pooled S x B x H blocks: n={xmax_vols.size}, "
      f"P95(Sj*) = {np.percentile(xmax_vols, 95) ** (1/3):.4f} m")

design_classical = BlastDesign(
    **{**design.__dict__,
      "mean_joint_spacing_m": SJ_CLASSICAL_PROFESSOR_M,
      "block_size_method": "equivalent_cube",
      "block_statistic": "p95",
    }
)
kco_result = predict_kco(design_classical, block_volumes_m3=xmax_vols)
print("\n" + kco_result.audit_table())

N_LOG_BINS = 10  # configurable; default 10 per the log-bins methodology
TAIL_LOW_PCT = 1.0   # P1
TAIL_HIGH_PCT = 99.0  # P99
WEIGHT_METHOD = "volume"  # professor's request: block-volume weighting
# Suffix appended to every output filename of this run so that previous
# outputs (count-weighted UCS=100: no suffix; volume-weighted with the
# 50 m-DFN Xmax: "_volume_weighted"; pre-correction JPS mapping:
# "_volume_weighted_xmaxSBH") are NOT overwritten.
OUT_SUFFIX = "_volume_weighted_xmaxSBH_jps80_wipfines"
kco_star_result = predict_kco_star(
    design, sj_star_vols, xmax_vols,
    xmax_block_size_method="equivalent_cube",
    xmax_block_percentile=95,
    class_method="log_bins_p1_p99_tails",
    n_log_bins=N_LOG_BINS,
    tail_low_pct=TAIL_LOW_PCT,
    tail_high_pct=TAIL_HIGH_PCT,
    percentile_edges=[0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
    representative_method="median",
    weight_method=WEIGHT_METHOD,
)
print("\n" + kco_star_result.audit_table())

# Count-weighted KCO* kept only for the side-by-side weight/X50 table
# (same classes, same per-class KCO chain; only w_i differs).
kco_star_result_count = predict_kco_star(
    design, sj_star_vols, xmax_vols,
    xmax_block_size_method="equivalent_cube",
    xmax_block_percentile=95,
    class_method="log_bins_p1_p99_tails",
    n_log_bins=N_LOG_BINS,
    tail_low_pct=TAIL_LOW_PCT,
    tail_high_pct=TAIL_HIGH_PCT,
    representative_method="median",
    weight_method="count",
)

unique_groups = kco_star_result.unique_curves()
print(f"\nDistinct individual KCO* class curves: {len(unique_groups)} "
      f"among {len(kco_star_result.classes)} classes")
for g in unique_groups:
    print(f"  JPS={g['jps']:>2}  X50={g['x50_mm']:.2f} mm  b={g['b']:.3f}  "
          f"classes={g['class_indices']}  w_{WEIGHT_METHOD}={g['weight']:.4f}")

# ---- Report how the N_LOG_BINS central log bins + 2 tail classes are
# populated (point 10: must be reported BEFORE the table, using all
# n_retained blocks, no cutoff) ----
print(f"\nSj* class population report ({N_LOG_BINS} equal-width bins in "
     f"ln(Sj*) inside [P{TAIL_LOW_PCT:g}, P{TAIL_HIGH_PCT:g}], plus explicit "
     f"lower-/upper-tail classes; all {kco_star_result.n_retained} blocks, "
     "no cutoff applied):")
for i, c in enumerate(kco_star_result.classes, start=1):
    tail_tag = f" [{c.is_tail} tail]" if c.is_tail else ""
    print(f"  class {i:>2}{tail_tag}: "
         f"[{c.bin_edge_low_m:.6g}, {c.bin_edge_high_m:.6g}] m "
         f"-> n={c.n_blocks:>4d}  w_count={c.weight_count:.4f}  "
         f"w_volume={c.weight_volume:.6f}")
if kco_star_result.warnings:
    print("\nWARNINGS (log bins / KCO*):")
    for msg in kco_star_result.warnings:
        print(f"  - {msg}")

# ---- Comparison plot (in-situ, KCO, KCO*, and WipFrag measured) ----
# Load the 2026-07-23 "travaillée" gradation for the Varenne production
# blast (B=3.4, S=4.1, H=14, Q=165); explicitly use this curve because it
# corresponds to the current KCO/KCO* run, not because of any visual fit.
wipfrag_size_mm, wipfrag_passing_pct = load_wipfrag_curve(
    WIPFRAG_XLSX, WIPFRAG_SHEET, WIPFRAG_CURVE)

# ---- Prof. Aubertin's crushed-zone missing-fines correction ----
# The WipFrag image-analysis curve misses the finest fraction.  The missing
# fines volume is estimated as 4 cylinders of crushed rock around each of
# 4 blastholes, with radius r_c from the dynamic pressure in the hole.
# Constants (professor-provided / Varenne blast design):
#   r_0 = d/2 = 0.057 m, rho_e = 1200 kg/m3, D = 4500 m/s,
#   C_0 = 150 MPa, C_0,dyn = 3*C_0 = 450 MPa,
#   S = 4.1 m, B = 3.4 m, H = 14.0 m.
_RHO_E = 1.2e3          # kg/m3
_D_EXP = 4500.0         # m/s
_R0 = 0.057             # m
_C0 = 150.0             # MPa
_C0_DYN = 3.0 * _C0     # MPa
_H = H_REAL
P_CRUSH_MPA = _RHO_E * _D_EXP**2 / 8.0 / 1e6   # MPa
R_C = _R0 * np.sqrt(P_CRUSH_MPA / _C0_DYN)     # m
# NOTE: full-cylinder volume 4*pi*r_c^2*H is Prof. Aubertin's explicit
# simplified estimate and is kept here. The crushed-rock annulus excluding
# the blasthole volume, 4*pi*(r_c^2 - r_0^2)*H, could be considered in a
# more refined analysis (Varenne: 3.287 m3 -> P_fines = 1.684 % instead of
# 3.858 m3 -> 1.977 %).
V_FINES = 4.0 * np.pi * R_C**2 * _H            # m3
V_BLOCK = S_REAL * B_REAL * _H                 # m3
P_FINES_PCT = 100.0 * V_FINES / V_BLOCK        # %
print("\n==== Prof. Aubertin crushed-zone missing-fines estimate ====")
print(f"P (detonation pressure)           = {P_CRUSH_MPA:.4f} MPa")
print(f"C_0,dyn                           = {_C0_DYN:.1f} MPa")
print(f"r_c (crushed-zone radius)         = {R_C:.6f} m")
print(f"V_fines = 4*pi*r_c^2*H            = {V_FINES:.6f} m3")
print(f"V_block = S*B*H                   = {V_BLOCK:.3f} m3")
print(f"P_x,fines = V_fines/V_block       = {P_FINES_PCT:.4f}%")

# Adjusted WipFrag CDF: add the missing fines at x = 1 mm and re-normalise
# the original curve so that it still reaches 100% at the top size.
# Mathematically: for x >= 1 mm,
#   P_adj(x) = P_fines + (100 - P_fines) * P_orig(x) / 100
# At x = 1 mm, P_adj = P_fines; for x < original minimum size, P_adj = P_fines.
def _wipfrag_adjusted(p_fines_pct):
    size = np.concatenate(([1.0], wipfrag_size_mm))
    passing = np.concatenate((
        [p_fines_pct],
        p_fines_pct + (100.0 - p_fines_pct) * wipfrag_passing_pct / 100.0))
    return size, passing


wipfrag_adj_size_mm, wipfrag_adj_passing_pct = _wipfrag_adjusted(P_FINES_PCT)

# Annulus variant (crushed rock only, blasthole volume excluded):
#   V_fines,ann = 4*pi*(r_c^2 - r_0^2)*H.  Same adjustment procedure.
V_FINES_ANN = 4.0 * np.pi * (R_C**2 - _R0**2) * _H     # m3
P_FINES_ANN_PCT = 100.0 * V_FINES_ANN / V_BLOCK        # %
print("\n==== Annulus variant (blasthole excluded) ====")
print(f"V_fines,ann = 4*pi*(r_c^2-r_0^2)*H = {V_FINES_ANN:.6f} m3")
print(f"P_x,fines,ann = V_fines,ann/V_block = {P_FINES_ANN_PCT:.4f}%")
wipfrag_ann_size_mm, wipfrag_ann_passing_pct = _wipfrag_adjusted(
    P_FINES_ANN_PCT)

# ---- Esen et al. (2003) CZI crushed-zone method (parallel estimate) ----
# Independent alternative to Prof. Aubertin's r_c above; the rest of the
# correction chain (4-cylinder V_f, P_f, WipFrag adjustment) is identical,
# so the ONLY difference between the two adjusted curves is the r_c method.
#   K   = E_d / (1 + nu_d)
#   CZI = P_b^3 / (K * sigma_c^2)
#   r_c = 0.812 * r_0 * CZI^0.219
# ======================================================================
# !!! EDITABLE INPUT -- ASSUMPTION, NOT YET CONFIRMED by Prof. Aubertin !!
# The input Excel contains E = 60 GPa (static Young's modulus); whether it
# may be used as the DYNAMIC modulus E_d is unconfirmed. Edit E_D_GPA here
# only -- the original input E is not touched.
# ======================================================================
E_D_GPA = 60.0          # GPa -- PROVISIONAL assumption (see note above)
NU_D = 0.25             # dynamic Poisson ratio (confirmed by Prof. Aubertin)
P_B_MPA = P_CRUSH_MPA   # borehole pressure rho_e D^2 / 8 = 3037.5 MPa
K_ESEN_MPA = E_D_GPA * 1e3 / (1.0 + NU_D)          # MPa
CZI = P_B_MPA**3 / (K_ESEN_MPA * _C0**2)           # dimensionless
R_C_ESEN = 0.812 * _R0 * CZI**0.219                # m
V_FINES_ESEN = 4.0 * np.pi * R_C_ESEN**2 * _H      # m3
P_FINES_ESEN_PCT = 100.0 * V_FINES_ESEN / V_BLOCK  # %
print("\n==== Esen et al. (2003) CZI crushed-zone estimate ====")
print(f"E_d (ASSUMPTION, unconfirmed)   = {E_D_GPA:.1f} GPa")
print(f"nu_d                            = {NU_D:.2f}")
print(f"P_b = rho_e D^2 / 8             = {P_B_MPA:.4f} MPa")
print(f"K = E_d/(1+nu_d)                = {K_ESEN_MPA:.1f} MPa")
print(f"CZI = P_b^3/(K sigma_c^2)       = {CZI:.4f}")
print(f"r_c = 0.812 r_0 CZI^0.219       = {R_C_ESEN:.6f} m "
      f"(r_c/r_0 = {R_C_ESEN / _R0:.4f})")
print(f"V_fines = 4*pi*r_c^2*H          = {V_FINES_ESEN:.6f} m3")
print(f"P_x,fines = V_fines/V_block     = {P_FINES_ESEN_PCT:.4f}%")
wipfrag_esen_size_mm, wipfrag_esen_passing_pct = _wipfrag_adjusted(
    P_FINES_ESEN_PCT)

# Styled to match the project's blockometry cumulative-distribution look
# (report_whiteboard.py: log-x axis, dashed grid, bold labels).
fig, ax = plt.subplots(figsize=(9.5, 5.4))
plot_main_comparison(
    kco_star_result.sj_star_dist, kco_result, kco_star_result,
    measured_sizes_mm=wipfrag_size_mm,
    measured_passing_pct=wipfrag_passing_pct,
    ax=ax,
    show_class_curves=True,   # every individual class Swebrec curve P_i(x)
    show_envelope=True,       # fuseau = [min_i P_i(x), max_i P_i(x)]
)
# NOTE: classical KCO and the volume-weighted KCO* are numerically very
# close (same Xmax, similar b; the JPS=80 classes carry ~99.96 % of the
# volume weight), so the KCO line would be hidden under the thicker KCO*
# line. Restyle KCO as a dashed line on top so both remain visible without
# changing any underlying numbers.
for line in ax.get_lines():
    if line.get_label() == "Classical KCO (post-blast, predicted)":
        line.set_linestyle("--")
        line.set_linewidth(2.2)
        line.set_zorder(10)
    if line.get_label() == "WipFrag (post-blast, measured)":
        line.set_label("WipFrag original (measured)")
ax.plot(wipfrag_adj_size_mm, wipfrag_adj_passing_pct, "s--",
        color="orangered", lw=1, label="WipFrag ajusté - fines", zorder=8)
ax.set_xlabel("Fragment / block size (mm)", fontsize=11, fontweight="bold")
ax.set_ylabel("Cumulative Probability (%)", fontsize=11, fontweight="bold")
ax.set_title("Fragment / Block Size Distribution -- VARENNE\n"
             "In-situ DFN vs KCO vs KCO* (volume-weighted, with class "
             "envelope) vs WipFrag original et ajusté - fines",
             fontsize=12, fontweight="bold")
ax.set_ylim(0, 100)
ax.grid(True, which="both", linestyle="--", alpha=0.4)
ax.legend(loc="upper left", fontsize=9)
fig.tight_layout()
out_dir = os.path.join(SCRIPT_DIR, "outputs", "VARENNE", "10_kco_comparison")
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir,
                        f"VARENNE_kco_vs_kco_star_comparison{OUT_SUFFIX}.png")
fig.savefig(out_path, dpi=200, bbox_inches="tight")
print(f"\nSaved comparison plot: {out_path}")

# ---- Final (presentation) figure: same style, clean legend labels, with
# Prof. Aubertin's full-cylinder WipFrag adjustment only. Saved to a
# separate file; the figure above is preserved. The annulus variant is kept
# in the code for comparison (archive outputs only) and is NOT plotted here.
fig2, ax2 = plt.subplots(figsize=(9.5, 5.4))
plot_main_comparison(
    kco_star_result.sj_star_dist, kco_result, kco_star_result,
    measured_sizes_mm=wipfrag_size_mm,
    measured_passing_pct=wipfrag_passing_pct,
    ax=ax2,
    show_class_curves=True,
    show_envelope=True,
)
for line in ax2.get_lines():
    _lbl = line.get_label()
    if _lbl == "Classical KCO (post-blast, predicted)":
        line.set_linestyle("--")
        line.set_linewidth(2.2)
        line.set_zorder(10)
        line.set_label("Classical KCO")
    elif _lbl == "WipFrag (post-blast, measured)":
        line.set_label("WipFrag original")
    elif _lbl.startswith("In-situ DFN"):
        line.set_label("In-situ DFN")
    elif _lbl.startswith("KCO* (volume-weighted"):
        line.set_label("KCO*")
ax2.plot(wipfrag_adj_size_mm, wipfrag_adj_passing_pct, "s--",
         color="orangered", lw=1,
         label=("WipFrag adjusted \u2013 Full cylinders "
                rf"($P_{{\mathrm{{fines}}}}={P_FINES_PCT:.2f}\%$)"),
         zorder=8)
ax2.text(0.985, 0.04,
         "Crushed-zone correction:\n"
         r"$V_f=4\pi r_c^2 H$"
         "\n"
         rf"$P_f={P_FINES_PCT:.2f}\%$",
         transform=ax2.transAxes, ha="right", va="bottom", fontsize=8,
         bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                   edgecolor="0.6", alpha=0.9), zorder=20)
ax2.set_xlabel("Fragment / block size (mm)", fontsize=11, fontweight="bold")
ax2.set_ylabel("Cumulative Probability (%)", fontsize=11, fontweight="bold")
ax2.set_title("Fragment / Block Size Distribution -- VARENNE\n"
              "In-situ DFN vs KCO vs KCO* (volume-weighted, with class "
              "envelope) vs WipFrag original / adjusted",
              fontsize=12, fontweight="bold")
ax2.set_ylim(0, 100)
ax2.grid(True, which="both", linestyle="--", alpha=0.4)
ax2.legend(loc="upper left", fontsize=9)
fig2.tight_layout()
out_path2 = os.path.join(
    out_dir,
    f"VARENNE_kco_vs_kco_star_comparison{OUT_SUFFIX}_final.png")
fig2.savefig(out_path2, dpi=200, bbox_inches="tight")
print(f"Saved final comparison plot: {out_path2}")

# ---- Crushed-zone method comparison figure: Aubertin vs Esen CZI ----
# Same curves as the final figure PLUS the Esen-adjusted WipFrag curve, so
# the effect of the r_c model can be compared directly.
fig3, ax3 = plt.subplots(figsize=(9.5, 5.4))
plot_main_comparison(
    kco_star_result.sj_star_dist, kco_result, kco_star_result,
    measured_sizes_mm=wipfrag_size_mm,
    measured_passing_pct=wipfrag_passing_pct,
    ax=ax3,
    show_class_curves=True,
    show_envelope=True,
)
for line in ax3.get_lines():
    _lbl = line.get_label()
    if _lbl == "Classical KCO (post-blast, predicted)":
        line.set_linestyle("--")
        line.set_linewidth(2.2)
        line.set_zorder(10)
        line.set_label("Classical KCO")
    elif _lbl == "WipFrag (post-blast, measured)":
        line.set_label("WipFrag original")
    elif _lbl.startswith("In-situ DFN"):
        line.set_label("In-situ DFN")
    elif _lbl.startswith("KCO* (volume-weighted"):
        line.set_label("KCO*")
ax3.plot(wipfrag_adj_size_mm, wipfrag_adj_passing_pct, "s--",
         color="orangered", lw=1,
         label=("WipFrag adjusted \u2013 Aubertin "
                rf"($P_{{\mathrm{{fines}}}}={P_FINES_PCT:.2f}\%$)"),
         zorder=8)
ax3.plot(wipfrag_esen_size_mm, wipfrag_esen_passing_pct, "^--",
         color="purple", lw=1,
         label=("WipFrag adjusted \u2013 Esen CZI "
                rf"($P_{{\mathrm{{fines}}}}={P_FINES_ESEN_PCT:.2f}\%$)"),
         zorder=8)
ax3.text(0.985, 0.04,
         "Crushed-zone correction: "
         r"$V_f=4\pi r_c^2 H$" "\n"
         rf"Aubertin: $r_c={R_C:.3f}$ m, $P_f={P_FINES_PCT:.2f}\%$" "\n"
         rf"Esen CZI: $r_c={R_C_ESEN:.3f}$ m, $P_f={P_FINES_ESEN_PCT:.2f}\%$"
         "\n"
         rf"($E_d={E_D_GPA:.0f}$ GPa assumed)",
         transform=ax3.transAxes, ha="right", va="bottom", fontsize=8,
         bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                   edgecolor="0.6", alpha=0.9), zorder=20)
ax3.set_xlabel("Fragment / block size (mm)", fontsize=11, fontweight="bold")
ax3.set_ylabel("Cumulative Probability (%)", fontsize=11, fontweight="bold")
ax3.set_title("Fragment / Block Size Distribution -- VARENNE\n"
              "In-situ DFN vs KCO vs KCO* vs WipFrag original / adjusted "
              "(Aubertin vs Esen CZI crushed-zone)",
              fontsize=12, fontweight="bold")
ax3.set_ylim(0, 100)
ax3.grid(True, which="both", linestyle="--", alpha=0.4)
ax3.legend(loc="upper left", fontsize=9)
fig3.tight_layout()
out_path3 = os.path.join(
    out_dir,
    f"VARENNE_kco_vs_kco_star_comparison{OUT_SUFFIX}_czmethods.png")
fig3.savefig(out_path3, dpi=200, bbox_inches="tight")
print(f"Saved crushed-zone method comparison plot: {out_path3}")

print("\nKCO*  percentiles:", kco_star_result.percentiles())

# ---- Percentile-size comparison table (D20, D50, D80, D90) ----
# Sizes (mm) at which 20%, 50%, 80% and 90% of the material passes,
# for the in-situ (pre-blast) DFN distribution, classical KCO, and KCO*.
_percentiles = (20, 50, 80, 90)
kco_pct = kco_result.percentiles(levels=_percentiles)
kco_star_pct = kco_star_result.percentiles(levels=_percentiles)
in_situ_mm = in_situ_size_at_passing(kco_star_result.sj_star_dist,
                                      _percentiles)
_wip_d = {p: float(np.interp(p, wipfrag_passing_pct, wipfrag_size_mm))
          for p in _percentiles}


def _wipfrag_adjusted_d(p_fines_pct):
    out = {}
    for p in _percentiles:
        if p <= p_fines_pct:
            out[p] = 1.0
        else:
            _p_orig = (p - p_fines_pct) / (100.0 - p_fines_pct) * 100.0
            out[p] = float(np.interp(_p_orig, wipfrag_passing_pct,
                                     wipfrag_size_mm))
    return out


_wip_d_adj = _wipfrag_adjusted_d(P_FINES_PCT)
_wip_d_ann = _wipfrag_adjusted_d(P_FINES_ANN_PCT)
_wip_d_esen = _wipfrag_adjusted_d(P_FINES_ESEN_PCT)
_wipfrag_D_table = pd.DataFrame({
    "D": [f"D{p}" for p in _percentiles],
    "WipFrag original (mm)": [_wip_d[p] for p in _percentiles],
    "WipFrag ajusté - fines (mm)": [_wip_d_adj[p] for p in _percentiles],
})
_wipfrag_cmp_table = pd.DataFrame({
    "D": [f"D{p}" for p in _percentiles],
    "WipFrag original (mm)": [_wip_d[p] for p in _percentiles],
    f"WipFrag adjusted - full cylinders, P_fines={P_FINES_PCT:.4f}% (mm)":
        [_wip_d_adj[p] for p in _percentiles],
    f"WipFrag adjusted - annulus, P_fines={P_FINES_ANN_PCT:.4f}% (mm)":
        [_wip_d_ann[p] for p in _percentiles],
}).round(2)
# Archive only (comparison of the two crushed-zone volume definitions);
# not part of the main results.
_archive_dir = os.path.join(out_dir, "archive_annulus_comparison")
os.makedirs(_archive_dir, exist_ok=True)
print("\n[archive] WipFrag original vs full-cylinder vs annulus D20/D50/D80/D90 (mm):")
print(_wipfrag_cmp_table.to_string(index=False))
_wipfrag_cmp_csv = os.path.join(
    _archive_dir,
    f"VARENNE_wipfrag_D_original_vs_fullcyl_vs_annulus{OUT_SUFFIX}.csv")
_wipfrag_cmp_table.to_csv(_wipfrag_cmp_csv, index=False)
print(f"Saved [archive] WipFrag full-cylinder vs annulus table: {_wipfrag_cmp_csv}")
_wipfrag_D_table = _wipfrag_D_table.round(2)
print("\nWipFrag original vs ajusté - fines D20/D50/D80/D90 (mm):")
print(_wipfrag_D_table.to_string(index=False))
_wipfrag_D_csv = os.path.join(
    out_dir, f"VARENNE_wipfrag_D20_D50_D80_D90_original_vs_adjusted{OUT_SUFFIX}.csv")
_wipfrag_D_table.to_csv(_wipfrag_D_csv, index=False)
print(f"Saved WipFrag D-table: {_wipfrag_D_csv}")

D_table = pd.DataFrame({
    "D": [f"D{p}" for p in _percentiles],
    "In-situ (mm)": [float(v) for v in in_situ_mm],
    "Classical KCO (mm)": [kco_pct[f"X{p}"] for p in _percentiles],
    "KCO* (mm)": [kco_star_pct[f"X{p}_star"] for p in _percentiles],
    "WipFrag original (mm)": [_wip_d[p] for p in _percentiles],
    "WipFrag ajusté - fines (mm)": [_wip_d_adj[p] for p in _percentiles],
})
D_table = D_table.round({"In-situ (mm)": 2,
                        "Classical KCO (mm)": 2,
                        "KCO* (mm)": 2,
                        "WipFrag original (mm)": 2,
                        "WipFrag ajusté - fines (mm)": 2})
print("\nPercentile sizes D20/D50/D80/D90 (mm):")
print(D_table.to_string(index=False))
D_table_csv = os.path.join(out_dir,
                           f"VARENNE_D20_D50_D80_D90_table{OUT_SUFFIX}.csv")
D_table.to_csv(D_table_csv, index=False)
print(f"Saved D-table: {D_table_csv}")

# ---- Crushed-zone method comparison table (Aubertin vs Esen CZI) ----
# Same 4-cylinder V_f / P_f / adjustment chain for both rows; the ONLY
# difference is the r_c model.
_cz_table = pd.DataFrame({
    "Method": ["Aubertin (current)", "Esen et al. (2003) CZI"],
    "P_b (MPa)": [P_CRUSH_MPA, P_B_MPA],
    "UCS (MPa)": [_C0, _C0],
    "E_d (GPa)": ["n/a", f"{E_D_GPA:.0f} (assumed)"],
    "nu_d": ["n/a", f"{NU_D:.2f}"],
    "r_c (m)": [R_C, R_C_ESEN],
    "r_c/r_0": [R_C / _R0, R_C_ESEN / _R0],
    "V_f (m3)": [V_FINES, V_FINES_ESEN],
    "P_f (%)": [P_FINES_PCT, P_FINES_ESEN_PCT],
    "adj D20 (mm)": [_wip_d_adj[20], _wip_d_esen[20]],
    "adj D50 (mm)": [_wip_d_adj[50], _wip_d_esen[50]],
    "adj D80 (mm)": [_wip_d_adj[80], _wip_d_esen[80]],
    "adj D90 (mm)": [_wip_d_adj[90], _wip_d_esen[90]],
})
print("\nCrushed-zone method comparison (Aubertin vs Esen CZI):")
print(_cz_table.to_string(index=False))
_cz_csv = os.path.join(
    out_dir, f"VARENNE_crushed_zone_methods_comparison{OUT_SUFFIX}.csv")
_cz_table.to_csv(_cz_csv, index=False)
print(f"Saved crushed-zone method table: {_cz_csv}")

print("\n==== WipFrag curve shift: Esen CZI vs original vs Aubertin ====")
for p in _percentiles:
    print(f"D{p}: original {_wip_d[p]:.2f} -> Esen {_wip_d_esen[p]:.2f} mm "
          f"(shift {_wip_d_esen[p] - _wip_d[p]:+.2f} mm); "
          f"Aubertin {_wip_d_adj[p]:.2f} mm "
          f"(Esen vs Aubertin {_wip_d_esen[p] - _wip_d_adj[p]:+.2f} mm)")

# ---- Sj* population (real pooled S x B x H, 2303 blocks) ----
# Reuses kco_star_result.sj_star_dist and kco_star_result.classes
# (already computed above from the real pooled population); does not
# recompute or invent anything.
sj_sorted_m = np.sort(kco_star_result.sj_star_dist.sj_star_m)
n_sj = sj_sorted_m.size
jps_colors = {10: "#ff7f0e", 20: "#d62728", 50: "#1f77b4", 80: "#2ca02c"}

# ---- Table: bin, Sj_lower, Sj_upper, n_blocks, weight (selected),
# weight_count, weight_volume, volume_sum_m3, Sj_representative, JPS, JF,
# RMD, BI, A, n, X50, Xmax, b ----
table_df = pd.DataFrame(kco_star_result.class_table_rows())
table_df = table_df.round({
    "Sj_lower": 6, "Sj_upper": 6, "weight": 6, "weight_count": 6,
    "weight_volume": 6, "volume_sum_m3": 6, "Sj_representative": 4,
    "JF": 4, "RMD": 4, "BI": 4, "A": 4, "n": 4, "X50": 2, "Xmax": 2, "b": 3,
})
print("\n" + table_df.to_string(index=False))

table_csv_path = os.path.join(
    out_dir, f"VARENNE_sj_star_log_bins_table{OUT_SUFFIX}.csv")
table_df.to_csv(table_csv_path, index=False)
print(f"\nSaved Sj* log-bin table: {table_csv_path}")

# ---- Count- vs volume-weighted KCO* summary (D20/D50/D80/D90) ----
count_pct = kco_star_result_count.percentiles(levels=_percentiles)
weight_cmp = pd.DataFrame({
    "D": [f"D{p}" for p in _percentiles],
    "KCO* count-weighted (mm)": [count_pct[f"X{p}_star"] for p in _percentiles],
    "KCO* volume-weighted (mm)": [kco_star_pct[f"X{p}_star"] for p in _percentiles],
}).round(2)
print("\nKCO* count- vs volume-weighted percentile sizes (mm):")
print(weight_cmp.to_string(index=False))
weight_cmp_csv = os.path.join(
    out_dir, f"VARENNE_kco_star_weighting_comparison{OUT_SUFFIX}.csv")
weight_cmp.to_csv(weight_cmp_csv, index=False)
print(f"Saved weighting comparison: {weight_cmp_csv}")

# ---- Code-generated results summary (Markdown) for the report ----
# Every number below is read from the computed result objects of THIS run
# (no hand-typed values), so the paper/presentation can be updated from it.
_wip_d = {p: float(np.interp(p, wipfrag_passing_pct, wipfrag_size_mm))
          for p in _percentiles}
_sj_all_for_xmax = kco_star_result.sj_star_dist.sj_star_m
_p95_sj = float(np.percentile(_sj_all_for_xmax, 95))
_md = []
_md.append(f"# VARENNE KCO / KCO* results summary (run suffix `{OUT_SUFFIX}`)\n")
_md.append("Generated by `run_kco_varenne_demo.py`; all values computed in this run.\n")
_md.append("## Methodology (Xmax)\n")
_md.append(
    "Xmax is obtained from the SAME pooled blast-cell DFN population used "
    "for Sj*: 30 realizations of the S x B x H cell "
    f"(S={S_REAL} m, B={B_REAL} m, H={H_REAL} m), "
    f"n={kco_star_result.n_retained} blocks, Sj* = V^(1/3). "
    f"Xmax = min(P95(Sj*), B, S) = min({_p95_sj:.5f}, {B_REAL}, {S_REAL}) = "
    f"{kco_star_result.xmax_mm / 1000:.5f} m = {kco_star_result.xmax_mm:.2f} mm "
    f"(governed by {kco_star_result.xmax_governed_by}). "
    "The 50x50x50 m DFN is used ONLY for the P32 calibration of the fracture "
    "families; it is NOT used for Xmax or Sj*.\n")
_md.append("## Classical KCO (mean joint spacing)\n")
_md.append("| Quantity | Value |\n|---|---|")
for k, v in (("mean Sj (m)", f"{sj_classical:.4f}"),
             ("JPS", kco_result.jps), ("JPA", kco_result.jpa),
             ("JF", f"{kco_result.jf:.1f}"), ("RMD", f"{kco_result.rmd:.1f}"),
             ("RDI", f"{kco_result.rdi:.2f}"), ("HF", f"{kco_result.hf:.2f}"),
             ("BI", f"{kco_result.bi:.2f}"), ("A", f"{kco_result.rock_factor:.4f}"),
             ("q used (kg/m3)", f"{kco_result.q_used:.3f}"),
             ("q from Q/(BSH) (kg/m3)", f"{kco_result.q_calculated:.4f}"),
             ("g(n)", f"{kco_result.g_n:.2f}"), ("n", f"{kco_result.n:.6f}"),
             ("X50 (mm)", f"{kco_result.x50_mm:.4f}"),
             ("Xmax (mm)", f"{kco_result.xmax_mm:.2f}"),
             ("Xmax governed by", kco_result.xmax_governed_by),
             ("b", f"{kco_result.b:.6f}")):
    _md.append(f"| {k} | {v} |")
_md.append("\n## KCO* 12-class table (volume-weighted)\n")
_md.append("| Class | Sj* low (m) | Sj* high (m) | N_i | N_i/N (%) | sum V_i (m3) "
           "| w_i^(V) | w_i^(V) (%) | Sj*_i (m) | JPS | JPA | JF | RMD | RDI | HF "
           "| BI | A_i | n_i | X50,i (mm) | Xmax (mm) | b_i |")
_md.append("|" + "---|" * 21)
for i, c in enumerate(kco_star_result.classes, 1):
    name = ("Lower tail" if c.is_tail == "lower" else
            "Upper tail" if c.is_tail == "upper" else
            f"Bin {i - 1}")
    _md.append(
        f"| {name} | {c.bin_edge_low_m:.6f} | {c.bin_edge_high_m:.6f} | {c.n_blocks} "
        f"| {100 * c.n_blocks / kco_star_result.n_retained:.4f} | {c.volume_sum_m3:.6f} "
        f"| {c.weight_volume:.9f} | {100 * c.weight_volume:.7f} | {c.sj_representative_m:.4f} "
        f"| {c.jps} | {c.jpa} | {c.jf:.0f} | {c.rmd:.0f} | {c.rdi:.1f} | {c.hf:.1f} "
        f"| {c.bi:.1f} | {c.rock_factor:.4f} | {c.n:.6f} | {c.x50_mm:.4f} "
        f"| {c.xmax_mm:.2f} | {c.b:.6f} |")
_md.append(f"\nTotals: sum N_i = {kco_star_result.n_retained}; "
           f"sum w_i^(V) = {sum(c.weight_volume for c in kco_star_result.classes):.12f}; "
           f"sum V_j = {float(np.sum(kco_star_result.sj_star_dist.block_volume_m3)):.4f} m3.\n")
_md.append("## Distinct KCO* class curves\n")
_md.append("| Classes | JPS | X50 (mm) | Xmax (mm) | b | sum w^(V) |\n|---|---|---|---|---|---|")
for g in unique_groups:
    _md.append(f"| C{', C'.join(str(k) for k in g['class_indices'])} | {g['jps']} "
               f"| {g['x50_mm']:.4f} | {g['xmax_mm']:.2f} | {g['b']:.6f} | {g['weight']:.7f} |")
_md.append("\n## Percentile sizes (mm)\n")
_md.append("| D | In-situ DFN | Classical KCO | KCO* volume-weighted "
           "| KCO* count-weighted (reference) | WipFrag original | WipFrag ajusté - fines "
           "| WipFrag adjusted - Esen CZI |\n|---|---|---|---|---|---|---|---|")
for p, ins in zip(_percentiles, in_situ_mm):
    _md.append(f"| D{p} | {float(ins):.2f} | {kco_pct[f'X{p}']:.2f} "
               f"| {kco_star_pct[f'X{p}_star']:.2f} | {count_pct[f'X{p}_star']:.2f} "
               f"| {_wip_d[p]:.2f} | {_wip_d_adj[p]:.2f} | {_wip_d_esen[p]:.2f} |")
_md.append("\nEnvelope: P_min(x) = min_i P_i(x), P_max(x) = max_i P_i(x) over the 12 "
           "class curves (min-max class envelope, not a confidence interval).\n")
_md.append("## WipFrag ajusté - fines (crushed-zone missing fines)\n")
_md.append("| Quantity | Value |\n|---|---|")
for k, v in (("r_0 = d/2 (m)", f"{_R0:.3f}"),
             ("rho_e (kg/m3)", f"{_RHO_E:.0f}"),
             ("D (m/s)", f"{_D_EXP:.0f}"),
             ("P = rho_e D^2 / 8 (MPa)", f"{P_CRUSH_MPA:.2f}"),
             ("C_0 (MPa)", f"{_C0:.0f}"),
             ("C_0,dyn = 3 C_0 (MPa)", f"{_C0_DYN:.0f}"),
             ("r_c = r_0 sqrt(P / C_0,dyn) (m)", f"{R_C:.5f}"),
             ("V_fines = 4 pi r_c^2 H (m3)", f"{V_FINES:.4f}"),
             ("V_block = S B H (m3)", f"{V_BLOCK:.3f}"),
             ("P_fines = V_fines / V_block (%)", f"{P_FINES_PCT:.4f}")):
    _md.append(f"| {k} | {v} |")
_md.append(
    "\nAdjusted CDF: P_adj(1 mm) = P_fines; for each original point x, "
    "P_adj(x) = P_fines + (100 - P_fines) P_orig(x) / 100. The original "
    "WipFrag data are unchanged.\n\n"
    "Note: V_fines = 4 pi r_c^2 H (full cylinders) is Prof. Aubertin's "
    "explicit simplified estimate and is used here. The crushed-rock annulus "
    "excluding the blasthole volume, 4 pi (r_c^2 - r_0^2) H = "
    f"{4.0 * np.pi * (R_C**2 - _R0**2) * _H:.4f} m3 "
    f"(P_fines = {100.0 * 4.0 * np.pi * (R_C**2 - _R0**2) * _H / V_BLOCK:.4f} %), "
    "could be considered in a more refined analysis.\n")
_md.append("Final presentation figure: "
           f"`VARENNE_kco_vs_kco_star_comparison{OUT_SUFFIX}_final.png` "
           "(In-situ DFN, Classical KCO, KCO*, WipFrag original, WipFrag "
           "adjusted - full cylinders). Annulus comparison archived in "
           "`archive_annulus_comparison/`.\n")
_md.append("## WipFrag adjusted - Esen et al. (2003) CZI crushed-zone\n")
_md.append(
    "Parallel crushed-zone estimate: same 4-cylinder V_f / P_f / WipFrag "
    "adjustment chain as Prof. Aubertin's method; the ONLY difference is "
    "the r_c model. "
    f"**E_d = {E_D_GPA:.0f} GPa is a PROVISIONAL ASSUMPTION** (input Excel "
    "E = 60 GPa, static; not yet confirmed as the dynamic modulus by "
    "Prof. Aubertin).\n")
_md.append("| Quantity | Value |\n|---|---|")
for k, v in (("E_d (GPa) -- ASSUMPTION", f"{E_D_GPA:.1f}"),
             ("nu_d", f"{NU_D:.2f}"),
             ("P_b = rho_e D^2 / 8 (MPa)", f"{P_B_MPA:.2f}"),
             ("K = E_d/(1+nu_d) (MPa)", f"{K_ESEN_MPA:.1f}"),
             ("sigma_c = UCS (MPa)", f"{_C0:.0f}"),
             ("CZI = P_b^3/(K sigma_c^2)", f"{CZI:.4f}"),
             ("r_c = 0.812 r_0 CZI^0.219 (m)", f"{R_C_ESEN:.5f}"),
             ("r_c/r_0", f"{R_C_ESEN / _R0:.4f}"),
             ("V_fines = 4 pi r_c^2 H (m3)", f"{V_FINES_ESEN:.4f}"),
             ("P_fines = V_fines/V_block (%)", f"{P_FINES_ESEN_PCT:.4f}")):
    _md.append(f"| {k} | {v} |")
_md.append("\n### Crushed-zone method comparison\n")
_md.append("| Method | P_b (MPa) | UCS (MPa) | E_d (GPa) | nu_d | r_c (m) "
           "| r_c/r_0 | V_f (m3) | P_f (%) | adj D20 | adj D50 | adj D80 "
           "| adj D90 |\n|---|---|---|---|---|---|---|---|---|---|---|---|")
for _, r in _cz_table.iterrows():
    _md.append("| " + " | ".join(str(v) for v in r) + " |")
_md.append(
    "\nMethod-comparison figure: "
    f"`VARENNE_kco_vs_kco_star_comparison{OUT_SUFFIX}_czmethods.png` "
    "(adds WipFrag adjusted - Esen CZI to the final-figure curves).\n")
summary_md_path = os.path.join(
    out_dir, f"VARENNE_kco_results_summary{OUT_SUFFIX}.md")
with open(summary_md_path, "w", encoding="utf-8") as fh:
    fh.write("\n".join(_md))
print(f"Saved results summary: {summary_md_path}")

# ---- Percentage-frequency histogram helper ----
# Y-axis = percentage of blocks in each bin: pct_bin = 100 * N_bin / N_total.
# No density/bin-width normalization is applied (unlike matplotlib's
# hist(..., density=True), which divides by the LINEAR bin width and would
# blow up for the smallest-Sj* log-spaced bins as width -> 0). By
# construction sum(pct_bin over all bins) == 100. No blocks are filtered or
# removed anywhere below; all n_sj = 2303 blocks are used in every bin.
def _log_bin_percent_hist(values_m, bins_edges):
    counts, edges = np.histogram(values_m, bins=bins_edges)
    assert counts.sum() == values_m.size, "no blocks may be dropped by binning"
    pct_per_bin = 100.0 * counts / values_m.size
    return edges, counts, pct_per_bin


def _draw_class_lines(ax, classes, y_top, label_prefix, label_fmt,
                      jps_labels=False, mark_p1_p99=False):
    is_log_bins = classes[0].bin_edge_low_m is not None
    if is_log_bins:
        edges_m = sorted(set([c.bin_edge_low_m for c in classes]
                             + [classes[-1].bin_edge_high_m]))
    else:
        edges_m = sorted(set([c.sj_min_m for c in classes]
                             + [classes[-1].sj_max_m]))

    # P1/P99 (if present) are drawn separately as thick blue lines below;
    # skip duplicate gray dotted lines at the exact same locations so the
    # 10 central-bin boundaries stand alone.
    lower_tail = next((c for c in classes if c.is_tail == "lower"), None)
    upper_tail = next((c for c in classes if c.is_tail == "upper"), None)
    p1_m = lower_tail.bin_edge_high_m if lower_tail is not None else None
    p99_m = upper_tail.bin_edge_low_m if upper_tail is not None else None
    skip_edges = {v for v in (p1_m, p99_m) if v is not None} if mark_p1_p99 else set()
    for edge in edges_m:
        if any(abs(edge - s) < 1e-12 for s in skip_edges):
            continue
        ax.axvline(edge, color="gray", lw=0.9, linestyle=":", zorder=3)

    # 12 representative Sj_i* as dashed lines; no per-line text labels
    # (avoids overlapping rotated text when representatives are close
    # together) -- tails vs central bins are distinguished by color and
    # identified explicitly in the legend below.
    for c in classes:
        line_color = "#8B008B" if c.is_tail else "#B22222"
        ax.axvline(c.sj_representative_m, color=line_color, lw=1.3,
                  linestyle="--", zorder=4)

    boundary_label = ("Central bin boundary (equal-width in ln Sj*)"
                      if is_log_bins else
                      "Percentile class boundary (P0..P100)")
    proxies = [
        Line2D([0], [0], color="gray", lw=0.9, linestyle=":",
              label=boundary_label),
        Line2D([0], [0], color="#B22222", lw=1.3, linestyle="--",
              label=f"{label_prefix} representative Sj* (central bin)"),
    ]
    if lower_tail is not None or upper_tail is not None:
        proxies.append(Line2D([0], [0], color="#8B008B", lw=1.3,
                             linestyle="--",
                             label=f"{label_prefix} representative Sj* "
                                   "(lower/upper tail)"))
    if mark_p1_p99:
        for p_val, p_label in ((p1_m, f"P{TAIL_LOW_PCT:g}"),
                               (p99_m, f"P{TAIL_HIGH_PCT:g}")):
            if p_val is None:
                continue
            ax.axvline(p_val, color="#1f4e8c", lw=2.5, zorder=6)
            ax.annotate(f"{p_label}={p_val:.4g} m", xy=(p_val, y_top),
                       xytext=(4, -2), textcoords="offset points",
                       fontsize=9, fontweight="bold", color="#1f4e8c",
                       rotation=90, va="top", ha="left", zorder=7)
        proxies.append(Line2D([0], [0], color="#1f4e8c", lw=2.5,
                             label="P1 / P99 (active binning-range edges)"))
    return proxies


bin_edges = np.geomspace(sj_sorted_m.min(), sj_sorted_m.max(), 60)
edges, counts, pct_per_bin = _log_bin_percent_hist(sj_sorted_m, bin_edges)
assert abs(pct_per_bin.sum() - 100.0) < 1e-9, (
    "histogram percentage-frequencies must sum to 100%"
)
kde = gaussian_kde(np.log(sj_sorted_m))
x_kde = np.geomspace(sj_sorted_m.min(), sj_sorted_m.max(), 400)
density_kde_lnx = kde(np.log(x_kde))  # density w.r.t. ln(Sj*), no /x
# Scale the smooth curve to the same percentage-of-population units as the
# bars: density_lnx * d(ln x) * 100 approximates the % of blocks in a bin
# of that ln-width (bin_edges is geomspace -> constant ln-width per bin).
dlnx_bin = np.diff(np.log(bin_edges))[0]
pct_kde = density_kde_lnx * dlnx_bin * 100.0

# ---- Fig 3: percentage-frequency histogram, JPS-colored classes ----
fig3, ax3 = plt.subplots(figsize=(9, 5.5))
ax3.bar(edges[:-1], pct_per_bin, width=np.diff(edges), align="edge",
       color="#888888", edgecolor="white", alpha=0.8,
       label=f"Sj* frequency (% of blocks, n={n_sj})")
ax3.plot(x_kde, pct_kde, color="black", lw=1.5,
        label="Sj* frequency (KDE, scaled to % per bin)")
for c in kco_star_result.classes:
    if c.is_tail:
        continue  # no shading for the tail classes; kept clean per figure spec
    span_lo = c.bin_edge_low_m if c.bin_edge_low_m is not None else c.sj_min_m
    span_hi = c.bin_edge_high_m if c.bin_edge_high_m is not None else c.sj_max_m
    color = jps_colors.get(c.jps, "gray")
    ax3.axvspan(span_lo, span_hi, alpha=0.12, color=color)
ymax3 = ax3.get_ylim()[1]
proxies3 = _draw_class_lines(ax3, kco_star_result.classes, ymax3,
                             "JPS-colored", "", jps_labels=True,
                             mark_p1_p99=True)
ax3.set_xscale("log")
ax3.set_xlabel("Sj* = V^(1/3) (m)", fontsize=11, fontweight="bold")
ax3.set_ylabel("Percentage of blocks (%)", fontsize=11,
              fontweight="bold")
_method_desc = {
    "log_bins": f"{kco_star_result.n_log_bins} equal-width log bins",
    "log_bins_p1_p99_tails": (
        f"{kco_star_result.n_log_bins} equal-width log bins in "
        f"[P{kco_star_result.tail_low_pct:g}, P{kco_star_result.tail_high_pct:g}] "
        "+ 2 tail classes"),
    "percentile": "percentile classes",
}.get(kco_star_result.class_method, kco_star_result.class_method)
ax3.set_title(
    f"Sj* Non-Cumulative Distribution with {_method_desc} -- VARENNE\n"
    "(real pooled S x B x H DFN, 30 realizations, percentage frequency, "
    f"n={n_sj} blocks)", fontsize=11, fontweight="bold")
ax3.grid(True, which="both", linestyle="--", alpha=0.3)
h3, l3 = ax3.get_legend_handles_labels()
ax3.legend(h3 + proxies3, l3 + [p.get_label() for p in proxies3],
          loc="upper left", fontsize=7.5)
fig3.tight_layout()

sj_hist_path = os.path.join(
    out_dir, f"VARENNE_sj_star_histogram_JPS{OUT_SUFFIX}.png")
fig3.savefig(sj_hist_path, dpi=200, bbox_inches="tight")
print(f"\nSaved Sj* non-cumulative (histogram/density) plot: {sj_hist_path}")

# ---- Fig 3b (comparison only): VOLUME-based version of Fig 3 ----
# Y-axis = percentage of TOTAL BLOCK VOLUME in each bin:
#   pct_vol_bin = 100 * sum_{j in bin} V_j / sum_j V_j.
# Identical Sj* data, 60 geometric histogram bins, P1/P99, 10 central log
# classes, 2 tails, class boundaries and representative Sj*_i as Fig 3;
# only the bar height statistic changes (volume share instead of block
# count share). Purely descriptive; it is NOT used by KCO/KCO*.
_sj_all = kco_star_result.sj_star_dist.sj_star_m
_vol_all = kco_star_result.sj_star_dist.block_volume_m3
vol_per_bin, _ = np.histogram(_sj_all, bins=bin_edges, weights=_vol_all)
assert abs(vol_per_bin.sum() - _vol_all.sum()) < 1e-6 * _vol_all.sum(), (
    "no block volume may be dropped by binning"
)
pct_vol_per_bin = 100.0 * vol_per_bin / _vol_all.sum()
assert abs(pct_vol_per_bin.sum() - 100.0) < 1e-9, (
    "volume percentages must sum to 100%"
)
kde_vol = gaussian_kde(np.log(_sj_all), weights=_vol_all)
pct_kde_vol = kde_vol(np.log(x_kde)) * dlnx_bin * 100.0

fig3b, ax3b = plt.subplots(figsize=(9, 5.5))
ax3b.bar(edges[:-1], pct_vol_per_bin, width=np.diff(edges), align="edge",
        color="#888888", edgecolor="white", alpha=0.8,
        label=f"Sj* volume share (% of total block volume, "
              f"sum V={_vol_all.sum():.1f} m3, n={n_sj})")
ax3b.plot(x_kde, pct_kde_vol, color="black", lw=1.5,
         label="Sj* volume share (volume-weighted KDE, scaled to % per bin)")
for c in kco_star_result.classes:
    if c.is_tail:
        continue
    span_lo = c.bin_edge_low_m if c.bin_edge_low_m is not None else c.sj_min_m
    span_hi = c.bin_edge_high_m if c.bin_edge_high_m is not None else c.sj_max_m
    ax3b.axvspan(span_lo, span_hi, alpha=0.12,
                 color=jps_colors.get(c.jps, "gray"))
ymax3b = ax3b.get_ylim()[1]
proxies3b = _draw_class_lines(ax3b, kco_star_result.classes, ymax3b,
                              "JPS-colored", "", jps_labels=True,
                              mark_p1_p99=True)
ax3b.set_xscale("log")
ax3b.set_xlabel("Sj* = V^(1/3) (m)", fontsize=11, fontweight="bold")
ax3b.set_ylabel("Percentage of total block volume (%)", fontsize=11,
               fontweight="bold")
ax3b.set_title(
    f"Sj* Non-Cumulative Distribution with {_method_desc} -- VARENNE\n"
    "(real pooled S x B x H DFN, 30 realizations, VOLUME share per bin, "
    f"n={n_sj} blocks) -- comparison only", fontsize=11, fontweight="bold")
ax3b.grid(True, which="both", linestyle="--", alpha=0.3)
h3b, l3b = ax3b.get_legend_handles_labels()
ax3b.legend(h3b + proxies3b, l3b + [p.get_label() for p in proxies3b],
           loc="upper left", fontsize=7.5)
fig3b.tight_layout()
sj_hist_vol_path = os.path.join(
    out_dir, f"VARENNE_sj_star_histogram_JPS_volume_share{OUT_SUFFIX}.png")
fig3b.savefig(sj_hist_vol_path, dpi=200, bbox_inches="tight")
print(f"Saved Sj* VOLUME-share histogram (comparison only): {sj_hist_vol_path}")

# ============================================================
# EXCEL EXPORT: underlying numerical data for every figure
# ============================================================
# All arrays below are the exact data used to draw the PNG figures
# (no digitization from images). Each sheet contains editable Excel
# charts linked directly to the cells; a CSV copy of every sheet is
# written to excel_export_csv/.
xlsx_path = os.path.join(
    out_dir, f"VARENNE_kco_figure_data{OUT_SUFFIX}.xlsx")
csv_dir = os.path.join(out_dir, "excel_export_csv")
os.makedirs(csv_dir, exist_ok=True)

wb = xlsxwriter.Workbook(xlsx_path, {"nan_inf_to_errors": True})
fmt_hdr = wb.add_format({"bold": True, "bg_color": "#D9E1F2", "border": 1})


def _write_col(ws, col, header, values):
    """Write a header + data column; NaN/None become blank cells."""
    ws.write(0, col, header, fmt_hdr)
    for r, v in enumerate(values, start=1):
        if v is None:
            ws.write_blank(r, col, None)
            continue
        if isinstance(v, np.generic):
            v = v.item()
        if isinstance(v, float) and not np.isfinite(v):
            ws.write_blank(r, col, None)
        else:
            ws.write(r, col, v)


def _sheet_csv(name, columns):
    """CSV copy of a sheet: columns may have different lengths (padded)."""
    maxlen = max(len(list(v)) for _, v in columns)
    data = {}
    for h, v in columns:
        vals = list(v)
        data[h] = vals + [np.nan] * (maxlen - len(vals))
    pd.DataFrame(data).to_csv(os.path.join(csv_dir, f"{name}.csv"),
                              index=False)


_VIRIDIS12 = ["#440154", "#482173", "#433E85", "#38598C", "#2D708E",
              "#25858E", "#1E9B8A", "#2AB07F", "#52C569", "#86D549",
              "#C2DF23", "#FDE725"]


def _logx_scatter(title, x_name, y_name, y_max=None,
                  x_min=None, x_max=None, hide_x_labels=False,
                  y_at_xmin=False):
    ch = wb.add_chart({"type": "scatter", "subtype": "straight"})
    xax = {"name": x_name, "log_base": 10, "num_font": {"size": 9},
           "major_gridlines": {"visible": True,
                               "line": {"color": "#D9D9D9", "width": 0.5}},
           "minor_gridlines": {"visible": True,
                               "line": {"color": "#F2F2F2", "width": 0.5}}}
    if x_min is not None:
        xax["min"] = x_min
    if x_max is not None:
        xax["max"] = x_max
    if hide_x_labels:
        xax["label_position"] = "none"
    if y_at_xmin:
        # pin the y-axis to the left edge: vertical axis crosses at x min
        # (xlsxwriter writes this into the y-axis element's c:crosses)
        xax["crossing"] = "min"
    ch.set_x_axis(xax)
    yax = {"name": y_name, "num_font": {"size": 9},
           "major_gridlines": {"visible": True,
                               "line": {"color": "#D9D9D9", "width": 0.5}},
           "minor_gridlines": {"visible": True,
                               "line": {"color": "#F2F2F2", "width": 0.5}}}
    if y_max is not None:
        yax.update({"min": 0, "max": y_max})
    ch.set_y_axis(yax)
    ch.set_title({"name": title, "name_font": {"size": 11, "bold": True}})
    ch.set_size({"width": 760, "height": 430})
    ch.set_legend({"font": {"size": 9}})
    return ch


def _write_seg_col(ws, col, tag, segments):
    """Write a gapped vertical-segment series (x,y cols, blank-separated)."""
    xs, ys = [], []
    for x, y0, y1 in segments:
        xs += [x, x, None]
        ys += [y0, y1, None]
    _write_col(ws, col, f"{tag} x", xs)
    _write_col(ws, col + 1, f"{tag} y", ys)
    return len(xs)


_SUP = {"-": "\u207b", "0": "\u2070", "1": "\u00b9", "2": "\u00b2",
        "3": "\u00b3", "4": "\u2074", "5": "\u2075", "6": "\u2076",
        "7": "\u2077", "8": "\u2078", "9": "\u2079"}


def _add_pow10_labels(ws, ch, sheet_name, col, powers):
    """Invisible y=0 series whose data labels show '10^n' at each decade,
    replacing the numeric log-axis tick labels (Excel cannot format axis
    labels as powers of ten). Returns the series index (for legend delete)."""
    xs = [10.0 ** p for p in powers]
    labels = ["10" + "".join(_SUP[d] for d in str(p)) for p in powers]
    _write_col(ws, col, "x-axis label x (mm)", xs)
    _write_col(ws, col + 1, "x-axis label y", [0.0] * len(xs))
    _write_col(ws, col + 2, "x-axis label text", labels)
    ch.add_series({
        "name": "_xlabels",
        "categories": [sheet_name, 1, col, len(xs), col],
        "values": [sheet_name, 1, col + 1, len(xs), col + 1],
        "line": {"none": True},
        "marker": {"type": "none"},
        "data_labels": {"custom": [{"value": t} for t in labels],
                        "position": "below",
                        "font": {"size": 9}},
    })


# ---- Sheet 1: In-situ_DFN_CDF (empirical CDF, 2303 blocks) ----
x_insitu_mm, p_insitu = in_situ_passing_from_sj_star(
    kco_star_result.sj_star_dist)
ws = wb.add_worksheet("In-situ_DFN_CDF")
_write_col(ws, 0, "Block size (mm)", x_insitu_mm)
_write_col(ws, 1, "Cumulative passing (%)", p_insitu)
n_is = len(x_insitu_mm)
# Helper columns replicating the reference PNG: D20/D50/D80/D90 markers,
# grey dotted droplines, red labels (same np.percentile convention as
# plot_reference_blocksize_curve.py).
_D_REFS = (20, 50, 80, 90)
_sizes_mm = np.sort(kco_star_result.sj_star_dist.sj_star_m) * 1000.0
_d_vals = {p: float(np.percentile(_sizes_mm, p)) for p in _D_REFS}
_xmin_ax = float(_sizes_mm.min() * 0.9)
_xmax_ax = float(_sizes_mm.max() * 1.1)
_drop_x, _drop_y = [], []
for p in _D_REFS:
    _drop_x += [_d_vals[p], _d_vals[p], _xmin_ax, None]
    _drop_y += [0.0, float(p), float(p), None]
_write_col(ws, 3, "D-ref dropline x (mm)", _drop_x)
_write_col(ws, 4, "D-ref dropline y (%)", _drop_y)
_write_col(ws, 6, "D-ref x (mm)", [_d_vals[p] for p in _D_REFS])
_write_col(ws, 7, "D-ref y (%)", [float(p) for p in _D_REFS])
_write_col(ws, 8, "D-ref label", [f"{_d_vals[p]:,.0f} mm" for p in _D_REFS])
ch = _logx_scatter(
    "Courbe de distribution r\u00e9f\u00e9rence \u2014 Varenne\n"
    f"({n_is} blocs, 30 r\u00e9alisations DFN S\u00d7B\u00d7H)",
    "Taille des blocs (mm)", "Passant cumul\u00e9 (%)",
    x_min=1.0, x_max=_xmax_ax, hide_x_labels=True)
ch.set_y_axis({"name": "Passant cumul\u00e9 (%)", "min": 0, "max": 102,
               "major_unit": 10, "num_font": {"size": 9},
               "major_gridlines": {"visible": True,
                                   "line": {"color": "#D9D9D9",
                                            "width": 0.5}},
               "minor_gridlines": {"visible": True,
                                   "line": {"color": "#F2F2F2",
                                            "width": 0.5}}})
ch.show_blanks_as("gap")
ch.add_series({
    "name": "In-situ blocks (pooled S\u00d7B\u00d7H DFN)",
    "categories": ["In-situ_DFN_CDF", 1, 0, n_is, 0],
    "values": ["In-situ_DFN_CDF", 1, 1, n_is, 1],
    "line": {"color": "#1f4e79", "width": 2.2},
})
ch.add_series({
    "name": "_droplines",
    "categories": ["In-situ_DFN_CDF", 1, 3, len(_drop_x), 3],
    "values": ["In-situ_DFN_CDF", 1, 4, len(_drop_y), 4],
    "line": {"color": "#808080", "width": 0.75,
             "dash_type": "round_dot"},
})
ch.add_series({
    "name": "_dref",
    "categories": ["In-situ_DFN_CDF", 1, 6, len(_D_REFS), 6],
    "values": ["In-situ_DFN_CDF", 1, 7, len(_D_REFS), 7],
    "line": {"none": True},
    "marker": {"type": "circle", "size": 6,
               "fill": {"color": "#c00000"},
               "line": {"color": "#c00000"}},
    "data_labels": {"custom": [{"value": f"{_d_vals[p]:,.0f} mm"}
                               for p in _D_REFS],
                    "position": "left",
                    "font": {"color": "#c00000", "size": 8}},
})
_add_pow10_labels(ws, ch, "In-situ_DFN_CDF", 10, [0, 1, 2, 3])
ch.set_legend({"font": {"size": 9}, "position": "top", "overlay": True,
               "delete_series": [1, 2, 3]})
ws.insert_chart("N2", ch)
_sheet_csv("In-situ_DFN_CDF", [
    ("Block size (mm)", x_insitu_mm),
    ("Cumulative passing (%)", p_insitu)])

# ---- Sheet 2: Sj_Distribution (histogram + KDE + class table) ----
ws = wb.add_worksheet("Sj_Distribution")
bin_labels = [f"{edges[i]:.4g} - {edges[i + 1]:.4g}"
              for i in range(len(counts))]
_write_col(ws, 0, "Bin lower edge Sj* (m)", edges[:-1])
_write_col(ws, 1, "Bin upper edge Sj* (m)", edges[1:])
_write_col(ws, 2, "Block count", counts)
_write_col(ws, 3, "Block frequency (%)", pct_per_bin)
_write_col(ws, 4, "Bin label (m)", bin_labels)
_write_col(ws, 6, "KDE x Sj* (m)", x_kde)
_write_col(ws, 7, "KDE density w.r.t. ln(Sj*)", density_kde_lnx)
_write_col(ws, 8, "KDE scaled frequency (% per bin)", pct_kde)
# KCO* class table (same classes as the histogram class lines)
cls_headers = ["Class", "Bin lower (m)", "Bin upper (m)", "n_blocks",
               "weight_count", "weight_volume", "Sj* representative (m)",
               "JPS", "JF", "RMD", "A", "X50 (mm)", "is_tail"]
for j, h in enumerate(cls_headers):
    ws.write(0, 10 + j, h, fmt_hdr)
for i, c in enumerate(kco_star_result.classes, start=1):
    row = [i, c.bin_edge_low_m, c.bin_edge_high_m, c.n_blocks,
           c.weight_count, c.weight_volume, c.sj_representative_m,
           c.jps, c.jf, c.rmd, c.rock_factor, c.x50_mm, c.is_tail or ""]
    for j, v in enumerate(row):
        if v is None:
            ws.write(i, 10 + j, "")
        elif isinstance(v, np.generic):
            ws.write(i, 10 + j, v.item())
        else:
            ws.write(i, 10 + j, v)
# P1 / P99 boundaries
_lower_tail = next((c for c in kco_star_result.classes
                    if c.is_tail == "lower"), None)
_upper_tail = next((c for c in kco_star_result.classes
                    if c.is_tail == "upper"), None)
ws.write(0, 24, f"P{TAIL_LOW_PCT:g} boundary (m)", fmt_hdr)
ws.write(1, 24, float(_lower_tail.bin_edge_high_m))
ws.write(0, 25, f"P{TAIL_HIGH_PCT:g} boundary (m)", fmt_hdr)
ws.write(1, 25, float(_upper_tail.bin_edge_low_m))
# ---- Composite chart replicating the PNG histogram ----
# Excel cannot draw log-spaced filled bars or axvspan shading; they are
# reproduced with thick vertical-line series (bars) and dense thin
# vertical-line series (JPS class spans) on a log-x scatter chart.
_ymax_hist = 12.0
_hc = 30  # first helper column (hidden below)
_jps_span_fill = {10: "#FBE2C5", 20: "#F6D0D0", 50: "#D0DBF0",
                  80: "#D5EBD5"}
_span_cols = {}
for jps in sorted({c.jps for c in kco_star_result.classes}):
    _segs = []
    for c in kco_star_result.classes:
        if c.jps != jps:
            continue
        _lo = (c.bin_edge_low_m if c.bin_edge_low_m is not None
               else c.sj_min_m)
        _hi = (c.bin_edge_high_m if c.bin_edge_high_m is not None
               else c.sj_max_m)
        for _xx in np.geomspace(_lo, _hi, 15):
            _segs.append((float(_xx), 0.0, _ymax_hist))
    _span_cols[jps] = (_hc, _write_seg_col(ws, _hc, f"span_jps{jps}", _segs))
    _hc += 2
_p1_m = float(_lower_tail.bin_edge_high_m)
_p99_m = float(_upper_tail.bin_edge_low_m)
_edges_all = sorted(
    {float(c.bin_edge_low_m) for c in kco_star_result.classes
     if c.bin_edge_low_m is not None}
    | {float(kco_star_result.classes[-1].bin_edge_high_m)})
_bnd_col = _hc
_bnd_n = _write_seg_col(
    ws, _hc, "bnd",
    [(e, 0.0, _ymax_hist) for e in _edges_all
     if abs(e - _p1_m) > 1e-12 and abs(e - _p99_m) > 1e-12])
_hc += 2
_crep_col = _hc
_crep_n = _write_seg_col(
    ws, _hc, "crep",
    [(float(c.sj_representative_m), 0.0, _ymax_hist)
     for c in kco_star_result.classes if not c.is_tail])
_hc += 2
_trep_col = _hc
_trep_n = _write_seg_col(
    ws, _hc, "trep",
    [(float(c.sj_representative_m), 0.0, _ymax_hist)
     for c in kco_star_result.classes if c.is_tail])
_hc += 2
_p1p99_col = _hc
_p1p99_n = _write_seg_col(
    ws, _hc, "p1p99",
    [(_p1_m, 0.0, _ymax_hist), (_p99_m, 0.0, _ymax_hist)])
_hc += 2
# histogram bars: real rectangles = horizontal-stripe fill + outline.
# Excel scatter lines have round caps and no fill, so each bar is drawn
# at 92% of its log-bin width -> rectangular bars with visible side gaps.
_bar_fill_x, _bar_fill_y = [], []
_bar_edge_x, _bar_edge_y = [], []
_dy = 0.1  # stripe spacing (%) < stripe line width -> solid fill
for i in range(len(counts)):
    _r = (edges[i + 1] / edges[i]) ** 0.04
    _xl, _xr = float(edges[i] * _r), float(edges[i + 1] / _r)
    _h = float(pct_per_bin[i])
    _yy = _dy / 2
    while _yy < _h:
        _bar_fill_x += [_xl, _xr, None]
        _bar_fill_y += [_yy, _yy, None]
        _yy += _dy
    _bar_edge_x += [_xl, _xl, _xr, _xr, _xl, None]
    _bar_edge_y += [0.0, _h, _h, 0.0, 0.0, None]
_bar_fill_col = _hc
_write_col(ws, _hc, "bar fill x", _bar_fill_x)
_write_col(ws, _hc + 1, "bar fill y", _bar_fill_y)
_hc += 2
_bar_edge_col = _hc
_write_col(ws, _hc, "bar edge x", _bar_edge_x)
_write_col(ws, _hc + 1, "bar edge y", _bar_edge_y)
_hc += 2
# P1 / P99 value labels (dummy points at the top of each line)
_p1p99_lbl_col = _hc
_write_col(ws, _hc, "P1P99 label x", [_p1_m, _p99_m])
_write_col(ws, _hc + 1, "P1P99 label y", [11.2, 11.2])
_hc += 2
ws.set_column(30, _hc - 1, None, None, {"hidden": True})

ch = _logx_scatter(
    f"Sj* Non-Cumulative Distribution -- VARENNE (n={n_sj} blocks)",
    "Sj* = V^(1/3) (m)", "Percentage of blocks (%)", y_max=_ymax_hist,
    x_min=1e-4, x_max=10.0, hide_x_labels=True, y_at_xmin=True)
ch.show_blanks_as("gap")
ch.show_hidden_data()
_del_series = []
_si = 0
for jps in sorted(_span_cols):
    _c0, _nn = _span_cols[jps]
    ch.add_series({
        "name": f"_span{jps}",
        "categories": ["Sj_Distribution", 1, _c0, _nn, _c0],
        "values": ["Sj_Distribution", 1, _c0 + 1, _nn, _c0 + 1],
        "line": {"color": _jps_span_fill[jps], "width": 0.75},
    })
    _del_series.append(_si)
    _si += 1
ch.add_series({
    "name": f"Sj* frequency (% of blocks, n={n_sj})",
    "categories": ["Sj_Distribution", 1, _bar_fill_col,
                   len(_bar_fill_x), _bar_fill_col],
    "values": ["Sj_Distribution", 1, _bar_fill_col + 1,
               len(_bar_fill_y), _bar_fill_col + 1],
    "line": {"color": "#909090", "width": 2.5},
})
_si += 1
ch.add_series({
    "name": "_baredge",
    "categories": ["Sj_Distribution", 1, _bar_edge_col,
                   len(_bar_edge_x), _bar_edge_col],
    "values": ["Sj_Distribution", 1, _bar_edge_col + 1,
               len(_bar_edge_y), _bar_edge_col + 1],
    "line": {"color": "#666666", "width": 0.9},
})
_del_series.append(_si)
_si += 1
ch.add_series({
    "name": "Central bin boundary (equal-width in ln Sj*)",
    "categories": ["Sj_Distribution", 1, _bnd_col, _bnd_n, _bnd_col],
    "values": ["Sj_Distribution", 1, _bnd_col + 1, _bnd_n, _bnd_col + 1],
    "line": {"color": "#808080", "width": 0.75,
             "dash_type": "round_dot"},
})
_si += 1
ch.add_series({
    "name": "Sj* frequency (KDE, scaled to % per bin)",
    "categories": ["Sj_Distribution", 1, 6, len(x_kde), 6],
    "values": ["Sj_Distribution", 1, 8, len(x_kde), 8],
    "line": {"color": "black", "width": 1.5},
})
_si += 1
ch.add_series({
    "name": "JPS-colored representative Sj* (central bin)",
    "categories": ["Sj_Distribution", 1, _crep_col, _crep_n, _crep_col],
    "values": ["Sj_Distribution", 1, _crep_col + 1, _crep_n, _crep_col + 1],
    "line": {"color": "#B22222", "width": 1.3, "dash_type": "dash"},
})
_si += 1
ch.add_series({
    "name": "JPS-colored representative Sj* (lower/upper tail)",
    "categories": ["Sj_Distribution", 1, _trep_col, _trep_n, _trep_col],
    "values": ["Sj_Distribution", 1, _trep_col + 1, _trep_n, _trep_col + 1],
    "line": {"color": "#8B008B", "width": 1.3, "dash_type": "dash"},
})
_si += 1
ch.add_series({
    "name": "P1 / P99 (active binning-range edges)",
    "categories": ["Sj_Distribution", 1, _p1p99_col, _p1p99_n, _p1p99_col],
    "values": ["Sj_Distribution", 1, _p1p99_col + 1, _p1p99_n,
               _p1p99_col + 1],
    "line": {"color": "#1f4e8c", "width": 2.5},
})
_si += 1
ch.add_series({
    "name": "_p1p99lbl",
    "categories": ["Sj_Distribution", 1, _p1p99_lbl_col, 2,
                   _p1p99_lbl_col],
    "values": ["Sj_Distribution", 1, _p1p99_lbl_col + 1, 2,
               _p1p99_lbl_col + 1],
    "line": {"none": True},
    "marker": {"type": "none"},
    "data_labels": {"custom": [
                        {"value": f"P1={_p1_m:.5g} m"},
                        {"value": f"P99={_p99_m:.4g} m"}],
                    "position": "above",
                    "font": {"size": 8, "color": "#1f4e8c",
                             "bold": True}},
})
_del_series.append(_si)
_si += 1
_add_pow10_labels(ws, ch, "Sj_Distribution", 26,
                  [-4, -3, -2, -1, 0, 1])
_del_series.append(_si)
ch.set_legend({"font": {"size": 8}, "overlay": True,
               "layout": {"x": 0.03, "y": 0.04,
                          "width": 0.38, "height": 0.30},
               "delete_series": _del_series})
ws.insert_chart("A63", ch)
_sheet_csv("Sj_Distribution", [
    ("Bin lower edge Sj* (m)", edges[:-1]),
    ("Bin upper edge Sj* (m)", edges[1:]),
    ("Block count", counts),
    ("Block frequency (%)", pct_per_bin),
    ("Bin label (m)", bin_labels),
    ("KDE x Sj* (m)", x_kde),
    ("KDE density w.r.t. ln(Sj*)", density_kde_lnx),
    ("KDE scaled frequency (% per bin)", pct_kde),
])

# ---- Sheet 3: KCO_Comparison_Curves (common log-x grid, 500 pts) ----
x_grid = np.geomspace(1.0, kco_star_result.xmax_mm, 500)
p_insitu_g = np.interp(x_grid, x_insitu_mm, p_insitu)
p_kco_g = np.asarray(kco_result.passing(x_grid), dtype=float)
p_star_g = np.asarray(kco_star_result.passing(x_grid), dtype=float)
p_env_min, p_env_max = kco_star_result.envelope(x_grid)
p_wip_o = np.interp(x_grid, wipfrag_size_mm, wipfrag_passing_pct)
p_wip_o[(x_grid < wipfrag_size_mm.min())
        | (x_grid > wipfrag_size_mm.max())] = np.nan
p_wip_a = np.interp(x_grid, wipfrag_adj_size_mm, wipfrag_adj_passing_pct)
p_wip_a[(x_grid < wipfrag_adj_size_mm.min())
        | (x_grid > wipfrag_adj_size_mm.max())] = np.nan
p_wip_e = np.interp(x_grid, wipfrag_esen_size_mm,
                    wipfrag_esen_passing_pct)
p_wip_e[(x_grid < wipfrag_esen_size_mm.min())
        | (x_grid > wipfrag_esen_size_mm.max())] = np.nan
ws = wb.add_worksheet("KCO_Comparison_Curves")
cmp_cols = [
    ("x (mm)", x_grid),
    ("In-situ DFN (%)", p_insitu_g),
    ("Classical KCO (%)", p_kco_g),
    ("KCO* (%)", p_star_g),
    ("KCO* class-envelope min (%)", p_env_min),
    ("KCO* class-envelope max (%)", p_env_max),
    ("WipFrag original (%)", p_wip_o),
    ("WipFrag adjusted - full cylinders (%)", p_wip_a),
    ("WipFrag adjusted - Esen CZI (%)", p_wip_e),
]
for j, (h, v) in enumerate(cmp_cols):
    _write_col(ws, j, h, v)
# Helper columns: envelope band (dense vertical segments, hidden) and the
# raw WipFrag measured points (markers, like the PNG).
_env_x, _env_y = [], []
for _xi, _lo, _hi in zip(x_grid, p_env_min, p_env_max):
    _env_x += [float(_xi), float(_xi), None]
    _env_y += [float(_lo), float(_hi), None]
_write_col(ws, 9, "envelope fill x (mm)", _env_x)
_write_col(ws, 10, "envelope fill y (%)", _env_y)
_write_col(ws, 12, "WipFrag original x (mm)", wipfrag_size_mm)
_write_col(ws, 13, "WipFrag original y (%)", wipfrag_passing_pct)
_write_col(ws, 15, "WipFrag adjusted x (mm)", wipfrag_adj_size_mm)
_write_col(ws, 16, "WipFrag adjusted y (%)", wipfrag_adj_passing_pct)
_write_col(ws, 45, "WipFrag adj Esen x (mm)", wipfrag_esen_size_mm)
_write_col(ws, 46, "WipFrag adj Esen y (%)", wipfrag_esen_passing_pct)
ws.set_column(9, 10, None, None, {"hidden": True})
ch = _logx_scatter(
    "Fragment / Block Size Distribution -- VARENNE",
    "Fragment / block size (mm)", "Cumulative Probability (%)",
    y_max=100, x_min=1, hide_x_labels=True)
ch.show_blanks_as("gap")
ch.show_hidden_data()
ch.add_series({
    "name": "In-situ DFN",
    "categories": ["KCO_Comparison_Curves", 1, 0, len(x_grid), 0],
    "values": ["KCO_Comparison_Curves", 1, 1, len(x_grid), 1],
    "line": {"color": "black", "width": 1.5},
})
ch.add_series({
    "name": "KCO* class envelope (min-max of P_i)",
    "categories": ["KCO_Comparison_Curves", 1, 9, len(_env_x), 9],
    "values": ["KCO_Comparison_Curves", 1, 10, len(_env_y), 10],
    "line": {"color": "#CFE0F1", "width": 0.75},
})
# Class curves: viridis colour + cycling dash + staggered white-face
# markers (markevery emulation) so coincident classes stay identifiable,
# exactly like the PNG. Marker coords go to hidden helper columns.
_groups = kco_star_result.unique_curves()
_group_of = {}
for g in _groups:
    for _rank, _k in enumerate(g["class_indices"]):
        _group_of[_k] = (_rank, len(g["class_indices"]))
_mstep = max(len(x_grid) // 10, 1)
_DASH4 = ["solid", "dash", "dash_dot", "round_dot"]
_MKT = ["circle", "square", "triangle", "diamond", "x", "star",
        "plus", "dot", "dash", "circle", "square", "triangle"]
_cls_labels = []
for i, c in enumerate(kco_star_result.classes, start=1):
    _col = _VIRIDIS12[(i - 1) % len(_VIRIDIS12)]
    ch.add_series({
        "name": (f"C{i}: Sj*={c.sj_representative_m:.3f} m, JPS={c.jps}, "
                 f"X50={c.x50_mm:.0f} mm"),
        "categories": ["KCOstar_Classes", 1, 0, len(x_grid), 0],
        "values": ["KCOstar_Classes", 1, i, len(x_grid), i],
        "line": {"color": _col, "width": 0.9,
                 "dash_type": _DASH4[(i - 1) % len(_DASH4)]},
    })
    _rank, _size = _group_of[i]
    _off = int(round(_rank * _mstep / max(_size, 1)))
    _mx = x_grid[_off::_mstep]
    _my = np.asarray(c.passing(_mx), dtype=float)
    _mc = 21 + 2 * (i - 1)
    _write_col(ws, _mc, f"C{i} marker x (mm)", _mx)
    _write_col(ws, _mc + 1, f"C{i} marker y (%)", _my)
    ch.add_series({
        "name": f"_Cm{i}",
        "categories": ["KCO_Comparison_Curves", 1, _mc, len(_mx), _mc],
        "values": ["KCO_Comparison_Curves", 1, _mc + 1, len(_my), _mc + 1],
        "line": {"none": True},
        "marker": {"type": _MKT[(i - 1) % len(_MKT)], "size": 5,
                   "fill": {"color": "white"},
                   "line": {"color": _col, "width": 0.9}},
    })
    _tail = f" [{c.is_tail} tail]" if c.is_tail else ""
    _cls_labels.append(
        f"C{i:>2}{_tail}: Sj*={c.sj_representative_m:.3f} m, "
        f"JPS={c.jps}, X50={c.x50_mm:.0f} mm, w={c.weight:.4f}")
ws.set_column(21, 44, None, None, {"hidden": True})
ch.add_series({
    "name": "Classical KCO",
    "categories": ["KCO_Comparison_Curves", 1, 0, len(x_grid), 0],
    "values": ["KCO_Comparison_Curves", 1, 2, len(x_grid), 2],
    "line": {"color": "#00B050", "width": 2.5, "dash_type": "dash"},
})
ch.add_series({
    "name": "KCO*",
    "categories": ["KCO_Comparison_Curves", 1, 0, len(x_grid), 0],
    "values": ["KCO_Comparison_Curves", 1, 3, len(x_grid), 3],
    "line": {"color": "blue", "width": 2.5},
})
ch.add_series({
    "name": "WipFrag original",
    "categories": ["KCO_Comparison_Curves", 1, 12,
                   len(wipfrag_size_mm), 12],
    "values": ["KCO_Comparison_Curves", 1, 13,
               len(wipfrag_passing_pct), 13],
    "line": {"color": "red", "width": 1.0, "dash_type": "dash"},
    "marker": {"type": "circle", "size": 5,
               "fill": {"color": "red"}, "line": {"color": "red"}},
})
ch.add_series({
    "name": "WipFrag adjusted \u2013 Full cylinders (P_fines = 1.98%)",
    "categories": ["KCO_Comparison_Curves", 1, 15,
                   len(wipfrag_adj_size_mm), 15],
    "values": ["KCO_Comparison_Curves", 1, 16,
               len(wipfrag_adj_passing_pct), 16],
    "line": {"color": "#FF4500", "width": 1.0, "dash_type": "dash"},
    "marker": {"type": "square", "size": 5,
               "fill": {"color": "#FF4500"},
               "line": {"color": "#FF4500"}},
})
ch.add_series({
    "name": "WipFrag adjusted \u2013 Esen CZI",
    "categories": ["KCO_Comparison_Curves", 1, 45,
                   len(wipfrag_esen_size_mm), 45],
    "values": ["KCO_Comparison_Curves", 1, 46,
               len(wipfrag_esen_passing_pct), 46],
    "line": {"color": "#7030A0", "width": 1.0, "dash_type": "dash"},
    "marker": {"type": "triangle", "size": 5,
               "fill": {"color": "#7030A0"},
               "line": {"color": "#7030A0"}},
})
_add_pow10_labels(ws, ch, "KCO_Comparison_Curves", 18, [0, 1, 2, 3])
ch.set_legend({"font": {"size": 8}, "overlay": True,
               "layout": {"x": 0.03, "y": 0.04,
                          "width": 0.42, "height": 0.32},
               "delete_series": list(range(2, 26)) + [31]})
ws.insert_chart("V2", ch)
ws.insert_textbox(12, 21,
                  "Crushed-zone correction:\n"
                  "V_f = 4\u03c0r_c\u00b2H\n"
                  f"Aubertin: r_c={R_C:.3f} m, P_f={P_FINES_PCT:.2f}%\n"
                  f"Esen CZI: r_c={R_C_ESEN:.3f} m, "
                  f"P_f={P_FINES_ESEN_PCT:.2f}%\n"
                  f"(E_d={E_D_GPA:.0f} GPa assumed)",
                  {"width": 190, "height": 72, "x_offset": 80,
                   "font": {"size": 8},
                   "fill": {"color": "#FFFFFF"},
                   "line": {"color": "#999999"}})
# Figure-level class legend (right of the chart), like the PNG.
_ovl = ["C" + ", C".join(str(k) for k in g["class_indices"])
        + f" coincide exactly (JPS={g['jps']})"
        for g in _groups if len(g["class_indices"]) > 1]
_cls_txt = (f"Individual KCO* class curves P_i(x): "
            f"{len(kco_star_result.classes)} classes computed, "
            f"{len(_groups)} distinct\n" + "\n".join(_ovl)
            + "\n\n" + "\n".join(_cls_labels))
ws.insert_textbox(1, 33, _cls_txt,
                  {"width": 360, "height": 430,
                   "font": {"size": 7, "name": "Consolas"},
                   "fill": {"color": "#FFFFFF"},
                   "line": {"color": "#999999"}})
_sheet_csv("KCO_Comparison_Curves", cmp_cols)

# ---- Sheet 4: KCOstar_Classes (per-class P_i(x) + parameters) ----
ws = wb.add_worksheet("KCOstar_Classes")
n_cls = len(kco_star_result.classes)
_write_col(ws, 0, "x (mm)", x_grid)
for i, c in enumerate(kco_star_result.classes, start=1):
    _write_col(ws, i, f"P_C{i} (%)",
               np.asarray(c.passing(x_grid), dtype=float))
# class parameter table to the right (col 14 = O)
par_headers = ["Class", "Sj* lower (m)", "Sj* upper (m)", "n_blocks",
               "weight_count", "weight_volume", "weight_used",
               "Sj* representative (m)", "JPS", "JF", "RMD", "BI", "A",
               "n", "X50 (mm)", "Xmax (mm)", "b", "is_tail"]
for j, h in enumerate(par_headers):
    ws.write(0, 14 + j, h, fmt_hdr)
for i, c in enumerate(kco_star_result.classes, start=1):
    row = [i, c.bin_edge_low_m, c.bin_edge_high_m, c.n_blocks,
           c.weight_count, c.weight_volume, c.weight,
           c.sj_representative_m, c.jps, c.jf, c.rmd, c.bi,
           c.rock_factor, c.n, c.x50_mm, c.xmax_mm, c.b,
           c.is_tail or ""]
    for j, v in enumerate(row):
        if v is None:
            ws.write(i, 14 + j, "")
        elif isinstance(v, np.generic):
            ws.write(i, 14 + j, v.item())
        else:
            ws.write(i, 14 + j, v)
ch = _logx_scatter(
    "Individual KCO* class curves P_i(x) -- VARENNE",
    "Fragment / block size (mm)", "Cumulative passing (%)", y_max=100,
    x_min=1, hide_x_labels=True)
for i, c in enumerate(kco_star_result.classes, start=1):
    ch.add_series({
        "name": f"C{i}: Sj*={c.sj_representative_m:.3f} m, JPS={c.jps}, "
                f"X50={c.x50_mm:.0f} mm",
        "categories": ["KCOstar_Classes", 1, 0, len(x_grid), 0],
        "values": ["KCOstar_Classes", 1, i, len(x_grid), i],
        "line": {"width": 1.0,
                 "color": _VIRIDIS12[(i - 1) % len(_VIRIDIS12)]},
    })
_add_pow10_labels(ws, ch, "KCOstar_Classes", 33, [0, 1, 2, 3])
ch.set_legend({"font": {"size": 9}, "delete_series": [12]})
ws.insert_chart("A505", ch)
_sheet_csv("KCOstar_Classes",
           [("x (mm)", x_grid)]
           + [(f"P_C{i} (%)", np.asarray(c.passing(x_grid), dtype=float))
              for i, c in enumerate(kco_star_result.classes, start=1)]
           + [(f"param_{h}", list(r))
              for h, r in zip(par_headers, zip(*[
                  [i, c.bin_edge_low_m, c.bin_edge_high_m, c.n_blocks,
                   c.weight_count, c.weight_volume, c.weight,
                   c.sj_representative_m, c.jps, c.jf, c.rmd, c.bi,
                   c.rock_factor, c.n, c.x50_mm, c.xmax_mm, c.b,
                   c.is_tail or ""]
                  for i, c in enumerate(kco_star_result.classes, start=1)]))])

wb.close()
print(f"\nSaved Excel figure-data workbook: {xlsx_path}")
print(f"Saved per-sheet CSV copies in: {csv_dir}")

print("\n" + "=" * 70)
print("REMINDER: s_anfo_pct, drill_accuracy_sd_m and jpa_case are ESTIMATES,")
print("not confirmed Varenne measurements. Re-run once confirmed.")
print("=" * 70)

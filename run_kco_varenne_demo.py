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
#       outputs/VARENNE/06_blockometry_plots/block_shape_data_VARENNE.csv
#       (the existing, calibrated 50x50x50 m DFN -- NOT boundary-limited
#       by a small S x B x H cell, appropriate for an Xmax/large-block
#       estimate). xmax_block_percentile=95 (per your professor's
#       95-or-99 guidance), xmax_block_size_method="equivalent_cube".
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
#     - rock_density_kg_m3    = 2700.0
#     - ucs_mpa                = 100.0
#     - youngs_modulus_gpa     = 60.0
#     - rock_mass_case         = "jointed" (reasonably justified: this
#                                 whole project characterises Varenne as
#                                 a jointed rock mass via DFN)
#     - jpa_case               = "strike_perpendicular_to_face"
#     - timing_scatter_factor_ns = 1.0 (Cunningham 2005 neutral baseline)
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

from run_site import SCRIPT_DIR
from kco_model import BlastDesign, predict_kco
from kco_star_model import (predict_kco_star, plot_main_comparison,
                            in_situ_size_at_passing)
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
    name="VARENNE (real geometry + ESTIMATED rock/explosive properties)",
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
    rock_density_kg_m3=2700.0,               # ESTIMATE
    ucs_mpa=100.0,                           # ESTIMATE
    youngs_modulus_gpa=60.0,                 # ESTIMATE
    rock_mass_case="jointed",
    jpa_case="strike_perpendicular_to_face", # ESTIMATE
    timing_scatter_factor_ns=1.0,
    shift_factor_mode="no_shift",
)

print("=" * 70)
print("INPUT PROVENANCE -- see run_kco_varenne_demo.py header for full list")
print("=" * 70)
print("REAL: B=3.4 m, S=4.1 m, H=14.0 m, D=114 mm, Q=165 kg, q=0.8 kg/m3")
print("ESTIMATED (unconfirmed): drill_accuracy_sd_m, s_anfo_pct, "
     "rock_density_kg_m3, ucs_mpa, youngs_modulus_gpa, jpa_case")
print("Observed but NOT usable as a curve: 3 oversize boulders ~1 m each")
print()

# ---- Classical KCO: real mean joint spacing + real large-domain Xmax ----
sj_classical = _load_mean_joint_spacing_varenne()
if sj_classical is None:
    raise RuntimeError("outputs/SPACING/raw_spacing_values.xlsx not found; "
                       "cannot get a real mean_joint_spacing_m for classical KCO")
print(f"Classical KCO mean_joint_spacing_m (real, from outputs/SPACING): "
     f"{sj_classical:.4f} m")

large_domain_vols = _load_block_volumes_varenne()
if large_domain_vols is None:
    raise RuntimeError(
        "outputs/VARENNE/06_blockometry_plots/block_shape_data_VARENNE.csv "
        "not found; cannot get a real Xmax block-volume population"
    )
print(f"Large-domain (50x50x50 m) block volumes loaded (real): "
     f"{large_domain_vols.size} blocks")

design_classical = BlastDesign(
    **{**design.__dict__,
      "mean_joint_spacing_m": sj_classical,
      "block_size_method": "equivalent_cube",
      "block_statistic": "p95",
    }
)
kco_result = predict_kco(design_classical, block_volumes_m3=large_domain_vols)
print("\n" + kco_result.audit_table())

# ---- KCO*: REAL S x B x H DFN block volumes (generated here) ----
print(f"\nGenerating REAL S x B x H DFN blast-cell realizations "
     f"(B={B_REAL}, S={S_REAL}, H={H_REAL} m, 30 seeds, PROVISIONAL count "
     "-- convergence not yet re-validated at these exact dimensions)...")
blastcell = generate_blast_cell_realizations(
    burden_m=B_REAL, spacing_m=S_REAL, bench_height_m=H_REAL,
    seeds=list(range(3001, 3031)),
    site_key="VARENNE",
)
sj_star_vols = blastcell["pooled_volumes_m3"]
print(f"Pooled S x B x H block volumes (real, 30 realizations): "
     f"{sj_star_vols.size} blocks")
print(f"Saved: {blastcell['pooled_csv_path']}")

N_LOG_BINS = 10  # configurable; default 10 per the log-bins methodology
TAIL_LOW_PCT = 1.0   # P1
TAIL_HIGH_PCT = 99.0  # P99
kco_star_result = predict_kco_star(
    design, sj_star_vols, large_domain_vols,
    xmax_block_size_method="equivalent_cube",
    xmax_block_percentile=95,
    class_method="log_bins_p1_p99_tails",
    n_log_bins=N_LOG_BINS,
    tail_low_pct=TAIL_LOW_PCT,
    tail_high_pct=TAIL_HIGH_PCT,
    percentile_edges=[0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
    representative_method="median",
)
print("\n" + kco_star_result.audit_table())

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
         f"-> n={c.n_blocks:>4d}  w={c.weight:.4f}")
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
# Styled to match the project's blockometry cumulative-distribution look
# (report_whiteboard.py: log-x axis, dashed grid, bold labels).
fig, ax = plt.subplots(figsize=(8, 4.8))
plot_main_comparison(
    kco_star_result.sj_star_dist, kco_result, kco_star_result,
    measured_sizes_mm=wipfrag_size_mm,
    measured_passing_pct=wipfrag_passing_pct,
    ax=ax,
)
# NOTE: classical KCO (X50=215.6mm) and KCO* (X50*=209.6mm) are numerically
# very close here (same Xmax, similar b), so the thin KCO line is visually
# hidden under the thicker KCO* line. Restyle KCO (drawn first, so it is
# ax.lines[1] after in-situ) as a dashed line, on top, so both remain
# visible without changing any underlying numbers.
for line in ax.get_lines():
    if line.get_label() == "Classical KCO (post-blast, predicted)":
        line.set_linestyle("--")
        line.set_linewidth(3.0)
        line.set_zorder(10)
    elif line.get_label() == "KCO* (post-blast, predicted)":
        line.set_zorder(5)
ax.set_xlabel("Fragment / block size (mm)", fontsize=11, fontweight="bold")
ax.set_ylabel("Cumulative Probability (%)", fontsize=11, fontweight="bold")
ax.set_title("Fragment / Block Size Distribution -- VARENNE\n"
             "In-situ DFN vs KCO vs KCO* vs WipFrag", fontsize=12,
             fontweight="bold")
ax.set_ylim(0, 100)
ax.grid(True, which="both", linestyle="--", alpha=0.4)
ax.legend(loc="lower right", fontsize=9)
fig.tight_layout()
out_dir = os.path.join(SCRIPT_DIR, "outputs", "VARENNE", "10_kco_comparison")
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, "VARENNE_kco_vs_kco_star_comparison.png")
fig.savefig(out_path, dpi=200, bbox_inches="tight")
print(f"\nSaved comparison plot: {out_path}")

print("\nKCO*  percentiles:", kco_star_result.percentiles())

# ---- Percentile-size comparison table (D20, D50, D80, D90) ----
# Sizes (mm) at which 20%, 50%, 80% and 90% of the material passes,
# for the in-situ (pre-blast) DFN distribution, classical KCO, and KCO*.
_percentiles = (20, 50, 80, 90)
kco_pct = kco_result.percentiles(levels=_percentiles)
kco_star_pct = kco_star_result.percentiles(levels=_percentiles)
in_situ_mm = in_situ_size_at_passing(kco_star_result.sj_star_dist,
                                      _percentiles)
D_table = pd.DataFrame({
    "D": [f"D{p}" for p in _percentiles],
    "In-situ (mm)": [float(v) for v in in_situ_mm],
    "Classical KCO (mm)": [kco_pct[f"X{p}"] for p in _percentiles],
    "KCO* (mm)": [kco_star_pct[f"X{p}_star"] for p in _percentiles],
})
D_table = D_table.round({"In-situ (mm)": 2,
                        "Classical KCO (mm)": 2,
                        "KCO* (mm)": 2})
print("\nPercentile sizes D20/D50/D80/D90 (mm):")
print(D_table.to_string(index=False))
D_table_csv = os.path.join(out_dir, "VARENNE_D20_D50_D80_D90_table.csv")
D_table.to_csv(D_table_csv, index=False)
print(f"Saved D-table: {D_table_csv}")

# ---- Sj* population (real pooled S x B x H, 2303 blocks) ----
# Reuses kco_star_result.sj_star_dist and kco_star_result.classes
# (already computed above from the real pooled population); does not
# recompute or invent anything.
sj_sorted_m = np.sort(kco_star_result.sj_star_dist.sj_star_m)
n_sj = sj_sorted_m.size
jps_colors = {20: "#d62728", 50: "#1f77b4", 80: "#2ca02c"}

# ---- Table: bin, Sj_lower, Sj_upper, n_blocks, weight, Sj_representative,
# JPS, JF, RMD, BI, A, n, X50, Xmax, b ----
table_df = pd.DataFrame(kco_star_result.class_table_rows())
table_df = table_df.round({
    "Sj_lower": 6, "Sj_upper": 6, "weight": 4, "Sj_representative": 4,
    "JF": 4, "RMD": 4, "BI": 4, "A": 4, "n": 4, "X50": 2, "Xmax": 2, "b": 3,
})
print("\n" + table_df.to_string(index=False))

table_csv_path = os.path.join(out_dir, "VARENNE_sj_star_log_bins_table.csv")
table_df.to_csv(table_csv_path, index=False)
print(f"\nSaved Sj* log-bin table: {table_csv_path}")

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

sj_hist_path = os.path.join(out_dir, "VARENNE_sj_star_histogram_JPS.png")
fig3.savefig(sj_hist_path, dpi=200, bbox_inches="tight")
print(f"\nSaved Sj* non-cumulative (histogram/density) plot: {sj_hist_path}")

print("\n" + "=" * 70)
print("REMINDER: rock/explosive properties above are ESTIMATES, not")
print("confirmed Varenne measurements. Re-run once confirmed.")
print("=" * 70)

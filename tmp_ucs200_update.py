"""UCS=200 MPa update for the Varenne KCO/KCO* diagnostic.

Does not modify the model.  Writes:
outputs/VARENNE/10_kco_comparison/VARENNE_kco_diagnostic_UCS200.md
"""
import os
import numpy as np
import pandas as pd
from run_site import SCRIPT_DIR
from kco_model import BlastDesign, predict_kco
from kco_star_model import predict_kco_star
from main import _load_mean_joint_spacing_varenne, _load_block_volumes_varenne

sj = _load_mean_joint_spacing_varenne()
large_domain_vols = _load_block_volumes_varenne()

# Load previously generated SxBxH blast-cell pooled volumes for KCO*
pool_path = os.path.join(SCRIPT_DIR, "outputs", "VARENNE", "08_blastcell_SxBxH",
                         "block_volumes_blastcell_pooled.csv")
sj_star_vols = pd.read_csv(pool_path)["volume"].dropna().values.astype(float)
sj_star_vols = sj_star_vols[np.isfinite(sj_star_vols) & (sj_star_vols > 0.0)]


def make_design(ucs_mpa: float):
    return BlastDesign(
        name="VARENNE UCS=200 diagnostic",
        hole_diameter_mm=114.0,
        burden_m=3.4,
        spacing_m=4.1,
        bench_height_m=14.0,
        subdrill_m=0.0,
        total_charge_m=14.0,
        bottom_charge_m=0.0,
        column_charge_m=0.0,
        drill_accuracy_sd_m=0.3,
        charge_per_hole_kg=165.0,
        powder_factor_reported_kg_m3=0.8,
        powder_factor_mode="reported",
        s_anfo_pct=100.0,
        explosive_name="Dyno Nobel XL900 emulsion",
        rock_density_kg_m3=2700.0,
        ucs_mpa=ucs_mpa,
        youngs_modulus_gpa=60.0,
        rock_mass_case="jointed",
        jpa_case="strike_perpendicular_to_face",
        timing_scatter_factor_ns=1.0,
        shift_factor_mode="no_shift",
        mean_joint_spacing_m=sj,
        block_size_method="equivalent_cube",
        block_statistic="p95",
    )


def kco_d(kr):
    p = kr.percentiles(levels=(20, 50, 80, 90))
    return {int(k.replace('X', '').replace('_star', '')): v for k, v in p.items()}


def kco_star_d(ksr):
    p = ksr.percentiles(levels=(20, 50, 80, 90))
    return {int(k.replace('X', '').replace('_star', '')): v for k, v in p.items()}


# WipFrag reference
wip_pct = np.array([1, 5, 10, 15, 20, 25, 40, 50, 60, 70, 80, 90, 99, 100], dtype=float)
wip_size = np.array([20.49, 219.93, 267.78, 315.64, 344.78, 373.77,
                     460.76, 540.75, 622.05, 729.86, 907.66,
                     1645.53, 2855.55, 2990.0], dtype=float)
wip_d = {t: float(np.exp(np.interp(t, wip_pct, np.log(wip_size)))) for t in (20, 50, 80, 90)}


def run_case(ucs_mpa: float, lines: list):
    d = make_design(ucs_mpa)
    kr = predict_kco(d, large_domain_vols)
    ksr = predict_kco_star(
        d, sj_star_vols, large_domain_vols,
        xmax_block_size_method="equivalent_cube",
        xmax_block_percentile=95,
        class_method="log_bins_p1_p99_tails",
        n_log_bins=10,
        tail_low_pct=1.0,
        tail_high_pct=99.0,
    )
    k = kco_d(kr)
    ks = kco_star_d(ksr)

    lines.append(f"## UCS = {ucs_mpa} MPa")
    lines.append("")
    lines.append("### Rock-factor and KCO parameters")
    lines.append(f"- JPS  = {kr.jps}")
    lines.append(f"- JPA  = {kr.jpa}")
    lines.append(f"- JF   = {kr.jf}")
    lines.append(f"- RMD  = {kr.rmd:.3f}")
    lines.append(f"- RDI  = {kr.rdi:.4f}")
    lines.append(f"- HF   = {kr.hf:.4f}  (UCS/5 because E={d.youngs_modulus_gpa} GPa >= 50)")
    lines.append(f"- BI   = {kr.bi:.4f}")
    lines.append(f"- A    = {kr.rock_factor:.4f}")
    lines.append(f"- n    = {kr.n:.4f}")
    lines.append(f"- g(n) = {kr.g_n:.4f}")
    lines.append(f"- X50 (classical) = {kr.x50_mm:.2f} mm")
    lines.append(f"- Xmax = {kr.xmax_mm:.2f} mm")
    lines.append(f"- b    = {kr.b:.4f}")
    lines.append("")
    lines.append("### D20/D50/D80/D90 (mm)")
    lines.append("| Curve | D20 | D50 | D80 | D90 |")
    lines.append("|---|---|---|---|---|")
    lines.append(f"| Classical KCO | {k[20]:.2f} | {k[50]:.2f} | {k[80]:.2f} | {k[90]:.2f} |")
    lines.append(f"| KCO* | {ks[20]:.2f} | {ks[50]:.2f} | {ks[80]:.2f} | {ks[90]:.2f} |")
    lines.append(f"| WipFrag 2026-07-23 | {wip_d[20]:.2f} | {wip_d[50]:.2f} | {wip_d[80]:.2f} | {wip_d[90]:.2f} |")
    lines.append("")
    lines.append("### Errors vs WipFrag (%)")
    lines.append("| Curve | D20 | D50 | D80 | D90 |")
    lines.append("|---|---|---|---|---|")
    for name, dic in [("Classical KCO", k), ("KCO*", ks)]:
        err = {t: 100.0 * (dic[t] - wip_d[t]) / wip_d[t] for t in (20, 50, 80, 90)}
        lines.append(f"| {name} | {err[20]:.1f} | {err[50]:.1f} | {err[80]:.1f} | {err[90]:.1f} |")
    lines.append("")
    return kr, ksr


lines = []
lines.append("# Varenne KCO/KCO* Diagnostic — Updated with Professor-Provided Intact-Rock Properties")
lines.append("")
lines.append("## Corrected parameter provenance")
lines.append("")
lines.append("| Parameter | Value | Unit | Source |")
lines.append("|---|---|---|---|")
lines.append("| Burden B | 3.4 | m | user-provided (real) |")
lines.append("| Spacing S | 4.1 | m | user-provided (real) |")
lines.append("| Bench height H | 14.0 | m | user-provided (real) |")
lines.append("| Hole diameter | 114.0 | mm | user-provided (real) |")
lines.append("| Charge per hole Q | 165.0 | kg | user-provided (real) |")
lines.append("| Powder factor (reported) | 0.8 | kg/m3 | user-provided (real) |")
lines.append("| Explosive | Dyno Nobel XL900 emulsion | - | user-provided (real) |")
lines.append("| s_ANFO | 100.0 | % | **placeholder** (ANFO-equivalent, pending product RWS) |")
lines.append("| **Rock density rho** | **2700.0** | kg/m3 | **professor-provided intact-rock value** |")
lines.append("| **UCS** | **200.0** | MPa | **professor-provided intact-rock value** |")
lines.append("| **Young's modulus E** | **60.0** | GPa | **professor-provided intact-rock value** |")
lines.append("| Rock mass case | jointed | - | justified by Varenne DFN (4 joint families) |")
lines.append("| JPA case | strike_perpendicular_to_face | - | justified by structural analysis |")
lines.append("| Drill accuracy W | 0.3 | m | **assumed/placeholder** |")
lines.append("| Timing scatter factor | 1.0 | - | **assumed/placeholder** |")
lines.append("| Mean joint spacing | measured | m | derived from DFN/SPACING |")
lines.append("")

lines.append("## Comparison: UCS = 100 MPa (previous) vs 200 MPa")
lines.append("")
lines.append("### UCS = 100 MPa (previous reference)")
lines.append("")
run_case(100.0, lines)

lines.append("### UCS = 200 MPa (professor-corrected)")
lines.append("")
run_case(200.0, lines)

lines.append("## Summary")
lines.append("")
lines.append("Correcting UCS from 100 MPa to 200 MPa (with E=60 GPa and rho=2700 kg/m3 unchanged) raises HF from 20.0 to 40.0, which raises A from about 7.05 to about 8.25.  This coarsens the KCO D50 from ~215 mm to ~252 mm and the KCO* D50 from ~209 mm to ~245 mm.  The WipFrag gap is reduced, but the predicted curve is still much finer than the measured one (D50 error remains around −53%).")

out_path = os.path.join(SCRIPT_DIR, "outputs", "VARENNE", "10_kco_comparison",
                        "VARENNE_kco_diagnostic_UCS200.md")
with open(out_path, "w", encoding="utf-8") as f:
    f.write("\n".join(lines))
print(f"Saved UCS=200 diagnostic: {out_path}")

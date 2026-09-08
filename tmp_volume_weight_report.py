"""Count- vs volume-weighted KCO* comparison (Varenne, UCS=200 MPa).

Read-only: uses the already pooled S x B x H block volumes and writes
nothing under outputs/.
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
pool_path = os.path.join(SCRIPT_DIR, "outputs", "VARENNE", "08_blastcell_SxBxH",
                         "block_volumes_blastcell_pooled.csv")
sj_star_vols = pd.read_csv(pool_path)["volume"].dropna().values.astype(float)
sj_star_vols = sj_star_vols[np.isfinite(sj_star_vols) & (sj_star_vols > 0.0)]

design = BlastDesign(
    name="VARENNE (corrected: UCS=200 MPa)",
    hole_diameter_mm=114.0, burden_m=3.4, spacing_m=4.1, bench_height_m=14.0,
    subdrill_m=0.0, total_charge_m=14.0, bottom_charge_m=0.0, column_charge_m=0.0,
    drill_accuracy_sd_m=0.3, charge_per_hole_kg=165.0,
    powder_factor_reported_kg_m3=0.8, powder_factor_mode="reported",
    s_anfo_pct=100.0, explosive_name="Dyno Nobel XL900 emulsion",
    rock_density_kg_m3=2700.0, ucs_mpa=200.0, youngs_modulus_gpa=60.0,
    rock_mass_case="jointed", jpa_case="strike_perpendicular_to_face",
    timing_scatter_factor_ns=1.0, shift_factor_mode="no_shift",
    mean_joint_spacing_m=sj, block_size_method="equivalent_cube",
    block_statistic="p95",
)

kco = predict_kco(design, large_domain_vols)
common = dict(xmax_block_size_method="equivalent_cube", xmax_block_percentile=95,
              class_method="log_bins_p1_p99_tails", n_log_bins=10,
              tail_low_pct=1.0, tail_high_pct=99.0, representative_method="median")
ks_count = predict_kco_star(design, sj_star_vols, large_domain_vols,
                            weight_method="count", **common)
ks_vol = predict_kco_star(design, sj_star_vols, large_domain_vols,
                          weight_method="volume", **common)

print(f"n_retained = {ks_vol.n_retained}, Xmax = {ks_vol.xmax_mm:.2f} mm "
      f"({ks_vol.xmax_governed_by})")
print(f"Classical KCO X50 = {kco.x50_mm:.2f} mm\n")

rows = []
for i, (cc, cv) in enumerate(zip(ks_count.classes, ks_vol.classes), start=1):
    rows.append({
        "class": i, "tail": cv.is_tail or "",
        "Sj_low_m": cv.bin_edge_low_m, "Sj_high_m": cv.bin_edge_high_m,
        "N_i": cv.n_blocks, "sumV_i_m3": cv.volume_sum_m3,
        "w_count": cc.weight, "w_volume": cv.weight,
        "Sj_rep_m": cv.sj_representative_m, "JPS": cv.jps,
        "X50_i_mm": cv.x50_mm, "b_i": cv.b,
    })
df = pd.DataFrame(rows)
pd.set_option("display.width", 200)
print(df.to_string(index=False, float_format=lambda v: f"{v:.5g}"))
print(f"\nsum w_count  = {df['w_count'].sum():.9f}")
print(f"sum w_volume = {df['w_volume'].sum():.9f}")
print(f"sum N_i      = {df['N_i'].sum()}")

lv = (20, 50, 80, 90)
pc = ks_count.percentiles(levels=lv)
pv = ks_vol.percentiles(levels=lv)
print("\nKCO* percentiles (mm):")
print(f"{'':>10}" + "".join(f"{f'X{p}':>10}" for p in lv))
print(f"{'count':>10}" + "".join(f"{pc[f'X{p}_star']:>10.2f}" for p in lv))
print(f"{'volume':>10}" + "".join(f"{pv[f'X{p}_star']:>10.2f}" for p in lv))

print("\nDistinct individual KCO* curves:")
for g in ks_vol.unique_curves():
    print(f"  JPS={g['jps']:>2}  X50={g['x50_mm']:.2f} mm  b={g['b']:.3f}  "
          f"classes={g['class_indices']}  w_volume={g['weight']:.4f}  "
          f"w_count={sum(ks_count.classes[k-1].weight for k in g['class_indices']):.4f}")

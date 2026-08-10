"""
extract_dfn_diameters_varenne.py

Extract DFN-generated fracture characteristics for VARENNE site.
"""

import os
import math
import numpy as np
import pandas as pd

from external_libraries import DFN
from utils.geometry import normal_from_dip_dipdir, build_center_quad, jitter_orientation
from utils.excel_loader import load_orientations_from_excel
from utils.persistence_params import load_size_params
from run_site import SITE_CONFIGS

SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
ASSETS_DIR   = os.path.join(SCRIPT_DIR, "assets")
OUTPUTS_DIR  = os.path.join(SCRIPT_DIR, "outputs")
COMBINED_DIR = os.path.join(OUTPUTS_DIR, "combined")
os.makedirs(COMBINED_DIR, exist_ok=True)

def export_dfn_fracture_characteristics_varenne():
    site_key = "VARENNE"
    cfg = SITE_CONFIGS.get(site_key)
    
    if not cfg:
        print(f"❌ Site {site_key} not found in SITE_CONFIGS")
        return None, None, None

    site_name = cfg["site_name"]
    a_terrain = float(cfg["a_terrain"])
    excel_name = cfg["excel_name"]
    family_ids = cfg["family_ids"]
    families = [dict(f) for f in cfg["families"]]
    dip_face = cfg["dip_face"]
    dipdir_face = cfg["dipdir_face"]
    trace_name = cfg["trace_name"]
    region_filter = cfg.get("region_filter")
    region_x = float(cfg.get("region_x", 50.0))
    region_y = float(cfg.get("region_y", 50.0))
    region_z = float(cfg.get("region_z", 50.0))
    viz_seed = int(cfg.get("viz_seed", 123))
    max_fracs = int(cfg.get("max_fracs", 3000))

    summary_csv = os.path.join(OUTPUTS_DIR, site_name, "02_calibration", "P32_calibrated_summary.csv")
    if not os.path.exists(summary_csv):
        print(f"❌ Calibration summary not found: {summary_csv}")
        return None, None, None

    summary_df = pd.read_csv(summary_csv)
    p32_cal = summary_df.sort_values("fam")["P32_calibrated"].to_numpy(dtype=float)

    print(f"\n=== {site_name} DFN Fracture Characteristics Extraction ===")

    excel_path = os.path.join(ASSETS_DIR, excel_name)
    orient_by_fam = load_orientations_from_excel(excel_path, sheet=0, valid_families=family_ids)

    size_params = load_size_params(
        trace_csv=os.path.join(ASSETS_DIR, trace_name),
        site_name=site_name,
        a_terrain=a_terrain,
        region_filter=region_filter,
        family_names=[family["name"] for family in families],
    )
    for family in families:
        params = size_params.get(family["name"])
        if params:
            family["mean"] = params["mean"]
            family["sd"] = params["sd"]

    n_fam = len(families)
    n_face = normal_from_dip_dipdir(dip_face, dipdir_face)
    center = [region_x / 2.0, region_y / 2.0, region_z / 2.0]
    half_u = 0.49 * min(region_x, region_y)
    half_v = 0.49 * min(region_y, region_z)
    Q1, Q2, Q3, Q4 = build_center_quad(center, n_face, half_u, half_v)

    print(f"  Generating DFN with P32_cal = {p32_cal}, seed = {viz_seed}")

    dfn = DFN()
    dfn.set_RegionMaxCorner([region_x, region_y, region_z])
    dfn.set_RandomSeed(viz_seed)
    np.random.seed(viz_seed)

    for _ in range(n_fam):
        dfn.add_FractureSet()

    dfn.add_QuadrilateralMapping(Q1, Q2, Q3, Q4)
    dfn.add_VolumeMapping()

    added_counts = [0] * n_fam
    inserted_rows = [[] for _ in range(n_fam)]

    for i in range(n_fam):
        target = float(p32_cal[i])
        if target <= 0:
            continue

        fam_id = family_ids[i]
        family = families[i]
        mu_log = np.log(float(family["mean"])**2 / np.sqrt(float(family["sd"])**2 + float(family["mean"])**2))
        sig_log = np.sqrt(np.log(1.0 + (float(family["sd"]) / float(family["mean"]))**2))

        while dfn.volumesMapping[0].get_P32(i) < target:
            if added_counts[i] >= max_fracs:
                break

            row = orient_by_fam[fam_id].sample(1).iloc[0]
            dip_val, dipdir_val = jitter_orientation(
                float(row["Dip"]),
                float(row["Dip Direction"]),
                fisher_k=family.get("fisher"),
            )
            diam = np.random.lognormal(mu_log, sig_log)
            cx = np.random.uniform(0.0, region_x)
            cy = np.random.uniform(0.0, region_y)
            cz = np.random.uniform(0.0, region_z)
            dfn.fractureSets[i].add_CircularFracture([cx, cy, cz], dipdir_val, dip_val, diam / 2.0)
            inserted_rows[i].append({
                "dip_deg": dip_val,
                "dipdir_deg": dipdir_val,
            })
            added_counts[i] += 1

    print(f"  Generated fractures: {added_counts}")

    all_rows = []
    for fam_idx in range(n_fam):
        fs = dfn.fractureSets[fam_idx]
        fam_id = family_ids[fam_idx]
        for frac_idx, frac in enumerate(fs.fractures):
            area = frac.get_Area()
            radius = math.sqrt(area / math.pi)
            diameter = 2.0 * radius
            orientation = inserted_rows[fam_idx][frac_idx]
            all_rows.append({
                "site": site_name,
                "family_id": fam_id,
                "family_name": f"fam{fam_id}",
                "fracture_id": frac_idx,
                "dip_deg": orientation["dip_deg"],
                "dipdir_deg": orientation["dipdir_deg"],
                "area_m2": area,
                "radius_m": radius,
                "diameter_m": diameter,
            })

    df_out = pd.DataFrame(all_rows)
    out_csv = os.path.join(OUTPUTS_DIR, site_name, "DFN_fracture_characteristics_VARENNE.csv")
    out_csv_combined = os.path.join(COMBINED_DIR, "DFN_fracture_characteristics_VARENNE.csv")
    df_out.to_csv(out_csv, index=False)
    df_out.to_csv(out_csv_combined, index=False)
    
    print(f"\n✅ Saved: {out_csv}")
    print(f"✅ Saved: {out_csv_combined}")
    print(f"Total fractures: {len(df_out)}")
    print(f"\nColumns: {list(df_out.columns)}")
    print(f"Families: {sorted(df_out['family_id'].unique())}")
    
    print("\n" + "="*60)
    print("WHERE DIP_DEG AND DIPDIR_DEG COME FROM:")
    print("="*60)
    print("1. Sample orientation from Excel (assets/" + excel_name + ")")
    print("2. Apply Fisher jitter via jitter_orientation()")
    print("3. Record jittered (dip_deg, dipdir_deg) with fracture")
    print("="*60)
    
    return df_out, out_csv, out_csv_combined

if __name__ == "__main__":
    export_dfn_diameters_varenne()

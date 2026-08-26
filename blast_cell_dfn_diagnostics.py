# ============================================================
# blast_cell_dfn_diagnostics.py
#
# PLACEHOLDER-ONLY diagnostics for the S x B x H blast-cell DFN
# workflow (blast_cell_dfn.py), run BEFORE the real Varenne S x B x H
# prediction to answer two open methodological questions:
#
#   1. Axis-swap sensitivity: does region_x/region_y = S/B vs B/S
#      change the pooled block-volume distribution when S != B
#      (anisotropic fracture sets, rectangular box)?
#   2. Realisation-count convergence: how many pooled seeds are needed
#      for the pooled Sj* distribution (and its percentile-class
#      representative values) to stabilise?
#
# ALL dimensions used when this module is run stand-alone are
# PLACEHOLDER TEST VALUES (labelled as such below) -- NOT the real
# Varenne production geometry, which is still unavailable. Every output
# file/folder is prefixed "PLACEHOLDER_" and written to a dedicated
# "09_PLACEHOLDER_diagnostics" folder, kept separate from
# blast_cell_dfn.py's production output folder
# ("08_blastcell_SxBxH"). These outputs MUST NOT be used as input to
# the final predict_kco_star() call.
#
# Does not modify blast_cell_dfn.py, run_site.py, kco_model.py or
# kco_star_model.py -- only reuses their functions (including
# blast_cell_dfn's private helpers, since the fracture-placement rule
# they implement must stay byte-for-byte identical to the production
# workflow being diagnosed).
# ============================================================

from __future__ import annotations

import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp

from run_site import SITE_CONFIGS, SCRIPT_DIR
from external_libraries.unblocks import Generator
from utils.excel_loader import load_orientations_from_excel
from blast_cell_dfn import (
    _load_calibrated_p32,
    _families_with_corrected_size,
    _generate_one_blastcell_dfn,
)
from kco_star_model import (
    sj_star_distribution_from_block_volumes,
    create_sj_percentile_classes,
)


def _load_common_inputs(site_key, calibration_summary_path, max_fracs):
    """Shared setup, identical to blast_cell_dfn.generate_blast_cell_realizations."""
    cfg = SITE_CONFIGS[site_key]
    family_ids = list(cfg["family_ids"])
    max_fracs = int(max_fracs if max_fracs is not None
                    else cfg.get("max_fracs", 3000))
    if calibration_summary_path is None:
        calibration_summary_path = os.path.join(
            SCRIPT_DIR, "outputs", site_key, "02_calibration",
            "P32_calibrated_summary.csv"
        )
    p32_targets = _load_calibrated_p32(calibration_summary_path, family_ids)
    excel_path = os.path.join(SCRIPT_DIR, "assets", cfg["excel_name"])
    orient_by_fam = load_orientations_from_excel(
        excel_path, sheet=0, valid_families=family_ids
    )
    families = _families_with_corrected_size(cfg)
    return family_ids, families, orient_by_fam, p32_targets, max_fracs


def _one_realization_volumes(region_x, region_y, region_z, seed,
                             family_ids, families, orient_by_fam,
                             p32_targets, max_fracs):
    dfn, warns = _generate_one_blastcell_dfn(
        region_x, region_y, region_z, seed,
        family_ids, families, orient_by_fam, p32_targets, max_fracs,
    )
    gen = Generator()
    gen.generate_RockMass(dfn)
    vols = np.array([float(v) for v in gen.get_Volumes(True)],
                    dtype=np.float64)
    return vols, warns


def _diag_out_root(site_key, subfolder):
    out_root = os.path.join(SCRIPT_DIR, "outputs", site_key,
                            "09_PLACEHOLDER_diagnostics", subfolder)
    os.makedirs(out_root, exist_ok=True)
    return out_root


# ============================================================
# 1. AXIS-SWAP SENSITIVITY TEST
# ============================================================
def run_axis_swap_test(burden_m, spacing_m, bench_height_m, seeds,
                       site_key="VARENNE", calibration_summary_path=None,
                       max_fracs=None, verbose=True):
    """
    Compare the pooled block-volume distribution under two axis
    conventions: region_x=S,region_y=B vs region_x=B,region_y=S (same
    seeds, same region_z=H for both). Does NOT change the default
    convention used by blast_cell_dfn.py (region_x=S, region_y=B); this
    is a diagnostic only.

    Returns:
        dict with per-convention summary stats, the two-sample
        Kolmogorov-Smirnov statistic/p-value on the pooled volumes, and
        paths to the saved CSV/plot.
    """
    out_root = _diag_out_root(site_key, "axis_swap")
    family_ids, families, orient_by_fam, p32_targets, max_fracs = \
        _load_common_inputs(site_key, calibration_summary_path, max_fracs)

    conventions = {
        "S_to_x_B_to_y (current default)": (spacing_m, burden_m, bench_height_m),
        "B_to_x_S_to_y (swapped)":         (burden_m, spacing_m, bench_height_m),
    }

    pooled = {}
    for label, (rx, ry, rz) in conventions.items():
        vols_list = []
        for seed in seeds:
            vols, _ = _one_realization_volumes(
                rx, ry, rz, seed, family_ids, families, orient_by_fam,
                p32_targets, max_fracs,
            )
            vols_list.append(vols)
        pooled[label] = np.concatenate(vols_list)

    def _stats(v):
        return {
            "n_blocks": int(v.size),
            "mean_m3": float(np.mean(v)),
            "P50_m3": float(np.percentile(v, 50)),
            "P95_m3": float(np.percentile(v, 95)),
            "P99_m3": float(np.percentile(v, 99)),
        }

    rows = [{"convention": label, **_stats(v)} for label, v in pooled.items()]
    summary_df = pd.DataFrame(rows)

    labels = list(pooled.keys())
    ks_stat, ks_p = ks_2samp(pooled[labels[0]], pooled[labels[1]])

    summary_csv = os.path.join(out_root, "PLACEHOLDER_axis_swap_summary.csv")
    summary_df.to_csv(summary_csv, index=False)

    plt.figure()
    for label, v in pooled.items():
        vs = np.sort(v)
        ecdf = np.arange(1, vs.size + 1) / vs.size
        plt.plot(vs, ecdf, drawstyle="steps-post", label=label)
    plt.xscale("log")
    plt.xlabel("Block volume (m3, log scale)")
    plt.ylabel("ECDF")
    plt.title(f"PLACEHOLDER axis-swap ECDF (KS D={ks_stat:.4f}, p={ks_p:.4g})")
    plt.legend()
    plt.grid(True, which="both", alpha=0.3)
    plot_path = os.path.join(out_root, "PLACEHOLDER_axis_swap_ecdf.png")
    plt.savefig(plot_path, dpi=200, bbox_inches="tight")
    plt.close()

    if verbose:
        print("\n=== PLACEHOLDER axis-swap sensitivity test ===")
        print(summary_df.to_string(index=False))
        print(f"KS statistic (max |CDF_A - CDF_B|) = {ks_stat:.4f}, "
             f"p-value = {ks_p:.4g}")
        print(f"Saved: {summary_csv}\nSaved: {plot_path}")

    return {
        "summary": summary_df,
        "ks_statistic": float(ks_stat),
        "ks_pvalue": float(ks_p),
        "pooled_volumes": pooled,
        "summary_csv": summary_csv,
        "ecdf_plot": plot_path,
    }


# ============================================================
# 2. REALISATION-COUNT CONVERGENCE ANALYSIS
# ============================================================
def run_convergence_analysis(
        burden_m, spacing_m, bench_height_m,
        seeds, checkpoints,
        site_key="VARENNE",
        calibration_summary_path=None,
        max_fracs=None,
        percentile_edges=(0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100),
        representative_method="median",
        verbose=True,
):
    """
    Generate len(seeds) realisations ONCE, then progressively pool the
    first n seeds (n in checkpoints) to track stabilisation of the
    pooled Sj* distribution. Uses the DEFAULT axis convention
    (region_x=spacing_m, region_y=burden_m, region_z=bench_height_m),
    unchanged from blast_cell_dfn.py.

    Args:
        seeds: Ordered list of distinct seeds; checkpoints are cumulative
            prefixes of this list (checkpoint n uses seeds[:n]).
        checkpoints: Increasing realisation counts to evaluate, e.g.
            [5, 10, 20, 30, 50]. Every value must be <= len(seeds).

    Returns:
        dict with the checkpoint summary table (incl. relative changes
        vs the previous checkpoint), the per-class representative-Sj*
        table, and paths to the saved CSV/plot.
    """
    if any(n > len(seeds) for n in checkpoints):
        raise ValueError("every checkpoint must be <= len(seeds)")

    out_root = _diag_out_root(site_key, "convergence")
    family_ids, families, orient_by_fam, p32_targets, max_fracs = \
        _load_common_inputs(site_key, calibration_summary_path, max_fracs)

    region_x, region_y, region_z = spacing_m, burden_m, bench_height_m

    per_seed_volumes = {}
    for seed in seeds:
        vols, _ = _one_realization_volumes(
            region_x, region_y, region_z, seed, family_ids, families,
            orient_by_fam, p32_targets, max_fracs,
        )
        per_seed_volumes[seed] = vols

    rows, class_rows = [], []
    for n in checkpoints:
        pooled_vols = np.concatenate([per_seed_volumes[s] for s in seeds[:n]])
        sj_dist = sj_star_distribution_from_block_volumes(pooled_vols)
        sj = sj_dist.sj_star_m

        rows.append({
            "n_realizations": n,
            "n_blocks_pooled": int(sj.size),
            "sj_star_P50_m": float(np.percentile(sj, 50)),
            "sj_star_P95_m": float(np.percentile(sj, 95)),
            "sj_star_P99_m": float(np.percentile(sj, 99)),
            "sj_star_mean_m": float(np.mean(sj)),
        })

        classes = create_sj_percentile_classes(
            sj_dist, percentile_edges=percentile_edges,
            representative_method=representative_method,
        )
        for c in classes:
            class_rows.append({
                "n_realizations": n,
                "class_label": f"{c.percentile_low:.0f}-{c.percentile_high:.0f}",
                "sj_representative_m": c.sj_representative_m,
                "n_blocks": c.n_blocks,
            })

    summary_df = pd.DataFrame(rows)
    class_df = pd.DataFrame(class_rows)

    for col in ["n_blocks_pooled", "sj_star_P50_m", "sj_star_P95_m",
               "sj_star_P99_m", "sj_star_mean_m"]:
        summary_df[f"{col}_rel_change"] = summary_df[col].pct_change().abs()

    class_pivot = class_df.pivot(index="n_realizations",
                                 columns="class_label",
                                 values="sj_representative_m")
    max_class_rel_change = class_pivot.pct_change().abs().max(axis=1)
    summary_df["max_class_sj_rel_change"] = summary_df["n_realizations"].map(
        max_class_rel_change.to_dict()
    )

    summary_csv = os.path.join(out_root, "PLACEHOLDER_convergence_summary.csv")
    class_csv = os.path.join(
        out_root, "PLACEHOLDER_convergence_class_representatives.csv")
    summary_df.to_csv(summary_csv, index=False)
    class_df.to_csv(class_csv, index=False)

    plt.figure()
    for col, lbl in [("sj_star_P50_m", "Sj* P50"),
                     ("sj_star_P95_m", "Sj* P95"),
                     ("sj_star_P99_m", "Sj* P99"),
                     ("sj_star_mean_m", "Sj* mean")]:
        plt.plot(summary_df["n_realizations"], summary_df[col],
                 marker="o", label=lbl)
    plt.xlabel("Number of pooled realizations")
    plt.ylabel("Sj* (m)")
    plt.title("PLACEHOLDER convergence of pooled Sj* distribution")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plot_path = os.path.join(out_root, "PLACEHOLDER_convergence_plot.png")
    plt.savefig(plot_path, dpi=200, bbox_inches="tight")
    plt.close()

    if verbose:
        print("\n=== PLACEHOLDER realisation-count convergence ===")
        print(summary_df.to_string(index=False))
        print(f"\nSaved: {summary_csv}\nSaved: {class_csv}\nSaved: {plot_path}")

    return {
        "summary": summary_df,
        "class_representatives": class_df,
        "summary_csv": summary_csv,
        "class_csv": class_csv,
        "plot": plot_path,
    }


if __name__ == "__main__":
    # ------------------------------------------------------------
    # PLACEHOLDER TEST DIMENSIONS ONLY -- NOT real Varenne geometry.
    # Anisotropic (S != B) on purpose, to make the axis-swap test
    # meaningful.
    # ------------------------------------------------------------
    _B, _S, _H = 3.0, 4.0, 12.0
    _SEEDS = list(range(2001, 2051))          # 50 realisations
    _CHECKPOINTS = [5, 10, 20, 30, 50]

    run_axis_swap_test(burden_m=_B, spacing_m=_S, bench_height_m=_H,
                       seeds=_SEEDS[:10])      # 10 seeds is enough to compare
    run_convergence_analysis(burden_m=_B, spacing_m=_S, bench_height_m=_H,
                             seeds=_SEEDS, checkpoints=_CHECKPOINTS)

# ============================================================
# analyze_sj_star_bin_ranges.py
#
# DIAGNOSTIC / COMPARISON ONLY -- does not modify kco_star_model.py,
# run_kco_varenne_demo.py, or any KCO*/KCO calculation. No block is
# ever filtered or deleted; this only changes where the 10 equal-width
# ln(Sj*) bin BOUNDARIES are placed.
#
# Question being answered: the current create_sj_log_bins() spaces its
# 10 bins evenly between ln(min(Sj*)) and ln(max(Sj*)) of the ACTUAL
# population. Because the lower tail contains a handful of extreme
# small-Sj* blocks (possible DFN boundary/sliver artifacts), several of
# the 10 bins get "spent" on that sliver tail instead of the bulk of
# the distribution. This script compares three ways of choosing the
# ln(Sj*) RANGE over which the 10 equal-width bins are laid out:
#
#   1. min-max   : current behaviour (ln(min), ln(max)) of full population.
#   2. P1-P99    : ln(Sj*) at the 1st/99th empirical percentile.
#   3. P5-P95    : ln(Sj*) at the 5th/95th empirical percentile.
#
# For options 2 and 3, the OUTERMOST bin edges are still stretched out
# to the true min/max so that every block remains assigned to some bin
# (no filtering) -- only the 8 (or 9) INTERIOR bin edges are placed
# using the robust central range, so more resolution is spent on the
# bulk of the distribution and the tails are absorbed into the first/
# last (necessarily wider) bin.
#
# Real pooled S x B x H DFN block volumes are used (same file the main
# demo script reads/regenerates), not placeholder data.
# ============================================================

import os

import numpy as np
import pandas as pd

from run_site import SCRIPT_DIR
from kco_star_model import sj_star_distribution_from_block_volumes


POOLED_CSV = os.path.join(
    SCRIPT_DIR, "outputs", "VARENNE", "08_blastcell_SxBxH",
    "block_volumes_blastcell_pooled.csv",
)
N_BINS = 10


def load_pooled_volumes(csv_path: str) -> np.ndarray:
    df = pd.read_csv(csv_path)
    for candidate in ("volume_m3", "volume"):
        if candidate in df.columns:
            return df[candidate].to_numpy(dtype=float)
    raise ValueError(
        f"could not find a volume column in {csv_path}; "
        f"columns present: {list(df.columns)}"
    )


def build_bins(sj_sorted: np.ndarray, ln_lo: float, ln_hi: float,
               n_bins: int = N_BINS) -> np.ndarray:
    """
    n_bins equal-width edges in ln(Sj*) between ln_lo and ln_hi, then
    stretch the outer edges to the true population min/max so every
    block is still assigned to some bin (no filtering).
    """
    ln_edges = np.linspace(ln_lo, ln_hi, n_bins + 1)
    edges_m = np.exp(ln_edges)
    edges_m[0] = sj_sorted[0]
    edges_m[-1] = sj_sorted[-1]
    return edges_m


def report_option(label: str, sj_sorted: np.ndarray, edges_m: np.ndarray,
                  n_outside_lo: int, n_outside_hi: int,
                  central_lo_m: float, central_hi_m: float) -> pd.DataFrame:
    n = sj_sorted.size
    bin_idx = np.digitize(sj_sorted, edges_m[1:-1], right=False)
    rows = []
    total_assigned = 0
    for b in range(N_BINS):
        mask = bin_idx == b
        n_b = int(mask.sum())
        total_assigned += n_b
        w_b = n_b / n
        rows.append({
            "bin": b + 1,
            "Sj_lower_m": edges_m[b],
            "Sj_upper_m": edges_m[b + 1],
            "n_blocks": n_b,
            "weight": w_b,
        })
    assert total_assigned == n, (
        f"{label}: internal error, assigned {total_assigned}, expected {n}"
    )
    df = pd.DataFrame(rows)

    print("\n" + "=" * 70)
    print(f"Option: {label}")
    print("=" * 70)
    print(f"Central range used for interior bin spacing: "
         f"[{central_lo_m:.6g}, {central_hi_m:.6g}] m")
    print(f"Blocks BELOW central range (absorbed into bin 1, tail): "
         f"{n_outside_lo} ({100 * n_outside_lo / n:.3f}%)")
    print(f"Blocks ABOVE central range (absorbed into bin {N_BINS}, tail): "
         f"{n_outside_hi} ({100 * n_outside_hi / n:.3f}%)")
    n_empty = int((df["n_blocks"] == 0).sum())
    n_thin = int(((df["n_blocks"] > 0) & (df["weight"] < 0.01)).sum())
    print(f"Empty bins: {n_empty} / {N_BINS}   "
         f"Non-empty bins with <1% of population: {n_thin} / {N_BINS}")
    with pd.option_context("display.float_format", lambda v: f"{v:.6g}"):
        print(df.to_string(index=False))
    return df


def main():
    v = load_pooled_volumes(POOLED_CSV)
    dist = sj_star_distribution_from_block_volumes(v)
    sj_sorted = np.sort(dist.sj_star_m)
    n = sj_sorted.size
    print(f"Loaded {v.size} pooled block volumes from {POOLED_CSV}")
    print(f"Retained (valid) Sj* population: n={n}, "
         f"min={sj_sorted[0]:.6g} m, max={sj_sorted[-1]:.6g} m")

    all_results = {}

    # ---- Option 1: min-max (current create_sj_log_bins behaviour) ----
    ln_lo, ln_hi = np.log(sj_sorted[0]), np.log(sj_sorted[-1])
    edges_minmax = build_bins(sj_sorted, ln_lo, ln_hi)
    all_results["min-max (current)"] = report_option(
        "min-max (current)", sj_sorted, edges_minmax,
        n_outside_lo=0, n_outside_hi=0,
        central_lo_m=sj_sorted[0], central_hi_m=sj_sorted[-1],
    )

    # ---- Option 2: P1-P99 ----
    p1 = np.percentile(sj_sorted, 1)
    p99 = np.percentile(sj_sorted, 99)
    edges_p1p99 = build_bins(sj_sorted, np.log(p1), np.log(p99))
    n_lo = int(np.sum(sj_sorted < p1))
    n_hi = int(np.sum(sj_sorted > p99))
    all_results["P1-P99"] = report_option(
        "P1-P99", sj_sorted, edges_p1p99,
        n_outside_lo=n_lo, n_outside_hi=n_hi,
        central_lo_m=p1, central_hi_m=p99,
    )

    # ---- Option 3: P5-P95 ----
    p5 = np.percentile(sj_sorted, 5)
    p95 = np.percentile(sj_sorted, 95)
    edges_p5p95 = build_bins(sj_sorted, np.log(p5), np.log(p95))
    n_lo = int(np.sum(sj_sorted < p5))
    n_hi = int(np.sum(sj_sorted > p95))
    all_results["P5-P95"] = report_option(
        "P5-P95", sj_sorted, edges_p5p95,
        n_outside_lo=n_lo, n_outside_hi=n_hi,
        central_lo_m=p5, central_hi_m=p95,
    )

    print("\n" + "=" * 70)
    print("SUMMARY: bulk-population bins vs. sliver-tail bins per option")
    print("=" * 70)
    for label, df in all_results.items():
        n_empty = int((df["n_blocks"] == 0).sum())
        n_thin = int(((df["n_blocks"] > 0) & (df["weight"] < 0.01)).sum())
        n_bulk = N_BINS - n_empty - n_thin
        print(f"  {label:<20s}: {n_bulk} bin(s) carry >=1% of the "
             f"population, {n_thin} thin (<1%), {n_empty} empty")

    out_dir = os.path.join(SCRIPT_DIR, "outputs", "VARENNE",
                           "10_kco_comparison")
    os.makedirs(out_dir, exist_ok=True)
    for label, df in all_results.items():
        safe_label = label.replace(" ", "_").replace("(", "").replace(")", "")
        out_path = os.path.join(
            out_dir, f"VARENNE_sj_star_bin_range_compare_{safe_label}.csv")
        df.to_csv(out_path, index=False)
        print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()

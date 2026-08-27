# ============================================================
# analyze_sj_star_p1_p99_tails.py
#
# DIAGNOSTIC / PROPOSAL ONLY -- does not modify kco_star_model.py,
# run_kco_varenne_demo.py, predict_kco_star(), or any KCO*/KCO
# calculation. No block is filtered or deleted.
#
# Per user request: the 10 equal-width ln(Sj*) bins are now placed
# EXACTLY inside [P1, P99] (no stretching to the true min/max as in
# analyze_sj_star_bin_ranges.py). Blocks below P1 and above P99 are
# NOT merged into bin 1 / bin 10 -- they are kept as two explicit,
# separate "lower-tail" and "upper-tail" populations, each with its
# own count, weight and representative Sj* (median), exactly like any
# other Sj* class/bin.
#
# This script only REPORTS the resulting 12-row partition (lower
# tail + 10 central bins + upper tail) and verifies that the weights
# sum to 1 by construction. It does NOT yet feed these into
# predict_kco_star() -- that wiring is a separate step, pending
# confirmation of the tail-weighting scheme proposed in the printed
# output / chat response.
# ============================================================

import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from run_site import SCRIPT_DIR
from kco_star_model import sj_star_distribution_from_block_volumes


POOLED_CSV = os.path.join(
    SCRIPT_DIR, "outputs", "VARENNE", "08_blastcell_SxBxH",
    "block_volumes_blastcell_pooled.csv",
)
N_CENTRAL_BINS = 10


def load_pooled_volumes(csv_path: str) -> np.ndarray:
    df = pd.read_csv(csv_path)
    for candidate in ("volume_m3", "volume"):
        if candidate in df.columns:
            return df[candidate].to_numpy(dtype=float)
    raise ValueError(
        f"could not find a volume column in {csv_path}; "
        f"columns present: {list(df.columns)}"
    )


def main():
    v = load_pooled_volumes(POOLED_CSV)
    dist = sj_star_distribution_from_block_volumes(v)
    sj_sorted = np.sort(dist.sj_star_m)
    n_total = sj_sorted.size

    p1 = float(np.percentile(sj_sorted, 1))
    p99 = float(np.percentile(sj_sorted, 99))
    print(f"n_total = {n_total}")
    print(f"P1  = {p1:.6g} m")
    print(f"P99 = {p99:.6g} m")

    lower_mask = sj_sorted < p1
    upper_mask = sj_sorted > p99
    central_mask = ~lower_mask & ~upper_mask

    sj_lower = sj_sorted[lower_mask]
    sj_upper = sj_sorted[upper_mask]
    sj_central = sj_sorted[central_mask]

    n_lower, n_upper, n_central = sj_lower.size, sj_upper.size, sj_central.size
    assert n_lower + n_upper + n_central == n_total

    # ---- 10 equal-width bins in ln(Sj*), STRICTLY inside [P1, P99] ----
    ln_edges = np.linspace(np.log(p1), np.log(p99), N_CENTRAL_BINS + 1)
    edges_m = np.exp(ln_edges)
    bin_idx = np.digitize(sj_central, edges_m[1:-1], right=False)

    rows = []
    rows.append({
        "class": "lower_tail (Sj* < P1)",
        "Sj_lower_m": sj_sorted[0],
        "Sj_upper_m": p1,
        "n_blocks": n_lower,
        "weight": n_lower / n_total,
        "Sj_representative_m": float(np.median(sj_lower)) if n_lower else np.nan,
    })
    total_assigned = 0
    for b in range(N_CENTRAL_BINS):
        mask = bin_idx == b
        n_b = int(mask.sum())
        total_assigned += n_b
        sj_rep = float(np.median(sj_central[mask])) if n_b else np.nan
        rows.append({
            "class": f"central_bin_{b + 1}",
            "Sj_lower_m": edges_m[b],
            "Sj_upper_m": edges_m[b + 1],
            "n_blocks": n_b,
            "weight": n_b / n_total,
            "Sj_representative_m": sj_rep,
        })
    assert total_assigned == n_central, (
        f"central bins assigned {total_assigned}, expected {n_central}"
    )
    rows.append({
        "class": "upper_tail (Sj* > P99)",
        "Sj_lower_m": p99,
        "Sj_upper_m": sj_sorted[-1],
        "n_blocks": n_upper,
        "weight": n_upper / n_total,
        "Sj_representative_m": float(np.median(sj_upper)) if n_upper else np.nan,
    })

    df = pd.DataFrame(rows)
    total_weight = df["weight"].sum()
    total_n = df["n_blocks"].sum()

    print(f"\nn_lower_tail = {n_lower}  ({100 * n_lower / n_total:.3f}%)")
    print(f"n_central    = {n_central}  ({100 * n_central / n_total:.3f}%)  "
         f"across {N_CENTRAL_BINS} bins")
    print(f"n_upper_tail = {n_upper}  ({100 * n_upper / n_total:.3f}%)")
    print(f"\nTotal blocks accounted for: {total_n} "
         f"(expected {n_total}) -> {'OK' if total_n == n_total else 'MISMATCH'}")
    print(f"Total weight: {total_weight:.9f} "
         f"(expected 1.0) -> {'OK' if abs(total_weight - 1.0) < 1e-9 else 'MISMATCH'}")

    with pd.option_context("display.float_format", lambda x: f"{x:.6g}"):
        print("\n" + df.to_string(index=False))

    out_dir = os.path.join(SCRIPT_DIR, "outputs", "VARENNE", "10_kco_comparison")
    os.makedirs(out_dir, exist_ok=True)
    table_path = os.path.join(out_dir, "VARENNE_sj_star_P1_P99_tails_table.csv")
    df.to_csv(table_path, index=False)
    print(f"\nSaved: {table_path}")

    # ---- Plot: histogram/density with P1 and P99 as thick labeled lines ----
    def _log_consistent_hist(values_m, bins_edges):
        counts, edges = np.histogram(values_m, bins=bins_edges)
        dlnx = np.diff(np.log(edges))
        density = counts / (values_m.size * dlnx)
        return edges, density

    hist_edges = np.geomspace(sj_sorted[0], sj_sorted[-1], 60)
    edges_h, density_h = _log_consistent_hist(sj_sorted, hist_edges)

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.bar(edges_h[:-1], density_h, width=np.diff(edges_h), align="edge",
          color="#888888", edgecolor="white", alpha=0.8,
          label=f"Sj* density per unit ln(Sj*) (n={n_total})")
    ax.axvspan(sj_sorted[0], p1, color="#B22222", alpha=0.10,
              label=f"lower tail < P1 (n={n_lower})")
    ax.axvspan(p99, sj_sorted[-1], color="#B22222", alpha=0.10,
              label=f"upper tail > P99 (n={n_upper})")
    for edge in edges_m:
        ax.axvline(edge, color="gray", lw=0.9, linestyle=":", zorder=3)
    ax.axvline(p1, color="#1f4e8c", lw=2.5, zorder=6)
    ax.axvline(p99, color="#1f4e8c", lw=2.5, zorder=6)
    ymax = ax.get_ylim()[1]
    ax.annotate(f"P1 = {p1:.4g} m", xy=(p1, ymax * 0.97),
               xytext=(4, 0), textcoords="offset points",
               fontsize=9, fontweight="bold", color="#1f4e8c",
               rotation=90, va="top")
    ax.annotate(f"P99 = {p99:.4g} m", xy=(p99, ymax * 0.97),
               xytext=(4, 0), textcoords="offset points",
               fontsize=9, fontweight="bold", color="#1f4e8c",
               rotation=90, va="top")
    ax.set_xscale("log")
    ax.set_xlabel("Sj* = V^(1/3) (m)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Probability density per unit ln(Sj*)", fontsize=11,
                 fontweight="bold")
    ax.set_title(
        f"Sj* Distribution -- {N_CENTRAL_BINS} bins in [P1, P99], "
        "tails kept separate -- VARENNE\n"
        f"(n={n_total}; lower tail n={n_lower}, upper tail n={n_upper})",
        fontsize=11, fontweight="bold")
    ax.grid(True, which="both", linestyle="--", alpha=0.3)
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig_path = os.path.join(out_dir, "VARENNE_sj_star_P1_P99_tails.png")
    fig.savefig(fig_path, dpi=200, bbox_inches="tight")
    print(f"Saved: {fig_path}")


if __name__ == "__main__":
    main()

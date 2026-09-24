"""
Reference cumulative in-situ block-size distribution for Varenne.

Uses the pooled block population from the 30 S x B x H blast-cell DFN
realizations (seeds 3001-3030, 2303 blocks) stored in
outputs/VARENNE/08_blastcell_SxBxH/block_volumes_blastcell_pooled.csv.

Block size = equivalent cube side S_j* = V_j^(1/3), expressed in mm.
Cumulative passing is count-based (same convention as the P95 used for Xmax).

Exports a clean high-resolution PNG for the slide
"Courbe de distribution reference pour Varenne".
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
POOLED_CSV = os.path.join(
    SCRIPT_DIR, "outputs", "VARENNE", "08_blastcell_SxBxH",
    "block_volumes_blastcell_pooled.csv",
)
OUT_PNG = os.path.join(
    SCRIPT_DIR, "outputs", "VARENNE", "10_kco_comparison",
    "VARENNE_reference_in_situ_blocksize_distribution.png",
)

D_REFS = (20, 50, 80, 90)


def main() -> None:
    df = pd.read_csv(POOLED_CSV)
    volumes = df["volume"].to_numpy(dtype=float)
    n_blocks = volumes.size

    # Equivalent cube side, mm
    sizes_mm = np.cbrt(volumes) * 1000.0
    sizes_sorted = np.sort(sizes_mm)

    # Count-based cumulative passing (%)
    passing_pct = 100.0 * np.arange(1, n_blocks + 1) / n_blocks

    # Reference D-values (percentiles of the block-size distribution)
    d_vals = {p: float(np.percentile(sizes_mm, p)) for p in D_REFS}

    fig, ax = plt.subplots(figsize=(8.0, 5.5))
    ax.plot(sizes_sorted, passing_pct, color="#1f4e79", lw=2.2,
            label="In-situ blocks (pooled S×B×H DFN)")

    # Mark D20 / D50 / D80 / D90
    for p in D_REFS:
        x = d_vals[p]
        ax.plot([x, x, 0.0], [0.0, p, p],
                color="grey", lw=0.9, ls=":", zorder=1)
        ax.plot([x], [p], marker="o", ms=6, color="#c00000", zorder=5)
        ax.annotate(f"{x:,.0f} mm",
                    xy=(x, p), xytext=(-8, -4),
                    textcoords="offset points",
                    fontsize=8, color="#c00000",
                    va="top", ha="right")

    ax.set_xscale("log")
    ax.set_xlabel("Taille des blocs (mm)", fontsize=12)
    ax.set_ylabel("Passant cumulé (%)", fontsize=12)
    ax.set_xlim(sizes_sorted.min() * 0.9, sizes_sorted.max() * 1.1)
    ax.set_ylim(0, 102)
    ax.set_yticks(np.arange(0, 101, 10))
    ax.grid(True, which="both", ls=":", lw=0.5, alpha=0.6)
    ax.legend(loc="upper left", fontsize=10, framealpha=0.9)
    ax.set_title(
        f"Courbe de distribution référence — Varenne\n"
        f"({n_blocks} blocs, 30 réalisations DFN S×B×H)",
        fontsize=12)

    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=400, bbox_inches="tight")
    plt.close(fig)

    print(f"n_blocks = {n_blocks}")
    for p in D_REFS:
        print(f"D{p} = {d_vals[p]:.2f} mm")
    print(f"saved: {OUT_PNG}")


if __name__ == "__main__":
    main()

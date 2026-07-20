"""
utils/persistence.py — Fracture persistence (disc radius) extraction and comparison plots.

For each fracture family in the calibrated DFN:
  - Extracts disc radii from fractureSets[i].fractures  (radius = sqrt(area/pi))
  - Loads measured trace lengths from the site CSV (corrected length column)
  - Saves radii to CSV and comparison plots (CDF, histogram) to the output dir
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def _radii_from_fracture_set(fs):
    """Return numpy array of disc radii (m) for a FractureSet."""
    radii = []
    for f in fs.fractures:
        area = float(f.get_Area())
        if area > 0:
            radii.append(np.sqrt(area / np.pi))
    return np.array(radii, dtype=float)


def _load_trace_lengths(assets_dir, csv_name, region_filter, fam_name):
    """
    Load corrected trace lengths for a family from the site CSV.
    Returns a float64 numpy array of positive lengths.
    """
    path = os.path.join(assets_dir, csv_name)
    if not os.path.exists(path):
        return np.array([], dtype=float)

    if path.endswith(".xlsx"):
        df = pd.read_excel(path)
    else:
        df = pd.read_csv(path)

    if region_filter and "REGION" in df.columns:
        df = df[df["REGION"].astype(str).str.upper() == region_filter.upper()]

    # Filter by family
    if "fam" in df.columns:
        df = df[df["fam"].astype(str).str.strip() == fam_name]

    if "corrected length" not in df.columns:
        return np.array([], dtype=float)

    vals = pd.to_numeric(df["corrected length"], errors="coerce").dropna().values
    return vals[vals > 0].astype(float)


def export_persistence(dfn_final, family_ids, family_names,
                       out_dir,
                       assets_dir, csv_name, region_filter,
                       site_label=""):
    """
    For each family:
      - Extract disc radii from the DFN
      - Load corrected trace lengths from the site CSV
      - Save radii CSV  → out_dir/persistence_radii_<fam>.csv
      - Save comparison plot → out_dir/persistence_comparison_<fam>.png/.pdf

    Parameters
    ----------
    dfn_final      : unblocks DFN object after final calibrated run
    family_ids     : list of int, e.g. [1, 3, 4]
    family_names   : list of str, e.g. ["fam1", "fam3", "fam4"]
    out_dir        : directory to write outputs (created if absent)
    assets_dir     : path to assets/ folder
    csv_name       : trace CSV filename (in assets/)
    region_filter  : REGION value to filter traces (or None/'' for no filter)
    site_label     : label shown in plot title
    """
    os.makedirs(out_dir, exist_ok=True)

    # The DFN fractureSets are stored in order of family insertion.
    # We iterate only over the real fracture families (ignore prism sets).
    n_fam = len(family_ids)

    for k, (fid, fname) in enumerate(zip(family_ids, family_names)):
        if k >= len(dfn_final.fractureSets):
            break

        fs = dfn_final.fractureSets[k]
        radii = _radii_from_fracture_set(fs)
        diameters = 2.0 * radii

        # --- Save radii CSV ---
        csv_out = os.path.join(out_dir, f"persistence_radii_{fname}.csv")
        pd.DataFrame({
            "radius_m":   radii,
            "diameter_m": diameters,
        }).to_csv(csv_out, index=False)

        # --- Load trace lengths ---
        traces = _load_trace_lengths(assets_dir, csv_name, region_filter, fname)

        # --- Comparison plot ---
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        title = f"{site_label} — {fname}  |  Persistence comparison"
        fig.suptitle(title, fontsize=12)

        # Left: CDF
        ax = axes[0]
        if len(diameters) > 0:
            x = np.sort(diameters)
            y = 100.0 * np.arange(1, len(x) + 1) / len(x)
            ax.plot(x, y, label=f"DFN diameters (n={len(x)})", color="steelblue")
        if len(traces) > 0:
            xt = np.sort(traces)
            yt = 100.0 * np.arange(1, len(xt) + 1) / len(xt)
            ax.plot(xt, yt, label=f"Trace lengths (n={len(xt)})",
                    color="darkorange", linestyle="--")
        ax.set_xlabel("Length / Diameter (m)")
        ax.set_ylabel("Cumulative %")
        ax.set_title("CDF")
        ax.set_ylim(0, 100)
        ax.grid(True)
        ax.legend()

        # Right: histogram (normalised density)
        ax = axes[1]
        all_vals = np.concatenate([diameters, traces]) if len(traces) > 0 else diameters
        if len(all_vals) > 0:
            bins = np.linspace(0, np.percentile(all_vals, 98) * 1.05, 30)
            if len(diameters) > 0:
                ax.hist(diameters, bins=bins, density=True, alpha=0.55,
                        label=f"DFN diameters (n={len(diameters)})", color="steelblue")
            if len(traces) > 0:
                ax.hist(traces, bins=bins, density=True, alpha=0.55,
                        label=f"Trace lengths (n={len(traces)})",
                        color="darkorange")
        ax.set_xlabel("Length / Diameter (m)")
        ax.set_ylabel("Density")
        ax.set_title("Histogram (normalised)")
        ax.grid(True)
        ax.legend()

        plt.tight_layout()
        for ext in ("png", "pdf"):
            fig.savefig(
                os.path.join(out_dir, f"persistence_comparison_{fname}.{ext}"),
                dpi=300, bbox_inches="tight"
            )
        plt.close(fig)

        n_tr = len(traces)
        print(f"  {fname}: {len(radii)} DFN discs "
              f"(r̄={np.mean(radii):.2f} m)  |  {n_tr} traces saved → {csv_out}")

    print(f"✅ Persistence outputs: {out_dir}")

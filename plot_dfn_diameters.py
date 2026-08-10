"""
plot_dfn_diameters.py — Visualise DFN-generated fracture diameters for VARENNE site.

This script reads the consolidated CSV produced by ``extract_dfn_diameters.py``
(``outputs/combined/DFN_fracture_characteristics_VARENNE.csv``) and generates:

1. **Per-family boxplots** (one figure per fracture family):
   Each figure shows side-by-side box-and-whisker plots for VARENNE site
   with data for that family.  Median lines are
   drawn in black.  Figures are saved as both PNG (300 dpi) and PDF.

2. **Combined grid figure** (all families on one canvas):
   A multi-panel grid layout (3 columns) with one subplot per family.
   Useful for a quick at-a-glance comparison across families.
   Saved as ``DFN_diameters_all_families.png/.pdf``.

3. **Summary statistics CSV**:
   Saved as ``DFN_diameter_summary_stats.csv`` with columns:
   site | family | n | mean | std | median | p10 | p90 | min | max

Prerequisites
-------------
Run ``extract_dfn_diameters.py`` (menu option [7]) before this script.
The input CSV must exist at::

    outputs/combined/DFN_fracture_characteristics_VARENNE.csv

All outputs are written to::

    outputs/combined/dfn_diameter_plots/
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
COMBINED_DIR = os.path.join(SCRIPT_DIR, "outputs", "combined")

CSV_IN  = os.path.join(COMBINED_DIR, "DFN_fracture_characteristics_VARENNE.csv")
OUT_DIR = os.path.join(COMBINED_DIR, "dfn_diameter_plots")
os.makedirs(OUT_DIR, exist_ok=True)

# ── load ─────────────────────────────────────────────────────────────────────
if not os.path.exists(CSV_IN):
    print(f"⚠️  Input file not found: {CSV_IN}")
    print(f"    Run extract_dfn_diameters.py first")
    exit(1)

df = pd.read_csv(CSV_IN)

SITES   = ["VARENNE"]
FAMILIES = sorted(df["family_name"].unique(),
                  key=lambda x: (int(x[3:]) if x[3:].isdigit() else 99))

COLORS = {
    "VARENNE":   "#1f77b4",
}

# ── summary CSV ──────────────────────────────────────────────────────────────
summary = (df.groupby(["site", "family_name"])["diameter_m"]
             .agg(n="count",
                  mean="mean",
                  std="std",
                  median="median",
                  p10=lambda x: x.quantile(0.10),
                  p90=lambda x: x.quantile(0.90),
                  min="min",
                  max="max")
             .reset_index())
summary_path = os.path.join(OUT_DIR, "DFN_diameter_summary_stats.csv")
summary.to_csv(summary_path, index=False, float_format="%.4f")
print(f"Summary stats saved: {summary_path}")
pd.set_option("display.float_format", "{:.3f}".format)
pd.set_option("display.max_rows", 60)
print(summary.to_string(index=False))

# ── one boxplot per family ────────────────────────────────────────────────────
for fam in FAMILIES:
    sub = df[df["family_name"] == fam]
    avail_sites = [s for s in SITES if s in sub["site"].values]

    data   = [sub.loc[sub["site"] == s, "diameter_m"].values for s in avail_sites]
    colors = [COLORS[s] for s in avail_sites]

    fig, ax = plt.subplots(figsize=(max(6, len(avail_sites) * 1.4), 5))
    bp = ax.boxplot(data, patch_artist=True, notch=False, vert=True,
                    medianprops=dict(color="black", linewidth=1.5))

    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.75)

    ax.set_xticks(range(1, len(avail_sites) + 1))
    ax.set_xticklabels(avail_sites, rotation=30, ha="right", fontsize=10)
    ax.set_ylabel("Fracture diameter (m)", fontsize=11)
    ax.set_title(f"DFN fracture diameters — {fam}", fontsize=12, fontweight="bold")
    ax.yaxis.set_minor_locator(mticker.AutoMinorLocator())
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.grid(axis="y", which="minor", linestyle=":", alpha=0.3)

    plt.tight_layout()
    png = os.path.join(OUT_DIR, f"DFN_diameters_{fam}.png")
    pdf = os.path.join(OUT_DIR, f"DFN_diameters_{fam}.pdf")
    plt.savefig(png, dpi=300, bbox_inches="tight")
    plt.savefig(pdf, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {png}")

# ── all families on one figure (grid) ────────────────────────────────────────
ncols  = 3
nrows  = -(-len(FAMILIES) // ncols)   # ceiling division
fig, axes = plt.subplots(nrows, ncols,
                         figsize=(ncols * 5, nrows * 4),
                         sharey=False)
axes = axes.flatten()

for ax, fam in zip(axes, FAMILIES):
    sub   = df[df["family"] == fam]
    avail = [s for s in SITES if s in sub["site"].values]
    data  = [sub.loc[sub["site"] == s, "diameter_m"].values for s in avail]
    bp = ax.boxplot(data, patch_artist=True,
                    medianprops=dict(color="black", linewidth=1.5))
    for patch, s in zip(bp["boxes"], avail):
        patch.set_facecolor(COLORS[s])
        patch.set_alpha(0.75)
    ax.set_xticks(range(1, len(avail) + 1))
    ax.set_xticklabels(avail, rotation=40, ha="right", fontsize=8)
    ax.set_ylabel("Diameter (m)", fontsize=9)
    ax.set_title(fam, fontsize=10, fontweight="bold")
    ax.grid(axis="y", linestyle="--", alpha=0.5)

for ax in axes[len(FAMILIES):]:
    ax.set_visible(False)

fig.suptitle("DFN-generated fracture diameters — all families (VARENNE site)", fontsize=13, fontweight="bold")
plt.tight_layout()
combined_png = os.path.join(OUT_DIR, "DFN_diameters_all_families.png")
combined_pdf = os.path.join(OUT_DIR, "DFN_diameters_all_families.pdf")
plt.savefig(combined_png, dpi=300, bbox_inches="tight")
plt.savefig(combined_pdf, bbox_inches="tight")
plt.close()
print(f"\n  Saved combined figure: {combined_png}")
print("\nDone.")

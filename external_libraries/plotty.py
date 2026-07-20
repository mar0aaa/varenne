# ============================================================
# plotty.py
# COMBINE BLOCK VOLUME DISTRIBUTIONS (BC1LEFT/BC1RIGHT/BC2/BC3LEFT/BC3RIGHT + BC_TOTAL)
# + OVERLAY + GRID
# + "FUSEAU" (band) built from curves:
#     - mode="quantile" (recommended): e.g., P10–P90
#     - or mode="minmax"
# + optional small margin on the band (so it doesn't stick to curves)
#
# HOW TO USE:
#   1) Put this file INSIDE the folder where your TXT files exist
#      (e.g. .../SOFTX_2020_237/build)
#   2) Run:  python3 plotty.py
#
# OUTPUT:
#   COMBINED_block_volume_grid.png / .pdf
#   COMBINED_block_volume_overlay.png / .pdf
# ============================================================

import os
import numpy as np
import matplotlib.pyplot as plt

# ----------------------------
# 1) Where are the TXT files?
# ----------------------------
BASE_DIR = os.getcwd()
# BASE_DIR = r"/home/mar0a/SOFTX_2020_237/build"

# ----------------------------
# 2) Which sites to plot?
#    Must match your EXPORT_PREFIX in each script
# ----------------------------
SITES = [
    ("BC_TOTAL",  "VIZ_calibrated_BC_TOTAL"),
    ("BC3-RIGHT", "VIZ_calibrated_BC3RIGHT"),
    ("BC3-LEFT",  "VIZ_calibrated_BC3LEFT"),
    ("BC2",       "VIZ_calibrated_BC2"),
    ("BC1-RIGHT", "VIZ_calibrated_BC1RIGHT"),
    ("BC1-LEFT",  "VIZ_calibrated_BC1LEFT"),
]

# Pick which volume file suffix you want:
VOLUME_SUFFIX = "_BlockVolumes_clean.txt"
# VOLUME_SUFFIX = "_BlockVolumes_with_prism.txt"

# ----------------------------
# 3) Plot controls
# ----------------------------
GRID_NCOLS = 2
MAKE_OVERLAY = True
MIN_POSITIVE = 1e-15

# Force x-axis to extend to 10^3 (as you asked)
FORCE_XMAX = 1e3   # set None to disable
FORCE_XMIN = None  # e.g. 1e-12, or None

# Legend placement (overlay)
LEGEND_LOC = "upper left"  # "upper left", "upper right", ...
LEGEND_BBOX = None         # e.g. (0.02, 0.98) if you want precise anchor

# ----------------------------
# 4) Fuseau (band) settings
# ----------------------------
ADD_FUSEAU = True

# Which curves define the band?
# You can include BC_TOTAL or not. Here: YES include BC_TOTAL.
FUSEAU_INCLUDE_NAMES = {"BC_TOTAL", "BC3-RIGHT", "BC3-LEFT", "BC2", "BC1-RIGHT", "BC1-LEFT"}

# Band mode:
#   "quantile" = recommended (less sensitive to extremes)
#   "minmax"   = strict envelope
FUSEAU_MODE = "quantile"

# Quantile band settings (used if mode="quantile")
Q_LOW = 0.10   # P10
Q_HIGH = 0.90  # P90

# Add a small margin so the band doesn't stick to curves:
# (these margins are applied after computing the band)
Y_MARGIN_PERCENT_POINTS = 2.0   # e.g., add +/-2 percentage points (0..100 axis)
X_MARGIN_RATIO = 0.00           # optional, usually keep 0 on log-x; set 0.05 for 5% if you want

# How dense the band x-grid should be (log-space)
FUSEAU_NX = 400

# Band transparency
FUSEAU_ALPHA = 0.15
FUSEAU_LABEL = "Fuseau"

# ----------------------------
# Helpers
# ----------------------------
def load_volumes(txt_path: str) -> np.ndarray:
    """Load volumes from TXT and clean."""
    v = np.loadtxt(txt_path, ndmin=1)
    v = np.asarray(v, dtype=float).ravel()
    v = v[np.isfinite(v)]
    v = v[v > 0]
    return v

def cdf_percent_smaller(vols: np.ndarray):
    """Empirical CDF in percent: % of blocks with volume <= x."""
    x = np.sort(vols)
    n = x.size
    y = 100.0 * (np.arange(1, n + 1) / n)
    return x, y

def interp_cdf_on_grid(x_sorted, y_sorted, xgrid):
    """
    Interpolate a CDF (x_sorted ascending, y in 0..100) onto xgrid in log-x.
    We do linear interpolation in x (already sorted).
    """
    # numpy.interp does 1D linear interpolation; outside range it clamps to end values.
    return np.interp(xgrid, x_sorted, y_sorted, left=y_sorted[0], right=y_sorted[-1])

# ----------------------------
# 5) Read all sites + global x-limits
# ----------------------------
data = []
global_min = np.inf
global_max = -np.inf

print("\n=== Reading volume files ===")
missing = []
for name, prefix in SITES:
    txt = os.path.join(BASE_DIR, f"{prefix}{VOLUME_SUFFIX}")
    if not os.path.exists(txt):
        missing.append(txt)
        continue

    vols = load_volumes(txt)
    if vols.size == 0:
        missing.append(txt + " (empty after cleaning)")
        continue

    x, y = cdf_percent_smaller(vols)

    vmin = max(np.min(x), MIN_POSITIVE)
    vmax = np.max(x)

    global_min = min(global_min, vmin)
    global_max = max(global_max, vmax)

    data.append({"name": name, "prefix": prefix, "txt": txt, "x": x, "y": y, "vmax": vmax})

    print(f"✅ {name:8s}  vmax = {vmax:.3f}   file = {txt}")

if missing:
    print("\n⚠️ Missing/empty volume files:")
    for m in missing:
        print(" -", m)

if len(data) == 0:
    raise RuntimeError(
        "No valid BlockVolumes_*.txt files found.\n"
        "Put plotty.py in the same folder as the TXT files, or set BASE_DIR."
    )

global_min = max(global_min, MIN_POSITIVE)

# Force x limits if requested
xmin_plot = global_min if FORCE_XMIN is None else max(float(FORCE_XMIN), MIN_POSITIVE)
xmax_plot = global_max if FORCE_XMAX is None else float(FORCE_XMAX)

print("\n=== Global axis range ===")
print("global_min =", global_min)
print("global_max =", global_max, " (forced ->", xmax_plot, ")")

# ----------------------------
# 6) MULTI-PANEL GRID FIGURE
# ----------------------------
n = len(data)
ncols = GRID_NCOLS
nrows = int(np.ceil(n / ncols))

fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(10, 4.2 * nrows))
axes = np.array(axes).reshape(-1)

for ax_i, ax in enumerate(axes):
    if ax_i >= n:
        ax.axis("off")
        continue

    d = data[ax_i]
    ax.plot(d["x"], d["y"])
    ax.set_xscale("log")
    ax.set_xlim(xmin_plot, xmax_plot)
    ax.set_ylim(0, 100)

    ax.set_title(f"Block Volume Distribution — {d['name']}")
    ax.set_xlabel("Block volume (m3)")
    ax.set_ylabel("% Volume Smaller Than")
    ax.grid(True)

fig.tight_layout()

out_png = os.path.join(BASE_DIR, "COMBINED_block_volume_grid.png")
out_pdf = os.path.join(BASE_DIR, "COMBINED_block_volume_grid.pdf")
fig.savefig(out_png, dpi=300, bbox_inches="tight")
fig.savefig(out_pdf, bbox_inches="tight")
print("\n✅ Saved:", out_png)
print("✅ Saved:", out_pdf)

# ----------------------------
# 7) OPTIONAL OVERLAY + FUSEAU
# ----------------------------
if MAKE_OVERLAY:
    fig2 = plt.figure(figsize=(10, 6))
    ax2 = fig2.add_subplot(111)

    # Plot curves
    for d in data:
        ax2.plot(d["x"], d["y"], label=d["name"])

    # ---- FUSEAU (band)
    if ADD_FUSEAU:
        # Select curves for band
        band_curves = [d for d in data if d["name"] in FUSEAU_INCLUDE_NAMES]
        if len(band_curves) >= 2:
            # Build common log-spaced x-grid over plotting range
            xg_min = xmin_plot
            xg_max = xmax_plot
            xgrid = np.logspace(np.log10(xg_min), np.log10(xg_max), FUSEAU_NX)

            # Interpolate each curve on xgrid
            Y = []
            for d in band_curves:
                yi = interp_cdf_on_grid(d["x"], d["y"], xgrid)
                Y.append(yi)
            Y = np.vstack(Y)  # shape (ncurves, nx)

            if FUSEAU_MODE.lower() == "minmax":
                y_low = np.min(Y, axis=0)
                y_high = np.max(Y, axis=0)
            else:
                # quantile mode
                y_low = np.quantile(Y, Q_LOW, axis=0)
                y_high = np.quantile(Y, Q_HIGH, axis=0)

            # Add small margins (so band doesn't "stick" to curves)
            if Y_MARGIN_PERCENT_POINTS and Y_MARGIN_PERCENT_POINTS > 0:
                y_low = y_low - Y_MARGIN_PERCENT_POINTS
                y_high = y_high + Y_MARGIN_PERCENT_POINTS

            # Clamp to [0, 100]
            y_low = np.clip(y_low, 0.0, 100.0)
            y_high = np.clip(y_high, 0.0, 100.0)

            # Optional x margin (usually keep 0 on log-x)
            if X_MARGIN_RATIO and X_MARGIN_RATIO > 0:
                # This would widen in x but is not very meaningful on log;
                # kept here only if you explicitly want it.
                pass

            ax2.fill_between(xgrid, y_low, y_high, alpha=FUSEAU_ALPHA, label=FUSEAU_LABEL)
        else:
            print("\n⚠️ Not enough curves selected for fuseau (need >= 2).")

    ax2.set_xscale("log")
    ax2.set_xlim(xmin_plot, xmax_plot)
    ax2.set_ylim(0, 100)
    ax2.set_title("Block Volume Distribution — All Sites (incl. BC_TOTAL)")
    ax2.set_xlabel("Block volume (m3)")
    ax2.set_ylabel("% Volume Smaller Than")
    ax2.grid(True)

    if LEGEND_BBOX is None:
        ax2.legend(loc=LEGEND_LOC)
    else:
        ax2.legend(loc=LEGEND_LOC, bbox_to_anchor=LEGEND_BBOX)

    fig2.tight_layout()

    out2_png = os.path.join(BASE_DIR, "COMBINED_block_volume_overlay.png")
    out2_pdf = os.path.join(BASE_DIR, "COMBINED_block_volume_overlay.pdf")
    fig2.savefig(out2_png, dpi=300, bbox_inches="tight")
    fig2.savefig(out2_pdf, bbox_inches="tight")
    print("\n✅ Saved:", out2_png)
    print("✅ Saved:", out2_pdf)

plt.show()



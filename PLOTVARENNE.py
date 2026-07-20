# ============================================================
# PLOTBCTOTAL.py
# COMBINE BLOCK VOLUME DISTRIBUTIONS (BC1LEFT/BC1RIGHT/BC2/BC3LEFT/BC3RIGHT + BC_TOTAL)
# + OVERLAY + GRID
# + "FUSEAU" (band) built from curves:
#     - mode="quantile" (recommended): e.g., P10–P90
#     - or mode="minmax"
# + optional small margin on the band (so it doesn't stick to curves)
#
# OUTPUT:
#   COMBINED_block_volume_grid.png / .pdf
#   COMBINED_block_volume_overlay.png / .pdf
# ============================================================

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FixedFormatter, NullFormatter

# ----------------------------
# 1) Where are the TXT files?
# ----------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR   = SCRIPT_DIR                                        # root for reading volume TXT files
BASE_DIR   = os.path.join(SCRIPT_DIR, "outputs", "VARENNE")   # output directory for figures
os.makedirs(BASE_DIR, exist_ok=True)

# ----------------------------
# 2) Which sites to plot?
#    Must match your EXPORT_PREFIX in each script
# ----------------------------
SITES = [
    ("VARENNE",  os.path.join("outputs", "VARENNE",  "05_block_volumes", "VIZ_calibrated_VARENNE")),
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
FORCE_XMIN = 1e-4  # e.g. 1e-12, or None

# Legend placement (overlay)
LEGEND_LOC = "upper left"  # "upper left", "upper right", ...
LEGEND_BBOX = None         # e.g. (0.02, 0.98) if you want precise anchor

# ----------------------------
# 4) Fuseau (band) settings
# ----------------------------
ADD_FUSEAU = True

# Which curves define the band?
# You can include BCTOTAL or not. Here: YES include BCTOTAL.
FUSEAU_INCLUDE_NAMES = {"BCTOTAL", "BC1LEFT", "BC1RIGHT", "BC2", "BC3LEFT", "BC3RIGHT"}

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
    """
    Load block volumes from a plain-text file and clean the result.

    Reads a whitespace-delimited text file produced by the DFN pipeline
    (``export_block_volumes_simple``), filters out non-finite values and
    volumes below the global ``MIN_POSITIVE`` threshold.

    Args:
        txt_path (str): Absolute or relative path to the block-volume TXT file.
            Each line should contain exactly one floating-point volume in m³.

    Returns:
        np.ndarray: 1-D array of positive, finite block volumes (m³).  May be
            empty if the file contains no valid values.
    """
    v = np.loadtxt(txt_path, ndmin=1)
    v = np.asarray(v, dtype=float).ravel()
    v = v[np.isfinite(v)]
    v = v[v > MIN_POSITIVE]
    return v


def compute_empirical_cdf(v: np.ndarray) -> tuple:
    """
    Compute a count-based empirical CDF on a log10 x-grid.

    The y-axis represents the cumulative percentage of blocks whose volume
    is ≤ x (count-based, not volume-weighted).  This matches the standard
    blockometry percentile convention used in the analysis reports.

    Steps:
        1. Sort *v* in ascending order.
        2. Assign y_i = 100 × i / N for each block (i = 1..N).
        3. Re-sample onto a uniform log10 grid between [FORCE_XMIN, FORCE_XMAX]
           (or the data’s own range if the force limits are None) using linear
           interpolation of log10(x).  This uniform grid is required so that
           all site curves can later be interpolated onto a common axis for
           fuseau (band) computation.

    Args:
        v (np.ndarray): 1-D array of positive, finite block volumes (m³).

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray]:
            ``(x_log10, y_percent, x_orig)`` where
            - *x_log10* is the uniform log10 grid,
            - *y_percent* is the interpolated CDF (0–100),
            - *x_orig* is ``10 ** x_log10`` (original volume scale).
    """
    x_sorted = np.sort(v)
    n_blocks = x_sorted.size
    y = 100.0 * (np.arange(1, n_blocks + 1) / n_blocks)

    x_log = np.log10(x_sorted)

    # Resample onto a uniform log grid for consistent fuseau interpolation
    if FORCE_XMIN is not None and FORCE_XMAX is not None:
        x_min_log = np.log10(FORCE_XMIN)
        x_max_log = np.log10(FORCE_XMAX)
    else:
        x_min_log = x_log[0]
        x_max_log = x_log[-1]

    x_grid     = np.linspace(x_min_log, x_max_log, 600)
    x_orig_grid = 10.0 ** x_grid
    y_grid     = np.interp(x_grid, x_log, y, left=y[0], right=y[-1])

    return x_grid, y_grid, x_orig_grid


def _style_ax(ax):
    """Apply reference-style formatting to a block volume CDF axis."""
    major_x = [1e-4, 1e-2, 1e0, 1e2]
    labels   = ["1.0E-4", "1.0E-2", "1.0E+0", "1.0E+2"]
    ax.xaxis.set_major_locator(FixedLocator(major_x))
    ax.xaxis.set_major_formatter(FixedFormatter(labels))
    ax.xaxis.set_minor_locator(FixedLocator([1e-3, 1e-1, 1e1]))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.grid(True, which="major", color="gray", linewidth=0.7, alpha=0.5)
    ax.grid(True, which="minor", color="gray", linewidth=0.5, alpha=0.3, linestyle=":")


def plot_single(ax, v: np.ndarray, label: str, color: str, linestyle: str = "-"):
    """
    Plot a single site’s empirical CDF on the given axes.

    Converts the block volumes to an empirical CDF via ``compute_empirical_cdf()``
    and draws a semi-log line (log x-scale) using ``ax.semilogx()``.

    Args:
        ax: A matplotlib ``Axes`` object.
        v (np.ndarray): Block volumes (m³) for the site.
        label (str): Legend label for the line (e.g. ``"BC1LEFT (n=342)"``).
        color (str): Hex or named colour string.
        linestyle (str): Matplotlib line style (default ``"-"`` for solid).
            Use ``"--"`` for the BCTOTAL total curve to distinguish it visually.
    """
    x_log, y, x_orig = compute_empirical_cdf(v)
    ax.semilogx(x_orig, y, color=color, linewidth=2.5, linestyle=linestyle, label=label)


# ----------------------------
# Main
# ----------------------------
print("Loading volumes...")
data = {}
colors_map = {
    "BC1LEFT":  "#1f77b4",
    "BC1RIGHT": "#ff7f0e",
    "BC2":      "#2ca02c",
    "BC3LEFT":  "#d62728",
    "BC3RIGHT": "#9467bd",
    "BCTOTAL":  "#000000",  # Black for total
}

for site_name, path_prefix in SITES:
    txt_file = os.path.join(DATA_DIR, path_prefix + VOLUME_SUFFIX)
    if not os.path.exists(txt_file):
        print(f"  ⚠️  {site_name}: {txt_file} not found")
        continue
    try:
        v = load_volumes(txt_file)
        data[site_name] = v
        print(f"  ✅ {site_name}: {len(v)} blocks")
    except Exception as e:
        print(f"  ❌ {site_name}: {e}")

if not data:
    print("❌ No valid data found!")
    exit(1)

# ----------------------------
# GRID PLOT
# ----------------------------
n_sites = len(data)
n_cols = GRID_NCOLS
n_rows = (n_sites + n_cols - 1) // n_cols

fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
if n_rows == 1 and n_cols == 1:
    axes = np.array([[axes]])
elif n_rows == 1 or n_cols == 1:
    axes = axes.reshape(n_rows, n_cols)

axes_flat = axes.flatten()

for idx, (site_name, v) in enumerate(data.items()):
    ax = axes_flat[idx]
    color = colors_map.get(site_name, "blue")
    plot_single(ax, v, f"{site_name} (n={len(v):,})", color)

    ax.set_xlabel("Block Volume (m³)")
    ax.set_ylabel("Cumulative Probability [% of blocks \u2264 x]")
    ax.set_title(f"{site_name}")
    ax.set_xlim([FORCE_XMIN or None, FORCE_XMAX or None])
    _style_ax(ax)
    ax.legend()

# Hide unused subplots
for idx in range(len(data), len(axes_flat)):
    axes_flat[idx].axis("off")

plt.suptitle("Block Volume Distributions (BC sites)", fontsize=14, fontweight="bold")
plt.tight_layout()

grid_png = os.path.join(BASE_DIR, "COMBINED_block_volume_grid.png")
grid_pdf = os.path.join(BASE_DIR, "COMBINED_block_volume_grid.pdf")
plt.savefig(grid_png, dpi=300, bbox_inches="tight")
plt.savefig(grid_pdf, bbox_inches="tight")
plt.close()
print(f"\n✅ {grid_png}")

# ----------------------------
# OVERLAY PLOT
# ----------------------------
if MAKE_OVERLAY:
    fig, ax = plt.subplots(figsize=(12, 7))

    # Plot individual sites
    for site_name, v in data.items():
        color = colors_map.get(site_name, "blue")
        linestyle = "-" if site_name != "BCTOTAL" else "--"
        plot_single(ax, v, f"{site_name} (n={len(v):,})", color, linestyle=linestyle)

    # Add fuseau (band)
    if ADD_FUSEAU:
        fuseau_sites = {k: v for k, v in data.items() if k in FUSEAU_INCLUDE_NAMES}
        
        if len(fuseau_sites) >= 2:
            if FUSEAU_MODE == "quantile":
                x_log_list = []
                y_list = []
                
                for site_name, v in fuseau_sites.items():
                    x_log, y, _ = compute_empirical_cdf(v)
                    x_log_list.append(x_log)
                    y_list.append(y)
                
                # Common x grid
                x_min = min(xl[0] for xl in x_log_list)
                x_max = max(xl[-1] for xl in x_log_list)
                x_common = np.linspace(x_min, x_max, FUSEAU_NX)
                
                # Interpolate all curves to common grid
                from scipy.interpolate import interp1d
                y_interp = []
                for i, x_log in enumerate(x_log_list):
                    f = interp1d(x_log, y_list[i], kind="linear", fill_value="extrapolate", bounds_error=False)
                    y_interp.append(f(x_common))
                
                y_interp = np.array(y_interp)
                y_low = np.quantile(y_interp, Q_LOW, axis=0)
                y_high = np.quantile(y_interp, Q_HIGH, axis=0)
                
                # Apply margin
                if Y_MARGIN_PERCENT_POINTS > 0:
                    y_low = np.maximum(y_low - Y_MARGIN_PERCENT_POINTS, 0)
                    y_high = np.minimum(y_high + Y_MARGIN_PERCENT_POINTS, 100)
                
                x_orig_band = 10**x_common
                ax.fill_between(x_orig_band, y_low, y_high, alpha=FUSEAU_ALPHA, color="gray", label=FUSEAU_LABEL)
            
            elif FUSEAU_MODE == "minmax":
                x_log_list = []
                y_list = []
                
                for site_name, v in fuseau_sites.items():
                    x_log, y, _ = compute_empirical_cdf(v)
                    x_log_list.append(x_log)
                    y_list.append(y)
                
                x_min = min(xl[0] for xl in x_log_list)
                x_max = max(xl[-1] for xl in x_log_list)
                x_common = np.linspace(x_min, x_max, FUSEAU_NX)
                
                from scipy.interpolate import interp1d
                y_interp = []
                for i, x_log in enumerate(x_log_list):
                    f = interp1d(x_log, y_list[i], kind="linear", fill_value="extrapolate", bounds_error=False)
                    y_interp.append(f(x_common))
                
                y_interp = np.array(y_interp)
                y_min = np.min(y_interp, axis=0)
                y_max = np.max(y_interp, axis=0)
                
                if Y_MARGIN_PERCENT_POINTS > 0:
                    y_min = np.maximum(y_min - Y_MARGIN_PERCENT_POINTS, 0)
                    y_max = np.minimum(y_max + Y_MARGIN_PERCENT_POINTS, 100)
                
                x_orig_band = 10**x_common
                ax.fill_between(x_orig_band, y_min, y_max, alpha=FUSEAU_ALPHA, color="gray", label=FUSEAU_LABEL)

    ax.set_xlabel("Block Volume (m³)", fontsize=12)
    ax.set_ylabel("Cumulative Probability [% of blocks \u2264 x]", fontsize=12)
    ax.set_title("Block Volume Distributions Overlay — BC sites", fontsize=14, fontweight="bold")
    ax.set_xlim([FORCE_XMIN or None, FORCE_XMAX or None])
    _style_ax(ax)
    ax.legend(loc=LEGEND_LOC, bbox_to_anchor=LEGEND_BBOX)

    plt.tight_layout()

    overlay_png = os.path.join(BASE_DIR, "COMBINED_block_volume_overlay.png")
    overlay_pdf = os.path.join(BASE_DIR, "COMBINED_block_volume_overlay.pdf")
    plt.savefig(overlay_png, dpi=300, bbox_inches="tight")
    plt.savefig(overlay_pdf, bbox_inches="tight")
    plt.close()
    print(f"✅ {overlay_png}")

print("\n✅ Done!")

# ============================================================
# SPACING — Perpendicular fracture spacing survival analysis
#            VARENNE site analysis
#
# Outputs saved to: outputs/SPACING/
#   - spacing_survival_<fam>.png
#   - spacing_survival_VARENNE_all_families.png
#   - spacing_fit_summary_LOGN.xlsx
#   - spacing_fit_summary_WIDE_LOGN.xlsx
#   - raw_spacing_values.xlsx
#   - fracture_counts_and_spacing_counts.xlsx
#   - spacing_summary_per_family_region.xlsx
#   - mean_spacing_wide.xlsx
# ============================================================

import os
import math
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ============================================================
# SETTINGS
# ============================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()

INPUT_FILE = os.path.join(SCRIPT_DIR, "assets", "varenne_traces.csv")
OUT_DIR    = os.path.join(SCRIPT_DIR, "outputs", "SPACING")

MIN_N  = 2
LOG_X  = True
S_MIN  = 0.01
CAP_Q  = 0.99
N_GRID = 250

COL_SX     = "Sx"
COL_SY     = "Sy"
COL_EX     = "Ex"
COL_EY     = "Ey"
COL_FAM    = "fam"
COL_REGION = "REGION"

# ============================================================
# HELPERS
# ============================================================
def _fam_sort_key(x):
    """
    Return an integer sort key for a fracture-family label string.

    Strips whitespace and the prefix ``"fam"`` (case-insensitive), then
    converts the remainder to an integer.  Returns 9999 for any label that
    cannot be parsed (e.g. a non-numeric suffix), so unknown families always
    sort to the end.

    Args:
        x: Anything convertible to str (family label).

    Returns:
        int: Sort key for stable ordering of family labels.
    """
    try:
        return int(float(str(x).strip().lower().replace("fam", "")))
    except Exception:
        return 9999


def normalize(v):
    """
    Return a unit vector in the direction of *v*.

    Args:
        v (array-like): Input vector of any length.

    Returns:
        np.ndarray: Unit vector.  If the input has zero norm, the original
            vector is returned unchanged to avoid division by zero.
    """
    v = np.asarray(v, dtype=float)
    n = np.linalg.norm(v)
    return v if n == 0 else v / n


def pca_best_fit_direction_xy(vectors_xy):
    """
    Find the dominant 2-D direction of a set of vectors using PCA.

    Centres the vectors, computes the 2×2 covariance matrix, and returns
    the eigenvector corresponding to the largest eigenvalue.  This direction
    captures the mean strike of the fracture traces used for spacing computation.

    Args:
        vectors_xy (np.ndarray): Array of shape (N, 2) with 2-D direction
            vectors (e.g. trace end − start).

    Returns:
        np.ndarray: Unit 2-D vector (shape (2,)) giving the dominant direction.
    """
    X = vectors_xy - vectors_xy.mean(axis=0)
    C = np.cov(X.T)
    eigvals, eigvecs = np.linalg.eig(C)
    v = eigvecs[:, np.argmax(eigvals)]
    return normalize(v.real)


def compute_perpendicular_spacing(df_group: pd.DataFrame) -> np.ndarray:
    """
    Compute perpendicular spacing between fracture traces in 2-D map view.

    Algorithm
    ---------
    1. Extract start (Sx, Sy) and end (Ex, Ey) coordinates from *df_group*.
    2. Compute per-trace direction vectors and derive the mean trace direction
       via unit-vector averaging.
    3. Build a perpendicular unit vector (90° rotation of the mean direction).
    4. Project each trace’s midpoint onto this perpendicular direction.
    5. Sort the projected values by absolute distance from the centroid and
       compute successive differences to obtain spacings.
    6. Filter out spacings below ``S_MIN`` (minimum meaningful spacing).

    The perpendicular spacing is a key rock-mass characterisation metric:
    it determines the fracture intensity per family as seen from the
    direction of maximum variability across the mapping face.

    Args:
        df_group (pd.DataFrame): Rows for one (family, region) combination.
            Must contain columns: ``Sx``, ``Sy``, ``Ex``, ``Ey`` (numeric).

    Returns:
        np.ndarray: 1-D array of positive perpendicular spacings (m).
            Empty array if fewer than 3 fractures are present (insufficient
            for at least one spacing) or if fewer than 2 direction vectors
            are non-degenerate.
    """
    if len(df_group) < 3:
        return np.array([])

    sx = df_group[COL_SX].to_numpy(dtype=float)
    sy = df_group[COL_SY].to_numpy(dtype=float)
    ex = df_group[COL_EX].to_numpy(dtype=float)
    ey = df_group[COL_EY].to_numpy(dtype=float)

    dx, dy = ex - sx, ey - sy

    dirs = []
    for i in range(len(df_group)):
        d = np.array([dx[i], dy[i]])
        if np.linalg.norm(d) > 1e-10:
            dirs.append(normalize(d))

    if len(dirs) < 2:
        return np.array([])

    dirs = np.array(dirs)
    mean_dir = normalize(np.mean(dirs, axis=0))
    
    perp_dir = np.array([-mean_dir[1], mean_dir[0]])

    centers = np.column_stack([
        (sx + ex) / 2.0,
        (sy + ey) / 2.0,
    ])

    projections = np.dot(centers - centers.mean(axis=0), perp_dir)
    projections_sorted = np.sort(np.abs(projections))

    spacings = np.diff(projections_sorted)
    spacings = spacings[spacings > S_MIN]
    
    return spacings


def survival_empirical(spacing: np.ndarray):
    """Empirical survival: P(S >= s) = (N - rank + 1) / N"""
    if spacing is None or len(spacing) < 2:
        return None, None
    s = np.array(spacing, dtype=float)
    s = s[np.isfinite(s) & (s > 0)]
    if len(s) < 2:
        return None, None
    s_sorted = np.sort(s)
    n = len(s_sorted)
    y = 100.0 * (1.0 - (np.arange(1, n + 1) - 1) / n)
    return s_sorted, y


def _norm_cdf(z):
    """Cumulative normal distribution"""
    z = np.asarray(z, dtype=float)
    return 0.5 * (1.0 + np.vectorize(math.erf)(z / math.sqrt(2.0)))


def lognormal_fit(spacing: np.ndarray, x_grid: np.ndarray):
    """Fit lognormal distribution and compute survival curve"""
    s = np.array(spacing, dtype=float)
    s = s[np.isfinite(s) & (s > 0)]
    if len(s) < 2:
        return None

    ls    = np.log(s)
    mu    = float(np.mean(ls))
    sigma = max(float(np.std(ls, ddof=0)), 1e-12)

    xg = np.where(x_grid <= 0, 1e-12, x_grid.astype(float))
    y  = 100.0 * (1.0 - _norm_cdf((np.log(xg) - mu) / sigma))

    x_emp, y_emp = survival_empirical(s)
    if x_emp is not None:
        y_i  = np.interp(np.log(x_emp), np.log(x_grid), y) if LOG_X else np.interp(x_emp, x_grid, y)
        rmse = float(np.sqrt(np.mean((y_i - y_emp) ** 2)))
    else:
        rmse = np.nan

    return {"mu_log": mu, "sigma_log": sigma, "y_model": y, "rmse": rmse}


def lognormal_mean_std(mu_log: float, sigma_log: float):
    """Compute mean and std from lognormal parameters"""
    mu, s = float(mu_log), float(sigma_log)
    mean  = float(np.exp(mu + 0.5 * s * s))
    var   = float((np.exp(s * s) - 1.0) * np.exp(2.0 * mu + s * s))
    return mean, float(np.sqrt(max(var, 0.0)))


os.makedirs(OUT_DIR, exist_ok=True)

# ============================================================
# LOAD DATA
# ============================================================
print(f"Loading trace data from: {INPUT_FILE}")
try:
    df = pd.read_csv(INPUT_FILE)
except Exception as e:
    print(f"❌ Failed to load: {e}")
    exit(1)

print(f"Columns: {list(df.columns)}")
print(f"Shape: {df.shape}")

# Check required columns
required_cols = [COL_SX, COL_SY, COL_EX, COL_EY, COL_FAM, COL_REGION]
missing = [c for c in required_cols if c not in df.columns]
if missing:
    print(f"❌ Missing columns: {missing}")
    exit(1)

# Standardize data
df.columns = df.columns.str.strip()
for c in [COL_SX, COL_SY, COL_EX, COL_EY]:
    df[c] = pd.to_numeric(df[c], errors="coerce")
df = df.dropna(subset=[COL_SX, COL_SY, COL_EX, COL_EY]).copy()

df[COL_FAM]    = df[COL_FAM].astype(str).str.strip()
df[COL_REGION] = df[COL_REGION].astype(str).str.strip().str.upper()

regions  = sorted(df[COL_REGION].dropna().unique().tolist())
families = sorted(df[COL_FAM].dropna().unique().tolist(), key=_fam_sort_key)
plot_order = regions + ["BCTOTAL"]

print(f"Regions found:  {regions}")
print(f"Families found: {families}")

rows = []
spacing_rows = []
fracture_rows = []

# ============================================================
# PER-FAMILY ANALYSIS WITH OVERLAY PLOTS
# ============================================================
for fam in families:
    df_f = df[df[COL_FAM] == fam].copy()

    spacing_dict        = {}
    fracture_count_dict = {}

    # Per-region spacing
    for region in regions:
        df_fr = df_f[df_f[COL_REGION] == region].copy()
        fracture_count_dict[region] = len(df_fr)
        spacing_dict[region]        = compute_perpendicular_spacing(df_fr)

    # Pooled BCTOTAL
    fracture_count_dict["BCTOTAL"] = len(df_f)
    spacing_dict["BCTOTAL"]        = compute_perpendicular_spacing(df_f)

    keys_valid = [k for k, v in spacing_dict.items() if v is not None and len(v) >= MIN_N]
    if not keys_valid:
        print(f"  {fam}: No valid spacing data")
        continue

    all_s = np.concatenate([spacing_dict[k] for k in keys_valid if spacing_dict[k] is not None])
    all_s = all_s[np.isfinite(all_s) & (all_s > 0)]
    if len(all_s) < 2:
        print(f"  {fam}: Not enough spacings")
        continue

    x_min  = float(np.min(all_s))
    x_max  = float(np.max(all_s))
    x_grid = (np.logspace(np.log10(x_min), np.log10(x_max), N_GRID) if LOG_X
              else np.linspace(x_min, x_max, N_GRID))

    # Create overlay plot
    fig, ax = plt.subplots(figsize=(11, 7), dpi=140)
    plotted_any = False

    for key in plot_order:
        if key not in spacing_dict:
            continue

        s           = spacing_dict[key]
        n_fractures = fracture_count_dict.get(key, 0)
        n_spacing   = len(s) if s is not None and len(s) > 0 else 0

        fracture_rows.append({
            "family":      fam,
            "region":      key,
            "n_fractures": int(n_fractures),
            "n_spacing":   int(n_spacing),
        })

        if s is None or len(s) < MIN_N:
            continue

        for val in s:
            spacing_rows.append({"family": fam, "region": key, "spacing": float(val)})

        x_emp, y_emp = survival_empirical(s)
        if x_emp is None:
            continue

        fit = lognormal_fit(s, x_grid)
        if fit is None:
            continue

        mu_log    = float(fit["mu_log"])
        sigma_log = float(fit["sigma_log"])
        y_model   = fit["y_model"]
        rmse      = fit["rmse"]
        mean_fit, std_fit = lognormal_mean_std(mu_log, sigma_log)

        lab = (f"{key} LOGN (μ={mu_log:.3g}, σ={sigma_log:.3g}, RMSE={rmse:.2g}, "
               f"n_spacing={len(s)}, n_frac={n_fractures})")

        if key == "BCTOTAL":
            ax.plot(x_emp, y_emp, linestyle="--", linewidth=2.0, color="black", alpha=0.45)
            ax.plot(x_grid, y_model, linewidth=4.2, color="black", label=lab)
        else:
            line, = ax.plot(x_grid, y_model, linewidth=2.8, label=lab)
            ax.plot(x_emp, y_emp, linestyle="--", linewidth=1.6,
                    color=line.get_color(), alpha=0.45)

        plotted_any = True

        rows.append({
            "family":      fam,
            "region":      key,
            "n_fractures": int(n_fractures),
            "n_spacing":   int(len(s)),
            "model":       "LOGNORMAL",
            "rmse":        rmse,
            "mu_log":      mu_log,
            "sigma_log":   sigma_log,
            "mean_fit":    mean_fit,
            "std_fit":     std_fit,
            "S_min_used":  S_MIN,
            "cap_q_used":  CAP_Q if CAP_Q is not None else "",
        })

    if plotted_any:
        ax.set_title(f"Occurrence (%) vs Spacing — {fam}\nEmpirical (dashed) + Lognormal fit (solid)")
        ax.set_xlabel("Perpendicular spacing (m)")
        ax.set_ylabel("Occurrence (%) (≥ x)")
        ax.set_ylim(0, 100)
        if LOG_X:
            ax.set_xscale("log")
        ax.grid(True, which="both", alpha=0.25)
        ax.axvline(S_MIN, linestyle=":", linewidth=1.2, alpha=0.6)
        ax.text(S_MIN, 5, f"S_min={S_MIN} m", rotation=90, va="bottom", ha="right", alpha=0.7)
        ax.legend(fontsize=9)
        fig.tight_layout()

        out_png = os.path.join(OUT_DIR, f"spacing_survival_{fam}.png")
        out_pdf = os.path.join(OUT_DIR, f"spacing_survival_{fam}.pdf")
        plt.savefig(out_png, dpi=300, bbox_inches="tight")
        plt.savefig(out_pdf, bbox_inches="tight")
        plt.close()
        print(f"  {fam}: saved {out_png}")
    else:
        plt.close()

# ============================================================
# SAVE OUTPUTS
# ============================================================
if rows:
    df_fit = pd.DataFrame(rows)
    out_1 = os.path.join(OUT_DIR, "spacing_fit_summary_LOGN.xlsx")
    df_fit.to_excel(out_1, index=False)
    print(f"\n✅ {out_1}")

    out_2 = os.path.join(OUT_DIR, "spacing_fit_summary_WIDE_LOGN.xlsx")
    df_fit_wide = df_fit.pivot_table(
        index="family", columns="region", 
        values=["mu_log", "sigma_log", "rmse", "n_spacing"]
    )
    df_fit_wide.to_excel(out_2)
    print(f"✅ {out_2}")

if spacing_rows:
    df_spacings = pd.DataFrame(spacing_rows)
    out_3 = os.path.join(OUT_DIR, "raw_spacing_values.xlsx")
    df_spacings.to_excel(out_3, index=False)
    print(f"✅ {out_3}")

if fracture_rows:
    df_frac = pd.DataFrame(fracture_rows)
    out_4 = os.path.join(OUT_DIR, "fracture_counts_and_spacing_counts.xlsx")
    df_frac.to_excel(out_4, index=False)
    print(f"✅ {out_4}")

    out_5 = os.path.join(OUT_DIR, "spacing_summary_per_family_region.xlsx")
    df_summary = (df_frac.groupby(["family", "region"])[["n_fractures", "n_spacing"]]
                  .sum()
                  .reset_index())
    df_summary.to_excel(out_5, index=False)
    print(f"✅ {out_5}")

# ============================================================
# COMBINED VARENNE PLOT — all families overlaid (VARENNE spacing)
# ============================================================
if rows:
    df_fit = pd.DataFrame(rows)
    df_total = df_fit[df_fit["region"] == "BCTOTAL"].copy()

    if not df_total.empty:
        fig_t, ax_t = plt.subplots(figsize=(11, 7), dpi=140)

        for _, row in df_total.iterrows():
            fam = row["family"]
            mu_log    = float(row["mu_log"])
            sigma_log = float(row["sigma_log"])
            rmse      = float(row["rmse"])
            n_sp      = int(row["n_spacing"])

            # Rebuild x_grid and curves from raw spacing data
            if spacing_rows:
                s_arr = np.array([r["spacing"] for r in spacing_rows
                                  if r["family"] == fam and r["region"] == "BCTOTAL"])
            else:
                s_arr = np.array([])

            if len(s_arr) < 2:
                continue

            s_min = float(np.min(s_arr))
            s_max = float(np.max(s_arr))
            xg = (np.logspace(np.log10(s_min), np.log10(s_max), N_GRID) if LOG_X
                  else np.linspace(s_min, s_max, N_GRID))
            y_model = 100.0 * (1.0 - _norm_cdf((np.log(xg) - mu_log) / sigma_log))

            x_emp, y_emp = survival_empirical(s_arr)

            lab = f"{fam} LOGN (μ={mu_log:.3g}, σ={sigma_log:.3g}, RMSE={rmse:.2g}, n={n_sp})"
            line, = ax_t.plot(xg, y_model, linewidth=2.8, label=lab)
            ax_t.plot(x_emp, y_emp, linestyle="--", linewidth=1.6,
                      color=line.get_color(), alpha=0.45)

        ax_t.set_title("Occurrence (%) vs Spacing — VARENNE (all families)\nEmpirical (dashed) + Lognormal fit (solid)")
        ax_t.set_xlabel("Perpendicular spacing S (m)")
        ax_t.set_ylabel("Occurrence (%) (≥ x)")
        ax_t.set_ylim(0, 100)
        if LOG_X:
            ax_t.set_xscale("log")
        ax_t.grid(True, which="both", alpha=0.25)
        ax_t.legend(fontsize=9)
        fig_t.tight_layout()

        out_tot_png = os.path.join(OUT_DIR, "spacing_survival_VARENNE_all_families.png")
        out_tot_pdf = os.path.join(OUT_DIR, "spacing_survival_VARENNE_all_families.pdf")
        plt.savefig(out_tot_png, dpi=300, bbox_inches="tight")
        plt.savefig(out_tot_pdf, bbox_inches="tight")
        plt.close()
        print(f"✅ {out_tot_png}")

print("\n✅ Done!")

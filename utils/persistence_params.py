"""
utils/persistence_params.py

Mauldon-corrected fracture size parameters for DFN generation.

Workflow
--------
1. Load raw trace lengths (Length column, NO size filtering) from a site CSV.
2. Compute mu_biased, sigma_biased, CoV per family.
3. Apply Mauldon / Priest circular-window first-order correction:
       mu_nb = mu_biased * (1 + mu_biased / (2R))
   where  R = sqrt(a_terrain / pi).
   Valid for R >= mu.  When R >> mu correction is small (correct behaviour).
4. Compute sigma_nb = CoV * mu_nb.
5. Cache results to assets/mauldon_corrections.csv for audit / manual override.

Manual override
---------------
Edit assets/mauldon_corrections.csv and set any mu_nb value manually.
The loader honours manual entries and skips auto-computation for those rows.
"""

import os
import numpy as np
import pandas as pd

_HERE         = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_HERE)
MAULDON_FILE  = os.path.join(_PROJECT_ROOT, "assets", "mauldon_corrections.csv")


# ============================================================
# CORE HELPERS
# ============================================================

def compute_biased_stats(lengths):
    """
    Arithmetic mean, std (ddof=1) and CoV from raw trace lengths.
    Drops NaN and non-positive values only -- NO other filtering.
    Returns (mu_biased, sigma_biased, cov_biased), all float (NaN if n < 2).
    """
    L = np.asarray(lengths, dtype=float)
    L = L[np.isfinite(L) & (L > 0)]
    if len(L) < 2:
        return np.nan, np.nan, np.nan
    mu    = float(np.mean(L))
    sigma = float(np.std(L, ddof=1))
    cov   = sigma / mu if mu > 0 else np.nan
    return mu, sigma, cov


def mauldon_correction_factor(mu_biased, a_terrain):
    """
    First-order Mauldon / Priest circular-window correction factor:
        factor = 1 + mu_biased / (2R)   where  R = sqrt(a_terrain / pi)

    Valid for R >= mu_biased.
    Returns 1.0 if inputs are invalid or R <= 0.
    """
    if not (np.isfinite(mu_biased) and mu_biased > 0
            and np.isfinite(a_terrain) and a_terrain > 0):
        return 1.0
    R = float(np.sqrt(a_terrain / np.pi))
    if R <= 0:
        return 1.0
    return 1.0 + mu_biased / (2.0 * R)


def compute_mauldon_mu_nb(trace_csv, a_terrain, region_filter=None,
                           family_names=None,
                           fam_col="fam", len_col="Length",
                           region_col="REGION"):
    """
    Compute Mauldon-corrected mean trace length (mu_nb) per family.

    Uses first-order Priest/Mauldon correction:
        mu_nb = mu_biased * (1 + mu_biased / (2R))   R = sqrt(a_terrain/pi)

    Returns
    -------
    dict  {fam_name: {"mu_nb": float, "mu_biased": float,
                      "R": float, "factor": float, "method": str}}
    """
    df = pd.read_csv(trace_csv)
    df.columns = df.columns.str.strip()

    if region_filter is not None and region_col in df.columns:
        sub_df = df[df[region_col].astype(str).str.upper() == region_filter.upper()].copy()
    else:
        sub_df = df.copy()

    sub_df[len_col] = pd.to_numeric(sub_df[len_col], errors="coerce")
    sub_df = sub_df.dropna(subset=[len_col]).copy()
    sub_df = sub_df[sub_df[len_col] > 0].copy()

    R = float(np.sqrt(float(a_terrain) / np.pi))

    fam_list = (family_names if family_names
                else sorted(sub_df[fam_col].dropna().astype(str).unique().tolist()))

    result = {}
    for fam in fam_list:
        grp = sub_df[sub_df[fam_col].astype(str) == str(fam)]
        L   = grp[len_col].to_numpy(float)
        mu_b, _, _ = compute_biased_stats(L)

        if not np.isfinite(mu_b):
            result[str(fam)] = {
                "mu_nb": np.nan, "mu_biased": mu_b,
                "R": R, "factor": np.nan, "method": "insufficient_data",
            }
            continue

        factor = mauldon_correction_factor(mu_b, a_terrain)
        mu_nb  = mu_b * factor

        result[str(fam)] = {
            "mu_nb":     mu_nb,
            "mu_biased": mu_b,
            "R":         R,
            "factor":    factor,
            "method":    "mauldon_priest_first_order",
        }

    return result


def _update_mauldon_csv(site_name, computed):
    """
    Write computed mu_nb values into assets/mauldon_corrections.csv.
    Always overwrites existing rows — values are always recomputed from data.
    """
    if os.path.exists(MAULDON_FILE):
        tbl = pd.read_csv(MAULDON_FILE)
        tbl.columns = tbl.columns.str.strip()
    else:
        tbl = pd.DataFrame(columns=["site", "fam", "mu_nb", "mu_biased", "R", "factor", "method"])

    # Ensure extra columns exist with appropriate dtypes
    for col in ("mu_biased", "R", "factor"):
        if col not in tbl.columns:
            tbl[col] = np.nan
    if "method" not in tbl.columns:
        tbl["method"] = ""

    for fam, info in computed.items():
        mu_nb = info.get("mu_nb", np.nan)
        mask  = (
            (tbl["site"].astype(str).str.upper() == site_name.upper()) &
            (tbl["fam"].astype(str) == str(fam))
        )
        new_vals = {
            "mu_nb":     round(float(mu_nb), 6) if np.isfinite(mu_nb) else np.nan,
            "mu_biased": round(float(info.get("mu_biased", np.nan)), 6),
            "R":         round(float(info.get("R", np.nan)), 4),
            "factor":    round(float(info.get("factor", np.nan)), 6),
            "method":    info.get("method", ""),
        }
        if mask.any():
            for col, val in new_vals.items():
                if col == "method":
                    tbl["method"] = tbl["method"].astype(object)
                tbl.loc[mask, col] = val
        else:
            row = {"site": site_name, "fam": fam}
            row.update(new_vals)
            tbl = pd.concat([tbl, pd.DataFrame([row])], ignore_index=True)

    tbl.to_csv(MAULDON_FILE, index=False)


# ============================================================
# PUBLIC API
# ============================================================

def load_size_params(trace_csv, site_name, a_terrain,
                     region_filter=None, family_names=None,
                     fam_col="fam", len_col="Length", region_col="REGION"):
    """
    Load Mauldon-corrected fracture size parameters for one site.

    Steps:
      1. Check mauldon_corrections.csv for existing mu_nb values.
      2. For missing values, auto-compute via first-order correction.
      3. Save newly computed values back to the CSV (empty cells only).
      4. Compute sigma_nb = CoV_biased * mu_nb per family.

    Returns
    -------
    dict
        {fam_name: {
            "mean"        : mu_nb      (corrected mean -- DFN diameter mean)
            "sd"          : sigma_nb   (= CoV_biased * mu_nb)
            "mu_biased"   : arithmetic mean of raw lengths
            "sigma_biased": arithmetic std of raw lengths
            "cov"         : CoV of raw lengths
            "factor"      : correction factor (mu_nb / mu_biased)
            "method"      : how mu_nb was determined
        }}
    """
    # ---- Load raw trace data ----
    df = pd.read_csv(trace_csv)
    df.columns = df.columns.str.strip()

    if region_filter is not None and region_col in df.columns:
        df_site = df[df[region_col].astype(str).str.upper() == region_filter.upper()].copy()
    else:
        df_site = df.copy()

    df_site[len_col] = pd.to_numeric(df_site[len_col], errors="coerce")
    df_site = df_site.dropna(subset=[len_col]).copy()
    df_site = df_site[df_site[len_col] > 0].copy()

    fam_list = (family_names if family_names
                else sorted(df_site[fam_col].dropna().astype(str).unique().tolist()))

    # ---- Read existing Mauldon table ----
    if os.path.exists(MAULDON_FILE):
        tbl = pd.read_csv(MAULDON_FILE)
        tbl.columns = tbl.columns.str.strip()
    else:
        tbl = pd.DataFrame(columns=["site", "fam", "mu_nb"])

    site_rows = tbl[tbl["site"].astype(str).str.upper() == site_name.upper()]

    # Always recompute all families from the trace CSV
    needs_compute = list(fam_list)
    manual = {}

    # ---- Auto-compute missing mu_nb ----
    computed = {}
    if needs_compute:
        computed = compute_mauldon_mu_nb(
            trace_csv=trace_csv,
            a_terrain=a_terrain,
            region_filter=region_filter,
            family_names=needs_compute,
            fam_col=fam_col,
            len_col=len_col,
            region_col=region_col,
        )
        _update_mauldon_csv(site_name, computed)

    # ---- Build result dict ----
    result = {}
    for fam in fam_list:
        sub = df_site[df_site[fam_col].astype(str) == str(fam)]
        mu_b, sigma_b, cov = compute_biased_stats(sub[len_col].to_numpy())

        if str(fam) in computed:
            info   = computed[str(fam)]
            mu_nb  = info["mu_nb"]
            factor = info.get("factor", np.nan)
            method = info["method"]
        else:
            mu_nb  = mu_b
            factor = 1.0
            method = "biased_mean_fallback"

        sigma_nb = float(cov * mu_nb) if (np.isfinite(cov) and np.isfinite(mu_nb)) else float(sigma_b)

        result[str(fam)] = {
            "mean":         mu_nb,
            "sd":           sigma_nb,
            "mu_biased":    mu_b,
            "sigma_biased": sigma_b,
            "cov":          cov,
            "factor":       factor,
            "method":       method,
        }

    return result


def biased_stats_table(trace_csv, region_filter=None, fam_col="fam",
                       len_col="Length", region_col="REGION"):
    """
    Compute biased stats for all families in a trace CSV.
    Returns a DataFrame: fam, n, mu_biased, sigma_biased, cov_biased.
    """
    df = pd.read_csv(trace_csv)
    df.columns = df.columns.str.strip()

    if region_filter is not None and region_col in df.columns:
        df = df[df[region_col].astype(str).str.upper() == region_filter.upper()].copy()

    df[len_col] = pd.to_numeric(df[len_col], errors="coerce")
    df = df.dropna(subset=[len_col]).copy()
    df = df[df[len_col] > 0].copy()

    rows = []
    for fam, grp in df.groupby(fam_col):
        mu, sigma, cov = compute_biased_stats(grp[len_col].to_numpy())
        rows.append({"fam": fam, "n": len(grp),
                     "mu_biased": mu, "sigma_biased": sigma, "cov_biased": cov})
    return pd.DataFrame(rows)

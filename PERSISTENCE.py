# ============================================================
# PERSISTENCE — Trace length / persistence survival analysis
#               ALL sites pooled (all bc corrected fam.xlsx)
#
# Outputs saved to: outputs/PERSISTENCE/
#   - persistence_survival_<fam>.png  (one plot per family)
#   - persistence_survival_BC_TOTAL_all_families.png
#   - persistence_fit_summary_EXP_LOGN.xlsx
#   - persistence_fit_summary_WIDE.xlsx
# ============================================================

import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.ticker import LogLocator, FuncFormatter
matplotlib.use("Agg")   # non-interactive — save to file only

# ============================================================
# SETTINGS
# ============================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()

INPUT_FILE = os.path.join(SCRIPT_DIR, "assets", "varenne_traces.csv")
OUT_DIR    = os.path.join(SCRIPT_DIR, "outputs", "PERSISTENCE")

MIN_N              = 2
LOG_X              = True
FORCE_START_AT_100 = True

# Optional physical filters on persistence length
L_MIN = 0.0   # example: 0.10 to remove traces < 10 cm
L_MAX = None  # None = auto-scale from data; example: 2.0 to cap at 2.0 m

# Optional outlier control by quantile
CAP_Q         = 0.95
MIN_N_FOR_CAP = 30

SMOOTH_EMPIRICAL = True
N_GRID           = 350

COL_FAM      = "fam"
COL_REGION   = "REGION"
COL_LEN_CORR = "corrected length"
COL_LEN_RAW  = "Length"
COORD_COLS   = ("Sx", "Sy", "Sz", "Ex", "Ey", "Ez")

# ============================================================
# HELPERS
# ============================================================
def compute_persistence(df_sub: pd.DataFrame) -> np.ndarray:
    """
    Persistence = raw measured trace length (Length column).
    corrected length is intentionally NOT used — all raw traces are kept
    for unbiased survival analysis.
    Falls back to Euclidean distance when Length is absent.
    """
    if COL_LEN_RAW in df_sub.columns:
        x = pd.to_numeric(df_sub[COL_LEN_RAW], errors="coerce").to_numpy()
        x = x[np.isfinite(x) & (x > 0)]
        if x.size > 0:
            return x

    missing = [c for c in COORD_COLS if c not in df_sub.columns]
    if missing:
        raise ValueError(f"Cannot compute persistence — missing columns: {missing}")

    Sx, Sy, Sz, Ex, Ey, Ez = [
        pd.to_numeric(df_sub[c], errors="coerce").to_numpy() for c in COORD_COLS
    ]
    mask = (
        np.isfinite(Sx) & np.isfinite(Sy) & np.isfinite(Sz) &
        np.isfinite(Ex) & np.isfinite(Ey) & np.isfinite(Ez)
    )
    Sx, Sy, Sz, Ex, Ey, Ez = Sx[mask], Sy[mask], Sz[mask], Ex[mask], Ey[mask], Ez[mask]
    L = np.sqrt((Ex - Sx)**2 + (Ey - Sy)**2 + (Ez - Sz)**2)
    return L[np.isfinite(L) & (L > 0)]


def clean_values(x: np.ndarray) -> np.ndarray:
    """
    Keep only finite positive persistence values,
    then apply optional physical filters L_MIN and L_MAX.
    """
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x) & (x > 0)]

    if L_MIN is not None and L_MIN > 0:
        x = x[x >= L_MIN]

    if L_MAX is not None:
        x = x[x <= L_MAX]

    return x


def get_raw_lengths(df_sub: pd.DataFrame) -> np.ndarray:
    """
    Return all finite positive raw trace lengths from a DataFrame subset.

    Unlike ``compute_persistence()``, this function applies NO quality filters
    (no L_MIN / L_MAX / cap) so it can be used for bias statistics that must
    reflect the original measurement population.

    Args:
        df_sub (pd.DataFrame): Row subset for one (family, region) combination.
            Must contain the column named by ``COL_LEN_RAW`` (``"Length"``).

    Returns:
        np.ndarray: 1-D array of positive finite length values.  Empty array
            if the column is absent or all values are non-numeric / non-positive.
    """
    if COL_LEN_RAW not in df_sub.columns:
        return np.array([])
    x = pd.to_numeric(df_sub[COL_LEN_RAW], errors="coerce").to_numpy()
    return x[np.isfinite(x) & (x > 0)]


def compute_biased_stats(lengths: np.ndarray):
    """
    Compute arithmetic mean, standard deviation, and coefficient of variation
    from raw (unfiltered) trace lengths.

    These statistics intentionally use the raw field measurements rather than
    the L_MIN/L_MAX filtered values so that Mauldon-correction look-ups are
    consistent with the original measurement population.

    Args:
        lengths (np.ndarray): Array of length values (any shape).  Non-finite
            and non-positive entries are removed internally.

    Returns:
        tuple[float, float, float]: ``(mu_biased, sigma_biased, cov_biased)``
            where *mu_biased* is the arithmetic mean, *sigma_biased* is the
            sample standard deviation (ddof=1), and *cov_biased* is sigma/mu.
            All three are ``np.nan`` if fewer than 2 valid values are found.
    """
    L = np.asarray(lengths, dtype=float)
    L = L[np.isfinite(L) & (L > 0)]
    if len(L) < 2:
        return np.nan, np.nan, np.nan
    mu    = float(np.mean(L))
    sigma = float(np.std(L, ddof=1))
    cov   = sigma / mu if mu > 0 else np.nan
    return mu, sigma, cov


def cap_values_safe(x: np.ndarray, cap_q, min_n_for_cap: int = 30):
    """
    Remove outliers above a quantile threshold (not clipping but filtering).

    When a dataset is large enough (>= ``min_n_for_cap`` values), traces
    longer than the ``cap_q`` quantile are dropped entirely.  This prevents
    a few extremely long outlier traces from dominating the tail of the
    survival curve without artificially truncating the bulk of the data.

    Args:
        x (np.ndarray): Input array of positive values.
        cap_q (float | None): Quantile threshold in [0, 1].  If ``None``, the
            array is returned unchanged.
        min_n_for_cap (int): Minimum sample size required before capping is
            applied.  Default is 30.

    Returns:
        tuple[np.ndarray, float]: ``(x_capped, cap_value)`` where
            *x_capped* is the filtered array and *cap_value* is the actual
            quantile value used (or ``np.nan`` if capping was not applied).
    """
    if x is None or len(x) == 0:
        return np.array([]), np.nan
    if cap_q is None:
        return x, np.nan
    if len(x) < min_n_for_cap:
        return x, np.nan
    cap = np.quantile(x, cap_q)
    return x[x <= cap], float(cap)


def survival_empirical(values: np.ndarray, force_start_at_100: bool = True):
    """
    Compute the empirical survival (complementary CDF) of a dataset.

    The survival function is defined as S(x) = 100 × P(X ≥ x), where the
    probability is estimated non-parametrically from the ranked sample.

    The Hazen plotting position is used:
        y_i = 100 × (1 − (i − 1) / n)
    so the smallest observation maps to 100 % and the largest to 100/n %.

    When ``force_start_at_100`` is True and ``LOG_X`` is False, an extra
    point at (0, 100) is prepended so the curve explicitly starts at the
    origin of the survival axis.

    Args:
        values (np.ndarray): Raw observations (any shape).  Cleaned through
            ``clean_values()`` before processing.
        force_start_at_100 (bool): Prepend the (0, 100) anchor point.

    Returns:
        tuple[np.ndarray | None, np.ndarray | None]: ``(x_sorted, y_percent)``
            or ``(None, None)`` if fewer than 2 valid values remain after
            cleaning.
    """
    v = clean_values(values)
    if len(v) < 2:
        return None, None
    v_sorted = np.sort(v)
    n = len(v_sorted)
    i = np.arange(1, n + 1)
    y = 100.0 * (1.0 - (i - 1) / n)
    if force_start_at_100 and not LOG_X:
        v_sorted = np.insert(v_sorted, 0, 0.0)
        y = np.insert(y, 0, 100.0)
    return v_sorted, y


def empirical_survival_on_grid(values: np.ndarray, x_grid: np.ndarray):
    """
    Evaluate the empirical survival function on a pre-defined x grid.

    For each grid point x, computes 100 × P(X ≥ x) directly from the cleaned
    sample, giving a smooth step-free representation suitable for overlay plots.

    Args:
        values (np.ndarray): Raw observations.  Cleaned internally.
        x_grid (np.ndarray): 1-D array of x values at which to evaluate S(x).

    Returns:
        np.ndarray | None: Array of survival percentages (shape matching
            *x_grid*), or ``None`` if fewer than 2 valid values remain.
    """
    v = clean_values(values)
    if len(v) < 2:
        return None
    return np.array([100.0 * np.mean(v >= x) for x in x_grid], dtype=float)


# ============================================================
# MODELS: EXP + LOGNORMAL ONLY
# ============================================================
def exp_fit(values: np.ndarray, x_grid: np.ndarray):
    """
    Fit an exponential distribution to survival data and evaluate it on a grid.

    The maximum-likelihood estimator for the exponential rate is
    λ = 1 / mean(x).  The survival function is S(x) = 100 × e^(-λx).

    Args:
        values (np.ndarray): Cleaned positive observations.
        x_grid (np.ndarray): x values at which to evaluate S(x).

    Returns:
        tuple | None: ``("EXP", {"lambda": lam}, y_grid)`` where *y_grid*
            is a float array of survival percentages.  Returns ``None`` if
            fewer than 2 valid values are found.
    """
    v = clean_values(values)
    if len(v) < 2:
        return None
    lam = 1.0 / float(np.mean(v))
    y = 100.0 * np.exp(-lam * x_grid)
    if not LOG_X and FORCE_START_AT_100 and x_grid[0] == 0.0:
        y[0] = 100.0
    return ("EXP", {"lambda": lam}, y)


def _norm_cdf(z):
    """
    Vectorised standard normal CDF using the error function.

    Computes Φ(z) = 0.5 × (1 + erf(z / sqrt(2))) element-wise using
    ``math.erf`` (scalar-safe, no scipy dependency).

    Args:
        z (array-like): Standardised variable(s).

    Returns:
        np.ndarray: CDF values in [0, 1] with the same shape as *z*.
    """
    z = np.asarray(z, dtype=float)
    erf_vec = np.vectorize(math.erf)
    return 0.5 * (1.0 + erf_vec(z / math.sqrt(2.0)))


def _log_tick_formatter(x, pos):
    """Format decade ticks as plain decimal values for log-scaled axes."""
    if x <= 0 or not np.isfinite(x):
        return ""
    return f"{x:g}"


def lognormal_fit(values: np.ndarray, x_grid: np.ndarray):
    """
    Fit a lognormal distribution to survival data and evaluate on a grid.

    Parameters are estimated by maximum likelihood on the log-transformed
    values:
        μ_log = mean(log x),   σ_log = std(log x, ddof=0)

    The survival function is S(x) = 100 × (1 − Φ((log x − μ) / σ)).

    Args:
        values (np.ndarray): Cleaned positive observations.
        x_grid (np.ndarray): x values at which to evaluate S(x).

    Returns:
        tuple | None: ``("LOGNORMAL", {"mu_log": μ, "sigma_log": σ}, y_grid)``
            or ``None`` if fewer than 2 valid values are found.
    """
    v = clean_values(values)
    if len(v) < 2:
        return None
    lv = np.log(v)
    mu = float(np.mean(lv))
    sigma = float(np.std(lv, ddof=0))
    sigma = max(sigma, 1e-12)
    xg = np.array(x_grid, dtype=float)
    xg_safe = np.where(xg <= 0, 1e-12, xg)
    z = (np.log(xg_safe) - mu) / sigma
    y = 100.0 * (1.0 - _norm_cdf(z))
    if not LOG_X and FORCE_START_AT_100 and xg[0] == 0.0:
        y[0] = 100.0
    return ("LOGNORMAL", {"mu_log": mu, "sigma_log": sigma}, y)


def lognormal_from_mean_std(mu_real: float, sigma_real: float, x_grid: np.ndarray):
    """
    Evaluate a lognormal survival curve from real-space mean and standard deviation.

    This is used to overlay a Mauldon-corrected DFN size distribution on the
    empirical trace-length survival plot.  Rather than fitting to the observed
    traces, the parameters come from a bias-corrected estimator.

    The conversion from real-space (μ, σ) to log-space (μ_log, σ_log) follows:
        σ_log = sqrt(log(1 + CoV²))
        μ_log = log(μ) − 0.5 × σ_log²

    Args:
        mu_real (float): Real-space mean of the distribution (e.g. Mauldon
            corrected mean fracture diameter in metres).
        sigma_real (float): Real-space standard deviation.
        x_grid (np.ndarray): x values at which to evaluate S(x).

    Returns:
        np.ndarray: Survival percentage values (shape matching *x_grid*).
    """
    cov2      = (sigma_real / mu_real) ** 2
    sigma_log = float(np.sqrt(np.log1p(cov2)))
    mu_log    = float(np.log(mu_real) - 0.5 * sigma_log ** 2)
    xg        = np.asarray(x_grid, dtype=float)
    xg_safe   = np.where(xg <= 0, 1e-12, xg)
    z         = (np.log(xg_safe) - mu_log) / sigma_log
    y         = 100.0 * (1.0 - _norm_cdf(z))
    if not LOG_X and FORCE_START_AT_100 and xg[0] == 0.0:
        y[0] = 100.0
    return y


def rmse_against_empirical(values: np.ndarray, x_grid: np.ndarray, model_y: np.ndarray):
    """
    Compute the root mean squared error (RMSE) between a model survival curve
    and the empirical survival function evaluated at the observed data points.

    The empirical survival is computed without the force_start_at_100 anchor
    to avoid artificially inflating the goodness-of-fit.  The model values are
    interpolated (in log x space if LOG_X=True) at each empirical observation.

    Args:
        values (np.ndarray): Raw observations used to build the empirical curve.
        x_grid (np.ndarray): The x grid on which *model_y* was evaluated.
        model_y (np.ndarray): Model survival percentages on *x_grid*.

    Returns:
        float | None: RMSE value in percentage points, or ``None`` if the
            empirical curve could not be computed.
    """
    v = clean_values(values)
    x_emp, y_emp = survival_empirical(v, force_start_at_100=False)
    if x_emp is None:
        return None
    if LOG_X:
        y_i = np.interp(np.log(x_emp), np.log(x_grid), model_y)
    else:
        y_i = np.interp(x_emp, x_grid, model_y)
    return float(np.sqrt(np.mean((y_i - y_emp) ** 2)))


def pick_best_model(values: np.ndarray, x_grid: np.ndarray):
    """
    Select the better-fitting survival model between Exponential and Lognormal.

    Fits both models and compares their RMSE against the empirical survival
    function.  The model with the lower RMSE is returned as the best choice.

    Args:
        values (np.ndarray): Cleaned positive observations.
        x_grid (np.ndarray): Evaluation grid.

    Returns:
        tuple | None: ``(model_name, params, y_best, rmse_best, rmse_other)``
            where *model_name* is ``"EXP"`` or ``"LOGNORMAL"``, *params* is the
            corresponding parameter dict, and *rmse_best* / *rmse_other* are
            the RMSE values for the winning / losing model.  Returns ``None``
            if either model cannot be fitted (insufficient data).
    """
    e  = exp_fit(values, x_grid)
    ln = lognormal_fit(values, x_grid)
    if e is None or ln is None:
        return None
    rmse_e  = rmse_against_empirical(values, x_grid, e[2])
    rmse_ln = rmse_against_empirical(values, x_grid, ln[2])
    if rmse_e is None or rmse_ln is None:
        return None
    if rmse_ln < rmse_e:
        return ("LOGNORMAL", ln[1], ln[2], rmse_ln, rmse_e)
    return ("EXP", e[1], e[2], rmse_e, rmse_ln)


# ============================================================
# PARAMETER CONVERSIONS
# ============================================================
def lognormal_mean_std_from_mu_sigma(mu_log: float, sigma_log: float):
    """
    Convert lognormal log-space parameters to real-space mean and standard deviation.

    Uses the standard moment formulas:
        mean = exp(μ + 0.5σ²)
        var  = (exp(σ²) − 1) × exp(2μ + σ²)

    Args:
        mu_log (float): Log-space mean (μ).
        sigma_log (float): Log-space standard deviation (σ).

    Returns:
        tuple[float, float]: ``(mean, std)`` in real space.
    """
    mu = float(mu_log)
    s  = float(sigma_log)
    mean = float(np.exp(mu + 0.5 * s * s))
    var  = float((np.exp(s * s) - 1.0) * np.exp(2.0 * mu + s * s))
    std  = float(np.sqrt(max(var, 0.0)))
    return mean, std


def exp_mean_std_from_lambda(lam: float):
    """
    Compute real-space mean and standard deviation from an exponential rate parameter.

    For an Exponential(λ) distribution:
        mean = std = 1 / λ

    Args:
        lam (float): Rate parameter λ > 0.

    Returns:
        tuple[float, float]: ``(mean, std)`` both equal to 1/λ.
            Returns ``(nan, nan)`` if λ ≤ 0.
    """
    lam = float(lam)
    if lam <= 0:
        return np.nan, np.nan
    m = 1.0 / lam
    return float(m), float(m)


# ============================================================
# MAIN
# ============================================================
def main():
    """
    Run the full trace-length persistence survival analysis for all BC sites.

    Workflow
    --------
    1. **Load data** — reads ``assets/all bc corrected fam.xlsx``, strips
       column names, and validates that the required ``fam`` and ``REGION``
       columns are present.
    2. **Load Mauldon corrections** (optional) — if
       ``assets/mauldon_corrections.csv`` exists, loads it for use as DFN
       reference curves on the per-region plots.
    3. **Per-family plots** — for each fracture family:
       a. Computes persistence values per region (raw Length column, falling
          back to Euclidean endpoint distance if missing).
       b. Applies optional L_MIN/L_MAX physical filters and quantile capping.
       c. Fits both Exponential and Lognormal models via ``pick_best_model()``
          and selects the best-fitting one by RMSE.
       d. If Mauldon corrections are available, overlays a Mauldon-corrected
          lognormal as the DFN reference curve on per-region sub-plots.
       e. Saves one PNG per family to ``outputs/PERSISTENCE/``.
    4. **Combined BC_TOTAL plot** — overlays all families’ TOTAL-pooled best
       fits on a single axes and saves a combined PNG.
    5. **Export summaries** — writes two Excel files:
       - ``persistence_fit_summary_EXP_LOGN.xlsx``: long-format per (family,
         region) table with fitted parameters, RMSE, biased stats, and cap info.
       - ``persistence_fit_summary_WIDE.xlsx``: pivot table with families as
         rows and regions as columns.

    All outputs are written to ``outputs/PERSISTENCE/``.
    """
    os.makedirs(OUT_DIR, exist_ok=True)

    if not os.path.exists(INPUT_FILE):
        raise FileNotFoundError(f"Input file not found: {INPUT_FILE}")

    print(f"Loading: {INPUT_FILE}")
    df = pd.read_csv(INPUT_FILE)
    df.columns = df.columns.str.strip()

    if COL_FAM not in df.columns or COL_REGION not in df.columns:
        raise ValueError(f"Excel must contain columns '{COL_FAM}' and '{COL_REGION}'")

    df[COL_FAM]    = df[COL_FAM].astype(str).str.strip()
    df[COL_REGION] = df[COL_REGION].astype(str).str.strip().str.upper()

    regions  = sorted(df[COL_REGION].dropna().unique().tolist())
    families = sorted(df[COL_FAM].dropna().unique().tolist())
    plot_order = regions + ["TOTAL"]

    print("Regions found:", regions)
    print("Families found:", families)

    # ---- Load Mauldon corrections (optional) ----
    _mauldon_file = os.path.join(SCRIPT_DIR, "assets", "mauldon_corrections.csv")
    _mauldon_df = None
    if os.path.exists(_mauldon_file):
        _mauldon_df = pd.read_csv(_mauldon_file)
        _mauldon_df.columns = _mauldon_df.columns.str.strip()
        # Drop rows with empty mu_nb
        _mauldon_df = _mauldon_df.dropna(subset=["mu_nb"])
        print(f"  Loaded Mauldon corrections ({len(_mauldon_df)} entries): {_mauldon_file}")
    else:
        print(f"  ⚠️  No Mauldon corrections found at: {_mauldon_file}")
        print(f"       Solid curves will use raw fitted models.")

    rows = []

    # ---- per-family plots ----
    for fam in families:
        df_f = df[df[COL_FAM] == fam].copy()

        data = {}
        for region in regions:
            x     = compute_persistence(df_f[df_f[COL_REGION] == region])
            x_raw = get_raw_lengths(df_f[df_f[COL_REGION] == region])
            x = clean_values(x)
            x, cap_used = cap_values_safe(x, CAP_Q, MIN_N_FOR_CAP)
            x = clean_values(x)
            if len(x) >= MIN_N:
                data[region] = {"x": x, "n": int(len(x)), "cap_used": cap_used, "x_raw": x_raw}

        xT     = compute_persistence(df_f)
        xT_raw = get_raw_lengths(df_f)
        xT = clean_values(xT)
        xT, cap_used_T = cap_values_safe(xT, CAP_Q, MIN_N_FOR_CAP)
        xT = clean_values(xT)
        if len(xT) >= MIN_N:
            data["TOTAL"] = {"x": xT, "n": int(len(xT)), "cap_used": cap_used_T, "x_raw": xT_raw}

        if not data:
            print(f"  Skipped {fam}: no valid persistence data")
            continue

        all_x = clean_values(np.concatenate([d["x"] for d in data.values()]))
        if len(all_x) < 2:
            print(f"  Skipped {fam}: not enough valid values")
            continue

        x_min = float(np.min(all_x))
        x_max = float(np.max(all_x))
        if LOG_X:
            eps = max(x_min * 0.1, 1e-6)
            x_grid = np.logspace(np.log10(eps), np.log10(x_max), N_GRID)
        else:
            x0 = 0.0 if FORCE_START_AT_100 else x_min
            x_grid = np.linspace(x0, x_max, N_GRID)

        fig, ax = plt.subplots(figsize=(11, 7), dpi=140)
        plotted_any = False

        for key in plot_order:
            if key not in data:
                continue

            x = data[key]["x"]
            n = data[key]["n"]
            best = pick_best_model(x, x_grid)
            if best is None:
                continue

            model_name, params, y_best, rmse_best, rmse_other = best

            if SMOOTH_EMPIRICAL:
                y_emp_plot = empirical_survival_on_grid(x, x_grid)
                if y_emp_plot is None:
                    x_emp, y_emp = survival_empirical(x, force_start_at_100=FORCE_START_AT_100)
                else:
                    x_emp, y_emp = x_grid, y_emp_plot
            else:
                x_emp, y_emp = survival_empirical(x, force_start_at_100=FORCE_START_AT_100)

            if model_name == "EXP":
                lab = f"{key} EXP (λ={params['lambda']:.3g}, RMSE={rmse_best:.2g}, n={n})"
            else:
                lab = f"{key} LOGN (μ={params['mu_log']:.3g}, σ={params['sigma_log']:.3g}, RMSE={rmse_best:.2g}, n={n})"

            # Biased stats from unfiltered raw lengths (for export + Mauldon lookup)
            x_raw_row = data[key].get("x_raw", np.array([]))
            mu_biased, sigma_biased, cov_biased = compute_biased_stats(x_raw_row)

            if key == "TOTAL":
                ax.plot(x_emp, y_emp, linestyle="--", linewidth=2.0, color="black", alpha=0.45)
                ax.plot(x_grid, y_best, linewidth=4.2, color="black", label=lab)
            else:
                # Use Mauldon-corrected lognormal for solid DFN curve (if available)
                site_key = key.replace("-", "")
                _mu_nb = None
                if _mauldon_df is not None:
                    _match = _mauldon_df[
                        (_mauldon_df["site"].astype(str).str.upper() == site_key.upper()) &
                        (_mauldon_df["fam"].astype(str) == str(fam))
                    ]
                    if not _match.empty:
                        _mu_nb = float(_match.iloc[0]["mu_nb"])

                if _mu_nb is not None and np.isfinite(cov_biased) and _mu_nb > 0:
                    _sigma_nb = cov_biased * _mu_nb
                    y_solid   = lognormal_from_mean_std(_mu_nb, _sigma_nb, x_grid)
                    solid_lab = (f"{key} DFN (Mauldon μ_nb={_mu_nb:.3g} m, "
                                 f"σ_nb={_sigma_nb:.3g} m, n={n})")
                else:
                    y_solid   = y_best
                    solid_lab = lab

                solid, = ax.plot(x_grid, y_solid, linewidth=2.8, label=solid_lab)
                c = solid.get_color()
                ax.plot(x_emp, y_emp, linestyle="--", linewidth=1.6, color=c, alpha=0.45)

            plotted_any = True

            if model_name == "EXP":
                lam = float(params["lambda"])
                mean_fit, std_fit = exp_mean_std_from_lambda(lam)
                mu_log = sigma_log = np.nan
            else:
                mu_log    = float(params["mu_log"])
                sigma_log = float(params["sigma_log"])
                mean_fit, std_fit = lognormal_mean_std_from_mu_sigma(mu_log, sigma_log)
                lam = np.nan

            cap_used = data[key]["cap_used"]
            rows.append({
                "family":        fam,
                "region":        key,
                "n":             n,
                "best_model":    model_name,
                "rmse_best":     rmse_best,
                "rmse_other":    rmse_other,
                "lambda_exp":    lam,
                "mu_log":        mu_log,
                "sigma_log":     sigma_log,
                "mean_fit":      mean_fit,
                "std_fit":       std_fit,
                "mu_biased":     mu_biased,
                "sigma_biased":  sigma_biased,
                "cov_biased":    cov_biased,
                "cap_q":         CAP_Q if CAP_Q is not None else "",
                "cap_applied":   bool(np.isfinite(cap_used)) if cap_used is not None else False,
                "cap_value":     cap_used if (cap_used is not None and np.isfinite(cap_used)) else np.nan,
                "L_min_used":    L_MIN,
                "L_max_used":    L_MAX,
            })

        ax.set_title(f"Occurrence (%) vs Persistence — {fam}\nEmpirical (dashed) + Best fit (solid: Exp or Lognormal)")
        ax.set_xlabel("Persistence / trace length (m)")
        ax.set_ylabel("Occurrence (%) (≥ x)")
        ax.set_ylim(0, 100)
        if LOG_X:
            ax.set_xscale("log")
            ax.xaxis.set_major_locator(LogLocator(base=10.0))
            ax.xaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1))
            ax.xaxis.set_major_formatter(FuncFormatter(_log_tick_formatter))
        ax.grid(True, which="both", alpha=0.25)
        if plotted_any:
            ax.legend()
        fig.tight_layout()

        if plotted_any:
            out_png = os.path.join(OUT_DIR, f"persistence_survival_{fam}.png")
            fig.savefig(out_png, dpi=200, bbox_inches="tight")
            print(f"  ✅ Saved: {out_png}")
        plt.close(fig)

    # ---- combined BC_TOTAL plot ----
    def _fam_sort_key(s):
        """
        Integer sort key for a fracture-family label string.

        Strips whitespace and the ``\"fam\"`` prefix, then converts the
        remaining characters to an integer.  Returns 9999 for labels
        that cannot be parsed numerically (placed at end of sort order).

        Args:
            s: Family label (anything convertible to str).

        Returns:
            int: Sort key; 9999 for unrecognised labels.
        """
        s = str(s).strip().lower().replace("fam", "")
        try:
            return int(float(s))
        except Exception:
            return 9999

    fig2, ax2 = plt.subplots(figsize=(11, 7), dpi=140)
    plotted_any_total = False

    for fam in sorted(families, key=_fam_sort_key):
        df_f = df[df[COL_FAM] == fam].copy()
        xT = compute_persistence(df_f)
        xT = clean_values(xT)
        xT, _ = cap_values_safe(xT, CAP_Q, MIN_N_FOR_CAP)
        xT = clean_values(xT)
        if len(xT) < MIN_N:
            continue

        x_min_T = float(np.min(xT))
        x_max_T = float(np.max(xT))
        if LOG_X:
            eps = max(x_min_T * 0.1, 1e-6)
            x_grid_T = np.logspace(np.log10(eps), np.log10(x_max_T), N_GRID)
        else:
            x0_T = 0.0 if FORCE_START_AT_100 else x_min_T
            x_grid_T = np.linspace(x0_T, x_max_T, N_GRID)

        best = pick_best_model(xT, x_grid_T)
        if best is None:
            continue

        model_name, params, y_best, rmse_best, _ = best

        if SMOOTH_EMPIRICAL:
            y_emp_plot = empirical_survival_on_grid(xT, x_grid_T)
            if y_emp_plot is None:
                x_emp, y_emp = survival_empirical(xT, force_start_at_100=FORCE_START_AT_100)
            else:
                x_emp, y_emp = x_grid_T, y_emp_plot
        else:
            x_emp, y_emp = survival_empirical(xT, force_start_at_100=FORCE_START_AT_100)

        fam_short = str(fam).replace("fam", "F")
        if model_name == "EXP":
            fit_lab = f"{fam_short} EXP-fit"
        else:
            fit_lab = f"{fam_short} LN-fit"

        solid, = ax2.plot(x_grid_T, y_best, linewidth=2.8, label=fit_lab)
        c = solid.get_color()
        ax2.plot(x_emp, y_emp, linestyle="--", linewidth=1.5, color=c, alpha=0.45,
             label=f"{fam_short} empirical")
        plotted_any_total = True

    ax2.set_title("Occurrence (%) vs Persistence — BC_TOTAL\nEmpirical (dashed) + Best fit (solid: Exp or Lognormal)")
    ax2.set_xlabel("Persistence / trace length (m)")
    ax2.set_ylabel("Occurrence (%) (≥ x)")
    ax2.set_ylim(0, 100)
    if LOG_X:
        ax2.set_xscale("log")
        ax2.set_xlim(0.1, 10.0)
        ax2.set_xticks([0.1, 1.0, 10.0])
        ax2.set_xticklabels(["0.1", "1", "10"])
        ax2.xaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1))
    ax2.grid(True, which="both", alpha=0.25)
    if plotted_any_total:
        ax2.legend()
    fig2.tight_layout()

    if plotted_any_total:
        out_total = os.path.join(OUT_DIR, "persistence_survival_BC_TOTAL_all_families.png")
        fig2.savefig(out_total, dpi=200, bbox_inches="tight")
        print(f"✅ Saved: {out_total}")
    plt.close(fig2)

    # ---- summary tables ----
    results_df = pd.DataFrame(rows)
    print(f"\nRows collected: {len(rows)}")

    if len(results_df) > 0:
        out_xlsx = os.path.join(OUT_DIR, "persistence_fit_summary_EXP_LOGN.xlsx")
        results_df.to_excel(out_xlsx, index=False)
        print(f"✅ Saved: {out_xlsx}")

        wide = results_df.pivot_table(
            index="family",
            columns="region",
            values=["best_model", "lambda_exp", "mu_log", "sigma_log", "mean_fit", "std_fit", "n", "rmse_best"],
            aggfunc="first"
        )
        out_wide = os.path.join(OUT_DIR, "persistence_fit_summary_WIDE.xlsx")
        wide.to_excel(out_wide)
        print(f"✅ Saved: {out_wide}")

        print("\n=== SUMMARY ===")
        print(results_df.to_string(index=False))
    else:
        print("\n⚠️  No valid results were generated.")


if __name__ == "__main__":
    main()

import numpy as np


def linear_fit_and_r2(x, y):
    """
    Fit y = a*x + b and return (a, b, R²).
    Returns (None, None, None) if fewer than 2 points.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) < 2:
        return None, None, None

    a, b = np.polyfit(x, y, 1)
    yhat = a * x + b
    ss_res = np.sum((y - yhat) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-16 else np.nan
    return float(a), float(b), float(r2)


def calibrated_p32_from_fit(p21_target, a, b, x_min=None, x_max=None):
    """
    Invert the linear fit P21 = a*P32 + b to find P32 given a P21 target.
    Returns (p32_value, flag) where flag is 'ok' or 'extrapolated'.
    """
    if a is None or abs(a) < 1e-15:
        return None, "slope~0"

    p32 = (p21_target - b) / a

    if x_min is not None and x_max is not None and (p32 < x_min or p32 > x_max):
        return float(p32), "extrapolated"

    return float(p32), "ok"


def update_p32_guess_auto(sweep_df, calibration_df, p32_guess, site_name: str = ""):
    """
    Check sweep coverage and automatically update p32_guess for any family
    whose calibrated P32 falls outside the swept range.
    Returns (new_guess, changed).
    """
    new_guess = np.array(p32_guess, dtype=float).copy()
    changed = False

    sep = "=" * 62
    print(f"\n{sep}")
    print(f"  SWEEP COVERAGE CHECK{' — ' + site_name if site_name else ''}")
    print(f"{sep}")

    for _, cal_row in calibration_df.iterrows():
        fam_id   = int(cal_row["fam"])
        fam_name = str(cal_row.get("fam_name", f"fam{fam_id}"))
        p32_cal  = float(cal_row["P32_calibrated"])
        flag     = str(cal_row.get("calibration_flag", ""))

        sub      = sweep_df[sweep_df["fam"] == fam_id]
        p32_min  = float(sub["P32_obt"].min())
        p32_max  = float(sub["P32_obt"].max())

        capping_suspected = False
        if "P32_target" in sub.columns:
            p32_target_max = float(sub["P32_target"].max())
            if p32_target_max > 0 and p32_max < 0.85 * p32_target_max:
                capping_suspected = True

        covered = p32_min <= p32_cal <= p32_max
        status  = "✅  ok" if (covered and "ok" in flag.lower()) else f"⚠️   {flag}"

        print(f"  {fam_name:6s}: sweep=[{p32_min:.5f}, {p32_max:.5f}]  "
              f"cal={p32_cal:.5f}  {status}")

        if not covered and np.isfinite(p32_cal) and p32_cal > 0:
            direction = "< sweep_min" if p32_cal < p32_min else "> sweep_max"
            idx = fam_id - 1
            # map fam_id to array index safely
            fam_ids = sorted(calibration_df["fam"].astype(int).tolist())
            if fam_id in fam_ids:
                idx = fam_ids.index(fam_id)
            print(f"           → auto-updating P32_guess[{idx}] "
                  f"{new_guess[idx]:.6f} → {p32_cal:.6f}  "
                  f"(cal {direction}; centring sweep on calibrated value)")
            new_guess[idx] = p32_cal
            changed = True

        if capping_suspected:
            print(f"           → ⚠️  P32_obt_max ({p32_max:.5f}) << P32_target_max "
                  f"({p32_target_max:.5f}): max_fracs may be limiting the sweep.")

    if not changed:
        print("\n  ✅  All families within sweep range — calibration is valid!")
    else:
        print(f"\n  → P32_guess updated to: {new_guess}  (re-running sweep...)")
    print(f"{sep}\n")

    return new_guess, changed
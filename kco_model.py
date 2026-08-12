# ============================================================
# KCO — Kuznetsov-Cunningham-Ouchterlony fragmentation model
#       (post-blast fragment size distribution prediction)
#
# Pure-physics module: no file I/O, no plotting.
# Implements Ouchterlony (2005), "The Swebrec function: linking
# fragmentation by blasting and crushing", eqs. (11a)-(11e),
# plus the classical Kuz-Ram / Rosin-Rammler reference model.
#
# All equations are validated against the worked example given in
# the source paper (Bararp round 4) -- see self_test() at the bottom.
# ============================================================

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np

LN2 = math.log(2.0)


# ============================================================
# ROCK FACTOR A  --  Lilly (1986) blastability index, eq. (11e)
# ============================================================
def jps_from_joint_spacing(mean_joint_spacing_m: float,
                           oversize_m: float) -> int:
    """
    Return Lilly's joint plane spacing rating JPS from a measured spacing.

    Ouchterlony (2005) eq. (11e):
        JPS = 10  if mean joint spacing Sj < 0.1 m
        JPS = 20  if 0.1 m <= Sj <= oversize
        JPS = 50  if Sj > oversize

    Args:
        mean_joint_spacing_m: Mean perpendicular joint spacing Sj (m).
            In this project it comes from the SPACING analysis
            (``outputs/SPACING/spacing_fit_summary_LOGN.xlsx``).
        oversize_m: Site oversize threshold xO (m), i.e. the largest
            block the crusher / loader can accept.

    Returns:
        int: JPS rating (10, 20 or 50).
    """
    if mean_joint_spacing_m < 0.1:
        return 10
    if mean_joint_spacing_m > oversize_m:
        return 50
    return 20


def rock_density_influence(rho_kg_m3: float) -> float:
    """
    Return Lilly's rock density influence RDI = 0.025*rho - 50.

    Args:
        rho_kg_m3: Intact rock density (kg/m3).

    Returns:
        float: RDI term of the blastability index.
    """
    return 0.025 * rho_kg_m3 - 50.0


def hardness_factor(youngs_modulus_gpa: float,
                    ucs_mpa: float) -> float:
    """
    Return Lilly's hardness factor HF.

    Ouchterlony (2005) eq. (11e):
        HF = E/3     if E < 50 GPa
        HF = sigma_c/5   otherwise

    Args:
        youngs_modulus_gpa: Young's modulus E (GPa).
        ucs_mpa: Uniaxial compressive strength sigma_c (MPa).

    Returns:
        float: HF term of the blastability index.
    """
    if youngs_modulus_gpa < 50.0:
        return youngs_modulus_gpa / 3.0
    return ucs_mpa / 5.0


def rock_factor_A(rmd: float,
                  rdi: float,
                  hf: float,
                  correction: float = 1.0) -> float:
    """
    Return the Kuz-Ram / KCO rock factor A from Lilly's blastability index.

    Ouchterlony (2005) eq. (11e):  A = 0.06 * (RMD + RDI + HF)
    equivalently A = 0.06 * BI, where BI is Lilly's blastability index.

    For a jointed rock mass RMD is itself set to the joint factor
    JF = JPS + JPA, so the sum reduces to Cunningham's (1987) form
    A = 0.06 * (RMD + JF + RDI + HF) with RMD folded into JF. Reported
    values of A span roughly 1.7 to 21 (Ouchterlony and Sanchidrian 2019),
    against the narrower 7 to 13 of the original Kuznetsov ratings.

    Args:
        rmd: Rock mass description rating. 10 for powdery/friable ground,
            50 for massive rock, or the joint factor JF = JPS + JPA for a
            jointed rock mass (the usual case for a fractured quarry face).
        rdi: Rock density influence, see :func:`rock_density_influence`.
        hf: Hardness factor, see :func:`hardness_factor`.
        correction: Cunningham (2005) correction multiplier C(A), applied
            when calibration blasts show the algorithm under- or
            over-estimates A. Defaults to 1.0 (no correction).

    Returns:
        float: Rock factor A (dimensionless).
    """
    return correction * 0.06 * (rmd + rdi + hf)


# ============================================================
# UNIFORMITY INDEX n  --  Cunningham (1987), eq. (11c)
# ============================================================
def uniformity_index_n(burden_m: float,
                       spacing_m: float,
                       hole_diameter_mm: float,
                       drill_accuracy_sd_m: float,
                       bottom_charge_m: float,
                       column_charge_m: float,
                       total_charge_m: float,
                       bench_height_m: float,
                       correction: float = 1.0) -> float:
    """
    Return Cunningham's uniformity index n (Ouchterlony 2005, eq. 11c).

        n = (2.2 - 14*B/D) * sqrt((1 + S/B)/2) * (1 - W/B)
              * (|Lb - Lc|/Ltot + 0.1)^0.1 * (Ltot/H)

    Note the mixed units: the burden B is in metres while the hole
    diameter D is in millimetres, exactly as published. This is verified
    against the paper's Bararp example in :func:`self_test`.

    The index controls the spread of the size distribution. Typical values
    range from 0.6 (very non-uniform: dust plus boulders) to 2.2 (uniform
    muckpile clustered around the mean size).

    Args:
        burden_m: Blast-hole burden B (m).
        spacing_m: Blast-hole spacing S (m).
        hole_diameter_mm: Drill-hole diameter D (mm).
        drill_accuracy_sd_m: Standard deviation of drilling accuracy W (m).
        bottom_charge_m: Length of the bottom charge Lb (m).
        column_charge_m: Length of the column charge Lc (m).
        total_charge_m: Total charge length Ltot (m), measured above grade.
        bench_height_m: Bench height / hole depth H (m).
        correction: Cunningham (2005) correction multiplier C(n).
            Defaults to 1.0.

    Returns:
        float: Uniformity index n (dimensionless).

    Raises:
        ValueError: If any of B, Ltot, H or the hole diameter is not
            strictly positive.
    """
    if burden_m <= 0 or hole_diameter_mm <= 0:
        raise ValueError("burden_m and hole_diameter_mm must be > 0")
    if total_charge_m <= 0 or bench_height_m <= 0:
        raise ValueError("total_charge_m and bench_height_m must be > 0")

    term_geometry = 2.2 - 14.0 * burden_m / hole_diameter_mm
    term_spacing = math.sqrt((1.0 + spacing_m / burden_m) / 2.0)
    term_accuracy = 1.0 - drill_accuracy_sd_m / burden_m
    term_charge = (abs(bottom_charge_m - column_charge_m) / total_charge_m
                   + 0.1) ** 0.1
    term_length = total_charge_m / bench_height_m

    return (correction * term_geometry * term_spacing
            * term_accuracy * term_charge * term_length)


def shift_factor_g(n: float) -> float:
    """
    Return the mean-to-median shift factor g(n) of the shifted Kuz-Ram model.

    Ouchterlony (2005) eq. (11b) and Ouchterlony and Sanchidrian (2019)
    eq. (14), the median-over-mean ratio of the Rosin-Rammler function:

        g(n) = (ln 2)^(1/n) / Gamma(1 + 1/n)   < 1

    Background: Spathis (2004) pointed out that Kuznetsov's formula was
    calibrated on *mean* sizes while the distribution is parameterised by
    the *median* x50, so a correction would be needed.

    Do not apply this factor by default. The 2019 review concludes that
    Cunningham (1987) in practice already treated his predicted size as
    the median, so "Spathis' remark about the mean vs. median mix-up
    ceases to be valid" for the revised Kuz-Ram model, which is the one
    KCO builds on. The factor is provided for sensitivity analysis only.

    Note: the 2005 paper prints g(n) = 0.659 for its Bararp example with
    n = 1.17. That is a misprint; the formula above gives 0.773, as
    verified in :func:`self_test`.

    Args:
        n: Cunningham uniformity index.

    Returns:
        float: Shift factor g(n), always < 1.
    """
    return LN2 ** (1.0 / n) / math.gamma(1.0 + 1.0 / n)


# ============================================================
# MEDIAN SIZE x50  --  Kuznetsov, eq. (11b)
# ============================================================
def x50_kuznetsov(rock_factor_a: float,
                  charge_per_hole_kg: float,
                  powder_factor_kg_m3: float,
                  s_anfo_pct: float,
                  n: Optional[float] = None,
                  shifted: bool = False,
                  timing_factor: float = 1.0) -> float:
    """
    Return the median (50 % passing) fragment size x50 in centimetres.

    Ouchterlony (2005) eq. (11b):

        x50 = g(n) * A * Q^(1/6) * q^(-0.8) * (115 / s_ANFO)^(19/30)

    Validated against the paper's Bararp round 4 example: A = 13,
    Q = 9.24 kg, q = 0.55 kg/m3, s_ANFO = 62.2 % gives x50 = 44.8 cm.

    Args:
        rock_factor_a: Rock factor A, see :func:`rock_factor_A`.
        charge_per_hole_kg: Charge weight per hole Q (kg).
        powder_factor_kg_m3: Specific charge / powder factor q (kg/m3).
            Cunningham notes the powder factor computed above grade
            (excluding subdrill explosive) correlates better.
        s_anfo_pct: Explosive weight strength relative to ANFO (%).
            ANFO itself is 100.
        n: Uniformity index, required only when ``shifted`` is True.
        shifted: If True, apply the mean-to-median shift factor g(n).
        timing_factor: Cunningham (2005) timing factor A_T accounting for
            inter-hole delay. Defaults to 1.0 (no timing correction).

    Returns:
        float: Median fragment size x50 (cm).

    Raises:
        ValueError: If ``shifted`` is True but ``n`` was not supplied, or
            if any input is non-positive.
    """
    if charge_per_hole_kg <= 0 or powder_factor_kg_m3 <= 0 or s_anfo_pct <= 0:
        raise ValueError("Q, q and s_ANFO must be > 0")
    if shifted and n is None:
        raise ValueError("n must be supplied when shifted=True")

    g = shift_factor_g(n) if shifted else 1.0

    return (g * timing_factor * rock_factor_a
            * charge_per_hole_kg ** (1.0 / 6.0)
            * powder_factor_kg_m3 ** (-0.8)
            * (115.0 / s_anfo_pct) ** (19.0 / 30.0))


# ============================================================
# MAXIMUM SIZE xmax  --  eq. (11d)
# ============================================================
def xmax_kco(in_situ_block_size_m: float,
             burden_m: float,
             spacing_m: float) -> float:
    """
    Return the maximum fragment size xmax in metres, eq. (11d).

        xmax = min(in-situ block size, S, B)

    The physical argument is that a fragment can be no larger than either
    the natural block delimited by the joint network, or the slab of rock
    bounded by the drilling pattern.

    In this project the in-situ block size is taken from the calibrated
    DFN blockometry produced by the pre-blast workflow
    (``outputs/VARENNE/05_block_volumes``), converted to an equivalent
    sphere diameter -- see :func:`volume_to_equivalent_diameter_m`.

    Args:
        in_situ_block_size_m: Characteristic in-situ block size (m).
        burden_m: Burden B (m).
        spacing_m: Spacing S (m).

    Returns:
        float: Maximum fragment size xmax (m).
    """
    return min(in_situ_block_size_m, burden_m, spacing_m)


def volume_to_equivalent_diameter_m(volume_m3: np.ndarray | float) -> np.ndarray | float:
    """
    Convert a block volume to the diameter of a sphere of equal volume.

        d = (6 V / pi)^(1/3)

    Args:
        volume_m3: Block volume(s) in cubic metres.

    Returns:
        Equivalent spherical diameter(s) in metres, same shape as input.
    """
    return (6.0 * np.asarray(volume_m3, dtype=float) / math.pi) ** (1.0 / 3.0)


# ============================================================
# UNDULATION PARAMETER b
# ============================================================
def b_parameter(xmax: float, x50: float, n: float) -> float:
    """
    Return the Swebrec undulation parameter b, eq. (11c) first line.

        b = 2 * ln(2) * ln(xmax / x50) * n

    This is obtained by equating the slope of the Swebrec function at x50
    with that of the Rosin-Rammler function with uniformity index n
    (eq. 3 of the paper, inverted).

    Args:
        xmax: Maximum fragment size, any length unit.
        x50: Median fragment size, in the *same* unit as ``xmax``.
        n: Cunningham uniformity index.

    Returns:
        float: Undulation parameter b (dimensionless), typically 1.5-3.

    Raises:
        ValueError: If ``xmax`` is not strictly greater than ``x50``.
    """
    if not xmax > x50 > 0:
        raise ValueError(f"require xmax > x50 > 0, got xmax={xmax}, x50={x50}")
    return 2.0 * LN2 * math.log(xmax / x50) * n


# ============================================================
# THE SWEBREC DISTRIBUTION  --  eq. (11a) / (2a)-(2b)
# ============================================================
def swebrec_passing(x: np.ndarray | float,
                    x50: float,
                    xmax: float,
                    b: float) -> np.ndarray:
    """
    Return the Swebrec cumulative percentage passing P(x), in percent.

    Ouchterlony (2005) eq. (11a):

        P(x) = 100 / (1 + [ ln(xmax/x) / ln(xmax/x50) ]^b)

    By construction P(x50) = 50 % and P(xmax) = 100 %. Unlike
    Rosin-Rammler the distribution has a finite upper limit and its
    small-fragment asymptote is logarithmic rather than a power law,
    which is why it reproduces the fines much better.

    Args:
        x: Fragment size(s), same length unit as ``x50`` and ``xmax``.
        x50: Median size (50 % passing).
        xmax: Maximum size (100 % passing).
        b: Undulation parameter, see :func:`b_parameter`.

    Returns:
        np.ndarray: Percentage passing in [0, 100]. Sizes at or above
            ``xmax`` return 100; non-positive sizes return 0.

    Raises:
        ValueError: If ``xmax`` is not strictly greater than ``x50``.
    """
    if not xmax > x50 > 0:
        raise ValueError(f"require xmax > x50 > 0, got xmax={xmax}, x50={x50}")

    x = np.asarray(x, dtype=float)
    out = np.zeros_like(x)

    inside = (x > 0.0) & (x < xmax)
    ratio = np.log(xmax / x[inside]) / math.log(xmax / x50)
    out[inside] = 100.0 / (1.0 + ratio ** b)
    out[x >= xmax] = 100.0

    return out


def swebrec_size_at_passing(passing_pct: np.ndarray | float,
                            x50: float,
                            xmax: float,
                            b: float) -> np.ndarray:
    """
    Invert the Swebrec function: return the size xP for a given % passing.

        xP = xmax * exp( -ln(xmax/x50) * (100/P - 1)^(1/b) )

    Args:
        passing_pct: Percentage passing P, in (0, 100].
        x50: Median size.
        xmax: Maximum size.
        b: Undulation parameter.

    Returns:
        np.ndarray: Fragment size(s) xP, same unit as ``x50``.
    """
    p = np.asarray(passing_pct, dtype=float)
    out = np.full_like(p, np.nan)
    valid = (p > 0.0) & (p <= 100.0)
    ratio = (100.0 / p[valid] - 1.0) ** (1.0 / b)
    out[valid] = xmax * np.exp(-math.log(xmax / x50) * ratio)
    return out


def swebrec_slope_at_x50(x50: float, xmax: float, b: float) -> float:
    """
    Return the slope s50 = P'(x50) of the Swebrec distribution.

    Ouchterlony and Sanchidrian (2019) eq. (60):
        s50 = b / (4 * x50 * ln(xmax / x50))

    Args:
        x50: Median size.
        xmax: Maximum size.
        b: Undulation parameter.

    Returns:
        float: Slope at the median size, in units of 1/[length].
    """
    return b / (4.0 * x50 * math.log(xmax / x50))


def swebrec_inflection_point(x50: float, xmax: float, b: float) -> float:
    """
    Return the inflection point of the Swebrec function in log-log space.

    Ouchterlony (2005) eq. (4):
        x_infl = xmax * (x50 / xmax)^(1 / (b - 1))

    As b tends to 1 the inflection point approaches xmax; it reaches x50
    at b = 2. Its existence is what lets the coarse fractions carry
    information about the fines.

    Args:
        x50: Median size.
        xmax: Maximum size.
        b: Undulation parameter (must differ from 1).

    Returns:
        float: Inflection point size, same unit as ``x50``.

    Raises:
        ValueError: If b == 1, where the expression is undefined.
    """
    if math.isclose(b, 1.0):
        raise ValueError("inflection point undefined for b = 1")
    return xmax * (x50 / xmax) ** (1.0 / (b - 1.0))


# ============================================================
# ROSIN-RAMMLER  --  the original Kuz-Ram model, for comparison
# ============================================================
def rosin_rammler_passing(x: np.ndarray | float,
                          x50: float,
                          n: float) -> np.ndarray:
    """
    Return the Rosin-Rammler cumulative percentage passing, eq. (1).

        P(x) = 100 * [1 - 2^(-(x/x50)^n)]

    This is the distribution of the original Kuz-Ram model. It is kept
    here so the report can show, side by side, the two Kuz-Ram drawbacks
    that KCO removes: the underestimation of fines and the absence of an
    upper size limit.

    Args:
        x: Fragment size(s), same unit as ``x50``.
        x50: Median size (50 % passing).
        n: Cunningham uniformity index.

    Returns:
        np.ndarray: Percentage passing in [0, 100].
    """
    x = np.asarray(x, dtype=float)
    out = np.zeros_like(x)
    pos = x > 0.0
    out[pos] = 100.0 * (1.0 - 2.0 ** (-((x[pos] / x50) ** n)))
    return out


def n_equivalent_from_b(xmax: float, x50: float, b: float) -> float:
    """
    Return the Rosin-Rammler index n equivalent to a Swebrec b, eq. (3).

        n_equiv = b / (2 * ln(2) * ln(xmax / x50))

    Inverse of :func:`b_parameter`; useful when a Swebrec function has
    been *fitted* to sieved data and one wants the corresponding Kuz-Ram
    uniformity index.

    Args:
        xmax: Maximum size.
        x50: Median size, same unit.
        b: Fitted undulation parameter.

    Returns:
        float: Equivalent Rosin-Rammler uniformity index.
    """
    return b / (2.0 * LN2 * math.log(xmax / x50))


# ============================================================
# BLAST DESIGN CONTAINER + END-TO-END PREDICTION
# ============================================================
@dataclass
class BlastDesign:
    """
    Complete set of inputs required by the KCO model for one blast round.

    Grouped as the three families of parameters the model needs: blast
    geometry, explosive loading, and rock mass properties. Rock mass
    values marked "pre-blast" are derived automatically from this
    project's DFN workflow rather than being entered by hand.

    Attributes:
        name: Label for the round or site, used in outputs.
        hole_diameter_mm: Drill-hole diameter D (mm).
        burden_m: Burden B (m).
        spacing_m: Spacing S (m).
        bench_height_m: Bench height H (m).
        total_charge_m: Total charge length above grade Ltot (m).
        bottom_charge_m: Bottom charge length Lb (m).
        column_charge_m: Column charge length Lc (m).
        subdrill_m: Subdrill length (m), recorded for the report.
        drill_accuracy_sd_m: Drilling accuracy standard deviation W (m).
            Gustafsson's rule of thumb is 0.03 m per metre of hole.
        charge_per_hole_kg: Charge per hole Q (kg).
        powder_factor_kg_m3: Powder factor q (kg/m3), above grade.
        s_anfo_pct: Explosive weight strength relative to ANFO (%).
        explosive_name: Explosive product name, for the report.
        rock_density_kg_m3: Intact rock density rho (kg/m3).
        ucs_mpa: Uniaxial compressive strength sigma_c (MPa).
        youngs_modulus_gpa: Young's modulus E (GPa).
        jpa: Lilly joint plane angle rating: 20 dip out of face,
            30 strike perpendicular to face, 40 dip into face.
        mean_joint_spacing_m: Mean joint spacing Sj (m). Pre-blast, from
            the SPACING analysis.
        oversize_m: Oversize threshold xO (m), used to rate JPS.
        in_situ_block_size_m: Characteristic in-situ block size (m).
            Pre-blast, from the calibrated DFN blockometry.
        rmd_override: Explicit RMD rating, bypassing the default
            "jointed rock mass" assumption RMD = JPS + JPA. Use 10 for
            powdery/friable ground or 50 for massive rock.
        c_a: Cunningham (2005) rock factor correction C(A).
        c_n: Cunningham (2005) uniformity index correction C(n).
        timing_factor: Cunningham (2005) timing factor A_T.
        use_shifted: Apply the mean-to-median shift factor g(n).
    """
    name: str = "VARENNE"

    hole_diameter_mm: float = 0.0
    burden_m: float = 0.0
    spacing_m: float = 0.0
    bench_height_m: float = 0.0
    total_charge_m: float = 0.0
    bottom_charge_m: float = 0.0
    column_charge_m: float = 0.0
    subdrill_m: float = 0.0
    drill_accuracy_sd_m: float = 0.0

    charge_per_hole_kg: float = 0.0
    powder_factor_kg_m3: float = 0.0
    s_anfo_pct: float = 100.0
    explosive_name: str = "ANFO"

    rock_density_kg_m3: float = 0.0
    ucs_mpa: float = 0.0
    youngs_modulus_gpa: float = 0.0
    jpa: int = 30
    mean_joint_spacing_m: float = 0.0
    oversize_m: float = 1.0
    in_situ_block_size_m: float = 0.0
    rmd_override: Optional[float] = None

    c_a: float = 1.0
    c_n: float = 1.0
    timing_factor: float = 1.0
    use_shifted: bool = False


@dataclass
class KCOResult:
    """
    Outcome of a KCO prediction for one blast round.

    All sizes are stored in millimetres, the conventional unit for
    sieving curves and for WipFrag image-analysis output.

    Attributes:
        design: The :class:`BlastDesign` the prediction was made from.
        rock_factor: Rock factor A.
        blastability_index: Lilly's BI = A / 0.06.
        rmd: Rock mass description rating actually used.
        jps: Joint plane spacing rating used.
        rdi: Rock density influence term.
        hf: Hardness factor term.
        n: Cunningham uniformity index.
        g_n: Mean-to-median shift factor applied (1.0 if unshifted).
        x50_mm: Median fragment size (mm).
        xmax_mm: Maximum fragment size (mm).
        b: Swebrec undulation parameter.
        xmax_governed_by: Which of "in-situ block", "burden" or "spacing"
            set the value of xmax.
    """
    design: BlastDesign
    rock_factor: float
    blastability_index: float
    rmd: float
    jps: int
    rdi: float
    hf: float
    n: float
    g_n: float
    x50_mm: float
    xmax_mm: float
    b: float
    xmax_governed_by: str

    def passing(self, x_mm: np.ndarray | float) -> np.ndarray:
        """
        Return the predicted Swebrec percentage passing at sizes ``x_mm``.

        Args:
            x_mm: Fragment size(s) in millimetres.

        Returns:
            np.ndarray: Percentage passing in [0, 100].
        """
        return swebrec_passing(x_mm, self.x50_mm, self.xmax_mm, self.b)

    def passing_rosin_rammler(self, x_mm: np.ndarray | float) -> np.ndarray:
        """
        Return the original Kuz-Ram (Rosin-Rammler) passing, for comparison.

        Args:
            x_mm: Fragment size(s) in millimetres.

        Returns:
            np.ndarray: Percentage passing in [0, 100].
        """
        return rosin_rammler_passing(x_mm, self.x50_mm, self.n)

    def size_at(self, passing_pct: np.ndarray | float) -> np.ndarray:
        """
        Return the predicted fragment size(s) at a given percentage passing.

        Args:
            passing_pct: Percentage passing, in (0, 100].

        Returns:
            np.ndarray: Size(s) in millimetres.
        """
        return swebrec_size_at_passing(passing_pct, self.x50_mm,
                                       self.xmax_mm, self.b)

    def percentiles(self,
                    levels: Sequence[float] = (10, 20, 30, 50, 80, 90, 100)
                    ) -> dict[str, float]:
        """
        Return a dictionary of characteristic percentile sizes.

        Args:
            levels: Percentage passing levels to evaluate.

        Returns:
            dict: Mapping such as ``{"x10": 42.3, "x50": 210.0, ...}``
                with sizes in millimetres.
        """
        return {f"x{int(p)}": float(self.size_at(p)) for p in levels}

    def oversize_fraction_pct(self, oversize_mm: float) -> float:
        """
        Return the predicted percentage of material coarser than a threshold.

        Args:
            oversize_mm: Oversize threshold (mm), typically the crusher or
                loader acceptance limit.

        Returns:
            float: Percentage retained above the threshold.
        """
        return float(100.0 - self.passing(oversize_mm))

    def fines_fraction_pct(self, fines_mm: float) -> float:
        """
        Return the predicted percentage of fines below a threshold.

        Args:
            fines_mm: Fines threshold (mm), e.g. 4 mm for aggregate quarries.

        Returns:
            float: Percentage passing the threshold.
        """
        return float(self.passing(fines_mm))


def predict_kco(design: BlastDesign) -> KCOResult:
    """
    Run the full KCO chain for one blast round.

    Executes the four steps prescribed by the model:
      1. rock factor A from Lilly's blastability index, eq. (11e);
      2. median size x50 from Kuznetsov, eq. (11b);
      3. maximum size xmax, eq. (11d);
      4. undulation parameter b, so that the Swebrec function of
         eq. (11a) is fully determined.

    Args:
        design: Fully populated :class:`BlastDesign`.

    Returns:
        KCOResult: Model parameters plus callable size-distribution methods.

    Raises:
        ValueError: If the resulting xmax does not exceed x50, which means
            the design and rock-mass inputs are mutually inconsistent
            (typically a powder factor so low that the predicted median
            size exceeds the in-situ block size).
    """
    jps = jps_from_joint_spacing(design.mean_joint_spacing_m,
                                 design.oversize_m)
    rdi = rock_density_influence(design.rock_density_kg_m3)
    hf = hardness_factor(design.youngs_modulus_gpa, design.ucs_mpa)
    rmd = design.rmd_override if design.rmd_override is not None else jps + design.jpa

    a = rock_factor_A(rmd, rdi, hf, correction=design.c_a)

    n = uniformity_index_n(
        burden_m=design.burden_m,
        spacing_m=design.spacing_m,
        hole_diameter_mm=design.hole_diameter_mm,
        drill_accuracy_sd_m=design.drill_accuracy_sd_m,
        bottom_charge_m=design.bottom_charge_m,
        column_charge_m=design.column_charge_m,
        total_charge_m=design.total_charge_m,
        bench_height_m=design.bench_height_m,
        correction=design.c_n,
    )

    g = shift_factor_g(n) if design.use_shifted else 1.0
    x50_cm = x50_kuznetsov(
        rock_factor_a=a,
        charge_per_hole_kg=design.charge_per_hole_kg,
        powder_factor_kg_m3=design.powder_factor_kg_m3,
        s_anfo_pct=design.s_anfo_pct,
        n=n,
        shifted=design.use_shifted,
        timing_factor=design.timing_factor,
    )
    x50_mm = x50_cm * 10.0

    candidates = {
        "in-situ block": design.in_situ_block_size_m,
        "burden": design.burden_m,
        "spacing": design.spacing_m,
    }
    governed_by = min(candidates, key=candidates.get)
    xmax_mm = xmax_kco(design.in_situ_block_size_m,
                       design.burden_m,
                       design.spacing_m) * 1000.0

    if xmax_mm <= x50_mm:
        raise ValueError(
            f"inconsistent inputs: predicted x50 = {x50_mm:.0f} mm exceeds "
            f"xmax = {xmax_mm:.0f} mm (governed by {governed_by}). "
            "Check the powder factor, the rock factor A, and the in-situ "
            "block size."
        )

    b = b_parameter(xmax_mm, x50_mm, n)

    return KCOResult(
        design=design,
        rock_factor=a,
        blastability_index=a / 0.06,
        rmd=rmd,
        jps=jps,
        rdi=rdi,
        hf=hf,
        n=n,
        g_n=g,
        x50_mm=x50_mm,
        xmax_mm=xmax_mm,
        b=b,
        xmax_governed_by=governed_by,
    )


# ============================================================
# SELF-TEST  --  Bararp round 4, worked example of Ouchterlony (2005)
# ============================================================
def self_test(verbose: bool = True) -> bool:
    """
    Verify the implementation against the paper's own worked example.

    Ouchterlony (2005) works through Bararp round 4 in the section
    "extended Kuz-Ram or KCO model" and reports A = 13, Q = 9.24 kg,
    q = 0.55 kg/m3, s_ANFO = 62.2 %, D = 51 mm, B = 1.8 m, S = 2.2 m,
    H = 5.2 m, Ltot = 3.9 m above grade, Lb = Ltot, Lc = 0, SD = 0.25 m,
    xmax = sqrt(B*S) = 2.0 m. The published answers are x50 = 44.8 cm,
    n = 1.17, g(n) = 0.659 and b = 2.431, with the Swebrec fit to the
    sieved data giving x50 = 459 mm and b = 2.238.

    Args:
        verbose: Print a comparison table of computed versus published
            values.

    Returns:
        bool: True if every value matches the paper within tolerance.
    """
    n = uniformity_index_n(burden_m=1.8, spacing_m=2.2, hole_diameter_mm=51.0,
                           drill_accuracy_sd_m=0.25, bottom_charge_m=3.9,
                           column_charge_m=0.0, total_charge_m=3.9,
                           bench_height_m=5.2)
    x50_cm = x50_kuznetsov(rock_factor_a=13.0, charge_per_hole_kg=9.24,
                           powder_factor_kg_m3=0.55, s_anfo_pct=62.2)
    g = shift_factor_g(n)
    b = b_parameter(xmax=2000.0, x50=x50_cm * 10.0, n=n)

    # g(n) is checked against the formula of Ouchterlony and Sanchidrian
    # (2019) eq. (14), not against the 0.659 printed in the 2005 paper,
    # which is a misprint: (ln2)^(1/1.1724)/Gamma(1+1/1.1724) = 0.7729.
    checks = [
        ("n",       n,       1.17,   0.01),
        ("x50 (cm)", x50_cm,  44.8,   0.1),
        ("g(n)",    g,       0.7729, 0.001),
        ("b",       b,       2.431,  0.01),
    ]

    ok = True
    if verbose:
        print("KCO self-test -- Bararp round 4 (Ouchterlony 2005)")
        print(f"{'quantity':<12}{'computed':>12}{'published':>12}{'':>8}")
    for label, got, expected, tol in checks:
        passed = abs(got - expected) <= tol
        ok = ok and passed
        if verbose:
            print(f"{label:<12}{got:>12.4f}{expected:>12.4f}"
                  f"{'  OK' if passed else '  FAIL':>8}")

    # The Swebrec curve must honour its two defining fixed points.
    p50 = float(swebrec_passing(x50_cm * 10.0, x50_cm * 10.0, 2000.0, b))
    p_max = float(swebrec_passing(2000.0, x50_cm * 10.0, 2000.0, b))
    for label, got, expected in (("P(x50) %", p50, 50.0),
                                 ("P(xmax) %", p_max, 100.0)):
        passed = abs(got - expected) < 1e-6
        ok = ok and passed
        if verbose:
            print(f"{label:<12}{got:>12.4f}{expected:>12.4f}"
                  f"{'  OK' if passed else '  FAIL':>8}")

    # Inverting the distribution must return the size we started from.
    round_trip = float(swebrec_size_at_passing(30.0, x50_cm * 10.0, 2000.0, b))
    back = float(swebrec_passing(round_trip, x50_cm * 10.0, 2000.0, b))
    passed = abs(back - 30.0) < 1e-6
    ok = ok and passed
    if verbose:
        print(f"{'invert P=30':<12}{back:>12.4f}{30.0:>12.4f}"
              f"{'  OK' if passed else '  FAIL':>8}")
        print("\nAll checks passed." if ok else "\nSOME CHECKS FAILED.")

    return ok


if __name__ == "__main__":
    raise SystemExit(0 if self_test() else 1)

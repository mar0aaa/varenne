# ============================================================
# KCO* — probabilistic extension of the deterministic KCO model
#
# THIS MODULE DOES NOT MODIFY kco_model.py. Classical/deterministic KCO
# remains exactly as implemented there and is the reference baseline.
#
# KCO* modelling hypothesis (NOT a published Cunningham/Ouchterlony
# equation):
#
#   Sj*_i = V_i ^ (1/3)     for every DFN in-situ block volume V_i
#
# Instead of reducing the whole DFN block-volume distribution to one
# representative joint spacing Sj (as classical KCO does), KCO*
# propagates the probabilistic block structure through the KCO chain:
#
#   DFN block volumes (S x B x H blast-cell sub-domain)
#         -> Sj* = V^(1/3) for every block               [KCO* hypothesis]
#         -> percentile classes of the Sj* distribution   [KCO* hypothesis]
#         -> representative Sj_i per class
#         -> published KCO equations run independently per class
#            (jps_from_joint_spacing, joint_factor, rock_mass_description,
#            rock_density_influence, hardness_factor, blastability_index,
#            rock_factor_A, uniformity_index_n, shift_factor_g,
#            x50_kuznetsov, xmax_kco, b_parameter, swebrec_passing)
#         -> P_i(x) : one Swebrec curve per class
#         -> P_KCO*(x) = sum_i w_i * P_i(x)                [KCO* hypothesis:
#            probability-weighted mixture of class curves; NOT a published
#            Cunningham/Ouchterlony combination rule]
#
# Every equation imported from kco_model.py is used unmodified. Every
# step introduced specifically for KCO* is explicitly labelled
# "KCO* modelling assumption" in its docstring.
#
# Xmax is a SEPARATE issue from Sj* (see kco_model.xmax_kco docstring):
# it is computed ONCE from a high percentile (P95 or P99, user-selected,
# no silent default) of a DFN block-size distribution, and is held
# CONSTANT across all Sj* classes.
#
# g(n): the professor has decided g(n) = 1 (shift_factor_mode="no_shift")
# for the KCO* baseline. n is still computed per class because it is
# required for the Swebrec b parameter.
#
# Units: block volumes m3, Sj* m, B/S/H m, hole diameter mm, all
# fragment sizes (X50, Xmax, x-grid) mm in outputs, exactly as in
# kco_model.py.
# ============================================================

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np

from kco_model import (
    BlastDesign,
    KCOResult,
    _validate_design,
    _joint_plane_angle_rating,
    jps_from_joint_spacing,
    joint_factor,
    joint_condition_factor,
    rock_mass_description,
    rock_density_influence,
    hardness_factor,
    blastability_index,
    rock_factor_A,
    calculate_powder_factor,
    uniformity_index_n,
    shift_factor_g,
    x50_kuznetsov,
    block_volume_to_equivalent_cube,
    characteristic_block_size_from_distribution,
    xmax_kco,
    b_parameter,
    swebrec_passing,
    swebrec_size_at_passing,
    compare_with_measured_fragmentation,
    B_TYPICAL_MIN,
    B_TYPICAL_MAX,
)


# ============================================================
# 1. Sj* PROBABILISTIC DISTRIBUTION FROM DFN BLOCK VOLUMES
#    KCO* modelling assumption: Sj* = V^(1/3) used as a joint-spacing
#    analogue. This reuses kco_model.block_volume_to_equivalent_cube,
#    which implements the identical V^(1/3) formula for a different
#    purpose (Xmax volume-to-length conversion); it is NOT a published
#    KCO joint-spacing rule.
# ============================================================
@dataclass
class SjStarDistribution:
    """
    Probabilistic Sj* = V^(1/3) distribution built from DFN block volumes.

    KCO* modelling assumption (see module docstring): Sj* is not a
    published KCO quantity.

    Attributes:
        block_volume_m3: Retained (finite, > 0) block volumes (m3).
        sj_star_m: Sj* = V^(1/3) for each retained block (m).
        n_retained: Number of valid blocks retained.
        n_rejected: Number of blocks discarded (non-finite or <= 0).
        summary: Summary statistics of sj_star_m (m): "min", "p5", "p10",
            "p20", "p25", "p50", "mean", "p75", "p80", "p90", "p95",
            "p99", "max", "std". Reporting only; NOT used automatically
            as the KCO* input (see :func:`create_sj_percentile_classes`).
    """
    block_volume_m3: np.ndarray
    sj_star_m: np.ndarray
    n_retained: int
    n_rejected: int
    summary: dict


def sj_star_distribution_from_block_volumes(block_volumes_m3) -> SjStarDistribution:
    """
    Build the probabilistic Sj* distribution from DFN block volumes.

    KCO* modelling assumption: Sj*_i = V_i^(1/3) for every valid block
    (see module docstring). Uses kco_model.block_volume_to_equivalent_cube
    for the cube-root conversion (same formula, reused, not duplicated).

    Filter: finite values only, V > 0. Non-equivalent-sphere: KCO*
    explicitly uses the cube-root definition only, never the equivalent
    sphere diameter.

    Args:
        block_volumes_m3: Iterable of DFN block volumes (m3), from the
            S x B x H blast-cell sub-domain (or whichever domain the
            caller has decided represents one blast round; this module
            does not generate the DFN itself).

    Returns:
        SjStarDistribution: retained volumes, Sj* values, counts and
        summary statistics (reporting only).

    Raises:
        ValueError: If no valid (finite, positive) volumes remain.
    """
    v_all = np.asarray(list(block_volumes_m3), dtype=float).ravel()
    valid = np.isfinite(v_all) & (v_all > 0.0)
    v = v_all[valid]
    n_rejected = int(v_all.size - v.size)
    if v.size == 0:
        raise ValueError(
            "no valid (finite, positive) block volumes supplied for the "
            "Sj* distribution"
        )

    sj_star = np.asarray(block_volume_to_equivalent_cube(v), dtype=float)

    pct_levels = {"p5": 5, "p10": 10, "p20": 20, "p25": 25, "p50": 50,
                 "p75": 75, "p80": 80, "p90": 90, "p95": 95, "p99": 99}
    summary = {
        "min": float(np.min(sj_star)),
        "max": float(np.max(sj_star)),
        "mean": float(np.mean(sj_star)),
        "std": float(np.std(sj_star, ddof=0)),
    }
    for label, p in pct_levels.items():
        summary[label] = float(np.percentile(sj_star, p))

    return SjStarDistribution(
        block_volume_m3=v,
        sj_star_m=sj_star,
        n_retained=int(v.size),
        n_rejected=n_rejected,
        summary=summary,
    )


# ============================================================
# 2. PERCENTILE CLASSES OF THE Sj* DISTRIBUTION
#    KCO* modelling assumption: dividing the probabilistic Sj*
#    distribution into percentile classes, and picking one
#    representative Sj per class, is a KCO* methodological choice
#    proposed for testing, not a published KCO procedure.
# ============================================================
@dataclass
class SjStarClass:
    """
    One percentile class of the Sj* distribution (KCO* input only, not
    yet propagated through KCO).

    Attributes:
        percentile_low: Lower percentile edge (0-100).
        percentile_high: Upper percentile edge (0-100).
        n_blocks: Number of DFN blocks assigned to this class.
        weight: Probability weight w_i = n_blocks / n_retained (computed
            from the actual retained population, not assumed equal to
            (percentile_high - percentile_low) / 100).
        volume_min_m3: Minimum block volume in this class.
        volume_max_m3: Maximum block volume in this class.
        sj_min_m: Minimum Sj* in this class.
        sj_max_m: Maximum Sj* in this class.
        sj_representative_m: Representative Sj* for this class, per
            ``representative_method``.
        sj_values_m: The actual Sj* values retained in this class
            (for later sensitivity analysis).
        bin_edge_low_m, bin_edge_high_m: Constructed bin boundaries
            (Sj_lower/Sj_upper), only set by
            :func:`create_sj_log_bins` / :func:`create_sj_log_bins_p1_p99_tails`
            (None for percentile classes, where the class boundary is
            defined by population count, not a fixed Sj* value).
        is_tail: None for a regular class/bin. "lower" or "upper" if
            this class is one of the two explicit tail populations
            produced by :func:`create_sj_log_bins_p1_p99_tails`
            (Sj* below P1 / above P99, kept separate rather than
            merged into the nearest central bin).
    """
    percentile_low: float
    percentile_high: float
    n_blocks: int
    weight: float
    volume_min_m3: float
    volume_max_m3: float
    sj_min_m: float
    sj_max_m: float
    sj_representative_m: float
    sj_values_m: np.ndarray
    bin_edge_low_m: Optional[float] = None
    bin_edge_high_m: Optional[float] = None
    is_tail: Optional[str] = None


def create_sj_percentile_classes(
        sj_star_dist: SjStarDistribution,
        percentile_edges: Sequence[float] = (0, 10, 20, 30, 40, 50, 60,
                                             70, 80, 90, 100),
        representative_method: str = "median",
) -> list[SjStarClass]:
    """
    Divide the Sj* distribution into configurable percentile classes.

    KCO* modelling assumption: this partitioning and the choice of
    representative statistic per class are KCO* methodology, not a
    published KCO procedure.

    Partitioning method: the retained Sj* values are sorted and each
    class is assigned a contiguous slice of the SORTED population using
    index boundaries ``round(percentile_edges[j] / 100 * n)``. This
    guarantees (by construction, verified by
    :func:`self_test_kco_star`):
        - every retained block belongs to exactly one class,
        - no block belongs to two classes,
        - class weights (n_blocks / n_retained) sum to 1.

    Args:
        sj_star_dist: Output of
            :func:`sj_star_distribution_from_block_volumes`.
        percentile_edges: Monotonically increasing percentile edges in
            [0, 100], e.g. ``[0,10,...,100]`` for ten 10 %-wide classes.
            Fully configurable; NOT hard-coded.
        representative_method: One of:
            "median": median of the Sj* values actually retained in
                the class (default; KCO* modelling choice, not a
                published KCO rule).
            "mean": arithmetic mean of the Sj* values in the class.
            "percentile_midpoint": Sj* evaluated (on the FULL
                distribution) at the midpoint percentile of the class
                edges, e.g. 5 % for the 0-10 % class.

    Returns:
        list[SjStarClass]: One entry per non-empty class, in ascending
        percentile order.

    Raises:
        ValueError: If ``percentile_edges`` is not monotonically
            increasing, not within [0, 100], has fewer than 2 edges, or
            if ``representative_method`` is not recognised.
    """
    edges = [float(e) for e in percentile_edges]
    if len(edges) < 2:
        raise ValueError("percentile_edges must have at least 2 values")
    if any(e < 0.0 or e > 100.0 for e in edges):
        raise ValueError("percentile_edges must lie within [0, 100]")
    if any(b <= a for a, b in zip(edges, edges[1:])):
        raise ValueError("percentile_edges must be strictly increasing")
    if representative_method not in ("median", "mean", "percentile_midpoint"):
        raise ValueError(
            "representative_method must be 'median', 'mean' or "
            "'percentile_midpoint'"
        )

    order = np.argsort(sj_star_dist.sj_star_m)
    sj_sorted = sj_star_dist.sj_star_m[order]
    v_sorted = sj_star_dist.block_volume_m3[order]
    n = sj_sorted.size

    idx = [int(round(e / 100.0 * n)) for e in edges]
    idx[0], idx[-1] = 0, n

    classes: list[SjStarClass] = []
    for j in range(len(idx) - 1):
        lo, hi = idx[j], idx[j + 1]
        if hi <= lo:
            continue  # empty class (edges too fine for this population)
        sj_slice = sj_sorted[lo:hi]
        v_slice = v_sorted[lo:hi]

        if representative_method == "median":
            sj_rep = float(np.median(sj_slice))
        elif representative_method == "mean":
            sj_rep = float(np.mean(sj_slice))
        else:  # percentile_midpoint
            mid_pct = 0.5 * (edges[j] + edges[j + 1])
            sj_rep = float(np.percentile(sj_star_dist.sj_star_m, mid_pct))

        classes.append(SjStarClass(
            percentile_low=edges[j],
            percentile_high=edges[j + 1],
            n_blocks=int(hi - lo),
            weight=float(hi - lo) / float(n),
            volume_min_m3=float(np.min(v_slice)),
            volume_max_m3=float(np.max(v_slice)),
            sj_min_m=float(np.min(sj_slice)),
            sj_max_m=float(np.max(sj_slice)),
            sj_representative_m=sj_rep,
            sj_values_m=sj_slice,
        ))
    return classes


# ============================================================
# 2b. LOG-SPACED BINS OF THE Sj* DISTRIBUTION (alternative/sensitivity
#     method to the equal-population percentile classes above)
#     KCO* modelling assumption: dividing the Sj* distribution into
#     equal-width bins along ln(Sj*) is a KCO* methodological choice
#     proposed for testing, not a published KCO procedure. This does
#     NOT replace create_sj_percentile_classes; both remain available,
#     selected via predict_kco_star(class_method=...).
# ============================================================
def create_sj_log_bins(
        sj_star_dist: SjStarDistribution,
        n_bins: int = 10,
        representative_method: str = "median",
) -> tuple[list[SjStarClass], list[str]]:
    """
    Divide the Sj* distribution into ``n_bins`` EQUAL-WIDTH bins along
    ln(Sj*), instead of the equal-POPULATION percentile classes of
    :func:`create_sj_percentile_classes`.

    KCO* modelling assumption (see module docstring): this is an
    ALTERNATIVE/sensitivity partitioning method, selected via
    ``predict_kco_star(class_method="log_bins")``; it does not replace
    or delete the default equal-population percentile-class method
    (still available via ``class_method="percentile"``).

    Bin construction: ``n_bins`` edges equally spaced in ``ln(Sj*)``
    between ``ln(min(Sj*))`` and ``ln(max(Sj*))`` of the ACTUAL
    retained population (not an arbitrary range), so the first and
    last bin edges always coincide exactly with the population's
    min/max Sj*. Every retained block is assigned to exactly one bin
    (:func:`numpy.digitize`); NO block is filtered, including any
    extremely small Sj* values that may be DFN boundary/sliver
    artifacts -- filtering/cutoffs are a separate, explicit modelling
    decision left to the caller and are never silently applied here.

    Args:
        sj_star_dist: Output of
            :func:`sj_star_distribution_from_block_volumes`.
        n_bins: Number of equal-width ln(Sj*) bins. Configurable;
            default 10 (matches the default 10 percentile classes).
        representative_method: "median" (default) or "mean" of the
            Sj* values actually retained in the bin. "percentile_
            midpoint" is not defined for log bins (there is no
            percentile axis here); if requested, the geometric mean of
            the bin edges is used instead.

    Returns:
        tuple[list[SjStarClass], list[str]]:
            - One SjStarClass per NON-EMPTY bin, in ascending Sj*
              order. ``percentile_low``/``percentile_high`` report the
              EMPIRICAL percentile rank range actually covered by the
              bin's members (for comparability with the percentile-
              class table only; they do NOT define the bin, unlike in
              :func:`create_sj_percentile_classes`).
              ``bin_edge_low_m``/``bin_edge_high_m`` report the actual
              constructed log-bin boundaries (Sj_lower/Sj_upper).
            - List of warning strings: one per EMPTY bin, and one for
              any non-empty bin holding < 1% of the population while
              sitting in the lower half of the Sj* range -- a signal
              that it may be dominated by the extreme-small-Sj* tail
              (see the DFN lower-tail investigation: blocks down to
              ~1.7e-11 m3, volume-negligible but count-relevant).
              These are diagnostics only; no block is removed.

    Raises:
        ValueError: If ``n_bins`` < 1, or ``representative_method`` is
            not recognised.
    """
    if n_bins < 1:
        raise ValueError("n_bins must be >= 1")
    if representative_method not in ("median", "mean", "percentile_midpoint"):
        raise ValueError(
            "representative_method must be 'median', 'mean' or "
            "'percentile_midpoint'"
        )

    order = np.argsort(sj_star_dist.sj_star_m)
    sj_sorted = sj_star_dist.sj_star_m[order]
    v_sorted = sj_star_dist.block_volume_m3[order]
    n = sj_sorted.size

    ln_sj = np.log(sj_sorted)
    ln_edges = np.linspace(ln_sj.min(), ln_sj.max(), n_bins + 1)
    edges_m = np.exp(ln_edges)
    edges_m[0] = sj_sorted[0]
    edges_m[-1] = sj_sorted[-1]

    bin_idx = np.digitize(sj_sorted, edges_m[1:-1], right=False)

    classes: list[SjStarClass] = []
    log_bin_warnings: list[str] = []
    total_assigned = 0
    for b in range(n_bins):
        mask = bin_idx == b
        n_b = int(mask.sum())
        total_assigned += n_b
        if n_b == 0:
            log_bin_warnings.append(
                f"log bin {b} [{edges_m[b]:.6g}, {edges_m[b + 1]:.6g}] m "
                "is EMPTY (0 blocks)"
            )
            continue

        sj_slice = sj_sorted[mask]
        v_slice = v_sorted[mask]
        rank_idx = np.where(mask)[0]
        p_low = 100.0 * rank_idx.min() / n
        p_high = 100.0 * (rank_idx.max() + 1) / n
        weight_b = n_b / float(n)

        if weight_b < 0.01 and edges_m[b + 1] < sj_sorted[-1] * 0.5:
            log_bin_warnings.append(
                f"log bin {b} [{edges_m[b]:.6g}, {edges_m[b + 1]:.6g}] m "
                f"holds only {n_b} block(s) ({100 * weight_b:.3f}% of "
                "population) -- may be dominated by extreme-small-Sj* "
                "boundary/sliver blocks; not filtered here, reported only"
            )

        if representative_method == "median":
            sj_rep = float(np.median(sj_slice))
        elif representative_method == "mean":
            sj_rep = float(np.mean(sj_slice))
        else:  # percentile_midpoint: no percentile axis, use geometric
              # mean of the bin edges instead
            sj_rep = float(np.sqrt(edges_m[b] * edges_m[b + 1]))

        classes.append(SjStarClass(
            percentile_low=p_low,
            percentile_high=p_high,
            n_blocks=n_b,
            weight=weight_b,
            volume_min_m3=float(np.min(v_slice)),
            volume_max_m3=float(np.max(v_slice)),
            sj_min_m=float(np.min(sj_slice)),
            sj_max_m=float(np.max(sj_slice)),
            sj_representative_m=sj_rep,
            sj_values_m=sj_slice,
            bin_edge_low_m=float(edges_m[b]),
            bin_edge_high_m=float(edges_m[b + 1]),
        ))

    assert total_assigned == n, (
        f"internal error: log bins assigned {total_assigned} blocks, "
        f"expected {n} (every block must land in exactly one bin)"
    )
    total_w = sum(c.weight for c in classes)
    if classes and abs(total_w - 1.0) > 1e-9:
        raise ValueError(
            f"internal error: log-bin weights sum to {total_w:.9f}, "
            "expected 1"
        )
    return classes, log_bin_warnings


# ============================================================
# 2c. LOG-SPACED BINS RESTRICTED TO [P_low, P_high], WITH TWO
#     EXPLICIT TAIL CLASSES (alternative to create_sj_log_bins)
#     KCO* modelling assumption: identical status to
#     create_sj_log_bins -- a KCO* methodological choice, not a
#     published KCO procedure. Selected via
#     predict_kco_star(class_method="log_bins_p1_p99_tails").
# ============================================================
def create_sj_log_bins_p1_p99_tails(
        sj_star_dist: SjStarDistribution,
        n_bins: int = 10,
        tail_low_pct: float = 1.0,
        tail_high_pct: float = 99.0,
        representative_method: str = "median",
) -> tuple[list[SjStarClass], list[str]]:
    """
    Divide the Sj* distribution into ``n_bins`` equal-width bins along
    ln(Sj*) placed STRICTLY inside [P_low, P_high] (default P1-P99),
    plus two explicit, separate "lower tail" (Sj* < P_low) and
    "upper tail" (Sj* > P_high) classes. No block is filtered: the
    two tails are not merged into bin 1 / bin n_bins (unlike
    :func:`create_sj_log_bins`, which stretches the outer bin edges
    to the population min/max); they are returned as their own
    :class:`SjStarClass` entries (``is_tail="lower"``/``"upper"``)
    with their own count, weight and representative Sj* (median of
    the tail's members), so every retained block is represented
    exactly once across all ``n_bins + 2`` classes and the class
    weights (n_blocks / n_retained) still sum to 1 by construction
    (lower tail + central bins + upper tail is a strict partition of
    the full population).

    Args:
        sj_star_dist: Output of
            :func:`sj_star_distribution_from_block_volumes`.
        n_bins: Number of equal-width ln(Sj*) bins inside
            [P_low, P_high]. Default 10.
        tail_low_pct: Lower percentile edge (0-100) defining the
            lower-tail cutoff. Default 1 (P1).
        tail_high_pct: Upper percentile edge (0-100) defining the
            upper-tail cutoff. Default 99 (P99).
        representative_method: "median" (default) or "mean" of the
            Sj* values retained in each class (tail or central bin).

    Returns:
        tuple[list[SjStarClass], list[str]]:
            - Classes in ascending Sj* order: [lower tail (if
              non-empty), central bin 1, ..., central bin n_bins (if
              non-empty), upper tail (if non-empty)].
              ``bin_edge_low_m``/``bin_edge_high_m`` report the actual
              boundaries (P_low/P_high for the tails, the constructed
              log-bin edges for central bins).
            - List of warning strings for any empty central bin.

    Raises:
        ValueError: If ``n_bins`` < 1, ``tail_low_pct``/
            ``tail_high_pct`` are not within [0, 100] with
            ``tail_low_pct < tail_high_pct``, or
            ``representative_method`` is not recognised.
    """
    if n_bins < 1:
        raise ValueError("n_bins must be >= 1")
    if not (0.0 <= tail_low_pct < tail_high_pct <= 100.0):
        raise ValueError(
            "require 0 <= tail_low_pct < tail_high_pct <= 100"
        )
    if representative_method not in ("median", "mean"):
        raise ValueError(
            "representative_method must be 'median' or 'mean' for "
            "create_sj_log_bins_p1_p99_tails"
        )

    def _representative(values: np.ndarray) -> float:
        return float(np.median(values) if representative_method == "median"
                    else np.mean(values))

    order = np.argsort(sj_star_dist.sj_star_m)
    sj_sorted = sj_star_dist.sj_star_m[order]
    v_sorted = sj_star_dist.block_volume_m3[order]
    n = sj_sorted.size

    p_low = float(np.percentile(sj_sorted, tail_low_pct))
    p_high = float(np.percentile(sj_sorted, tail_high_pct))

    lower_mask = sj_sorted < p_low
    upper_mask = sj_sorted > p_high
    central_mask = ~lower_mask & ~upper_mask

    classes: list[SjStarClass] = []
    warn: list[str] = []

    n_lower = int(lower_mask.sum())
    if n_lower > 0:
        sj_l, v_l = sj_sorted[lower_mask], v_sorted[lower_mask]
        classes.append(SjStarClass(
            percentile_low=0.0, percentile_high=tail_low_pct,
            n_blocks=n_lower, weight=n_lower / n,
            volume_min_m3=float(np.min(v_l)), volume_max_m3=float(np.max(v_l)),
            sj_min_m=float(np.min(sj_l)), sj_max_m=float(np.max(sj_l)),
            sj_representative_m=_representative(sj_l), sj_values_m=sj_l,
            bin_edge_low_m=float(sj_sorted[0]), bin_edge_high_m=p_low,
            is_tail="lower",
        ))

    sj_c, v_c = sj_sorted[central_mask], v_sorted[central_mask]
    n_central = sj_c.size
    ln_edges = np.linspace(np.log(p_low), np.log(p_high), n_bins + 1)
    edges_m = np.exp(ln_edges)
    bin_idx = np.digitize(sj_c, edges_m[1:-1], right=False)
    total_assigned_central = 0
    for b in range(n_bins):
        mask = bin_idx == b
        n_b = int(mask.sum())
        total_assigned_central += n_b
        if n_b == 0:
            warn.append(
                f"central log bin {b} [{edges_m[b]:.6g}, "
                f"{edges_m[b + 1]:.6g}] m is EMPTY (0 blocks)"
            )
            continue
        sj_b, v_b = sj_c[mask], v_c[mask]
        classes.append(SjStarClass(
            percentile_low=float(100.0 * np.searchsorted(sj_sorted, sj_b.min()) / n),
            percentile_high=float(100.0 * np.searchsorted(sj_sorted, sj_b.max(), side="right") / n),
            n_blocks=n_b, weight=n_b / n,
            volume_min_m3=float(np.min(v_b)), volume_max_m3=float(np.max(v_b)),
            sj_min_m=float(np.min(sj_b)), sj_max_m=float(np.max(sj_b)),
            sj_representative_m=_representative(sj_b), sj_values_m=sj_b,
            bin_edge_low_m=float(edges_m[b]), bin_edge_high_m=float(edges_m[b + 1]),
            is_tail=None,
        ))
    assert total_assigned_central == n_central, (
        f"internal error: central bins assigned {total_assigned_central}, "
        f"expected {n_central}"
    )

    n_upper = int(upper_mask.sum())
    if n_upper > 0:
        sj_u, v_u = sj_sorted[upper_mask], v_sorted[upper_mask]
        classes.append(SjStarClass(
            percentile_low=tail_high_pct, percentile_high=100.0,
            n_blocks=n_upper, weight=n_upper / n,
            volume_min_m3=float(np.min(v_u)), volume_max_m3=float(np.max(v_u)),
            sj_min_m=float(np.min(sj_u)), sj_max_m=float(np.max(sj_u)),
            sj_representative_m=_representative(sj_u), sj_values_m=sj_u,
            bin_edge_low_m=p_high, bin_edge_high_m=float(sj_sorted[-1]),
            is_tail="upper",
        ))

    total_assigned = n_lower + total_assigned_central + n_upper
    assert total_assigned == n, (
        f"internal error: log_bins_p1_p99_tails assigned {total_assigned} "
        f"blocks, expected {n} (every block must be represented exactly "
        "once, across tails + central bins)"
    )
    total_w = sum(c.weight for c in classes)
    if classes and abs(total_w - 1.0) > 1e-9:
        raise ValueError(
            f"internal error: log_bins_p1_p99_tails weights sum to "
            f"{total_w:.9f}, expected 1"
        )
    return classes, warn


# ============================================================
# 3. PROPAGATE EACH Sj* CLASS THROUGH THE PUBLISHED KCO EQUATIONS
# ============================================================
@dataclass
class KCOStarClassResult:
    """
    Full KCO chain evaluated for one Sj* class, using published KCO
    equations (reused unmodified from kco_model.py) with Sj_i as the
    only varying structural input; all other blast-design parameters
    are held constant (see :func:`predict_kco_star`).

    Attributes:
        percentile_low, percentile_high: Class percentile edges.
        weight: Probability weight w_i (n_blocks / n_retained).
        n_blocks: Number of DFN blocks represented by this class.
        volume_min_m3, volume_max_m3: Block-volume range in this class.
        sj_min_m, sj_max_m: Sj* range in this class.
        sj_representative_m: Sj_i used to run KCO for this class.
        jps, jpa, jf, rmd, rdi, hf, bi, rock_factor: Published KCO
            intermediate quantities for this class (jpa, rdi, hf
            constant across all classes; jps/jf/rmd/bi/rock_factor
            vary with sj_representative_m).
        n: Cunningham (2005) uniformity index for this class.
        g_n: Shift factor applied (1.0 under the agreed KCO* baseline).
        x50_mm: Median fragment size for this class (mm).
        xmax_mm: Maximum fragment size (mm); CONSTANT across all
            classes (see :func:`predict_kco_star`).
        b: Swebrec undulation parameter for this class.
        x20_mm, x50_check_mm, x80_mm: Class-curve percentile sizes,
            for audit only.
        bin_edge_low_m, bin_edge_high_m: Constructed log-bin
            boundaries (Sj_lower/Sj_upper), only set when
            ``class_method in ("log_bins", "log_bins_p1_p99_tails")``
            (None for percentile classes; see :class:`SjStarClass`).
        is_tail: None for a regular class/bin. "lower"/"upper" if this
            is one of the two explicit tail classes produced by
            ``class_method="log_bins_p1_p99_tails"`` (see
            :func:`create_sj_log_bins_p1_p99_tails`); propagated
            through the same, unmodified KCO chain as every other
            class.
    """
    percentile_low: float
    percentile_high: float
    weight: float
    n_blocks: int
    volume_min_m3: float
    volume_max_m3: float
    sj_min_m: float
    sj_max_m: float
    sj_representative_m: float
    jps: int
    jpa: int
    jf: float
    rmd: float
    rdi: float
    hf: float
    bi: float
    rock_factor: float
    n: float
    g_n: float
    x50_mm: float
    xmax_mm: float
    b: float
    x20_mm: float
    x50_check_mm: float
    x80_mm: float
    bin_edge_low_m: Optional[float] = None
    bin_edge_high_m: Optional[float] = None
    is_tail: Optional[str] = None
    weight_count: Optional[float] = None
    weight_volume: Optional[float] = None
    volume_sum_m3: Optional[float] = None

    def passing(self, x_mm) -> np.ndarray:
        """Class-specific Swebrec percentage passing P_i(x) (published KCO)."""
        return swebrec_passing(x_mm, self.x50_mm, self.xmax_mm, self.b)


# ============================================================
# 4. KCO* RESULT: WEIGHTED MIXTURE OF CLASS CURVES
# ============================================================
@dataclass
class KCOStarResult:
    """
    Full, traceable KCO* prediction: a probability-weighted mixture of
    per-class Swebrec curves.

    KCO* modelling assumption (see module docstring): the weighted
    mixture combination rule

        P_KCO*(x) = sum_i w_i * P_i(x)

    is a KCO* methodology proposed for testing; it is NOT a published
    Cunningham/Ouchterlony equation.

    Attributes:
        design: The BlastDesign holding the constant blast parameters
            (mean_joint_spacing_m, block_size_method and
            in_situ_block_size_m are NOT used by KCO*; Sj* and Xmax are
            supplied separately, see :func:`predict_kco_star`).
        classes: One KCOStarClassResult per non-empty percentile class.
        percentile_edges: The percentile edges used to build the classes.
        representative_method: "median", "mean" or "percentile_midpoint".
        xmax_block_percentile: Percentile (e.g. 95 or 99) of the Xmax
            block-size distribution used; user-selected, never silent.
        xmax_block_size_method: "equivalent_cube" or "equivalent_sphere"
            volume-to-length convention used for Xmax.
        xmax_mm: The single Xmax value (mm) shared by every class.
        xmax_governed_by: "in-situ block", "burden" or "spacing".
        n_retained: Number of DFN blocks retained for the Sj*
            distribution.
        n_rejected: Number of DFN blocks discarded.
        sj_star_dist: The full SjStarDistribution (summary stats,
            reporting only).
        warnings: All warnings raised during the prediction.
        class_method: "percentile" (default, equal-population classes)
            or "log_bins" (equal-width bins along ln(Sj*); see
            :func:`create_sj_log_bins`).
        n_log_bins: Number of log bins used, only set when
            ``class_method=="log_bins"`` (None otherwise).
    """
    design: BlastDesign
    classes: list
    percentile_edges: list
    representative_method: str
    xmax_block_percentile: float
    xmax_block_size_method: str
    xmax_mm: float
    xmax_governed_by: str
    n_retained: int
    n_rejected: int
    sj_star_dist: SjStarDistribution
    warnings: list = field(default_factory=list)
    class_method: str = "percentile"
    n_log_bins: Optional[int] = None
    tail_low_pct: Optional[float] = None
    tail_high_pct: Optional[float] = None
    weight_method: str = "count"

    def weights(self) -> np.ndarray:
        return np.array([c.weight for c in self.classes], dtype=float)

    def envelope(self, x_mm) -> tuple[np.ndarray, np.ndarray]:
        """
        Return (P_min(x), P_max(x)) over the individual class Swebrec
        curves P_i(x): the KCO* class envelope ("fuseau").

        This is the range of the individual class curves, NOT a
        confidence interval around the weighted mixture.
        """
        return kco_star_class_envelope(x_mm, self.classes)

    def unique_curves(self, tol: float = 1e-9) -> list:
        """
        Group classes that produce numerically identical Swebrec curves
        (same X50, Xmax and b). Returns a list of dicts with keys
        "x50_mm", "b", "jps", "class_indices" (1-based) and
        "weight" (summed weight of the group).
        """
        groups: list = []
        for i, c in enumerate(self.classes, start=1):
            for g in groups:
                if (abs(g["x50_mm"] - c.x50_mm) < tol
                        and abs(g["b"] - c.b) < tol
                        and abs(g["xmax_mm"] - c.xmax_mm) < tol):
                    g["class_indices"].append(i)
                    g["weight"] += c.weight
                    break
            else:
                groups.append({"x50_mm": c.x50_mm, "b": c.b,
                               "xmax_mm": c.xmax_mm, "jps": c.jps,
                               "class_indices": [i], "weight": c.weight})
        return groups

    def passing(self, x_mm) -> np.ndarray:
        """
        Return the KCO* probability-weighted cumulative percentage
        passing P_KCO*(x).

        KCO* modelling assumption: weighted mixture of the class
        Swebrec curves, see module/class docstring.
        """
        return kco_star_passing(x_mm, self.classes)

    def size_at(self, passing_pct) -> np.ndarray:
        """
        Invert the combined KCO* curve numerically (no closed form for
        a mixture of Swebrec curves).
        """
        return kco_star_size_at_passing(passing_pct, self.classes)

    def percentiles(self,
                    levels: Sequence[float] = (10, 20, 30, 50, 80, 90, 100)
                    ) -> dict:
        """Return characteristic KCO* percentile sizes (mm)."""
        return {f"X{int(p)}_star": float(np.atleast_1d(self.size_at(p))[0])
                for p in levels}

    def audit_table(self) -> str:
        """Return a formatted audit table, one row per Sj* class."""
        w = (7, 8, 9, 8, 7, 7, 7, 8, 8, 9, 9, 9, 9, 6)
        header = (f"{'P_low':>{w[0]}}{'P_high':>{w[1]}}{'n_blk':>{w[2]}}"
                  f"{'weight':>{w[3]}}{'JPS':>{w[4]}}{'JF':>{w[5]}}"
                  f"{'RMD':>{w[6]}}{'BI':>{w[7]}}{'A':>{w[8]}}"
                  f"{'n':>{w[9]}}{'X50(mm)':>{w[10]}}{'Xmax(mm)':>{w[11]}}"
                  f"{'b':>{w[12]}}{'tail':>{w[13]}}")
        lines = [header, "-" * len(header)]
        for c in self.classes:
            lines.append(
                f"{c.percentile_low:>{w[0]}.1f}{c.percentile_high:>{w[1]}.1f}"
                f"{c.n_blocks:>{w[2]}d}{c.weight:>{w[3]}.4f}"
                f"{c.jps:>{w[4]}d}{c.jf:>{w[5]}.3g}{c.rmd:>{w[6]}.3g}"
                f"{c.bi:>{w[7]}.3g}{c.rock_factor:>{w[8]}.3g}"
                f"{c.n:>{w[9]}.3g}{c.x50_mm:>{w[10]}.4g}"
                f"{c.xmax_mm:>{w[11]}.4g}{c.b:>{w[12]}.3g}"
                f"{(c.is_tail or ''):>{w[13]}}"
            )
        lines.append("")
        lines.append(f"sum(weights) = {sum(c.weight for c in self.classes):.6f}")
        lines.append(f"weight_method = {self.weight_method}")
        lines.append(f"n_retained = {self.n_retained}, "
                     f"n_rejected = {self.n_rejected}")
        lines.append(f"class_method = {self.class_method}"
                     + (f" (n_log_bins={self.n_log_bins})"
                        if self.class_method == "log_bins" else "")
                     + (f" (n_log_bins={self.n_log_bins}, "
                        f"tail_low_pct={self.tail_low_pct}, "
                        f"tail_high_pct={self.tail_high_pct})"
                        if self.class_method == "log_bins_p1_p99_tails" else ""))
        lines.append(f"representative_method = {self.representative_method}")
        lines.append(f"xmax_block_percentile = P{self.xmax_block_percentile:.0f}, "
                     f"method = {self.xmax_block_size_method}")
        lines.append(f"Xmax = {self.xmax_mm:.4g} mm "
                     f"(governed by {self.xmax_governed_by})")
        lines.append(f"g(n) mode = {self.design.shift_factor_mode}")
        if self.warnings:
            lines.append("")
            lines.append("Warnings:")
            for msg in self.warnings:
                lines.append(f"  - {msg}")
        return "\n".join(lines)

    def class_table_rows(self) -> list:
        """
        Return one dict per Sj* class with the columns: bin, Sj_lower,
        Sj_upper, n_blocks, weight, Sj_representative, JPS, JF, RMD,
        BI, A, n, X50, Xmax, b, is_tail.

        Works identically across all ``class_method`` values: when
        ``class_method in ("log_bins", "log_bins_p1_p99_tails")``,
        Sj_lower/Sj_upper are the constructed bin/tail edges
        (``bin_edge_low_m``/``bin_edge_high_m``); when
        ``class_method=="percentile"``, they fall back to the actual
        Sj* data range of the class (``sj_min_m``/``sj_max_m``), since
        percentile classes have no fixed bin edge. ``is_tail`` is
        None/"lower"/"upper" (only non-None for
        ``class_method=="log_bins_p1_p99_tails"``).
        """
        rows = []
        for i, c in enumerate(self.classes, start=1):
            sj_lower = (c.bin_edge_low_m if c.bin_edge_low_m is not None
                       else c.sj_min_m)
            sj_upper = (c.bin_edge_high_m if c.bin_edge_high_m is not None
                       else c.sj_max_m)
            rows.append({
                "bin": i,
                "Sj_lower": sj_lower,
                "Sj_upper": sj_upper,
                "n_blocks": c.n_blocks,
                "weight": c.weight,
                "weight_count": c.weight_count,
                "weight_volume": c.weight_volume,
                "volume_sum_m3": c.volume_sum_m3,
                "Sj_representative": c.sj_representative_m,
                "JPS": c.jps,
                "JF": c.jf,
                "RMD": c.rmd,
                "BI": c.bi,
                "A": c.rock_factor,
                "n": c.n,
                "X50": c.x50_mm,
                "Xmax": c.xmax_mm,
                "b": c.b,
                "is_tail": c.is_tail,
            })
        return rows


# ============================================================
# 5. WEIGHTED-MIXTURE CURVE FUNCTIONS
# ============================================================
def kco_star_passing(x, classes: list) -> np.ndarray:
    """
    Return the KCO* probability-weighted cumulative percentage passing.

    KCO* modelling assumption:

        P_KCO*(x) = sum_i w_i * P_i(x)

    where P_i(x) is the published Swebrec curve of class i
    (:func:`kco_model.swebrec_passing`, unmodified) and w_i is the
    class probability weight. This weighted-mixture rule is proposed
    KCO* methodology, NOT a published Cunningham/Ouchterlony equation.

    Args:
        x: Fragment size(s) (mm).
        classes: List of :class:`KCOStarClassResult`.

    Returns:
        np.ndarray: Percentage passing in [0, 100].

    Raises:
        ValueError: If ``classes`` is empty or weights do not sum to
            ~1 (tolerance 1e-6).
    """
    if not classes:
        raise ValueError("no classes supplied")
    total_w = sum(c.weight for c in classes)
    if abs(total_w - 1.0) > 1e-6:
        raise ValueError(f"class weights must sum to 1, got {total_w:.6f}")

    x = np.asarray(x, dtype=float)
    out = np.zeros_like(x)
    for c in classes:
        out = out + c.weight * swebrec_passing(x, c.x50_mm, c.xmax_mm, c.b)
    return out


def kco_star_class_envelope(x, classes: list) -> tuple[np.ndarray, np.ndarray]:
    """
    Return the KCO* class envelope ("fuseau"):

        P_min(x) = min_i P_i(x),   P_max(x) = max_i P_i(x)

    over the individual class Swebrec curves P_i(x). This is the range
    spanned by the individual KCO class curves, NOT a confidence
    interval around the weighted mixture.

    Args:
        x: Fragment size(s) (mm).
        classes: List of :class:`KCOStarClassResult`.

    Returns:
        tuple[np.ndarray, np.ndarray]: (P_min, P_max), percent passing.
    """
    if not classes:
        raise ValueError("no classes supplied")
    x = np.asarray(x, dtype=float)
    stack = np.vstack([swebrec_passing(x, c.x50_mm, c.xmax_mm, c.b)
                       for c in classes])
    return stack.min(axis=0), stack.max(axis=0)


def kco_star_size_at_passing(passing_pct, classes: list,
                             n_grid: int = 4000) -> np.ndarray:
    """
    Invert the combined KCO* curve numerically.

    KCO* modelling assumption: there is no closed-form inverse for a
    weighted mixture of Swebrec curves (unlike the single-class
    :func:`kco_model.swebrec_size_at_passing`). A fine log-spaced size
    grid is built between a small fraction of the smallest class X50
    and the shared Xmax, P_KCO*(x) is evaluated on it
    (:func:`kco_star_passing`, monotonic by construction), and the
    requested percentile(s) are obtained by interpolation (linear in
    passing %, logarithmic in size), matching the interpolation
    convention already used in
    :func:`kco_model.compare_with_measured_fragmentation`.

    Args:
        passing_pct: Percentage(s) passing in (0, 100].
        classes: List of :class:`KCOStarClassResult`.
        n_grid: Number of grid points used for the numerical inversion.

    Returns:
        np.ndarray: Fragment size(s) (mm). NaN outside the achievable
        passing range of the grid.
    """
    if not classes:
        raise ValueError("no classes supplied")
    xmax_mm = classes[0].xmax_mm
    x_min = min(c.x50_mm for c in classes) * 1e-3
    x_grid = np.geomspace(max(x_min, 1e-6), xmax_mm, n_grid)
    p_grid = kco_star_passing(x_grid, classes)

    keep = np.concatenate(([True], np.diff(p_grid) > 0))
    p_keep, x_keep = p_grid[keep], x_grid[keep]

    p = np.atleast_1d(np.asarray(passing_pct, dtype=float))
    out = np.full_like(p, np.nan)
    valid = (p > 0.0) & (p <= 100.0) & (p >= p_keep.min()) & (p <= p_keep.max())
    out[valid] = np.exp(np.interp(p[valid], p_keep, np.log(x_keep)))
    return out


# ============================================================
# 6. IN-SITU (PRE-BLAST) DISTRIBUTION FROM Sj*
# ============================================================
def in_situ_passing_from_sj_star(sj_star_dist: SjStarDistribution
                                 ) -> tuple[np.ndarray, np.ndarray]:
    """
    Return the empirical in-situ (pre-blast) cumulative size distribution
    built directly from Sj* = V^(1/3), for comparison against the
    post-blast KCO / KCO* / WipFrag curves.

    This is the pre-blast block structure, NOT a fragmentation
    prediction: it must be clearly labelled as such on any comparison
    plot (see :func:`plot_main_comparison`).

    Args:
        sj_star_dist: Output of
            :func:`sj_star_distribution_from_block_volumes`.

    Returns:
        tuple[np.ndarray, np.ndarray]: (size_mm, passing_pct), both
        sorted ascending, empirical CDF (rank / n * 100).
    """
    sj_sorted_mm = np.sort(sj_star_dist.sj_star_m) * 1000.0
    n = sj_sorted_mm.size
    passing_pct = 100.0 * (np.arange(1, n + 1) - 0.5) / n
    return sj_sorted_mm, passing_pct


def in_situ_size_at_passing(sj_star_dist: SjStarDistribution,
                            passing_pct) -> np.ndarray:
    """
    Return the in-situ (pre-blast) Sj*-based size(s) at given percentage(s)
    passing, by interpolation of the empirical CDF from
    :func:`in_situ_passing_from_sj_star`.

    Args:
        sj_star_dist: Output of
            :func:`sj_star_distribution_from_block_volumes`.
        passing_pct: Percentage(s) passing in (0, 100].

    Returns:
        np.ndarray: In-situ size(s) (mm).
    """
    x_mm, p = in_situ_passing_from_sj_star(sj_star_dist)
    p_arr = np.atleast_1d(np.asarray(passing_pct, dtype=float))
    out = np.full_like(p_arr, np.nan)
    valid = (p_arr > 0.0) & (p_arr <= 100.0)
    out[valid] = np.exp(np.interp(p_arr[valid], p, np.log(x_mm)))
    return out


# ============================================================
# 7. END-TO-END KCO* PREDICTION
# ============================================================
def predict_kco_star(
        design: BlastDesign,
        sj_block_volumes_m3,
        xmax_block_volumes_m3,
        xmax_block_size_method: str,
        xmax_block_percentile: float,
        percentile_edges: Sequence[float] = (0, 10, 20, 30, 40, 50, 60,
                                             70, 80, 90, 100),
        representative_method: str = "median",
        class_method: str = "percentile",
        n_log_bins: int = 10,
        tail_low_pct: float = 1.0,
        tail_high_pct: float = 99.0,
        weight_method: str = "count",
) -> KCOStarResult:
    """
    Run the full KCO* chain: DFN block volumes -> Sj* distribution ->
    percentile classes (or log-spaced bins) -> published KCO equations
    per class -> probability-weighted mixture curve.

    Every equation applied to each class is imported unmodified from
    kco_model.py. Only the structural input Sj_i varies between classes;
    B, S, H, D, Q, q, s_ANFO, rock density/UCS/E, JPA, W and timing
    parameters are held constant (taken once from ``design``), per the
    KCO* methodology requested. Xmax is computed ONCE (see module
    docstring: it is a separate issue from Sj*) and shared by every
    class.

    Args:
        design: Fully populated BlastDesign holding the CONSTANT blast
            parameters. ``mean_joint_spacing_m``, ``block_size_method``
            and ``in_situ_block_size_m`` are NOT used by KCO* (Sj* and
            Xmax are supplied explicitly through the other arguments).
        sj_block_volumes_m3: DFN block volumes (m3) from the domain
            chosen to represent the probabilistic in-situ block
            structure of one blast cell (e.g. an S x B x H sub-domain).
            This module does not generate the DFN; the caller supplies
            an already-generated block-volume population.
        xmax_block_volumes_m3: DFN block volumes (m3) used to derive
            the single, class-independent Xmax (see module docstring,
            section "Xmax is a SEPARATE issue"). May be the same or a
            different population from ``sj_block_volumes_m3``; this is
            a modelling choice the caller must make explicitly (no
            default; see the ambiguity flagged before implementation).
        xmax_block_size_method: "equivalent_cube" or
            "equivalent_sphere"; the volume-to-length convention used
            for Xmax. No default.
        xmax_block_percentile: Percentile (0, 100] of the Xmax
            block-size distribution, e.g. 95 or 99. No default; must
            be chosen explicitly (see kco_model.characteristic_block_
            size_from_distribution).
        percentile_edges: Percentile class edges for the Sj*
            distribution, e.g. ``[0,10,...,100]``. Fully configurable.
        representative_method: "median" (default), "mean" or
            "percentile_midpoint"; see
            :func:`create_sj_percentile_classes` /
            :func:`create_sj_log_bins`.
        class_method: "percentile" (default; equal-population classes,
            see :func:`create_sj_percentile_classes`), "log_bins"
            (equal-width bins along ln(Sj*) spanning the full
            population, see :func:`create_sj_log_bins`), or
            "log_bins_p1_p99_tails" (equal-width bins strictly inside
            [P_tail_low, P_tail_high] plus two explicit lower-/
            upper-tail classes, see
            :func:`create_sj_log_bins_p1_p99_tails`) -- KCO* modelling
            choice of HOW the Sj* distribution is partitioned before
            the published KCO chain is run once per representative
            Sj_i; it does not alter any published equation.
        n_log_bins: Number of log-spaced bins, used when
            ``class_method in ("log_bins", "log_bins_p1_p99_tails")``.
            Default 10.
        tail_low_pct, tail_high_pct: Percentile edges (0-100) defining
            the tail cutoffs, only used when
            ``class_method=="log_bins_p1_p99_tails"``. Default 1/99.
        weight_method: "count" (default) -> w_i = N_i / N_tot (block-
            count weighting); "volume" -> w_i = sum_{j in C_i} V_j /
            sum_j V_j (block-volume weighting, V_j = Sj*_j^3). Both
            weights are stored on every class result (``weight_count``,
            ``weight_volume``); ``weight`` is the one selected here and
            used in the mixture P_KCO*(x) = sum_i w_i P_i(x).

    Returns:
        KCOStarResult: The full, traceable KCO* prediction.

    Raises:
        ValueError: On any invalid input, propagated unmodified from the
            reused kco_model.py functions, or if the resulting Xmax does
            not exceed X50 for some class.
    """
    warnings = list(_validate_design(design))

    # ---- JPA resolution: duplicated input-dispatch logic from
    # predict_kco (not a standalone function in kco_model.py; no
    # equation is altered here, only which categorical input is read). ----
    if design.jpa_rating is not None:
        if design.jpa_rating not in (20, 30, 40):
            raise ValueError("jpa_rating must be 20, 30 or 40")
        jpa = design.jpa_rating
    elif design.jpa_case is not None:
        jpa = _joint_plane_angle_rating(design.jpa_case,
                                       design.jpa_mapping_version)
    else:
        raise ValueError("supply either jpa_case or jpa_rating")

    if design.jf_includes_jcf:
        if design.joint_condition is None:
            raise ValueError("jf_includes_jcf=True requires joint_condition")
        jcf: Optional[float] = joint_condition_factor(design.joint_condition)
    else:
        jcf = None

    # ---- constant (Sj*-independent) terms, computed once ----
    rdi = rock_density_influence(design.rock_density_kg_m3)
    hf = hardness_factor(design.youngs_modulus_gpa, design.ucs_mpa)

    # ---- powder factor: duplicated reported-vs-calculated dispatch
    # from predict_kco (same reasoning as JPA above). ----
    q_calculated = calculate_powder_factor(design.charge_per_hole_kg,
                                           design.burden_m,
                                           design.spacing_m,
                                           design.bench_height_m)
    q_used = (design.powder_factor_reported_kg_m3
              if design.powder_factor_mode == "reported"
              else q_calculated)
    if q_used is None:
        raise ValueError("powder_factor_mode='reported' requires "
                         "powder_factor_reported_kg_m3")

    # ---- 1. Sj* distribution and classes (KCO* input) ----
    sj_star_dist = sj_star_distribution_from_block_volumes(sj_block_volumes_m3)
    if class_method == "percentile":
        classes_in = create_sj_percentile_classes(
            sj_star_dist,
            percentile_edges=percentile_edges,
            representative_method=representative_method,
        )
    elif class_method == "log_bins":
        classes_in, log_bin_warnings = create_sj_log_bins(
            sj_star_dist,
            n_bins=n_log_bins,
            representative_method=representative_method,
        )
        warnings.extend(log_bin_warnings)
    elif class_method == "log_bins_p1_p99_tails":
        classes_in, log_bin_warnings = create_sj_log_bins_p1_p99_tails(
            sj_star_dist,
            n_bins=n_log_bins,
            tail_low_pct=tail_low_pct,
            tail_high_pct=tail_high_pct,
            representative_method=representative_method,
        )
        warnings.extend(log_bin_warnings)
    else:
        raise ValueError(
            "class_method must be 'percentile', 'log_bins' or "
            "'log_bins_p1_p99_tails'"
        )
    if not classes_in:
        raise ValueError("no non-empty Sj* classes were produced; check "
                         "percentile_edges/n_log_bins against the "
                         "population size")

    # ---- 1b. class weights: block-count (N_i/N_tot) and block-volume
    # (sum V_j in class / sum V_j total, with V_j = Sj*_j^3) ----
    if weight_method not in ("count", "volume"):
        raise ValueError("weight_method must be 'count' or 'volume'")
    class_vol_sums = np.array(
        [float(np.sum(np.asarray(c.sj_values_m, dtype=float) ** 3))
         for c in classes_in], dtype=float)
    total_vol = float(np.sum(sj_star_dist.block_volume_m3))
    if not np.isclose(class_vol_sums.sum(), total_vol, rtol=1e-9, atol=0.0):
        raise ValueError(
            f"internal error: class volume sums {class_vol_sums.sum():.9g} "
            f"!= total retained volume {total_vol:.9g}")
    weights_count = np.array([c.weight for c in classes_in], dtype=float)
    weights_volume = class_vol_sums / total_vol
    if abs(weights_volume.sum() - 1.0) > 1e-9:
        raise ValueError(
            f"internal error: volume weights sum to {weights_volume.sum():.9f}")
    weights_used = weights_volume if weight_method == "volume" else weights_count

    # ---- 2. Xmax: separate issue, computed once, shared by all classes ----
    block_size_m, xmax_info = characteristic_block_size_from_distribution(
        xmax_block_volumes_m3,
        block_size_method=xmax_block_size_method,
        block_percentile=xmax_block_percentile,
    )
    warnings.extend(xmax_info["warnings"])
    xmax_m, governed_by = xmax_kco(block_size_m, design.burden_m,
                                   design.spacing_m)
    xmax_mm = xmax_m * 1000.0

    # ---- 3. propagate each Sj* class through the published KCO chain ----
    class_results: list[KCOStarClassResult] = []
    for k, cin in enumerate(classes_in):
        jps_i = jps_from_joint_spacing(cin.sj_representative_m,
                                       design.burden_m, design.spacing_m)
        jf_i = joint_factor(jps_i, jpa, jcf)
        rmd_i = rock_mass_description(design.rock_mass_case, jf=jf_i)
        bi_i = blastability_index(rmd_i, rdi, hf)
        a_i = rock_factor_A(bi_i)
        if a_i <= 0:
            raise ValueError(
                f"class [{cin.percentile_low:.1f}-{cin.percentile_high:.1f}%]: "
                f"rock factor A = {a_i:.3g} is not positive"
            )

        n_i = uniformity_index_n(
            burden_m=design.burden_m,
            spacing_m=design.spacing_m,
            hole_diameter_mm=design.hole_diameter_mm,
            drill_accuracy_sd_m=design.drill_accuracy_sd_m,
            charge_length_m=design.total_charge_m,
            bench_height_m=design.bench_height_m,
            rock_factor_A=a_i,
            timing_scatter_factor_ns=design.timing_scatter_factor_ns,
        )
        g_n_i = shift_factor_g(n_i, mode=design.shift_factor_mode)
        x50_i_cm = x50_kuznetsov(
            rock_factor_a=a_i,
            charge_per_hole_kg=design.charge_per_hole_kg,
            powder_factor_kg_m3=q_used,
            s_anfo_pct=design.s_anfo_pct,
            g_n=g_n_i,
            timing_factor=design.timing_factor,
        )
        x50_i_mm = x50_i_cm * 10.0

        if xmax_mm <= x50_i_mm:
            raise ValueError(
                f"class [{cin.percentile_low:.1f}-{cin.percentile_high:.1f}%]: "
                f"X50 = {x50_i_mm:.0f} mm is not smaller than the shared "
                f"Xmax = {xmax_mm:.0f} mm (governed by {governed_by}); "
                "check the Xmax percentile/method choice against this "
                "class's representative Sj*"
            )
        b_i = b_parameter(xmax_mm, x50_i_mm, n_i)
        if b_i < B_TYPICAL_MIN or b_i > B_TYPICAL_MAX:
            warnings.append(
                f"class [{cin.percentile_low:.1f}-{cin.percentile_high:.1f}%]: "
                f"b = {b_i:.2f} is outside the typical Swebrec range "
                f"{B_TYPICAL_MIN:.0f}-{B_TYPICAL_MAX:.0f}"
            )

        x20_i, x50_check_i, x80_i = (
            float(swebrec_size_at_passing(p, x50_i_mm, xmax_mm, b_i))
            for p in (20.0, 50.0, 80.0)
        )

        class_results.append(KCOStarClassResult(
            percentile_low=cin.percentile_low,
            percentile_high=cin.percentile_high,
            weight=float(weights_used[k]),
            n_blocks=cin.n_blocks,
            volume_min_m3=cin.volume_min_m3,
            volume_max_m3=cin.volume_max_m3,
            sj_min_m=cin.sj_min_m,
            sj_max_m=cin.sj_max_m,
            sj_representative_m=cin.sj_representative_m,
            jps=jps_i, jpa=jpa, jf=jf_i, rmd=rmd_i, rdi=rdi, hf=hf,
            bi=bi_i, rock_factor=a_i, n=n_i, g_n=g_n_i,
            x50_mm=x50_i_mm, xmax_mm=xmax_mm, b=b_i,
            x20_mm=x20_i, x50_check_mm=x50_check_i, x80_mm=x80_i,
            bin_edge_low_m=cin.bin_edge_low_m,
            bin_edge_high_m=cin.bin_edge_high_m,
            is_tail=cin.is_tail,
            weight_count=float(weights_count[k]),
            weight_volume=float(weights_volume[k]),
            volume_sum_m3=float(class_vol_sums[k]),
        ))

    total_w = sum(c.weight for c in class_results)
    if abs(total_w - 1.0) > 1e-6:
        raise ValueError(f"internal error: class weights sum to {total_w:.6f}, "
                         "expected 1")

    return KCOStarResult(
        design=design,
        classes=class_results,
        percentile_edges=list(percentile_edges),
        representative_method=representative_method,
        xmax_block_percentile=xmax_block_percentile,
        xmax_block_size_method=xmax_block_size_method,
        xmax_mm=xmax_mm,
        xmax_governed_by=governed_by,
        n_retained=sj_star_dist.n_retained,
        n_rejected=sj_star_dist.n_rejected,
        sj_star_dist=sj_star_dist,
        warnings=warnings,
        class_method=class_method,
        n_log_bins=(n_log_bins if class_method in
                   ("log_bins", "log_bins_p1_p99_tails") else None),
        tail_low_pct=(tail_low_pct if class_method == "log_bins_p1_p99_tails"
                     else None),
        tail_high_pct=(tail_high_pct if class_method == "log_bins_p1_p99_tails"
                      else None),
        weight_method=weight_method,
    )


# ============================================================
# 8. SHIFT FROM IN-SITU TO POST-BLAST (percentile deltas/ratios)
# ============================================================
def compare_in_situ_to_postblast(x_post: dict, x_in_situ: dict) -> dict:
    """
    Quantify the change between in-situ (pre-blast) and post-blast
    (predicted or measured) percentile sizes.

    Sign convention: DeltaX_P = X_P_post - X_P_in_situ. A NEGATIVE
    DeltaX_P means the post-blast size at that percentile is SMALLER
    than the in-situ size (fragmentation reduced block size), a
    POSITIVE DeltaX_P means it is larger. This convention is stated
    explicitly and not interpreted physically beyond this definition.

    Args:
        x_post: dict with keys like "X20", "X50", "X80" -> size (mm),
            for classical KCO, KCO*, or WipFrag.
        x_in_situ: dict with the same keys -> in-situ size (mm).

    Returns:
        dict: For every key present in both inputs:
            "delta_<key>_mm", "ratio_<key>", "relative_change_pct".

    Raises:
        ValueError: If no common keys exist, or an in-situ value is 0.
    """
    common = sorted(set(x_post) & set(x_in_situ))
    if not common:
        raise ValueError("x_post and x_in_situ share no common percentile keys")
    out: dict = {}
    for key in common:
        post = float(x_post[key])
        pre = float(x_in_situ[key])
        if pre == 0:
            raise ValueError(f"in-situ value for {key} is 0; cannot compute ratio")
        out[f"delta_{key}_mm"] = post - pre
        out[f"ratio_{key}"] = post / pre
        out[f"relative_change_{key}_pct"] = 100.0 * (post - pre) / pre
    return out


# ============================================================
# 9. VALIDATION AGAINST MEASURED (WipFrag) FRAGMENTATION
# ============================================================
def compare_kco_star_with_measured_fragmentation(
        result: KCOStarResult,
        measured_sizes_mm,
        measured_passing_pct) -> dict:
    """
    Compare a KCO* prediction against measured post-blast fragmentation.

    Thin wrapper that reuses
    :func:`kco_model.compare_with_measured_fragmentation` UNMODIFIED:
    ``KCOStarResult`` implements the same ``.passing(x_mm)`` and
    ``.size_at(passing_pct)`` interface as ``KCOResult``, so the
    existing validation logic (RMSE/MAE/R^2, X20/X50/X80 errors)
    applies without any change.

    Args:
        result: A KCOStarResult from :func:`predict_kco_star`.
        measured_sizes_mm: Measured fragment sizes (mm).
        measured_passing_pct: Measured cumulative percentage passing.

    Returns:
        dict: Same metric set as
        :func:`kco_model.compare_with_measured_fragmentation`.
    """
    return compare_with_measured_fragmentation(
        result, measured_sizes_mm, measured_passing_pct)


def build_kco_vs_kco_star_comparison_table(
        kco_result: KCOResult,
        kco_star_result: KCOStarResult,
        measured_sizes_mm,
        measured_passing_pct) -> dict:
    """
    Build a direct classical-KCO vs KCO* comparison against measured
    (WipFrag) fragmentation.

    Reuses :func:`kco_model.compare_with_measured_fragmentation` for
    classical KCO and :func:`compare_kco_star_with_measured_fragmentation`
    for KCO*; no metric is recomputed independently.

    Args:
        kco_result: A KCOResult from kco_model.predict_kco.
        kco_star_result: A KCOStarResult from :func:`predict_kco_star`.
        measured_sizes_mm: Measured fragment sizes (mm).
        measured_passing_pct: Measured cumulative percentage passing.

    Returns:
        dict: {"KCO": {...metrics...}, "KCO*": {...metrics...}}.
    """
    return {
        "KCO": compare_with_measured_fragmentation(
            kco_result, measured_sizes_mm, measured_passing_pct),
        "KCO*": compare_kco_star_with_measured_fragmentation(
            kco_star_result, measured_sizes_mm, measured_passing_pct),
    }


# ============================================================
# 10. PLOTS (no auto-display; caller decides when to show/save)
# ============================================================
def plot_dfn_block_distribution(sj_star_dist: SjStarDistribution, ax=None):
    """Plot 1: cumulative DFN block-volume/size distribution."""
    import matplotlib.pyplot as plt
    if ax is None:
        _, ax = plt.subplots()
    v_sorted = np.sort(sj_star_dist.block_volume_m3)
    n = v_sorted.size
    cdf = 100.0 * (np.arange(1, n + 1) - 0.5) / n
    ax.plot(v_sorted, cdf, marker=".", linestyle="none", markersize=3)
    ax.set_xscale("log")
    ax.set_xlabel("DFN block volume (m3)")
    ax.set_ylabel("Cumulative passing (%)")
    ax.set_title("In-situ DFN block-volume distribution")
    return ax


def plot_sj_star_classes(sj_star_dist: SjStarDistribution,
                         classes: list, ax=None):
    """Plot 2: Sj* cumulative distribution with percentile classes marked."""
    import matplotlib.pyplot as plt
    if ax is None:
        _, ax = plt.subplots()
    x_mm, p = in_situ_passing_from_sj_star(sj_star_dist)
    ax.plot(x_mm, p, color="black", lw=1, label="Sj* empirical CDF")
    for c in classes:
        ax.axvspan(c.sj_min_m * 1000.0, c.sj_max_m * 1000.0,
                  alpha=0.15)
        ax.axvline(c.sj_representative_m * 1000.0, color="red",
                  lw=0.5, linestyle="--")
    ax.set_xscale("log")
    ax.set_xlabel("Sj* = V^(1/3) (mm)")
    ax.set_ylabel("Cumulative passing (%)")
    ax.set_title("Sj* distribution with percentile classes (KCO* input)")
    ax.legend()
    return ax


def plot_kco_star_classes_and_curve(result: KCOStarResult, ax=None,
                                    n_points: int = 500):
    """Plot 3: all class-specific KCO curves (thin) + final KCO* curve."""
    import matplotlib.pyplot as plt
    if ax is None:
        _, ax = plt.subplots()
    x_grid = np.geomspace(1.0, result.xmax_mm, n_points)
    for c in result.classes:
        ax.plot(x_grid, c.passing(x_grid), lw=0.6, alpha=0.5, color="gray")
    ax.plot(x_grid, result.passing(x_grid), lw=2, color="blue",
           label="KCO* (weighted mixture)")
    ax.set_xscale("log")
    ax.set_xlabel("Fragment size (mm)")
    ax.set_ylabel("Cumulative passing (%)")
    ax.set_title("KCO* class curves and combined distribution")
    ax.legend()
    return ax


def plot_main_comparison(sj_star_dist: SjStarDistribution,
                         kco_result: Optional[KCOResult],
                         kco_star_result: Optional[KCOStarResult],
                         measured_sizes_mm=None,
                         measured_passing_pct=None,
                         ax=None, n_points: int = 500,
                         show_class_curves: bool = False,
                         show_envelope: bool = False):
    """
    Plot 4: in-situ DFN (pre-blast) vs classical KCO vs KCO* vs WipFrag
    (post-blast), all on the same fragment-size axis. The in-situ curve
    represents pre-blast block structure; KCO/KCO*/WipFrag represent
    predicted or measured post-blast fragmentation.

    Optional:
        show_class_curves: draw every individual class Swebrec curve
            P_i(x) as a thin, semi-transparent line.
        show_envelope: shade the KCO* class envelope ("fuseau") between
            P_min(x) = min_i P_i(x) and P_max(x) = max_i P_i(x). This is
            the range of the individual class curves, not a confidence
            interval around the weighted mixture.
    """
    import matplotlib.pyplot as plt
    if ax is None:
        _, ax = plt.subplots()
    x_in_situ, p_in_situ = in_situ_passing_from_sj_star(sj_star_dist)
    ax.plot(x_in_situ, p_in_situ, color="black", lw=1.5,
           label="In-situ DFN (pre-blast, Sj*)")
    if kco_star_result is not None and (show_class_curves or show_envelope):
        x_grid = np.geomspace(1.0, kco_star_result.xmax_mm, n_points)
        if show_envelope:
            p_min, p_max = kco_star_result.envelope(x_grid)
            ax.fill_between(x_grid, p_min, p_max, color="tab:blue",
                            alpha=0.18, lw=0, zorder=1,
                            label="KCO* class envelope (min-max of P_i)")
        if show_class_curves:
            groups = kco_star_result.unique_curves()
            n_cls = len(kco_star_result.classes)
            cmap = plt.get_cmap("viridis")
            markers = ["o", "s", "^", "v", "D", "P", "X", "*", "<", ">",
                       "h", "p"]
            linestyles = ["-", "--", "-.", ":"]
            # Every class is plotted individually with its own colour,
            # line style and marker. Classes with identical (X50, Xmax, b)
            # fall on exactly the same curve; their markers are staggered
            # along x (markevery offset = rank in the group) so that each
            # coincident class remains individually identifiable without
            # altering any numerical value.
            group_of = {}
            for g in groups:
                for rank, k in enumerate(g["class_indices"]):
                    group_of[k] = (rank, len(g["class_indices"]))
            class_handles, class_labels = [], []
            for i, c in enumerate(kco_star_result.classes, start=1):
                rank, size = group_of[i]
                step = max(n_points // 10, 1)
                offset = int(round(rank * step / max(size, 1)))
                tail = (f" [{c.is_tail} tail]" if c.is_tail else "")
                label = (f"C{i:>2}{tail}: Sj*={c.sj_representative_m:.3f} m, "
                         f"JPS={c.jps}, X50={c.x50_mm:.0f} mm, "
                         f"w={c.weight:.4f}")
                ln, = ax.plot(x_grid, c.passing(x_grid),
                              color=cmap((i - 1) / max(n_cls - 1, 1)),
                              lw=0.9, alpha=0.9, zorder=7,
                              linestyle=linestyles[(i - 1) % len(linestyles)],
                              marker=markers[(i - 1) % len(markers)],
                              markersize=4.5, markevery=(offset, step),
                              markerfacecolor="white", markeredgewidth=0.9,
                              label="_" + label)  # hidden from main legend
                class_handles.append(ln)
                class_labels.append(label)
            overlap_lines = []
            for g in groups:
                idx = g["class_indices"]
                if len(idx) > 1:
                    overlap_lines.append(
                        "C" + ", C".join(str(k) for k in idx)
                        + f" coincide exactly (JPS={g['jps']})")
            title = (f"Individual KCO* class curves P_i(x): {n_cls} classes "
                     f"computed, {len(groups)} distinct"
                     + ("\n" + "\n".join(overlap_lines) if overlap_lines
                        else ""))
            # Figure-level legend (not ax.add_artist) so that it is kept
            # by savefig(bbox_inches="tight") when placed outside the axes.
            ax.figure.legend(handles=class_handles, labels=class_labels,
                             loc="center left", bbox_to_anchor=(1.01, 0.5),
                             bbox_transform=ax.transAxes, fontsize=7,
                             title=title, title_fontsize=7.5, frameon=True)
    if kco_result is not None:
        x_grid = np.geomspace(1.0, kco_result.xmax_mm, n_points)
        ax.plot(x_grid, kco_result.passing(x_grid), color="green", lw=1.5,
               label="Classical KCO (post-blast, predicted)")
    if kco_star_result is not None:
        x_grid = np.geomspace(1.0, kco_star_result.xmax_mm, n_points)
        wm = ("volume-weighted" if kco_star_result.weight_method == "volume"
              else "count-weighted")
        ax.plot(x_grid, kco_star_result.passing(x_grid), color="blue", lw=2.5,
               zorder=6, label=f"KCO* ({wm}, post-blast, predicted)")
    if measured_sizes_mm is not None and measured_passing_pct is not None:
        ax.plot(measured_sizes_mm, measured_passing_pct, "o--", color="red",
               lw=1, label="WipFrag (post-blast, measured)")
    ax.set_xscale("log")
    ax.set_xlabel("Fragment size (mm)")
    ax.set_ylabel("Cumulative passing (%)")
    ax.set_title("In-situ vs classical KCO vs KCO* vs measured (WipFrag)")
    ax.legend()
    return ax


def plot_percentile_shift(shift_data: dict, ax=None):
    """
    Plot 5: percentile shift (X20, X50, X80) across in-situ / KCO / KCO*
    / WipFrag.

    Args:
        shift_data: dict like {"In-situ": {"X20":.., "X50":.., "X80":..},
            "KCO": {...}, "KCO*": {...}, "WipFrag": {...}}. Any subset
            of series/percentiles may be supplied.
    """
    import matplotlib.pyplot as plt
    if ax is None:
        _, ax = plt.subplots()
    percentiles = ["X20", "X50", "X80"]
    for series_name, values in shift_data.items():
        ys = [values.get(p, np.nan) for p in percentiles]
        ax.plot(percentiles, ys, marker="o", label=series_name)
    ax.set_ylabel("Fragment size (mm)")
    ax.set_title("Percentile shift: in-situ vs KCO vs KCO* vs measured")
    ax.legend()
    return ax


# ============================================================
# 11. SELF-TEST
# ============================================================
def _reference_design() -> BlastDesign:
    """Bårarp-round-4-based BlastDesign used only for KCO* self-tests."""
    return BlastDesign(
        name="KCO* self-test",
        hole_diameter_mm=51.0, burden_m=1.8, spacing_m=2.2,
        bench_height_m=5.2, total_charge_m=3.9, bottom_charge_m=3.9,
        column_charge_m=0.0, drill_accuracy_sd_m=0.25,
        charge_per_hole_kg=9.24, powder_factor_reported_kg_m3=0.55,
        powder_factor_mode="reported", s_anfo_pct=62.2,
        rock_density_kg_m3=2700.0, ucs_mpa=100.0, youngs_modulus_gpa=60.0,
        rock_mass_case="jointed", jpa_case="strike_perpendicular_to_face",
        timing_scatter_factor_ns=1.0, shift_factor_mode="no_shift",
    )


def self_test_kco_star(verbose: bool = True) -> bool:
    """
    Regression/consistency checks for the KCO* implementation.

    Does NOT validate KCO* against any field site; checks internal
    consistency and that classical KCO (kco_model.py) is unaffected.

    Returns:
        bool: True if every check passes.
    """
    ok = True

    def check(label, passed):
        nonlocal ok
        ok = ok and passed
        if verbose:
            print(f"{label:<55}{'OK' if passed else 'FAIL':>6}")

    # 1-2. Sj* = V^(1/3) on simple cases.
    dist = sj_star_distribution_from_block_volumes([1.0, 0.125])
    check("V=1 -> Sj*=1", abs(float(dist.sj_star_m[0]) - 1.0) < 1e-9)
    check("V=0.125 -> Sj*=0.5", abs(float(dist.sj_star_m[1]) - 0.5) < 1e-9)

    # 3. invalid V <= 0 (and non-finite) rejected/filtered.
    dist2 = sj_star_distribution_from_block_volumes(
        [1.0, 0.0, -2.0, float("nan"), float("inf"), 8.0])
    check("invalid V<=0/non-finite filtered",
         dist2.n_retained == 2 and dist2.n_rejected == 4)

    # Synthetic DFN block-volume population for the class/curve checks.
    rng = np.random.default_rng(42)
    sj_volumes = rng.lognormal(mean=math.log(0.15), sigma=0.9, size=600)
    sj_volumes = sj_volumes[(sj_volumes > 1e-4) & (sj_volumes < 3.0)]
    xmax_volumes = rng.lognormal(mean=math.log(0.6), sigma=0.9, size=600)
    xmax_volumes = xmax_volumes[(xmax_volumes > 1e-3) & (xmax_volumes < 15.0)]

    design = _reference_design()
    result = predict_kco_star(
        design, sj_volumes, xmax_volumes,
        xmax_block_size_method="equivalent_cube",
        xmax_block_percentile=95,
        percentile_edges=[0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
        representative_method="median",
    )

    # 4/5/6/7. classes cover the population exactly once, weights sum to 1.
    n_sum = sum(c.n_blocks for c in result.classes)
    check("classes cover full retained population",
         n_sum == result.n_retained)
    check("class weights sum to 1",
         abs(sum(c.weight for c in result.classes) - 1.0) < 1e-9)

    sj_dist = result.sj_star_dist
    order = np.argsort(sj_dist.sj_star_m)
    sj_sorted = sj_dist.sj_star_m[order]
    covered = np.zeros(sj_sorted.size, dtype=bool)
    pos = 0
    no_overlap = True
    for c in result.classes:
        n = c.n_blocks
        if np.any(covered[pos:pos + n]):
            no_overlap = False
        covered[pos:pos + n] = True
        pos += n
    check("no block in two classes / every block in one class",
         no_overlap and covered.all())

    # 8/9. KCO* passing curve in [0,100] and monotonic.
    x_grid = np.geomspace(1.0, result.xmax_mm, 500)
    p_grid = result.passing(x_grid)
    check("KCO* passing in [0,100]",
         bool(np.all(p_grid >= -1e-9) and np.all(p_grid <= 100.0 + 1e-9)))
    check("KCO* passing monotonic non-decreasing",
         bool(np.all(np.diff(p_grid) >= -1e-9)))

    # 10. inversion gives P(X50*) ~ 50%.
    x50_star = float(np.atleast_1d(result.size_at(50.0))[0])
    p_at_x50_star = float(np.atleast_1d(result.passing(x50_star))[0])
    check("P(X50*) approx 50%", abs(p_at_x50_star - 50.0) < 1.0)

    # 11. one-class KCO* behaves like the direct KCO chain with that Sj.
    result_one = predict_kco_star(
        design, sj_volumes, xmax_volumes,
        xmax_block_size_method="equivalent_cube",
        xmax_block_percentile=95,
        percentile_edges=[0, 100],
        representative_method="median",
    )
    c = result_one.classes[0]
    jps_ref = jps_from_joint_spacing(c.sj_representative_m,
                                    design.burden_m, design.spacing_m)
    jpa_ref = _joint_plane_angle_rating(design.jpa_case,
                                       design.jpa_mapping_version)
    jf_ref = joint_factor(jps_ref, jpa_ref)
    rmd_ref = rock_mass_description(design.rock_mass_case, jf=jf_ref)
    rdi_ref = rock_density_influence(design.rock_density_kg_m3)
    hf_ref = hardness_factor(design.youngs_modulus_gpa, design.ucs_mpa)
    bi_ref = blastability_index(rmd_ref, rdi_ref, hf_ref)
    a_ref = rock_factor_A(bi_ref)
    check("one-class KCO* matches direct KCO chain (rock factor A)",
         abs(c.rock_factor - a_ref) < 1e-9)

    # 12. changing only Sj* changes only Sj*-downstream quantities.
    result_alt = predict_kco_star(
        design, xmax_volumes, xmax_volumes,  # different Sj* population
        xmax_block_size_method="equivalent_cube",
        xmax_block_percentile=95,
        percentile_edges=[0, 100],
        representative_method="median",
    )
    c_alt = result_alt.classes[0]
    check("Sj*-independent terms unchanged (RDI)",
         abs(c.rdi - c_alt.rdi) < 1e-9)
    check("Sj*-independent terms unchanged (HF)",
         abs(c.hf - c_alt.hf) < 1e-9)
    check("Sj*-dependent term differs (Sj representative)",
         abs(c.sj_representative_m - c_alt.sj_representative_m) > 1e-9)

    # 13. normal KCO (kco_model.py) unaffected by KCO* module.
    import kco_model
    check("classical KCO self-test unaffected", kco_model.self_test(verbose=False))

    # 14-17. log_bins alternative class_method: coverage, weights, no
    # overlap, and every block assigned to exactly one bin.
    result_log = predict_kco_star(
        design, sj_volumes, xmax_volumes,
        xmax_block_size_method="equivalent_cube",
        xmax_block_percentile=95,
        class_method="log_bins",
        n_log_bins=10,
        representative_method="median",
    )
    n_sum_log = sum(c.n_blocks for c in result_log.classes)
    check("log_bins: classes cover full retained population",
         n_sum_log == result_log.n_retained)
    check("log_bins: class weights sum to 1",
         abs(sum(c.weight for c in result_log.classes) - 1.0) < 1e-9)

    check("log_bins: Sj_min/Sj_max non-overlapping and ordered",
         bool(np.all(np.diff(
             [c.sj_min_m for c in result_log.classes]) > 0)) and
         all(c.sj_min_m <= c.sj_representative_m <= c.sj_max_m
             for c in result_log.classes))
    check("log_bins: bin edges non-decreasing across classes",
         bool(np.all(np.diff(
             [c.bin_edge_low_m for c in result_log.classes]) >= 0)))
    check("log_bins: KCO* passing in [0,100]",
         bool(np.all(result_log.passing(
             np.geomspace(1.0, result_log.xmax_mm, 300)) >= -1e-9)))

    # 18-23. log_bins_p1_p99_tails: 2 explicit tail classes + coverage,
    # weights, tail flags, and P1/P99 boundaries.
    result_tails = predict_kco_star(
        design, sj_volumes, xmax_volumes,
        xmax_block_size_method="equivalent_cube",
        xmax_block_percentile=95,
        class_method="log_bins_p1_p99_tails",
        n_log_bins=10,
        tail_low_pct=1.0,
        tail_high_pct=99.0,
        representative_method="median",
    )
    n_sum_tails = sum(c.n_blocks for c in result_tails.classes)
    check("log_bins_p1_p99_tails: classes cover full retained population",
         n_sum_tails == result_tails.n_retained)
    check("log_bins_p1_p99_tails: class weights sum to 1",
         abs(sum(c.weight for c in result_tails.classes) - 1.0) < 1e-9)
    tail_flags = [c.is_tail for c in result_tails.classes]
    check("log_bins_p1_p99_tails: exactly one lower and one upper tail class",
         tail_flags.count("lower") == 1 and tail_flags.count("upper") == 1)
    check("log_bins_p1_p99_tails: lower tail is first, upper tail is last",
         tail_flags[0] == "lower" and tail_flags[-1] == "upper")
    p1_ref = float(np.percentile(sj_dist_tails_sorted := np.sort(
        sj_star_distribution_from_block_volumes(sj_volumes).sj_star_m), 1.0))
    p99_ref = float(np.percentile(sj_dist_tails_sorted, 99.0))
    lower_class = result_tails.classes[0]
    upper_class = result_tails.classes[-1]
    check("log_bins_p1_p99_tails: lower tail upper edge == P1",
         abs(lower_class.bin_edge_high_m - p1_ref) < 1e-9)
    check("log_bins_p1_p99_tails: upper tail lower edge == P99",
         abs(upper_class.bin_edge_low_m - p99_ref) < 1e-9)
    check("log_bins_p1_p99_tails: KCO* passing in [0,100]",
         bool(np.all(result_tails.passing(
             np.geomspace(1.0, result_tails.xmax_mm, 300)) >= -1e-9)))

    # 24-28. volume weighting: sums to 1, equals sum(V)/total, count
    # weights preserved, class chain (X50_i) unchanged, envelope brackets
    # the mixture.
    result_vol = predict_kco_star(
        design, sj_volumes, xmax_volumes,
        xmax_block_size_method="equivalent_cube",
        xmax_block_percentile=95,
        class_method="log_bins_p1_p99_tails",
        n_log_bins=10,
        representative_method="median",
        weight_method="volume",
    )
    check("volume weights sum to 1",
         abs(sum(c.weight for c in result_vol.classes) - 1.0) < 1e-9)
    v_tot = float(np.sum(result_vol.sj_star_dist.block_volume_m3))
    check("volume weight_i == sum(V in class)/sum(V)",
         all(abs(c.weight - c.volume_sum_m3 / v_tot) < 1e-12
             for c in result_vol.classes))
    check("count weights still stored and sum to 1",
         abs(sum(c.weight_count for c in result_vol.classes) - 1.0) < 1e-9)
    check("per-class X50 unchanged by weighting method",
         all(abs(a.x50_mm - b.x50_mm) < 1e-9
             for a, b in zip(result_tails.classes, result_vol.classes)))
    xg = np.geomspace(1.0, result_vol.xmax_mm, 300)
    p_lo, p_hi = result_vol.envelope(xg)
    p_mix = result_vol.passing(xg)
    check("envelope P_min <= P_KCO* <= P_max",
         bool(np.all(p_lo - 1e-9 <= p_mix) and np.all(p_mix <= p_hi + 1e-9)))

    if verbose:
        print("\nAll KCO* checks passed." if ok else "\nSOME KCO* CHECKS FAILED.")
    return ok


if __name__ == "__main__":
    raise SystemExit(0 if self_test_kco_star() else 1)

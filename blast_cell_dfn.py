# ============================================================
# blast_cell_dfn.py
#
# S x B x H "blast-cell" DFN workflow for KCO*.
#
# THIS MODULE DOES NOT MODIFY run_site.py OR THE EXISTING 50x50x50 m
# VARENNE DFN. That large-domain workflow (sweep, P32<->P21 field
# calibration, block/persistence export) is left completely untouched
# and remains the source of truth for the calibrated fracture intensity.
#
# What this module does instead:
#   1. Reads the ALREADY-CALIBRATED P32 per family from the output of the
#      existing 50x50x50 m pipeline
#      (outputs/<site>/02_calibration/P32_calibrated_summary.csv).
#      It does NOT recompute or re-calibrate P32; if that file is
#      missing or a family's calibration did not succeed, this module
#      raises an error rather than inventing a value.
#   2. Reuses, unchanged, the same structural DFN parameters as the
#      50x50x50 m run: fracture-family orientations (DIPSVARENNE.xlsx),
#      Fisher K per family (SITE_CONFIGS), and the Mauldon-corrected
#      fracture-size mean/sd (utils.persistence_params.load_size_params,
#      itself driven by a_terrain and the field trace CSV -- both
#      independent of domain size, see prior inspection).
#   3. Places fractures directly inside a NEW, smaller
#      S(spacing) x B(burden) x H(bench height) box until each family's
#      calibrated P32 is reached -- the identical per-fracture placement
#      rule used inside run_site.run_one() (orientation = field
#      measurement + Fisher-K jitter, diameter ~ lognormal(mean, sd),
#      uniform position in the box). That rule is a closure inside
#      run_site.py, not a standalone function, so it is DUPLICATED here
#      (not modified, not re-derived) in
#      :func:`_generate_one_blastcell_dfn`.
#   4. Generates MULTIPLE independent realisations (one per seed) of
#      this same S x B x H probabilistic block structure and pools their
#      clean block volumes into a single population, per your explicit
#      instruction. This pooling is the assumption that several
#      independent DFN realisations of one blast-cell's structural
#      statistics approximate the probabilistic in-situ block-volume
#      distribution for that cell; it is not a spatial tiling of
#      multiple physically distinct cells.
#
# No P21 / mapping-quad calibration is performed here: the calibrated
# P32 values already encode the field-matched fracture intensity from
# the large-domain run. This module only changes the DOMAIN SIZE in
# which that same intensity is realised.
#
# Axis convention (arbitrary, documented, has no effect on the
# resulting block-volume statistics because fracture orientations are
# sampled from absolute field dip/dip-direction measurements, not
# relative to the box axes -- see prior inspection):
#     region_x = spacing_m   (S)
#     region_y = burden_m    (B)
#     region_z = bench_height_m (H, vertical)
#
# Output (per call), under outputs/<site_key>/08_blastcell_SxBxH/:
#     blastcell_config.csv                    -- run parameters, audit
#     BlockVolumes_blastcell_seed<seed>.txt    -- one file per realisation
#     block_volumes_blastcell_pooled.csv       -- columns: seed, volume
#                                                  (this is what feeds
#                                                  sj_block_volumes_m3)
# ============================================================

from __future__ import annotations

import os

import numpy as np
import pandas as pd

from run_site import SITE_CONFIGS, SCRIPT_DIR
from external_libraries.unblocks import DFN, Generator
from utils.geometry import jitter_orientation
from utils.excel_loader import load_orientations_from_excel
from utils.persistence_params import load_size_params


def _load_calibrated_p32(calibration_summary_path: str,
                         family_ids: list) -> np.ndarray:
    """
    Load the already-calibrated P32 per family from the existing
    50x50x50 m pipeline's calibration output. Never recomputed here.

    Args:
        calibration_summary_path: Path to
            ``outputs/<site>/02_calibration/P32_calibrated_summary.csv``.
        family_ids: Family id order to align the returned array to
            (matches ``SITE_CONFIGS[site]["family_ids"]``).

    Returns:
        np.ndarray: Calibrated P32 (1/m) in ``family_ids`` order.

    Raises:
        FileNotFoundError: If the calibration summary does not exist
            (the 50x50x50 m pipeline has not been run yet).
        ValueError: If a required family is missing, any P32 value is
            non-finite, or any family's ``calibration_flag`` is not
            ``"ok"``.
    """
    if not os.path.exists(calibration_summary_path):
        raise FileNotFoundError(
            f"Calibrated P32 summary not found: {calibration_summary_path}\n"
            "This module reuses the existing 50x50x50 m VARENNE "
            "calibration; it does not recompute it. Run the large-domain "
            "pipeline first (python varenne.py, or run_site.py VARENNE)."
        )
    tbl = pd.read_csv(calibration_summary_path).set_index("fam")
    missing = [fid for fid in family_ids if fid not in tbl.index]
    if missing:
        raise ValueError(
            f"calibration summary {calibration_summary_path} is missing "
            f"family id(s) {missing}"
        )
    p32 = tbl.loc[family_ids, "P32_calibrated"].to_numpy(dtype=float)
    flags = tbl.loc[family_ids, "calibration_flag"].tolist()
    if not np.all(np.isfinite(p32)):
        raise ValueError(
            f"calibrated P32 in {calibration_summary_path} contains a "
            "non-finite value; cannot proceed"
        )
    bad = [(fid, fl) for fid, fl in zip(family_ids, flags) if fl != "ok"]
    if bad:
        raise ValueError(
            f"calibration_flag is not 'ok' for family/flag pairs {bad} in "
            f"{calibration_summary_path}; resolve the 50x50x50 m "
            "calibration before generating blast-cell realisations"
        )
    return p32


def _families_with_corrected_size(cfg: dict) -> list:
    """
    Return ``cfg["families"]`` with Mauldon-corrected mean/sd merged in.

    Duplicated from the equivalent merge block inside
    ``run_site.run_site`` (not a standalone function there); the
    correction itself is computed by the UNMODIFIED
    :func:`utils.persistence_params.load_size_params`, only the
    dict-merge glue is repeated here.
    """
    families = [dict(f) for f in cfg["families"]]
    size_params = load_size_params(
        trace_csv=os.path.join(SCRIPT_DIR, "assets", cfg["trace_name"]),
        site_name=cfg["site_name"],
        a_terrain=float(cfg["a_terrain"]),
        region_filter=cfg.get("region_filter"),
        family_names=[f["name"] for f in families],
    )
    for f in families:
        sp = size_params.get(f["name"])
        if sp:
            f["mean"] = sp["mean"]
            f["sd"] = sp["sd"]
    return families


def _generate_one_blastcell_dfn(region_x: float, region_y: float,
                                region_z: float, seed: int,
                                family_ids: list, families: list,
                                orient_by_fam: dict,
                                p32_targets: np.ndarray,
                                max_fracs: int) -> tuple:
    """
    Place fractures inside one region_x x region_y x region_z box until
    each family's ALREADY-CALIBRATED P32 target is reached.

    Same per-fracture placement rule as ``run_site.run_one()``
    (orientation = field measurement + Fisher-K jitter, diameter ~
    lognormal(mean, sd), uniform position in the box); duplicated here
    (not modified) because it is a closure inside run_site.py, not a
    standalone function. Unlike run_one(), NO quadrilateral/P21 mapping
    is added -- only ``add_VolumeMapping()``, since P32 is a whole-region
    volumetric measure and no field-P21 calibration is repeated here.

    Args:
        region_x, region_y, region_z: Box dimensions (m).
        seed: Random seed (numpy and the DFN library).
        family_ids: Family id list, in the same order as ``families``
            and ``p32_targets``.
        families: List of family dicts with "name", "fisher", "mean",
            "sd" (Mauldon-corrected).
        orient_by_fam: {fam_id: DataFrame[Dip, Dip Direction]}.
        p32_targets: Calibrated P32 target per family (1/m), same order
            as ``family_ids``.
        max_fracs: Safety cap on fractures added per family.

    Returns:
        tuple[DFN, list[str]]: The populated DFN object and any
        warnings (e.g. max_fracs reached for a family).
    """
    n_fam = len(families)
    warnings: list = []

    dfn = DFN()
    dfn.set_RegionMaxCorner([region_x, region_y, region_z])
    dfn.set_RandomSeed(int(seed))
    np.random.seed(int(seed))

    for _ in range(n_fam):
        dfn.add_FractureSet()
    dfn.add_VolumeMapping()

    added_counts = [0] * n_fam
    for i in range(n_fam):
        target = float(p32_targets[i])
        if target <= 0:
            continue

        fam_id = family_ids[i]
        f = families[i]
        mu, sd = float(f["mean"]), float(f["sd"])
        mu_log = np.log(mu ** 2 / np.sqrt(sd ** 2 + mu ** 2))
        sig_log = np.sqrt(np.log(1.0 + (sd / mu) ** 2))

        while dfn.volumesMapping[0].get_P32(i) < target:
            if added_counts[i] >= max_fracs:
                warnings.append(
                    f"seed {seed}: max_fracs={max_fracs} reached for "
                    f"fam{fam_id} before P32 target {target:.6f} was "
                    "reached"
                )
                break
            row = orient_by_fam[fam_id].sample(1).iloc[0]
            dip_val, dipdir_val = jitter_orientation(
                float(row["Dip"]), float(row["Dip Direction"]),
                fisher_k=f.get("fisher"),
            )
            diam = np.random.lognormal(mu_log, sig_log)
            cx = np.random.uniform(0.0, region_x)
            cy = np.random.uniform(0.0, region_y)
            cz = np.random.uniform(0.0, region_z)
            dfn.fractureSets[i].add_CircularFracture(
                [cx, cy, cz], dipdir_val, dip_val, diam / 2.0
            )
            added_counts[i] += 1

    return dfn, warnings


def generate_blast_cell_realizations(
        burden_m: float,
        spacing_m: float,
        bench_height_m: float,
        seeds,
        site_key: str = "VARENNE",
        calibration_summary_path: str = None,
        max_fracs: int = None,
        out_subdir: str = "08_blastcell_SxBxH",
) -> dict:
    """
    Generate multiple S x B x H blast-cell DFN realisations and pool
    their clean block volumes, for the KCO* probabilistic Sj*
    calculation.

    Reuses, unmodified: the calibrated P32 per family from the existing
    50x50x50 m pipeline, the field fracture orientations, Fisher K, and
    the Mauldon-corrected fracture-size mean/sd. Does not touch
    run_site.py or its outputs.

    Args:
        burden_m: Burden B (m). Must be > 0. No default -- must be the
            actual production-blast value.
        spacing_m: Hole spacing S (m). Must be > 0.
        bench_height_m: Bench height H (m). Must be > 0.
        seeds: Non-empty list of distinct random seeds, one DFN
            realisation per seed. Their pooled block volumes form the
            Sj* population.
        site_key: Key into ``run_site.SITE_CONFIGS`` to reuse structural
            parameters from. Default "VARENNE".
        calibration_summary_path: Path to the existing calibrated P32
            summary. Defaults to
            ``outputs/<site_key>/02_calibration/P32_calibrated_summary.csv``
            (the file produced by the existing 50x50x50 m pipeline).
        max_fracs: Safety cap on fractures added per family per
            realisation. Defaults to ``SITE_CONFIGS[site_key]["max_fracs"]``.
        out_subdir: Output subfolder name under
            ``outputs/<site_key>/``. Default "08_blastcell_SxBxH".

    Returns:
        dict:
            "pooled_volumes_m3": np.ndarray, all realisations combined.
            "per_seed_volumes_m3": {seed: np.ndarray}.
            "pooled_csv_path": str, path to the pooled CSV (columns
                "seed", "volume") -- pass its "volume" column as
                ``sj_block_volumes_m3`` to ``predict_kco_star``.
            "config_path": str, path to the audit CSV of run parameters.
            "out_root": str, output directory.
            "warnings": list[str].

    Raises:
        ValueError: If burden_m/spacing_m/bench_height_m are not > 0,
            or ``seeds`` is empty.
        FileNotFoundError: If the calibrated P32 summary does not exist.
    """
    if burden_m <= 0 or spacing_m <= 0 or bench_height_m <= 0:
        raise ValueError(
            "burden_m, spacing_m and bench_height_m must be > 0"
        )
    seeds = list(seeds)
    if not seeds:
        raise ValueError("seeds must be a non-empty list of distinct seeds")
    if len(set(seeds)) != len(seeds):
        raise ValueError("seeds must be distinct")

    cfg = SITE_CONFIGS[site_key]
    family_ids = list(cfg["family_ids"])
    max_fracs = int(max_fracs if max_fracs is not None
                    else cfg.get("max_fracs", 3000))

    if calibration_summary_path is None:
        calibration_summary_path = os.path.join(
            SCRIPT_DIR, "outputs", site_key, "02_calibration",
            "P32_calibrated_summary.csv"
        )
    p32_targets = _load_calibrated_p32(calibration_summary_path, family_ids)

    excel_path = os.path.join(SCRIPT_DIR, "assets", cfg["excel_name"])
    orient_by_fam = load_orientations_from_excel(
        excel_path, sheet=0, valid_families=family_ids
    )
    families = _families_with_corrected_size(cfg)

    # Axis convention: see module docstring. Arbitrary but documented;
    # does not affect the resulting block-volume statistics.
    region_x, region_y, region_z = spacing_m, burden_m, bench_height_m

    out_root = os.path.join(SCRIPT_DIR, "outputs", site_key, out_subdir)
    os.makedirs(out_root, exist_ok=True)

    all_warnings: list = []
    per_seed_volumes: dict = {}
    pooled_rows: list = []

    for seed in seeds:
        dfn, warns = _generate_one_blastcell_dfn(
            region_x, region_y, region_z, seed,
            family_ids, families, orient_by_fam, p32_targets, max_fracs,
        )
        all_warnings.extend(warns)

        gen = Generator()
        gen.generate_RockMass(dfn)
        vols = np.array([float(v) for v in gen.get_Volumes(True)],
                        dtype=np.float64)
        per_seed_volumes[int(seed)] = vols

        seed_txt = os.path.join(
            out_root, f"BlockVolumes_blastcell_seed{seed}.txt")
        np.savetxt(seed_txt, vols)

        for v in vols:
            pooled_rows.append({"seed": int(seed), "volume": float(v)})

    if not pooled_rows:
        raise ValueError(
            "no blocks were generated in any realisation; check "
            "burden_m/spacing_m/bench_height_m and the calibrated P32 "
            "values"
        )

    pooled_df = pd.DataFrame(pooled_rows)
    pooled_csv = os.path.join(out_root, "block_volumes_blastcell_pooled.csv")
    pooled_df.to_csv(pooled_csv, index=False)

    config_path = os.path.join(out_root, "blastcell_config.csv")
    pd.DataFrame([{
        "site_key": site_key,
        "spacing_m_S": spacing_m,
        "burden_m_B": burden_m,
        "bench_height_m_H": bench_height_m,
        "region_x_assigned_to": "spacing_m",
        "region_y_assigned_to": "burden_m",
        "region_z_assigned_to": "bench_height_m",
        "seeds": ";".join(str(s) for s in seeds),
        "calibration_summary_source": calibration_summary_path,
        "max_fracs": max_fracs,
        "n_realizations": len(seeds),
        "n_blocks_total": len(pooled_df),
    }]).to_csv(config_path, index=False)

    if all_warnings:
        print(f"⚠️ {len(all_warnings)} warning(s) during blast-cell "
             "generation; see returned 'warnings'.")

    return {
        "pooled_volumes_m3": pooled_df["volume"].to_numpy(dtype=float),
        "per_seed_volumes_m3": per_seed_volumes,
        "pooled_csv_path": pooled_csv,
        "config_path": config_path,
        "out_root": out_root,
        "warnings": all_warnings,
    }

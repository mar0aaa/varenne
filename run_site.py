# ============================================================
# run_site.py
#
# 1. En-tête, imports et configuration (lignes 1–120)
#   Lignes 1–19 : Commentaires d’en-tête expliquant le but du fichier : centraliser toute la chaîne d’analyse DFN pour chaque site, la calibration, l’export, etc. On précise aussi la structure des dossiers de sortie.
#   Lignes 21–38 : Imports des librairies standards (numpy, pandas, matplotlib) et des modules spécialisés du projet (DFN, outils de géométrie, régression, export, etc.).
#   Lignes 40–120 : Définition du dictionnaire SITE_CONFIGS : chaque site (BC1LEFT, BC1RIGHT, etc.) a sa propre configuration (surface, familles, fichiers, seeds, etc.). Cela permet de piloter tout le pipeline avec un seul objet.
# ============================================================

import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from external_libraries.unblocks import DFN, Generator
from external_libraries import plotTools

from utils.geometry import normal_from_dip_dipdir, build_center_quad, quad_area, jitter_orientation
from utils.regression import linear_fit_and_r2, calibrated_p32_from_fit, update_p32_guess_auto
from utils.persistence_params import load_size_params
from utils.prism import add_triangular_prism_Z, delete_fragments_inside_prism_vtk
from utils.export import export_block_volumes_simple, save_all_open_figures
from utils.excel_loader import load_orientations_from_excel
from utils.persistence import export_persistence

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()

# ============================================================
# SITE CONFIGURATIONS  — edit only this section
# ============================================================
# ============================================================
# SITE CONFIGURATION — Varenne (single site)
# ============================================================
SITE_CONFIGS = {

    "VARENNE": {
        "site_name":     "VARENNE",
        "a_terrain":     1480.61,          # m² — Average triangle surface from DIPS
        "excel_name":    "DIPSVARENNE.xlsx",
        "family_ids":    [1, 2, 3, 4],
        "families": [
            # mean = Mauldon (1990) pooled circular scanline: π·Rc·Σn / (2·Σm)
            # sd   = CoV(raw traces) × mean
            dict(name="fam1", fisher=23.8756, mean=1.4332, sd=0.5044),  # Set 1m: F3 — 96 poles
            dict(name="fam2", fisher=30.0506, mean=1.6382, sd=0.4045),  # Set 2m: F1 — 242 poles
            dict(name="fam3", fisher=32.4488, mean=1.3765, sd=0.3048),  # Set 3m: F2 — 189 poles
            dict(name="fam4", fisher=38.5074, mean=1.8158, sd=0.6419),  # Set 4m: F4 — 63 poles
        ],
        "dip_face":      72.0,   # from DIPS
        "dipdir_face":   50.0,   # from DIPS
        "p32_guess":     [0.08, 0.12, 0.10, 0.07],
        "trace_name":    "varenne_traces.csv",
        "region_filter": "VARENNE",
        "factors":       list(np.geomspace(0.4, 2.5, 10)),
        "seeds":         [10, 30, 50],
        "max_fracs":     6000,
    },
}

def run_site(cfg):
        # ------------------------------------------------------------------------
        # 2. Fonction principale : run_site(cfg) (lignes 123–555)
        # a) Initialisation et création des dossiers (lignes 124–162)
        #    On extrait tous les paramètres nécessaires depuis le dictionnaire de configuration du site.
        #    On définit les chemins pour chaque dossier de sortie (sweep, calibration, VTK, volumes, blockométrie, persistance).
        #    On crée tous ces dossiers si besoin.
        # ------------------------------------------------------------------------
    """
    Run the full DFN workflow for one site configuration dict.
    
    REQUIRED cfg keys:
        site_name (str)       — Site identifier for output naming
        a_terrain (float)     — Terrain mapping area in m² (CloudCompare)
        excel_name (str)      — Excel file with dip/dipdir distributions (in assets/)
        family_ids (list)     — Family numbers to use [1,3,4] etc.
        families (list)       — List of dicts: name, fisher, mean, sd
        dip_face (float)      — Dip angle of mapping surface
        dipdir_face (float)   — Dip direction of mapping surface
        p32_guess (list)      — Initial P32 guesses per family
        trace_name (str)      — CSV trace file with fam/Length columns (in assets/)
        region_filter (str)   — Filter CSV by REGION column (or None)
    
    OPTIONAL cfg keys:
        region_x, region_y, region_z (default 50.0)  — Domain dimensions
        factors (default geomspace(0.2, 8, 18))      — P32 sweep factors
        seeds (default [10..300 by 10])              — Random seeds
        max_fracs (default 3000)                     — Max fractures per family
        viz_seed (default 123)                       — Seed for final DFN
    
    OUTPUTS:
        Creates outputs/<site_name>/ with 7 subdirectories:
        01_sweep/, 02_calibration/, 03_dfn_vtk/, 04_blocks_vtk/,
        05_block_volumes/, 06_blockometry_plots/, 07_persistence/
    """
    site_name     = cfg["site_name"]
        # ------------------------------------------------------------------------
        # b) Chargement des données d’orientation (lignes 164–172)
        #    On charge les orientations (pendage/direction) depuis le fichier Excel du site, pour chaque famille.
        #    On vérifie que chaque famille a bien des données.
        # ------------------------------------------------------------------------
    a_terrain     = float(cfg["a_terrain"])
    excel_name    = cfg["excel_name"]
    family_ids    = list(cfg["family_ids"])
    families      = cfg["families"]
    dip_face      = float(cfg["dip_face"])
    dipdir_face   = float(cfg["dipdir_face"])
    p32_guess     = np.array(cfg["p32_guess"], dtype=float)
    trace_name    = cfg["trace_name"]
    region_filter = cfg.get("region_filter", None)

    region_x = float(cfg.get("region_x", 50.0))
    region_y = float(cfg.get("region_y", 50.0))
    region_z = float(cfg.get("region_z", 50.0))

    factors   = cfg.get("factors", list(np.geomspace(0.2, 8, 18)))
    seeds     = cfg.get("seeds", list(range(10, 310, 10)))
    MAX_FRACS = int(cfg.get("max_fracs", 3000))
    viz_seed  = int(cfg.get("viz_seed", 123))

    n_fam = len(families)

    # ---- Output directories ----
    OUT_ROOT        = os.path.join(SCRIPT_DIR, "outputs", site_name)
    DIR_SWEEP       = os.path.join(OUT_ROOT, "01_sweep")
    DIR_CALIB       = os.path.join(OUT_ROOT, "02_calibration")
    DIR_CALIB_PLOTS = os.path.join(DIR_CALIB, "plots")
    DIR_DFN_VTK     = os.path.join(OUT_ROOT, "03_dfn_vtk")
    DIR_BLOCKS_VTK  = os.path.join(OUT_ROOT, "04_blocks_vtk")
    DIR_VOLUMES     = os.path.join(OUT_ROOT, "05_block_volumes")
    DIR_BLOCK_PLOTS = os.path.join(OUT_ROOT, "06_blockometry_plots")

    DIR_PERSISTENCE = os.path.join(OUT_ROOT, "07_persistence")
    for _d in [DIR_SWEEP, DIR_CALIB, DIR_CALIB_PLOTS,
               DIR_DFN_VTK, DIR_BLOCKS_VTK, DIR_VOLUMES, DIR_BLOCK_PLOTS, DIR_PERSISTENCE]:
        os.makedirs(_d, exist_ok=True)

    print(f"\n📁 Output root: {OUT_ROOT}")

    EXPORT_PREFIX     = f"VIZ_calibrated_{site_name}"
    RESULTS_CSV       = os.path.join(DIR_SWEEP,      f"P32_to_P21_sweep_results_{site_name}.csv")
    DFN_VTK_CLEAN     = os.path.join(DIR_DFN_VTK,   f"{EXPORT_PREFIX}_DFN_clean")
    DFN_VTK_PRISM     = os.path.join(DIR_DFN_VTK,   f"{EXPORT_PREFIX}_DFN_with_prism")
    BLOCKS_VTK_CLEAN  = os.path.join(DIR_BLOCKS_VTK, f"{EXPORT_PREFIX}_Blocks_clean")
    BLOCKS_VTK_PRISM  = os.path.join(DIR_BLOCKS_VTK, f"{EXPORT_PREFIX}_Blocks_with_prism")
    BLOCKS_VTK_VOID   = os.path.join(DIR_BLOCKS_VTK, f"{EXPORT_PREFIX}_Blocks_void.vtk")
    VOLUMES_TXT_CLEAN = os.path.join(DIR_VOLUMES,    f"{EXPORT_PREFIX}_BlockVolumes_clean.txt")
    VOLUMES_TXT_PRISM = os.path.join(DIR_VOLUMES,    f"{EXPORT_PREFIX}_BlockVolumes_with_prism.txt")

    # ---- Load orientations from Excel ----
    excel_path = os.path.join(SCRIPT_DIR, "assets", excel_name)
        # ------------------------------------------------------------------------
        # c) Construction du plan de cartographie (lignes 174–181)
        #    On calcule la normale au plan de cartographie à partir du pendage/direction.
        #    On construit un quadrilatère centré dans le domaine, qui servira de surface de référence pour le calcul de P21.
        #    On affiche les aires du modèle et du terrain.
        # ------------------------------------------------------------------------
    orient_by_fam = load_orientations_from_excel(excel_path, sheet=0, valid_families=family_ids)

    print(f"\n=== Loaded terrain orientations from Excel ===")
    print(f"Excel: {excel_path}")
    print(f"Using families: {family_ids}")
    for fid in family_ids:
        if fid not in orient_by_fam or len(orient_by_fam[fid]) == 0:
            raise ValueError(f"Missing orientations in Excel for fam{fid}.")
        print(f"fam{fid}: n = {len(orient_by_fam[fid])}")

    # ---- Build mapping quad ----
    n_face = normal_from_dip_dipdir(dip_face, dipdir_face)
        # ------------------------------------------------------------------------
        # d) Chargement des traces terrain et calcul de P21 cible (lignes 183–200)
        #    On lit le CSV des traces terrain, on filtre éventuellement par région.
        #    On calcule la somme des longueurs de traces par famille.
        #    On calcule la densité linéique P21 cible pour chaque famille (somme des longueurs divisée par la surface terrain).
        # ------------------------------------------------------------------------
    center = [region_x / 2.0, region_y / 2.0, region_z / 2.0]
    half_u = 0.49 * min(region_x, region_y)
    half_v = 0.49 * min(region_y, region_z)

    Q1, Q2, Q3, Q4 = build_center_quad(center, n_face, half_u, half_v)
    A_section_model = quad_area(Q1, Q2, Q3, Q4)

    print(f"\n=== Areas ===")
    print(f"A_section_model (Q1..Q4 in model) = {A_section_model:.3f} m²")
    print(f"A_TERRAIN (CloudCompare)          = {a_terrain:.3f} m²")

    # ---- Load trace CSV → P21_target ----
    trace_path = os.path.join(SCRIPT_DIR, "assets", trace_name)
        # ------------------------------------------------------------------------
        # e) Chargement des paramètres de taille de fracture (lignes 202–225)
        #    On tente de charger les paramètres corrigés (Mauldon) pour la taille des fractures.
        #    Si le fichier n’existe pas, on garde les valeurs par défaut.
        #    On affiche les paramètres utilisés pour chaque famille.
        # ------------------------------------------------------------------------
    df = pd.read_csv(trace_path)
    if "fam" not in df.columns or "Length" not in df.columns:
        raise ValueError("CSV must contain columns: 'fam' and 'Length'")

    if region_filter is not None and "REGION" in df.columns:
        df = df[df["REGION"].astype(str).str.upper() == region_filter.upper()]

    df["Length"] = pd.to_numeric(df["Length"], errors="coerce")
    df = df.dropna(subset=["Length", "fam"]).copy()

    sumL = df.groupby(df["fam"].astype(str))["Length"].sum()
    P21_target = np.array(
        [float(sumL.get(fam["name"], 0.0)) / a_terrain for fam in families],
        dtype=float
    )

    print(f"\n=== P21 TARGET (TERRAIN) ===")
    for i, fam in enumerate(families):
        print(f"{fam['name']}: P21_target = {P21_target[i]:.6f} 1/m")

    # ---- Load Mauldon-corrected fracture size params ----
    try:
            # ------------------------------------------------------------------------
            # f) Fonction interne run_one() : génération d’une réalisation DFN (lignes 227–293)
            #    Cette fonction génère un DFN pour un jeu de paramètres donné (P32 cible, seed).
            #    Pour chaque famille, on ajoute des fractures jusqu’à atteindre l’intensité volumique cible (P32).
            #    Chaque fracture : orientation tirée des mesures terrain (avec jitter), diamètre tiré d’une loi log-normale, position aléatoire dans le domaine.
            #    On retourne les valeurs obtenues de P32 et P21 (et le DFN si besoin).
            # ------------------------------------------------------------------------
        fam_names_local = [f["name"] for f in families]
        _size_params = load_size_params(
            trace_csv=trace_path,
            site_name=site_name,
            a_terrain=a_terrain,
            region_filter=region_filter,
            family_names=fam_names_local,
        )
        families = [dict(f) for f in families]  # shallow copy — don't mutate cfg
        for f in families:
            sp = _size_params.get(f["name"])
            if sp:
                f["mean"] = sp["mean"]
                f["sd"]   = sp["sd"]
        print(f"\n=== Fracture size params (Mauldon-corrected) ===")
        for f in families:
            print(f"  {f['name']}: mean={f['mean']:.4f} m  sd={f['sd']:.4f} m")
    except FileNotFoundError as e:
        print(f"\n⚠️  Mauldon correction not applied — {e}")
        if not all("mean" in f and "sd" in f for f in families):
            raise RuntimeError(
                f"No Mauldon table and no fallback mean/sd for site '{site_name}'.\n"
                f"Create: assets/mauldon_corrections.csv  (columns: site, fam, mu_nb)"
            )
        print("  → Using hardcoded mean/sd from SITE_CONFIGS.")
    except KeyError as e:
        raise KeyError(f"Missing Mauldon entry for site '{site_name}': {e}") from e

    # ---- run_one: generate one DFN realisation ----
    def run_one(P32_targets, seed=100, return_dfn=False):
            # ------------------------------------------------------------------------
            # g) Balayage et calibration P32 → P21 (lignes 295–374)
            #    On balaye plusieurs valeurs de P32 (facteurs) et plusieurs seeds.
            #    Pour chaque combinaison, on génère des DFN, on calcule les moyennes de P32 et P21 obtenues.
            #    Pour chaque famille, on fait une régression linéaire P21 = a·P32 + b pour trouver la valeur optimale de P32 qui permet d’atteindre la cible P21.
            #    Si la calibration sort de l’intervalle balayé, on recalcule automatiquement (jusqu’à 5 itérations max).
            # ------------------------------------------------------------------------
        """
        Generate one DFN realisation for a single site configuration.

        For each fracture family, fractures are added one by one to the DFN
        until the volumetric fracture intensity P32 (m²/m³) reaches the
        requested target.  Each fracture is placed by:
            1. Sampling a dip / dip-direction pair from the real orientation
               measurements and applying Fisher jitter (``jitter_orientation``).
            2. Drawing a diameter from a lognormal distribution parameterised
               by the Mauldon-corrected mean and standard deviation.
            3. Placing the fracture centre uniformly at random inside the
               3-D domain [0, region_x] × [0, region_y] × [0, region_z].

        After all families are built, P32 and P21 are read back from the DFN
        mapping objects and returned.

        Args:
            P32_targets (array-like): Target P32 (m²/m³) for each family,
                in the same order as *families*.
            seed (int): Random seed for both numpy (fracture sampling) and the
                DFN library.  Default 100.
            return_dfn (bool): If True, the function also returns the populated
                ``DFN`` object (needed for VTK export and block generation).

        Returns:
            tuple: ``(P32_obt, P21_obt)`` — numpy arrays of shape ``(n_fam,)``
                with achieved P32 and P21 values.
            If *return_dfn* is True, returns ``(P32_obt, P21_obt, dfn)``.
        """
        dfn = DFN()
        dfn.set_RegionMaxCorner([region_x, region_y, region_z])
        dfn.set_RandomSeed(int(seed))
        np.random.seed(int(seed))

        for _ in range(n_fam):
            dfn.add_FractureSet()

        dfn.add_QuadrilateralMapping(Q1, Q2, Q3, Q4)
        dfn.add_VolumeMapping()

        added_counts = [0] * n_fam

        for i in range(n_fam):
            target = float(P32_targets[i])
            if target <= 0:
                continue

            fam_id = family_ids[i]
            f = families[i]
            _mu_log  = np.log(float(f["mean"])**2 / np.sqrt(float(f["sd"])**2 + float(f["mean"])**2))
            _sig_log = np.sqrt(np.log(1.0 + (float(f["sd"]) / float(f["mean"]))**2))

            while dfn.volumesMapping[0].get_P32(i) < target:
                if added_counts[i] >= MAX_FRACS:
                    print(f"⚠️ MAX_FRACS={MAX_FRACS} reached for fam{fam_id}.")
                    break

                row        = orient_by_fam[fam_id].sample(1).iloc[0]
                dip_val, dipdir_val = jitter_orientation(
                    float(row["Dip"]),
                    float(row["Dip Direction"]),
                    fisher_k=f.get("fisher"),
                )
                diam = np.random.lognormal(_mu_log, _sig_log)
                cx   = np.random.uniform(0.0, region_x)
                cy   = np.random.uniform(0.0, region_y)
                cz   = np.random.uniform(0.0, region_z)
                dfn.fractureSets[i].add_CircularFracture(
                    [cx, cy, cz], dipdir_val, dip_val, diam / 2.0
                )
                added_counts[i] += 1

        P32_obt = np.array([dfn.volumesMapping[0].get_P32(i) for i in range(n_fam)], dtype=float)
        P21_obt = np.array([dfn.surfacesMapping[0].get_P21(i) for i in range(n_fam)], dtype=float)

        if return_dfn:
            return P32_obt, P21_obt, dfn
        return P32_obt, P21_obt

    # ---- Iterative Sweep + Calibration (auto-recentre if extrapolated) ----
    MAX_SWEEP_ITER = 5
        # ------------------------------------------------------------------------
        # h) Export des résultats de balayage et calibration (lignes 376–414)
        #    On sauvegarde les résultats du balayage (CSV).
        #    On génère et sauvegarde les figures de calibration (P32 vs P21 pour chaque famille, avec la droite de régression et la cible).
        # ------------------------------------------------------------------------
    for _sweep_iter in range(MAX_SWEEP_ITER):
        rows = []
        print(f"\n=== Sweep P32 factors — iteration {_sweep_iter+1}/{MAX_SWEEP_ITER} "
              f"({len(factors)} steps × {len(seeds)} seeds) ===")
        for k, fac in enumerate(factors):
            P32_targets = fac * p32_guess

            P32_list, P21_list = [], []
            for s in seeds:
                P32_obt, P21_obt = run_one(P32_targets, seed=s, return_dfn=False)
                P32_list.append(P32_obt)
                P21_list.append(P21_obt)

            P32_mean = np.mean(P32_list, axis=0)
            P21_mean = np.mean(P21_list, axis=0)

            print(f"  [{k+1}/{len(factors)}] factor={fac:.4f}  P32={P32_mean}  P21={P21_mean}")

            for i in range(n_fam):
                rows.append({
                    "factor":     float(fac),
                    "fam":        family_ids[i],
                    "fam_name":   families[i]["name"],
                    "P32_target": float(P32_targets[i]),
                    "P32_obt":    float(P32_mean[i]),
                    "P21_obt":    float(P21_mean[i]),
                    "P21_target": float(P21_target[i]),
                    "seed_count": len(seeds),
                })

        res = pd.DataFrame(rows)

        # ---- Calibration ----
        summary_rows = []
        print(f"\n=== Calibrated P32 estimate (per family) [REGRESSION] ===")
        for i, fam in enumerate(families):
            fam_id = family_ids[i]
            sub = res[res["fam"] == fam_id].copy().sort_values("P32_obt")

            x = sub["P32_obt"].to_numpy(float)
            y = sub["P21_obt"].to_numpy(float)

            a, b, r2 = linear_fit_and_r2(x, y)
            p32_star, flag = calibrated_p32_from_fit(
                P21_target[i], a, b,
                x_min=float(np.min(x)), x_max=float(np.max(x))
            )

            print(f"  {fam['name']}: P32_cal={p32_star:.6f}, flag={flag}, R²={r2:.4f}")

            summary_rows.append({
                "fam":               fam_id,
                "fam_name":          fam["name"],
                "P21_target":        float(P21_target[i]),
                "fit_a_P21_vs_P32":  a,
                "fit_b_P21_vs_P32":  b,
                "fit_R2":            r2,
                "P32_calibrated":    p32_star,
                "calibration_flag":  flag,
            })

        summary = pd.DataFrame(summary_rows)

        # ---- Auto-update p32_guess if any family extrapolated ----
        p32_guess, _changed = update_p32_guess_auto(
            res, summary, p32_guess, site_name=site_name
        )
        if not _changed:
            break  # all families within sweep range — done
    else:
        print(f"⚠️  Max sweep iterations ({MAX_SWEEP_ITER}) reached — check coverage manually.")

    res.to_csv(RESULTS_CSV, index=False)
        # ------------------------------------------------------------------------
        # i) Génération du DFN final calibré et exports (lignes 416–474)
        #    On génère le DFN final avec les P32 calibrés.
        #    On exporte les fichiers VTK pour le DFN et les blocs (clean).
        #    On exporte les volumes de blocs, figures de blockométrie (distribution des volumes, diagrammes de forme, etc.).
        # ------------------------------------------------------------------------
    print(f"\nSaved sweep results: {RESULTS_CSV}")

    # ---- Calibration plots ----
    for fam_id in sorted(res["fam"].unique()):
        sub = res[res["fam"] == fam_id].copy().sort_values("P32_obt")
        fam_name = sub["fam_name"].iloc[0]
        p21_t    = float(sub["P21_target"].iloc[0])

        row     = summary.loc[summary["fam"] == fam_id].iloc[0]
        a_fit   = row["fit_a_P21_vs_P32"]
        b_fit   = row["fit_b_P21_vs_P32"]
        r2      = row["fit_R2"]
        p32_cal = row["P32_calibrated"]
        flag    = row["calibration_flag"]

        x = sub["P32_obt"].to_numpy(float)
        y = sub["P21_obt"].to_numpy(float)

        plt.figure()
        plt.plot(x, y, marker="o", linestyle="None", label="DFN sweep (means)")

        if pd.notna(a_fit) and pd.notna(b_fit):
            xfit = np.linspace(float(np.min(x)), float(np.max(x)), 200)
            yfit = a_fit * xfit + b_fit
            plt.plot(xfit, yfit, linestyle="-",
                     label=f"fit: P21 = {a_fit:.3g}·P32 + {b_fit:.3g}  (R²={r2:.4f})")

        plt.axhline(p21_t, linestyle="--", label="P21 target")

        if pd.notna(p32_cal):
            lbl = "P32 calibrated" if flag == "ok" else "P32 calibrated (extrap.)"
            plt.scatter([p32_cal], [p21_t], marker="x", s=80, label=lbl)

        title = (f"{fam_name} | P32_cal ≈ {p32_cal:.6f} 1/m"
                 if pd.notna(p32_cal) else f"{fam_name} | calibration failed")
        plt.title(title)
        plt.xlabel("P32_obt (1/m)")
        plt.ylabel("P21_obt (1/m)")
        plt.grid(True)
        plt.legend()

        plt.savefig(os.path.join(DIR_CALIB_PLOTS, f"P32_vs_P21_{fam_name}.png"),
                    dpi=300, bbox_inches="tight")
        plt.savefig(os.path.join(DIR_CALIB_PLOTS, f"P32_vs_P21_{fam_name}.pdf"),
                    bbox_inches="tight")
        plt.close()

    summary_path = os.path.join(DIR_CALIB, "P32_calibrated_summary.csv")
    summary.to_csv(summary_path, index=False)
    print(f"\nCalibration plots saved in: {DIR_CALIB_PLOTS}")
    print(f"Summary saved: {summary_path}")
    print("\n=== Summary table ===")
    print(summary.to_string(index=False))

    # ---- Final DFN + Block exports ----
    P32_cal = summary.sort_values("fam")["P32_calibrated"].to_numpy(dtype=float)
    # ------------------------------------------------------------------------
    # j) Validation de la persistance (lignes 476–495)
    #    On exporte les CSVs et figures de persistance (comparaison des longueurs de traces DFN vs terrain).
    # ------------------------------------------------------------------------

    if np.any(~np.isfinite(P32_cal)):
        print("\n⚠️ Some families failed calibration. Final DFN export skipped.")
        return

    print(f"\n=== Generating FINAL calibrated DFN ===")
    print("P32_calibrated per family:", P32_cal)

    P32_obt_viz, P21_obt_viz, dfn_final = run_one(P32_cal, seed=viz_seed, return_dfn=True)

    # Append P21_model to summary CSV
    fam_ids_sorted = sorted(summary["fam"].unique())
    for j, fid in enumerate(fam_ids_sorted):
        summary.loc[summary["fam"] == fid, "P21_model"] = float(P21_obt_viz[j])
    summary.to_csv(summary_path, index=False)

    dfn_final.export_DFNVtk(DFN_VTK_CLEAN)
    print(f"✅ Exported CLEAN DFN: {DFN_VTK_CLEAN}.vtk")

    gen_clean = Generator()
    gen_clean.generate_RockMass(dfn_final)
    gen_clean.export_BlocksVtk(BLOCKS_VTK_CLEAN)
    print(f"✅ Exported Blocks CLEAN: {BLOCKS_VTK_CLEAN}.vtk")

    export_block_volumes_simple(gen_clean, VOLUMES_TXT_CLEAN)

    try:
            # ------------------------------------------------------------------------
            # k) Ajout et gestion du prisme/excavation (lignes 497–540)
            #    On définit un prisme triangulaire dans le modèle (pour simuler une excavation).
            #    On ajoute ce prisme au DFN, on génère les blocs coupés par le prisme, et ceux à l’intérieur (void).
            #    On exporte les fichiers VTK correspondants (DFN avec prisme, blocs avec prisme, blocs void).
            # ------------------------------------------------------------------------
        vols  = gen_clean.get_Volumes(True)
        alpha = gen_clean.get_AlphaValues(True)
        beta  = gen_clean.get_BetaValues(True)
        if len(vols) > 0:
            plotTools.blockVolumeDistribution(vols)
            plotTools.blockShapeDiagram(alpha, beta, vols, 0.05)
            plotTools.BlockShapeDistribution(alpha, beta, vols)
            save_all_open_figures(DIR_BLOCK_PLOTS, prefix=EXPORT_PREFIX)
            plt.close("all")
            import pandas as _pd
            _shape_csv = os.path.join(DIR_BLOCK_PLOTS, f"block_shape_data_{site_name}.csv")
            _pd.DataFrame({"volume": vols, "alpha": alpha, "beta": beta}).to_csv(_shape_csv, index=False)
            print(f"✅ Block shape data saved: {_shape_csv}")
    except Exception as e:
        print(f"⚠️ Blockometry plotting failed: {repr(e)}")

    # ---- Persistence validation ----
    try:
        _dir_persistence = os.path.join(OUT_ROOT, "07_persistence")
        _fam_names = [f["name"] for f in families]
        _assets_dir = os.path.join(SCRIPT_DIR, "assets")
        export_persistence(
            dfn_final, family_ids, _fam_names,
            _dir_persistence,
            _assets_dir, trace_name, region_filter,
            site_label=site_name,
        )
        print(f"✅ Persistence exports saved in: {_dir_persistence}")
    except Exception as e:
        import traceback
        print(f"⚠️ Persistence export failed: {repr(e)}")
        traceback.print_exc()

    EPS         = 1e-6
        # ------------------------------------------------------------------------
        # l) Résumé final des sorties (lignes 542–555)
        #    On affiche un arbre récapitulatif de tous les fichiers générés et de leur emplacement.
        # ------------------------------------------------------------------------
    z_top_prism = region_z - EPS
    z_bot_prism = region_z * 0.85
    x_wall      = region_x - EPS
    cy          = region_y / 2.0
    half_h      = 4.0
    depth_in    = 4.0

    A_pt = [x_wall,             cy - half_h, z_top_prism]
    B_pt = [x_wall,             cy + half_h, z_top_prism]
    C_pt = [x_wall - depth_in,  cy,          z_top_prism]

    print(f"\n=== Prism definition ===")
    print(f"z_bot = {z_bot_prism:.4f},  z_top = {z_top_prism:.4f}")
    print(f"A = {A_pt},  B = {B_pt},  C = {C_pt}")

    add_triangular_prism_Z(dfn_final, z_top=z_top_prism, z_bot=z_bot_prism,
                           A=A_pt, B=B_pt, C=C_pt)

    dfn_final.export_DFNVtk(DFN_VTK_PRISM)
    print(f"✅ Exported DFN WITH prism: {DFN_VTK_PRISM}.vtk")

    gen_prism = Generator()
    gen_prism.generate_RockMass(dfn_final)
    gen_prism.export_BlocksVtk(BLOCKS_VTK_PRISM)
    print(f"✅ Exported Blocks WITH prism: {BLOCKS_VTK_PRISM}.vtk")

    export_block_volumes_simple(gen_prism, VOLUMES_TXT_PRISM)

    A_xy = (A_pt[0], A_pt[1])
    B_xy = (B_pt[0], B_pt[1])
    C_xy = (C_pt[0], C_pt[1])

    delete_fragments_inside_prism_vtk(
        BLOCKS_VTK_PRISM + ".vtk",
        BLOCKS_VTK_VOID,
        A_xy=A_xy, B_xy=B_xy, C_xy=C_xy,
        z_bot=z_bot_prism, z_top=z_top_prism,
        inside_ratio_thresh=0.95,
    )

    # ---- Summary tree ----
    print("\n" + "=" * 55)
    print(f"📁 OUTPUT: {OUT_ROOT}/")
    print("=" * 55)
    print(f"  ├── 01_sweep/             P32_to_P21_sweep_results_{site_name}.csv")
    print(f"  ├── 02_calibration/       P32_calibrated_summary.csv")
    print(f"  │   └── plots/            P32_vs_P21_fam*.png/.pdf")
    print(f"  ├── 03_dfn_vtk/           *_DFN_clean.vtk  +  *_DFN_with_prism.vtk")
    print(f"  ├── 04_blocks_vtk/        clean / prism / void")
    print(f"  ├── 05_block_volumes/     *.txt")
    print(f"  ├── 06_blockometry_plots/ *.png/.pdf")
    print(f"  └── 07_persistence/       persistence_*.csv + *.png/.pdf")
    print("=" * 55)
    print("\n✅ Done.")


# ============================================================
# Allow running this file directly for a single site
# ============================================================
if __name__ == "__main__":
    # ============================================================
    # 3. Exécution directe du script (lignes 558–563)
    #   Si on lance run_site.py directement en ligne de commande, on peut passer le nom du site en argument (python run_site.py BC1LEFT).
    #   Cela appelle run_site(SITE_CONFIGS[site]) pour lancer toute la chaîne sur le site choisi.
    # ============================================================
    import argparse

    parser = argparse.ArgumentParser(description="Run DFN analysis for a single site.")
    parser.add_argument("site", choices=list(SITE_CONFIGS.keys()),
                        help="Site to run")
    args = parser.parse_args()

    run_site(SITE_CONFIGS[args.site])
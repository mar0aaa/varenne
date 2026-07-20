# ============================================================
# VARENNE — DFN + blockometry analysis  (single site)
# Run directly:  python varenne.py
# Or via menu:   python main.py  →  [1]
#
# This script generates:
#   • DFN sweep across P32 factors
#   • P32 calibration against field P21
#   • Block volumetry and shape analysis
#   • Persistence comparisons
#   • Prism (excavation) analysis
# ============================================================
"""
varenne.py — Entry point for the Varenne site DFN analysis.

This module resolves the project root directory, inserts it into sys.path
so all sibling modules are importable, then delegates the full analysis
pipeline to ``run_site()`` from ``run_site.py`` using the pre-configured
``SITE_CONFIGS["VARENNE"]`` dictionary.

The VARENNE site configuration includes:
    - Fracture families: fam1, fam2, fam3, fam4
    - Terrain mapping area: set in SITE_CONFIGS (update from CloudCompare)
    - Mapping face orientation: update dip_face / dipdir_face in run_site.py
    - Orientation file:  assets/DIPSVARENNE.xlsx
    - Trace file:        assets/varenne_traces.csv

Running this script triggers the following pipeline stages inside run_site():
    1. Load dip/dip-direction orientations from DIPSVARENNE.xlsx
    2. Load Mauldon-corrected fracture size parameters
    3. Compute P21 targets from the field trace CSV
    4. Sweep P32 factors and fit P32 → P21 regression
    5. Auto-recentre sweep if any family falls outside the covered range
    6. Generate the final calibrated DFN (VTK)
    7. Generate block meshes (clean + prism + void VTKs)
    8. Export block volumes and blockometry figures
    9. Export persistence comparison plots and CSVs

Outputs are written to:  outputs/VARENNE/
"""

import sys
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()
sys.path.insert(0, SCRIPT_DIR)

from run_site import SITE_CONFIGS, run_site

if __name__ == "__main__":
    run_site(SITE_CONFIGS["VARENNE"])

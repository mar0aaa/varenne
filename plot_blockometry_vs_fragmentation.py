"""
Overlay in-situ blockometry and WipFrag fragmentation curves on one figure.

The figure shows % passing vs particle size using either:
- equivalent diameter (mm), or
- equivalent spherical volume (m^3).

Inputs (defaults):
- In-situ block volumes:
    outputs/BCTOTAL/05_block_volumes/VIZ_calibrated_BCTOTAL_BlockVolumes_clean.txt
- WipFrag results:
    assets/Fragmentation wipfrag results.xlsx

Output (defaults):
- outputs/BCTOTAL/08_fragmentation_comparison/blockometry_vs_fragmentation.png
- outputs/BCTOTAL/08_fragmentation_comparison/blockometry_vs_fragmentation.pdf
"""

import argparse
import os
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

DEFAULT_BLOCK_VOLUMES = os.path.join(
    SCRIPT_DIR,
    "outputs",
    "BCTOTAL",
    "05_block_volumes",
    "VIZ_calibrated_BCTOTAL_BlockVolumes_clean.txt",
)
DEFAULT_WIPFRAG_XLSX = os.path.join(SCRIPT_DIR, "assets", "Fragmentation wipfrag results.xlsx")
DEFAULT_OUT_DIR = os.path.join(SCRIPT_DIR, "outputs", "BCTOTAL", "08_fragmentation_comparison")


def _safe_numeric(series: pd.Series) -> pd.Series:
    """
    Coerce a pandas Series to numeric dtype, replacing non-parseable values with NaN.

    A thin convenience wrapper around ``pd.to_numeric(errors="coerce")``.

    Args:
        series (pd.Series): Input column of any dtype.

    Returns:
        pd.Series: Numeric series with non-convertible entries replaced by NaN.
    """
    return pd.to_numeric(series, errors="coerce")


def _extract_zone_name(sheet: pd.DataFrame, start_row: int, col: int, fallback: str) -> str:
    """
    Search a few rows above a WipFrag header row to extract the zone label.

    WipFrag Excel exports typically place a descriptive zone name (e.g.
    ``"Zone A"`` or a chainage range) one to four rows above the column
    headers.  This function scans upward from *start_row* to find the first
    non-empty cell that does not look like a WipFrag internal header
    (``"chainage"`` or ``"fraction"``).

    Args:
        sheet (pd.DataFrame): Full raw sheet read with ``header=None``.
        start_row (int): 0-indexed row of the ``"Fraction passante"`` header.
        col (int): Column index to search in.
        fallback (str): String returned when no suitable label is found.

    Returns:
        str: The extracted zone name, or *fallback* if nothing suitable was found.
    """
    for r in range(max(0, start_row - 4), start_row):
        val = sheet.iat[r, col]
        if isinstance(val, str) and val.strip():
            txt = val.strip()
            if "chainage" not in txt.lower() and "fraction" not in txt.lower():
                return txt
    return fallback


def read_wipfrag_curves(xlsx_path: str) -> List[Tuple[str, np.ndarray, np.ndarray]]:
    """
    Parse WipFrag sheet and return a list of curves: (name, size_mm, passing_percent).

    The parser is robust to merged/blank rows and uses text markers
    "Fraction passante" + "Grosseur" to locate each curve block.
    """
    raw = pd.read_excel(xlsx_path, sheet_name=0, header=None)
    curves = []

    n_rows, n_cols = raw.shape
    for r in range(n_rows):
        for c in range(n_cols - 1):
            left = raw.iat[r, c]
            right = raw.iat[r, c + 1]

            left_s = left.strip().lower() if isinstance(left, str) else ""
            right_s = right.strip().lower() if isinstance(right, str) else ""

            if "fraction" in left_s and "pass" in left_s and "grosseur" in right_s:
                zone_name = _extract_zone_name(raw, r, c, fallback=f"Zone_{len(curves)+1}")

                block = raw.iloc[r + 1 :, [c, c + 1]].copy()
                block.columns = ["fraction", "size_mm"]
                block["fraction"] = _safe_numeric(block["fraction"])
                block["size_mm"] = _safe_numeric(block["size_mm"])
                block = block.dropna(subset=["fraction", "size_mm"])  # keep numeric rows only

                if block.empty:
                    continue

                y = block["fraction"].to_numpy(dtype=float)
                x = block["size_mm"].to_numpy(dtype=float)

                # WipFrag may store fraction in [0,1] or percent [0,100].
                if np.nanmax(y) <= 1.5:
                    y = 100.0 * y

                # Sort by size for clean plotting.
                order = np.argsort(x)
                x = x[order]
                y = y[order]

                curves.append((zone_name, x, y))

    if not curves:
        raise ValueError(
            "No WipFrag curve found. Expected columns like 'Fraction passante' and 'Grosseur ... (mm)'."
        )

    return curves


def read_blockometry_curve(volumes_txt: str) -> Tuple[np.ndarray, np.ndarray]:
    """Return sorted block volumes (m^3) and cumulative passing (%) weighted by volume."""
    vols = np.loadtxt(volumes_txt, ndmin=1)
    vols = np.asarray(vols, dtype=float).ravel()
    vols = vols[np.isfinite(vols)]
    vols = vols[vols > 0]
    if vols.size == 0:
        raise ValueError(f"No valid positive block volumes found in: {volumes_txt}")

    vols_sorted = np.sort(vols)
    passing = 100.0 * np.cumsum(vols_sorted) / np.sum(vols_sorted)
    return vols_sorted, passing


def volumes_to_equivalent_diameter_mm(volumes_m3: np.ndarray) -> np.ndarray:
    """Equivalent sphere diameter in mm from volume in m^3: d = (6V/pi)^(1/3)."""
    diam_m = np.cbrt((6.0 * volumes_m3) / np.pi)
    return 1000.0 * diam_m


def diameter_mm_to_equivalent_volume_m3(diameter_mm: np.ndarray) -> np.ndarray:
    """Equivalent sphere volume in m^3 from diameter in mm: V = pi/6 * d^3."""
    d_m = diameter_mm / 1000.0
    return (np.pi / 6.0) * np.power(d_m, 3)


def make_overlay_plot(
    block_volumes_txt: str,
    wipfrag_xlsx: str,
    out_dir: str,
    x_mode: str = "diameter",
) -> None:
    """
    Produce an overlay figure comparing DFN-predicted block sizes with WipFrag data.

    Reads two data sources:

    1. **In-situ blockometry** — block volumes (m³) exported from the BCTOTAL
       DFN pipeline, converted to a cumulative passing curve weighted by volume.
    2. **WipFrag post-blast fragmentation** — one or more granulometry curves
       parsed from the WipFrag Excel export, each representing a different
       blast zone or measurement campaign.

    Both datasets are plotted on a shared semi-logarithmic axis (log x-scale)
    showing percent passing versus particle size.  This allows a direct visual
    comparison of pre-blast natural block sizes and post-blast fragment sizes.

    Args:
        block_volumes_txt (str): Path to the plain-text block-volume file
            (one volume per line, in m³).
        wipfrag_xlsx (str): Path to the WipFrag Excel workbook.  The first
            sheet is parsed; each curve block is identified by the column
            header pair ``"Fraction passante"`` / ``"Grosseur"``.
        out_dir (str): Directory in which to save the output figures.  Created
            if it does not exist.
        x_mode (str): X-axis units.  Either:
            - ``"diameter"`` (default): equivalent sphere diameter in mm.
            - ``"volume"``: equivalent sphere volume in m³.

    Returns:
        None

    Side-effects:
        Saves ``blockometry_vs_fragmentation_<suffix>.png`` and
        ``blockometry_vs_fragmentation_<suffix>.pdf`` into *out_dir*.
        Prints one status line per saved file.

    Raises:
        ValueError: If *x_mode* is not ``"diameter"`` or ``"volume"``.
    """
    os.makedirs(out_dir, exist_ok=True)

    vols_sorted, passing_blocko = read_blockometry_curve(block_volumes_txt)
    wipfrag_curves = read_wipfrag_curves(wipfrag_xlsx)

    if x_mode == "diameter":
        x_blocko = volumes_to_equivalent_diameter_mm(vols_sorted)
        x_label = "Particle size (equivalent diameter, mm)"
        title = "DFN-predicted pre-blast vs WipFrag post-blast"
    elif x_mode == "volume":
        x_blocko = vols_sorted
        x_label = "Particle size (equivalent volume, m^3)"
        title = "DFN-predicted pre-blast vs WipFrag post-blast"
    else:
        raise ValueError("x_mode must be 'diameter' or 'volume'.")

    fig, ax = plt.subplots(figsize=(9.5, 6.0))

    ax.plot(
        x_blocko,
        passing_blocko,
        color="red",
        linewidth=2.2,
        label="DFN-predicted pre-blast (model volumes)",
    )

    for zone_name, size_mm, passing_pct in wipfrag_curves:
        if x_mode == "diameter":
            x = size_mm
        else:
            x = diameter_mm_to_equivalent_volume_m3(size_mm)

        ax.plot(
            x,
            passing_pct,
            marker="o",
            markersize=4,
            linewidth=1.8,
            label=f"WipFrag post-blast - {zone_name}",
        )

    ax.set_xscale("log")
    ax.set_ylim(0, 100)
    ax.set_xlabel(x_label)
    ax.set_ylabel("Percent passing (%)")
    ax.set_title(title)
    ax.grid(True, which="both", linestyle="--", alpha=0.4)
    ax.legend(loc="upper left")
    fig.tight_layout()

    suffix = "diameter_mm" if x_mode == "diameter" else "volume_m3"
    out_png = os.path.join(out_dir, f"blockometry_vs_fragmentation_{suffix}.png")
    out_pdf = os.path.join(out_dir, f"blockometry_vs_fragmentation_{suffix}.pdf")
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {out_png}")
    print(f"Saved: {out_pdf}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Overlay in-situ blockometry and WipFrag fragmentation curves."
    )
    parser.add_argument(
        "--block-volumes",
        default=DEFAULT_BLOCK_VOLUMES,
        help="Path to block volume TXT (m^3).",
    )
    parser.add_argument(
        "--wipfrag",
        default=DEFAULT_WIPFRAG_XLSX,
        help="Path to WipFrag Excel file.",
    )
    parser.add_argument(
        "--out-dir",
        default=DEFAULT_OUT_DIR,
        help="Output directory for figures.",
    )
    parser.add_argument(
        "--x-mode",
        choices=["diameter", "volume"],
        default="diameter",
        help="Plot x-axis as equivalent diameter (mm) or equivalent volume (m^3).",
    )

    args = parser.parse_args()
    make_overlay_plot(
        block_volumes_txt=args.block_volumes,
        wipfrag_xlsx=args.wipfrag,
        out_dir=args.out_dir,
        x_mode=args.x_mode,
    )

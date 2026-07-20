import os
import numpy as np
import matplotlib.pyplot as plt


def export_block_volumes_simple(gen, out_txt):
    """Export block volumes from a Generator to a plain text file."""
    raw = gen.get_Volumes(True)
    vols = np.array([float(v) for v in raw], dtype=np.float64)
    np.savetxt(out_txt, vols)
    print(f"✅ Block volumes saved: {os.path.abspath(out_txt)}")
    return vols


def save_all_open_figures(out_dir, prefix="VIZ"):
    """Save all currently open matplotlib figures as PNG and PDF."""
    fig_nums = plt.get_fignums()
    for i, num in enumerate(fig_nums, start=1):
        fig = plt.figure(num)
        png = os.path.join(out_dir, f"{prefix}_blocko_fig{i}.png")
        pdf = os.path.join(out_dir, f"{prefix}_blocko_fig{i}.pdf")
        fig.savefig(png, dpi=300, bbox_inches="tight")
        fig.savefig(pdf, bbox_inches="tight")
    print(f"✅ Saved {len(fig_nums)} blockometry figures in: {out_dir}")


def print_blockometry_summary(vols: np.ndarray, label: str = "") -> None:
    """Print percentile + Palmström summary for a block volume array."""
    vols = np.asarray(vols, dtype=float)
    vols = vols[np.isfinite(vols) & (vols > 0)]
    if vols.size == 0:
        print(f"  ⚠️  No valid volumes for {label}")
        return

    n = vols.size
    print(f"  === Blockometry Summary: {label} (n={n}) ===")

    p20 = np.percentile(vols, 20)
    p50 = np.percentile(vols, 50)
    p80 = np.percentile(vols, 80)
    p90 = np.percentile(vols, 90)
    print(f"    D20 (20th pct)  = {p20:.4f} m³   →  L = {p20**(1/3):.3f} m")
    print(f"    D50 (median)    = {p50:.4f} m³   →  L = {p50**(1/3):.3f} m")
    print(f"    D80 (80th pct)  = {p80:.4f} m³   →  L = {p80**(1/3):.3f} m")
    print(f"    D90 (90th pct)  = {p90:.4f} m³   →  L = {p90**(1/3):.3f} m")
    print(f"    Mean            = {vols.mean():.4f} m³   →  L = {vols.mean()**(1/3):.3f} m")

    print("    --- % blocks LARGER than threshold ---")
    for t in [0.01, 0.1, 1.0, 10.0]:
        pct = 100.0 * np.mean(vols > t)
        print(f"      > {t:>5.2f} m³ : {pct:.1f}%")

    def _palmstrom_class(v):
        if v < 1e-4:  return "Very small (crushed)"
        if v < 1e-3:  return "Small"
        if v < 0.01:  return "Moderately small"
        if v < 0.1:   return "Medium"
        if v < 1.0:   return "Large"
        return "Very large"

    print(f"    Median block class (Palmström): {_palmstrom_class(p50)}")
    print()
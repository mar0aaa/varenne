#!/usr/bin/env python3
"""
render_block_volume_3d.py
=========================
Regenerates the 3-D block-volume visualization for BCTOTAL.

Colorbar is set to log scale from 1e-4 m³ to 200 m³ (matching
Figure 15 in-situ block size distribution), replacing the previous
auto-scaled version that only showed the range 10–160 m³.

Output: outputs/BCTOTAL/04_blocks_vtk/blocksbctotal.png
"""

import os
import numpy as np
import pyvista as pv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colorbar import ColorbarBase

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

VTK_PATH = os.path.join(
    SCRIPT_DIR, "outputs", "BCTOTAL", "04_blocks_vtk",
    "VIZ_calibrated_BCTOTAL_Blocks_clean.vtk"
)
TMP_PNG = os.path.join(SCRIPT_DIR, "outputs", "BCTOTAL", "04_blocks_vtk", "_tmp_render.png")
OUT_PNG = os.path.join(SCRIPT_DIR, "outputs", "BCTOTAL", "04_blocks_vtk", "blocksbctotal.png")

# Colorbar limits — match Figure 15 (1E-4 to ~1E+2 m³)
VMIN = 1e-4   # m³  (dark blue = smallest meaningful blocks)
VMAX = 200.0  # m³  (red = largest blocks ~1.6 × 10²)

# ── Load VTK ──────────────────────────────────────────────────────────────────
pv.OFF_SCREEN = True
mesh = pv.read(VTK_PATH)

# Log-transform the volume scalar so the colormap spans the full log range
vol = mesh["volume"].copy()
vol_clamped = np.clip(vol, VMIN, VMAX)
log_vol = np.log10(vol_clamped)
mesh["log_volume"] = log_vol

log_vmin = np.log10(VMIN)   # -4
log_vmax = np.log10(VMAX)   #  ~2.3

# ── PyVista off-screen render (no scalar bar — we add it via matplotlib) ──────
pl = pv.Plotter(off_screen=True, window_size=(900, 700))
pl.set_background("dimgray")
pl.add_mesh(
    mesh,
    scalars="log_volume",
    cmap="coolwarm",
    clim=[log_vmin, log_vmax],
    show_scalar_bar=False,
    lighting=True,
    smooth_shading=False,
)
pl.camera_position = "iso"
pl.camera.azimuth  = -30
pl.camera.elevation = 15
pl.screenshot(TMP_PNG, return_img=False)
pl.close()

# ── Compose final image with matplotlib colorbar ──────────────────────────────
render_img = plt.imread(TMP_PNG)

fig = plt.figure(figsize=(7, 6.5), facecolor="#404040")

# 3-D render occupies the top 88% of the figure
ax_img = fig.add_axes([0.0, 0.12, 1.0, 0.88])
ax_img.imshow(render_img)
ax_img.axis("off")

# Colorbar below the render
ax_cb = fig.add_axes([0.18, 0.06, 0.64, 0.045])

norm = mcolors.LogNorm(vmin=VMIN, vmax=VMAX)
cmap = plt.get_cmap("coolwarm")
cb = ColorbarBase(ax_cb, cmap=cmap, norm=norm, orientation="horizontal")

# Ticks at powers of 10 that fall within range
tick_vals = [v for v in [1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2]
             if VMIN <= v <= VMAX]
cb.set_ticks(tick_vals)
cb.set_ticklabels([f"$10^{{{int(np.log10(t))}}}$" for t in tick_vals])
cb.ax.tick_params(colors="white", labelsize=9)
cb.set_label("Block volume (m³)", color="white", fontsize=11)
cb.outline.set_edgecolor("white")

# Annotate min/max
ax_cb.text(0.0, -1.8, f"{VMIN:.0e} m³", color="white", fontsize=9,
           ha="left", va="top", transform=ax_cb.transAxes)
ax_cb.text(1.0, -1.8, f"$1.6 \\times 10^2$ m³", color="white", fontsize=9,
           ha="right", va="top", transform=ax_cb.transAxes)

fig.savefig(OUT_PNG, dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
plt.close(fig)

# Clean up temp file
os.remove(TMP_PNG)

print(f"✅ Saved: {OUT_PNG}")
print(f"   Colorbar: {VMIN:.0e} m³ (dark blue)  →  {VMAX:.0f} m³ (red)")
print(f"   Log scale to match Figure 15 (1E-04 to 1E+02 m³)")

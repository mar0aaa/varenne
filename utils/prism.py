import numpy as np
from .vtk_io import read_legacy_vtk_unstructured_grid, write_legacy_vtk_unstructured_grid


def add_triangular_prism_Z(dfn, *, z_top, z_bot, A, B, C):
    """
    Add a triangular prism (vertical extrusion) to a DFN as cutting surfaces.
    A, B, C are the 3D top vertices (z = z_top).
    """
    dfn.add_FractureSet()
    fs = dfn.fractureSets[-1]

    A2 = [A[0], A[1], z_bot]
    B2 = [B[0], B[1], z_bot]
    C2 = [C[0], C[1], z_bot]

    # Top and bottom caps
    fs.add_TriangularFracture(A, B, C)
    fs.add_TriangularFracture(A2, C2, B2)

    # Side faces (2 triangles each)
    fs.add_TriangularFracture(A, B, B2)
    fs.add_TriangularFracture(A, B2, A2)

    fs.add_TriangularFracture(B, C, C2)
    fs.add_TriangularFracture(B, C2, B2)

    fs.add_TriangularFracture(C, A, A2)
    fs.add_TriangularFracture(C, A2, C2)

    return fs


def _point_in_triangle_2d(P, A, B, C, eps=1e-12):
    px, py = P
    ax, ay = A
    bx, by = B
    cx, cy = C

    def sign(x1, y1, x2, y2, x3, y3):
        return (x1 - x3) * (y2 - y3) - (x2 - x3) * (y1 - y3)

    d1 = sign(px, py, ax, ay, bx, by)
    d2 = sign(px, py, bx, by, cx, cy)
    d3 = sign(px, py, cx, cy, ax, ay)

    has_neg = (d1 < -eps) or (d2 < -eps) or (d3 < -eps)
    has_pos = (d1 > eps) or (d2 > eps) or (d3 > eps)
    return not (has_neg and has_pos)


def delete_fragments_inside_prism_vtk(in_vtk, out_vtk, *, A_xy, B_xy, C_xy,
                                      z_bot, z_top,
                                      inside_ratio_thresh=0.95, eps=1e-9):
    """
    Remove from a VTK block file all cells whose vertices are mostly
    inside the triangular prism defined by (A_xy, B_xy, C_xy) and [z_bot, z_top].
    Writes the filtered result to out_vtk.
    """
    points, cells, cell_types, cell_data, point_data = read_legacy_vtk_unstructured_grid(in_vtk)

    zmin = min(z_bot, z_top) - eps
    zmax = max(z_bot, z_top) + eps

    tri_x = [A_xy[0], B_xy[0], C_xy[0]]
    tri_y = [A_xy[1], B_xy[1], C_xy[1]]
    pxmin, pxmax = min(tri_x) - eps, max(tri_x) + eps
    pymin, pymax = min(tri_y) - eps, max(tri_y) + eps

    keep_idx = []
    removed_idx = []

    for ci, c in enumerate(cells):
        pts = points[np.array(c, dtype=int)]
        mn = pts.min(axis=0)
        mx = pts.max(axis=0)

        if (mx[0] < pxmin or mn[0] > pxmax or
                mx[1] < pymin or mn[1] > pymax or
                mx[2] < zmin or mn[2] > zmax):
            keep_idx.append(ci)
            continue

        inside = sum(
            1 for pid in c
            if zmin <= points[int(pid)][2] <= zmax
            and _point_in_triangle_2d(
                (points[int(pid)][0], points[int(pid)][1]),
                A_xy, B_xy, C_xy
            )
        )

        ratio = inside / max(len(c), 1)
        if ratio >= inside_ratio_thresh:
            removed_idx.append(ci)
        else:
            keep_idx.append(ci)

    new_cells = [cells[i] for i in keep_idx]
    new_types = [cell_types[i] for i in keep_idx]

    new_cell_data = {}
    for name, arr in (cell_data or {}).items():
        arr = np.asarray(arr)
        new_cell_data[name] = arr[np.array(keep_idx, dtype=int)] if len(arr) == len(cells) else arr

    removed_vol = None
    if cell_data and "volume" in cell_data and len(cell_data["volume"]) == len(cells):
        removed_vol = float(np.sum(cell_data["volume"][np.array(removed_idx, dtype=int)]))

    write_legacy_vtk_unstructured_grid(
        out_vtk, points, new_cells, new_types,
        cell_data=new_cell_data, point_data=point_data,
        title="Blocks with void"
    )

    print(f"✅ Void VTK written: {out_vtk}")
    print(f"   Removed cells: {len(removed_idx)} / {len(cells)}")
    if removed_vol is not None:
        print(f"   Approx removed volume: {removed_vol:.6f} m³")
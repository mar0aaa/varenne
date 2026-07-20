import numpy as np


def _read_tokens(path):
    with open(path, "r") as f:
        return f.read().replace("\r", "\n").split()


def read_legacy_vtk_unstructured_grid(path):
    """Parse a legacy ASCII VTK unstructured grid file."""
    toks = _read_tokens(path)
    i = 0
    points = None
    cells = []
    cell_types = []
    cell_data = {}
    point_data = {}

    def expect(word):
        nonlocal i
        if toks[i] != word:
            raise ValueError(f"Expected {word} but got {toks[i]} at token {i}")
        i += 1

    while i < len(toks) and toks[i] != "POINTS":
        i += 1
    if i >= len(toks):
        raise ValueError("No POINTS section found.")
    expect("POINTS")
    npts = int(toks[i]); i += 1
    _ptype = toks[i]; i += 1
    coords = list(map(float, toks[i:i + 3 * npts])); i += 3 * npts
    points = np.array(coords, dtype=float).reshape((npts, 3))

    while i < len(toks) and toks[i] != "CELLS":
        i += 1
    if i >= len(toks):
        raise ValueError("No CELLS section found.")
    expect("CELLS")
    ncells = int(toks[i]); i += 1
    _total_ints = int(toks[i]); i += 1

    for _ in range(ncells):
        k = int(toks[i]); i += 1
        ids = list(map(int, toks[i:i + k])); i += k
        cells.append(ids)

    while i < len(toks) and toks[i] != "CELL_TYPES":
        i += 1
    if i >= len(toks):
        raise ValueError("No CELL_TYPES section found.")
    expect("CELL_TYPES")
    nct = int(toks[i]); i += 1
    cell_types = list(map(int, toks[i:i + nct])); i += nct

    while i < len(toks):
        if toks[i] == "CELL_DATA":
            i += 1
            ncd = int(toks[i]); i += 1
            while i < len(toks) and toks[i] == "SCALARS":
                i += 1
                name = toks[i]; i += 1
                _dtype = toks[i]; i += 1
                # numComp is optional in the VTK spec
                if toks[i] not in ("LOOKUP_TABLE", "SCALARS", "CELL_DATA", "POINT_DATA"):
                    ncomp = int(toks[i]); i += 1
                else:
                    ncomp = 1
                if ncomp != 1:
                    raise ValueError("Only SCALARS with 1 component supported.")
                expect("LOOKUP_TABLE")
                _lt = toks[i]; i += 1
                arr = np.array(list(map(float, toks[i:i + ncd])), dtype=float)
                i += ncd
                cell_data[name] = arr
        elif toks[i] == "POINT_DATA":
            i += 1
            npd = int(toks[i]); i += 1
            while i < len(toks) and toks[i] == "SCALARS":
                i += 1
                name = toks[i]; i += 1
                _dtype = toks[i]; i += 1
                # numComp is optional in the VTK spec
                if toks[i] not in ("LOOKUP_TABLE", "SCALARS", "CELL_DATA", "POINT_DATA"):
                    ncomp = int(toks[i]); i += 1
                else:
                    ncomp = 1
                if ncomp != 1:
                    raise ValueError("Only SCALARS with 1 component supported.")
                expect("LOOKUP_TABLE")
                _lt = toks[i]; i += 1
                arr = np.array(list(map(float, toks[i:i + npd])), dtype=float)
                i += npd
                point_data[name] = arr
        else:
            i += 1

    return points, cells, cell_types, cell_data, point_data


def write_legacy_vtk_unstructured_grid(path, points, cells, cell_types, *,
                                       cell_data=None, point_data=None, title="Blocks"):
    """Write a legacy ASCII VTK unstructured grid file."""
    points = np.asarray(points, dtype=float)
    npts = points.shape[0]
    ncells = len(cells)
    total_ints = sum(1 + len(c) for c in cells)

    with open(path, "w") as f:
        f.write("# vtk DataFile Version 3.0\n")
        f.write(f"{title}\n")
        f.write("ASCII\n")
        f.write("DATASET UNSTRUCTURED_GRID\n")

        f.write(f"POINTS {npts} float\n")
        for p in points:
            f.write(f"{p[0]:.9g} {p[1]:.9g} {p[2]:.9g}\n")

        f.write(f"CELLS {ncells} {total_ints}\n")
        for c in cells:
            f.write(str(len(c)) + " " + " ".join(str(int(x)) for x in c) + "\n")

        f.write(f"CELL_TYPES {ncells}\n")
        for t in cell_types:
            f.write(str(int(t)) + "\n")

        if cell_data:
            f.write(f"CELL_DATA {ncells}\n")
            for name, arr in cell_data.items():
                arr = np.asarray(arr)
                f.write(f"SCALARS {name} float 1\n")
                f.write("LOOKUP_TABLE default\n")
                for v in arr:
                    f.write(f"{float(v):.9g}\n")

        if point_data:
            f.write(f"POINT_DATA {npts}\n")
            for name, arr in point_data.items():
                arr = np.asarray(arr)
                f.write(f"SCALARS {name} float 1\n")
                f.write("LOOKUP_TABLE default\n")
                for v in arr:
                    f.write(f"{float(v):.9g}\n")
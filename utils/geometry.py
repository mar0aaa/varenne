import numpy as np


def normal_from_dip_dipdir(dip_deg, dipdir_deg):
    """Compute unit normal vector from dip and dip direction (degrees)."""
    dip = np.deg2rad(dip_deg)
    dipdir = np.deg2rad(dipdir_deg)
    strike = dipdir - np.pi / 2.0

    s = np.array([np.sin(strike), np.cos(strike), 0.0])
    d = np.array([
        np.sin(dipdir) * np.cos(dip),
        np.cos(dipdir) * np.cos(dip),
        -np.sin(dip)
    ])
    n = np.cross(s, d)
    n = n / np.linalg.norm(n)
    return n


def build_center_quad(center, n, half_u, half_v):
    """Build a centered quadrilateral (4 corners) from a center, normal, and half-extents."""
    n = np.array(n, dtype=float)
    n = n / np.linalg.norm(n)

    ref = np.array([0.0, 0.0, 1.0])
    if abs(np.dot(n, ref)) > 0.95:
        ref = np.array([1.0, 0.0, 0.0])

    u = np.cross(n, ref)
    u = u / np.linalg.norm(u)
    v = np.cross(n, u)
    v = v / np.linalg.norm(v)

    c = np.array(center, dtype=float)
    p1 = c - half_u * u - half_v * v
    p2 = c + half_u * u - half_v * v
    p3 = c + half_u * u + half_v * v
    p4 = c - half_u * u + half_v * v
    return p1.tolist(), p2.tolist(), p3.tolist(), p4.tolist()


def quad_area(Q1, Q2, Q3, Q4):
    """Compute area of a quadrilateral defined by 4 corner points."""
    Q1 = np.array(Q1, dtype=float)
    Q2 = np.array(Q2, dtype=float)
    Q3 = np.array(Q3, dtype=float)
    Q4 = np.array(Q4, dtype=float)
    a1 = 0.5 * np.linalg.norm(np.cross(Q2 - Q1, Q3 - Q1))
    a2 = 0.5 * np.linalg.norm(np.cross(Q3 - Q1, Q4 - Q1))
    return float(a1 + a2)


def _reflect_dip_to_range(dip_deg):
    """Reflect dip into the physical range [0, 90] degrees."""
    dip = float(dip_deg)
    while dip < 0.0 or dip > 90.0:
        if dip < 0.0:
            dip = -dip
        if dip > 90.0:
            dip = 180.0 - dip
    return dip


def jitter_orientation(dip_deg, dipdir_deg, fisher_k=None,
                       min_sigma_deg=0.35, max_sigma_deg=4.0):
    """
    Apply a small angular perturbation so DFN orientations are continuous
    rather than exact repeats of measured rows.

    The perturbation scale is tied to Fisher concentration using the simple
    approximation sigma ~= degrees(1 / sqrt(kappa)). Higher kappa therefore
    gives tighter jitter.
    """
    if fisher_k is not None and np.isfinite(fisher_k) and fisher_k > 0:
        sigma_deg = np.degrees(1.0 / np.sqrt(float(fisher_k)))
    else:
        sigma_deg = 2.0

    sigma_deg = float(np.clip(sigma_deg, min_sigma_deg, max_sigma_deg))

    dip_jittered = _reflect_dip_to_range(np.random.normal(float(dip_deg), sigma_deg))
    dipdir_jittered = float(np.mod(np.random.normal(float(dipdir_deg), sigma_deg), 360.0))
    return dip_jittered, dipdir_jittered
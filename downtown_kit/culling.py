"""
Hyperplane extreme-point visibility test against an observer bounding box.
"""

import numpy as np


def cull_building_backfaces(verts: np.ndarray, faces: np.ndarray, b_box, exempt_faces=None):
    """
    Culls building faces that can never face any observer inside b_box = (xmin, xmax, ymin, ymax, zmin, zmax).
    Returns: verts_kept, faces_kept, num_culled.
    """
    if faces.shape[0] == 0:
        return verts, faces, 0

    bx_min, bx_max, by_min, by_max, bz_min, bz_max = b_box
    vA = verts[faces[:, 0]]
    vB = verts[faces[:, 1]]
    vC = verts[faces[:, 2]]
    centroids = (vA + vB + vC) * (1.0 / 3.0)
    raw_normals = np.cross(vB - vA, vC - vA)
    unit_normals = raw_normals / np.maximum(np.linalg.norm(raw_normals, axis=1, keepdims=True), 1e-12)

    # Extreme point inside observer box aligned with the normal
    p_star = np.column_stack((
        np.where(unit_normals[:, 0] > 0.0, bx_max, bx_min),
        np.where(unit_normals[:, 1] > 0.0, by_max, by_min),
        np.where(unit_normals[:, 2] > 0.0, bz_max, bz_min)
    ))

    max_vis_dot = np.einsum("ij,ij->i", unit_normals, p_star - centroids)
    cull_mask = max_vis_dot <= 0.0

    if exempt_faces is not None and len(exempt_faces) == faces.shape[0]:
        cull_mask = cull_mask & (~exempt_faces)

    faces_kept = faces[~cull_mask]
    used_v = np.unique(faces_kept)
    remap = np.full(verts.shape[0], -1, dtype=np.int32)
    remap[used_v] = np.arange(used_v.shape[0], dtype=np.int32)
    return verts[used_v], remap[faces_kept], int(np.sum(cull_mask))

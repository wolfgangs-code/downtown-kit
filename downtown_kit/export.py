"""
glTF 2.0 binary (.glb) mesh packaging, mesh deduplication, and PBR material attachment.
"""

import os
import numpy as np
import trimesh
from PIL import Image


def save_metric_glb(glb_path: str, vertices: np.ndarray, faces: np.ndarray) -> float:
    """Exports an untextured metric glTF binary mesh. Returns file size in MB."""
    os.makedirs(os.path.dirname(os.path.abspath(glb_path)) or ".", exist_ok=True)
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    mesh.export(glb_path, file_type="glb")
    return os.path.getsize(glb_path) / (1024.0 * 1024.0)


def load_mesh_for_texturing(glb_path: str, decimals: int = 3):
    """
    Loads an existing GLB, rounds vertices, deduplicates vertices, re-indexes faces,
    and removes degenerate triangles. Returns cleaned (verts, faces).
    """
    mesh = trimesh.load(glb_path, force="mesh", process=False)
    rounded_v = np.round(mesh.vertices, decimals=decimals)
    _, unique_idx, inv_idx = np.unique(rounded_v, axis=0, return_index=True, return_inverse=True)
    verts = mesh.vertices[unique_idx].astype(np.float32)  # In METERS
    faces = inv_idx[mesh.faces].astype(np.int32)
    valid_f = (faces[:, 0] != faces[:, 1]) & (faces[:, 1] != faces[:, 2]) & (faces[:, 2] != faces[:, 0])
    return verts, faces[valid_f]


def create_pbr_material(
    name: str,
    base_color_image: Image.Image,
    metallic_factor: float = 0.03,
    roughness_factor: float = 0.85
) -> trimesh.visual.material.PBRMaterial:
    """Constructs a standard glTF PBR metallic-roughness material."""
    return trimesh.visual.material.PBRMaterial(
        name=name,
        baseColorTexture=base_color_image,
        metallicFactor=metallic_factor,
        roughnessFactor=roughness_factor
    )


def save_textured_glb(
    glb_path: str,
    vertices: np.ndarray,
    faces: np.ndarray,
    uvs: np.ndarray,
    material: trimesh.visual.material.PBRMaterial
) -> float:
    """Binds UVs and PBR material to mesh and exports final textured GLB. Returns file size in MB."""
    os.makedirs(os.path.dirname(os.path.abspath(glb_path)) or ".", exist_ok=True)
    clean_mesh = trimesh.Trimesh(
        vertices=vertices,
        faces=faces,
        visual=trimesh.visual.TextureVisuals(uv=uvs, material=material),
        process=False
    )
    clean_mesh.export(glb_path, file_type="glb")
    return os.path.getsize(glb_path) / (1024.0 * 1024.0)

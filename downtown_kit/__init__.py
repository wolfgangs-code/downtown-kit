"""
downtown-kit: Geospatial 3D digital twin mesh and texture tools for game engines.
"""

from .coords import GeoFrame, METERS_TO_US_FT, FT_TO_METERS
from .export import save_metric_glb, load_mesh_for_texturing, create_pbr_material, save_textured_glb
from .dem import (
    stream_high_res_core_dem,
    stream_distant_dem,
    build_dem_interpolator,
    get_datum_elevation
)
from .imagery import (
    tile_to_quadkey,
    lat_lon_to_tile,
    tile_to_lat_lon,
    stream_and_cache_ve_mosaic
)
from .osm import fetch_osm_elements, parse_osm_infrastructure, extract_valid_polygons
from .culling import cull_building_backfaces

__all__ = [
    "GeoFrame",
    "METERS_TO_US_FT",
    "FT_TO_METERS",
    "save_metric_glb",
    "load_mesh_for_texturing",
    "create_pbr_material",
    "save_textured_glb",
    "stream_high_res_core_dem",
    "stream_distant_dem",
    "build_dem_interpolator",
    "get_datum_elevation",
    "tile_to_quadkey",
    "lat_lon_to_tile",
    "tile_to_lat_lon",
    "stream_and_cache_ve_mosaic",
    "fetch_osm_elements",
    "parse_osm_infrastructure",
    "extract_valid_polygons",
    "cull_building_backfaces"
]

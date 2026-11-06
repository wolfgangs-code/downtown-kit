"""
Microsoft VirtualEarth (Bing Maps) Quadkey indexing and concurrent tile mosaic downloading.
"""

import os
import math
import numpy as np
from PIL import Image
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from osgeo import gdal, osr


def tile_to_quadkey(xtile: int, ytile: int, zoom: int) -> str:
    """Encodes Bing Maps Quadkey string from slippy tile coordinates."""
    quadkey = []
    for i in range(zoom, 0, -1):
        digit = 0
        mask = 1 << (i - 1)
        if (xtile & mask) != 0:
            digit += 1
        if (ytile & mask) != 0:
            digit += 2
        quadkey.append(str(digit))
    return "".join(quadkey)


def lat_lon_to_tile(lat: float, lon: float, zoom: int):
    """Calculates slippy tile indices from WGS84 coordinates."""
    lat_rad = math.radians(lat)
    n = 2.0 ** zoom
    xtile = int((lon + 180.0) / 360.0 * n)
    ytile = int((1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n)
    return xtile, ytile


def tile_to_lat_lon(xtile: int, ytile: int, zoom: int):
    """Calculates northwest corner WGS84 coordinates of slippy tile."""
    n = 2.0 ** zoom
    lon = xtile / n * 360.0 - 180.0
    lat_rad = math.atan(math.sinh(math.pi * (1.0 - 2.0 * ytile / n)))
    lat = math.degrees(lat_rad)
    return lat, lon


def fetch_single_ve_tile(tx: int, ty: int, zoom: int) -> np.ndarray:
    """Downloads a single 256x256 aerial JPEG tile via Bing Quadkey."""
    quadkey = tile_to_quadkey(tx, ty, zoom)
    subdomain = (tx + ty) % 4
    url = f"https://ecn.t{subdomain}.tiles.virtualearth.net/tiles/a{quadkey}.jpeg?g=1&pr=odbl&n=z"
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    try:
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=10) as resp:
            img = Image.open(resp).convert("RGB")
            return np.array(img, dtype=np.uint8)
    except Exception:
        return np.full((256, 256, 3), 128, dtype=np.uint8)


def stream_and_cache_ve_mosaic(bbox_wgs, zoom_level: int, cache_tif_path: str, max_workers: int = 16):
    """
    Downloads and mosaics aerial tiles covering bbox_wgs = (min_lon, min_lat, max_lon, max_lat).
    Saves and caches as georeferenced GeoTIFF.
    Returns: canvas (H, W, 3) uint8, (nw_lon, nw_lat, pixel_w, pixel_h).
    """
    if os.path.exists(cache_tif_path):
        ds = gdal.Open(cache_tif_path)
        rgb = np.dstack([
            ds.GetRasterBand(1).ReadAsArray(),
            ds.GetRasterBand(2).ReadAsArray(),
            ds.GetRasterBand(3).ReadAsArray()
        ])
        gt = ds.GetGeoTransform()
        ds = None
        return rgb, (gt[0], gt[3], gt[1], gt[5])

    min_lon, min_lat, max_lon, max_lat = bbox_wgs
    x_min, y_min_tile = lat_lon_to_tile(max_lat, min_lon, zoom_level)
    x_max, y_max_tile = lat_lon_to_tile(min_lat, max_lon, zoom_level)

    x_start, x_end = min(x_min, x_max), max(x_min, x_max)
    y_start, y_end = min(y_min_tile, y_max_tile), max(y_min_tile, y_max_tile)

    num_x = (x_end - x_start) + 1
    num_y = (y_end - y_start) + 1
    canvas_w = num_x * 256
    canvas_h = num_y * 256

    print(f"  Streaming {num_x}x{num_y} tiles (Zoom {zoom_level}) -> {canvas_w}x{canvas_h} px...")

    coords = [(tx, ty) for ty in range(y_start, y_end + 1) for tx in range(x_start, x_end + 1)]
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        tile_images = list(pool.map(lambda pt: fetch_single_ve_tile(pt[0], pt[1], zoom_level), coords))

    canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
    idx = 0
    for iy in range(num_y):
        for ix in range(num_x):
            canvas[iy * 256:(iy + 1) * 256, ix * 256:(ix + 1) * 256, :] = tile_images[idx]
            idx += 1

    nw_lat, nw_lon = tile_to_lat_lon(x_start, y_start, zoom_level)
    se_lat, se_lon = tile_to_lat_lon(x_end + 1, y_end + 1, zoom_level)

    pixel_w = (se_lon - nw_lon) / float(canvas_w)
    pixel_h = (se_lat - nw_lat) / float(canvas_h)

    os.makedirs(os.path.dirname(os.path.abspath(cache_tif_path)) or ".", exist_ok=True)
    driver = gdal.GetDriverByName("GTiff")
    out_ds = driver.Create(cache_tif_path, canvas_w, canvas_h, 3, gdal.GDT_Byte, options=["COMPRESS=LZW"])
    out_ds.GetRasterBand(1).WriteArray(canvas[:, :, 0])
    out_ds.GetRasterBand(2).WriteArray(canvas[:, :, 1])
    out_ds.GetRasterBand(3).WriteArray(canvas[:, :, 2])
    out_ds.SetGeoTransform([nw_lon, pixel_w, 0.0, nw_lat, 0.0, pixel_h])
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(4326)
    out_ds.SetProjection(srs.ExportToWkt())
    out_ds.FlushCache()
    out_ds = None

    return canvas, (nw_lon, nw_lat, pixel_w, pixel_h)

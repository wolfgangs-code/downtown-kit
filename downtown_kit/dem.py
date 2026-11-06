"""
USGS 3DEP LiDAR digital elevation model (DEM) ingestion, GDAL warping, and interpolation.
"""

import os
import json
import urllib.request
import urllib.parse
import numpy as np
from scipy import ndimage
from scipy.interpolate import RegularGridInterpolator
from osgeo import gdal

from .coords import GeoFrame, METERS_TO_US_FT


def stream_high_res_core_dem(frame: GeoFrame, cache_filename: str, half_span: float):
    """
    Ingests 1-meter USGS 3DEP LiDAR DEM within the specified half-span.
    Falls back sequentially: USGS 1m COGs -> 3DEP ImageServer -> S3 1/3-arcsec tile.
    """
    if os.path.exists(cache_filename):
        print(f"Loading cached core 1-meter elevation: {cache_filename} ...")
        ds = gdal.Open(cache_filename)
        arr_m = ds.GetRasterBand(1).ReadAsArray().astype(np.float64)
        warp_dim = arr_m.shape[1]
        ds = None
        return arr_m, warp_dim

    min_x, max_x = frame.x0_ft - half_span, frame.x0_ft + half_span
    min_y, max_y = frame.y0_ft - half_span, frame.y0_ft + half_span
    warp_dim = int(np.ceil((half_span * 2.0) / 3.28084)) + 1

    c_lons, c_lats = frame.to_wgs84.transform([min_x, max_x, min_x, max_x], [min_y, min_y, max_y, max_y])
    bbox = (min(c_lons), min(c_lats), max(c_lons), max(c_lats))
    bbox_str = f"{bbox[0]:.6f},{bbox[1]:.6f},{bbox[2]:.6f},{bbox[3]:.6f}"

    print(f"Streaming high-resolution 1-meter LiDAR DEM ({warp_dim}x{warp_dim} raster)...")
    candidate_urls = []
    try:
        q = urllib.parse.urlencode({
            "datasets": "Digital Elevation Model (DEM) 1 meter",
            "bbox": bbox_str,
            "prodFormats": "GeoTIFF",
            "outputFormat": "JSON"
        })
        req = urllib.request.Request(
            f"https://tnmaccess.nationalmap.gov/api/v1/products?{q}",
            headers={"User-Agent": "GeoTwin1mLiDAR/Pipeline"}
        )
        with urllib.request.urlopen(req, timeout=20) as r:
            payload = json.loads(r.read().decode("utf-8"))
        for it in payload.get("items", []):
            dl = it.get("downloadURL", "")
            if dl.endswith(".tif"):
                candidate_urls.append(f"/vsicurl/{dl}")
    except Exception:
        pass

    wcs_url = (
        f"/vsicurl/https://elevation.nationalmap.gov/arcgis/rest/services/3DEPElevation/ImageServer/exportImage?"
        f"bbox={bbox[0]:.6f},{bbox[1]:.6f},{bbox[2]:.6f},{bbox[3]:.6f}"
        f"&bboxSR=4326&imageSR={frame.epsg_sp}&size={warp_dim},{warp_dim}&format=tiff&pixelType=F32&f=image"
    )

    attempts = []
    if candidate_urls:
        attempts.append(("USGS 1-Meter COG Tiles", candidate_urls))
    attempts.append(("USGS 3DEP ImageServer", [wcs_url]))
    attempts.append((
        "USGS 1/3-arcsec Seamless Tile",
        ["/vsicurl/https://prd-tnm.s3.amazonaws.com/StagedProducts/Elevation/13/TIFF/current/n36w091/USGS_13_n36w091.tif"]
    ))

    arr_m = None
    for label, urls in attempts:
        try:
            opts = gdal.WarpOptions(
                format="MEM",
                dstSRS=f"EPSG:{frame.epsg_sp}",
                outputBounds=(min_x, min_y, max_x, max_y),
                width=warp_dim,
                height=warp_dim,
                resampleAlg=gdal.GRA_Bilinear,
                outputType=gdal.GDT_Float32,
                multithread=True
            )
            ds = gdal.Warp("", urls, options=opts)
            if ds:
                raw_data = ds.GetRasterBand(1).ReadAsArray().astype(np.float64)
                valid = np.isfinite(raw_data) & (raw_data > -1000.0)
                if np.mean(valid) >= 0.70:
                    if not np.all(valid):
                        _, idx = ndimage.distance_transform_edt(~valid, return_distances=True, return_indices=True)
                        raw_data = raw_data[tuple(idx)]
                    arr_m = raw_data
                    print(f"  Ingested LiDAR DEM from: {label}")
                    break
        except Exception:
            pass

    if arr_m is None:
        raise RuntimeError("Failed to ingest 1-meter LiDAR DEM from any configured endpoint.")

    os.makedirs(os.path.dirname(os.path.abspath(cache_filename)) or ".", exist_ok=True)
    driver = gdal.GetDriverByName("GTiff")
    out_ds = driver.Create(cache_filename, warp_dim, warp_dim, 1, gdal.GDT_Float32)
    out_ds.GetRasterBand(1).WriteArray(arr_m.astype(np.float32))
    out_ds.FlushCache()
    out_ds = None
    return arr_m, warp_dim


def stream_distant_dem(frame: GeoFrame, cache_filename: str, half_span: float):
    """Ingests regional elevation from USGS Seamless 1/3-arcsec DEM."""
    if os.path.exists(cache_filename):
        print(f"Loading cached distant elevation: {cache_filename} ...")
        ds = gdal.Open(cache_filename)
        arr_m = ds.GetRasterBand(1).ReadAsArray().astype(np.float64)
        warp_dim = arr_m.shape[1]
        ds = None
        return arr_m, warp_dim

    min_x, max_x = frame.x0_ft - half_span, frame.x0_ft + half_span
    min_y, max_y = frame.y0_ft - half_span, frame.y0_ft + half_span
    warp_dim = 1400

    urls = ["/vsicurl/https://prd-tnm.s3.amazonaws.com/StagedProducts/Elevation/13/TIFF/USGS_Seamless_DEM_13.vrt"]
    opts = gdal.WarpOptions(
        format="MEM",
        dstSRS=f"EPSG:{frame.epsg_sp}",
        outputBounds=(min_x, min_y, max_x, max_y),
        width=warp_dim,
        height=warp_dim,
        resampleAlg=gdal.GRA_Bilinear,
        outputType=gdal.GDT_Float32,
        multithread=True
    )
    ds = gdal.Warp("", urls, options=opts)
    raw_data = ds.GetRasterBand(1).ReadAsArray().astype(np.float64)
    valid = np.isfinite(raw_data) & (raw_data > -1000.0)
    if not np.all(valid):
        _, idx = ndimage.distance_transform_edt(~valid, return_distances=True, return_indices=True)
        raw_data = raw_data[tuple(idx)]

    os.makedirs(os.path.dirname(os.path.abspath(cache_filename)) or ".", exist_ok=True)
    driver = gdal.GetDriverByName("GTiff")
    out_ds = driver.Create(cache_filename, warp_dim, warp_dim, 1, gdal.GDT_Float32)
    out_ds.GetRasterBand(1).WriteArray(raw_data.astype(np.float32))
    out_ds.FlushCache()
    out_ds = None
    return raw_data, warp_dim


def build_dem_interpolator(arr_m: np.ndarray, half_span: float) -> RegularGridInterpolator:
    """Constructs a 2D bilinear elevation interpolator in feet over [-half_span, half_span]."""
    dim = arr_m.shape[1]
    arr_ft = arr_m * METERS_TO_US_FT
    dx = np.linspace(-half_span, half_span, dim, dtype=np.float64)
    dy = np.linspace(half_span, -half_span, dim, dtype=np.float64)
    return RegularGridInterpolator(
        (dy[::-1], dx),
        arr_ft[::-1, :],
        method="linear",
        bounds_error=False,
        fill_value=None
    )


def get_datum_elevation(interp_fn: RegularGridInterpolator) -> float:
    """Evaluates elevation at the exact coordinate origin (0.0, 0.0)."""
    return float(interp_fn(np.array([[0.0, 0.0]]))[0])

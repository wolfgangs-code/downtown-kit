"""
Geodetic projections, rotated local street grid coordinate frames, and glTF metric conversions.
"""

import numpy as np
from pyproj import CRS, Transformer

METERS_TO_US_FT = 3937.0 / 1200.0
FT_TO_METERS = 1200.0 / 3937.0


class GeoFrame:
    """Manages geodetic projections, local rotated Cartesian grids, and glTF 2.0 metric conventions."""

    def __init__(
        self,
        lat_origin: float,
        lon_origin: float,
        epsg_sp: int = 2274,
        rotation_deg_ccw: float = 0.0,
        offset_west_ft: float = 0.0,
        offset_south_ft: float = 0.0,
        anchor_shift_ft: float = 0.0,
    ):
        self.lat_origin = lat_origin
        self.lon_origin = lon_origin
        self.epsg_sp = epsg_sp
        self.rotation_deg_ccw = rotation_deg_ccw
        self.offset_west_ft = offset_west_ft
        self.offset_south_ft = offset_south_ft
        self.anchor_shift_ft = anchor_shift_ft

        self.crs_wgs84 = CRS.from_epsg(4326)
        self.crs_sp = CRS.from_epsg(epsg_sp)

        self.to_state_plane = Transformer.from_crs(self.crs_wgs84, self.crs_sp, always_xy=True)
        self.to_wgs84 = Transformer.from_crs(self.crs_sp, self.crs_wgs84, always_xy=True)

        self.theta_rad = np.deg2rad(rotation_deg_ccw)
        self.cos_t = float(np.cos(self.theta_rad))
        self.sin_t = float(np.sin(self.theta_rad))

        self.x0_raw, self.y0_raw = self.to_state_plane.transform(lon_origin, lat_origin)
        self.x0_ft = self.x0_raw + anchor_shift_ft - offset_west_ft
        self.y0_ft = self.y0_raw + anchor_shift_ft - offset_south_ft

        self.lon_shifted, self.lat_shifted = self.to_wgs84.transform(self.x0_ft, self.y0_ft)

    def local_to_relative_sp(self, x_local, y_local):
        """Rotate local Cartesian coordinates to relative State Plane offsets."""
        dx_sp = x_local * self.cos_t + y_local * self.sin_t
        dy_sp = -x_local * self.sin_t + y_local * self.cos_t
        return dx_sp, dy_sp

    def local_to_state_plane(self, x_local, y_local):
        """Convert local Cartesian (X, Y) feet to State Plane (X, Y) feet."""
        dx_sp, dy_sp = self.local_to_relative_sp(x_local, y_local)
        return self.x0_ft + dx_sp, self.y0_ft + dy_sp

    def state_plane_to_local(self, xs_sp, ys_sp):
        """Convert State Plane coordinates (US Survey Feet) to local rotated Cartesian coordinates."""
        dx = np.asarray(xs_sp, dtype=np.float64) - self.x0_ft
        dy = np.asarray(ys_sp, dtype=np.float64) - self.y0_ft
        x_loc = dx * self.cos_t - dy * self.sin_t
        y_loc = dx * self.sin_t + dy * self.cos_t
        return np.column_stack((x_loc, y_loc))

    def wgs84_to_local(self, lons, lats):
        """Convert WGS84 lon/lat to local rotated Cartesian coordinates in feet."""
        xs_sp, ys_sp = self.to_state_plane.transform(lons, lats)
        return self.state_plane_to_local(xs_sp, ys_sp)

    def local_to_wgs84(self, x_local, y_local):
        """Convert local rotated Cartesian feet to WGS84 (lon, lat)."""
        xs_sp, ys_sp = self.local_to_state_plane(x_local, y_local)
        return self.to_wgs84.transform(xs_sp, ys_sp)

    def stream_span_ft(self, span_ft: float, buffer: float = 1.05) -> float:
        """Compute the true-north enclosing bounding span covering CCW diagonal rotation."""
        return span_ft * (self.cos_t + self.sin_t) * buffer

    def bbox_wgs84_for_span(self, half_span_ft: float):
        """Return (min_lon, min_lat, max_lon, max_lat) in WGS84 for a bounding square."""
        c_lons, c_lats = self.to_wgs84.transform(
            [self.x0_ft - half_span_ft, self.x0_ft + half_span_ft, self.x0_ft - half_span_ft, self.x0_ft + half_span_ft],
            [self.y0_ft - half_span_ft, self.y0_ft - half_span_ft, self.y0_ft + half_span_ft, self.y0_ft + half_span_ft]
        )
        return min(c_lons), min(c_lats), max(c_lons), max(c_lats)

    def to_gltf_coords(self, v_local: np.ndarray) -> np.ndarray:
        """Converts local US Survey Feet (X, Y, Z) into metric glTF 2.0 Y-Up in METERS: (X*s, Z*s, -Y*s)."""
        if v_local.shape[0] == 0:
            return np.empty((0, 3), dtype=np.float32)
        v_meters = v_local * FT_TO_METERS
        return np.column_stack((v_meters[:, 0], v_meters[:, 2], -v_meters[:, 1])).astype(np.float32)

    def from_gltf_coords(self, v_gltf: np.ndarray):
        """Converts metric glTF 2.0 Y-Up (X_m, Y_m, Z_m) back to local feet (X, Y, Z_elev)."""
        x_m = v_gltf[:, 0].astype(np.float64)
        z_m = v_gltf[:, 1].astype(np.float64)
        y_m = -v_gltf[:, 2].astype(np.float64)
        return x_m * METERS_TO_US_FT, y_m * METERS_TO_US_FT, z_m * METERS_TO_US_FT

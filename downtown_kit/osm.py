"""
OpenStreetMap (OSM) Overpass query engine, multi-mirror failover, and multipolygon topology assembler.
"""

import os
import json
import requests
import numpy as np
from shapely.geometry import Polygon, MultiPolygon, LineString
from shapely.ops import linemerge, polygonize, unary_union

from .coords import GeoFrame

OVERPASS_MIRRORS = [
    "https://overpass-api.de/api/interpreter",
    "https://lz4.overpass-api.de/api/interpreter",
    "https://z.overpass-api.de/api/interpreter",
    "https://maps.mail.ru/osm/tools/overpass/api/interpreter"
]


def fetch_osm_elements(bbox_wgs84, cache_file: str, query_type: str = "core_skybox", mirrors=None):
    """
    Fetches OSM elements via Overpass API with mirror fallback.
    Caches results locally to cache_file.
    """
    if os.path.exists(cache_file):
        with open(cache_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        if len(data) > 0:
            print(f"Loaded {len(data):,} cached OSM elements from {os.path.basename(cache_file)}.")
            return data

    mirrors = mirrors or OVERPASS_MIRRORS
    s, w, n, e = bbox_wgs84[1], bbox_wgs84[0], bbox_wgs84[3], bbox_wgs84[2]

    if query_type == "core_skybox":
        query = f"""
        [out:json][timeout:60];
        (
          way["building"]({s:.6f},{w:.6f},{n:.6f},{e:.6f});
          way["building:part"]({s:.6f},{w:.6f},{n:.6f},{e:.6f});
          relation["building"]({s:.6f},{w:.6f},{n:.6f},{e:.6f});
          relation["building:part"]({s:.6f},{w:.6f},{n:.6f},{e:.6f});
          way["bridge"="yes"]({s:.6f},{w:.6f},{n:.6f},{e:.6f});
          way["man_made"="bridge"]({s:.6f},{w:.6f},{n:.6f},{e:.6f});
          way["railway"="monorail"]({s:.6f},{w:.6f},{n:.6f},{e:.6f});
        );
        out geom;
        """
    else:  # distant
        query = f"""
        [out:json][timeout:45];
        (
          way["building"]["height"]({s:.6f},{w:.6f},{n:.6f},{e:.6f});
          way["building"]["building:levels"]({s:.6f},{w:.6f},{n:.6f},{e:.6f});
          way["bridge"="yes"]({s:.6f},{w:.6f},{n:.6f},{e:.6f});
        );
        out geom;
        """

    headers = {"User-Agent": "GeoTwin3DPipeline/OSM"}
    for ep in mirrors:
        try:
            r = requests.post(ep, data={"data": query}, headers=headers, timeout=50)
            if r.status_code == 200:
                elems = r.json().get("elements", [])
                if len(elems) > 0:
                    os.makedirs(os.path.dirname(os.path.abspath(cache_file)) or ".", exist_ok=True)
                    with open(cache_file, "w", encoding="utf-8") as f:
                        json.dump(elems, f)
                    return elems
        except Exception:
            pass

    return []


def extract_valid_polygons(geom_obj):
    """Sanitizes geometry and extracts non-empty Polygons."""
    if geom_obj is None or geom_obj.is_empty:
        return []
    if not geom_obj.is_valid:
        geom_obj = geom_obj.buffer(0)
    if geom_obj.is_empty:
        return []
    if geom_obj.geom_type == "Polygon":
        return [geom_obj]
    elif geom_obj.geom_type in ("MultiPolygon", "GeometryCollection"):
        return [g for g in geom_obj.geoms if g.geom_type == "Polygon" and not g.is_empty]
    return []


def parse_osm_infrastructure(osm_elements, frame: GeoFrame):
    """
    Parses raw OSM elements into:
    1. parsed_building_items: list of (list_of_polygons, tags_dict)
    2. parsed_bridge_items: list of (xy_loc_array, tags_dict)
    Coordinates are returned in local rotated Cartesian feet.
    """
    ways_by_id = {el["id"]: el for el in osm_elements if el.get("type") == "way"}

    def _geom_to_local_xy(geom):
        lons = [p["lon"] for p in geom]
        lats = [p["lat"] for p in geom]
        return frame.wgs84_to_local(lons, lats)

    parsed_building_items = []
    parsed_bridge_items = []
    consumed_multipolygon_ways = set()

    # 1. Process Relations (Courtyard buildings)
    for el in osm_elements:
        if el.get("type") != "relation":
            continue
        tags = dict(el.get("tags", {}))
        tags["_osm_id"] = el.get("id")
        members = el.get("members", [])
        if tags.get("type") != "multipolygon" and not any(m.get("role") == "inner" for m in members):
            continue

        outer_lines, inner_lines = [], []
        member_way_ids = []
        for m in members:
            if m.get("type") != "way":
                continue
            wid = m.get("ref")
            m_geom = m.get("geometry") or (ways_by_id.get(wid, {}).get("geometry"))
            if not m_geom or len(m_geom) < 2:
                continue
            ls = LineString(_geom_to_local_xy(m_geom))
            role = m.get("role", "outer")
            if role in ("outer", ""):
                outer_lines.append(ls)
                member_way_ids.append(wid)
                if wid in ways_by_id:
                    for k, v in ways_by_id[wid].get("tags", {}).items():
                        if k not in tags:
                            tags[k] = v
            elif role == "inner":
                inner_lines.append(ls)
                member_way_ids.append(wid)

        if not outer_lines:
            continue
        try:
            merged_outer = linemerge(outer_lines)
            outer_polys = list(polygonize(merged_outer)) if merged_outer.geom_type == "MultiLineString" else [Polygon(merged_outer)]
            if not outer_polys:
                continue
            outer_geom = unary_union(outer_polys)
            if inner_lines:
                merged_inner = linemerge(inner_lines)
                inner_polys = list(polygonize(merged_inner)) if merged_inner.geom_type == "MultiLineString" else [Polygon(merged_inner)]
                final_geom = outer_geom.difference(unary_union(inner_polys)) if inner_polys else outer_geom
            else:
                final_geom = outer_geom

            polys = extract_valid_polygons(final_geom)
            if polys:
                parsed_building_items.append((polys, tags))
                for wid in member_way_ids:
                    consumed_multipolygon_ways.add(wid)
        except Exception:
            continue

    # 2. Process Standalone Ways (Buildings & Elevated Bridge Decks)
    for el in osm_elements:
        if el.get("type") != "way" or el.get("id") in consumed_multipolygon_ways:
            continue
        geom = el.get("geometry", [])
        if len(geom) < 2:
            continue
        tags = dict(el.get("tags", {}))
        tags["_osm_id"] = el.get("id")
        xy_loc = _geom_to_local_xy(geom)

        is_bridge = (
            tags.get("bridge") in ("yes", "viaduct") or
            tags.get("man_made") == "bridge" or
            tags.get("railway") == "monorail"
        )
        if is_bridge:
            parsed_bridge_items.append((xy_loc, tags))
            continue

        if len(geom) >= 4 and ("building" in tags or "building:part" in tags):
            try:
                polys = extract_valid_polygons(Polygon(xy_loc))
                if polys:
                    parsed_building_items.append((polys, tags))
            except Exception:
                continue

    return parsed_building_items, parsed_bridge_items

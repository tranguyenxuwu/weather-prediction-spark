"""
Land Mask Utility
=================
Determines whether a (lat, lon) coordinate is on land or ocean using the
world-countries.json GeoJSON file and shapely's PreparedGeometry for fast
point-in-polygon queries.

Usage:
    from helpers.land_mask import is_land, is_ocean, filter_ocean_only
"""

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from shapely.geometry import Point, shape, MultiPolygon
from shapely.prepared import prep

log = logging.getLogger("LandMask")

# ── Module-level cache ────────────────────────────────────────────────────────

_PREPARED_LAND: "list | None" = None


def _load_land_polygons():
    """Load and cache prepared land polygons from world-countries.json."""
    global _PREPARED_LAND

    if _PREPARED_LAND is not None:
        return _PREPARED_LAND

    geojson_path = Path(__file__).resolve().parent.parent / "world-countries.json"
    if not geojson_path.exists():
        raise FileNotFoundError(
            f"GeoJSON not found at {geojson_path}. "
            "Download from https://raw.githubusercontent.com/"
            "python-visualization/folium/main/examples/data/world-countries.json"
        )

    log.info(f"Loading land polygons from {geojson_path.name}...")
    with open(geojson_path, "r") as f:
        data = json.load(f)

    polygons = []
    for feature in data["features"]:
        try:
            geom = shape(feature["geometry"])
            if geom.is_valid:
                polygons.append(prep(geom))
        except Exception as e:
            name = feature.get("properties", {}).get("name", "?")
            log.debug(f"   Skipping {name}: {e}")

    _PREPARED_LAND = polygons
    log.info(f"   Loaded {len(polygons)} country polygons")
    return _PREPARED_LAND


# ── Public API ────────────────────────────────────────────────────────────────


def is_land(lat: float, lon: float) -> bool:
    """Return True if the coordinate falls inside any country polygon."""
    prepared = _load_land_polygons()
    pt = Point(lon, lat)  # shapely uses (x=lon, y=lat)
    return any(p.contains(pt) for p in prepared)


def is_ocean(lat: float, lon: float) -> bool:
    """Return True if the coordinate is NOT on land."""
    return not is_land(lat, lon)


def filter_ocean_only(
    df: pd.DataFrame,
    lat_col: str = "lat",
    lon_col: str = "lon",
) -> pd.DataFrame:
    """
    Drop rows whose (lat, lon) falls on land.

    Uses vectorised batch checking for efficiency: builds the unique set of
    (lat, lon) pairs, tests each once, then filters the full DataFrame.

    Returns a copy — the original DataFrame is not modified.
    """
    prepared = _load_land_polygons()

    # Get unique coordinate pairs to minimise point-in-polygon calls
    coords = df[[lat_col, lon_col]].drop_duplicates()
    n_unique = len(coords)

    def _check(row):
        pt = Point(row[lon_col], row[lat_col])
        return any(p.contains(pt) for p in prepared)

    log.info(f"   Checking {n_unique:,} unique cells for land/ocean...")
    coords["_is_land"] = coords.apply(_check, axis=1)

    n_land = coords["_is_land"].sum()
    n_ocean = n_unique - n_land
    log.info(f"   Land: {n_land:,} | Ocean: {n_ocean:,} "
             f"({n_ocean / n_unique:.1%} ocean)")

    # Merge the flag back and filter
    result = df.merge(coords, on=[lat_col, lon_col], how="left")
    result = result[~result["_is_land"]].drop(columns=["_is_land"])

    return result.reset_index(drop=True)

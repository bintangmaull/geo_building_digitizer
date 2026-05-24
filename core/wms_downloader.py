"""
wms_downloader.py
Modul untuk mengunduh citra dari WMS/XYZ tile service dan
menyambungkannya (stitch) menjadi satu file GeoTIFF georeferensi.

Mendukung:
- XYZ tile endpoints (Google, ESRI, Bing, OSM, dll.)
- WMS standar GetMap endpoints
- Polygon/Rectangle AOI dalam EPSG:4326
"""

import os
import io
import math
import struct
import threading
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import numpy as np

# ── Public WMS/XYZ Presets ──────────────────────────────────────────────────
WMS_PRESETS = {
    "Google Satellite": {
        "url": "https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}",
        "type": "xyz",
        "max_zoom": 20,
        "attribution": "© Google",
    },
    "ESRI World Imagery": {
        "url": "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        "type": "xyz",
        "max_zoom": 19,
        "attribution": "© Esri, Maxar, Earthstar Geographics",
    },
    "Bing Aerial": {
        "url": "https://ecn.t3.tiles.virtualearth.net/tiles/a{q}.jpeg?g=1",
        "type": "bing",
        "max_zoom": 19,
        "attribution": "© Microsoft Bing",
    },
    "OpenStreetMap": {
        "url": "https://tile.openstreetmap.org/{z}/{x}/{y}.png",
        "type": "xyz",
        "max_zoom": 19,
        "attribution": "© OpenStreetMap contributors",
    },
}

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "image/webp,image/apng,image/*,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.google.com/",
}


# ── Tile Math Utilities ──────────────────────────────────────────────────────

def _lon_to_tile_x(lon: float, zoom: int) -> int:
    return int((lon + 180.0) / 360.0 * (1 << zoom))


def _lat_to_tile_y(lat: float, zoom: int) -> int:
    lat_r = math.radians(lat)
    return int((1.0 - math.log(math.tan(lat_r) + 1.0 / math.cos(lat_r)) / math.pi) / 2.0 * (1 << zoom))


def _tile_to_lon(x: int, zoom: int) -> float:
    return x / (1 << zoom) * 360.0 - 180.0


def _tile_to_lat(y: int, zoom: int) -> float:
    n = math.pi - 2.0 * math.pi * y / (1 << zoom)
    return math.degrees(math.atan(math.sinh(n)))


def _bbox_to_tiles(
    min_lon: float, min_lat: float, max_lon: float, max_lat: float, zoom: int
) -> Tuple[int, int, int, int]:
    """Return (x_min, y_min, x_max, y_max) tile indices."""
    x_min = _lon_to_tile_x(min_lon, zoom)
    x_max = _lon_to_tile_x(max_lon, zoom)
    y_min = _lat_to_tile_y(max_lat, zoom)  # Note: y flipped
    y_max = _lat_to_tile_y(min_lat, zoom)
    return x_min, y_min, x_max, y_max


def _bing_quadkey(x: int, y: int, z: int) -> str:
    """Convert XYZ tile indices to Bing quadkey."""
    quadkey = ""
    for i in range(z, 0, -1):
        digit = 0
        mask = 1 << (i - 1)
        if x & mask:
            digit += 1
        if y & mask:
            digit += 2
        quadkey += str(digit)
    return quadkey


def estimate_tile_count(
    min_lon: float, min_lat: float, max_lon: float, max_lat: float, zoom: int
) -> int:
    """Estimate number of tiles needed to cover the AOI."""
    x_min, y_min, x_max, y_max = _bbox_to_tiles(min_lon, min_lat, max_lon, max_lat, zoom)
    nx = x_max - x_min + 1
    ny = y_max - y_min + 1
    return max(nx * ny, 0)


def estimate_download_size_mb(tile_count: int) -> float:
    """Rough estimate: ~30KB per tile on average."""
    return tile_count * 30 / 1024


def get_best_zoom(
    min_lon: float, min_lat: float, max_lon: float, max_lat: float,
    max_zoom: int = 20, max_tiles: int = 400
) -> int:
    """
    Find the highest zoom level where tile count <= max_tiles.
    This protects against accidentally downloading thousands of tiles.
    """
    for z in range(max_zoom, 10, -1):
        count = estimate_tile_count(min_lon, min_lat, max_lon, max_lat, z)
        if count <= max_tiles:
            return z
    return 12  # fallback


# ── Core Download Function ───────────────────────────────────────────────────

def download_wms_aoi(
    wms_url: str,
    aoi_bounds: Tuple[float, float, float, float],  # (min_lon, min_lat, max_lon, max_lat)
    output_path: str,
    zoom: Optional[int] = None,
    wms_type: str = "xyz",
    max_tiles: int = 600,
    log_callback: Optional[Callable] = None,
    progress_callback: Optional[Callable] = None,
    cancel_check: Optional[Callable] = None,
) -> str:
    """
    Download tiles from a WMS/XYZ service covering the AOI and stitch into GeoTIFF.

    Args:
        wms_url: URL template with {x}, {y}, {z} placeholders (or {q} for Bing)
        aoi_bounds: (min_lon, min_lat, max_lon, max_lat) in EPSG:4326
        output_path: Destination .tif file path
        zoom: Zoom level (None = auto-detect best)
        wms_type: "xyz", "bing", or "wms"
        max_tiles: Safety cap on tiles
        log_callback: fn(message, level)
        progress_callback: fn(percent, message)
        cancel_check: fn() -> bool

    Returns:
        Path to the created GeoTIFF
    """
    import requests
    from PIL import Image
    import rasterio
    from rasterio.transform import from_bounds
    from rasterio.crs import CRS

    def log(msg, level="info"):
        if log_callback:
            log_callback(msg, level)

    def progress(pct, msg=""):
        if progress_callback:
            progress_callback(pct, msg)

    def is_cancelled():
        return cancel_check() if cancel_check else False

    min_lon, min_lat, max_lon, max_lat = aoi_bounds

    # Auto-detect best zoom level
    if zoom is None:
        zoom = get_best_zoom(min_lon, min_lat, max_lon, max_lat, max_zoom=20, max_tiles=max_tiles)
        log(f"🔍 Zoom level otomatis dipilih: {zoom}", "info")

    x_min, y_min, x_max, y_max = _bbox_to_tiles(min_lon, min_lat, max_lon, max_lat, zoom)
    nx = x_max - x_min + 1
    ny = y_max - y_min + 1
    total_tiles = nx * ny

    log(f"📥 Mengunduh {total_tiles} tile ({nx}×{ny}) pada zoom {zoom}...", "system")
    log(f"   Estimasi ukuran: ~{estimate_download_size_mb(total_tiles):.1f} MB", "info")
    progress(5, f"Mulai download {total_tiles} tile WMS...")

    if total_tiles > max_tiles:
        raise ValueError(
            f"AOI terlalu besar: {total_tiles} tile diperlukan (maksimum {max_tiles}). "
            f"Coba perkecil area AOI atau gunakan zoom lebih rendah."
        )

    # Calculate true geo extent of the tile grid
    tile_lon_min = _tile_to_lon(x_min, zoom)
    tile_lat_max = _tile_to_lat(y_min, zoom)
    tile_lon_max = _tile_to_lon(x_max + 1, zoom)
    tile_lat_min = _tile_to_lat(y_max + 1, zoom)

    TILE_SIZE = 256
    canvas = np.zeros((ny * TILE_SIZE, nx * TILE_SIZE, 3), dtype=np.uint8)

    session = requests.Session()
    session.headers.update(HEADERS)

    downloaded = 0
    failed = 0

    for row_idx, ty in enumerate(range(y_min, y_max + 1)):
        for col_idx, tx in enumerate(range(x_min, x_max + 1)):
            if is_cancelled():
                raise RuntimeError("Download dibatalkan oleh pengguna.")

            # Build URL
            if wms_type == "bing":
                q = _bing_quadkey(tx, ty, zoom)
                url = wms_url.replace("{q}", q)
            else:
                url = wms_url.replace("{x}", str(tx)).replace("{y}", str(ty)).replace("{z}", str(zoom))

            try:
                resp = session.get(url, timeout=15)
                resp.raise_for_status()
                img = Image.open(io.BytesIO(resp.content)).convert("RGB")
                img_arr = np.array(img)

                # Paste into canvas
                y0 = row_idx * TILE_SIZE
                x0 = col_idx * TILE_SIZE
                canvas[y0:y0 + TILE_SIZE, x0:x0 + TILE_SIZE] = img_arr
                downloaded += 1

            except Exception as e:
                log(f"⚠️ Tile ({tx},{ty}) gagal: {e}", "warning")
                failed += 1

            total_done = row_idx * nx + col_idx + 1
            pct = 5 + int((total_done / total_tiles) * 80)
            progress(pct, f"Download tile {total_done}/{total_tiles}...")

    log(f"✅ Download selesai: {downloaded} berhasil, {failed} gagal", "success")
    progress(87, "Menyimpan GeoTIFF...")

    # Save as GeoTIFF using rasterio
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Web Mercator bounds
    # Convert lat/lon extent to EPSG:3857 (Web Mercator)
    WEB_MERC_MAX = 20037508.342789244

    def lon_to_merc_x(lon):
        return lon * WEB_MERC_MAX / 180.0

    def lat_to_merc_y(lat):
        lat_r = math.radians(lat)
        y = math.log(math.tan(math.pi / 4 + lat_r / 2))
        return y * WEB_MERC_MAX / math.pi

    merc_x_min = lon_to_merc_x(tile_lon_min)
    merc_x_max = lon_to_merc_x(tile_lon_max)
    merc_y_min = lat_to_merc_y(tile_lat_min)
    merc_y_max = lat_to_merc_y(tile_lat_max)

    transform = from_bounds(merc_x_min, merc_y_min, merc_x_max, merc_y_max, canvas.shape[1], canvas.shape[0])

    # Fix: Instead of using CRS.from_epsg(3857) which queries the PROJ database 
    # (and crashes if PostGIS overrides PROJ_LIB with an incompatible version),
    # we explicitly define Web Mercator using its PROJ4 string. 
    # This completely bypasses the need for proj.db.
    proj4_3857 = "+proj=merc +a=6378137 +b=6378137 +lat_ts=0.0 +lon_0=0.0 +x_0=0.0 +y_0=0 +k=1.0 +units=m +nadgrids=@null +wktext +no_defs"
    
    try:
        crs = CRS.from_string(proj4_3857)
        with rasterio.open(
            output_path,
            "w",
            driver="GTiff",
            height=canvas.shape[0],
            width=canvas.shape[1],
            count=3,
            dtype=np.uint8,
            crs=crs,
            transform=transform,
            compress="lzw",
        ) as dst:
            for band in range(3):
                dst.write(canvas[:, :, band], band + 1)
    except Exception as e:
        log(f"Error saving GeoTIFF: {e}", "error")
        raise

    file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
    log(f"💾 GeoTIFF disimpan: {Path(output_path).name} ({file_size_mb:.1f} MB)", "success")
    progress(100, f"GeoTIFF siap: {Path(output_path).name}")

    return output_path


def get_aoi_bounds_from_polygon(polygon_coords: List[Tuple[float, float]]) -> Tuple[float, float, float, float]:
    """
    Compute bounding box from a list of (lon, lat) coordinate pairs.
    Returns (min_lon, min_lat, max_lon, max_lat).
    """
    lons = [c[0] for c in polygon_coords]
    lats = [c[1] for c in polygon_coords]
    return min(lons), min(lats), max(lons), max(lats)

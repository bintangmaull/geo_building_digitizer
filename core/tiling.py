"""
tiling.py
Adaptive sliding window tiling for large raster images.
Splits large GeoTIFF into tiles, processes them, and merges results back.
Handles overlap to prevent objects from being cut at tile boundaries.
"""

import os
import math
import numpy as np
from pathlib import Path
from typing import Callable, Generator, List, Optional, Tuple


# Default tile configuration
DEFAULT_TILE_SIZE = 1024   # pixels
DEFAULT_OVERLAP = 128      # pixels (12.5% overlap for 1024 tile)


def calculate_tile_grid(
    width: int,
    height: int,
    tile_size: int = DEFAULT_TILE_SIZE,
    overlap: int = DEFAULT_OVERLAP,
) -> List[Tuple[int, int, int, int]]:
    """
    Calculate tile coordinates (col_off, row_off, width, height) for a given image.
    Uses sliding window with overlap.

    Returns:
        List of (col_off, row_off, tile_w, tile_h) tuples in pixel coordinates
    """
    tiles = []
    step = tile_size - overlap

    row_off = 0
    while row_off < height:
        col_off = 0
        tile_h = min(tile_size, height - row_off)
        while col_off < width:
            tile_w = min(tile_size, width - col_off)
            tiles.append((col_off, row_off, tile_w, tile_h))
            col_off += step
        row_off += step

    return tiles


def read_tile_as_array(
    raster_path: str,
    col_off: int,
    row_off: int,
    tile_w: int,
    tile_h: int,
) -> Tuple[np.ndarray, dict]:
    """
    Read a tile window from a raster file as a numpy array (H, W, C) uint8.
    Also returns the tile's transform metadata for georeferencing.

    Returns:
        (rgb_array, tile_meta) where rgb_array is (H, W, 3) uint8
    """
    import rasterio
    from rasterio.windows import Window

    with rasterio.open(raster_path) as src:
        window = Window(col_off, row_off, tile_w, tile_h)
        tile_meta = {
            "transform": src.window_transform(window),
            "crs": src.crs,
            "col_off": col_off,
            "row_off": row_off,
            "tile_w": tile_w,
            "tile_h": tile_h,
        }

        # Read RGB bands (first 3 bands, or duplicate if grayscale)
        band_count = src.count
        if band_count >= 3:
            data = src.read([1, 2, 3], window=window)  # (3, H, W)
        elif band_count == 1:
            gray = src.read(1, window=window)
            data = np.stack([gray, gray, gray], axis=0)
        else:
            data = src.read(list(range(1, min(band_count + 1, 4))), window=window)

        # Normalize/clip to uint8
        result = np.clip(data, 0, 255).astype(np.uint8)

        # Convert (3, H, W) → (H, W, 3) for SAM/OpenCV compatibility
        rgb_array = np.transpose(result, (1, 2, 0))

    return rgb_array, tile_meta


def save_tile_as_geotiff(
    array: np.ndarray,
    tile_meta: dict,
    output_path: str,
) -> None:
    """Save a numpy RGB array as a GeoTIFF with proper georeference."""
    import rasterio
    from rasterio.transform import from_bounds

    h, w = array.shape[:2]
    if array.ndim == 2:
        array = array[np.newaxis, ...]  # (1, H, W)
    elif array.ndim == 3:
        array = np.transpose(array, (2, 0, 1))  # (C, H, W)

    with rasterio.open(
        output_path, "w",
        driver="GTiff",
        height=h, width=w,
        count=array.shape[0],
        dtype=array.dtype,
        crs=tile_meta["crs"],
        transform=tile_meta["transform"],
    ) as dst:
        dst.write(array)


def is_tile_empty(rgb_array: np.ndarray, edge_threshold: float = 0.004) -> bool:
    """
    Check if a tile is likely to contain buildings.
    Uses bilateral filtering, Canny edge density, and greenness index.
    Returns True if empty (likely no buildings), False otherwise.
    """
    import cv2

    # 1. Convert to gray
    gray = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2GRAY)

    # 2. Bilateral filter: smooths textures (like leaves/grass) but preserves sharp edges (like roofs)
    smoothed = cv2.bilateralFilter(gray, 9, 75, 75)

    # 3. Canny edge detection
    edges = cv2.Canny(smoothed, 50, 150)

    # 4. Calculate edge density (fraction of non-zero pixels)
    edge_pixels = np.count_nonzero(edges)
    total_pixels = edges.size
    density = edge_pixels / total_pixels

    # 5. Calculate greenness index to skip dense forests/grasslands
    r = rgb_array[:, :, 0].astype(float)
    g = rgb_array[:, :, 1].astype(float)
    b = rgb_array[:, :, 2].astype(float)
    greenness = g - np.maximum(r, b)
    mean_green = np.mean(greenness)

    # Combined heuristic: dense vegetation might have some texture edges, but it's very green
    if mean_green > 25.0 and density < 0.012:
        return True

    return density < edge_threshold


def tiles_generator(
    raster_path: str,
    tile_size: int = DEFAULT_TILE_SIZE,
    overlap: int = DEFAULT_OVERLAP,
    temp_dir: Optional[str] = None,
    progress_callback: Optional[Callable[[int, int, str], None]] = None,
    enable_filtering: bool = False,
) -> Generator[Tuple[str, dict, int, int], None, None]:
    """
    Generator that yields (tile_tiff_path, tile_meta, tile_idx, total_tiles)
    for each tile of the input raster.

    Saves tiles as temporary GeoTIFF files for SAM-Geo compatibility.

    Args:
        raster_path: Path to input GeoTIFF
        tile_size: Size of each tile in pixels
        overlap: Overlap in pixels between adjacent tiles
        temp_dir: Directory to save temporary tile files
        progress_callback: Optional function(tile_idx, total_tiles, message)
        enable_filtering: Whether to skip empty tiles
    """
    import rasterio

    if temp_dir is None:
        temp_dir = os.path.join(os.path.dirname(raster_path), "tiles_temp")
    os.makedirs(temp_dir, exist_ok=True)

    with rasterio.open(raster_path) as src:
        width = src.width
        height = src.height

    tiles = calculate_tile_grid(width, height, tile_size, overlap)
    total = len(tiles)

    for idx, (col_off, row_off, tile_w, tile_h) in enumerate(tiles):
        if progress_callback:
            progress_callback(idx, total, f"Mempersiapkan tile {idx+1}/{total}")

        tile_path = os.path.join(temp_dir, f"tile_{idx:04d}_{col_off}_{row_off}.tif")

        # Always read array first to determine if empty or to get metadata
        array, tile_meta = read_tile_as_array(raster_path, col_off, row_off, tile_w, tile_h)

        if enable_filtering and is_tile_empty(array):
            continue

        if not os.path.isfile(tile_path):
            save_tile_as_geotiff(array, tile_meta, tile_path)

        yield tile_path, tile_meta, idx, total


def cleanup_temp_tiles(temp_dir: str) -> None:
    """Remove temporary tile files."""
    import shutil
    if os.path.isdir(temp_dir):
        shutil.rmtree(temp_dir, ignore_errors=True)

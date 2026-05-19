"""
ecw_handler.py
Handles ECW file detection and conversion to GeoTIFF using GDAL from QGIS installation.
Provides fallback if ECW driver is not available (graceful error with user guidance).
"""

import os
import subprocess
import shutil
import tempfile
import platform
from pathlib import Path
from typing import Callable, Optional


# Common GDAL binary locations (Windows)
GDAL_SEARCH_PATHS = [
    r"C:\Program Files\QGIS 3.30.2\apps\qgis\bin",
    r"C:\Program Files\QGIS 3.30.2\apps\qgis-lts\bin",
    r"C:\OSGeo4W\bin",
    r"C:\OSGeo4W64\bin",
    r"C:\Program Files\GDAL",
    r"C:\Miniconda3\Library\bin",
    r"C:\ProgramData\Miniconda3\Library\bin",
]


def find_gdal_translate() -> Optional[str]:
    """Search for gdal_translate executable in common locations and PATH."""
    # Check PATH first
    found = shutil.which("gdal_translate")
    if found:
        return found

    # Search known locations
    for search_path in GDAL_SEARCH_PATHS:
        candidate = os.path.join(search_path, "gdal_translate.exe")
        if os.path.isfile(candidate):
            return candidate

    # Search QGIS installation directories recursively (limited depth)
    qgis_dirs = [
        r"C:\Program Files",
        r"C:\Program Files (x86)",
    ]
    for base in qgis_dirs:
        if not os.path.isdir(base):
            continue
        for entry in os.scandir(base):
            if "QGIS" in entry.name and entry.is_dir():
                for root, dirs, files in os.walk(entry.path):
                    if "gdal_translate.exe" in files:
                        return os.path.join(root, "gdal_translate.exe")

    return None


def check_ecw_support(gdal_bin: str) -> bool:
    """Check if the found GDAL binary supports ECW format."""
    gdalinfo = os.path.join(os.path.dirname(gdal_bin), "gdalinfo.exe")
    if not os.path.isfile(gdalinfo):
        return False
    try:
        result = subprocess.run(
            [gdalinfo, "--formats"],
            capture_output=True, text=True, timeout=10
        )
        return "ECW" in result.stdout
    except Exception:
        return False


def convert_ecw_to_geotiff(
    ecw_path: str,
    output_dir: str,
    gdal_translate_path: str,
    progress_callback: Optional[Callable[[int, str], None]] = None,
) -> str:
    """
    Convert an ECW file to GeoTIFF using gdal_translate.
    Returns path to the converted GeoTIFF file.

    Args:
        ecw_path: Path to input ECW file
        output_dir: Directory to save the output GeoTIFF
        gdal_translate_path: Path to gdal_translate executable
        progress_callback: Optional function(percent, message) for progress updates

    Returns:
        str: Path to output GeoTIFF file

    Raises:
        FileNotFoundError: If ECW file or gdal_translate not found
        RuntimeError: If conversion fails
    """
    if not os.path.isfile(ecw_path):
        raise FileNotFoundError(f"ECW file not found: {ecw_path}")
    if not os.path.isfile(gdal_translate_path):
        raise FileNotFoundError(f"gdal_translate not found: {gdal_translate_path}")

    os.makedirs(output_dir, exist_ok=True)
    basename = Path(ecw_path).stem
    output_path = os.path.join(output_dir, f"{basename}_converted.tif")

    if progress_callback:
        progress_callback(5, f"Mengkonversi ECW → GeoTIFF: {Path(ecw_path).name}")

    # Build gdal_translate command
    # -co options for Cloud Optimized GeoTIFF (COG) for efficient reading
    cmd = [
        gdal_translate_path,
        "-of", "GTiff",
        "-co", "TILED=YES",
        "-co", "COMPRESS=LZW",
        "-co", "BIGTIFF=YES",   # Handle files > 4GB
        "-co", "NUM_THREADS=ALL_CPUS",
        ecw_path,
        output_path,
    ]

    if progress_callback:
        progress_callback(10, "Menjalankan konversi GDAL...")

    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            creationflags=subprocess.CREATE_NO_WINDOW if platform.system() == "Windows" else 0,
        )
        output_lines = []
        for line in process.stdout:
            line = line.strip()
            output_lines.append(line)
            # Parse GDAL progress output (format: "10...20...30...")
            if "..." in line:
                try:
                    parts = line.replace(".", " ").split()
                    for p in parts:
                        if p.isdigit():
                            pct = int(p)
                            if progress_callback:
                                progress_callback(10 + int(pct * 0.15), f"Konversi ECW: {pct}%")
                except Exception:
                    pass

        process.wait()
        if process.returncode != 0:
            raise RuntimeError(
                f"gdal_translate gagal (exit code {process.returncode}):\n"
                + "\n".join(output_lines[-5:])
            )
    except subprocess.TimeoutExpired:
        process.kill()
        raise RuntimeError("Konversi ECW timeout (>300 detik)")

    if not os.path.isfile(output_path):
        raise RuntimeError(f"Output GeoTIFF tidak ditemukan setelah konversi: {output_path}")

    if progress_callback:
        progress_callback(25, f"Konversi selesai: {Path(output_path).name}")

    return output_path


def get_raster_info(raster_path: str) -> dict:
    """
    Read basic metadata from a raster file (GeoTIFF or ECW).
    Uses rasterio if available, falls back to subprocess gdalinfo.

    Returns dict with: width, height, crs, bounds, resolution, band_count, dtype
    """
    try:
        import rasterio
        with rasterio.open(raster_path) as src:
            bounds = src.bounds
            res = src.res
            return {
                "width": src.width,
                "height": src.height,
                "crs": str(src.crs),
                "epsg": src.crs.to_epsg() if src.crs else None,
                "bounds": {
                    "left": bounds.left,
                    "bottom": bounds.bottom,
                    "right": bounds.right,
                    "top": bounds.top,
                },
                "resolution_x": res[1],
                "resolution_y": res[0],
                "band_count": src.count,
                "dtype": str(src.dtypes[0]),
                "nodata": src.nodata,
            }
    except ImportError:
        raise RuntimeError("rasterio tidak terinstal. Jalankan: pip install rasterio")
    except Exception as e:
        raise RuntimeError(f"Gagal membaca metadata raster: {e}")

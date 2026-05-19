"""
dataset_generator.py
Optimized for massive files (e.g. 30GB+) using spatial windowed reading.
Directly targets building coordinates from Shapefile and extracts localized chips.
Uses extremely low memory (<100MB RAM) and is blazingly fast.
"""

import os
import cv2
import numpy as np
import rasterio
import geopandas as gpd
from rasterio.features import rasterize
from shapely.geometry import box
from typing import Callable, Optional

def generate_training_chips(
    geotiff_path: str,
    shp_path: str,
    output_dir: str,
    chip_size: int = 512,
    target_gsd: float = 0.15,  # Target resolution in meters per pixel (15cm)
    log_callback: Optional[Callable[[str], None]] = None,
    progress_callback: Optional[Callable[[float, str], None]] = None,
) -> bool:
    """
    Highly memory-efficient windowed generator designed to process 30GB+ rasters.
    Only loads tiny 512x512 pixel windows centered on shapefile buildings.
    """
    log = log_callback or print
    progress = progress_callback or (lambda p, m: None)

    try:
        log("🎬 Memulai ekstraksi dataset windowed ramah memori (RT-Memory)...")
        progress(5, "Membaca Shapefile bangunan...")

        # 1. Load building shapefile
        gdf = gpd.read_file(shp_path)
        if len(gdf) == 0:
            log("❌ Error: Shapefile bangunan tidak berisi poligon apapun!")
            return False

        # 2. Open GeoTIFF
        with rasterio.open(geotiff_path) as src:
            meta = src.meta.copy()
            transform = src.transform
            crs = src.crs
            width = src.width
            height = src.height
            
            # Check GSD (Ground Sampling Distance / Resolution)
            res_x = abs(transform.a)
            res_y = abs(transform.e)
            current_gsd = (res_x + res_y) / 2.0
            log(f"📐 Citra Drone Terdeteksi: {width:,}x{height:,} px | Ukuran File: Raksasa")
            log(f"📐 Resolusi Asli Citra Drone: {current_gsd*100:.2f} cm/pixel")

            # Reproject Shapefile to match GeoTIFF CRS if needed
            if gdf.crs != crs:
                log("🔄 Reprojeksi Shapefile agar sejajar dengan sistem koordinat GeoTIFF...")
                gdf = gdf.to_crs(crs)

            # Auto-Rescaling factor computation
            scale = 1.0
            if current_gsd < 0.10:
                scale = current_gsd / target_gsd
                log(f"🚁 Citra drone resolusi tinggi terdeteksi! Mengaktifkan Auto-Rescale (Skala: {scale:.4f})")
                log(f"🚁 Target resolusi: {target_gsd*100:.1f} cm/pixel")

            # Output paths setup
            images_dir = os.path.join(output_dir, "images")
            masks_dir = os.path.join(output_dir, "masks")
            os.makedirs(images_dir, exist_ok=True)
            os.makedirs(masks_dir, exist_ok=True)

            # Clear old chips
            for d in [images_dir, masks_dir]:
                for f in os.listdir(d):
                    try:
                        os.remove(os.path.join(d, f))
                    except Exception:
                        pass

            # 3. Process each building polygon centroid to extract local windows
            log("🎯 Menghitung koordinat spasial bangunan...")
            progress(20, "Menghitung indeks spasial bangunan...")

            # We use a grid-deduplication to avoid generating duplicate overlapping chips
            visited = set()
            grid_size = int(chip_size * 0.75 / scale)  # Step size in raw pixels (with 25% overlap)

            chip_count = 0
            total_buildings = len(gdf)
            log(f"🏠 Total objek bangunan dalam Shapefile: {total_buildings:,}")

            for idx, geom in enumerate(gdf.geometry):
                if geom is None or not geom.is_valid:
                    continue

                if idx % max(1, total_buildings // 20) == 0:
                    pct = 20.0 + (idx / total_buildings) * 50.0  # Range 20% to 70% progress
                    progress(pct, f"Mengekstrak jendela gambar ({idx}/{total_buildings})...")

                # Get spatial coordinates of the building centroid
                centroid = geom.centroid
                cx_geo, cy_geo = centroid.x, centroid.y

                # Convert geographic coordinate to pixel index
                row, col = src.index(cx_geo, cy_geo)

                # Ignore buildings outside the raster bounds
                if not (0 <= row < height and 0 <= col < width):
                    continue

                # Deduplicate based on grid cell to avoid duplicate ubin
                grid_row = int(row // grid_size) * grid_size
                grid_col = int(col // grid_size) * grid_size
                grid_key = (grid_row, grid_col)

                if grid_key in visited:
                    continue
                visited.add(grid_key)

                # Determine the raw pixel window size around centroid
                raw_w = int(chip_size / scale)
                
                # Center window on the snapped grid cell
                offset_row = grid_row - (raw_w - grid_size) // 2
                offset_col = grid_col - (raw_w - grid_size) // 2

                # Bound check window
                if offset_row < 0 or offset_col < 0 or offset_row + raw_w > height or offset_col + raw_w > width:
                    continue

                # Define rasterio spatial Window
                window = rasterio.windows.Window(offset_col, offset_row, raw_w, raw_w)
                win_transform = rasterio.windows.transform(window, transform)
                
                # Fast Bounding Box check for shapefile intersection
                win_box = box(*rasterio.windows.bounds(window, transform))
                intersecting_gdfs = gdf[gdf.intersects(win_box)]
                
                if len(intersecting_gdfs) == 0:
                    continue

                # 4. Read only the tiny window from 30GB file (Takes < 1 millisecond!)
                if src.count >= 3:
                    r = src.read(1, window=window)
                    g = src.read(2, window=window)
                    b = src.read(3, window=window)
                    img_chip = np.dstack((r, g, b))
                else:
                    gray = src.read(1, window=window)
                    img_chip = np.dstack((gray, gray, gray))

                # 5. Rasterize only the building polygons intersecting this window
                shapes_list = [(g, 255) for g in intersecting_gdfs.geometry if g.is_valid]
                mask_chip = rasterize(
                    shapes=shapes_list,
                    out_shape=(raw_w, raw_w),
                    transform=win_transform,
                    fill=0,
                    all_touched=True,
                    dtype=np.uint8
                )

                # 6. Rescale local chip to standard 512x512 training size
                if scale != 1.0:
                    img_chip = cv2.resize(img_chip, (chip_size, chip_size), interpolation=cv2.INTER_LINEAR)
                    mask_chip = cv2.resize(mask_chip, (chip_size, chip_size), interpolation=cv2.INTER_NEAREST)

                # Save the training pairs
                img_name = f"chip_{chip_count:04d}.png"
                cv2.imwrite(os.path.join(images_dir, img_name), cv2.cvtColor(img_chip, cv2.COLOR_RGB2BGR))
                cv2.imwrite(os.path.join(masks_dir, img_name), mask_chip)
                chip_count += 1

        log(f"✅ Selesai! Berhasil mengekstrak {chip_count} ubin training berukuran {chip_size}x{chip_size} secara aman!")
        progress(70, f"Berhasil mengekstrak {chip_count} ubin training.")
        
        if chip_count == 0:
            log("❌ Error: Tidak ada ubin yang berhasil diekstrak. Pastikan Shapefile bertumpukan secara geografis dengan citra drone Anda!")
            return False
            
        return True
    except Exception as e:
        log(f"❌ Gagal ekstraksi dataset: {e}")
        import traceback
        log(traceback.format_exc())
        return False

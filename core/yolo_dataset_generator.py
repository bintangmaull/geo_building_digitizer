"""
yolo_dataset_generator.py
Optimized for massive files (e.g. 30GB+) using spatial windowed reading.
Directly targets building coordinates from Shapefile and extracts localized chips.
Outputs YOLOv8 compatible dataset format (.jpg/.png + .txt labels).
"""

import os
import cv2
import yaml
import numpy as np
import rasterio
import geopandas as gpd
from shapely.geometry import box
from typing import Callable, Optional

def generate_yolo_dataset(
    geotiff_path: str,
    shp_path: str,
    output_dir: str,
    chip_size: int = 640,
    target_gsd: float = 0.15,
    target_class_name: str = "bangunan",
    mode: str = "bbox",  # 'bbox' or 'segmentation'
    log_callback: Optional[Callable[[str], None]] = None,
    progress_callback: Optional[Callable[[float, str], None]] = None,
) -> bool:
    """
    Highly memory-efficient generator for YOLO dataset format.
    Creates dataset.yaml, images/train, and labels/train.
    """
    log = log_callback or print
    progress = progress_callback or (lambda p, m: None)

    try:
        log("🎬 Memulai ekstraksi dataset YOLO (RT-Memory)...")
        progress(5, f"Membaca Shapefile {target_class_name}...")

        # 1. Load building shapefile
        gdf = gpd.read_file(shp_path)
        if len(gdf) == 0:
            log(f"❌ Error: Shapefile {target_class_name} tidak berisi poligon apapun!")
            return False

        # 2. Open GeoTIFF
        with rasterio.open(geotiff_path) as src:
            transform = src.transform
            crs = src.crs
            width = src.width
            height = src.height
            
            # Check GSD (Ground Sampling Distance / Resolution)
            res_x = abs(transform.a)
            res_y = abs(transform.e)
            current_gsd = (res_x + res_y) / 2.0
            log(f"📐 Citra Drone: {width:,}x{height:,} px | Resolusi Asli: {current_gsd*100:.2f} cm/pixel")

            # Reproject Shapefile to match GeoTIFF CRS if needed
            if gdf.crs != crs:
                log("🔄 Reprojeksi Shapefile agar sejajar dengan sistem koordinat GeoTIFF...")
                gdf = gdf.to_crs(crs)

            # Auto-Rescaling factor computation
            scale = 1.0
            if current_gsd < 0.10:
                scale = current_gsd / target_gsd
                log(f"🚁 Target resolusi: {target_gsd*100:.1f} cm/pixel (Skala: {scale:.4f})")

            # Output paths setup
            images_dir = os.path.join(output_dir, "images", "train")
            labels_dir = os.path.join(output_dir, "labels", "train")
            os.makedirs(images_dir, exist_ok=True)
            os.makedirs(labels_dir, exist_ok=True)

            # Clear old chips
            for d in [images_dir, labels_dir]:
                for f in os.listdir(d):
                    try:
                        os.remove(os.path.join(d, f))
                    except Exception:
                        pass

            # 3. Create dataset.yaml
            yaml_path = os.path.join(output_dir, "dataset.yaml")
            dataset_yaml = {
                "path": os.path.abspath(output_dir),
                "train": "images/train",
                "val": "images/train",
                "names": {0: target_class_name}
            }
            with open(yaml_path, "w") as f:
                yaml.dump(dataset_yaml, f, default_flow_style=False)

            log("🎯 Menghitung kotak batas spasial...")
            progress(20, "Mengekstrak ubin gambar YOLO...")

            grid_size = int(chip_size / scale)
            step = int(grid_size * 0.75) # 25% overlap

            chip_count = 0
            
            # Tentukan area pencarian berdasarkan bounding box keseluruhan Shapefile
            minx, miny, maxx, maxy = gdf.total_bounds
            r1, c1 = src.index(minx, maxy) # Top Left
            r2, c2 = src.index(maxx, miny) # Bottom Right
            
            start_row = max(0, int(min(r1, r2)))
            end_row   = min(height, int(max(r1, r2)))
            start_col = max(0, int(min(c1, c2)))
            end_col   = min(width, int(max(c1, c2)))
            
            # Buat daftar koordinat grid
            grid_windows = []
            for r in range(start_row, end_row, step):
                for c in range(start_col, end_col, step):
                    grid_windows.append((r, c))
                    
            total_windows = len(grid_windows)
            log(f"🎯 Memindai {total_windows:,} area grid untuk mencari {target_class_name}...")

            for idx, (row, col) in enumerate(grid_windows):
                if idx % max(1, total_windows // 20) == 0:
                    pct = 20.0 + (idx / total_windows) * 50.0
                    progress(pct, f"Mengekstrak dataset YOLO ({idx}/{total_windows})...")

                if row + grid_size > height or col + grid_size > width:
                    continue # Skip grid yang keluar dari batas gambar

                window = rasterio.windows.Window(col, row, grid_size, grid_size)
                win_transform = rasterio.windows.transform(window, transform)
                
                win_box = box(*rasterio.windows.bounds(window, transform))
                intersecting_gdfs = gdf[gdf.intersects(win_box)]
                
                if len(intersecting_gdfs) == 0:
                    continue

                # Read only the tiny window
                if src.count >= 3:
                    r = src.read(1, window=window)
                    g = src.read(2, window=window)
                    b = src.read(3, window=window)
                    img_chip = np.dstack((r, g, b))
                else:
                    gray = src.read(1, window=window)
                    img_chip = np.dstack((gray, gray, gray))

                if scale != 1.0:
                    img_chip = cv2.resize(img_chip, (chip_size, chip_size), interpolation=cv2.INTER_LINEAR)

                # Process YOLO labels
                yolo_labels = []
                for _, building in intersecting_gdfs.iterrows():
                    b_geom = building.geometry
                    if not b_geom.is_valid:
                        continue
                        
                    # Get intersection with window bounds
                    intersection = b_geom.intersection(win_box)
                    if intersection.is_empty:
                        continue
                        
                    # Get bounding box of the intersection
                    minx, miny, maxx, maxy = intersection.bounds
                    
                    # Convert geographic coordinates back to local window pixel coordinates
                    if mode == "segmentation":
                        # YOLO Segmentation format: class x1 y1 x2 y2 ... xn yn (normalized)
                        if intersection.geom_type == 'Polygon':
                            polys = [intersection]
                        elif intersection.geom_type == 'MultiPolygon':
                            polys = list(intersection.geoms)
                        else:
                            continue
                            
                        for poly in polys:
                            coords = list(poly.exterior.coords)
                            if len(coords) < 3:
                                continue
                                
                            norm_coords = []
                            for pt in coords:
                                lon, lat = pt[0], pt[1]
                                r, c = rasterio.transform.rowcol(win_transform, lon, lat)
                                px_x = max(0, min(c * scale, chip_size))
                                px_y = max(0, min(r * scale, chip_size))
                                norm_coords.append(f"{px_x / chip_size:.6f} {px_y / chip_size:.6f}")
                                
                            if len(norm_coords) > 2:
                                yolo_labels.append(f"0 {' '.join(norm_coords)}")
                    else:
                        # Standard YOLO Bounding Box format
                        minx, miny, maxx, maxy = intersection.bounds
                        
                        r1, c1 = rasterio.transform.rowcol(win_transform, minx, miny) # top left
                        r2, c2 = rasterio.transform.rowcol(win_transform, maxx, maxy) # bottom right
                        
                        px_minx = min(c1, c2) * scale
                        px_maxx = max(c1, c2) * scale
                        px_miny = min(r1, r2) * scale
                        px_maxy = max(r1, r2) * scale
                        
                        px_minx = max(0, min(px_minx, chip_size))
                        px_maxx = max(0, min(px_maxx, chip_size))
                        px_miny = max(0, min(px_miny, chip_size))
                        px_maxy = max(0, min(px_maxy, chip_size))
                        
                        box_w = px_maxx - px_minx
                        box_h = px_maxy - px_miny
                        
                        if box_w < 5 or box_h < 5:
                            continue # Ignore extremely tiny slivers
                            
                        x_center = (px_minx + box_w / 2.0) / chip_size
                        y_center = (px_miny + box_h / 2.0) / chip_size
                        norm_w = box_w / chip_size
                        norm_h = box_h / chip_size
                        
                        yolo_labels.append(f"0 {x_center:.6f} {y_center:.6f} {norm_w:.6f} {norm_h:.6f}")

                if not yolo_labels:
                    continue # Skip saving image if no valid labels

                # Save the training pairs
                img_name = f"chip_{chip_count:04d}.png"
                txt_name = f"chip_{chip_count:04d}.txt"
                
                cv2.imwrite(os.path.join(images_dir, img_name), cv2.cvtColor(img_chip, cv2.COLOR_RGB2BGR))
                with open(os.path.join(labels_dir, txt_name), "w") as f:
                    f.write("\n".join(yolo_labels))
                    
                chip_count += 1

        log(f"✅ Selesai! Berhasil mengekstrak {chip_count} ubin training YOLO secara aman!")
        progress(70, f"Berhasil mengekstrak {chip_count} ubin YOLO.")
        
        if chip_count == 0:
            log("❌ Error: Tidak ada ubin yang berhasil diekstrak.")
            return False
            
        return True
    except Exception as e:
        log(f"❌ Gagal ekstraksi dataset YOLO: {e}")
        import traceback
        log(traceback.format_exc())
        return False

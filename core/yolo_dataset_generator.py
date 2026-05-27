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
    shp_paths: dict,
    output_dir: str,
    chip_size: int = 640,
    target_gsd: float = 0.15,
    mode: str = "bbox",  # 'bbox' or 'segmentation'
    log_callback: Optional[Callable[[str], None]] = None,
    progress_callback: Optional[Callable[[float, str], None]] = None,
) -> bool:
    """
    Highly memory-efficient generator for YOLO dataset format.
    Creates dataset.yaml, images/train, and labels/train.
    Supports multi-class training via a dictionary of shapefiles.
    """
    log = log_callback or print
    progress = progress_callback or (lambda p, m: None)

    try:
        log("🎬 Memulai ekstraksi dataset YOLO (RT-Memory) Multi-Class...")
        progress(5, f"Membaca {len(shp_paths)} Shapefile...")

        if not shp_paths:
            log("❌ Error: Tidak ada shapefile yang diberikan!")
            return False

        # 1. Load shapefiles
        gdfs = []
        gdf_negative = None
        class_names = {}
        current_class_id = 0
        
        for class_name, path in shp_paths.items():
            gdf = gpd.read_file(path)
            if len(gdf) > 0:
                if class_name == "lainnya":
                    # Simpan sebagai area background/negatif (tidak dijadikan kelas)
                    log("🔍 Memuat Shapefile 'Lainnya' sebagai area Background/Negative Samples.")
                    gdf_negative = gdf
                else:
                    gdfs.append((current_class_id, class_name, gdf))
                    class_names[current_class_id] = class_name
                    current_class_id += 1
            else:
                log(f"⚠️ Peringatan: Shapefile {class_name} kosong dan akan diabaikan.")

        if not gdfs:
            log("❌ Error: Semua Shapefile target (selain Lainnya) kosong!")
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

            # Reproject Shapefiles to match GeoTIFF CRS if needed
            for i, (cid, cname, gdf) in enumerate(gdfs):
                if gdf.crs != crs:
                    log(f"🔄 Reprojeksi Shapefile {cname} agar sejajar dengan sistem koordinat GeoTIFF...")
                    gdfs[i] = (cid, cname, gdf.to_crs(crs))

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
                "names": class_names
            }
            with open(yaml_path, "w") as f:
                yaml.dump(dataset_yaml, f, default_flow_style=False)

            log("🎯 Menghitung kotak batas spasial...")
            progress(20, "Mengekstrak ubin gambar YOLO...")

            grid_size = int(chip_size / scale)
            step = int(grid_size * 0.75) # 25% overlap

            chip_count = 0
            
            # Tentukan area pencarian berdasarkan bounding box keseluruhan dari semua Shapefile
            minx_all, miny_all, maxx_all, maxy_all = float('inf'), float('inf'), float('-inf'), float('-inf')
            
            # Helper to merge bounds
            def merge_bounds(gdf, minx, miny, maxx, maxy):
                bx1, by1, bx2, by2 = gdf.total_bounds
                return min(minx, bx1), min(miny, by1), max(maxx, bx2), max(maxy, by2)
                
            for _, _, gdf in gdfs:
                minx_all, miny_all, maxx_all, maxy_all = merge_bounds(gdf, minx_all, miny_all, maxx_all, maxy_all)
                
            if gdf_negative is not None:
                minx_all, miny_all, maxx_all, maxy_all = merge_bounds(gdf_negative, minx_all, miny_all, maxx_all, maxy_all)

            r1, c1 = src.index(minx_all, maxy_all) # Top Left
            r2, c2 = src.index(maxx_all, miny_all) # Bottom Right
            
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
            log(f"🎯 Memindai {total_windows:,} area grid untuk multi-kelas...")

            for idx, (row, col) in enumerate(grid_windows):
                if idx % max(1, total_windows // 20) == 0:
                    pct = 20.0 + (idx / total_windows) * 50.0
                    progress(pct, f"Mengekstrak dataset YOLO ({idx}/{total_windows})...")

                if row + grid_size > height or col + grid_size > width:
                    continue # Skip grid yang keluar dari batas gambar

                window = rasterio.windows.Window(col, row, grid_size, grid_size)
                win_transform = rasterio.windows.transform(window, transform)
                
                win_box = box(*rasterio.windows.bounds(window, transform))
                
                # Cek interseksi untuk semua kelas utama
                intersecting_classes = []
                total_intersections = 0
                for cid, cname, gdf in gdfs:
                    intersecting_gdfs = gdf[gdf.intersects(win_box)]
                    if len(intersecting_gdfs) > 0:
                        intersecting_classes.append((cid, intersecting_gdfs))
                        total_intersections += len(intersecting_gdfs)
                
                # Logika Negative Sampling
                is_negative_sample = False
                if total_intersections == 0:
                    if gdf_negative is not None:
                        neg_intersect = gdf_negative[gdf_negative.intersects(win_box)]
                        if len(neg_intersect) > 0:
                            is_negative_sample = True
                            
                    if not is_negative_sample:
                        continue # Skip jika tidak ada objek utama dan bukan area negatif

                # Read only the tiny window
                if src.count >= 3:
                    r_band = src.read(1, window=window)
                    g_band = src.read(2, window=window)
                    b_band = src.read(3, window=window)
                    img_chip = np.dstack((r_band, g_band, b_band))
                else:
                    gray = src.read(1, window=window)
                    img_chip = np.dstack((gray, gray, gray))

                if scale != 1.0:
                    img_chip = cv2.resize(img_chip, (chip_size, chip_size), interpolation=cv2.INTER_LINEAR)

                yolo_labels = []
                
                if not is_negative_sample:
                    # Process YOLO labels untuk objek utama
                    for cid, intersecting_gdfs in intersecting_classes:
                        for _, feature in intersecting_gdfs.iterrows():
                            geom = feature.geometry
                            if not geom.is_valid:
                                continue
                                
                            # Get intersection with window bounds
                            intersection = geom.intersection(win_box)
                            if intersection.is_empty:
                                continue
                                
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
                                        r_px, c_px = rasterio.transform.rowcol(win_transform, lon, lat)
                                        px_x = max(0, min(c_px * scale, chip_size))
                                        px_y = max(0, min(r_px * scale, chip_size))
                                        norm_coords.append(f"{px_x / chip_size:.6f} {px_y / chip_size:.6f}")
                                        
                                    if len(norm_coords) > 2:
                                        yolo_labels.append(f"{cid} {' '.join(norm_coords)}")
                            else:
                                # Standard YOLO Bounding Box format
                                minx, miny, maxx, maxy = intersection.bounds
                                
                                r1_px, c1_px = rasterio.transform.rowcol(win_transform, minx, miny) # top left
                                r2_px, c2_px = rasterio.transform.rowcol(win_transform, maxx, maxy) # bottom right
                                
                                px_minx = min(c1_px, c2_px) * scale
                                px_maxx = max(c1_px, c2_px) * scale
                                px_miny = min(r1_px, r2_px) * scale
                                px_maxy = max(r1_px, r2_px) * scale
                                
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
                                
                                yolo_labels.append(f"{cid} {x_center:.6f} {y_center:.6f} {norm_w:.6f} {norm_h:.6f}")

                    if not yolo_labels:
                        continue # Skip saving image if no valid labels, unless it was a negative sample
                
                # Note: jika is_negative_sample = True, maka yolo_labels = [] (kosong)
                # Ini akan menciptakan file .txt kosong yang dibaca YOLO sebagai Background Image.

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

"""
vegetation_processor.py
Processor untuk digitasi Vegetasi dari citra drone/satelit.

Objek yang dideteksi:
    - Hutan / kawasan bervegetasi lebat
    - Taman kota / RTH (Ruang Terbuka Hijau)
    - Sawah / ladang pertanian
    - Kebun (perkebunan)
    - Semak / rumput

Pipeline:
    1. Tiling raster input
    2. Deteksi kandidat area vegetasi via OpenCV:
       - ExG (Excess Green Index): 2G - R - B > threshold
       - HSV green range: Hue 35–85
    3. Segmentasi SAM menggunakan point prompts dari deteksi CV
    4. Post-processing khusus vegetasi:
       - Konfirmasi greenness piksel > threshold
       - Filter area minimum >= 10 m²
       - TIDAK ada regularisasi sudut 90° (vegetasi organik)
       - TIDAK ada filter bayangan (vegetasi bisa teduh)
       - TIDAK ada filter aspek rasio (vegetasi bisa berbentuk apa saja)

CATATAN: File ini BERDIRI SENDIRI dan tidak mengimpor dari postprocess.py
         atau sam_processor.py (kode bangunan tidak disentuh).
"""

import os
import gc
import threading
import numpy as np
from pathlib import Path
from typing import Callable, List, Optional, Tuple


class VegetationProcessor:
    """
    Processor mandiri untuk deteksi dan segmentasi Vegetasi.
    Menggunakan SAM untuk segmentasi dengan prompt dari deteksi ExG/HSV OpenCV.
    """

    OBJECT_CLASS = "Vegetasi"
    OBJECT_COLOR = "#22C55E"   # Hijau — untuk preview panel

    def __init__(
        self,
        sam_model_name: str = "SAM2-Small (Seimbang, ~185MB)",
        models_dir: str = "models",
        device: str = "auto",
        exg_threshold: float = 15.0,
        greenness_confirm: float = 10.0,
        detection_mode: str = "predetect",   # "predetect" | "automatic" | "yolo"
        yolo_model_name: Optional[str] = None,
        log_callback: Optional[Callable[[str], None]] = None,
        progress_callback: Optional[Callable[[int, str], None]] = None,
    ):
        self.sam_model_name = sam_model_name
        self.models_dir = os.path.abspath(models_dir)
        self.exg_threshold = exg_threshold
        self.greenness_confirm = greenness_confirm
        self.detection_mode = detection_mode
        self.yolo_model_name = yolo_model_name

        self.log_callback = log_callback or (lambda msg: print(msg))
        self.progress_callback = progress_callback or (lambda pct, msg: None)
        self._cancel_event = threading.Event()
        self._sam = None
        self._yolo = None
        self._is_sam2 = False

        if device == "auto":
            try:
                import torch
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                self.device = "cpu"
        else:
            self.device = device

    def cancel(self):
        self._cancel_event.set()

    def is_cancelled(self) -> bool:
        return self._cancel_event.is_set()

    def reset_cancel(self):
        self._cancel_event.clear()

    def _log(self, msg: str):
        self.log_callback(f"[Vegetasi] {msg}")

    def _progress(self, pct: int, msg: str):
        self.progress_callback(pct, msg)

    # ──────────────────────────────────────────────────────────────
    # Model Loading
    # ──────────────────────────────────────────────────────────────

    def load_model(self):
        """Muat model SAM/SAM2 ke memori."""
        self._log(f"Memuat model SAM: {self.sam_model_name} [Mode: {self.detection_mode}]")
        is_auto = (self.detection_mode == "automatic")

        MODEL_CONFIGS = {
            "SAM2-Tiny (Cepat, ~155MB)":        {"type": "sam2", "model_id": "sam2-hiera-tiny"},
            "SAM2-Small (Seimbang, ~185MB)":     {"type": "sam2", "model_id": "sam2-hiera-small"},
            "SAM2-Base+ (Akurat, ~325MB)":       {"type": "sam2", "model_id": "sam2-hiera-base-plus"},
            "SAM2-Large (Sangat Akurat, ~898MB)":{"type": "sam2", "model_id": "sam2-hiera-large"},
            "SAM-B (Seimbang, ~375MB)":          {"type": "vit_b", "checkpoint": "sam_vit_b_01ec64.pth",
                                                   "download_url": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"},
            "SAM-L (Akurat, ~1.2GB)":            {"type": "vit_l", "checkpoint": "sam_vit_l_0b3195.pth",
                                                   "download_url": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth"},
            "SAM-H (Sangat Akurat, ~2.4GB)":     {"type": "vit_h", "checkpoint": "sam_vit_h_4b8939.pth",
                                                   "download_url": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"},
        }

        cfg = MODEL_CONFIGS.get(self.sam_model_name,
                                MODEL_CONFIGS["SAM2-Small (Seimbang, ~185MB)"])
        model_type = cfg.get("type")

        try:
            if model_type == "sam2":
                from samgeo import SamGeo2
                self._sam = SamGeo2(
                    model_id=cfg["model_id"],
                    device=self.device,
                    automatic=is_auto,
                    points_per_side=32 if is_auto else None,
                    pred_iou_thresh=0.72 if is_auto else None,
                    stability_score_thresh=0.75 if is_auto else None,
                    min_mask_region_area=100 if is_auto else None,
                )
                self._is_sam2 = True
                self._log(f"SAM2 ({cfg['model_id']}) dimuat di {self.device.upper()}")
            else:
                from samgeo import SamGeo
                model_path = os.path.join(self.models_dir, cfg["checkpoint"])
                if not os.path.exists(model_path):
                    import urllib.request
                    self._log(f"Mengunduh {cfg['checkpoint']}...")
                    urllib.request.urlretrieve(cfg["download_url"], model_path)
                self._sam = SamGeo(
                    model_type=model_type,
                    checkpoint=model_path,
                    device=self.device,
                    automatic=is_auto,
                    sam_kwargs={
                        "points_per_side": 32,
                        "pred_iou_thresh": 0.72,
                        "stability_score_thresh": 0.75,
                        "crop_n_layers": 0,
                        "min_mask_region_area": 100,
                    } if is_auto else None,
                )
                self._log(f"SAM ({model_type}) dimuat di {self.device.upper()}")
        except Exception as e:
            raise RuntimeError(f"Gagal memuat model SAM untuk Vegetasi: {e}")

        # Load YOLO if needed
        if self.detection_mode == "yolo" and self.yolo_model_name:
            try:
                from ultralytics import YOLO
                yolo_path = os.path.join(self.models_dir, self.yolo_model_name)
                if not os.path.exists(yolo_path):
                    self._log(f"⚠️ Model YOLO {self.yolo_model_name} tidak ditemukan!", "warning")
                else:
                    self._yolo = YOLO(yolo_path)
                    self._log(f"Model YOLO {self.yolo_model_name} dimuat.")
            except ImportError:
                self._log("⚠️ Gagal memuat YOLO (ultralytics belum terinstal)", "warning")
            except Exception as e:
                self._log(f"⚠️ Gagal memuat YOLO: {e}", "warning")

    def unload_model(self):
        """Bebaskan model dari memori."""
        if self._sam is not None:
            del self._sam
            self._sam = None
            
        if self._yolo is not None:
            del self._yolo
            self._yolo = None

        gc.collect()
        if self.device == "cuda":
            try:
                import torch
                torch.cuda.empty_cache()
            except Exception:
                pass
        self._log("Model dibebaskan dari memori")

    # ──────────────────────────────────────────────────────────────
    # Deteksi Kandidat Vegetasi (OpenCV)
    # ──────────────────────────────────────────────────────────────

    def detect_vegetation_points(
        self,
        image_path: str,
        min_area_px: float = 100.0,
        max_area_px: float = 5000000.0,
    ) -> Tuple[List[List[int]], List[int]]:
        """
        Deteksi kandidat area vegetasi menggunakan OpenCV.

        Karakteristik spektral vegetasi dari citra drone:
        - ExG (Excess Green Index) = 2G - R - B > threshold (biasanya > 15)
        - HSV Hue: 35–85 (hijau dari kuning-hijau hingga hijau tua)
        - Saturation > 30 (vegetasi bukan abu-abu/putih)

        Pola deteksi:
        - Titik di tengah setiap patch vegetasi yang cukup besar
        - SAM kemudian segmentasi batas vegetasi yang lebih akurat

        Returns:
            (point_coords, point_labels) — centroid patch vegetasi
        """
        import cv2
        import numpy as np

        img = cv2.imread(image_path)
        if img is None:
            return [], []

        h_img, w_img = img.shape[:2]
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # Perhatikan: cv2.imread mengembalikan BGR, bukan RGB!
        b_ch, g_ch, r_ch = cv2.split(img.astype(float))

        # --- 1. ExG Index (Excess Green) ---
        # ExG = 2G - R - B; piksel vegetasi biasanya > 15
        exg = 2.0 * g_ch - r_ch - b_ch
        exg_mask = (exg > self.exg_threshold).astype(np.uint8) * 255

        # --- 2. HSV Green Range ---
        # Hijau dalam BGR-to-HSV: Hue 35–85
        lower_green = np.array([35, 30, 30])
        upper_green = np.array([85, 255, 255])
        hsv_green_mask = cv2.inRange(hsv, lower_green, upper_green)

        # --- 3. Gabungkan kedua indikator (OR) ---
        veg_combined = cv2.bitwise_or(exg_mask, hsv_green_mask)

        # --- 4. Morfologi untuk menghilangkan noise dan menggabungkan patch ---
        kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
        veg_processed = cv2.morphologyEx(veg_combined, cv2.MORPH_OPEN, kernel_open)
        veg_processed = cv2.morphologyEx(veg_processed, cv2.MORPH_CLOSE, kernel_close)

        # --- 5. Temukan contour & ekstrak centroid ---
        contours, _ = cv2.findContours(veg_processed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        coords = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < min_area_px or area > max_area_px:
                continue

            M = cv2.moments(cnt)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                if 0 <= cx < w_img and 0 <= cy < h_img:
                    coords.append([cx, cy])

        if not coords:
            return [], []

        labels = [1] * len(coords)
        return coords, labels

    # ──────────────────────────────────────────────────────────────
    # Segmentasi SAM Single Tile
    # ──────────────────────────────────────────────────────────────

    def process_tile(
        self,
        tile_path: str,
        output_mask_path: str,
        point_coords: Optional[List[List[int]]] = None,
        point_labels: Optional[List[int]] = None,
        boxes: Optional[List[List[float]]] = None,
    ) -> bool:
        """Segmentasi SAM untuk satu tile citra vegetasi."""
        if self.is_cancelled():
            return False
        if self._sam is None:
            raise RuntimeError("Model belum dimuat. Panggil load_model() terlebih dahulu.")

        if not point_coords and not boxes:
            return False

        try:
            import rasterio
            import numpy as np

            self._sam.set_image(tile_path)

            with rasterio.open(tile_path) as src:
                h, w = src.height, src.width
                meta = src.meta.copy()
                meta.update(dtype="uint16", count=1, nodata=0)

            master_mask = np.zeros((h, w), dtype=np.uint16)
            temp_dir = os.path.dirname(output_mask_path)
            temp_path = os.path.join(temp_dir, f"_tmp_veg_{Path(tile_path).name}")

            if boxes:
                for idx, box in enumerate(boxes):
                    if self.is_cancelled(): return False
                    try:
                        if os.path.exists(temp_path): os.remove(temp_path)
                        x1, y1, x2, y2 = box
                        x1, x2 = max(0, min(x1, w)), max(0, min(x2, w))
                        y1, y2 = max(0, min(y1, h)), max(0, min(y2, h))
                        if abs(x2 - x1) < 2 or abs(y2 - y1) < 2:
                            continue
                        self._sam.predict(boxes=[x1, y1, x2, y2], output=temp_path)
                        if os.path.exists(temp_path):
                            with rasterio.open(temp_path) as p_src:
                                pmask = p_src.read(1)
                                veg_id = idx + 1
                                master_mask = np.where(pmask > 0, veg_id, master_mask)
                    except Exception as pe:
                        self._log(f"  ⚠️ Gagal segmentasi box {box}: {pe}")
            elif point_coords and point_labels:
                for idx, (coord, label) in enumerate(zip(point_coords, point_labels)):
                    if self.is_cancelled():
                        return False
                    try:
                        if os.path.exists(temp_path):
                            os.remove(temp_path)
                        self._sam.predict(
                            point_coords=[coord],
                            point_labels=[label],
                            output=temp_path,
                        )
                        if os.path.exists(temp_path):
                            with rasterio.open(temp_path) as p_src:
                                pmask = p_src.read(1)
                                veg_id = idx + 1
                                master_mask = np.where(pmask > 0, veg_id, master_mask)
                    except Exception as pe:
                        self._log(f"  ⚠️ Gagal segmentasi titik {coord}: {pe}")

            if os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except Exception:
                    pass

            with rasterio.open(output_mask_path, "w", **meta) as dst:
                dst.write(master_mask.astype(np.uint16), 1)

            return True

        except Exception as e:
            self._log(f"Error proses tile {Path(tile_path).name}: {e}")
            return False

    # ──────────────────────────────────────────────────────────────
    # Pipeline Utama
    # ──────────────────────────────────────────────────────────────

    def process_raster(
        self,
        raster_path: str,
        masks_dir: str,
        tile_size: int = 1024,
        overlap: int = 128,
        min_area_m2: float = 10.0,
        max_area_m2: float = 50000000.0,
        tile_progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> List[Tuple[str, dict]]:
        """Tile raster dan proses setiap tile untuk deteksi vegetasi."""
        from core.tiling import tiles_generator

        os.makedirs(masks_dir, exist_ok=True)
        results = []

        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        tiles_temp_dir = os.path.join(project_root, "temp", "tiles")

        self._log(f"Memulai tiling raster untuk deteksi vegetasi [{self.detection_mode}]: {Path(raster_path).name}")

        for tile_path, tile_meta, idx, total in tiles_generator(
            raster_path=raster_path,
            tile_size=tile_size,
            overlap=overlap,
            temp_dir=tiles_temp_dir,
            enable_filtering=False,
        ):
            if self.is_cancelled():
                self._log("Proses vegetasi dibatalkan.")
                break

            tile_name = Path(tile_path).stem
            mask_path = os.path.join(masks_dir, f"mask_veg_{tile_name}.tif")

            self._log(f"[{idx+1}/{total}] Tile vegetasi: {tile_name}")
            if tile_progress_callback:
                tile_progress_callback(idx + 1, total)

            if os.path.isfile(mask_path):
                os.remove(mask_path)

            transform = tile_meta["transform"]
            res_x = abs(transform.a)
            res_y = abs(transform.e)
            pixel_area_m2 = res_x * res_y
            min_px = min_area_m2 / pixel_area_m2 if pixel_area_m2 > 0 else 100.0
            max_px = max_area_m2 / pixel_area_m2 if pixel_area_m2 > 0 else 50000000.0

            if self.detection_mode == "automatic":
                self._log("  -> Mode Otomatis (Grid Buta)")
                success = self._process_tile_automatic(tile_path, mask_path)
            elif self.detection_mode == "yolo":
                if self._yolo is None:
                    self._log("  -> Gagal (YOLO tidak dimuat)")
                    continue
                yolo_results = self._yolo(tile_path, conf=0.25, verbose=False)
                if len(yolo_results) > 0 and len(yolo_results[0].boxes) > 0:
                    boxes = yolo_results[0].boxes.xyxy.cpu().numpy().tolist()
                    self._log(f"  -> YOLO Deteksi: {len(boxes)} kotak")
                    success = self.process_tile(tile_path, mask_path, boxes=boxes)
                else:
                    self._log("  -> YOLO Deteksi: 0 kotak")
                    continue
            else:
                point_coords, point_labels = self.detect_vegetation_points(tile_path, min_px, max_px)
                self._log(f"  -> Kandidat vegetasi: {len(point_coords)} titik")
                if not point_coords:
                    self._log("  -> Dilewati (tidak ada kandidat vegetasi)")
                    continue
                success = self.process_tile(tile_path, mask_path, point_coords=point_coords, point_labels=point_labels)

            if success and os.path.isfile(mask_path):
                results.append((mask_path, tile_meta))
                self._log(f"  -> Berhasil: {Path(mask_path).name}")
            else:
                self._log("  -> Gagal atau dibatalkan")

        if self.device == "cuda":
            try:
                import torch
                torch.cuda.empty_cache()
            except ImportError:
                pass
        gc.collect()

        return results

    def _process_tile_automatic(self, tile_path: str, output_mask_path: str) -> bool:
        """Mode Otomatis: SAM generate() grid buta tanpa prompt pre-deteksi."""
        if self.is_cancelled():
            return False
        if self._sam is None:
            raise RuntimeError("Model belum dimuat.")
        try:
            self._sam.generate(tile_path, output=output_mask_path)
            return os.path.isfile(output_mask_path)
        except Exception as e:
            self._log(f"Error generate otomatis {Path(tile_path).name}: {e}")
            return False

    # ──────────────────────────────────────────────────────────────
    # Post-processing Khusus Vegetasi
    # ──────────────────────────────────────────────────────────────

    def postprocess(
        self,
        mask_results: List[Tuple[str, dict]],
        original_raster_path: str,
        min_area_m2: float = 10.0,
        max_area_m2: float = 50000000.0,
        log_callback: Optional[Callable[[str], None]] = None,
    ) -> "geopandas.GeoDataFrame":
        """
        Post-processing khusus untuk poligon vegetasi:
        - Merge tile masks → vector
        - Filter area minimum >= 10 m²
        - Konfirmasi greenness piksel > threshold
        - Dissolve/gabung patch yang bersentuhan (vegetasi sering tersambung)
        - TIDAK ada regularisasi sudut 90°
        - Tambah kolom 'class' = 'Vegetasi'
        """
        import geopandas as gpd
        import rasterio
        from shapely.ops import unary_union

        log = log_callback or (lambda x: None)

        log("=== POST-PROCESSING VEGETASI ===")

        all_gdfs = []
        for mask_path, tile_meta in mask_results:
            if not os.path.isfile(mask_path):
                continue
            try:
                gdf_tile = self._mask_to_polygons(mask_path)
                if len(gdf_tile) > 0:
                    all_gdfs.append(gdf_tile)
            except Exception as e:
                log(f"  Gagal vektorisasi {Path(mask_path).name}: {e}")

        if not all_gdfs:
            log("Tidak ada poligon vegetasi yang berhasil diekstrak.")
            with rasterio.open(original_raster_path) as src:
                return gpd.GeoDataFrame(geometry=[], crs=src.crs)

        merged = gpd.pd.concat(all_gdfs, ignore_index=True)
        gdf = gpd.GeoDataFrame(merged, geometry="geometry", crs=all_gdfs[0].crs)
        log(f"Total poligon vegetasi awal: {len(gdf)}")

        # Deduplikasi
        gdf = self._deduplicate(gdf, log)

        # Filter area
        gdf = self._filter_area(gdf, min_area_m2, max_area_m2, log)

        # Konfirmasi greenness
        gdf = self._confirm_vegetation_pixels(gdf, original_raster_path, log)

        # Tambah kolom class
        gdf["class"] = self.OBJECT_CLASS

        log(f"=== POST-PROCESSING VEGETASI SELESAI: {len(gdf)} poligon ===")
        return gdf.reset_index(drop=True)

    # ──────────────────────────────────────────────────────────────
    # Helper Methods
    # ──────────────────────────────────────────────────────────────

    def _mask_to_polygons(self, mask_path: str) -> "geopandas.GeoDataFrame":
        import rasterio
        from rasterio.features import shapes
        import geopandas as gpd
        from shapely.geometry import shape
        import numpy as np

        with rasterio.open(mask_path) as src:
            data = src.read(1)
            transform = src.transform
            crs = src.crs

        unique_vals = np.unique(data)
        geoms = []
        for val in unique_vals:
            if val == 0:
                continue
            instance_mask = (data == val).astype(np.uint8)
            for geom_dict, v in shapes(instance_mask, mask=instance_mask, transform=transform):
                if v == 1:
                    geom = shape(geom_dict)
                    if geom.area > 0:
                        geoms.append(geom)

        if not geoms:
            return gpd.GeoDataFrame(geometry=[], crs=crs)
        return gpd.GeoDataFrame(geometry=geoms, crs=crs)

    def _deduplicate(self, gdf, log):
        gdf = gdf.copy()
        gdf["geometry"] = gdf.geometry.buffer(0)
        keep = [True] * len(gdf)
        geoms = list(gdf.geometry)
        for i in range(len(geoms)):
            if not keep[i]:
                continue
            for j in range(i + 1, len(geoms)):
                if not keep[j]:
                    continue
                try:
                    inter = geoms[i].intersection(geoms[j]).area
                    if inter == 0:
                        continue
                    union = geoms[i].union(geoms[j]).area
                    iou = inter / union if union > 0 else 0
                    if iou > 0.5:
                        keep[j] = False
                except Exception:
                    pass
        result = gdf.iloc[[k for k, v in enumerate(keep) if v]].copy()
        log(f"Poligon vegetasi setelah deduplication: {len(result)}")
        return result.reset_index(drop=True)

    def _filter_area(self, gdf, min_area_m2, max_area_m2, log):
        if len(gdf) == 0:
            return gdf
        if gdf.crs and gdf.crs.is_geographic:
            gdf_metric = gdf.to_crs(gdf.estimate_utm_crs())
        else:
            gdf_metric = gdf.copy()
        areas = gdf_metric.geometry.area
        mask = (areas >= min_area_m2) & (areas <= max_area_m2)
        result = gdf[mask].copy()
        log(f"Filter area vegetasi ({min_area_m2}–{max_area_m2} m²): {len(gdf)} -> {len(result)}")
        return result.reset_index(drop=True)

    def _confirm_vegetation_pixels(
        self,
        gdf: "geopandas.GeoDataFrame",
        raster_path: str,
        log,
    ) -> "geopandas.GeoDataFrame":
        """
        Konfirmasi poligon adalah vegetasi dengan cek greenness piksel.
        Greenness = mean_G - max(mean_R, mean_B) > threshold
        """
        try:
            import rasterio
            from rasterio.mask import mask as rio_mask
            from shapely.geometry import mapping

            with rasterio.open(raster_path) as src:
                raster_crs = src.crs
                if gdf.crs and gdf.crs != raster_crs:
                    gdf_raster = gdf.to_crs(raster_crs)
                else:
                    gdf_raster = gdf

                keep_mask = []
                for geom in gdf_raster.geometry:
                    try:
                        out_image, _ = rio_mask(src, [mapping(geom)], crop=True, nodata=0)
                        if out_image.shape[0] < 3:
                            keep_mask.append(True)
                            continue
                        r = out_image[0].astype(float)
                        g = out_image[1].astype(float)
                        b = out_image[2].astype(float)
                        valid = r > 0
                        if valid.sum() == 0:
                            keep_mask.append(True)
                            continue
                        mean_r = r[valid].mean()
                        mean_g = g[valid].mean()
                        mean_b = b[valid].mean()
                        greenness = mean_g - max(mean_r, mean_b)
                        keep_mask.append(greenness > self.greenness_confirm)
                    except Exception:
                        keep_mask.append(True)

            result = gdf[keep_mask].copy()
            removed = len(gdf) - len(result)
            log(f"Konfirmasi piksel vegetasi (greenness>{self.greenness_confirm}): dihapus {removed} poligon non-hijau")
            return result.reset_index(drop=True)
        except Exception as e:
            log(f"Konfirmasi piksel vegetasi gagal: {e}. Melanjutkan tanpa konfirmasi.")
            return gdf

"""
road_processor.py
Processor untuk digitasi Jalan & Infrastruktur dari citra drone/satelit.

Objek yang dideteksi:
    - Jalan (aspal, beton, tanah)
    - Jembatan
    - Rel kereta api
    - Area parkir (sebagai poligon lebar)

Pipeline:
    1. Tiling raster input
    2. Deteksi kandidat area jalan via OpenCV (warna abu-abu aspal, low saturation)
    3. Segmentasi SAM menggunakan point/box prompts dari deteksi CV
    4. Post-processing khusus jalan:
       - Filter elongated (aspek rasio >= 2.0)
       - Filter lebar minimal (>= 3 m)
       - Tidak ada regularisasi 90° (jalan bisa melengkung)
       - Tidak ada filter vegetasi/bayangan

CATATAN: File ini BERDIRI SENDIRI dan tidak mengimpor dari postprocess.py
         atau sam_processor.py (kode bangunan tidak disentuh).
"""

import os
import gc
import threading
import numpy as np
from pathlib import Path
from typing import Callable, List, Optional, Tuple


class RoadProcessor:
    """
    Processor mandiri untuk deteksi dan segmentasi Jalan & Infrastruktur.
    Menggunakan SAM untuk segmentasi dengan prompt dari deteksi warna OpenCV.
    """

    OBJECT_CLASS = "Jalan"
    OBJECT_COLOR = "#FF6B35"   # Orange — untuk preview panel

    def __init__(
        self,
        sam_model_name: str = "SAM2-Small (Seimbang, ~185MB)",
        models_dir: str = "models",
        device: str = "auto",
        points_per_side: int = 32,
        detection_mode: str = "predetect",   # "predetect" | "automatic" | "yolo"
        yolo_model_name: Optional[str] = None,
        log_callback: Optional[Callable[[str], None]] = None,
        progress_callback: Optional[Callable[[int, str], None]] = None,
    ):
        self.sam_model_name = sam_model_name
        self.models_dir = os.path.abspath(models_dir)
        self.points_per_side = points_per_side
        self.detection_mode = detection_mode
        self.yolo_model_name = yolo_model_name

        self.log_callback = log_callback or (lambda msg: print(msg))
        self.progress_callback = progress_callback or (lambda pct, msg: None)
        self._cancel_event = threading.Event()
        self._sam = None
        self._yolo = None
        self._is_sam2 = False

        # Determine device
        if device == "auto":
            try:
                import torch
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                self.device = "cpu"
        else:
            self.device = device

    def cancel(self):
        """Request cancellation."""
        self._cancel_event.set()

    def is_cancelled(self) -> bool:
        return self._cancel_event.is_set()

    def reset_cancel(self):
        self._cancel_event.clear()

    def _log(self, msg: str):
        self.log_callback(f"[Jalan] {msg}")

    def _progress(self, pct: int, msg: str):
        self.progress_callback(pct, msg)

    # ──────────────────────────────────────────────────────────────
    # Model Loading
    # ──────────────────────────────────────────────────────────────

    def load_model(self):
        """Muat model SAM/SAM2 ke memori."""
        self._log(f"Memuat model SAM: {self.sam_model_name} [Mode: {self.detection_mode}]")

        # Jika automatic, SAM perlu diinisialisasi dengan automatic=True
        is_auto = (self.detection_mode == "automatic")

        # Import MODEL_CONFIGS dari SAM processor (hanya konfigurasi, bukan kode pipeline)
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
                    points_per_side=self.points_per_side if is_auto else None,
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
                        "points_per_side": self.points_per_side,
                        "pred_iou_thresh": 0.72,
                        "stability_score_thresh": 0.75,
                        "crop_n_layers": 0,
                        "min_mask_region_area": 100,
                    } if is_auto else None,
                )
                self._log(f"SAM ({model_type}) dimuat di {self.device.upper()}")
        except Exception as e:
            raise RuntimeError(f"Gagal memuat model SAM untuk Jalan: {e}")

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
    # Deteksi Kandidat Jalan (OpenCV, berbasis warna)
    # ──────────────────────────────────────────────────────────────

    def detect_road_points(
        self,
        image_path: str,
        min_area_px: float = 500.0,
        max_area_px: float = 500000.0,
    ) -> Tuple[List[List[int]], List[int]]:
        """
        Deteksi kandidat area jalan menggunakan OpenCV.

        Karakteristik spektral jalan dari citra drone:
        - Warna abu-abu (aspal) atau putih (beton): saturation rendah, brightness sedang-tinggi
        - Bukan hijau (vegetasi) dan bukan sangat gelap (bayangan)
        - Bentuk memanjang (aspek rasio > 2.0)

        Returns:
            (point_coords, point_labels) — centroid dari kandidat jalan
        """
        import cv2
        import numpy as np

        img = cv2.imread(image_path)
        if img is None:
            return [], []

        h_img, w_img = img.shape[:2]
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # --- 1. Mask Vegetasi (hijau) — untuk eksklusi ---
        lower_green = np.array([35, 30, 30])
        upper_green = np.array([85, 255, 255])
        green_mask = cv2.inRange(hsv, lower_green, upper_green)

        b, g, r = cv2.split(img.astype(float))
        exg = 2.0 * g - r - b
        exg_mask = (exg > 15).astype(np.uint8) * 255
        veg_mask = cv2.bitwise_or(green_mask, exg_mask)

        # --- 2. Mask Bayangan (sangat gelap) ---
        lower_shadow = np.array([0, 0, 0])
        upper_shadow = np.array([180, 255, 50])
        shadow_mask = cv2.inRange(hsv, lower_shadow, upper_shadow)

        # --- 3. Deteksi warna aspal/beton (abu-abu, saturation rendah) ---
        # Aspal: HSV Saturation rendah (< 60), Value sedang (40–220)
        lower_road = np.array([0, 0, 40])
        upper_road = np.array([180, 60, 220])
        road_mask = cv2.inRange(hsv, lower_road, upper_road)

        # Eksklusi vegetasi dan bayangan dari mask jalan
        road_clean = cv2.bitwise_and(road_mask, cv2.bitwise_not(veg_mask))
        road_clean = cv2.bitwise_and(road_clean, cv2.bitwise_not(shadow_mask))

        # --- 4. Morfologi untuk menutup gap pada jalan ---
        kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
        road_processed = cv2.morphologyEx(road_clean, cv2.MORPH_OPEN, kernel_open)
        road_processed = cv2.morphologyEx(road_processed, cv2.MORPH_CLOSE, kernel_close)

        # --- 5. Temukan contour & filter berdasarkan aspek rasio ---
        contours, _ = cv2.findContours(road_processed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        coords = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < min_area_px or area > max_area_px:
                continue

            x, y, w, h = cv2.boundingRect(cnt)
            aspect = max(w, h) / max(min(w, h), 1)

            # Jalan biasanya memanjang: aspek rasio >= 2.0
            # Atau area besar (persimpangan/parkiran) aspek rasio bisa kecil
            if aspect >= 2.0 or area >= 5000:
                M = cv2.moments(cnt)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    if 0 <= cx < w_img and 0 <= cy < h_img:
                        if veg_mask[cy, cx] == 0:
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
        """Segmentasi SAM untuk satu tile citra jalan."""
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
            temp_path = os.path.join(temp_dir, f"_tmp_road_{Path(tile_path).name}")

            if boxes:
                for idx, box in enumerate(boxes):
                    if self.is_cancelled(): return False
                    try:
                        if os.path.exists(temp_path): os.remove(temp_path)
                        # Clip box to valid coordinates to prevent SAM errors
                        x1, y1, x2, y2 = box
                        x1, x2 = max(0, min(x1, w)), max(0, min(x2, w))
                        y1, y2 = max(0, min(y1, h)), max(0, min(y2, h))
                        if abs(x2 - x1) < 2 or abs(y2 - y1) < 2:
                            continue
                            
                        self._sam.predict(
                            boxes=[x1, y1, x2, y2],
                            output=temp_path,
                        )
                        if os.path.exists(temp_path):
                            with rasterio.open(temp_path) as p_src:
                                pmask = p_src.read(1)
                                road_id = idx + 1
                                master_mask = np.where(pmask > 0, road_id, master_mask)
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
                                road_id = idx + 1
                                master_mask = np.where(pmask > 0, road_id, master_mask)
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
    # Pipeline Utama: Proses Seluruh Raster
    # ──────────────────────────────────────────────────────────────

    def process_raster(
        self,
        raster_path: str,
        masks_dir: str,
        tile_size: int = 1024,
        overlap: int = 128,
        min_area_m2: float = 30.0,
        max_area_m2: float = 500000.0,
        tile_progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> List[Tuple[str, dict]]:
        """
        Tile raster dan proses setiap tile untuk deteksi jalan.

        Returns:
            List of (mask_tiff_path, tile_meta)
        """
        from core.tiling import tiles_generator

        os.makedirs(masks_dir, exist_ok=True)
        results = []

        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        tiles_temp_dir = os.path.join(project_root, "temp", "tiles")

        self._log(f"Memulai tiling raster untuk deteksi jalan [{self.detection_mode}]: {Path(raster_path).name}")

        for tile_path, tile_meta, idx, total in tiles_generator(
            raster_path=raster_path,
            tile_size=tile_size,
            overlap=overlap,
            temp_dir=tiles_temp_dir,
            enable_filtering=False,
        ):
            if self.is_cancelled():
                self._log("Proses jalan dibatalkan.")
                break

            tile_name = Path(tile_path).stem
            mask_path = os.path.join(masks_dir, f"mask_road_{tile_name}.tif")

            self._log(f"[{idx+1}/{total}] Tile jalan: {tile_name}")
            if tile_progress_callback:
                tile_progress_callback(idx + 1, total)

            if os.path.isfile(mask_path):
                os.remove(mask_path)

            # Hitung batas area dalam piksel
            transform = tile_meta["transform"]
            res_x = abs(transform.a)
            res_y = abs(transform.e)
            pixel_area_m2 = res_x * res_y
            min_px = min_area_m2 / pixel_area_m2 if pixel_area_m2 > 0 else 500.0
            max_px = max_area_m2 / pixel_area_m2 if pixel_area_m2 > 0 else 5000000.0

            if self.detection_mode == "automatic":
                # Mode Otomatis: SAM generate() tanpa prompt
                self._log("  -> Mode Otomatis (Grid Buta)")
                success = self._process_tile_automatic(tile_path, mask_path)
            elif self.detection_mode == "yolo":
                # Mode YOLO
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
                # Mode Pra-Deteksi Warna: OpenCV → titik prompt → SAM
                point_coords, point_labels = self.detect_road_points(tile_path, min_px, max_px)
                self._log(f"  -> Kandidat jalan: {len(point_coords)} titik")
                if not point_coords:
                    self._log("  -> Dilewati (tidak ada kandidat jalan)")
                    continue
                success = self.process_tile(tile_path, mask_path, point_coords=point_coords, point_labels=point_labels)

            if success and os.path.isfile(mask_path):
                results.append((mask_path, tile_meta))
                self._log(f"  -> Berhasil: {Path(mask_path).name}")
            else:
                self._log("  -> Gagal atau dibatalkan")

        # Bebaskan GPU
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
    # Post-processing Khusus Jalan
    # ──────────────────────────────────────────────────────────────

    def postprocess(
        self,
        mask_results: List[Tuple[str, dict]],
        original_raster_path: str,
        min_area_m2: float = 30.0,
        max_area_m2: float = 500000.0,
        log_callback: Optional[Callable[[str], None]] = None,
    ) -> "geopandas.GeoDataFrame":
        """
        Post-processing khusus untuk poligon jalan:
        - Merge tile mask → vector
        - Filter area (min luas jalan)
        - Filter aspek rasio >= 1.5 (jalan bersifat memanjang)
        - TIDAK ada regularisasi sudut 90° (jalan bisa melengkung)
        - TIDAK ada filter vegetasi/bayangan
        - Tambah kolom 'class' = 'Jalan'
        """
        import geopandas as gpd
        import rasterio

        log = log_callback or (lambda x: None)

        log("=== POST-PROCESSING JALAN ===")

        # 1. Merge tile masks → polygons
        log("Menggabungkan mask tile jalan...")
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
            log("Tidak ada poligon jalan yang berhasil diekstrak.")
            with rasterio.open(original_raster_path) as src:
                return gpd.GeoDataFrame(geometry=[], crs=src.crs)

        merged = gpd.pd.concat(all_gdfs, ignore_index=True)
        gdf = gpd.GeoDataFrame(merged, geometry="geometry", crs=all_gdfs[0].crs)
        log(f"Total poligon jalan awal: {len(gdf)}")

        # 2. Deduplikasi overlap antar tile
        gdf = self._deduplicate(gdf, log)

        # 3. Filter area
        gdf = self._filter_area(gdf, min_area_m2, max_area_m2, log)

        # 4. Filter aspek rasio — jalan harus memanjang (>= 1.5)
        gdf = self._filter_aspect_ratio(gdf, min_ratio=1.5, log_callback=log)

        # 5. Tambahkan kolom class
        gdf["class"] = self.OBJECT_CLASS

        log(f"=== POST-PROCESSING JALAN SELESAI: {len(gdf)} poligon ===")
        return gdf.reset_index(drop=True)

    # ──────────────────────────────────────────────────────────────
    # Helper Methods
    # ──────────────────────────────────────────────────────────────

    def _mask_to_polygons(self, mask_path: str) -> "geopandas.GeoDataFrame":
        """Konversi raster mask ke vector polygon."""
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
        """Hapus duplikat dari overlap tile (IoU > 0.5)."""
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
        result = gdf[[v for v in keep]].copy() if False else gdf.iloc[[k for k, v in enumerate(keep) if v]].copy()
        log(f"Poligon jalan setelah deduplication: {len(result)}")
        return result.reset_index(drop=True)

    def _filter_area(self, gdf, min_area_m2, max_area_m2, log):
        """Filter berdasarkan luas dalam m²."""
        if len(gdf) == 0:
            return gdf
        if gdf.crs and gdf.crs.is_geographic:
            utm_crs = gdf.estimate_utm_crs()
            gdf_metric = gdf.to_crs(utm_crs)
        else:
            gdf_metric = gdf.copy()
        areas = gdf_metric.geometry.area
        mask = (areas >= min_area_m2) & (areas <= max_area_m2)
        result = gdf[mask].copy()
        log(f"Filter area jalan ({min_area_m2}–{max_area_m2} m²): {len(gdf)} -> {len(result)}")
        return result.reset_index(drop=True)

    def _filter_aspect_ratio(self, gdf, min_ratio: float = 1.5, log_callback=None):
        """
        Filter jalan berdasarkan aspek rasio MINIMUM.
        Berbeda dari filter bangunan yang pakai maksimum — jalan justru harus memanjang.
        """
        log = log_callback or (lambda x: None)
        if len(gdf) == 0:
            return gdf

        def get_aspect(geom):
            try:
                mbr = geom.minimum_rotated_rectangle
                if mbr is None or mbr.is_empty:
                    return 1.0
                coords = list(mbr.exterior.coords)
                edges = []
                for i in range(len(coords) - 1):
                    dx = coords[i+1][0] - coords[i][0]
                    dy = coords[i+1][1] - coords[i][1]
                    edges.append((dx**2 + dy**2) ** 0.5)
                if len(edges) < 2:
                    return 1.0
                s1, s2 = edges[0], edges[1]
                return max(s1, s2) / max(min(s1, s2), 1e-9)
            except Exception:
                return 1.0

        ratios = gdf.geometry.apply(get_aspect)
        # Jalan: aspek rasio >= min_ratio ATAU area yang sangat besar (persimpangan)
        if gdf.crs and gdf.crs.is_geographic:
            utm_crs = gdf.estimate_utm_crs()
            gdf_metric = gdf.to_crs(utm_crs)
        else:
            gdf_metric = gdf.copy()
        large_area = gdf_metric.geometry.area >= 500.0   # persimpangan >= 500 m²

        mask = (ratios >= min_ratio) | large_area
        result = gdf[mask].copy()
        log(f"Filter aspek rasio jalan (>={min_ratio} atau area>=500m²): {len(gdf)} -> {len(result)}")
        return result.reset_index(drop=True)

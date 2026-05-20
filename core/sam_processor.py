"""
sam_processor.py
SAM-Geo processing wrapper. Handles model loading, tile-by-tile inference,
and mask collection with thread-safe cancellation support.
Optimized for NVIDIA GPU (CUDA) with CPU fallback.
"""

import os
import gc
import threading
import numpy as np
from pathlib import Path
from typing import Callable, List, Optional, Tuple


class SAMProcessor:
    """
    Thread-safe SAM-Geo processor for automatic building segmentation.
    Supports GPU (CUDA) and CPU inference.
    Provides progress callbacks and cancellation support.
    """

    MODEL_CONFIGS = {
        "MobileSAM (Cepat, ~40MB)": {
            "type": "mobile_sam",
            "checkpoint": "mobile_sam.pt",
            "download_url": "https://github.com/ChaoningZhang/MobileSAM/raw/master/weights/mobile_sam.pt",
        },
        "SAM-B (Seimbang, ~375MB)": {
            "type": "vit_b",
            "checkpoint": "sam_vit_b_01ec64.pth",
            "download_url": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth",
        },
        "SAM-B (Custom - Bangunan Lokal)": {
            "type": "vit_b",
            "checkpoint": "sam_bangunan_lokal.pth",
            "download_url": None,
        },
        "SAM-L (Akurat, ~1.2GB)": {
            "type": "vit_l",
            "checkpoint": "sam_vit_l_0b3195.pth",
            "download_url": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth",
        },
        "SAM-H (Sangat Akurat, ~2.4GB)": {
            "type": "vit_h",
            "checkpoint": "sam_vit_h_4b8939.pth",
            "download_url": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
        },
        # ── SAM2 (Meta, generasi terbaru) ──
        "SAM2-Tiny (Cepat, ~155MB)": {
            "type": "sam2",
            "model_id": "sam2-hiera-tiny",
            "checkpoint": None,  # diunduh otomatis oleh SamGeo2
        },
        "SAM2-Small (Seimbang, ~185MB)": {
            "type": "sam2",
            "model_id": "sam2-hiera-small",
            "checkpoint": None,
        },
        "SAM2-Base+ (Akurat, ~325MB)": {
            "type": "sam2",
            "model_id": "sam2-hiera-base-plus",
            "checkpoint": None,
        },
        "SAM2-Large (Sangat Akurat, ~898MB)": {
            "type": "sam2",
            "model_id": "sam2-hiera-large",
            "checkpoint": None,
        },
    }

    def __init__(
        self,
        model_name: str = "SAM-B (Seimbang, ~375MB)",
        models_dir: str = "models",
        device: str = "auto",
        points_per_side: int = 48,
        mode: str = "Otomatis (Grid Buta)",
        yolo_model_name: str = "yolov8n.pt",
        log_callback: Optional[Callable[[str], None]] = None,
        progress_callback: Optional[Callable[[int, str], None]] = None,
    ):
        self.model_name = model_name
        self.models_dir = os.path.abspath(models_dir)
        self.points_per_side = points_per_side
        self.yolo_model_name = yolo_model_name
        
        if "YOLO" in mode:
            self.mode = "yolo"
        elif "Pra-Deteksi" in mode:
            self.mode = "prompt"
        else:
            self.mode = "automatic"
            
        self.log_callback = log_callback or (lambda msg: print(msg))
        self.progress_callback = progress_callback or (lambda pct, msg: None)
        self._cancel_event = threading.Event()
        self._sam = None
        self._yolo = None
        self._is_sam2 = False  # flag untuk membedakan SAM1 vs SAM2

        # Determine device
        if device == "auto":
            try:
                import torch
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                self.device = "cpu"
        else:
            self.device = device

        os.makedirs(self.models_dir, exist_ok=True)

    @property
    def config(self) -> dict:
        return self.MODEL_CONFIGS.get(self.model_name, self.MODEL_CONFIGS["MobileSAM (Cepat, ~40MB)"])

    def cancel(self):
        """Request cancellation of ongoing processing."""
        self._cancel_event.set()

    def is_cancelled(self) -> bool:
        return self._cancel_event.is_set()

    def reset_cancel(self):
        self._cancel_event.clear()

    def _log(self, msg: str):
        self.log_callback(f"[SAM] {msg}")

    def _progress(self, pct: int, msg: str):
        self.progress_callback(pct, msg)

    def _ensure_model_downloaded(self) -> str:
        """Download model checkpoint if not already present. Returns local path.
        Note: SAM2 models are auto-downloaded by SamGeo2, so this is only called for SAM1.
        """
        import urllib.request

        cfg = self.config

        # SAM2: tidak perlu download manual (SamGeo2 mengurus sendiri via HuggingFace)
        if cfg.get("type") == "sam2":
            return ""

        model_path = os.path.join(self.models_dir, cfg["checkpoint"])

        if os.path.isfile(model_path):
            self._log(f"Model sudah ada: {cfg['checkpoint']}")
            return model_path

        # Jika file tidak ada dan tidak ada URL unduhan (model kustom), berikan error yang jelas
        if not cfg.get("download_url"):
            raise RuntimeError(
                f"Model kustom '{cfg['checkpoint']}' tidak ditemukan di folder 'models'!\n"
                "Silakan lakukan training model terlebih dahulu melalui panel di sidebar."
            )

        self._log(f"Mengunduh model {self.model_name}...")
        self._log(f"URL: {cfg['download_url']}")
        self._progress(2, f"Mengunduh model: {cfg['checkpoint']}")

        def _reporthook(block_num, block_size, total_size):
            if total_size > 0:
                pct = min(int(block_num * block_size / total_size * 100), 100)
                self._progress(2 + int(pct * 0.18), f"Mengunduh model: {pct}%")

        try:
            urllib.request.urlretrieve(cfg["download_url"], model_path, reporthook=_reporthook)
        except Exception as e:
            if os.path.isfile(model_path):
                os.remove(model_path)
            raise RuntimeError(f"Gagal mengunduh model: {e}")

        self._log(f"Model berhasil diunduh: {model_path}")
        return model_path

    def load_model(self):
        """Load SAM/SAM2 and optionally YOLO model into memory."""
        cfg = self.config
        model_type = cfg.get("type")

        # ── YOLO branch ──────────────────────────────────────────────
        if self.mode == "yolo":
            try:
                from ultralytics import YOLO
                # Extract actual model filename
                exact_yolo = self.yolo_model_name.split()[0]
                if exact_yolo == "yolo_bangunan_lokal.pt":
                    yolo_path = os.path.join(self.models_dir, exact_yolo)
                else:
                    yolo_path = exact_yolo
                self._log(f"Memuat model YOLO: {yolo_path}")
                self._yolo = YOLO(yolo_path)
            except Exception as e:
                self._log(f"⚠️ Gagal memuat YOLO, dialihkan ke grid mode: {e}", "warning")
                self.mode = "automatic"

        # ── SAM2 branch ──────────────────────────────────────────────
        if model_type == "sam2":
            self._load_sam2(cfg)
            return

        # ── SAM1 branch (vit_b, vit_l, vit_h, mobile_sam) ────────────
        model_path = self._ensure_model_downloaded()

        self._log(f"Memuat model {self.model_name} ke {self.device.upper()}...")
        self._progress(20, f"Memuat model ke {self.device.upper()}...")

        automatic_bool = (self.mode == "automatic")

        try:
            from samgeo import SamGeo

            # Fallback for mobile_sam which is deprecated in segment-geospatial
            if model_type == "mobile_sam":
                self._log("⚠️ MobileSAM tidak didukung. Dialihkan ke SAM-B...")
                model_type = "vit_b"
                model_path = os.path.join(self.models_dir, "sam_vit_b_01ec64.pth")
                if not os.path.exists(model_path):
                    import urllib.request
                    urllib.request.urlretrieve(
                        "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth",
                        model_path
                    )

            self._sam = SamGeo(
                model_type=model_type,
                checkpoint=model_path,
                device=self.device,
                automatic=automatic_bool,
                sam_kwargs={
                    "points_per_side": self.points_per_side,
                    "pred_iou_thresh": 0.72,
                    "stability_score_thresh": 0.75,
                    "crop_n_layers": 0,
                    "min_mask_region_area": 100,
                } if automatic_bool else None,
            )
            self._log(f"Model berhasil dimuat di {self.device.upper()}")
        except ImportError as e:
            raise RuntimeError(
                f"segment-geospatial belum terinstal: {e}\n"
                "Jalankan: pip install segment-geospatial"
            )
        except Exception as e:
            raise RuntimeError(f"Gagal memuat model SAM: {e}")

    def _load_sam2(self, cfg: dict):
        """Load SAM2 model via SamGeo2."""
        model_id = cfg["model_id"]
        self._log(f"Memuat {self.model_name} ke {self.device.upper()} (SAM2)...")
        self._progress(20, f"Memuat SAM2: {model_id}...")
        
        automatic_bool = (self.mode == "automatic")
        
        try:
            from samgeo import SamGeo2
            self._sam = SamGeo2(
                model_id=model_id,
                device=self.device,
                automatic=automatic_bool,
                points_per_side=self.points_per_side if automatic_bool else None,
                pred_iou_thresh=0.82 if automatic_bool else None,
                stability_score_thresh=0.88 if automatic_bool else None,
                min_mask_region_area=100 if automatic_bool else None,
                crop_n_layers=0 if automatic_bool else None,
            )
            self._is_sam2 = True
            self._log(f"SAM2 ({model_id}) berhasil dimuat di {self.device.upper()}")
        except Exception as e:
            raise RuntimeError(f"Gagal memuat model SAM2: {e}")

    def detect_building_points(
        self,
        image_path: str,
        min_area_px: float = 100.0,
        max_area_px: float = 250000.0,
    ) -> Tuple[List[List[int]], List[int]]:
        """
        Detect candidate building centroid coordinates automatically using OpenCV.
        Uses HSV color thresholding (for red/orange terracotta tiles & bright metal roofs)
        and masks out vegetation (using HSV green range & ExG index).
        Returns:
            (point_coords, point_labels) where:
                point_coords: List of [x, y] coordinates
                point_labels: List of 1s (indicating foreground points)
        """
        import cv2
        import numpy as np

        img = cv2.imread(image_path)
        if img is None:
            return [], []

        # Convert to HSV for robust color filtering (cv2.imread returns BGR!)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        
        # 1. Vegetation Mask (Green):
        # Hue for green is typically between 35 and 85 in BGR-to-HSV
        lower_green = np.array([35, 30, 30])
        upper_green = np.array([85, 255, 255])
        green_mask = cv2.inRange(hsv, lower_green, upper_green)
        
        # Also calculate Excess Green Index (ExG = 2*G - R - B) for more accurate vegetation detection
        # Since cv2.imread returns BGR, we split it as b, g, r!
        b, g, r = cv2.split(img.astype(float))
        exg = 2.0 * g - r - b
        exg_mask = (exg > 15).astype(np.uint8) * 255
        
        # Combine green masks
        veg_mask = cv2.bitwise_or(green_mask, exg_mask)
        
        # 2. Extract potential building roof colors:
        # A. Red/Orange/Brown tiles (genteng):
        # Hue: 0-20 and 160-180 for Red in BGR-to-HSV
        lower_red1 = np.array([0, 30, 40])
        upper_red1 = np.array([9, 255, 255])
        lower_red2 = np.array([170, 30, 40])
        upper_red2 = np.array([180, 255, 255])
        red_mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        red_mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        red_mask = cv2.bitwise_or(red_mask1, red_mask2)
        
        # B. Bright roofs (metal/grey/white):
        # High value, low saturation
        lower_bright = np.array([0, 0, 200])
        upper_bright = np.array([180, 60, 255])
        bright_mask = cv2.inRange(hsv, lower_bright, upper_bright)
        
        # Remove vegetation from individual candidate masks
        red_clean = cv2.bitwise_and(red_mask, cv2.bitwise_not(veg_mask))
        bright_clean = cv2.bitwise_and(bright_mask, cv2.bitwise_not(veg_mask))
        
        # Apply morphological operations
        kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
        
        # Process masks separately
        red_processed = cv2.morphologyEx(red_clean, cv2.MORPH_OPEN, kernel_open)
        red_processed = cv2.morphologyEx(red_processed, cv2.MORPH_CLOSE, kernel_close)
        
        bright_processed = cv2.morphologyEx(bright_clean, cv2.MORPH_OPEN, kernel_open)
        bright_processed = cv2.morphologyEx(bright_processed, cv2.MORPH_CLOSE, kernel_close)
        
        # Find contours separately
        contours_red, _ = cv2.findContours(red_processed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours_bright, _ = cv2.findContours(bright_processed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        coords = []
        max_allowed_px = min(max_area_px, 12500.0)
        relaxed_min_px = min(min_area_px, 200.0)
        
        # Helper to extract points from a list of contours
        def extract_points_from_contours(contours_list):
            for cnt in contours_list:
                area = cv2.contourArea(cnt)
                if relaxed_min_px <= area <= max_allowed_px:
                    x, y, w, h = cv2.boundingRect(cnt)
                    aspect_ratio = float(w) / h if h > 0 else 0
                    if 0.2 <= aspect_ratio <= 5.0:
                        rect = cv2.minAreaRect(cnt)
                        rect_area = rect[1][0] * rect[1][1]
                        if rect_area > 0:
                            solidity = area / rect_area
                            if solidity > 0.40:
                                M = cv2.moments(cnt)
                                if M["m00"] != 0:
                                    cx = int(M["m10"] / M["m00"])
                                    cy = int(M["m01"] / M["m00"])
                                    if 0 <= cx < img.shape[1] and 0 <= cy < img.shape[0]:
                                        if veg_mask[cy, cx] == 0:
                                            # Avoid duplicate coordinates
                                            if [cx, cy] not in coords:
                                                coords.append([cx, cy])
                                                
        # Run extraction on both lists of contours
        extract_points_from_contours(contours_red)
        extract_points_from_contours(contours_bright)

        # 3. Fallback: If we found no points, let's relax criteria for grey/darker roofs
        if len(coords) == 0:
            # We look at general non-vegetation, non-shadow, non-green areas with strict compactness
            non_veg = cv2.bitwise_not(veg_mask)
            shadow_mask = cv2.inRange(hsv, np.array([0, 0, 0]), np.array([180, 255, 45]))
            non_veg_no_shadow = cv2.bitwise_and(non_veg, cv2.bitwise_not(shadow_mask))
            
            processed_fallback = cv2.morphologyEx(non_veg_no_shadow, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)))
            processed_fallback = cv2.morphologyEx(processed_fallback, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (11, 11)))
            contours, _ = cv2.findContours(processed_fallback, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for cnt in contours:
                area = cv2.contourArea(cnt)
                # Keep fallback sizes strictly smaller (e.g. max 350 m2 / 8,000 pixels) to avoid agricultural fields
                if min_area_px <= area <= min(max_area_px, 8000.0):
                    x, y, w, h = cv2.boundingRect(cnt)
                    aspect_ratio = float(w) / h if h > 0 else 0
                    if 0.25 <= aspect_ratio <= 4.0:
                        rect = cv2.minAreaRect(cnt)
                        rect_area = rect[1][0] * rect[1][1]
                        if rect_area > 0:
                            solidity = area / rect_area
                            if solidity > 0.50:  # Strict compactness
                                M = cv2.moments(cnt)
                                if M["m00"] != 0:
                                    cx = int(M["m10"] / M["m00"])
                                    cy = int(M["m01"] / M["m00"])
                                    if 0 <= cx < img.shape[1] and 0 <= cy < img.shape[0]:
                                        if veg_mask[cy, cx] == 0:
                                            coords.append([cx, cy])

        if not coords:
            return [], []

        labels = [1] * len(coords)
        return coords, labels

    def process_tile(
        self,
        tile_path: str,
        output_mask_path: str,
        point_coords: Optional[List[List[int]]] = None,
        point_labels: Optional[List[int]] = None,
        box_prompts: Optional[List[List[int]]] = None,
    ) -> bool:
        """
        Run SAM automatic or prompt-based segmentation on a single tile.
        In prompt mode, segments each point individually to avoid merging buildings.
        Returns True if successful, False if cancelled.
        """
        if self.is_cancelled():
            return False
        if self._sam is None:
            raise RuntimeError("Model belum dimuat. Panggil load_model() terlebih dahulu.")

        try:
            if (self.mode == "prompt" and point_coords) or (self.mode == "yolo" and box_prompts):
                import rasterio
                
                # 1. Set the image once to encode it (heavy operation done only once)
                self._sam.set_image(tile_path)
                
                # 2. Get image dimensions
                with rasterio.open(tile_path) as src:
                    h, w = src.height, src.width
                    meta = src.meta.copy()
                    meta.update(dtype="uint16", count=1, nodata=0)
                
                # Initialize master mask
                master_mask = np.zeros((h, w), dtype=np.uint16)
                
                # Temp path for individual point/box prediction
                temp_dir = os.path.dirname(output_mask_path)
                temp_point_path = os.path.join(temp_dir, f"temp_predict_{os.path.basename(tile_path)}")
                
                # Loop through each prompt separately to prevent SAM from merging them
                items_to_loop = box_prompts if self.mode == "yolo" else list(zip(point_coords, point_labels))
                
                for idx, item in enumerate(items_to_loop):
                    if self.is_cancelled():
                        return False
                    
                    try:
                        if os.path.exists(temp_point_path):
                            os.remove(temp_point_path)
                            
                        # Predict single building
                        if self.mode == "yolo":
                            self._sam.predict(
                                boxes=item,
                                output=temp_point_path,
                            )
                        else:
                            coord, label = item
                            self._sam.predict(
                                point_coords=[coord],
                                point_labels=[label],
                                output=temp_point_path,
                            )
                        
                        # Read individual binary mask
                        if os.path.exists(temp_point_path):
                            with rasterio.open(temp_point_path) as p_src:
                                point_mask = p_src.read(1)
                                # Assign a unique ID to this building's pixels to keep them separate in vectorization
                                building_id = idx + 1
                                master_mask = np.where(point_mask > 0, building_id, master_mask)
                    except Exception as pe:
                        self._log(f"      ⚠️ Gagal segmentasi item {item}: {pe}", "warning")
                
                # Cleanup temp file
                if os.path.exists(temp_point_path):
                    try:
                        os.remove(temp_point_path)
                    except Exception:
                        pass
                
                # Write final combined master mask to disk
                with rasterio.open(output_mask_path, "w", **meta) as dst:
                    dst.write(master_mask.astype(np.uint16), 1)
                    
            else:
                try:
                    self._sam.generate(
                        source=tile_path,
                        output=output_mask_path,
                        foreground=True,
                        unique=False,
                    )
                except TypeError:
                    self._sam.generate(
                        source=tile_path,
                        output=output_mask_path,
                        unique=False,
                    )
            return True
        except Exception as e:
            self._log(f"Error pada tile {Path(tile_path).name}: {e}")
            return False

    def process_raster(
        self,
        raster_path: str,
        masks_dir: str,
        tile_size: int = 1024,
        overlap: int = 128,
        enable_filtering: bool = False,
        min_area_m2: float = 20.0,
        max_area_m2: float = 5000.0,
        tile_progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> List[Tuple[str, dict]]:
        """
        Full pipeline: tile the raster and process each tile with SAM.

        Returns:
            Tuple containing:
            - List of (mask_tiff_path, tile_meta) for all successfully processed tiles
            - List of global YOLO bounding boxes (if yolo mode is active)
        """
        from core.tiling import tiles_generator

        os.makedirs(masks_dir, exist_ok=True)
        results = []
        global_yolo_boxes = []

        self._log(f"Memulai proses tiling raster: {Path(raster_path).name}")

        # Place tiles in dedicated temp/tiles folder under project root (not beside raster)
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        tiles_temp_dir = os.path.join(project_root, "temp", "tiles")

        for tile_path, tile_meta, idx, total in tiles_generator(
            raster_path=raster_path,
            tile_size=tile_size,
            overlap=overlap,
            temp_dir=tiles_temp_dir,
            enable_filtering=enable_filtering,
        ):
            if self.is_cancelled():
                self._log("Proses dibatalkan oleh pengguna.")
                break

            tile_name = Path(tile_path).stem
            mask_path = os.path.join(masks_dir, f"mask_{tile_name}.tif")

            self._log(f"[{idx+1}/{total}] Memproses tile: {tile_name}")
            if tile_progress_callback:
                tile_progress_callback(idx + 1, total)

            # Selalu proses ulang — hapus mask lama agar parameter baru berlaku
            if os.path.isfile(mask_path):
                os.remove(mask_path)

            point_coords, point_labels = None, None
            box_prompts = None
            
            if self.mode == "yolo" and self._yolo:
                import cv2
                img = cv2.imread(tile_path)
                # Run YOLO inference
                yolo_results = self._yolo(img, verbose=False)
                boxes = yolo_results[0].boxes.xyxy.cpu().numpy() # [x1, y1, x2, y2]
                if len(boxes) > 0:
                    box_prompts = boxes.tolist()
                    self._log(f"  -> YOLO menemukan {len(box_prompts)} bangunan")
                    # Convert to global coordinates for preview
                    transform = tile_meta["transform"]
                    for box in box_prompts:
                        x1, y1, x2, y2 = box
                        geo_x1, geo_y1 = transform * (x1, y1)
                        geo_x2, geo_y2 = transform * (x2, y2)
                        global_yolo_boxes.append([min(geo_x1,geo_x2), min(geo_y1,geo_y2), max(geo_x1,geo_x2), max(geo_y1,geo_y2)])
                else:
                    self._log("  -> Dilewati (YOLO tidak menemukan bangunan)")
                    continue
            elif self.mode == "prompt":
                # Compute resolution from tile metadata transform
                transform = tile_meta["transform"]
                res_x = abs(transform.a)
                res_y = abs(transform.e)
                pixel_area_m2 = res_x * res_y

                min_area_px = min_area_m2 / pixel_area_m2 if pixel_area_m2 > 0 else 100.0
                max_area_px = max_area_m2 / pixel_area_m2 if pixel_area_m2 > 0 else 25000.0

                point_coords, point_labels = self.detect_building_points(
                    tile_path, min_area_px, max_area_px
                )
                self._log(f"  -> Pra-deteksi menemukan {len(point_coords)} calon bangunan")
                if not point_coords:
                    self._log("  -> Dilewati (tidak ada bangunan terdeteksi)")
                    continue

            success = self.process_tile(
                tile_path, mask_path,
                point_coords=point_coords,
                point_labels=point_labels,
                box_prompts=box_prompts
            )
            if success and os.path.isfile(mask_path):
                results.append((mask_path, tile_meta))
                self._log(f"  -> Berhasil: {Path(mask_path).name}")
            else:
                self._log(f"  -> Gagal atau dibatalkan")

        # Free GPU memory
        if self.device == "cuda":
            try:
                import torch
                torch.cuda.empty_cache()
            except ImportError:
                pass
        gc.collect()

        return results, global_yolo_boxes

    def unload_model(self):
        """Free model from memory."""
        if self._sam is not None:
            del self._sam
            self._sam = None
            gc.collect()
            if self.device == "cuda":
                try:
                    import torch
                    torch.cuda.empty_cache()
                except Exception:
                    pass
            self._log("Model berhasil diunload dari memori")

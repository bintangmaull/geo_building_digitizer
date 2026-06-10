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
        yolo_conf: float = 0.15,
        gemini_api_key: str = "",
        log_callback: Optional[Callable[[str], None]] = None,
        progress_callback: Optional[Callable[[int, str], None]] = None,
    ):
        self.model_name = model_name
        self.models_dir = os.path.abspath(models_dir)
        self.points_per_side = points_per_side
        self.yolo_model_name = yolo_model_name
        self.yolo_conf = yolo_conf
        self.gemini_api_key = gemini_api_key
        
        if "Full Eksperimental" in mode:
            self.mode = "gemini"
        elif "Gemini Bounding Box" in mode:
            self.mode = "gemini_bbox"
        elif "YOLO" in mode:
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
        """Load the required models based on current mode."""
        if self.mode == "gemini":
            from core.objects.gemini_processor import GeminiProcessor
            self._sam = GeminiProcessor(api_key=self.gemini_api_key, log_callback=self._log)
            if not self._sam.is_ready():
                raise RuntimeError("Gemini API gagal dimuat. Periksa API Key dan koneksi internet.")
            self._progress(30, "Gemini API siap")
            return

        if self.mode == "gemini_bbox":
            from core.objects.gemini_processor import GeminiProcessor
            self._gemini = GeminiProcessor(api_key=self.gemini_api_key, log_callback=self._log)
            if not self._gemini.is_ready():
                raise RuntimeError("Gemini API gagal dimuat. Periksa API Key dan koneksi internet.")
            self._progress(5, "Gemini API siap")
            # Tetap lanjut memuat SAM di bawah karena mode ini butuh SAM juga

        if self.is_cancelled():
            return

        cfg = self.config
        model_type = cfg.get("type")

        # ── YOLO branch ──────────────────────────────────────────────
        if self.mode == "yolo":
            try:
                from ultralytics import YOLO
                # Extract actual model filename
                exact_yolo = self.yolo_model_name.split()[0]
                possible_path = os.path.join(self.models_dir, exact_yolo)
                if os.path.exists(possible_path):
                    yolo_path = possible_path
                else:
                    yolo_path = exact_yolo
                import ultralytics.nn.tasks as tasks
                try:
                    from core.models.uav_yolov12_modules import PConv, SKNet
                    tasks.PConv = PConv
                    tasks.SKNet = SKNet
                except ImportError:
                    pass
                self._log(f"Memuat model YOLO: {yolo_path}")
                self._yolo = YOLO(yolo_path)
            except Exception as e:
                self._log(f"⚠️ Gagal memuat YOLO, dialihkan ke grid mode: {e}")
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
        global_indices: Optional[List[int]] = None,
    ) -> bool:
        """
        Run SAM prediction on a single tile.
        If box_prompts are provided, it predicts using boxes.
        If global_indices is provided, it assigns those values to the instance mask pixels.
        """
        if self.is_cancelled():
            return False
        if self._sam is None:
            raise RuntimeError("Model belum dimuat. Panggil load_model() terlebih dahulu.")

        try:
            if ((self.mode == "prompt" and point_coords) or 
               ((self.mode == "yolo" or self.mode == "gemini_bbox") and box_prompts is not None and len(box_prompts) > 0)):
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
                if (self.mode == "yolo" or self.mode == "gemini_bbox") and box_prompts is not None:
                    # Sort Smallest First! This ensures small buildings claim their pixels
                    # instead of being swallowed by the SAM bleed of larger neighboring buildings.
                    if global_indices is None:
                        global_indices = list(range(1, len(box_prompts) + 1))
                        
                    items_to_loop = sorted(
                        zip(box_prompts, global_indices),
                        key=lambda x: (x[0][2] - x[0][0]) * (x[0][3] - x[0][1]),
                        reverse=False
                    )
                else:
                    items_to_loop = list(zip(point_coords, point_labels)) if point_coords else []
                
                
                for idx, item in enumerate(items_to_loop):
                    if self.is_cancelled():
                        return False
                    
                    try:
                        if os.path.exists(temp_point_path):
                            os.remove(temp_point_path)
                            
                        # Predict single building
                        if self.mode == "yolo" or self.mode == "gemini_bbox":
                            box_coord, g_idx = item
                            self._sam.predict(
                                boxes=box_coord.tolist() if isinstance(box_coord, np.ndarray) else box_coord,
                                output=temp_point_path
                            )
                        else:
                            self._sam.predict(
                                point_coords=item[0],
                                point_labels=item[1],
                                output=temp_point_path
                            )
                        
                        # Read individual binary mask
                        if os.path.exists(temp_point_path):
                            with rasterio.open(temp_point_path) as temp_src:
                                point_mask = temp_src.read(1)
                                if self.mode == "yolo" or self.mode == "gemini_bbox":
                                    # Crop point_mask strictly to its bounding box
                                    # This prevents SAM from bleeding into neighboring buildings!
                                    bx1, by1, bx2, by2 = [int(round(c)) for c in box_coord]
                                    bx1, by1 = max(0, bx1), max(0, by1)
                                    bx2, by2 = min(w, bx2), min(h, by2)
                                    cropped_mask = np.zeros_like(point_mask)
                                    cropped_mask[by1:by2, bx1:bx2] = point_mask[by1:by2, bx1:bx2]
                                    point_mask = cropped_mask

                                # Assign a unique ID to this building's pixels to keep them separate in vectorization
                                building_id = g_idx if (self.mode == "yolo" or self.mode == "gemini_bbox") else (idx + 1)
                                # Overwrite where master_mask is empty (0)
                                master_mask = np.where((point_mask > 0) & (master_mask == 0), building_id, master_mask)
                    except Exception as pe:
                        self._log(f"      ⚠️ Gagal segmentasi item {item}: {pe}")
                
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
                if self.mode in ("yolo", "gemini_bbox"):
                    # Jika tidak ada kotak pembatas yang ditemukan, cukup buat mask kosong
                    import rasterio
                    with rasterio.open(tile_path) as src:
                        meta = src.meta.copy()
                        h, w = src.height, src.width
                    meta.update(dtype="uint16", count=1, nodata=0)
                    with rasterio.open(output_mask_path, "w", **meta) as dst:
                        dst.write(np.zeros((h, w), dtype=np.uint16), 1)
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

    def _convert_to_global_boxes(self, boxes, x0, y0, transform):
        """Helper to convert local boxes to global coordinates."""
        global_boxes = []
        for box in boxes:
            x1, y1, x2, y2 = box
            geo_x1, geo_y1 = transform * (x1, y1)
            geo_x2, geo_y2 = transform * (x2, y2)
            global_boxes.append([min(geo_x1,geo_x2), min(geo_y1,geo_y2), max(geo_x1,geo_x2), max(geo_y1,geo_y2)])
        return global_boxes

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

        if self.mode == "gemini":
            self._log("Fase 1: Mengeksekusi Gemini API pada semua tile...")
            for tile_path, tile_meta, idx, total in tiles_generator(
                raster_path=raster_path,
                tile_size=tile_size,
                overlap=overlap,
                temp_dir=tiles_temp_dir,
                enable_filtering=enable_filtering,
            ):
                if self.is_cancelled():
                    self._log("Proses dibatalkan oleh pengguna.")
                    return results, global_yolo_boxes
                
                self._log(f"[{idx+1}/{total}] Fase 1 - Gemini API: {Path(tile_path).stem}")
                if tile_progress_callback:
                    tile_progress_callback(idx + 1, total)
                
                mask_filename = f"mask_gemini_{Path(tile_path).stem}.tif"
                mask_path = os.path.join(masks_dir, mask_filename)
                
                success = self.process_tile(tile_path, mask_path)
                if success and os.path.exists(mask_path):
                    results.append((mask_path, tile_meta))
                    
            return results, global_yolo_boxes

        # ── FASE 1: Deteksi YOLO / Pra-pemrosesan di Semua Tile ──
        self._log("Fase 1: Mengeksekusi Pra-pemrosesan (YOLO/Gemini) pada semua tile...")
        all_yolo_detections = []
        tiles_data = []

        # We need to compute total tiles upfront to show progress properly if needed,
        # but tiles_generator yields them.
        for tile_path, tile_meta, idx, total in tiles_generator(
            raster_path=raster_path,
            tile_size=tile_size,
            overlap=overlap,
            temp_dir=tiles_temp_dir,
            enable_filtering=enable_filtering,
        ):
            if self.is_cancelled():
                self._log("Proses dibatalkan oleh pengguna.")
                return results, global_yolo_boxes

            import rasterio
            with rasterio.open(tile_path) as src:
                tile_meta = src.meta.copy()
                w, h = src.width, src.height
                transform = src.transform
                
            x0, y0 = transform * (0, 0)
            x1, y1 = transform * (w, h)
            minx, maxx = min(x0, x1), max(x0, x1)
            miny, maxy = min(y0, y1), max(y0, y1)
            
            tile_dict = {
                "tile_path": tile_path,
                "tile_meta": tile_meta,
                "bbox": [minx, miny, maxx, maxy],
                "idx": idx,
                "total": total,
                "point_coords": None,
                "point_labels": None,
                "box_prompts": None
            }

            self._log(f"[{idx+1}/{total}] Fase 1 - Pra-pemrosesan: {Path(tile_path).stem}")
            if tile_progress_callback:
                # Provide progress for phase 1
                tile_progress_callback(idx + 1, total * 2)

            if self.mode == "gemini_bbox" and hasattr(self, "_gemini"):
                boxes_np = self._gemini.detect_boxes(tile_path)
                tile_dict["box_prompts"] = boxes_np
                self._log(f"   Mendapat {len(boxes_np)} kotak bangunan dari Gemini.")

            elif self.mode == "yolo" and self._yolo:
                import cv2
                img = cv2.imread(tile_path)
                yolo_results = self._yolo(
                    img, 
                    verbose=False,
                    imgsz=max(img.shape[0], img.shape[1]),
                    conf=self.yolo_conf,
                    iou=0.6,
                    max_det=3000
                )
                boxes = yolo_results[0].boxes.xyxy.cpu().numpy()
                confidences = yolo_results[0].boxes.conf.cpu().numpy()
                
                if len(boxes) > 0:
                    self._log(f"  -> YOLO menemukan {len(boxes)} bangunan")
                    transform = tile_meta["transform"]
                    tile_idx = len(tiles_data)
                    for box, conf in zip(boxes, confidences):
                        x1, y1, x2, y2 = box
                        geo_x1, geo_y1 = transform * (x1, y1)
                        geo_x2, geo_y2 = transform * (x2, y2)
                        global_box = [min(geo_x1,geo_x2), min(geo_y1,geo_y2), max(geo_x1,geo_x2), max(geo_y1,geo_y2)]
                        all_yolo_detections.append({
                            "global_box": global_box,
                            "conf": float(conf),
                            "local_box": box.tolist(),
                            "tile_idx": tile_idx
                        })
            elif self.mode == "prompt":
                transform = tile_meta["transform"]
                res_x = abs(transform.a)
                res_y = abs(transform.e)
                pixel_area_m2 = res_x * res_y

                min_area_px = min_area_m2 / pixel_area_m2 if pixel_area_m2 > 0 else 100.0
                max_area_px = max_area_m2 / pixel_area_m2 if pixel_area_m2 > 0 else 25000.0

                point_coords, point_labels = self.detect_building_points(
                    tile_path, min_area_px, max_area_px
                )
                tile_dict["point_coords"] = point_coords
                tile_dict["point_labels"] = point_labels
                
            tiles_data.append(tile_dict)

        # ── FASE 2: Global NMS (Khusus YOLO) ──
        if self.mode == "yolo" and self._yolo:
            if len(all_yolo_detections) > 0:
                self._log(f"Fase 2: Menjalankan Global NMS pada {len(all_yolo_detections)} kotak awal...")
                try:
                    import torch
                    import torchvision
                    
                    boxes_tensor = torch.tensor([d["global_box"] for d in all_yolo_detections], dtype=torch.float32)
                    scores_tensor = torch.tensor([d["conf"] for d in all_yolo_detections], dtype=torch.float32)
                    
                    # 1. Standard NMS (increased threshold from 0.3 to 0.5 for dense buildings)
                    keep_indices = torchvision.ops.nms(boxes_tensor, scores_tensor, 0.5).tolist()
                    
                    # 2. IoM NMS (Intersection over Minimum Area)
                    # We want to keep the most confident predictions.
                    # Sort by CONFIDENCE (Highest First).
                    final_keep = []
                    kept_dets = [all_yolo_detections[i] for i in keep_indices]
                    kept_dets.sort(
                        key=lambda d: d["conf"], 
                        reverse=True
                    )
                    
                    for det in kept_dets:
                        boxA = det["global_box"]
                        areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
                        keep = True
                        for f_det in final_keep:
                            boxB = f_det["global_box"]
                            areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
                            
                            ix1 = max(boxA[0], boxB[0])
                            iy1 = max(boxA[1], boxB[1])
                            ix2 = min(boxA[2], boxB[2])
                            iy2 = min(boxA[3], boxB[3])
                            
                            w = max(0, ix2 - ix1)
                            h = max(0, iy2 - iy1)
                            inter = w * h
                            
                            min_area = min(areaA, areaB)
                            if min_area > 0 and (inter / min_area) > 0.85:
                                keep = False
                                break
                                
                        if keep:
                            final_keep.append(det)
                            
                    self._log(f"  -> Global NMS selesai. Tersisa BB unik: {len(final_keep)}")
                    
                    for global_idx, det in enumerate(final_keep):
                        box = det["global_box"]
                        global_yolo_boxes.append(box)
                        
                        # Distribusikan kotak global ke SEMUA tile yang berpotongan
                        for t_idx, t_data in enumerate(tiles_data):
                            t_box = t_data["bbox"]  # [minx, miny, maxx, maxy]
                            # Cek intersection
                            if not (box[2] < t_box[0] or box[0] > t_box[2] or box[3] < t_box[1] or box[1] > t_box[3]):
                                # Convert global box ke local tile pixels
                                t_transform = tiles_data[t_idx]["tile_meta"]["transform"]
                                inv_transform = ~t_transform
                                px1, py1 = inv_transform * (box[0], box[1])
                                px2, py2 = inv_transform * (box[2], box[3])
                                local_x1 = float(min(px1, px2))
                                local_y1 = float(min(py1, py2))
                                local_x2 = float(max(px1, px2))
                                local_y2 = float(max(py1, py2))
                                
                                if tiles_data[t_idx].get("box_prompts") is None:
                                    tiles_data[t_idx]["box_prompts"] = []
                                    tiles_data[t_idx]["global_indices"] = []
                                tiles_data[t_idx]["box_prompts"].append([local_x1, local_y1, local_x2, local_y2])
                                tiles_data[t_idx]["global_indices"].append(global_idx + 1) # Use 1-based index for mask

                except ImportError:
                    self._log("⚠️ PyTorch tidak tersedia untuk NMS. Menggunakan semua deteksi YOLO.")
                    for global_idx, det in enumerate(all_yolo_detections):
                        box = det["global_box"]
                        global_yolo_boxes.append(box)
                        for t_idx, t_data in enumerate(tiles_data):
                            t_box = t_data["bbox"]
                            if not (box[2] < t_box[0] or box[0] > t_box[2] or box[3] < t_box[1] or box[1] > t_box[3]):
                                t_transform = tiles_data[t_idx]["tile_meta"]["transform"]
                                inv_transform = ~t_transform
                                px1, py1 = inv_transform * (box[0], box[1])
                                px2, py2 = inv_transform * (box[2], box[3])
                                local_x1 = float(min(px1, px2))
                                local_y1 = float(min(py1, py2))
                                local_x2 = float(max(px1, px2))
                                local_y2 = float(max(py1, py2))
                                if tiles_data[t_idx].get("box_prompts") is None:
                                    tiles_data[t_idx]["box_prompts"] = []
                                    tiles_data[t_idx]["global_indices"] = []
                                tiles_data[t_idx]["box_prompts"].append([local_x1, local_y1, local_x2, local_y2])
                                tiles_data[t_idx]["global_indices"].append(global_idx + 1)

        # ── FASE 3: SAM Segmentasi ──
        total_tiles = len(tiles_data)
        self._log(f"Fase 3: Mengeksekusi SAM pada {total_tiles} tile...")
        
        for tile_idx, t_data in enumerate(tiles_data):
            if self.is_cancelled():
                self._log("Proses dibatalkan oleh pengguna.")
                break
                
            tile_path = t_data["tile_path"]
            tile_meta = t_data["tile_meta"]
            idx = t_data["idx"]
            total = t_data["total"]
            
            self._log(f"[{idx+1}/{total}] Fase 3 - SAM Segmentasi: {Path(tile_path).stem}")
            if tile_progress_callback:
                tile_progress_callback(total + idx + 1, total * 2)
                
            if self.mode == "yolo":
                if t_data["box_prompts"] is None or len(t_data["box_prompts"]) == 0:
                    self._log("  -> Dilewati (Tidak ada objek unik tersisa)")
                    continue
                else:
                    self._log(f"  -> SAM mengeksekusi {len(t_data['box_prompts'])} Bounding Box unik...")
            elif self.mode == "prompt":
                if not t_data["point_coords"]:
                    self._log("  -> Dilewati (Tidak ada calon bangunan)")
                    continue
                    
            tile_name = Path(tile_path).stem
            mask_path = os.path.join(masks_dir, f"mask_{tile_name}.tif")
            
            if os.path.isfile(mask_path):
                os.remove(mask_path)
                
            success = self.process_tile(
                tile_path, mask_path,
                point_coords=t_data["point_coords"],
                point_labels=t_data["point_labels"],
                box_prompts=t_data["box_prompts"],
                global_indices=t_data.get("global_indices")
            )
            
            if success and os.path.isfile(mask_path):
                results.append((mask_path, tile_meta))
                self._log(f"  -> Berhasil: {Path(mask_path).name}")

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

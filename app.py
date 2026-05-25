"""
app.py
Main application entry point for SAM-Geo Building Digitizer.
Manages the overall window layout, threading, and pipeline orchestration.

Layout:
  ┌────────────┬─────────────────────────────┐
  │            │                             │
  │  Sidebar   │      Preview Panel          │
  │ (controls) │       (map + overlay)       │
  │            │                             │
  │            ├─────────────────────────────┤
  │            │       Log Panel             │
  │            │  (real-time logs + progress)│
  └────────────┴─────────────────────────────┘
"""

import os
import sys
import threading
import traceback
import tkinter as tk
import customtkinter as ctk
from pathlib import Path
from tkinter import messagebox

# ── Theme Setup ──────────────────────────────────────────────────────────────
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

# Add project root to path
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

# ── Import UI Components ──────────────────────────────────────────────────────
from ui.sidebar import Sidebar
from ui.preview_panel import PreviewPanel
from ui.log_panel import LogPanel


class SAMGeoApp(ctk.CTk):
    """Main application window."""

    APP_TITLE = "SAM-Geo Multi-Object Digitizer"
    APP_VERSION = "1.3"
    MIN_WIDTH = 1200
    MIN_HEIGHT = 720

    def __init__(self):
        super().__init__()

        self._processing_thread: threading.Thread | None = None
        self._processor = None   # SAMProcessor instance
        self._result_gdf = None  # Final building polygons

        self._setup_window()
        self._build_layout()
        self._check_environment()

    def _setup_window(self):
        self.title(f"{self.APP_TITLE} v{self.APP_VERSION}")
        self.geometry("1400x820")
        self.minsize(self.MIN_WIDTH, self.MIN_HEIGHT)
        self.configure(fg_color="#070D1A")

        # Window icon (try to set)
        try:
            self.iconbitmap(os.path.join(ROOT, "assets", "icon.ico"))
        except Exception:
            pass

        # Handle close button
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    def _build_layout(self):
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=1)

        # ── Left: Sidebar ─────────────────────────────────────────────────────
        self.sidebar = Sidebar(
            self,
            on_run=self._on_run,
            on_stop=self._on_stop,
            on_train=self._on_train,
            on_train_yolo=self._on_train_yolo,
            on_wms_ready=self._on_wms_ready,
            on_yolo_toggle=self._on_yolo_toggle,
            on_build_fingerprint=self._on_build_fingerprint,
            width=240,
            fg_color="#0A1628",
            corner_radius=0,
        )
        self.sidebar.grid(row=0, column=0, sticky="nsew")

        # ── Right: Vertical split (Preview + Log) ─────────────────────────────
        right_frame = ctk.CTkFrame(self, fg_color="#070D1A", corner_radius=0)
        right_frame.grid(row=0, column=1, sticky="nsew")
        right_frame.grid_rowconfigure(0, weight=3)
        right_frame.grid_rowconfigure(1, weight=1)
        right_frame.grid_columnconfigure(0, weight=1)

        # Preview Panel (top-right)
        self.preview = PreviewPanel(
            right_frame,
            on_maximize_toggle=self._on_preview_maximize,
            fg_color="#0A0F1E",
            corner_radius=0,
        )
        self.preview.grid(row=0, column=0, sticky="nsew", padx=(1, 0), pady=(0, 1))

        # Log Panel (bottom-right)
        self.log_panel = LogPanel(
            right_frame,
            on_minimize_toggle=self._on_log_minimize,
            fg_color="#0A1628",
            corner_radius=0,
        )
        self.log_panel.grid(row=1, column=0, sticky="nsew", padx=(1, 0))

    def _on_log_minimize(self, is_minimized):
        """Callback to resize the right split frame when the log panel is minimized."""
        right_frame = self.log_panel.master
        if is_minimized:
            # When minimized, set the log panel row weight to 0 and set its height to exactly the header size
            right_frame.grid_rowconfigure(0, weight=1)
            right_frame.grid_rowconfigure(1, weight=0, minsize=40)
        else:
            # When restored, set the weights back to original 3:1 split
            right_frame.grid_rowconfigure(0, weight=3)
            right_frame.grid_rowconfigure(1, weight=1, minsize=0)

    def _on_preview_maximize(self, is_maximized):
        """Callback to expand the preview panel to full screen, hiding other components."""
        right_frame = self.preview.master
        if is_maximized:
            # 1. Hide the sidebar and the log panel
            self.sidebar.grid_remove()
            self.log_panel.grid_remove()
            
            # 2. Make right_frame span the ENTIRE window columns (0 and 1)
            right_frame.grid(row=0, column=0, columnspan=2, sticky="nsew")
            
            # 3. Make preview panel fill the entire right_frame height
            right_frame.grid_rowconfigure(0, weight=1)
            right_frame.grid_rowconfigure(1, weight=0, minsize=0)
        else:
            # 1. Restore right_frame to column 1 only
            right_frame.grid(row=0, column=1, sticky="nsew")
            
            # 2. Restore sidebar to column 0
            self.sidebar.grid(row=0, column=0, sticky="nsew")
            
            # 3. Restore log panel
            self.log_panel.grid(row=1, column=0, sticky="nsew", padx=(1, 0))
            
            # 4. Restore row configuring weights in right_frame
            if self.log_panel.is_minimized:
                right_frame.grid_rowconfigure(0, weight=1)
                right_frame.grid_rowconfigure(1, weight=0, minsize=40)
            else:
                right_frame.grid_rowconfigure(0, weight=3)
                right_frame.grid_rowconfigure(1, weight=1, minsize=0)

    def _check_environment(self):
        """Verify required packages and GPU availability at startup."""
        self.log_panel.log("=== SAM-Geo Multi-Object Digitizer v1.3 ===", "system")
        self.log_panel.log("Memeriksa environment...", "system")

        missing = []

        # Check PyTorch
        try:
            import torch
            cuda_ok = torch.cuda.is_available()
            gpu_name = torch.cuda.get_device_name(0) if cuda_ok else "Tidak ada"
            self.log_panel.log(
                f"✅ PyTorch {torch.__version__} | "
                f"{'GPU: ' + gpu_name if cuda_ok else 'CPU only'}",
                "success" if cuda_ok else "warning"
            )
        except ImportError:
            missing.append("torch")
            self.log_panel.log("❌ PyTorch belum terinstal", "error")

        # Check rasterio
        try:
            import rasterio
            self.log_panel.log(f"✅ rasterio {rasterio.__version__}", "success")
        except ImportError:
            missing.append("rasterio")
            self.log_panel.log("❌ rasterio belum terinstal", "error")

        # Check geopandas
        try:
            import geopandas
            self.log_panel.log(f"✅ geopandas {geopandas.__version__}", "success")
        except ImportError:
            missing.append("geopandas")
            self.log_panel.log("❌ geopandas belum terinstal", "error")

        # Check samgeo
        try:
            import samgeo
            self.log_panel.log(f"✅ segment-geospatial (samgeo)", "success")
        except ImportError:
            missing.append("segment-geospatial")
            self.log_panel.log("❌ segment-geospatial belum terinstal", "error")

        # Check GDAL for ECW
        from core.ecw_handler import find_gdal_translate
        gdal_path = find_gdal_translate()
        if gdal_path:
            self.log_panel.log(f"✅ GDAL: {gdal_path}", "success")
            from core.ecw_handler import check_ecw_support
            has_ecw = check_ecw_support(gdal_path)
            if has_ecw:
                self.log_panel.log("✅ ECW driver: TERSEDIA", "success")
            else:
                self.log_panel.log(
                    "⚠️ ECW driver: TIDAK TERSEDIA — hanya GeoTIFF yang didukung",
                    "warning"
                )
        else:
            self.log_panel.log(
                "⚠️ GDAL tidak ditemukan — hanya GeoTIFF yang didukung",
                "warning"
            )

        if missing:
            self.log_panel.log(
                f"\n⚠️ Library belum terinstal: {', '.join(missing)}\n"
                f"Jalankan: pip install {' '.join(missing)}",
                "warning"
            )
        else:
            self.log_panel.log("\n✅ Semua library siap! Pilih file raster untuk memulai.", "success")

    def _on_wms_ready(self, geotiff_path: str):
        """Called when WMS AOI download completes — preview the result GeoTIFF."""
        self.log_panel.log(f"🌐 Citra WMS siap: {geotiff_path}", "success")
        self.preview.load_raster_preview(geotiff_path)

    def _on_yolo_toggle(self, show: bool):
        """Called when YOLO preview toggle is clicked in the sidebar."""
        if hasattr(self, "preview") and self.preview:
            self.preview.toggle_yolo_boxes(show)

    def _log(self, message: str, level: str = "info"):
        """Thread-safe log callback."""
        self.after(0, lambda: self.log_panel.log(message, level))

    def _progress(self, percent: int, message: str = ""):
        """Thread-safe progress callback."""
        self.after(0, lambda: self.log_panel.set_overall_progress(percent, message))

    def _tile_progress(self, current: int, total: int):
        """Thread-safe tile progress callback."""
        self.after(0, lambda: self.log_panel.set_tile_progress(current, total))
        self.after(0, lambda: self.preview.set_active_tile(current - 1))

    def _on_run(self, params: dict):
        """Called when user clicks Run.
        Routes to building-only pipeline OR multi-object pipeline depending on selection.
        """
        if self._processing_thread and self._processing_thread.is_alive():
            return

        # Validasi: minimal satu objek harus dipilih
        enabled = params.get("enabled_objects", {"building": True})
        if not any(enabled.values()):
            from tkinter import messagebox
            messagebox.showwarning(
                "Tidak Ada Objek Dipilih",
                "Pilih minimal satu objek digitasi\n(Bangunan, Jalan, Badan Air, atau Vegetasi)."
            )
            return

        self.log_panel.reset()
        self.preview.clear_results()
        self._result_gdf = None

        # Tentukan pipeline yang tepat
        only_building = enabled.get("building", False) and \
                        not any(v for k, v in enabled.items() if k != "building")

        if only_building:
            # Gunakan pipeline bangunan yang sudah ada (tidak diubah)
            target = self._run_pipeline
        else:
            # Gunakan pipeline multi-objek
            target = self._run_multi_object_pipeline

        self._processing_thread = threading.Thread(
            target=target,
            args=(params,),
            daemon=True,
        )
        self._processing_thread.start()

    def _on_stop(self):
        """Cancel ongoing processing."""
        if self._processor:
            self._processor.cancel()
            self._log("🛑 Permintaan berhenti dikirim...", "warning")

    def _on_train(self, geotiff_path: str, shp_path: str):
        """Starts the AI model training process in a background thread."""
        self.log_panel.reset()
        self._log("=== MEMULAI PROSES TRAINING MODEL AI ===", "system")
        
        train_thread = threading.Thread(
            target=self._run_training_pipeline,
            args=(geotiff_path, shp_path),
            daemon=True,
        )
        train_thread.start()

    def _train_progress(self, percent: float, message: str = ""):
        """Thread-safe training progress callback."""
        self.after(0, lambda: self.sidebar.set_training_progress(percent, message))
        if message:
            self._log(f"⚡ [Progress] {message}", "info")

    def _run_training_pipeline(self, geotiff_path: str, shp_path: str):
        """Runs the dataset generation and PyTorch fine-tuning in background."""
        import shutil
        from core.dataset_generator import generate_training_chips
        from train import train_model

        dataset_dir = os.path.join(ROOT, "dataset")
        
        try:
            # 1. Dataset Generation
            self._train_progress(0, "Mempersiapkan dataset...")
            success = generate_training_chips(
                geotiff_path=geotiff_path,
                shp_path=shp_path,
                output_dir=dataset_dir,
                chip_size=512,
                target_gsd=0.15,
                log_callback=self._log,
                progress_callback=self._train_progress,
            )
            
            if not success:
                self.after(0, lambda: self.sidebar.set_training_idle("Ekstraksi data gagal!", False))
                return

            # 2. PyTorch fine-tuning on GPU
            self._train_progress(70, "Menghubungkan ke GPU & Memulai training...")
            success = train_model(
                dataset_dir=dataset_dir,
                epochs=15,
                batch_size=2,
                lr=1e-4,
                log_callback=self._log,
                progress_callback=self._train_progress,
            )

            if success:
                self._log("🎉 TRAINING SELESAI DENGAN SUKSES!", "success")
                self._log("💾 Model Anda disimpan di: models/sam_bangunan_lokal.pth", "success")
                self._log("💡 Model ini siap digunakan untuk mendigitasi citra drone baru Anda!", "success")
                self.after(0, lambda: self.sidebar.set_training_idle("Latihan sukses! Model siap.", True))
            else:
                self._log("❌ Proses training gagal. Periksa log di atas.", "error")
                self.after(0, lambda: self.sidebar.set_training_idle("Training di GPU gagal!", False))

        except Exception as e:
            self._log(f"❌ Error selama training SAM: {e}", "error")
            self._log(traceback.format_exc(), "error")
            self.after(0, lambda: self.sidebar.set_training_idle("Error fatal training!", False))

    def _on_train_yolo(self, geotiff_path: str, shp_path: str, target_obj: str = "Bangunan"):
        """Starts the YOLO AI model training process in a background thread."""
        self.log_panel.reset()
        self._log("=== MEMULAI PROSES TRAINING YOLO ===", "system")
        
        train_yolo_thread = threading.Thread(
            target=self._run_training_yolo_pipeline,
            args=(geotiff_path, shp_path, target_obj),
            daemon=True,
        )
        train_yolo_thread.start()

    def _train_yolo_progress(self, percent: float, message: str = ""):
        """Thread-safe YOLO training progress callback."""
        self.after(0, lambda: self.sidebar.set_training_yolo_progress(percent, message))
        if message:
            self._log(f"⚡ [YOLO Progress] {message}", "info")

    def _run_training_yolo_pipeline(self, geotiff_path: str, shp_path: str, target_obj: str = "Bangunan"):
        """Runs YOLO dataset generation and training in background."""
        import shutil
        from core.yolo_dataset_generator import generate_yolo_dataset
        from train_yolo import train_yolo_model

        dataset_dir = os.path.join(ROOT, "dataset_yolo")
        target_obj_lower = target_obj.lower().replace("badan ", "").replace(" ", "_")
        output_model_name = f"yolo_{target_obj_lower}_lokal.pt"
        
        try:
            # 1. Dataset Generation
            self._train_yolo_progress(0, f"Mempersiapkan dataset YOLO untuk {target_obj}...")
            
            gen_mode = "segmentation" if target_obj_lower == "jalan" else "bbox"
            
            success = generate_yolo_dataset(
                geotiff_path=geotiff_path,
                shp_path=shp_path,
                output_dir=dataset_dir,
                chip_size=640,
                target_gsd=0.15,
                target_class_name=target_obj_lower,
                mode=gen_mode,
                log_callback=self._log,
                progress_callback=self._train_yolo_progress,
            )
            
            if not success:
                self.after(0, lambda: self.sidebar.set_training_yolo_idle("Ekstraksi data YOLO gagal!", False))
                return

            # 2. YOLO fine-tuning
            self._train_yolo_progress(70, "Menghubungkan ke GPU & Memulai training YOLO...")

            # Extract base model name from current UI selection if possible, otherwise default to yolov8n.pt
            base_model = self.sidebar.yolo_var.get().split()[0]
            
            if target_obj_lower == "jalan":
                base_model = os.path.join(ROOT, "core", "models", "uav-yolov12-seg.yaml")
                self._log("🛣️ Menggunakan arsitektur kustom UAV-YOLO12-Seg untuk jalan", "system")
            elif base_model == "yolo_bangunan_lokal.pt":
                custom_model_path = os.path.join(ROOT, "models", "yolo_bangunan_lokal.pt")
                if os.path.exists(custom_model_path):
                    base_model = custom_model_path
                    self._log(f"🔄 Melanjutkan training (fine-tuning) dari model kustom YOLO yang sudah ada: {custom_model_path}", "system")
                else:
                    base_model = "yolo12n.pt"
                    self._log(f"⚠️ Model kustom tidak ditemukan, memulai training baru dengan base model: {base_model}", "warning")

            success = train_yolo_model(
                dataset_dir=dataset_dir,
                epochs=100, # YOLO usually needs a bit more epochs
                batch_size=4,
                base_model=base_model,
                output_model_name=output_model_name,
                log_callback=self._log,
                progress_callback=self._train_yolo_progress,
            )

            if success:
                self._log("🎉 TRAINING YOLO SELESAI DENGAN SUKSES!", "success")
                self._log(f"💾 Model disimpan di: models/{output_model_name}", "success")
                self.after(0, lambda: self.sidebar.set_training_yolo_idle("Latihan sukses! Model siap.", True))
                # Update sidebar var to select the new custom model automatically
                self.after(0, lambda: self.sidebar.yolo_var.set(f"{output_model_name} (Custom)"))
            else:
                self._log("❌ Proses training YOLO gagal.", "error")
                self.after(0, lambda: self.sidebar.set_training_yolo_idle("Training YOLO gagal!", False))

        except Exception as e:
            self._log(f"❌ Error selama training YOLO: {e}", "error")
            self._log(traceback.format_exc(), "error")
            self.after(0, lambda: self.sidebar.set_training_yolo_idle("Error fatal training YOLO!", False))

    def _on_build_fingerprint(self, ref_raster: str, ref_shp: str, output_path: str):
        """Starts the Road Fingerprint builder in a background thread."""
        self.log_panel.reset()
        self._log("=== MEMBANGUN ROAD FINGERPRINT ===", "system")
        self._log(f"Citra referensi: {ref_raster}")
        self._log(f"Polygon referensi: {ref_shp}")
        
        thread = threading.Thread(
            target=self._run_build_fingerprint,
            args=(ref_raster, ref_shp, output_path),
            daemon=True,
        )
        thread.start()

    def _fp_progress(self, percent: float, message: str = ""):
        """Thread-safe fingerprint progress callback."""
        self.after(0, lambda: self.sidebar.set_fingerprint_progress(percent, message))
        if message:
            self._log(f"🔬 [Fingerprint] {message}", "info")

    def _run_build_fingerprint(self, ref_raster: str, ref_shp: str, output_path: str):
        from core.objects.road_fingerprint import RoadFingerprintBuilder
        try:
            builder = RoadFingerprintBuilder(
                log_callback=self._log,
                progress_callback=self._fp_progress,
            )
            result = builder.build(ref_raster, ref_shp, output_path)
            
            if result:
                self._log("🎉 ROAD FINGERPRINT BERHASIL DIBUAT!", "success")
                self._log(f"💾 Disimpan di: {output_path}", "success")
                self.after(0, lambda: self.sidebar.set_fingerprint_idle(f"✅ Disimpan: {os.path.basename(output_path)}", True))
                self.after(0, lambda: self.sidebar.road_fp_path_var.set(output_path))
            else:
                self._log("❌ Gagal membuat fingerprint (dibatalkan atau kosong).", "error")
                self.after(0, lambda: self.sidebar.set_fingerprint_idle("Gagal membuat fingerprint!", False))
                
        except Exception as e:
            self._log(f"❌ Error membuat fingerprint: {e}", "error")
            self._log(traceback.format_exc(), "error")
            self.after(0, lambda: self.sidebar.set_fingerprint_idle("Error fatal!", False))

    def _run_pipeline(self, params: dict):
        """
        Full processing pipeline, runs in background thread.
        All UI updates are dispatched to main thread via self.after().
        """
        input_path = params["input_path"]
        output_dir = params["output_dir"]
        model_name = params["model_name"]
        tile_size = params["tile_size"]
        min_area = params["min_area_m2"]
        max_area = params["max_area_m2"]

        import shutil
        temp_dir = os.path.join(ROOT, "temp")
        masks_dir = os.path.join(temp_dir, "masks")
        tiles_dir = os.path.join(temp_dir, "tiles")

        # Bersihkan cache lama (tiles & masks) agar tidak terjadi bug tumpang tindih antar citra
        self._log("🧹 Membersihkan cache dan file temporary lama...", "system")
        if os.path.exists(masks_dir):
            shutil.rmtree(masks_dir, ignore_errors=True)
        if os.path.exists(tiles_dir):
            shutil.rmtree(tiles_dir, ignore_errors=True)

        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(masks_dir, exist_ok=True)
        os.makedirs(tiles_dir, exist_ok=True)

        try:
            # ─────────────────────────────────────────────────
            # STEP 1: Handle ECW → GeoTIFF conversion
            # ─────────────────────────────────────────────────
            self._progress(5, "Memeriksa format file input...")
            ext = Path(input_path).suffix.lower()

            if ext == ".ecw":
                self._log("📥 File ECW terdeteksi, memulai konversi ke GeoTIFF...", "system")
                from core.ecw_handler import find_gdal_translate, convert_ecw_to_geotiff
                gdal_path = find_gdal_translate()
                if not gdal_path:
                    raise RuntimeError(
                        "GDAL tidak ditemukan. File ECW tidak dapat dikonversi.\n"
                        "Solusi: Install OSGeo4W atau konversi ke GeoTIFF menggunakan QGIS."
                    )
                raster_path = convert_ecw_to_geotiff(
                    input_path, temp_dir, gdal_path,
                    progress_callback=self._progress,
                )
                self._log(f"✅ Konversi selesai: {Path(raster_path).name}", "success")
            else:
                raster_path = input_path
                self._progress(25, "File GeoTIFF siap diproses")
                self._log(f"📥 File GeoTIFF: {Path(raster_path).name}", "system")

            # ─────────────────────────────────────────────────
            # STEP 2: Load and display raster preview
            # ─────────────────────────────────────────────────
            self._progress(27, "Memuat preview citra...")
            self.after(0, lambda: self.preview.load_raster_preview(raster_path))

            # ─────────────────────────────────────────────────
            # STEP 3: Calculate tile grid for display
            # ─────────────────────────────────────────────────
            self._progress(28, "Menghitung grid tile...")
            self._log("📐 Menghitung grid tile...", "system")

            import rasterio
            from core.tiling import calculate_tile_grid
            with rasterio.open(raster_path) as src:
                w, h = src.width, src.height
                transform = src.transform

            tiles_px = calculate_tile_grid(w, h, tile_size, tile_size // 8)
            total_tiles = len(tiles_px)
            self._log(f"📐 Total tile: {total_tiles} ({w}×{h} px, tile {tile_size}px)", "system")

            # Convert tile pixel coords to geo coords for preview
            tiles_geo = []
            for (col_off, row_off, tw, th) in tiles_px:
                x0, y0 = transform * (col_off, row_off)
                x1, y1 = transform * (col_off + tw, row_off + th)
                tiles_geo.append((min(x0,x1), min(y0,y1), max(x0,x1), max(y0,y1)))

            self.after(0, lambda tg=tiles_geo: self.preview.set_tile_grid(tg))

            # ─────────────────────────────────────────────────
            # STEP 4: Load SAM model
            # ─────────────────────────────────────────────────
            self._progress(30, f"Memuat model {model_name}...")
            self._log(f"🤖 Memuat model: {model_name}", "system")

            from core.sam_processor import SAMProcessor
            self._processor = SAMProcessor(
                model_name=model_name,
                models_dir=os.path.join(ROOT, "models"),
                device="auto",
                points_per_side=params.get("points_per_side", 48),
                mode=params.get("mode", "Otomatis (Grid Buta)"),
                yolo_model_name=params.get("yolo_model", "yolo12n.pt (YOLO12 Nano - Terbaru)"),
                log_callback=self._log,
                progress_callback=self._progress,
            )
            self._processor.reset_cancel()
            self._processor.load_model()

            if self._processor.is_cancelled():
                self._log("🛑 Proses dihentikan saat loading model.", "warning")
                return

            # ─────────────────────────────────────────────────
            # STEP 5: Process tiles with SAM
            # ─────────────────────────────────────────────────
            self._progress(40, "Memulai segmentasi SAM...")
            self._log(f"🚀 Memulai segmentasi pada {total_tiles} tile...", "sam")

            mask_results, yolo_boxes = self._processor.process_raster(
                raster_path=raster_path,
                masks_dir=masks_dir,
                tile_size=tile_size,
                overlap=tile_size // 8,
                enable_filtering=params.get("enable_tile_filtering", True),
                min_area_m2=min_area,
                max_area_m2=max_area,
                tile_progress_callback=lambda cur, tot: (
                    self._tile_progress(cur, tot),
                    self._progress(
                        40 + int((cur / tot) * 25),
                        f"SAM tile {cur}/{tot}"
                    )
                )[-1],  # Return last value (for lambda compatibility)
            )

            if self._processor.is_cancelled():
                self._log("🛑 Proses dihentikan oleh pengguna.", "warning")
                return

            self._log(f"✅ Segmentasi selesai: {len(mask_results)} tile diproses", "success")

            # ─────────────────────────────────────────────────
            # STEP 6: Post-processing
            # ─────────────────────────────────────────────────
            self._progress(68, "Memulai post-processing...")
            self._log("🔧 Memulai post-processing...", "system")

            from core.postprocess import run_postprocess_pipeline
            result_gdf = run_postprocess_pipeline(
                mask_paths_and_metas=mask_results,
                original_raster_path=raster_path,
                min_area_m2=min_area,
                max_area_m2=max_area,
                max_aspect_ratio=8.0,
                enable_shadow_filter=params["enable_shadow_filter"],
                enable_vegetation_filter=params.get("enable_vegetation_filter", True),
                enable_regularization=params["enable_regularization"],
                simplify_tolerance=params.get("simplify_tolerance", 0.75),
                log_callback=self._log,
                progress_callback=self._progress,
            )

            self._result_gdf = result_gdf
            self._last_building_gdf = result_gdf  # Untuk priority clipping multi-objek

            if len(result_gdf) == 0:
                self._log("⚠️ Tidak ada bangunan yang terdeteksi. Coba turunkan nilai min area.", "warning")
                return

            # Update preview with results
            show_yolo = params.get("show_yolo_preview", False)
            self.after(0, lambda gdf=result_gdf, yb=yolo_boxes if show_yolo else None: self.preview.set_result_polygons(gdf, yb))

            # ─────────────────────────────────────────────────
            # STEP 7: Export to Shapefile
            # ─────────────────────────────────────────────────
            self._progress(96, "Mengekspor ke Shapefile...")
            self._log("💾 Mengekspor hasil ke Shapefile...", "system")

            input_stem = Path(input_path).stem
            output_shp = os.path.join(output_dir, f"{input_stem}_bangunan.shp")

            from core.exporter import export_to_shapefile, get_output_summary
            export_to_shapefile(
                result_gdf,
                output_shp,
                source_raster_path=raster_path,
                log_callback=self._log,
            )

            summary = get_output_summary(output_shp)
            self._progress(100, f"✅ Selesai! {summary.get('count', len(result_gdf))} bangunan diekspor")
            self._log(
                f"\n{'='*50}\n"
                f"✅ DIGITASI SELESAI\n"
                f"   Bangunan terdeteksi: {summary.get('count', len(result_gdf)):,}\n"
                f"   Total luas: {summary.get('total_area_m2', 0):,.1f} m²\n"
                f"   Rata-rata luas: {summary.get('avg_area_m2', 0):,.1f} m²\n"
                f"   Output: {output_shp}\n"
                f"   Ukuran file: {summary.get('file_size_kb', 0):,} KB\n"
                f"{'='*50}",
                "success"
            )

            # Show success dialog
            self.after(0, lambda: messagebox.showinfo(
                "Digitasi Selesai",
                f"✅ {summary.get('count', len(result_gdf)):,} bangunan berhasil didigitasi!\n\n"
                f"File output:\n{output_shp}\n\n"
                f"Buka file .shp di QGIS atau ArcGIS.",
            ))

        except Exception as e:
            err_msg = f"❌ ERROR: {e}\n{traceback.format_exc()}"
            self._log(err_msg, "error")
            self._progress(0, "Error — lihat log")
            self.after(0, lambda msg=str(e): messagebox.showerror(
                "Error",
                f"Terjadi kesalahan:\n\n{msg}\n\nLihat log untuk detail."
            ))

        finally:
            # Unload model to free GPU memory
            if self._processor:
                try:
                    self._processor.unload_model()
                except Exception:
                    pass
                self._processor = None
            # Re-enable buttons
            self.after(0, self.sidebar.set_idle)

    # ─────────────────────────────────────────────────────────────────────────
    # PIPELINE MULTI-OBJEK (Jalan, Air, Vegetasi, dan/atau Bangunan)
    # Metode ini berdiri sendiri dan TIDAK mengubah _run_pipeline() di atas.
    # ─────────────────────────────────────────────────────────────────────────

    def _run_multi_object_pipeline(self, params: dict):
        """
        Pipeline multi-objek: jalankan digitasi untuk semua kelas terpilih.
        Jika bangunan juga dipilih, pipeline bangunan dijalankan terpisah TERLEBIH DAHULU
        (menggunakan _run_pipeline yang sudah ada), lalu objek lain menyusul.
        Semua hasil non-bangunan digabung ke satu shapefile gabungan dengan kolom 'class'.
        """
        input_path    = params["input_path"]
        output_dir    = params["output_dir"]
        model_name    = params["model_name"]
        tile_size     = params["tile_size"]
        enabled       = params.get("enabled_objects", {"building": False})

        import shutil
        temp_dir  = os.path.join(ROOT, "temp")
        masks_dir = os.path.join(temp_dir, "masks")
        tiles_dir = os.path.join(temp_dir, "tiles")

        self._log("🧹 Membersihkan cache dan file temporary lama...", "system")
        if os.path.exists(masks_dir):
            shutil.rmtree(masks_dir, ignore_errors=True)
        if os.path.exists(tiles_dir):
            shutil.rmtree(tiles_dir, ignore_errors=True)

        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(masks_dir, exist_ok=True)
        os.makedirs(tiles_dir, exist_ok=True)

        try:
            # ── STEP 1: Handle ECW → GeoTIFF
            self._progress(5, "Memeriksa format file input...")
            ext = Path(input_path).suffix.lower()
            if ext == ".ecw":
                self._log("📥 File ECW terdeteksi, memulai konversi ke GeoTIFF...", "system")
                from core.ecw_handler import find_gdal_translate, convert_ecw_to_geotiff
                gdal_path = find_gdal_translate()
                if not gdal_path:
                    raise RuntimeError("GDAL tidak ditemukan. File ECW tidak dapat dikonversi.")
                raster_path = convert_ecw_to_geotiff(
                    input_path, temp_dir, gdal_path, progress_callback=self._progress
                )
                self._log(f"✅ Konversi selesai: {Path(raster_path).name}", "success")
            else:
                raster_path = input_path
                self._progress(10, "File GeoTIFF siap diproses")
                self._log(f"📥 File GeoTIFF: {Path(raster_path).name}", "system")

            # ── STEP 2: Preview raster
            self._progress(12, "Memuat preview citra...")
            self.after(0, lambda: self.preview.load_raster_preview(raster_path))

            input_stem = Path(input_path).stem

            # ── STEP 3: Bangunan (jika diaktifkan) — gunakan pipeline bangunan yang ada
            if enabled.get("building", False):
                self._log("\n🏠 Memulai digitasi BANGUNAN...", "system")
                self._progress(15, "Memulai digitasi bangunan...")

                # Buat parameter khusus untuk pipeline bangunan (tanpa multi-object routing)
                building_params = dict(params)
                building_params["enabled_objects"] = {"building": True}  # Paksa hanya bangunan

                # Panggil _run_pipeline langsung (blocking di thread ini sudah aman)
                self._run_pipeline(building_params)
                # _run_pipeline sudah memanggil set_idle di akhir, perlu di-override
                # Kita set kembali tombol ke loading state untuk objek berikutnya
                self.after(0, lambda: self.sidebar.btn_run.configure(
                    state="disabled", text="⏳  Memproses objek lain..."
                ))

            # ── STEP 4: Objek non-bangunan
            non_building_active = {k: v for k, v in enabled.items() if k != "building" and v}
            if non_building_active:
                self._progress(50, "Memulai digitasi multi-objek...")

                from core.multi_object_runner import (
                    run_multi_object_digitization,
                    export_combined_shapefile,
                    get_multi_object_summary,
                )

                combined_gdf = run_multi_object_digitization(
                    raster_path=raster_path,
                    output_dir=output_dir,
                    input_stem=input_stem,
                    sam_model_name=model_name,
                    tile_size=tile_size,
                    enabled_objects=enabled,
                    params=params,
                    masks_base_dir=masks_dir,
                    building_gdf=getattr(self, "_last_building_gdf", None),
                    log_callback=self._log,
                    progress_callback=self._progress,
                    cancel_check=lambda: (self._processor.is_cancelled()
                                         if self._processor else False),
                )

                if combined_gdf is not None and len(combined_gdf) > 0:
                    # Tampilkan di preview dengan warna per kelas
                    self.after(0, lambda gdf=combined_gdf:
                               self.preview.set_multi_object_polygons(gdf))

                    # Ekspor ke Shapefile gabungan
                    self._progress(92, "Mengekspor ke Shapefile gabungan...")
                    output_shp = os.path.join(output_dir, f"{input_stem}_digitasi.shp")
                    export_combined_shapefile(
                        combined_gdf,
                        output_shp,
                        source_raster_path=raster_path,
                        log_callback=self._log,
                    )

                    summary = get_multi_object_summary(output_shp)
                    self._progress(100, f"✅ Selesai! {summary.get('total', len(combined_gdf))} poligon diekspor")

                    # Susun pesan ringkasan per kelas
                    by_class = summary.get("by_class", {})
                    class_lines = "\n".join(
                        f"   - {cls}: {info['count']:,} poligon "
                        f"({info.get('total_area_m2', 0):,.1f} m²)"
                        for cls, info in by_class.items()
                    )

                    self._log(
                        f"\n{'='*50}\n"
                        f"✅ DIGITASI MULTI-OBJEK SELESAI\n"
                        f"{class_lines}\n"
                        f"   Total: {summary.get('total', len(combined_gdf)):,} poligon\n"
                        f"   Output: {output_shp}\n"
                        f"   Ukuran file: {summary.get('file_size_kb', 0):,} KB\n"
                        f"{'='*50}",
                        "success"
                    )

                    rincian_lines = "\n".join(
                        f"  {cls}: {info['count']:,}"
                        for cls, info in by_class.items()
                    )
                    self.after(0, lambda msg_body=(
                        f"✅ {summary.get('total', len(combined_gdf)):,} poligon berhasil didigitasi!\n\n"
                        f"Rincian:\n{rincian_lines}\n\n"
                        f"File output:\n{output_shp}\n\n"
                        f"Buka file .shp di QGIS atau ArcGIS."
                    ): messagebox.showinfo("Digitasi Selesai", msg_body))

                else:
                    if not enabled.get("building", False):
                        self._log("⚠️ Tidak ada objek yang terdeteksi.", "warning")
                        self._progress(0, "Tidak ada hasil")

        except Exception as e:
            err_msg = f"❌ ERROR: {e}\n{traceback.format_exc()}"
            self._log(err_msg, "error")
            self._progress(0, "Error — lihat log")
            self.after(0, lambda msg=str(e): messagebox.showerror(
                "Error", f"Terjadi kesalahan:\n\n{msg}\n\nLihat log untuk detail."
            ))

        finally:
            if self._processor:
                try:
                    self._processor.unload_model()
                except Exception:
                    pass
                self._processor = None
            self.after(0, self.sidebar.set_idle)

    def _on_close(self):
        """Graceful shutdown."""
        if self._processing_thread and self._processing_thread.is_alive():
            if messagebox.askyesno(
                "Keluar",
                "Proses masih berjalan. Yakin ingin keluar?\nProses akan dihentikan.",
            ):
                if self._processor:
                    self._processor.cancel()
                self.destroy()
        else:
            self.destroy()


def main():
    app = SAMGeoApp()
    app.mainloop()


if __name__ == "__main__":
    main()

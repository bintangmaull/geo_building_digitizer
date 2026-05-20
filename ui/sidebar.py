"""
sidebar.py
Left-side control panel with file input, model selection,
processing parameters, and action buttons.
"""

import os
import customtkinter as ctk
import tkinter as tk
from tkinter import filedialog, messagebox
from typing import Callable, Optional


class Sidebar(ctk.CTkScrollableFrame):
    """
    Left control panel with all user-configurable settings.
    """

    MODEL_OPTIONS = [
        # ── SAM2 (Generasi Terbaru - Direkomendasikan) ──
        "SAM2-Tiny (Cepat, ~155MB)",
        "SAM2-Small (Seimbang, ~185MB)",
        "SAM2-Base+ (Akurat, ~325MB)",
        "SAM2-Large (Sangat Akurat, ~898MB)",
        # ── SAM1 (Generasi Pertama) ──
        "SAM-B (Seimbang, ~375MB)",
        "SAM-B (Custom - Bangunan Lokal)",
        "MobileSAM (Cepat, ~40MB)",
        "SAM-L (Akurat, ~1.2GB)",
        "SAM-H (Sangat Akurat, ~2.4GB)",
    ]

    TILE_OPTIONS = ["512", "1024", "2048"]

    def __init__(self, parent, on_run: Callable, on_stop: Callable, on_train: Callable, on_train_yolo: Callable, **kwargs):
        super().__init__(parent, **kwargs)
        self.on_run = on_run
        self.on_stop = on_stop
        self.on_train = on_train
        self.on_train_yolo = on_train_yolo
        self._input_path = tk.StringVar()
        self._output_dir = tk.StringVar(value=str(os.path.join(os.getcwd(), "output")))
        self._build_ui()

    def _section_label(self, text: str, row: int):
        """Create a styled section header label."""
        lbl = ctk.CTkLabel(
            self,
            text=text,
            font=ctk.CTkFont(family="Segoe UI", size=10, weight="bold"),
            text_color="#475569",
            anchor="w",
        )
        lbl.grid(row=row, column=0, columnspan=2, sticky="ew", padx=12, pady=(14, 2))

    def _divider(self, row: int):
        frame = ctk.CTkFrame(self, height=1, fg_color="#1E293B")
        frame.grid(row=row, column=0, columnspan=2, sticky="ew", padx=8, pady=2)

    def _build_ui(self):
        self.grid_columnconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=0)

        row = 0

        # ── App Title ──────────────────────────────────
        title_frame = ctk.CTkFrame(self, fg_color="#0F172A", corner_radius=12)
        title_frame.grid(row=row, column=0, columnspan=2, sticky="ew", padx=8, pady=(12, 8))
        title_frame.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(
            title_frame,
            text="🛰️",
            font=ctk.CTkFont(size=28),
        ).grid(row=0, column=0, pady=(12, 0))

        ctk.CTkLabel(
            title_frame,
            text="SAM-Geo Digitizer",
            font=ctk.CTkFont(family="Segoe UI", size=15, weight="bold"),
            text_color="#E2E8F0",
        ).grid(row=1, column=0)

        ctk.CTkLabel(
            title_frame,
            text="Digitasi Bangunan Otomatis",
            font=ctk.CTkFont(family="Segoe UI", size=10),
            text_color="#475569",
        ).grid(row=2, column=0, pady=(0, 12))

        row += 1; self._divider(row); row += 1

        # ── 1. Input File ──────────────────────────────
        self._section_label("📂  FILE INPUT", row); row += 1

        self.lbl_input = ctk.CTkLabel(
            self,
            text="Belum ada file dipilih",
            font=ctk.CTkFont(size=10),
            text_color="#64748B",
            wraplength=180,
            justify="left",
            anchor="w",
        )
        self.lbl_input.grid(row=row, column=0, columnspan=2, sticky="ew", padx=12, pady=(0, 4))
        row += 1

        ctk.CTkButton(
            self,
            text="📂  Pilih File ECW / TIF",
            font=ctk.CTkFont(size=12, weight="bold"),
            height=38,
            corner_radius=8,
            fg_color="#6366F1",
            hover_color="#4F46E5",
            command=self._browse_input,
        ).grid(row=row, column=0, columnspan=2, sticky="ew", padx=12, pady=(0, 4))
        row += 1

        # Raster info display
        self.info_frame = ctk.CTkFrame(self, fg_color="#0F172A", corner_radius=8)
        self.info_frame.grid(row=row, column=0, columnspan=2, sticky="ew", padx=12, pady=(0, 4))
        self.info_frame.grid_columnconfigure(1, weight=1)

        self.info_labels = {}
        for i, (key, label) in enumerate([
            ("size", "Ukuran"),
            ("crs", "CRS"),
            ("resolution", "Resolusi"),
            ("bands", "Band"),
        ]):
            ctk.CTkLabel(
                self.info_frame,
                text=f"{label}:",
                font=ctk.CTkFont(size=9),
                text_color="#475569",
                anchor="w",
            ).grid(row=i, column=0, sticky="w", padx=(8, 4), pady=1)

            self.info_labels[key] = ctk.CTkLabel(
                self.info_frame,
                text="—",
                font=ctk.CTkFont(size=9),
                text_color="#94A3B8",
                anchor="w",
            )
            self.info_labels[key].grid(row=i, column=1, sticky="ew", padx=(0, 8), pady=1)
        row += 1

        # ── 2. Output ──────────────────────────────────
        row += 1; self._divider(row); row += 1
        self._section_label("📤  OUTPUT", row); row += 1

        self.lbl_output = ctk.CTkLabel(
            self,
            text=self._output_dir.get(),
            font=ctk.CTkFont(size=10),
            text_color="#64748B",
            wraplength=180,
            justify="left",
            anchor="w",
        )
        self.lbl_output.grid(row=row, column=0, columnspan=2, sticky="ew", padx=12, pady=(0, 4))
        row += 1

        ctk.CTkButton(
            self,
            text="📁  Pilih Folder Output",
            font=ctk.CTkFont(size=11),
            height=32,
            corner_radius=8,
            fg_color="#1E293B",
            hover_color="#334155",
            text_color="#94A3B8",
            command=self._browse_output,
        ).grid(row=row, column=0, columnspan=2, sticky="ew", padx=12, pady=(0, 4))
        row += 1

        # ── 3. Model Selection ─────────────────────────
        row += 1; self._divider(row); row += 1
        self._section_label("🤖  MODEL SAM", row); row += 1

        self.model_var = tk.StringVar(value="SAM2-Small (Seimbang, ~185MB)")
        self.model_menu = ctk.CTkOptionMenu(
            self,
            values=self.MODEL_OPTIONS,
            variable=self.model_var,
            font=ctk.CTkFont(size=11),
            height=32,
            fg_color="#1E293B",
            button_color="#334155",
            button_hover_color="#475569",
            dropdown_fg_color="#1E293B",
            dropdown_hover_color="#334155",
            text_color="#E2E8F0",
        )
        self.model_menu.grid(row=row, column=0, columnspan=2, sticky="ew", padx=12, pady=(0, 4))
        row += 1

        # Mode Digitasi
        ctk.CTkLabel(
            self, text="Mode Digitasi:",
            font=ctk.CTkFont(size=11), text_color="#CBD5E1", anchor="w",
        ).grid(row=row, column=0, sticky="w", padx=12, pady=(4, 0))

        self.mode_var = tk.StringVar(value="Pra-Deteksi (CV + Point Prompts)")
        self.mode_menu = ctk.CTkOptionMenu(
            self,
            values=["Otomatis (Grid Buta)", "Pra-Deteksi (CV + Point Prompts)", "Pra-Deteksi (YOLO Bounding Box)"],
            variable=self.mode_var,
            font=ctk.CTkFont(size=11),
            height=28,
            fg_color="#1E293B",
            button_color="#334155",
            button_hover_color="#475569",
            dropdown_fg_color="#1E293B",
            dropdown_hover_color="#334155",
            text_color="#E2E8F0",
        )
        self.mode_menu.grid(row=row, column=1, sticky="ew", padx=(4, 12), pady=(4, 0))
        row += 1

        # YOLO Model Selector (Only relevant if YOLO mode is selected)
        ctk.CTkLabel(
            self, text="Model YOLO:",
            font=ctk.CTkFont(size=11), text_color="#CBD5E1", anchor="w",
        ).grid(row=row, column=0, sticky="w", padx=12, pady=(4, 0))

        self.yolo_var = tk.StringVar(value="yolov8n.pt (Nano - Cepat)")
        self.yolo_menu = ctk.CTkOptionMenu(
            self,
            values=["yolov8n.pt (Nano - Cepat)", "yolov8s.pt (Small - Seimbang)", "yolo_bangunan_lokal.pt (Custom)"],
            variable=self.yolo_var,
            font=ctk.CTkFont(size=11),
            height=28,
            fg_color="#1E293B",
            button_color="#334155",
            button_hover_color="#475569",
            dropdown_fg_color="#1E293B",
            dropdown_hover_color="#334155",
            text_color="#E2E8F0",
        )
        self.yolo_menu.grid(row=row, column=1, sticky="ew", padx=(4, 12), pady=(4, 0))
        row += 1

        self.show_yolo_preview_var = tk.BooleanVar(value=True)
        self.switch_yolo_preview = ctk.CTkSwitch(
            self, text="Tampilkan Preview Bounding Box (YOLO)",
            variable=self.show_yolo_preview_var,
            font=ctk.CTkFont(size=11), text_color="#E2E8F0",
        )
        self.switch_yolo_preview.grid(row=row, column=0, columnspan=2, sticky="w", padx=12, pady=(8, 4))
        row += 1

        # ── 4. Tile Size ───────────────────────────────
        row += 1; self._divider(row); row += 1
        self._section_label("⚙️  PARAMETER PROSES", row); row += 1

        ctk.CTkLabel(
            self, text="Ukuran Tile (px):",
            font=ctk.CTkFont(size=11), text_color="#CBD5E1", anchor="w",
        ).grid(row=row, column=0, sticky="w", padx=12)

        self.tile_var = tk.StringVar(value="1024")
        tile_seg = ctk.CTkSegmentedButton(
            self,
            values=self.TILE_OPTIONS,
            variable=self.tile_var,
            font=ctk.CTkFont(size=11),
            height=28,
            fg_color="#1E293B",
            selected_color="#6366F1",
            selected_hover_color="#4F46E5",
            unselected_color="#1E293B",
            unselected_hover_color="#334155",
            text_color="#E2E8F0",
        )
        tile_seg.grid(row=row, column=1, sticky="ew", padx=(4, 12))
        row += 1

        # Luas minimum
        ctk.CTkLabel(
            self, text="Luas Min (m²):",
            font=ctk.CTkFont(size=11), text_color="#CBD5E1", anchor="w",
        ).grid(row=row, column=0, sticky="w", padx=12, pady=(4, 0))

        self.min_area_var = ctk.CTkEntry(
            self, placeholder_text="20", width=80, height=28,
            font=ctk.CTkFont(size=11),
            fg_color="#1E293B", border_color="#334155", text_color="#E2E8F0",
        )
        self.min_area_var.grid(row=row, column=1, sticky="ew", padx=(4, 12), pady=(4, 0))
        self.min_area_var.insert(0, "20")
        row += 1

        # Luas maksimum
        ctk.CTkLabel(
            self, text="Luas Maks (m²):",
            font=ctk.CTkFont(size=11), text_color="#CBD5E1", anchor="w",
        ).grid(row=row, column=0, sticky="w", padx=12, pady=(4, 0))

        self.max_area_var = ctk.CTkEntry(
            self, placeholder_text="100000", width=80, height=28,
            font=ctk.CTkFont(size=11),
            fg_color="#1E293B", border_color="#334155", text_color="#E2E8F0",
        )
        self.max_area_var.grid(row=row, column=1, sticky="ew", padx=(4, 12), pady=(4, 0))
        self.max_area_var.insert(0, "5000")
        row += 1

        # Kerapatan Titik (Points per Side)
        ctk.CTkLabel(
            self, text="Kerapatan Titik:",
            font=ctk.CTkFont(size=11), text_color="#CBD5E1", anchor="w",
        ).grid(row=row, column=0, sticky="w", padx=12, pady=(4, 0))

        self.pts_var = tk.StringVar(value="48 (Seimbang)")
        self.pts_menu = ctk.CTkOptionMenu(
            self,
            values=["32 (Cepat)", "48 (Seimbang)", "64 (Akurat)"],
            variable=self.pts_var,
            font=ctk.CTkFont(size=11),
            height=28,
            fg_color="#1E293B",
            button_color="#334155",
            button_hover_color="#475569",
            dropdown_fg_color="#1E293B",
            dropdown_hover_color="#334155",
            text_color="#E2E8F0",
        )
        self.pts_menu.grid(row=row, column=1, sticky="ew", padx=(4, 12), pady=(4, 0))
        row += 1

        # ── 5. Toggles ────────────────────────────────
        row += 1

        self.filter_empty_var = tk.BooleanVar(value=True)
        ctk.CTkCheckBox(
            self,
            text="Filter Ubin Kosong (Skip Forest/Water)",
            variable=self.filter_empty_var,
            font=ctk.CTkFont(size=11),
            text_color="#CBD5E1",
            checkmark_color="#0F172A",
            fg_color="#6366F1",
            hover_color="#4F46E5",
            border_color="#334155",
            height=24,
        ).grid(row=row, column=0, columnspan=2, sticky="w", padx=12, pady=(4, 0))
        row += 1

        self.shadow_var = tk.BooleanVar(value=True)
        ctk.CTkCheckBox(
            self,
            text="Filter Bayangan",
            variable=self.shadow_var,
            font=ctk.CTkFont(size=11),
            text_color="#CBD5E1",
            checkmark_color="#0F172A",
            fg_color="#6366F1",
            hover_color="#4F46E5",
            border_color="#334155",
            height=24,
        ).grid(row=row, column=0, columnspan=2, sticky="w", padx=12, pady=(4, 0))
        row += 1

        self.vegetation_var = tk.BooleanVar(value=True)
        ctk.CTkCheckBox(
            self,
            text="Filter Vegetasi (Pohon/Halaman)",
            variable=self.vegetation_var,
            font=ctk.CTkFont(size=11),
            text_color="#CBD5E1",
            checkmark_color="#0F172A",
            fg_color="#6366F1",
            hover_color="#4F46E5",
            border_color="#334155",
            height=24,
        ).grid(row=row, column=0, columnspan=2, sticky="w", padx=12, pady=(2, 0))
        row += 1

        self.regularize_var = tk.BooleanVar(value=True)
        ctk.CTkCheckBox(
            self,
            text="Regularisasi Sudut Bangunan",
            variable=self.regularize_var,
            font=ctk.CTkFont(size=11),
            text_color="#CBD5E1",
            checkmark_color="#0F172A",
            fg_color="#6366F1",
            hover_color="#4F46E5",
            border_color="#334155",
            height=24,
        ).grid(row=row, column=0, columnspan=2, sticky="w", padx=12, pady=(4, 0))
        row += 1

        # ── 6. Action Buttons ──────────────────────────
        row += 1; self._divider(row); row += 1

        self.btn_run = ctk.CTkButton(
            self,
            text="▶  JALANKAN DIGITASI",
            font=ctk.CTkFont(family="Segoe UI", size=13, weight="bold"),
            height=46,
            corner_radius=10,
            fg_color="#10B981",
            hover_color="#059669",
            text_color="#FFFFFF",
            command=self._on_run_clicked,
        )
        self.btn_run.grid(row=row, column=0, columnspan=2, sticky="ew", padx=12, pady=(8, 4))
        row += 1

        self.btn_stop = ctk.CTkButton(
            self,
            text="⏹  HENTIKAN",
            font=ctk.CTkFont(family="Segoe UI", size=12, weight="bold"),
            height=36,
            corner_radius=10,
            fg_color="#1E293B",
            hover_color="#7F1D1D",
            text_color="#F87171",
            command=self._on_stop_clicked,
            state="disabled",
        )
        self.btn_stop.grid(row=row, column=0, columnspan=2, sticky="ew", padx=12, pady=(0, 12))
        row += 1

        # ── 6b. Training Model AI ──────────────────────
        row += 1; self._divider(row); row += 1
        self._section_label("🧠  TRAINING MODEL SAM (FINE-TUNING)", row); row += 1
        
        self.lbl_train_geotiff = ctk.CTkLabel(
            self,
            text="1. File Citra Drone (GeoTIFF):",
            font=ctk.CTkFont(size=10, weight="bold"),
            text_color="#CBD5E1",
            anchor="w",
        )
        self.lbl_train_geotiff.grid(row=row, column=0, columnspan=2, sticky="w", padx=12, pady=(4, 0))
        row += 1
        
        self.entry_train_geotiff = ctk.CTkEntry(
            self,
            placeholder_text="Path citra .tif...",
            font=ctk.CTkFont(size=11),
            height=30,
            fg_color="#0F172A",
            border_color="#334155",
            text_color="#E2E8F0",
        )
        self.entry_train_geotiff.grid(row=row, column=0, sticky="ew", padx=(12, 4), pady=(0, 4))
        
        self.btn_browse_train_geotiff = ctk.CTkButton(
            self,
            text="📂",
            width=36,
            height=30,
            corner_radius=6,
            fg_color="#334155",
            hover_color="#475569",
            command=self._browse_train_geotiff,
        )
        self.btn_browse_train_geotiff.grid(row=row, column=1, sticky="w", padx=(0, 12), pady=(0, 4))
        row += 1
        
        self.lbl_train_shp = ctk.CTkLabel(
            self,
            text="2. File Digitasi Bangunan (Shapefile .shp):",
            font=ctk.CTkFont(size=10, weight="bold"),
            text_color="#CBD5E1",
            anchor="w",
        )
        self.lbl_train_shp.grid(row=row, column=0, columnspan=2, sticky="w", padx=12, pady=(4, 0))
        row += 1
        
        self.entry_train_shp = ctk.CTkEntry(
            self,
            placeholder_text="Path polygon .shp...",
            font=ctk.CTkFont(size=11),
            height=30,
            fg_color="#0F172A",
            border_color="#334155",
            text_color="#E2E8F0",
        )
        self.entry_train_shp.grid(row=row, column=0, sticky="ew", padx=(12, 4), pady=(0, 4))
        
        self.btn_browse_train_shp = ctk.CTkButton(
            self,
            text="📂",
            width=36,
            height=30,
            corner_radius=6,
            fg_color="#334155",
            hover_color="#475569",
            command=self._browse_train_shp,
        )
        self.btn_browse_train_shp.grid(row=row, column=1, sticky="w", padx=(0, 12), pady=(0, 4))
        row += 1
        
        self.btn_train_run = ctk.CTkButton(
            self,
            text="🔥  MULAI TRAINING AI",
            font=ctk.CTkFont(family="Segoe UI", size=12, weight="bold"),
            height=38,
            corner_radius=8,
            fg_color="#D97706",
            hover_color="#B45309",
            text_color="#FFFFFF",
            command=self._on_train_clicked,
        )
        self.btn_train_run.grid(row=row, column=0, columnspan=2, sticky="ew", padx=12, pady=(6, 4))
        row += 1
        
        self.lbl_train_status = ctk.CTkLabel(
            self,
            text="Status: Idle (Siap training)",
            font=ctk.CTkFont(size=10),
            text_color="#64748B",
            anchor="w",
            wraplength=180,
            justify="left",
        )
        self.lbl_train_status.grid(row=row, column=0, columnspan=2, sticky="ew", padx=12, pady=(2, 0))
        row += 1
        
        self.train_progress = ctk.CTkProgressBar(
            self,
            height=8,
            corner_radius=4,
            fg_color="#1E293B",
            progress_color="#D97706",
        )
        self.train_progress.grid(row=row, column=0, columnspan=2, sticky="ew", padx=12, pady=(4, 12))
        self.train_progress.set(0)
        row += 1

        # ── 6c. Training Model YOLOv8 ──────────────────
        row += 1; self._divider(row); row += 1
        self._section_label("🎯  TRAINING MODEL YOLOv8", row); row += 1
        
        self.lbl_train_yolo_geotiff = ctk.CTkLabel(
            self,
            text="1. File Citra Drone (GeoTIFF):",
            font=ctk.CTkFont(size=10, weight="bold"),
            text_color="#CBD5E1",
            anchor="w",
        )
        self.lbl_train_yolo_geotiff.grid(row=row, column=0, columnspan=2, sticky="w", padx=12, pady=(4, 0))
        row += 1
        
        self.entry_train_yolo_geotiff = ctk.CTkEntry(
            self,
            placeholder_text="Path citra .tif...",
            font=ctk.CTkFont(size=11),
            height=30,
            fg_color="#0F172A",
            border_color="#334155",
            text_color="#E2E8F0",
        )
        self.entry_train_yolo_geotiff.grid(row=row, column=0, sticky="ew", padx=(12, 4), pady=(0, 4))
        
        self.btn_browse_train_yolo_geotiff = ctk.CTkButton(
            self,
            text="📂",
            width=36,
            height=30,
            corner_radius=6,
            fg_color="#334155",
            hover_color="#475569",
            command=self._browse_train_yolo_geotiff,
        )
        self.btn_browse_train_yolo_geotiff.grid(row=row, column=1, sticky="w", padx=(0, 12), pady=(0, 4))
        row += 1
        
        self.lbl_train_yolo_shp = ctk.CTkLabel(
            self,
            text="2. File Digitasi Bangunan (Shapefile .shp):",
            font=ctk.CTkFont(size=10, weight="bold"),
            text_color="#CBD5E1",
            anchor="w",
        )
        self.lbl_train_yolo_shp.grid(row=row, column=0, columnspan=2, sticky="w", padx=12, pady=(4, 0))
        row += 1
        
        self.entry_train_yolo_shp = ctk.CTkEntry(
            self,
            placeholder_text="Path polygon .shp...",
            font=ctk.CTkFont(size=11),
            height=30,
            fg_color="#0F172A",
            border_color="#334155",
            text_color="#E2E8F0",
        )
        self.entry_train_yolo_shp.grid(row=row, column=0, sticky="ew", padx=(12, 4), pady=(0, 4))
        
        self.btn_browse_train_yolo_shp = ctk.CTkButton(
            self,
            text="📂",
            width=36,
            height=30,
            corner_radius=6,
            fg_color="#334155",
            hover_color="#475569",
            command=self._browse_train_yolo_shp,
        )
        self.btn_browse_train_yolo_shp.grid(row=row, column=1, sticky="w", padx=(0, 12), pady=(0, 4))
        row += 1
        
        self.btn_train_yolo_run = ctk.CTkButton(
            self,
            text="🎯  MULAI TRAINING YOLO",
            font=ctk.CTkFont(family="Segoe UI", size=12, weight="bold"),
            height=38,
            corner_radius=8,
            fg_color="#0EA5E9",
            hover_color="#0284C7",
            text_color="#FFFFFF",
            command=self._on_train_yolo_clicked,
        )
        self.btn_train_yolo_run.grid(row=row, column=0, columnspan=2, sticky="ew", padx=12, pady=(6, 4))
        row += 1
        
        self.lbl_train_yolo_status = ctk.CTkLabel(
            self,
            text="Status: Idle (Siap training)",
            font=ctk.CTkFont(size=10),
            text_color="#64748B",
            anchor="w",
            wraplength=180,
            justify="left",
        )
        self.lbl_train_yolo_status.grid(row=row, column=0, columnspan=2, sticky="ew", padx=12, pady=(2, 0))
        row += 1
        
        self.train_yolo_progress = ctk.CTkProgressBar(
            self,
            height=8,
            corner_radius=4,
            fg_color="#1E293B",
            progress_color="#0EA5E9",
        )
        self.train_yolo_progress.grid(row=row, column=0, columnspan=2, sticky="ew", padx=12, pady=(4, 12))
        self.train_yolo_progress.set(0)
        row += 1

        # ── 7. Footer ──────────────────────────────────
        ctk.CTkLabel(
            self,
            text="SAM-Geo Building Digitizer v1.2\nGPU: NVIDIA RTX 3050 (CUDA)",
            font=ctk.CTkFont(size=9),
            text_color="#1E293B",
            justify="center",
        ).grid(row=row, column=0, columnspan=2, pady=(8, 4))

    def _browse_input(self):
        path = filedialog.askopenfilename(
            title="Pilih File Raster",
            filetypes=[
                ("Raster Files", "*.ecw *.tif *.tiff *.geotiff"),
                ("ECW Files", "*.ecw"),
                ("GeoTIFF Files", "*.tif *.tiff"),
                ("All Files", "*.*"),
            ]
        )
        if path:
            self._input_path.set(path)
            display_name = os.path.basename(path)
            self.lbl_input.configure(text=display_name, text_color="#E2E8F0")
            # Try to load metadata
            self._load_raster_info(path)

    def _browse_output(self):
        path = filedialog.askdirectory(title="Pilih Folder Output")
        if path:
            self._output_dir.set(path)
            self.lbl_output.configure(text=path, text_color="#E2E8F0")

    def _load_raster_info(self, path: str):
        """Attempt to read and display raster metadata."""
        try:
            from core.ecw_handler import get_raster_info
            info = get_raster_info(path)
            self.info_labels["size"].configure(
                text=f"{info['width']:,} × {info['height']:,} px"
            )
            self.info_labels["crs"].configure(
                text=f"EPSG:{info['epsg']}" if info.get('epsg') else info.get('crs', '—')[:20]
            )
            res_x = info.get('resolution_x', 0)
            res_y = info.get('resolution_y', 0)
            self.info_labels["resolution"].configure(
                text=f"{res_x:.4f} × {res_y:.4f}"
            )
            self.info_labels["bands"].configure(
                text=f"{info['band_count']} ({info['dtype']})"
            )
        except Exception as e:
            for key in self.info_labels:
                self.info_labels[key].configure(text="N/A")

    def _on_run_clicked(self):
        if not self._input_path.get():
            messagebox.showwarning("Input Kosong", "Silakan pilih file ECW atau GeoTIFF terlebih dahulu.")
            return
        self.btn_run.configure(state="disabled", text="⏳  Memproses...")
        self.btn_stop.configure(state="normal")
        self.on_run(self.get_params())

    def _on_stop_clicked(self):
        self.btn_stop.configure(state="disabled")
        self.on_stop()

    def set_idle(self):
        """Reset button states after processing."""
        self.btn_run.configure(state="normal", text="▶  JALANKAN DIGITASI")
        self.btn_stop.configure(state="disabled")

    def get_params(self) -> dict:
        """Return all parameter values as a dict."""
        try:
            min_area = float(self.min_area_var.get())
        except ValueError:
            min_area = 20.0
        try:
            max_area = float(self.max_area_var.get())
        except ValueError:
            max_area = 100000.0

        # Parse points_per_side from selection string
        pts_str = self.pts_var.get()
        pts_val = 48
        if "32" in pts_str:
            pts_val = 32
        elif "64" in pts_str:
            pts_val = 64

        return {
            "input_path": self._input_path.get(),
            "output_dir": self._output_dir.get(),
            "model_name": self.model_var.get(),
            "mode": self.mode_var.get(),
            "yolo_model": self.yolo_var.get(),
            "show_yolo_preview": self.show_yolo_preview_var.get(),
            "enable_tile_filtering": self.filter_empty_var.get(),
            "tile_size": int(self.tile_var.get()),
            "min_area_m2": min_area,
            "max_area_m2": max_area,
            "points_per_side": pts_val,
            "enable_shadow_filter": self.shadow_var.get(),
            "enable_vegetation_filter": self.vegetation_var.get(),
            "enable_regularization": self.regularize_var.get(),
        }

    def _browse_train_geotiff(self):
        path = filedialog.askopenfilename(
            title="Pilih Citra GeoTIFF untuk Training",
            filetypes=[
                ("GeoTIFF Files", "*.tif *.tiff"),
                ("All Files", "*.*"),
            ]
        )
        if path:
            self.entry_train_geotiff.delete(0, tk.END)
            self.entry_train_geotiff.insert(0, path)
            
    def _browse_train_shp(self):
        path = filedialog.askopenfilename(
            title="Pilih Shapefile Bangunan untuk Training",
            filetypes=[
                ("Shapefile Files", "*.shp"),
                ("All Files", "*.*"),
            ]
        )
        if path:
            self.entry_train_shp.delete(0, tk.END)
            self.entry_train_shp.insert(0, path)

    def _on_train_clicked(self):
        geotiff = self.entry_train_geotiff.get().strip()
        shp = self.entry_train_shp.get().strip()
        
        if not geotiff or not shp:
            messagebox.showwarning("File Belum Lengkap", "Silakan pilih kedua file (Citra GeoTIFF & Shapefile) terlebih dahulu.")
            return
            
        if not os.path.exists(geotiff):
            messagebox.showerror("File Tidak Ditemukan", f"Citra GeoTIFF tidak ditemukan di path:\n{geotiff}")
            return
            
        if not os.path.exists(shp):
            messagebox.showerror("File Tidak Ditemukan", f"Shapefile tidak ditemukan di path:\n{shp}")
            return
            
        self.btn_train_run.configure(state="disabled", text="⏳  Training...")
        self.on_train(geotiff, shp)

    def set_training_progress(self, percent: float, status_msg: str):
        """Update progress bar and status text for training."""
        self.train_progress.set(percent / 100.0)
        self.lbl_train_status.configure(text=f"Status: {status_msg}", text_color="#F59E0B")
        
    def set_training_idle(self, final_msg: str = "Idle (Siap training)", is_success: bool = True):
        """Reset button states after training."""
        self.btn_train_run.configure(state="normal", text="🔥  MULAI TRAINING SAM")
        self.train_progress.set(1.0 if is_success else 0.0)
        self.lbl_train_status.configure(
            text=f"Status: {final_msg}",
            text_color="#10B981" if is_success else "#EF4444"
        )

    def _browse_train_yolo_geotiff(self):
        path = filedialog.askopenfilename(
            title="Pilih Citra GeoTIFF untuk Training YOLO",
            filetypes=[
                ("GeoTIFF Files", "*.tif *.tiff"),
                ("All Files", "*.*"),
            ]
        )
        if path:
            self.entry_train_yolo_geotiff.delete(0, tk.END)
            self.entry_train_yolo_geotiff.insert(0, path)
            
    def _browse_train_yolo_shp(self):
        path = filedialog.askopenfilename(
            title="Pilih Shapefile Bangunan untuk Training YOLO",
            filetypes=[
                ("Shapefile Files", "*.shp"),
                ("All Files", "*.*"),
            ]
        )
        if path:
            self.entry_train_yolo_shp.delete(0, tk.END)
            self.entry_train_yolo_shp.insert(0, path)

    def _on_train_yolo_clicked(self):
        geotiff = self.entry_train_yolo_geotiff.get().strip()
        shp = self.entry_train_yolo_shp.get().strip()
        
        if not geotiff or not shp:
            messagebox.showwarning("File Belum Lengkap", "Silakan pilih kedua file (Citra GeoTIFF & Shapefile) terlebih dahulu.")
            return
            
        if not os.path.exists(geotiff):
            messagebox.showerror("File Tidak Ditemukan", f"Citra GeoTIFF tidak ditemukan di path:\n{geotiff}")
            return
            
        if not os.path.exists(shp):
            messagebox.showerror("File Tidak Ditemukan", f"Shapefile tidak ditemukan di path:\n{shp}")
            return
            
        self.btn_train_yolo_run.configure(state="disabled", text="⏳  Training YOLO...")
        self.on_train_yolo(geotiff, shp)

    def set_training_yolo_progress(self, percent: float, status_msg: str):
        """Update progress bar and status text for YOLO training."""
        self.train_yolo_progress.set(percent / 100.0)
        self.lbl_train_yolo_status.configure(text=f"Status: {status_msg}", text_color="#F59E0B")
        
    def set_training_yolo_idle(self, final_msg: str = "Idle (Siap training)", is_success: bool = True):
        """Reset button states after YOLO training."""
        self.btn_train_yolo_run.configure(state="normal", text="🎯  MULAI TRAINING YOLO")
        self.train_yolo_progress.set(1.0 if is_success else 0.0)
        self.lbl_train_yolo_status.configure(
            text=f"Status: {final_msg}",
            text_color="#10B981" if is_success else "#EF4444"
        )

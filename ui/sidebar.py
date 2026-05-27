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

    def __init__(self, parent, on_run: Callable, on_stop: Callable, on_train: Callable, on_train_yolo: Callable, on_wms_ready: Optional[Callable] = None, on_yolo_toggle: Optional[Callable] = None, on_build_fingerprint: Optional[Callable] = None, **kwargs):
        super().__init__(parent, **kwargs)
        self.on_run = on_run
        self.on_stop = on_stop
        self.on_train = on_train
        self.on_train_yolo = on_train_yolo
        self.on_wms_ready = on_wms_ready
        self.on_yolo_toggle = on_yolo_toggle
        self.on_build_fingerprint = on_build_fingerprint
        self._input_path = tk.StringVar()
        self._output_dir = tk.StringVar(value=str(os.path.join(os.getcwd(), "output")))
        self._input_mode = tk.StringVar(value="📁 File Lokal")  # or "🌐 WMS Online"
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
            text="Digitasi Geospasial Otomatis",
            font=ctk.CTkFont(family="Segoe UI", size=10),
            text_color="#475569",
        ).grid(row=2, column=0, pady=(0, 12))

        row += 1; self._divider(row); row += 1

        # ── 1. Input File ──────────────────────────────
        self._section_label("📂  FILE INPUT", row); row += 1

        # Input mode toggle: File Lokal vs WMS Online
        self.input_mode_seg = ctk.CTkSegmentedButton(
            self,
            values=["📁 File Lokal", "🌐 WMS Online"],
            variable=self._input_mode,
            font=ctk.CTkFont(size=11), height=30,
            fg_color="#1E293B",
            selected_color="#6366F1",
            selected_hover_color="#4F46E5",
            unselected_color="#1E293B",
            unselected_hover_color="#334155",
            text_color="#E2E8F0",
            command=self._on_input_mode_changed,
        )
        self.input_mode_seg.grid(row=row, column=0, columnspan=2, sticky="ew", padx=12, pady=(0, 6))
        row += 1

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

        # Button: Pilih File (local mode)
        self.btn_browse_file = ctk.CTkButton(
            self,
            text="📂  Pilih File ECW / TIF",
            font=ctk.CTkFont(size=12, weight="bold"),
            height=38,
            corner_radius=8,
            fg_color="#6366F1",
            hover_color="#4F46E5",
            command=self._browse_input,
        )
        self.btn_browse_file.grid(row=row, column=0, columnspan=2, sticky="ew", padx=12, pady=(0, 4))

        # Button: Buka Panel WMS (online mode, hidden by default)
        self.btn_open_wms = ctk.CTkButton(
            self,
            text="🌐  Buka Panel WMS",
            font=ctk.CTkFont(size=12, weight="bold"),
            height=38,
            corner_radius=8,
            fg_color="#0EA5E9",
            hover_color="#0284C7",
            command=self._open_wms_panel,
        )
        # Hidden initially, shown only in WMS mode
        self.btn_open_wms.grid(row=row, column=0, columnspan=2, sticky="ew", padx=12, pady=(0, 4))
        self.btn_open_wms.grid_remove()
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

        # Dynamically find custom YOLO models in the models directory
        models_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
        custom_yolo_models = []
        if os.path.exists(models_dir):
            for f in os.listdir(models_dir):
                if f.startswith("yolo_") and f.endswith(".pt"):
                    custom_yolo_models.append(f"{f} (Custom)")
                    
        yolo_values = [
            "yolo12n.pt (YOLO12 Nano - Terbaru)",
            "yolo12s.pt (YOLO12 Small - Akurat)",
            "yolov8n.pt (YOLOv8 Nano - Klasik)", 
            "yolov8s.pt (YOLOv8 Small - Seimbang)", 
        ] + custom_yolo_models

        self.yolo_var = tk.StringVar(value="yolo12n.pt (YOLO12 Nano - Terbaru)")
        self.yolo_menu = ctk.CTkOptionMenu(
            self,
            values=yolo_values,
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
            command=self._on_yolo_switch,
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

        # Overlap Tile
        ctk.CTkLabel(
            self, text="Overlap Tile:",
            font=ctk.CTkFont(size=11), text_color="#CBD5E1", anchor="w",
        ).grid(row=row, column=0, sticky="w", padx=12, pady=(4, 0))

        self.overlap_var = tk.StringVar(value="25% (Aman)")
        self.overlap_menu = ctk.CTkOptionMenu(
            self,
            values=["12.5% (Cepat)", "25% (Aman)", "50% (Sangat Aman)"],
            variable=self.overlap_var,
            font=ctk.CTkFont(size=11),
            height=28,
            fg_color="#1E293B",
            button_color="#334155",
            button_hover_color="#475569",
            dropdown_fg_color="#1E293B",
            dropdown_hover_color="#334155",
            text_color="#E2E8F0",
        )
        self.overlap_menu.grid(row=row, column=1, sticky="ew", padx=(4, 12), pady=(4, 0))
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

        # Toleransi Simplifikasi
        ctk.CTkLabel(
            self, text="Simplifikasi Garis (m):",
            font=ctk.CTkFont(size=11), text_color="#CBD5E1", anchor="w",
        ).grid(row=row, column=0, sticky="w", padx=12, pady=(4, 0))

        self.simplify_var = ctk.CTkEntry(
            self, placeholder_text="0.75", width=80, height=24,
            font=ctk.CTkFont(size=11),
            fg_color="#1E293B", border_color="#334155", text_color="#E2E8F0",
        )
        self.simplify_var.grid(row=row, column=1, sticky="ew", padx=(4, 12), pady=(4, 0))
        self.simplify_var.insert(0, "0.75")
        row += 1

        # YOLO Confidence
        ctk.CTkLabel(
            self, text="YOLO Confidence (%):",
            font=ctk.CTkFont(size=11), text_color="#CBD5E1", anchor="w",
        ).grid(row=row, column=0, sticky="w", padx=12, pady=(4, 0))

        self.yolo_conf_var = ctk.CTkEntry(
            self, placeholder_text="15", width=80, height=24,
            font=ctk.CTkFont(size=11),
            fg_color="#1E293B", border_color="#334155", text_color="#E2E8F0",
        )
        self.yolo_conf_var.grid(row=row, column=1, sticky="ew", padx=(4, 12), pady=(4, 0))
        self.yolo_conf_var.insert(0, "15")
        row += 1

        # ── 5b. Objek Digitasi ─────────────────────────────
        row += 1; self._divider(row); row += 1
        self._section_label("🗺️  OBJEK DIGITASI", row); row += 1

        info_lbl = ctk.CTkLabel(
            self,
            text="Pilih objek yang akan didigitasi oleh sistem:",
            font=ctk.CTkFont(size=10),
            text_color="#64748B",
            anchor="w",
            wraplength=200,
            justify="left",
        )
        info_lbl.grid(row=row, column=0, columnspan=2, sticky="ew", padx=12, pady=(0, 6))
        row += 1

        # Checkbox: Bangunan (default ON) — mode ikuti Mode Digitasi utama
        self.obj_building_var = tk.BooleanVar(value=True)
        self.chk_building = ctk.CTkCheckBox(
            self,
            text="🏠  Bangunan",
            variable=self.obj_building_var,
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color="#00D4FF",
            checkmark_color="#0F172A",
            fg_color="#00D4FF",
            hover_color="#00B8D9",
            border_color="#334155",
            height=26,
        )
        self.chk_building.grid(row=row, column=0, columnspan=2, sticky="w", padx=12, pady=(2, 0))
        row += 1

        # Label mode bangunan (ikuti dropdown Mode Digitasi di atas)
        ctk.CTkLabel(
            self, text="   Mode: ikuti pilihan Mode Digitasi",
            font=ctk.CTkFont(size=9), text_color="#475569", anchor="w",
        ).grid(row=row, column=0, columnspan=2, sticky="w", padx=12, pady=(0, 4))
        row += 1

        # ── Jalan & Infrastruktur ────────────────────────────────
        self.obj_road_var = tk.BooleanVar(value=False)
        self.chk_road = ctk.CTkCheckBox(
            self,
            text="🛣️  Jalan & Infrastruktur",
            variable=self.obj_road_var,
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color="#FF6B35",
            checkmark_color="#0F172A",
            fg_color="#FF6B35",
            hover_color="#E55A25",
            border_color="#334155",
            height=26,
            command=self._on_road_toggled,
        )
        self.chk_road.grid(row=row, column=0, columnspan=2, sticky="w", padx=12, pady=(6, 0))
        row += 1

        # Mode dropdown
        ctk.CTkLabel(
            self, text="   Mode:",
            font=ctk.CTkFont(size=10), text_color="#64748B", anchor="w",
        ).grid(row=row, column=0, sticky="w", padx=(24, 0), pady=(0, 2))

        self.road_mode_var = tk.StringVar(value="Road Fingerprint (Adaptif)")
        self._road_mode_menu = ctk.CTkOptionMenu(
            self,
            values=["Road Fingerprint (Adaptif)", "Pra-Deteksi Warna (Lama)"],
            variable=self.road_mode_var,
            font=ctk.CTkFont(size=10),
            height=24,
            fg_color="#1E293B",
            button_color="#334155",
            button_hover_color="#475569",
            dropdown_fg_color="#1E293B",
            dropdown_hover_color="#334155",
            text_color="#CBD5E1",
            command=self._on_road_mode_changed,
        )
        self._road_mode_menu.grid(row=row, column=1, sticky="ew", padx=(0, 12), pady=(0, 2))
        row += 1

        # ── Sub-panel Road Fingerprint ───────────────────────────
        self._road_fp_frame = ctk.CTkFrame(
            self, fg_color="#0F172A", corner_radius=8
        )
        self._road_fp_frame.grid(
            row=row, column=0, columnspan=2, sticky="ew", padx=12, pady=(0, 6)
        )
        self._road_fp_frame.grid_columnconfigure(0, weight=1)
        row += 1

        # Toggle: Buat Baru / Load Existing
        self.road_fp_source_var = tk.StringVar(value="🔬 Buat Baru")
        self._road_fp_seg = ctk.CTkSegmentedButton(
            self._road_fp_frame,
            values=["🔬 Buat Baru", "📂 Load Existing"],
            variable=self.road_fp_source_var,
            font=ctk.CTkFont(size=10),
            height=26,
            fg_color="#1E293B",
            selected_color="#FF6B35",
            selected_hover_color="#E55A25",
            unselected_color="#1E293B",
            unselected_hover_color="#334155",
            text_color="#E2E8F0",
            command=self._on_road_fp_source_changed,
        )
        self._road_fp_seg.grid(row=0, column=0, sticky="ew", padx=8, pady=(8, 4))

        # ── Buat Baru — form ──────────────────────────────────────
        self._road_build_frame = ctk.CTkFrame(
            self._road_fp_frame, fg_color="transparent"
        )
        self._road_build_frame.grid(row=1, column=0, sticky="ew", padx=4, pady=2)
        self._road_build_frame.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(
            self._road_build_frame,
            text="Citra Referensi (.tif / .ecw):",
            font=ctk.CTkFont(size=9, weight="bold"),
            text_color="#94A3B8", anchor="w",
        ).grid(row=0, column=0, columnspan=2, sticky="w", padx=4, pady=(4, 0))

        self.road_ref_raster_var = tk.StringVar()
        self._road_ref_raster_entry = ctk.CTkEntry(
            self._road_build_frame,
            textvariable=self.road_ref_raster_var,
            placeholder_text="Path citra referensi ...",
            font=ctk.CTkFont(size=10),
            height=26,
            fg_color="#0F172A",
            border_color="#334155",
            text_color="#E2E8F0",
        )
        self._road_ref_raster_entry.grid(row=1, column=0, sticky="ew", padx=(4, 2), pady=1)

        ctk.CTkButton(
            self._road_build_frame,
            text="📂", width=30, height=26,
            corner_radius=6, fg_color="#334155", hover_color="#475569",
            command=self._browse_road_ref_raster,
        ).grid(row=1, column=1, sticky="w", padx=(0, 4), pady=1)

        ctk.CTkLabel(
            self._road_build_frame,
            text="Polygon Jalan Referensi (.shp):",
            font=ctk.CTkFont(size=9, weight="bold"),
            text_color="#94A3B8", anchor="w",
        ).grid(row=2, column=0, columnspan=2, sticky="w", padx=4, pady=(4, 0))

        self.road_ref_shp_var = tk.StringVar()
        self._road_ref_shp_entry = ctk.CTkEntry(
            self._road_build_frame,
            textvariable=self.road_ref_shp_var,
            placeholder_text="Path polygon jalan .shp ...",
            font=ctk.CTkFont(size=10),
            height=26,
            fg_color="#0F172A",
            border_color="#334155",
            text_color="#E2E8F0",
        )
        self._road_ref_shp_entry.grid(row=3, column=0, sticky="ew", padx=(4, 2), pady=1)

        ctk.CTkButton(
            self._road_build_frame,
            text="📂", width=30, height=26,
            corner_radius=6, fg_color="#334155", hover_color="#475569",
            command=self._browse_road_ref_shp,
        ).grid(row=3, column=1, sticky="w", padx=(0, 4), pady=1)

        self.btn_build_fingerprint = ctk.CTkButton(
            self._road_build_frame,
            text="🔬  BUAT FINGERPRINT",
            font=ctk.CTkFont(size=11, weight="bold"),
            height=32, corner_radius=7,
            fg_color="#FF6B35", hover_color="#E55A25",
            text_color="#FFFFFF",
            command=self._on_build_fingerprint_clicked,
        )
        self.btn_build_fingerprint.grid(
            row=4, column=0, columnspan=2, sticky="ew", padx=4, pady=(6, 2)
        )

        self.lbl_fp_status = ctk.CTkLabel(
            self._road_build_frame,
            text="Status: Belum ada fingerprint",
            font=ctk.CTkFont(size=9),
            text_color="#64748B", anchor="w", wraplength=180,
        )
        self.lbl_fp_status.grid(row=5, column=0, columnspan=2, sticky="ew", padx=4, pady=(2, 0))

        self.fp_progress = ctk.CTkProgressBar(
            self._road_build_frame,
            height=6, corner_radius=3,
            fg_color="#1E293B", progress_color="#FF6B35",
        )
        self.fp_progress.grid(
            row=6, column=0, columnspan=2, sticky="ew", padx=4, pady=(2, 6)
        )
        self.fp_progress.set(0)

        # ── Load Existing — form ──────────────────────────────────
        self._road_load_frame = ctk.CTkFrame(
            self._road_fp_frame, fg_color="transparent"
        )
        self._road_load_frame.grid(row=2, column=0, sticky="ew", padx=4, pady=2)
        self._road_load_frame.grid_columnconfigure(0, weight=1)
        self._road_load_frame.grid_remove()   # Hidden by default

        ctk.CTkLabel(
            self._road_load_frame,
            text="File Fingerprint (.json):",
            font=ctk.CTkFont(size=9, weight="bold"),
            text_color="#94A3B8", anchor="w",
        ).grid(row=0, column=0, columnspan=2, sticky="w", padx=4, pady=(4, 0))

        self.road_fp_path_var = tk.StringVar()
        self._road_fp_path_entry = ctk.CTkEntry(
            self._road_load_frame,
            textvariable=self.road_fp_path_var,
            placeholder_text="Path road_fingerprint.json ...",
            font=ctk.CTkFont(size=10),
            height=26,
            fg_color="#0F172A",
            border_color="#334155",
            text_color="#E2E8F0",
        )
        self._road_fp_path_entry.grid(row=1, column=0, sticky="ew", padx=(4, 2), pady=(2, 6))

        ctk.CTkButton(
            self._road_load_frame,
            text="📂", width=30, height=26,
            corner_radius=6, fg_color="#334155", hover_color="#475569",
            command=self._browse_road_fp_json,
        ).grid(row=1, column=1, sticky="w", padx=(0, 4), pady=(2, 6))

        # ── Badan Air ───────────────────────────────────────────
        self.obj_water_var = tk.BooleanVar(value=False)
        self.chk_water = ctk.CTkCheckBox(
            self,
            text="💧  Badan Air",
            variable=self.obj_water_var,
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color="#3B82F6",
            checkmark_color="#0F172A",
            fg_color="#3B82F6",
            hover_color="#2563EB",
            border_color="#334155",
            height=26,
        )
        self.chk_water.grid(row=row, column=0, columnspan=2, sticky="w", padx=12, pady=(6, 0))
        row += 1

        ctk.CTkLabel(
            self, text="   Mode:",
            font=ctk.CTkFont(size=10), text_color="#64748B", anchor="w",
        ).grid(row=row, column=0, sticky="w", padx=(24, 0), pady=(0, 4))

        self.water_mode_var = tk.StringVar(value="SegFormer (Rekomendasi)")
        ctk.CTkOptionMenu(
            self,
            values=["SegFormer (Rekomendasi)", "Pra-Deteksi Warna", "Pra-Deteksi (YOLO Bounding Box)", "Otomatis SAM (Grid)"],
            variable=self.water_mode_var,
            font=ctk.CTkFont(size=10),
            height=24,
            fg_color="#1E293B",
            button_color="#334155",
            button_hover_color="#475569",
            dropdown_fg_color="#1E293B",
            dropdown_hover_color="#334155",
            text_color="#CBD5E1",
        ).grid(row=row, column=1, sticky="ew", padx=(0, 12), pady=(0, 4))
        row += 1

        ctk.CTkLabel(
            self, text="   Model YOLO:",
            font=ctk.CTkFont(size=10), text_color="#64748B", anchor="w",
        ).grid(row=row, column=0, sticky="w", padx=(24, 0), pady=(0, 4))

        self.water_yolo_var = tk.StringVar(value=yolo_values[0])
        ctk.CTkOptionMenu(
            self,
            values=yolo_values,
            variable=self.water_yolo_var,
            font=ctk.CTkFont(size=10),
            height=24,
            fg_color="#1E293B",
            button_color="#334155",
            button_hover_color="#475569",
            dropdown_fg_color="#1E293B",
            dropdown_hover_color="#334155",
            text_color="#CBD5E1",
        ).grid(row=row, column=1, sticky="ew", padx=(0, 12), pady=(0, 4))
        row += 1

        # ── Vegetasi ─────────────────────────────────────────────
        self.obj_veg_var = tk.BooleanVar(value=False)
        self.chk_veg = ctk.CTkCheckBox(
            self,
            text="🌿  Vegetasi",
            variable=self.obj_veg_var,
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color="#22C55E",
            checkmark_color="#0F172A",
            fg_color="#22C55E",
            hover_color="#16A34A",
            border_color="#334155",
            height=26,
        )
        self.chk_veg.grid(row=row, column=0, columnspan=2, sticky="w", padx=12, pady=(6, 0))
        row += 1

        ctk.CTkLabel(
            self, text="   Mode:",
            font=ctk.CTkFont(size=10), text_color="#64748B", anchor="w",
        ).grid(row=row, column=0, sticky="w", padx=(24, 0), pady=(0, 4))

        self.veg_mode_var = tk.StringVar(value="SegFormer (Rekomendasi)")
        ctk.CTkOptionMenu(
            self,
            values=["SegFormer (Rekomendasi)", "Pra-Deteksi Warna", "Pra-Deteksi (YOLO Bounding Box)", "Otomatis SAM (Grid)"],
            variable=self.veg_mode_var,
            font=ctk.CTkFont(size=10),
            height=24,
            fg_color="#1E293B",
            button_color="#334155",
            button_hover_color="#475569",
            dropdown_fg_color="#1E293B",
            dropdown_hover_color="#334155",
            text_color="#CBD5E1",
        ).grid(row=row, column=1, sticky="ew", padx=(0, 12), pady=(0, 6))
        row += 1

        ctk.CTkLabel(
            self, text="   Model YOLO:",
            font=ctk.CTkFont(size=10), text_color="#64748B", anchor="w",
        ).grid(row=row, column=0, sticky="w", padx=(24, 0), pady=(0, 6))

        self.veg_yolo_var = tk.StringVar(value=yolo_values[0])
        ctk.CTkOptionMenu(
            self,
            values=yolo_values,
            variable=self.veg_yolo_var,
            font=ctk.CTkFont(size=10),
            height=24,
            fg_color="#1E293B",
            button_color="#334155",
            button_hover_color="#475569",
            dropdown_fg_color="#1E293B",
            dropdown_hover_color="#334155",
            text_color="#CBD5E1",
        ).grid(row=row, column=1, sticky="ew", padx=(0, 12), pady=(0, 6))
        row += 1

        # ── 6. Action Buttons ──────────────────────────────
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

        # ── 6c. Training Model YOLO ──────────────────
        row += 1; self._divider(row); row += 1
        self._section_label("🎯  TRAINING MODEL YOLO", row); row += 1

        self.lbl_train_yolo_base = ctk.CTkLabel(
            self,
            text="Base Model (Mulai dari mana):",
            font=ctk.CTkFont(size=10, weight="bold"),
            text_color="#CBD5E1",
            anchor="w",
        )
        self.lbl_train_yolo_base.grid(row=row, column=0, columnspan=2, sticky="w", padx=12, pady=(4, 0))
        row += 1
        
        self.train_yolo_base_var = tk.StringVar(value="Otomatis (Lanjutkan jika ada)")
        self.train_yolo_base_menu = ctk.CTkOptionMenu(
            self,
            values=["Otomatis (Lanjutkan jika ada)"] + yolo_values,
            variable=self.train_yolo_base_var,
            font=ctk.CTkFont(size=11),
            height=28,
            fg_color="#1E293B",
            button_color="#334155",
            button_hover_color="#475569",
            dropdown_fg_color="#1E293B",
            dropdown_hover_color="#334155",
            text_color="#E2E8F0",
        )
        self.train_yolo_base_menu.grid(row=row, column=0, columnspan=2, sticky="ew", padx=12, pady=(0, 4))
        row += 1
        
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
        
        self.lbl_train_yolo_shp_b = ctk.CTkLabel(
            self,
            text="2. Shapefile Bangunan (Opsional):",
            font=ctk.CTkFont(size=10, weight="bold"),
            text_color="#CBD5E1",
            anchor="w",
        )
        self.lbl_train_yolo_shp_b.grid(row=row, column=0, columnspan=2, sticky="w", padx=12, pady=(4, 0))
        row += 1
        
        self.entry_train_yolo_shp_b = ctk.CTkEntry(
            self,
            placeholder_text="Path polygon bangunan .shp...",
            font=ctk.CTkFont(size=11),
            height=30,
            fg_color="#0F172A",
            border_color="#334155",
            text_color="#E2E8F0",
        )
        self.entry_train_yolo_shp_b.grid(row=row, column=0, sticky="ew", padx=(12, 4), pady=(0, 4))
        
        self.btn_browse_train_yolo_shp_b = ctk.CTkButton(
            self,
            text="📂",
            width=36,
            height=30,
            corner_radius=6,
            fg_color="#334155",
            hover_color="#475569",
            command=lambda: self._browse_train_yolo_shp_multi(self.entry_train_yolo_shp_b, "Bangunan"),
        )
        self.btn_browse_train_yolo_shp_b.grid(row=row, column=1, sticky="w", padx=(0, 12), pady=(0, 4))
        row += 1

        self.lbl_train_yolo_shp_j = ctk.CTkLabel(
            self,
            text="3. Shapefile Jalan (Opsional):",
            font=ctk.CTkFont(size=10, weight="bold"),
            text_color="#CBD5E1",
            anchor="w",
        )
        self.lbl_train_yolo_shp_j.grid(row=row, column=0, columnspan=2, sticky="w", padx=12, pady=(4, 0))
        row += 1
        
        self.entry_train_yolo_shp_j = ctk.CTkEntry(
            self,
            placeholder_text="Path polygon jalan .shp...",
            font=ctk.CTkFont(size=11),
            height=30,
            fg_color="#0F172A",
            border_color="#334155",
            text_color="#E2E8F0",
        )
        self.entry_train_yolo_shp_j.grid(row=row, column=0, sticky="ew", padx=(12, 4), pady=(0, 4))
        
        self.btn_browse_train_yolo_shp_j = ctk.CTkButton(
            self,
            text="📂",
            width=36,
            height=30,
            corner_radius=6,
            fg_color="#334155",
            hover_color="#475569",
            command=lambda: self._browse_train_yolo_shp_multi(self.entry_train_yolo_shp_j, "Jalan"),
        )
        self.btn_browse_train_yolo_shp_j.grid(row=row, column=1, sticky="w", padx=(0, 12), pady=(0, 4))
        row += 1
        
        self.lbl_train_yolo_shp_l = ctk.CTkLabel(
            self,
            text="4. Shapefile Lainnya (Opsional):",
            font=ctk.CTkFont(size=10, weight="bold"),
            text_color="#CBD5E1",
            anchor="w",
        )
        self.lbl_train_yolo_shp_l.grid(row=row, column=0, columnspan=2, sticky="w", padx=12, pady=(4, 0))
        row += 1
        
        self.entry_train_yolo_shp_l = ctk.CTkEntry(
            self,
            placeholder_text="Path polygon lainnya .shp...",
            font=ctk.CTkFont(size=11),
            height=30,
            fg_color="#0F172A",
            border_color="#334155",
            text_color="#E2E8F0",
        )
        self.entry_train_yolo_shp_l.grid(row=row, column=0, sticky="ew", padx=(12, 4), pady=(0, 10))
        
        self.btn_browse_train_yolo_shp_l = ctk.CTkButton(
            self,
            text="📂",
            width=36,
            height=30,
            corner_radius=6,
            fg_color="#334155",
            hover_color="#475569",
            command=lambda: self._browse_train_yolo_shp_multi(self.entry_train_yolo_shp_l, "Lainnya"),
        )
        self.btn_browse_train_yolo_shp_l.grid(row=row, column=1, sticky="w", padx=(0, 12), pady=(0, 10))
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

        # Sub-panel fingerprint jalan — hidden by default, tampil saat checkbox Jalan dicentang
        self._road_fp_frame.grid_remove()

    def _on_input_mode_changed(self, value: str):
        """Toggle between local file and WMS online input modes."""
        if value == "🌐 WMS Online":
            self.btn_browse_file.grid_remove()
            self.btn_open_wms.grid()
            self.lbl_input.configure(
                text="Pilih sumber WMS dan gambar area AOI",
                text_color="#0EA5E9",
            )
            for key in self.info_labels:
                self.info_labels[key].configure(text="—")
        else:
            self.btn_open_wms.grid_remove()
            self.btn_browse_file.grid()
            if not self._input_path.get():
                self.lbl_input.configure(
                    text="Belum ada file dipilih",
                    text_color="#64748B",
                )

    def _open_wms_panel(self):
        """Open the WMS panel dialog."""
        from ui.wms_panel import WmsPanel
        output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "temp")
        os.makedirs(output_dir, exist_ok=True)

        panel = WmsPanel(
            parent=self.winfo_toplevel(),
            output_dir=output_dir,
            on_ready_callback=self._on_wms_geotiff_ready,
            log_callback=None,  # Will be set by app.py if needed
        )

    def _on_wms_geotiff_ready(self, geotiff_path: str):
        """Called when WMS AOI download is complete."""
        self._input_path.set(geotiff_path)
        display_name = os.path.basename(geotiff_path)
        self.lbl_input.configure(
            text=f"🌐 {display_name}",
            text_color="#0EA5E9",
        )
        self._load_raster_info(geotiff_path)
        if self.on_wms_ready:
            self.on_wms_ready(geotiff_path)

    def _on_yolo_switch(self):
        """Called when YOLO preview toggle is clicked."""
        if getattr(self, "on_yolo_toggle", None):
            self.on_yolo_toggle(self.show_yolo_preview_var.get())

    def set_wms_geotiff(self, geotiff_path: str):
        """Programmatically set the WMS-downloaded GeoTIFF as input."""
        self._on_wms_geotiff_ready(geotiff_path)

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
        try:
            simplify_tol = float(self.simplify_var.get())
        except Exception:
            simplify_tol = 0.75
            
        try:
            yolo_conf = float(self.yolo_conf_var.get()) / 100.0
            yolo_conf = max(0.01, min(0.99, yolo_conf))
        except Exception:
            yolo_conf = 0.15

        # Parse points_per_side from selection string
        pts_str = self.pts_var.get()
        pts_val = 48
        if "32" in pts_str:
            pts_val = 32
        elif "64" in pts_str:
            pts_val = 64
            
        # Parse overlap string
        overlap_str = self.overlap_var.get()
        overlap_ratio = 0.25
        if "12.5" in overlap_str:
            overlap_ratio = 0.125
        elif "50" in overlap_str:
            overlap_ratio = 0.5

        # ── Resolve road fingerprint path ───────────────────────
        fp_source = self.road_fp_source_var.get()
        if fp_source == "🔬 Buat Baru":
            # Fingerprint akan dibuat saat run — path default di output_dir
            fp_json_path = os.path.join(
                self._output_dir.get(), "road_fingerprint.json"
            )
        else:
            fp_json_path = self.road_fp_path_var.get().strip()

        return {
            "input_path": self._input_path.get(),
            "output_dir": self._output_dir.get(),
            "model_name": self.model_var.get(),
            "mode": self.mode_var.get(),
            "yolo_model": self.yolo_var.get(),
            "yolo_conf": yolo_conf,
            "show_yolo_preview": self.show_yolo_preview_var.get(),
            "enable_tile_filtering": self.filter_empty_var.get(),
            "tile_size": int(self.tile_var.get()),
            "overlap_ratio": overlap_ratio,
            "min_area_m2": min_area,
            "max_area_m2": max_area,
            "points_per_side": pts_val,
            "enable_shadow_filter": self.shadow_var.get(),
            "enable_vegetation_filter": self.vegetation_var.get(),
            "enable_regularization": self.regularize_var.get(),
            "simplify_tolerance": simplify_tol,
            # ── Objek Digitasi ──────────────────────────────────
            "enabled_objects": {
                "building":   self.obj_building_var.get(),
                "road":       self.obj_road_var.get(),
                "water":      self.obj_water_var.get(),
                "vegetation": self.obj_veg_var.get(),
            },
            # ── Mode per objek non-bangunan ────────────────────────
            "object_modes": {
                "road":       self.road_mode_var.get(),
                "water":      self.water_mode_var.get(),
                "vegetation": self.veg_mode_var.get(),
            },
            "object_yolo_models": {
                "water":      self.water_yolo_var.get(),
                "vegetation": self.veg_yolo_var.get(),
            },
            # ── Road Fingerprint config ─────────────────────────
            "road_config": {
                "mode":             "fingerprint" if "Fingerprint" in self.road_mode_var.get() else "color_predetect",
                "fp_source":        fp_source,
                "fingerprint_path": fp_json_path,
                "ref_raster":       self.road_ref_raster_var.get().strip(),
                "ref_shp":          self.road_ref_shp_var.get().strip(),
            },
        }

    # ══════════════════════════════════════════════════════════════
    # Road Fingerprint Event Handlers
    # ══════════════════════════════════════════════════════════════

    def _on_road_toggled(self):
        """Show/hide fingerprint sub-panel when Jalan checkbox is toggled."""
        if self.obj_road_var.get():
            self._road_fp_frame.grid()
        else:
            self._road_fp_frame.grid_remove()

    def _on_road_mode_changed(self, value: str):
        """Show/hide fingerprint sub-panel based on road mode."""
        if "Fingerprint" in value:
            self._road_fp_frame.grid()
        else:
            self._road_fp_frame.grid_remove()

    def _on_road_fp_source_changed(self, value: str):
        """Toggle between Buat Baru and Load Existing sub-forms."""
        if value == "🔬 Buat Baru":
            self._road_build_frame.grid()
            self._road_load_frame.grid_remove()
        else:
            self._road_build_frame.grid_remove()
            self._road_load_frame.grid()

    def _browse_road_ref_raster(self):
        """Browse for reference raster file."""
        path = filedialog.askopenfilename(
            title="Pilih Citra Referensi (ECW / GeoTIFF)",
            filetypes=[
                ("Raster Files", "*.ecw *.tif *.tiff *.geotiff"),
                ("All Files", "*.*"),
            ]
        )
        if path:
            self.road_ref_raster_var.set(path)

    def _browse_road_ref_shp(self):
        """Browse for reference road polygon SHP."""
        path = filedialog.askopenfilename(
            title="Pilih Polygon Jalan Referensi (Shapefile)",
            filetypes=[
                ("Shapefile", "*.shp"),
                ("All Files", "*.*"),
            ]
        )
        if path:
            self.road_ref_shp_var.set(path)

    def _browse_road_fp_json(self):
        """Browse for existing fingerprint JSON or YOLO pt."""
        path = filedialog.askopenfilename(
            title="Pilih File Road Fingerprint (.json) atau YOLO (.pt)",
            filetypes=[
                ("Model / Fingerprint", "*.pt *.json"),
                ("YOLO Model", "*.pt"),
                ("JSON Files", "*.json"),
                ("All Files", "*.*"),
            ]
        )
        if path:
            self.road_fp_path_var.set(path)
            self.lbl_fp_status.configure(
                text=f"✅ Fingerprint dimuat: {os.path.basename(path)}",
                text_color="#10B981",
            )
            self.fp_progress.set(1.0)

    def _on_build_fingerprint_clicked(self):
        """Validate inputs and trigger fingerprint building (Fase 1-4)."""
        ref_raster = self.road_ref_raster_var.get().strip()
        ref_shp    = self.road_ref_shp_var.get().strip()

        if not ref_raster:
            messagebox.showwarning(
                "Input Kosong",
                "Silakan pilih Citra Referensi terlebih dahulu."
            )
            return
        if not ref_shp:
            messagebox.showwarning(
                "Input Kosong",
                "Silakan pilih Polygon Jalan Referensi (.shp) terlebih dahulu."
            )
            return
        if not os.path.exists(ref_raster):
            messagebox.showerror(
                "File Tidak Ditemukan",
                f"Citra referensi tidak ditemukan:\n{ref_raster}"
            )
            return
        if not os.path.exists(ref_shp):
            messagebox.showerror(
                "File Tidak Ditemukan",
                f"Shapefile tidak ditemukan:\n{ref_shp}"
            )
            return

        # Output path: di folder output
        output_path = os.path.join(
            self._output_dir.get(), "road_fingerprint.json"
        )

        # Trigger via callback (akan ditangkap app.py)
        self.btn_build_fingerprint.configure(
            state="disabled", text="⏳  Membangun Fingerprint..."
        )
        if hasattr(self, "on_build_fingerprint") and self.on_build_fingerprint:
            self.on_build_fingerprint(ref_raster, ref_shp, output_path)
        else:
            # Fallback: jalankan langsung di thread
            import threading
            from core.objects.road_fingerprint import RoadFingerprintBuilder

            def _run():
                builder = RoadFingerprintBuilder(
                    log_callback=lambda m: print(m),
                    progress_callback=lambda p, m: self.after(
                        0, lambda: self.set_fingerprint_progress(p, m)
                    ),
                )
                try:
                    builder.build(ref_raster, ref_shp, output_path)
                    self.after(0, lambda: self.set_fingerprint_idle(
                        f"✅ Fingerprint disimpan: {os.path.basename(output_path)}",
                        is_success=True
                    ))
                    self.after(0, lambda: self.road_fp_path_var.set(output_path))
                except Exception as exc:
                    self.after(0, lambda: self.set_fingerprint_idle(
                        f"❌ Gagal: {exc}", is_success=False
                    ))

            threading.Thread(target=_run, daemon=True).start()

    def set_fingerprint_progress(self, percent: float, msg: str):
        """Update fingerprint build progress (called from worker thread via .after)."""
        self.fp_progress.set(percent / 100.0)
        self.lbl_fp_status.configure(
            text=f"⏳ {msg}", text_color="#F59E0B"
        )

    def set_fingerprint_idle(self, final_msg: str, is_success: bool = True):
        """Reset fingerprint button after build completes or fails."""
        self.btn_build_fingerprint.configure(
            state="normal", text="🔬  BUAT FINGERPRINT"
        )
        self.fp_progress.set(1.0 if is_success else 0.0)
        self.lbl_fp_status.configure(
            text=final_msg,
            text_color="#10B981" if is_success else "#EF4444",
        )

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
            
    def _browse_train_yolo_shp_multi(self, entry_widget, target_name):
        path = filedialog.askopenfilename(
            title=f"Pilih Shapefile {target_name} untuk Training YOLO",
            filetypes=[
                ("Shapefile Files", "*.shp"),
                ("All Files", "*.*"),
            ]
        )
        if path:
            entry_widget.delete(0, tk.END)
            entry_widget.insert(0, path)

    def _on_train_yolo_clicked(self):
        geotiff = self.entry_train_yolo_geotiff.get().strip()
        shp_b = self.entry_train_yolo_shp_b.get().strip()
        shp_j = self.entry_train_yolo_shp_j.get().strip()
        shp_l = self.entry_train_yolo_shp_l.get().strip()
        
        if not geotiff:
            messagebox.showwarning("File Belum Lengkap", "Silakan pilih Citra GeoTIFF terlebih dahulu.")
            return
            
        if not shp_b and not shp_j and not shp_l:
            messagebox.showwarning("File Belum Lengkap", "Silakan isi minimal satu Shapefile (Bangunan, Jalan, atau Lainnya).")
            return
            
        if not os.path.exists(geotiff):
            messagebox.showerror("File Tidak Ditemukan", f"Citra GeoTIFF tidak ditemukan di path:\n{geotiff}")
            return
            
        shp_paths = {}
        if shp_b:
            if not os.path.exists(shp_b):
                messagebox.showerror("File Tidak Ditemukan", f"Shapefile Bangunan tidak ditemukan di path:\n{shp_b}")
                return
            shp_paths["bangunan"] = shp_b
            
        if shp_j:
            if not os.path.exists(shp_j):
                messagebox.showerror("File Tidak Ditemukan", f"Shapefile Jalan tidak ditemukan di path:\n{shp_j}")
                return
            shp_paths["jalan"] = shp_j
            
        if shp_l:
            if not os.path.exists(shp_l):
                messagebox.showerror("File Tidak Ditemukan", f"Shapefile Lainnya tidak ditemukan di path:\n{shp_l}")
                return
            shp_paths["lainnya"] = shp_l
            
        self.btn_train_yolo_run.configure(state="disabled", text="⏳  Training YOLO...")
        self.on_train_yolo(geotiff, shp_paths)

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

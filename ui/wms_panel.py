"""
wms_panel.py
Panel interaktif untuk memilih sumber WMS dan menggambar Area of Interest (AOI).
Menggunakan tkintermapview untuk menampilkan peta tile dan mendukung:
- Gambar Rectangle AOI (klik + drag)
- Gambar Polygon AOI bebas (klik titik-titik, double-click untuk menutup)
"""

import os
import threading
import tkinter as tk
import customtkinter as ctk
from tkinter import messagebox
from typing import Callable, List, Optional, Tuple

from core.wms_downloader import (
    WMS_PRESETS,
    estimate_tile_count,
    estimate_download_size_mb,
    get_best_zoom,
    download_wms_aoi,
    get_aoi_bounds_from_polygon,
)


class WmsPanel(ctk.CTkToplevel):
    """
    Dialog window untuk memilih WMS dan menggambar AOI.
    Setelah download selesai, memanggil on_ready_callback(geotiff_path).
    """

    def __init__(
        self,
        parent,
        output_dir: str,
        on_ready_callback: Callable[[str], None],
        log_callback: Optional[Callable] = None,
    ):
        super().__init__(parent)
        self.output_dir = output_dir
        self.on_ready = on_ready_callback
        self.log_callback = log_callback

        self._aoi_mode = tk.StringVar(value="rectangle")  # "rectangle" or "polygon"
        self._draw_mode_active = False
        self._rect_start = None
        self._polygon_points: List[Tuple[float, float]] = []
        self._current_aoi: Optional[List[Tuple[float, float]]] = None  # polygon AOI in (lon, lat)
        self._current_aoi_type = None  # "rectangle" or "polygon"
        self._map_markers = []
        self._map_polygon = None
        self._download_thread: Optional[threading.Thread] = None

        # Preset + custom URL
        self._preset_names = list(WMS_PRESETS.keys()) + ["URL Kustom..."]
        self._selected_preset = tk.StringVar(value=list(WMS_PRESETS.keys())[0])
        self._custom_url = tk.StringVar()
        self._wms_type = tk.StringVar(value="xyz")

        self._build_window()
        self._init_map()

    def _build_window(self):
        self.title("🌐 WMS - Pilih Area of Interest (AOI)")
        self.geometry("1100x750")
        self.minsize(900, 600)
        self.configure(fg_color="#070D1A")
        self.grab_set()  # Modal

        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=0)  # Left panel
        self.grid_columnconfigure(1, weight=1)  # Map

        # ── Left Control Panel ──────────────────────────────────────────────
        left = ctk.CTkScrollableFrame(self, width=280, fg_color="#0A1628", corner_radius=0)
        left.grid(row=0, column=0, sticky="nsew")
        left.grid_columnconfigure(0, weight=1)

        row = 0

        # Title
        title_frame = ctk.CTkFrame(left, fg_color="#0F172A", corner_radius=12)
        title_frame.grid(row=row, column=0, sticky="ew", padx=8, pady=(12, 8))
        title_frame.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(title_frame, text="🌐", font=ctk.CTkFont(size=26)).grid(row=0, column=0, pady=(10, 0))
        ctk.CTkLabel(
            title_frame, text="Input WMS / Peta Online",
            font=ctk.CTkFont(family="Segoe UI", size=13, weight="bold"),
            text_color="#E2E8F0",
        ).grid(row=1, column=0)
        ctk.CTkLabel(
            title_frame, text="Gambar AOI lalu download citra",
            font=ctk.CTkFont(size=10), text_color="#475569",
        ).grid(row=2, column=0, pady=(0, 10))
        row += 1

        # Divider helper
        def divider(r):
            ctk.CTkFrame(left, height=1, fg_color="#1E293B").grid(
                row=r, column=0, sticky="ew", padx=8, pady=2)

        def sec_lbl(text, r):
            ctk.CTkLabel(
                left, text=text,
                font=ctk.CTkFont(family="Segoe UI", size=10, weight="bold"),
                text_color="#475569", anchor="w",
            ).grid(row=r, column=0, sticky="ew", padx=12, pady=(14, 2))

        # ── Sumber WMS ─────────────────────────────────────────────────────
        divider(row); row += 1
        sec_lbl("🗺️  SUMBER CITRA", row); row += 1

        self.preset_menu = ctk.CTkOptionMenu(
            left,
            values=self._preset_names,
            variable=self._selected_preset,
            font=ctk.CTkFont(size=11), height=32,
            fg_color="#1E293B", button_color="#334155",
            button_hover_color="#475569", dropdown_fg_color="#1E293B",
            dropdown_hover_color="#334155", text_color="#E2E8F0",
            command=self._on_preset_changed,
        )
        self.preset_menu.grid(row=row, column=0, sticky="ew", padx=12, pady=(0, 4))
        row += 1

        # Custom URL (hidden by default)
        self.custom_url_frame = ctk.CTkFrame(left, fg_color="transparent")
        self.custom_url_frame.grid(row=row, column=0, sticky="ew", padx=12, pady=(0, 4))
        self.custom_url_frame.grid_columnconfigure(0, weight=1)
        self.custom_url_frame.grid_remove()  # Hidden initially

        ctk.CTkLabel(
            self.custom_url_frame, text="URL Template ({x}/{y}/{z}):",
            font=ctk.CTkFont(size=10), text_color="#94A3B8", anchor="w",
        ).grid(row=0, column=0, sticky="ew")
        self.entry_custom_url = ctk.CTkEntry(
            self.custom_url_frame,
            textvariable=self._custom_url,
            placeholder_text="https://example.com/tiles/{z}/{x}/{y}.png",
            font=ctk.CTkFont(size=10), height=30,
            fg_color="#0F172A", border_color="#334155", text_color="#E2E8F0",
        )
        self.entry_custom_url.grid(row=1, column=0, sticky="ew")

        ctk.CTkLabel(
            self.custom_url_frame, text="Tipe WMS:",
            font=ctk.CTkFont(size=10), text_color="#94A3B8", anchor="w",
        ).grid(row=2, column=0, sticky="ew", pady=(4, 0))
        ctk.CTkOptionMenu(
            self.custom_url_frame,
            values=["xyz", "bing"],
            variable=self._wms_type,
            font=ctk.CTkFont(size=10), height=26,
            fg_color="#1E293B", button_color="#334155",
            button_hover_color="#475569", dropdown_fg_color="#1E293B",
            dropdown_hover_color="#334155", text_color="#E2E8F0",
        ).grid(row=3, column=0, sticky="ew", pady=(2, 0))
        row += 1

        # Attribution
        self.lbl_attribution = ctk.CTkLabel(
            left,
            text=f"© {WMS_PRESETS[self._selected_preset.get()].get('attribution', '')}",
            font=ctk.CTkFont(size=9), text_color="#334155",
            anchor="w", wraplength=240,
        )
        self.lbl_attribution.grid(row=row, column=0, sticky="ew", padx=12, pady=(0, 4))
        row += 1

        # ── Tool Gambar AOI ─────────────────────────────────────────────────
        divider(row); row += 1
        sec_lbl("✏️  GAMBAR AREA OF INTEREST", row); row += 1

        ctk.CTkLabel(
            left, text="Mode gambar AOI:",
            font=ctk.CTkFont(size=11), text_color="#CBD5E1", anchor="w",
        ).grid(row=row, column=0, sticky="ew", padx=12)
        row += 1

        mode_seg = ctk.CTkSegmentedButton(
            left,
            values=["🔲 Rectangle", "🔷 Polygon"],
            font=ctk.CTkFont(size=11), height=30,
            fg_color="#1E293B", selected_color="#6366F1",
            selected_hover_color="#4F46E5", unselected_color="#1E293B",
            unselected_hover_color="#334155", text_color="#E2E8F0",
            command=self._on_mode_changed,
        )
        mode_seg.grid(row=row, column=0, sticky="ew", padx=12, pady=(0, 4))
        mode_seg.set("🔲 Rectangle")
        row += 1

        self.btn_draw = ctk.CTkButton(
            left,
            text="▶  Mulai Menggambar AOI",
            font=ctk.CTkFont(size=12, weight="bold"), height=36,
            corner_radius=8, fg_color="#6366F1", hover_color="#4F46E5",
            command=self._start_drawing,
        )
        self.btn_draw.grid(row=row, column=0, sticky="ew", padx=12, pady=(2, 2))
        row += 1

        self.btn_clear_aoi = ctk.CTkButton(
            left,
            text="🗑  Hapus AOI",
            font=ctk.CTkFont(size=11), height=30,
            corner_radius=8, fg_color="#1E293B", hover_color="#7F1D1D",
            text_color="#F87171", command=self._clear_aoi,
        )
        self.btn_clear_aoi.grid(row=row, column=0, sticky="ew", padx=12, pady=(0, 4))
        row += 1

        # Instruksi kontekstual
        self.lbl_instruction = ctk.CTkLabel(
            left,
            text="Klik 'Mulai Menggambar' lalu klik-drag di peta untuk membuat area seleksi.",
            font=ctk.CTkFont(size=10), text_color="#475569",
            anchor="w", wraplength=240, justify="left",
        )
        self.lbl_instruction.grid(row=row, column=0, sticky="ew", padx=12, pady=(0, 8))
        row += 1

        # ── Info AOI ────────────────────────────────────────────────────────
        divider(row); row += 1
        sec_lbl("📊  INFO AOI", row); row += 1

        self.info_frame = ctk.CTkFrame(left, fg_color="#0F172A", corner_radius=8)
        self.info_frame.grid(row=row, column=0, sticky="ew", padx=12, pady=(0, 4))
        self.info_frame.grid_columnconfigure(1, weight=1)

        self._info_labels = {}
        for i, (key, label) in enumerate([
            ("zoom", "Zoom Level"),
            ("tiles", "Jumlah Tile"),
            ("size_mb", "Est. Ukuran"),
            ("area_km2", "Luas AOI"),
        ]):
            ctk.CTkLabel(
                self.info_frame, text=f"{label}:",
                font=ctk.CTkFont(size=9), text_color="#475569", anchor="w",
            ).grid(row=i, column=0, sticky="w", padx=(8, 4), pady=1)
            self._info_labels[key] = ctk.CTkLabel(
                self.info_frame, text="—",
                font=ctk.CTkFont(size=9), text_color="#94A3B8", anchor="w",
            )
            self._info_labels[key].grid(row=i, column=1, sticky="ew", padx=(0, 8), pady=1)
        row += 1

        # ── Download ────────────────────────────────────────────────────────
        divider(row); row += 1
        sec_lbl("⬇️  DOWNLOAD & PROSES", row); row += 1

        self.btn_download = ctk.CTkButton(
            left,
            text="⬇  Download AOI & Proses SAM",
            font=ctk.CTkFont(family="Segoe UI", size=12, weight="bold"),
            height=46, corner_radius=10,
            fg_color="#10B981", hover_color="#059669",
            text_color="#FFFFFF", state="disabled",
            command=self._start_download,
        )
        self.btn_download.grid(row=row, column=0, sticky="ew", padx=12, pady=(4, 4))
        row += 1

        self.lbl_download_status = ctk.CTkLabel(
            left, text="Gambar AOI terlebih dahulu",
            font=ctk.CTkFont(size=10), text_color="#475569",
            anchor="w", wraplength=240, justify="left",
        )
        self.lbl_download_status.grid(row=row, column=0, sticky="ew", padx=12)
        row += 1

        self.download_progress = ctk.CTkProgressBar(
            left, height=8, corner_radius=4,
            fg_color="#1E293B", progress_color="#10B981",
        )
        self.download_progress.grid(row=row, column=0, sticky="ew", padx=12, pady=(4, 8))
        self.download_progress.set(0)
        row += 1

        # ── Right: Map Area ─────────────────────────────────────────────────
        map_container = ctk.CTkFrame(self, fg_color="#0A0F1E", corner_radius=0)
        map_container.grid(row=0, column=1, sticky="nsew")
        map_container.grid_rowconfigure(0, weight=0)
        map_container.grid_rowconfigure(1, weight=1)
        map_container.grid_columnconfigure(0, weight=1)

        # Header bar atas peta
        map_header = ctk.CTkFrame(map_container, fg_color="#0F172A", height=38, corner_radius=0)
        map_header.grid(row=0, column=0, sticky="ew")
        map_header.grid_columnconfigure(1, weight=1)
        map_header.grid_propagate(False)

        ctk.CTkLabel(
            map_header, text="🗺️  PETA INTERAKTIF",
            font=ctk.CTkFont(family="Segoe UI", size=11, weight="bold"),
            text_color="#64748B",
        ).grid(row=0, column=0, padx=12, pady=8, sticky="w")

        self.lbl_map_status = ctk.CTkLabel(
            map_header, text="Zoom dan pan peta, lalu gambar AOI",
            font=ctk.CTkFont(size=10), text_color="#334155",
        )
        self.lbl_map_status.grid(row=0, column=1, padx=8, pady=8, sticky="e")

        # Placeholder frame for map (will be replaced by tkintermapview if available)
        self.map_frame = ctk.CTkFrame(map_container, fg_color="#0A0F1E", corner_radius=0)
        self.map_frame.grid(row=1, column=0, sticky="nsew")
        self.map_frame.grid_rowconfigure(0, weight=1)
        self.map_frame.grid_columnconfigure(0, weight=1)

        # Bottom status bar
        self.lbl_coords = ctk.CTkLabel(
            map_container, text="",
            font=ctk.CTkFont(family="Consolas", size=9),
            text_color="#1E293B", anchor="w",
        )
        self.lbl_coords.grid(row=2, column=0, sticky="ew", padx=8, pady=2)

    def _init_map(self):
        """Initialize the map widget (tkintermapview if available, fallback otherwise)."""
        try:
            import tkintermapview
            self._init_tkintermapview(tkintermapview)
        except ImportError:
            self._init_fallback_map()

    def _init_tkintermapview(self, tkintermapview):
        """Use tkintermapview for a proper interactive map."""
        preset = WMS_PRESETS.get(self._selected_preset.get(), {})
        tile_url = preset.get("url", "https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}")
        # tkintermapview uses {x},{y},{z} format natively

        self.map_widget = tkintermapview.TkinterMapView(
            self.map_frame,
            width=800, height=600,
            corner_radius=0,
        )
        self.map_widget.pack(fill="both", expand=True)

        # Use the selected tile server
        self.map_widget.set_tile_server(tile_url, max_zoom=20)
        self.map_widget.set_position(-6.9175, 107.6191)  # Default: Bandung
        self.map_widget.set_zoom(14)

        self._map_type = "tkintermapview"
        self.lbl_map_status.configure(text="Klik 'Mulai Menggambar' untuk membuat AOI")

        # Mouse events for drawing
        self.map_widget.canvas.bind("<Motion>", self._on_map_mouse_move_tkmap)

    def _init_fallback_map(self):
        """Fallback: matplotlib-based map with offline tile simulation."""
        import matplotlib
        matplotlib.use("TkAgg")
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

        self.fig_map = Figure(figsize=(8, 6), dpi=96, facecolor="#0A0F1E")
        self.ax_map = self.fig_map.add_axes([0, 0, 1, 1], facecolor="#0D1B2A")
        self.ax_map.set_axis_off()
        self.ax_map.text(
            0.5, 0.55,
            "🌐  Peta Interaktif",
            ha="center", va="center", fontsize=18, color="#334155",
            fontweight="bold", transform=self.ax_map.transAxes,
        )
        self.ax_map.text(
            0.5, 0.45,
            "Install tkintermapview untuk peta interaktif:\npip install tkintermapview",
            ha="center", va="center", fontsize=11, color="#1E3A5F",
            transform=self.ax_map.transAxes,
        )
        self.ax_map.text(
            0.5, 0.35,
            "Saat ini: masukkan koordinat AOI secara manual di bawah.",
            ha="center", va="center", fontsize=10, color="#475569",
            transform=self.ax_map.transAxes,
        )

        self.canvas_map = FigureCanvasTkAgg(self.fig_map, master=self.map_frame)
        self.canvas_map.draw()
        self.canvas_map.get_tk_widget().pack(fill="both", expand=True)

        self._map_type = "fallback"

        # Show manual coordinate input
        self._build_manual_aoi_input()
        self.lbl_map_status.configure(
            text="⚠️ tkintermapview tidak terinstal — gunakan input koordinat manual"
        )

    def _build_manual_aoi_input(self):
        """Fallback: Input koordinat AOI secara manual."""
        manual_frame = ctk.CTkFrame(self.map_frame, fg_color="#0F172A", corner_radius=8)
        manual_frame.place(relx=0.5, rely=0.72, anchor="center")
        manual_frame.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(
            manual_frame, text="Masukkan koordinat AOI (WGS84 / Desimal):",
            font=ctk.CTkFont(size=11, weight="bold"), text_color="#CBD5E1",
        ).grid(row=0, column=0, columnspan=4, padx=12, pady=(10, 6))

        fields = [
            ("Min Lon:", "min_lon", "107.5"), ("Min Lat:", "min_lat", "-7.1"),
            ("Max Lon:", "max_lon", "107.7"), ("Max Lat:", "max_lat", "-6.9"),
        ]
        self._manual_vars = {}
        for i, (lbl, key, default) in enumerate(fields):
            col = (i % 2) * 2
            r = 1 + i // 2
            ctk.CTkLabel(
                manual_frame, text=lbl,
                font=ctk.CTkFont(size=10), text_color="#94A3B8",
            ).grid(row=r, column=col, padx=(12, 2), pady=3, sticky="e")
            var = tk.StringVar(value=default)
            self._manual_vars[key] = var
            ctk.CTkEntry(
                manual_frame, textvariable=var, width=90, height=28,
                font=ctk.CTkFont(size=10),
                fg_color="#1E293B", border_color="#334155", text_color="#E2E8F0",
            ).grid(row=r, column=col + 1, padx=(0, 8), pady=3, sticky="w")

        ctk.CTkButton(
            manual_frame, text="✅  Set AOI dari Koordinat",
            font=ctk.CTkFont(size=11, weight="bold"), height=34, corner_radius=8,
            fg_color="#6366F1", hover_color="#4F46E5",
            command=self._set_aoi_from_manual,
        ).grid(row=3, column=0, columnspan=4, padx=12, pady=(6, 10), sticky="ew")

    def _set_aoi_from_manual(self):
        """Set AOI from manually entered coordinates."""
        try:
            min_lon = float(self._manual_vars["min_lon"].get())
            min_lat = float(self._manual_vars["min_lat"].get())
            max_lon = float(self._manual_vars["max_lon"].get())
            max_lat = float(self._manual_vars["max_lat"].get())
        except ValueError:
            messagebox.showerror("Error", "Koordinat tidak valid. Gunakan angka desimal.", parent=self)
            return

        polygon = [
            (min_lon, min_lat), (max_lon, min_lat),
            (max_lon, max_lat), (min_lon, max_lat), (min_lon, min_lat),
        ]
        self._set_aoi_polygon(polygon, "rectangle")

    def _on_preset_changed(self, value):
        """Handle WMS preset selection change."""
        if value == "URL Kustom...":
            self.custom_url_frame.grid()
            self.lbl_attribution.configure(text="")
        else:
            self.custom_url_frame.grid_remove()
            preset = WMS_PRESETS.get(value, {})
            self.lbl_attribution.configure(
                text=f"© {preset.get('attribution', '')}"
            )
            if hasattr(self, "map_widget") and self._map_type == "tkintermapview":
                try:
                    self.map_widget.set_tile_server(preset.get("url", ""), max_zoom=20)
                except Exception:
                    pass

    def _on_mode_changed(self, value):
        """Switch between rectangle and polygon draw modes."""
        if "Rectangle" in value:
            self._aoi_mode.set("rectangle")
            self.lbl_instruction.configure(
                text="Klik 'Mulai Menggambar' lalu klik-drag di peta untuk membuat area seleksi persegi."
            )
        else:
            self._aoi_mode.set("polygon")
            self.lbl_instruction.configure(
                text="Klik 'Mulai Menggambar', klik titik-titik di peta untuk membuat polygon. "
                     "Double-klik untuk menutup polygon."
            )

    def _start_drawing(self):
        """Activate drawing mode."""
        self._draw_mode_active = True
        self._polygon_points.clear()
        self._rect_start = None

        if self._map_type == "tkintermapview" and hasattr(self, "map_widget"):
            self._clear_map_overlays()
            canvas = self.map_widget.canvas
            if self._aoi_mode.get() == "rectangle":
                canvas.bind("<ButtonPress-1>", self._on_rect_press)
                canvas.bind("<B1-Motion>", self._on_rect_drag)
                canvas.bind("<ButtonRelease-1>", self._on_rect_release)
                self.lbl_map_status.configure(text="🖊  Klik dan drag untuk menggambar rectangle AOI")
            else:
                canvas.bind("<ButtonPress-1>", self._on_poly_click)
                canvas.bind("<Double-Button-1>", self._on_poly_double_click)
                self.lbl_map_status.configure(text="🖊  Klik titik-titik polygon. Double-klik untuk selesai")

        self.btn_draw.configure(text="⏹  Gambar Aktif...", fg_color="#475569", state="disabled")

    def _stop_drawing(self):
        """Deactivate drawing mode and restore normal map interaction."""
        self._draw_mode_active = False
        if self._map_type == "tkintermapview" and hasattr(self, "map_widget"):
            canvas = self.map_widget.canvas
            canvas.unbind("<ButtonPress-1>")
            canvas.unbind("<B1-Motion>")
            canvas.unbind("<ButtonRelease-1>")
            canvas.unbind("<Double-Button-1>")
        self.btn_draw.configure(text="▶  Mulai Menggambar AOI", fg_color="#6366F1", state="normal")

    # ── Rectangle Drawing ───────────────────────────────────────────────────

    def _on_rect_press(self, event):
        self._rect_start = (event.x, event.y)
        self._rect_canvas_item = None

    def _on_rect_drag(self, event):
        if self._rect_start is None:
            return
        canvas = self.map_widget.canvas
        if hasattr(self, "_rect_canvas_item") and self._rect_canvas_item:
            canvas.delete(self._rect_canvas_item)
        x0, y0 = self._rect_start
        self._rect_canvas_item = canvas.create_rectangle(
            x0, y0, event.x, event.y,
            outline="#FBBF24", width=2, fill="", dash=(4, 4),
        )

    def _on_rect_release(self, event):
        if self._rect_start is None:
            return
        x0_px, y0_px = self._rect_start
        x1_px, y1_px = event.x, event.y

        if abs(x1_px - x0_px) < 10 or abs(y1_px - y0_px) < 10:
            self._stop_drawing()
            return

        # Convert canvas pixel → lat/lon
        try:
            lat0, lon0 = self.map_widget.convert_canvas_coords_to_decimal_coords(x0_px, y0_px)
            lat1, lon1 = self.map_widget.convert_canvas_coords_to_decimal_coords(x1_px, y1_px)
        except Exception:
            self._stop_drawing()
            return

        min_lon, max_lon = min(lon0, lon1), max(lon0, lon1)
        min_lat, max_lat = min(lat0, lat1), max(lat0, lat1)

        polygon = [
            (min_lon, min_lat), (max_lon, min_lat),
            (max_lon, max_lat), (min_lon, max_lat), (min_lon, min_lat),
        ]
        self._stop_drawing()
        self._set_aoi_polygon(polygon, "rectangle")

    # ── Polygon Drawing ─────────────────────────────────────────────────────

    def _on_poly_click(self, event):
        try:
            lat, lon = self.map_widget.convert_canvas_coords_to_decimal_coords(event.x, event.y)
        except Exception:
            return
        self._polygon_points.append((lon, lat))

        # Draw marker
        marker = self.map_widget.set_marker(
            lat, lon, text="",
            marker_color_circle="#6366F1", marker_color_outside="#4F46E5",
        )
        self._map_markers.append(marker)
        self.lbl_map_status.configure(
            text=f"🖊  {len(self._polygon_points)} titik. Double-klik untuk selesai."
        )

    def _on_poly_double_click(self, event):
        if len(self._polygon_points) < 3:
            messagebox.showwarning(
                "Polygon Tidak Valid",
                "Polygon membutuhkan minimal 3 titik.",
                parent=self,
            )
            return
        # Close polygon
        pts = self._polygon_points + [self._polygon_points[0]]
        self._stop_drawing()
        self._set_aoi_polygon(pts, "polygon")

    # ── AOI Logic ───────────────────────────────────────────────────────────

    def _set_aoi_polygon(self, polygon_coords, aoi_type):
        """Store and display the drawn AOI polygon."""
        self._current_aoi = polygon_coords
        self._current_aoi_type = aoi_type

        min_lon, min_lat, max_lon, max_lat = get_aoi_bounds_from_polygon(polygon_coords)

        # Display on map
        if self._map_type == "tkintermapview" and hasattr(self, "map_widget"):
            self._clear_map_overlays()
            try:
                # Draw polygon on map
                lat_lon_pts = [(lat, lon) for lon, lat in polygon_coords]
                self._map_polygon = self.map_widget.set_polygon(
                    lat_lon_pts,
                    fill_color="#FBBF24",
                    outline_color="#F59E0B",
                    border_width=3,
                    name="AOI",
                )
            except Exception as e:
                pass

        # Update AOI info
        zoom = get_best_zoom(min_lon, min_lat, max_lon, max_lat, max_zoom=20, max_tiles=600)
        tile_count = estimate_tile_count(min_lon, min_lat, max_lon, max_lat, zoom)
        size_mb = estimate_download_size_mb(tile_count)

        # Calculate area in km² (approximate)
        lat_mid = (min_lat + max_lat) / 2
        import math
        dx = (max_lon - min_lon) * math.cos(math.radians(lat_mid)) * 111.32
        dy = (max_lat - min_lat) * 110.54
        area_km2 = abs(dx * dy)

        self._info_labels["zoom"].configure(text=str(zoom))
        self._info_labels["tiles"].configure(text=f"{tile_count:,}")
        self._info_labels["size_mb"].configure(text=f"~{size_mb:.1f} MB")
        self._info_labels["area_km2"].configure(text=f"~{area_km2:.2f} km²")

        self.lbl_map_status.configure(
            text=f"✅ AOI ditetapkan: {tile_count} tile | ~{size_mb:.1f} MB | ~{area_km2:.2f} km²"
        )
        self.lbl_download_status.configure(
            text=f"Siap download {tile_count} tile (~{size_mb:.1f} MB)"
        )
        self.btn_download.configure(state="normal")

    def _clear_aoi(self):
        """Clear the current AOI selection."""
        self._current_aoi = None
        self._current_aoi_type = None
        self._polygon_points.clear()
        self._clear_map_overlays()

        for key in self._info_labels:
            self._info_labels[key].configure(text="—")
        self.lbl_map_status.configure(text="AOI dihapus. Gambar AOI baru.")
        self.lbl_download_status.configure(text="Gambar AOI terlebih dahulu")
        self.btn_download.configure(state="disabled")

    def _clear_map_overlays(self):
        """Remove all drawing overlays from the map."""
        if self._map_type == "tkintermapview" and hasattr(self, "map_widget"):
            for m in self._map_markers:
                try:
                    m.delete()
                except Exception:
                    pass
            self._map_markers.clear()
            if self._map_polygon:
                try:
                    self._map_polygon.delete()
                except Exception:
                    pass
                self._map_polygon = None
            # Also remove any rectangle canvas drawing
            if hasattr(self, "_rect_canvas_item") and self._rect_canvas_item:
                try:
                    self.map_widget.canvas.delete(self._rect_canvas_item)
                except Exception:
                    pass
                self._rect_canvas_item = None

    def _on_map_mouse_move_tkmap(self, event):
        """Update coordinate display on mouse move."""
        if hasattr(self, "map_widget"):
            try:
                lat, lon = self.map_widget.convert_canvas_coords_to_decimal_coords(event.x, event.y)
                self.lbl_coords.configure(text=f"Lat: {lat:.6f}  Lon: {lon:.6f}")
            except Exception:
                pass

    # ── Download ─────────────────────────────────────────────────────────────

    def _get_current_wms_info(self):
        """Return (url, type) of the currently selected WMS."""
        preset_name = self._selected_preset.get()
        if preset_name == "URL Kustom...":
            url = self._custom_url.get().strip()
            wms_type = self._wms_type.get()
        else:
            preset = WMS_PRESETS.get(preset_name, {})
            url = preset.get("url", "")
            wms_type = preset.get("type", "xyz")
        return url, wms_type

    def _start_download(self):
        """Initiate the tile download in a background thread."""
        if self._current_aoi is None:
            messagebox.showwarning("AOI Belum Ditetapkan", "Gambar area AOI terlebih dahulu.", parent=self)
            return

        url, wms_type = self._get_current_wms_info()
        if not url:
            messagebox.showerror("URL Kosong", "Pilih sumber WMS atau masukkan URL kustom.", parent=self)
            return

        aoi_bounds = get_aoi_bounds_from_polygon(self._current_aoi)
        preset_name = self._selected_preset.get()
        safe_name = preset_name.replace(" ", "_").replace("/", "-").replace(".", "")

        output_filename = f"wms_{safe_name}_aoi.tif"
        output_path = os.path.join(self.output_dir, output_filename)

        self.btn_download.configure(state="disabled", text="⏳  Mengunduh...")
        self.btn_draw.configure(state="disabled")
        self.download_progress.set(0)
        self.lbl_download_status.configure(text="Memulai download...")

        self._cancelled = False

        self._download_thread = threading.Thread(
            target=self._run_download,
            args=(url, aoi_bounds, output_path, wms_type),
            daemon=True,
        )
        self._download_thread.start()

    def _run_download(self, url, aoi_bounds, output_path, wms_type):
        """Background download thread."""
        try:
            result_path = download_wms_aoi(
                wms_url=url,
                aoi_bounds=aoi_bounds,
                output_path=output_path,
                zoom=None,  # Auto
                wms_type=wms_type,
                max_tiles=600,
                log_callback=self.log_callback,
                progress_callback=self._progress_callback,
                cancel_check=lambda: getattr(self, "_cancelled", False),
            )
            self.after(0, lambda: self._on_download_complete(result_path))
        except Exception as e:
            self.after(0, lambda err=str(e): self._on_download_error(err))

    def _progress_callback(self, percent: int, message: str = ""):
        self.after(0, lambda: self.download_progress.set(percent / 100.0))
        if message:
            self.after(0, lambda m=message: self.lbl_download_status.configure(text=m))

    def _on_download_complete(self, geotiff_path: str):
        self.download_progress.set(1.0)
        self.lbl_download_status.configure(
            text=f"✅ Selesai! {os.path.basename(geotiff_path)}"
        )
        self.btn_download.configure(state="normal", text="⬇  Download AOI & Proses SAM")
        self.btn_draw.configure(state="normal")

        if self.on_ready:
            self.on_ready(geotiff_path)
        self.destroy()

    def _on_download_error(self, error_msg: str):
        self.btn_download.configure(state="normal", text="⬇  Download AOI & Proses SAM")
        self.btn_draw.configure(state="normal")
        self.lbl_download_status.configure(text=f"❌ Gagal: {error_msg[:80]}")
        messagebox.showerror("Download Gagal", f"Terjadi kesalahan:\n\n{error_msg}", parent=self)

"""
preview_panel.py
Center panel for raster preview with matplotlib canvas.
Shows the input image with overlay of detected building polygons.
Highlights the currently-processing tile in yellow.
"""

import tkinter as tk
import customtkinter as ctk
import numpy as np
from typing import Optional


class PreviewPanel(ctk.CTkFrame):
    """
    Image preview panel using matplotlib embedded in Tkinter.
    Displays the input raster with overlay of detected polygons.
    """

    def __init__(self, parent, on_maximize_toggle=None, **kwargs):
        super().__init__(parent, **kwargs)
        self._raster_path: Optional[str] = None
        self._gdf = None         # GeoDataFrame with result polygons
        self._yolo_boxes = None  # List of YOLO boxes (geo coords)
        self._tiles = []         # List of tile rects
        self._active_tile_idx = -1
        self._raster_array = None  # Cached preview image
        self._raster_extent = None  # (left, right, bottom, top)
        self.on_maximize_toggle = on_maximize_toggle
        self.is_maximized = False
        self._build_ui()

    def _build_ui(self):
        self.grid_rowconfigure(1, weight=1)
        self.grid_columnconfigure(0, weight=1)

        # ── Header Bar ──────────────────────────────────
        header = ctk.CTkFrame(self, fg_color="#0F172A", corner_radius=0, height=40)
        header.grid(row=0, column=0, sticky="ew")
        header.grid_columnconfigure(1, weight=1)
        header.grid_propagate(False)

        ctk.CTkLabel(
            header,
            text="🗺️  PREVIEW CITRA",
            font=ctk.CTkFont(family="Segoe UI", size=11, weight="bold"),
            text_color="#64748B",
        ).grid(row=0, column=0, padx=12, pady=10, sticky="w")

        # Stats bar
        self.stats_frame = ctk.CTkFrame(header, fg_color="transparent")
        self.stats_frame.grid(row=0, column=1, sticky="e", padx=12)

        self.lbl_count = ctk.CTkLabel(
            self.stats_frame,
            text="",
            font=ctk.CTkFont(size=10),
            text_color="#4ADE80",
        )
        self.lbl_count.pack(side="right", padx=4)

        self.lbl_tile_status = ctk.CTkLabel(
            self.stats_frame,
            text="",
            font=ctk.CTkFont(size=10),
            text_color="#FBBF24",
        )
        self.lbl_tile_status.pack(side="right", padx=4)

        # Maximize Button (rightmost in stats_frame)
        self.btn_maximize = ctk.CTkButton(
            self.stats_frame,
            text="🗖 Maximize Preview",
            width=120,
            height=22,
            font=ctk.CTkFont(family="Segoe UI", size=10, weight="bold"),
            fg_color="#1E293B",
            hover_color="#334155",
            text_color="#94A3B8",
            command=self.toggle_maximize,
        )
        self.btn_maximize.pack(side="right", padx=(8, 4))

    def toggle_maximize(self):
        """Toggle maximized status of this preview panel."""
        self.is_maximized = not self.is_maximized
        if self.is_maximized:
            self.btn_maximize.configure(text="🗖 Restore View")
        else:
            self.btn_maximize.configure(text="🗖 Maximize Preview")

        if self.on_maximize_toggle:
            self.on_maximize_toggle(self.is_maximized)

        # ── Matplotlib Canvas ────────────────────────────
        import matplotlib
        matplotlib.use("TkAgg")
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

        self.fig = Figure(figsize=(8, 6), dpi=96, facecolor="#0A0F1E")
        self.ax = self.fig.add_axes([0, 0, 1, 1], facecolor="#0A0F1E")
        self.ax.set_axis_off()

        # Placeholder text
        self.ax.text(
            0.5, 0.5,
            "Pilih file raster untuk melihat preview",
            ha="center", va="center",
            fontsize=13, color="#334155",
            transform=self.ax.transAxes,
        )

        self.canvas = FigureCanvasTkAgg(self.fig, master=self)
        self.canvas.draw()
        self.canvas.get_tk_widget().grid(row=1, column=0, sticky="nsew")

        # ── Toolbar ─────────────────────────────────────
        toolbar_frame = ctk.CTkFrame(self, fg_color="#0A0F1E", height=30)
        toolbar_frame.grid(row=2, column=0, sticky="ew")
        toolbar_frame.grid_columnconfigure(0, weight=1)

        self.lbl_coords = ctk.CTkLabel(
            toolbar_frame,
            text="",
            font=ctk.CTkFont(family="Consolas", size=9),
            text_color="#334155",
            anchor="w",
        )
        self.lbl_coords.grid(row=0, column=0, sticky="w", padx=8)

        # Mouse event binding for zoom & pan
        self._pan_start_x = None
        self._pan_start_y = None
        self._pan_xlim = None
        self._pan_ylim = None
        self._is_panning = False

        self.canvas.mpl_connect("scroll_event", self._on_zoom)
        self.canvas.mpl_connect("button_press_event", self._on_pan_start)
        self.canvas.mpl_connect("button_release_event", self._on_pan_end)
        self.canvas.mpl_connect("motion_notify_event", self._on_mouse_move)

    def _on_zoom(self, event):
        if event.inaxes != self.ax:
            return
        
        # Get the current limits
        cur_xlim = self.ax.get_xlim()
        cur_ylim = self.ax.get_ylim()
        
        xdata = event.xdata
        ydata = event.ydata
        
        # Zoom factor (scroll up to zoom in, scroll down to zoom out)
        scale_factor = 0.85 if event.button == "up" else 1.15
        
        # New width and height of the viewport
        new_width = (cur_xlim[1] - cur_xlim[0]) * scale_factor
        new_height = (cur_ylim[1] - cur_ylim[0]) * scale_factor
        
        relx = (cur_xlim[1] - xdata) / (cur_xlim[1] - cur_xlim[0])
        rely = (cur_ylim[1] - ydata) / (cur_ylim[1] - cur_ylim[0])
        
        self.ax.set_xlim([xdata - new_width * (1 - relx), xdata + new_width * relx])
        self.ax.set_ylim([ydata - new_height * (1 - rely), ydata + new_height * rely])
        
        self.canvas.draw_idle()

    def _on_pan_start(self, event):
        if event.inaxes != self.ax:
            return
            
        # Double-click anywhere to reset zoom/pan to the default full-extent
        if event.dblclick:
            self.reset_view()
            return

        if event.button in [1, 3]: # Left or Right Click drag to pan
            self._pan_start_x = event.x
            self._pan_start_y = event.y
            self._pan_xlim = np.array(self.ax.get_xlim())
            self._pan_ylim = np.array(self.ax.get_ylim())
            self._is_panning = True

    def _on_pan_end(self, event):
        self._is_panning = False

    def _on_mouse_move(self, event):
        # Update coordinates on toolbar
        if event.xdata is not None and event.ydata is not None:
            self.lbl_coords.configure(
                text=f"X: {event.xdata:.4f}   Y: {event.ydata:.4f}  [Scroll: Zoom | Drag: Pan | Double-Click: Reset]"
            )
            
        # Perform panning if active
        if getattr(self, "_is_panning", False) and event.inaxes == self.ax:
            dx_pix = event.x - self._pan_start_x
            dy_pix = event.y - self._pan_start_y
            
            bbox = self.ax.get_window_extent()
            width_pix = bbox.width
            height_pix = bbox.height
            
            width_data = self._pan_xlim[1] - self._pan_xlim[0]
            height_data = self._pan_ylim[1] - self._pan_ylim[0]
            
            dx_data = dx_pix * (width_data / width_pix)
            dy_data = dy_pix * (height_data / height_pix)
            
            self.ax.set_xlim(self._pan_xlim - dx_data)
            self.ax.set_ylim(self._pan_ylim - dy_data)
            self.canvas.draw_idle()

    def reset_view(self):
        """Reset view limits back to full raster extent."""
        if getattr(self, "_raster_extent", None) is not None:
            self.ax.set_xlim(self._raster_extent[0], self._raster_extent[1])
            self.ax.set_ylim(self._raster_extent[2], self._raster_extent[3])
            self.canvas.draw_idle()

    def load_raster_preview(self, raster_path: str):
        """
        Load and display a thumbnail of the raster.
        Uses adaptive downsampling to keep GUI responsive.
        """
        self._raster_path = raster_path

        try:
            import rasterio
            from rasterio.enums import Resampling

            with rasterio.open(raster_path) as src:
                # Downsample to max 2048x2048 for preview
                max_size = 2048
                scale = min(max_size / src.width, max_size / src.height, 1.0)
                out_w = max(int(src.width * scale), 1)
                out_h = max(int(src.height * scale), 1)

                bounds = src.bounds
                self._raster_extent = [bounds.left, bounds.right, bounds.bottom, bounds.top]

                if src.count >= 3:
                    data = src.read(
                        [1, 2, 3],
                        out_shape=(3, out_h, out_w),
                        resampling=Resampling.bilinear,
                    )
                    rgb = np.transpose(data, (1, 2, 0)).astype(np.float32)
                else:
                    gray = src.read(
                        1,
                        out_shape=(1, out_h, out_w),
                        resampling=Resampling.bilinear,
                    )
                    rgb = np.transpose(
                        np.stack([gray, gray, gray], axis=0), (1, 2, 0)
                    ).astype(np.float32)

                # Contrast stretch (2%-98% percentile)
                for c in range(3):
                    channel = rgb[:, :, c]
                    nz = channel[channel > 0]
                    if len(nz) > 0:
                        p2, p98 = np.percentile(nz, (2, 98))
                        rgb[:, :, c] = np.clip((channel - p2) / max(p98 - p2, 1) * 255, 0, 255)
                    else:
                        rgb[:, :, c] = channel

                self._raster_array = rgb.astype(np.uint8)

            self._redraw()
        except Exception as e:
            self.ax.clear()
            self.ax.set_facecolor("#0A0F1E")
            self.ax.text(
                0.5, 0.5, f"Gagal memuat preview:\n{e}",
                ha="center", va="center", fontsize=10, color="#F87171",
                transform=self.ax.transAxes,
            )
            self.ax.set_axis_off()
            self.canvas.draw()

    def _redraw(self):
        """Redraw everything: raster + tile highlights + polygon overlay."""
        self.ax.clear()
        self.ax.set_facecolor("#0A0F1E")
        self.ax.set_axis_off()

        if self._raster_array is not None:
            ext = self._raster_extent
            self.ax.imshow(
                self._raster_array,
                extent=ext,
                aspect="equal",
                interpolation="bilinear",
            )

        import matplotlib.patches as patches
        
        # Draw tile grid (light gray outline)
        if self._tiles and self._raster_extent:
            for i, (x0, y0, x1, y1) in enumerate(self._tiles):
                color = "#FBBF24" if i == self._active_tile_idx else "#1E3A5F"
                alpha = 0.6 if i == self._active_tile_idx else 0.15
                lw = 2.0 if i == self._active_tile_idx else 0.5
                rect = patches.Rectangle(
                    (x0, y0), x1 - x0, y1 - y0,
                    linewidth=lw, edgecolor=color,
                    facecolor=color if i == self._active_tile_idx else "none",
                    alpha=alpha,
                )
                self.ax.add_patch(rect)

        # Draw polygon overlay
        if self._gdf is not None and len(self._gdf) > 0:
            try:
                import matplotlib.patches as mpatches
                from matplotlib.patches import Polygon as MplPolygon
                from matplotlib.collections import PatchCollection
                from shapely.geometry import MultiPolygon

                polys = []
                for geom in self._gdf.geometry:
                    if geom is None or geom.is_empty:
                        continue
                    if isinstance(geom, MultiPolygon):
                        for part in geom.geoms:
                            coords = np.array(part.exterior.coords)
                            polys.append(MplPolygon(coords, closed=True))
                    else:
                        try:
                            coords = np.array(geom.exterior.coords)
                            polys.append(MplPolygon(coords, closed=True))
                        except Exception:
                            pass

                if polys:
                    collection = PatchCollection(
                        polys,
                        facecolor="#F43F5E",
                        edgecolor="#FCA5A5",
                        alpha=0.35,
                        linewidth=0.8,
                    )
                    self.ax.add_collection(collection)

                # Draw YOLO bounding boxes if any
                if self._yolo_boxes:
                    for bx in self._yolo_boxes:
                        # bx is [xmin, ymin, xmax, ymax]
                        rect = patches.Rectangle(
                            (bx[0], bx[1]), bx[2] - bx[0], bx[3] - bx[1],
                            linewidth=1.2, edgecolor="#EF4444", facecolor="none",
                            alpha=0.8
                        )
                        self.ax.add_patch(rect)

                self.lbl_count.configure(
                    text=f"✅ {len(self._gdf):,} bangunan terdeteksi"
                )
            except Exception as e:
                self.lbl_count.configure(text=f"Overlay error: {e}")

        self.canvas.draw_idle()

    def set_tile_grid(self, tiles_geo: list):
        """
        Set tile bounding boxes in geographic coordinates for display.
        tiles_geo: list of (left, bottom, right, top) in CRS units
        """
        self._tiles = [(t[0], t[1], t[2], t[3]) for t in tiles_geo]
        self._redraw()

    def set_active_tile(self, idx: int):
        """Highlight the currently-processing tile."""
        self._active_tile_idx = idx
        self.lbl_tile_status.configure(text=f"Tile {idx+1}" if idx >= 0 else "")
        self._redraw()

    def set_result_polygons(self, gdf, yolo_boxes=None):
        """Display result polygons as overlay."""
        self._gdf = gdf
        self._yolo_boxes = yolo_boxes
        self._active_tile_idx = -1
        self._redraw()

    def clear_results(self):
        """Remove polygon overlay."""
        self._gdf = None
        self._yolo_boxes = None
        self._active_tile_idx = -1
        self.lbl_count.configure(text="")
        self.lbl_tile_status.configure(text="")
        self._redraw()

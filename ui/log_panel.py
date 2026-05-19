"""
log_panel.py
Real-time scrollable log panel and dual-layer progress bar widget.
Built with CustomTkinter for consistent dark-mode aesthetic.
"""

import customtkinter as ctk
import tkinter as tk
from datetime import datetime
from typing import Optional


class LogPanel(ctk.CTkFrame):
    """
    Scrollable log panel with timestamp, color-coded messages,
    and dual progress bars (tile-level + overall).
    """

    LOG_COLORS = {
        "info":    "#94A3B8",   # Slate gray - normal messages
        "success": "#4ADE80",   # Green - success
        "warning": "#FBBF24",   # Amber - warning
        "error":   "#F87171",   # Red - error
        "sam":     "#818CF8",   # Indigo - SAM messages
        "system":  "#38BDF8",   # Sky blue - system messages
    }

    def __init__(self, parent, on_minimize_toggle=None, **kwargs):
        super().__init__(parent, **kwargs)
        self.is_minimized = False
        self.on_minimize_toggle = on_minimize_toggle
        self._build_ui()

    def _build_ui(self):
        self.grid_rowconfigure(1, weight=1)
        self.grid_columnconfigure(0, weight=1)

        # ── Header ──────────────────────────────────────
        header = ctk.CTkFrame(self, fg_color="transparent")
        header.grid(row=0, column=0, sticky="ew", padx=12, pady=(10, 4))
        header.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(
            header,
            text="📋  LOG PROSES",
            font=ctk.CTkFont(family="Segoe UI", size=11, weight="bold"),
            text_color="#64748B",
        ).grid(row=0, column=0, sticky="w")

        # Minimize Button
        self.btn_minimize = ctk.CTkButton(
            header,
            text="➖ Minimize",
            width=75, height=22,
            font=ctk.CTkFont(size=10),
            fg_color="#1E293B",
            hover_color="#334155",
            text_color="#94A3B8",
            command=self.toggle_minimize,
        )
        self.btn_minimize.grid(row=0, column=1, sticky="e", padx=(0, 6))

        # Clear Button
        self.btn_clear = ctk.CTkButton(
            header,
            text="Hapus",
            width=60, height=22,
            font=ctk.CTkFont(size=10),
            fg_color="#1E293B",
            hover_color="#334155",
            text_color="#94A3B8",
            command=self.clear,
        )
        self.btn_clear.grid(row=0, column=2, sticky="e")

        # ── Log Text Area ────────────────────────────────
        self.text = tk.Text(
            self,
            wrap="word",
            state="disabled",
            bg="#0F172A",
            fg="#94A3B8",
            insertbackground="#94A3B8",
            font=("Consolas", 9),
            bd=0,
            padx=10,
            pady=8,
            cursor="arrow",
            selectbackground="#1E3A5F",
            selectforeground="#E2E8F0",
            relief="flat",
        )
        self.text.grid(row=1, column=0, sticky="nsew", padx=(8, 0), pady=0)

        self.scrollbar = ctk.CTkScrollbar(self, command=self.text.yview)
        self.scrollbar.grid(row=1, column=1, sticky="ns", padx=(0, 4))
        self.text.configure(yscrollcommand=self.scrollbar.set)

        # Configure color tags
        for tag, color in self.LOG_COLORS.items():
            self.text.tag_configure(tag, foreground=color)
        self.text.tag_configure("timestamp", foreground="#334155")
        self.text.tag_configure("bold", font=("Consolas", 9, "bold"))

        # ── Progress Bars ────────────────────────────────
        self.progress_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.progress_frame.grid(row=2, column=0, columnspan=2, sticky="ew", padx=12, pady=(4, 10))
        self.progress_frame.grid_columnconfigure(1, weight=1)

        # Tile progress
        ctk.CTkLabel(
            self.progress_frame,
            text="Tile:",
            font=ctk.CTkFont(size=10),
            text_color="#475569",
            width=60,
            anchor="e",
        ).grid(row=0, column=0, sticky="e", padx=(0, 6))

        self.tile_progress = ctk.CTkProgressBar(
            self.progress_frame,
            height=8,
            progress_color="#6366F1",
            fg_color="#1E293B",
            corner_radius=4,
        )
        self.tile_progress.grid(row=0, column=1, sticky="ew")
        self.tile_progress.set(0)

        self.tile_label = ctk.CTkLabel(
            self.progress_frame,
            text="0/0",
            font=ctk.CTkFont(size=9),
            text_color="#475569",
            width=45,
        )
        self.tile_label.grid(row=0, column=2, padx=(6, 0))

        # Overall progress
        ctk.CTkLabel(
            self.progress_frame,
            text="Total:",
            font=ctk.CTkFont(size=10),
            text_color="#475569",
            width=60,
            anchor="e",
        ).grid(row=1, column=0, sticky="e", padx=(0, 6), pady=(4, 0))

        self.overall_progress = ctk.CTkProgressBar(
            self.progress_frame,
            height=10,
            progress_color="#10B981",
            fg_color="#1E293B",
            corner_radius=4,
        )
        self.overall_progress.grid(row=1, column=1, sticky="ew", pady=(4, 0))

        self.overall_label = ctk.CTkLabel(
            self.progress_frame,
            text="0%",
            font=ctk.CTkFont(size=9),
            text_color="#475569",
            width=45,
        )
        self.overall_label.grid(row=1, column=2, padx=(6, 0), pady=(4, 0))

        # Status message below progress
        self.status_label = ctk.CTkLabel(
            self,
            text="Siap",
            font=ctk.CTkFont(size=10),
            text_color="#475569",
            anchor="w",
        )
        self.status_label.grid(row=3, column=0, columnspan=2, sticky="ew", padx=12, pady=(0, 8))

    def toggle_minimize(self):
        """Toggle minimizing the log panel body, calling the parent callback if provided."""
        self.is_minimized = not self.is_minimized
        if self.is_minimized:
            self.text.grid_remove()
            if hasattr(self, 'scrollbar'):
                self.scrollbar.grid_remove()
            self.progress_frame.grid_remove()
            self.status_label.grid_remove()
            self.btn_minimize.configure(text="🗖 Maximize")
            self.btn_clear.grid_remove()
        else:
            self.text.grid(row=1, column=0, sticky="nsew", padx=(8, 0), pady=0)
            if hasattr(self, 'scrollbar'):
                self.scrollbar.grid(row=1, column=1, sticky="ns", padx=(0, 4))
            self.progress_frame.grid(row=2, column=0, columnspan=2, sticky="ew", padx=12, pady=(4, 10))
            self.status_label.grid(row=3, column=0, columnspan=2, sticky="ew", padx=12, pady=(0, 8))
            self.btn_minimize.configure(text="➖ Minimize")
            self.btn_clear.grid(row=0, column=2, sticky="e")

        if self.on_minimize_toggle:
            self.on_minimize_toggle(self.is_minimized)

    def log(self, message: str, level: str = "info"):
        """Append a log message with timestamp and color coding."""
        self.text.configure(state="normal")
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.text.insert("end", f"[{timestamp}] ", "timestamp")

        # Auto-detect level from message content
        if level == "info":
            if any(kw in message.lower() for kw in ["error", "gagal", "failed"]):
                level = "error"
            elif any(kw in message.lower() for kw in ["selesai", "berhasil", "sukses", "done"]):
                level = "success"
            elif any(kw in message.lower() for kw in ["peringatan", "warning", "skip"]):
                level = "warning"
            elif message.startswith("[SAM]"):
                level = "sam"
            elif any(kw in message.lower() for kw in ["system", "cuda", "gpu", "memuat"]):
                level = "system"

        self.text.insert("end", message + "\n", level)
        self.text.configure(state="disabled")
        self.text.see("end")

    def clear(self):
        """Clear all log messages."""
        self.text.configure(state="normal")
        self.text.delete("1.0", "end")
        self.text.configure(state="disabled")

    def set_tile_progress(self, current: int, total: int):
        """Update tile-level progress bar."""
        if total > 0:
            self.tile_progress.set(current / total)
            self.tile_label.configure(text=f"{current}/{total}")
        else:
            self.tile_progress.set(0)
            self.tile_label.configure(text="0/0")

    def set_overall_progress(self, percent: int, message: str = ""):
        """Update overall progress bar (0–100)."""
        self.overall_progress.set(percent / 100)
        self.overall_label.configure(text=f"{percent}%")
        if message:
            self.status_label.configure(text=message)

    def reset(self):
        """Reset all progress bars and status."""
        self.tile_progress.set(0)
        self.tile_label.configure(text="0/0")
        self.overall_progress.set(0)
        self.overall_label.configure(text="0%")
        self.status_label.configure(text="Siap")

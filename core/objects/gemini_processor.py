"""
gemini_processor.py
Experimental processor menggunakan Gemini API (Vision) untuk mendeteksi dan mendigitasi
bangunan langsung ke poligon, menggantikan SAM & YOLO sepenuhnya.
"""

import os
import json
import numpy as np
from typing import Optional, Callable
from pathlib import Path


class GeminiProcessor:
    """
    Experimental processor to fully utilize Gemini API for detecting and digitizing
    buildings directly into polygons, bypassing SAM and YOLO entirely.
    """

    # Model yang didukung
    DEFAULT_MODEL = "gemini-3.1-flash-lite"

    def __init__(
        self,
        api_key: str,
        log_callback: Optional[Callable[[str], None]] = None,
        model_name: str = DEFAULT_MODEL,
    ):
        self.api_key = api_key
        self.model_name = model_name
        self.log_callback = log_callback or (lambda msg: print(msg))
        self._is_ready = False
        self.model = None

        if not self.api_key:
            self._log("⚠️ Gemini API Key kosong. Silakan masukkan API Key di UI.")
            return

        try:
            import google.generativeai as genai
            genai.configure(api_key=self.api_key)
            self.model = genai.GenerativeModel(self.model_name)
            self._is_ready = True
            self._log(f"✅ Gemini API siap menggunakan model: {self.model_name}")
        except ImportError:
            self._log("⚠️ Library 'google-generativeai' belum terinstal. Jalankan: pip install google-generativeai")
        except Exception as e:
            self._log(f"⚠️ Gagal inisialisasi Gemini API: {e}")

    def _log(self, msg: str):
        """Kirim pesan log."""
        self.log_callback(f"[GEMINI] {msg}")

    def is_ready(self) -> bool:
        return self._is_ready

    def _open_tile_as_rgb(self, tile_path: str):
        """
        Buka file tile GeoTIFF dan konversi ke PIL Image RGB.
        Menangani file multispektral (>3 band) maupun single-band.
        """
        import numpy as np
        from PIL import Image
        import rasterio
        from rasterio.enums import ColorInterp

        with rasterio.open(tile_path) as src:
            band_count = src.count
            if band_count >= 3:
                # Ambil 3 band pertama (asumsi RGB atau BGR)
                r = src.read(1).astype(np.float32)
                g = src.read(2).astype(np.float32)
                b = src.read(3).astype(np.float32)
            elif band_count == 1:
                # Grayscale → duplikasi ke 3 channel
                gray = src.read(1).astype(np.float32)
                r = g = b = gray
            else:
                # 2 band — ambil band 1 sebagai gray
                gray = src.read(1).astype(np.float32)
                r = g = b = gray

        def normalize(arr: np.ndarray) -> np.ndarray:
            lo, hi = arr.min(), arr.max()
            if hi == lo:
                return np.zeros_like(arr, dtype=np.uint8)
            return ((arr - lo) / (hi - lo) * 255).clip(0, 255).astype(np.uint8)

        rgb = np.dstack([normalize(r), normalize(g), normalize(b)])
        return Image.fromarray(rgb, mode="RGB")

    def process_tile(self, tile_path: str, output_mask_path: str) -> bool:
        """
        Kirim tile ke Gemini, dapatkan JSON koordinat poligon, lalu gambar menjadi binary mask
        (agar kompatibel dengan pipeline postprocess.py yang mengharapkan .tif mask).
        
        Returns True jika berhasil (termasuk jika tidak ada bangunan ditemukan).
        Returns False jika terjadi error fatal (error jaringan, format JSON salah, dll).
        """
        if not self._is_ready:
            self._log("Processor belum siap. Periksa API Key dan koneksi internet.")
            return False

        tile_name = Path(tile_path).name

        try:
            import rasterio
            import cv2

            # Dapatkan dimensi dan metadata asli dari tile
            with rasterio.open(tile_path) as src:
                h, w = src.height, src.width
                meta = src.meta.copy()

            # Update metadata output agar kompatibel dengan postprocess pipeline
            meta.update(dtype="uint16", count=1, nodata=0)

            # Buka citra sebagai RGB untuk dikirim ke Gemini
            self._log(f"Mengirim {tile_name} ke Gemini API ({w}×{h} px)...")
            img = self._open_tile_as_rgb(tile_path)

            # ── Prompt ──────────────────────────────────────────────────
            prompt = (
                "You are an expert GIS mapping assistant analyzing an aerial/satellite image tile. "
                "Your task: exhaustively detect and digitize EVERY SINGLE visible building (houses, factories, warehouses, any roofed structure). "
                f"Image size: {w}×{h} pixels. Pixel coordinate origin is top-left corner: x goes right, y goes down. "
                "There are likely dozens of buildings in this image. DO NOT STOP after finding just one or two. "
                "For EACH AND EVERY building found, provide the OUTER boundary polygon as an ordered list of [x, y] pixel coordinates. "
                "Trace the visible roof edge as precisely as possible, using 4–10 points per polygon. "
                "If NO buildings are visible, return an empty list for buildings. "
                "IMPORTANT: Return ONLY a raw JSON object. Do NOT wrap it in markdown code blocks. "
                "Required JSON format: "
                '{"buildings": [ [[x1,y1],[x2,y2],...], [[x1,y1],...] ]}'
            )

            # ── Panggil API Gemini ──────────────────────────────────────
            response = self.model.generate_content([prompt, img])
            res_text = response.text.strip()

            # Bersihkan artefak markdown jika ada
            if res_text.startswith("```json"):
                res_text = res_text[7:]
            if res_text.startswith("```"):
                res_text = res_text[3:]
            if res_text.endswith("```"):
                res_text = res_text[:-3]
            res_text = res_text.strip()

            # ── Parse JSON ──────────────────────────────────────────────
            try:
                data = json.loads(res_text)
            except json.JSONDecodeError as je:
                self._log(f"⚠️ Respons Gemini bukan JSON valid pada {tile_name}: {je}")
                self._log(f"   Respons mentah (50 char pertama): {res_text[:50]!r}")
                # Tulis mask kosong agar tile ini tidak menyebabkan error di postprocess
                self._write_empty_mask(output_mask_path, h, w, meta)
                return True  # bukan error fatal, lanjutkan tile berikutnya

            buildings = data.get("buildings", [])
            self._log(f"   Gemini menemukan {len(buildings)} bangunan di {tile_name}.")

            # ── Gambar poligon ke mask numpy ────────────────────────────
            master_mask = np.zeros((h, w), dtype=np.uint16)

            for idx, polygon_coords in enumerate(buildings):
                if len(polygon_coords) < 3:
                    continue  # poligon minimal 3 titik

                pts = np.array(polygon_coords, dtype=np.int32).reshape((-1, 1, 2))
                # Klem koordinat agar tidak melebihi batas citra
                pts[:, :, 0] = np.clip(pts[:, :, 0], 0, w - 1)
                pts[:, :, 1] = np.clip(pts[:, :, 1], 0, h - 1)

                building_id = idx + 1  # ID unik, tidak boleh 0 (nodata)
                cv2.fillPoly(master_mask, [pts], building_id)

            # ── Simpan mask ke GeoTIFF ───────────────────────────────────
            with rasterio.open(output_mask_path, "w", **meta) as dst:
                dst.write(master_mask.astype(np.uint16), 1)

            return True

        except Exception as e:
            self._log(f"❌ Error fatal memproses {tile_name}: {type(e).__name__}: {e}")
            return False

    def detect_boxes(self, tile_path: str) -> np.ndarray:
        """
        Kirim tile ke Gemini, dapatkan JSON bounding box bangunan.
        Berbeda dengan process_tile (yang mengembalikan poligon langsung),
        fungsi ini hanya mencari kotak [xmin, ymin, xmax, ymax] untuk
        diumpankan ke model SAM.
        
        Returns:
            np.ndarray berukuran (N, 4) berisi [xmin, ymin, xmax, ymax].
            Mengembalikan array kosong jika tidak ada bangunan atau terjadi error.
        """
        if not self._is_ready:
            self._log("Processor belum siap. Periksa API Key dan koneksi internet.")
            return np.array([])

        tile_name = Path(tile_path).name

        try:
            import rasterio

            with rasterio.open(tile_path) as src:
                h, w = src.height, src.width

            self._log(f"Mendeteksi kotak pembatas pada {tile_name} ({w}×{h} px)...")
            img = self._open_tile_as_rgb(tile_path)

            prompt = (
                "You are an expert GIS mapping assistant analyzing an aerial/satellite image tile. "
                "Your task: exhaustively detect EVERY SINGLE visible building (houses, factories, warehouses, any roofed structure). "
                f"Image size: {w}×{h} pixels. Pixel coordinate origin is top-left corner: x goes right, y goes down. "
                "There are likely dozens of buildings in this image. DO NOT STOP after finding just one or two. "
                "For EACH AND EVERY building found, provide its bounding box as [xmin, ymin, xmax, ymax]. "
                "If NO buildings are visible, return an empty list for boxes. "
                "IMPORTANT: Return ONLY a raw JSON object. Do NOT wrap it in markdown code blocks. "
                "Required JSON format: "
                '{"boxes": [ [xmin, ymin, xmax, ymax], [xmin, ymin, xmax, ymax], ... ]}'
            )

            response = self.model.generate_content([prompt, img])
            res_text = response.text.strip()

            # Bersihkan artefak markdown jika ada
            if res_text.startswith("```json"):
                res_text = res_text[7:]
            if res_text.startswith("```"):
                res_text = res_text[3:]
            if res_text.endswith("```"):
                res_text = res_text[:-3]
            res_text = res_text.strip()

            try:
                data = json.loads(res_text)
            except json.JSONDecodeError as je:
                self._log(f"⚠️ Respons Gemini bukan JSON valid pada {tile_name}: {je}")
                return np.array([])

            boxes = data.get("boxes", [])
            self._log(f"   Gemini menemukan {len(boxes)} kotak bangunan di {tile_name}.")

            if len(boxes) == 0:
                return np.array([])

            boxes_np = np.array(boxes, dtype=np.float32)
            
            # Validasi ukuran: harus N x 4
            if boxes_np.ndim == 1 and len(boxes_np) == 4:
                boxes_np = boxes_np.reshape(1, 4)
            elif boxes_np.ndim != 2 or boxes_np.shape[1] != 4:
                self._log(f"⚠️ Format array kotak tidak valid: {boxes_np.shape}")
                return np.array([])

            # Klem koordinat agar tidak melebihi batas citra
            boxes_np[:, 0] = np.clip(boxes_np[:, 0], 0, w - 1)
            boxes_np[:, 1] = np.clip(boxes_np[:, 1], 0, h - 1)
            boxes_np[:, 2] = np.clip(boxes_np[:, 2], 0, w - 1)
            boxes_np[:, 3] = np.clip(boxes_np[:, 3], 0, h - 1)

            # Validasi xmin < xmax dan ymin < ymax
            valid_idx = (boxes_np[:, 2] > boxes_np[:, 0]) & (boxes_np[:, 3] > boxes_np[:, 1])
            boxes_np = boxes_np[valid_idx]

            return boxes_np

        except Exception as e:
            self._log(f"❌ Error fatal mendeteksi kotak pada {tile_name}: {type(e).__name__}: {e}")
            return np.array([])

    def _write_empty_mask(self, output_mask_path: str, h: int, w: int, meta: dict):
        """Tulis mask kosong (all-zeros) agar tile tidak menyebabkan error postprocess."""
        try:
            import rasterio
            empty = np.zeros((h, w), dtype=np.uint16)
            with rasterio.open(output_mask_path, "w", **meta) as dst:
                dst.write(empty, 1)
        except Exception:
            pass  # Jika bahkan ini gagal, biarkan saja

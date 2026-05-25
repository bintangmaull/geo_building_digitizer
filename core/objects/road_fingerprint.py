"""
road_fingerprint.py
Fase 1–4 dari pipeline digitasi jalan:

  Fase 1 — Sampling    : Piksel dari dalam/luar polygon SHP jalan referensi
  Fase 2 — Normalisasi : Percentile stretch adaptif (bukan hardcode warna)
  Fase 3 — Feature Ext : HSV, ExG, texture (local std), brightness
  Fase 4 — Fingerprint : MinCovDet robust covariance → Mahalanobis road profile

Output: road_fingerprint.json — bisa dipakai ulang antar proyek tanpa SHP lagi.

CATATAN: Modul ini berdiri sendiri, tidak bergantung pada postprocess.py
         (pipeline bangunan tidak disentuh).
"""

import numpy as np
import json
import warnings
import threading
from pathlib import Path
from typing import Callable, Optional, Tuple

warnings.filterwarnings("ignore")


class RoadFingerprintBuilder:
    """
    Builder untuk Road Fingerprint dari pasangan citra referensi + SHP polygon jalan.

    Cara pemakaian:
        builder = RoadFingerprintBuilder(log_callback=..., progress_callback=...)
        result  = builder.build(raster_path, shp_path, output_path)
    """

    def __init__(
        self,
        max_samples: int = 50_000,
        texture_ksize: int = 7,
        percentile_low: float = 2.0,
        percentile_high: float = 98.0,
        log_callback: Optional[Callable[[str], None]] = None,
        progress_callback: Optional[Callable[[int, str], None]] = None,
    ):
        self.max_samples = max_samples
        self.texture_ksize = texture_ksize
        self.percentile_low = percentile_low
        self.percentile_high = percentile_high
        self.log_callback = log_callback or (lambda msg: print(msg))
        self.progress_callback = progress_callback or (lambda pct, msg: None)
        self._cancel_event = threading.Event()

    def cancel(self):
        self._cancel_event.set()

    def is_cancelled(self) -> bool:
        return self._cancel_event.is_set()

    def reset_cancel(self):
        self._cancel_event.clear()

    def _log(self, msg: str):
        self.log_callback(f"[Fingerprint] {msg}")

    def _progress(self, pct: int, msg: str):
        self.progress_callback(pct, msg)

    # ══════════════════════════════════════════════════════════════════════════
    # FASE 1: SAMPLING
    # ══════════════════════════════════════════════════════════════════════════

    def sample_pixels(
        self,
        raster_path: str,
        shp_path: str,
    ) -> Tuple[np.ndarray, np.ndarray, dict, np.ndarray, np.ndarray]:
        """
        Ekstrak piksel di dalam polygon SHP (jalan) dan di luar (non-jalan).

        Returns:
            road_idx       : (N,) index flat piksel jalan
            nonroad_idx    : (M,) index flat piksel non-jalan
            meta           : dict metadata raster
            img_raw        : (H, W, C) citra asli float32 (sudah di-crop ke bounding box SHP)
            road_mask      : (H, W) boolean True = piksel jalan
        """
        try:
            import rasterio
            from rasterio.mask import mask as rasterio_mask
            import geopandas as gpd
            from shapely.geometry import mapping
        except ImportError as e:
            raise ImportError(f"Dependensi tidak ditemukan: {e}. "
                              f"Install: pip install rasterio geopandas shapely")

        self._log("Fase 1: Sampling piksel dari citra referensi ...")
        self._progress(10, "Membaca citra referensi & SHP jalan ...")

        with rasterio.open(raster_path) as src:
            meta = {
                "crs":       str(src.crs),
                "transform": list(src.transform),
                "count":     src.count,
                "dtype":     str(src.dtypes[0]),
                "nodata":    src.nodata,
            }

            # Baca polygon jalan dari SHP
            gdf = gpd.read_file(shp_path).to_crs(src.crs)
            shapes = [geom for geom in gdf.geometry if geom is not None]

            if not shapes:
                raise ValueError(f"SHP tidak memiliki geometri valid: {shp_path}")

            # Dapatkan bounding box total dari SHP
            minx, miny, maxx, maxy = gdf.total_bounds
            
            # Tambahkan buffer ~200 piksel agar mendapat sampel background (non-jalan)
            px_width, px_height = src.res
            buffer_x = px_width * 200
            buffer_y = px_height * 200
            
            minx = max(src.bounds.left, minx - buffer_x)
            miny = max(src.bounds.bottom, miny - buffer_y)
            maxx = min(src.bounds.right, maxx + buffer_x)
            maxy = min(src.bounds.top, maxy + buffer_y)

            from rasterio.windows import from_bounds, Window
            import rasterio.windows
            window = from_bounds(minx, miny, maxx, maxy, src.transform)
            # Pastikan window tidak keluar dari batas citra
            full_window = Window(0, 0, src.width, src.height)
            window = window.intersection(full_window)

            # Baca citra yang sudah di-crop & di-buffer
            n_bands = min(src.count, 3)
            img = src.read(list(range(1, n_bands + 1)), window=window)
            img = np.moveaxis(img, 0, -1).astype(np.float32)  # (H, W, C)
            
            out_transform = rasterio.windows.transform(window, src.transform)

            # Buat road mask eksak dari SHP di dalam window
            from rasterio.features import geometry_mask
            road_mask = geometry_mask(
                shapes,
                transform=out_transform,
                invert=True,  # True = piksel di dalam polygon jalan
                out_shape=(img.shape[0], img.shape[1])
            )
            
            # Update meta
            meta.update({
                "height": img.shape[0],
                "width": img.shape[1],
                "transform": out_transform
            })

        if self.is_cancelled():
            return None, None, None, None, None

        H, W, C = img.shape
        flat_img  = img.reshape(-1, C)
        flat_mask = road_mask.reshape(-1)

        # Handle nodata
        if meta["nodata"] is not None:
            valid = ~np.any(flat_img == meta["nodata"], axis=1)
        else:
            valid = np.all(flat_img > 0, axis=1)

        road_idx    = np.where(flat_mask & valid)[0]
        nonroad_idx = np.where(~flat_mask & valid)[0]

        if len(road_idx) == 0:
            raise ValueError("Tidak ada piksel jalan ditemukan di dalam polygon SHP. "
                             "Pastikan SHP berada di area yang sama dengan citra referensi.")

        # Subsample
        rng = np.random.default_rng(42)
        if len(road_idx) > self.max_samples:
            road_idx = rng.choice(road_idx, self.max_samples, replace=False)
        if len(nonroad_idx) > self.max_samples:
            nonroad_idx = rng.choice(nonroad_idx, self.max_samples, replace=False)

        road_pixels    = flat_img[road_idx]
        nonroad_pixels = flat_img[nonroad_idx]

        self._log(f"  ✓ Piksel jalan    : {len(road_idx):,}")
        self._log(f"  ✓ Piksel non-jalan: {len(nonroad_idx):,}")
        self._log(f"  ✓ Ukuran crop     : {W}x{H} piksel")
        self._log(f"  ✓ Band digunakan  : {C} (RGB)")

        return road_idx, nonroad_idx, meta, img, road_mask

    # ══════════════════════════════════════════════════════════════════════════
    # FASE 2: NORMALISASI
    # ══════════════════════════════════════════════════════════════════════════

    def compute_norm_params(self, img: np.ndarray) -> dict:
        """
        Hitung parameter normalisasi percentile stretch dari citra.
        Adaptif — tidak bergantung pada nilai warna yang di-hardcode.

        Returns:
            norm_params: {"band_0": {"p_low": ..., "p_high": ...}, ...}
        """
        self._log("Fase 2: Menghitung parameter normalisasi (percentile stretch) ...")
        self._progress(25, "Menghitung normalisasi citra ...")

        C = img.shape[2]
        norm_params = {}
        for c in range(C):
            band = img[:, :, c].ravel()
            band = band[band > 0]
            p_low  = float(np.percentile(band, self.percentile_low))
            p_high = float(np.percentile(band, self.percentile_high))
            norm_params[f"band_{c}"] = {"p_low": p_low, "p_high": p_high}
            self._log(f"  Band {c}: [{p_low:.1f}, {p_high:.1f}]")

        return norm_params

    @staticmethod
    def normalize_image(img: np.ndarray, norm_params: dict) -> np.ndarray:
        """
        Terapkan percentile stretch ke citra.
        Bisa dipanggil secara static saat scoring citra target.
        """
        img_norm = img.copy().astype(np.float32)
        C = img.shape[2]
        for c in range(C):
            key = f"band_{c}"
            if key not in norm_params:
                continue
            p_low  = norm_params[key]["p_low"]
            p_high = norm_params[key]["p_high"]
            img_norm[:, :, c] = (
                np.clip((img[:, :, c] - p_low) / (p_high - p_low + 1e-6), 0, 1) * 255
            )
        return img_norm

    # ══════════════════════════════════════════════════════════════════════════
    # FASE 3: FEATURE EXTRACTION
    # ══════════════════════════════════════════════════════════════════════════

    @staticmethod
    def extract_features(
        img_norm: np.ndarray,
        texture_ksize: int = 7,
    ) -> Tuple[np.ndarray, list]:
        """
        Ekstrak 6 fitur per piksel dari citra ternormalisasi:
          - hue, saturation, value (HSV)
          - brightness (mean RGB)
          - exg (Excess Green Index = 2G - R - B)
          - texture (local std dev via sliding window)

        Returns:
            features   : (H, W, 6) float32
            feat_names : list of str
        """
        try:
            import cv2
        except ImportError:
            raise ImportError("opencv-python diperlukan. Install: pip install opencv-python")

        H, W, C = img_norm.shape
        img_uint8 = img_norm.astype(np.uint8)

        # HSV
        hsv      = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2HSV).astype(np.float32)
        feat_H   = hsv[:, :, 0]
        feat_S   = hsv[:, :, 1]
        feat_V   = hsv[:, :, 2]

        # Brightness (mean RGB)
        feat_brightness = img_norm.mean(axis=2)

        # ExG = 2G - R - B
        R = img_norm[:, :, 0]
        G = img_norm[:, :, 1]
        B = img_norm[:, :, 2] if C >= 3 else np.zeros((H, W), np.float32)
        feat_exg = 2.0 * G - R - B

        # Texture: local std dev via sliding window
        gray    = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2GRAY).astype(np.float32)
        gray_sq = gray ** 2
        kernel  = np.ones((texture_ksize, texture_ksize), np.float32) / (texture_ksize ** 2)
        mean_   = cv2.filter2D(gray, -1, kernel)
        mean_sq = cv2.filter2D(gray_sq, -1, kernel)
        feat_texture = np.sqrt(np.maximum(mean_sq - mean_ ** 2, 0))

        features = np.stack([
            feat_H, feat_S, feat_V,
            feat_brightness, feat_exg, feat_texture,
        ], axis=-1)  # (H, W, 6)

        feat_names = ["hue", "saturation", "value", "brightness", "exg", "texture"]
        return features, feat_names

    @staticmethod
    def get_feature_pixels(features: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Ambil piksel fitur berdasarkan boolean mask."""
        H, W, F = features.shape
        flat_feat = features.reshape(-1, F)
        flat_mask = mask.reshape(-1)
        return flat_feat[flat_mask]

    @staticmethod
    def extract_features_sampled(
        img_norm: np.ndarray,
        road_idx: np.ndarray,
        nonroad_idx: np.ndarray,
        texture_ksize: int = 7,
    ) -> Tuple[np.ndarray, np.ndarray, list]:
        """
        Ekstrak 6 fitur hanya untuk indeks piksel yang disampel guna menghemat memori.
        Mencegah Out-of-Memory (OOM) pada citra resolusi sangat tinggi.
        """
        try:
            import cv2
        except ImportError:
            raise ImportError("opencv-python diperlukan. Install: pip install opencv-python")

        H, W, C = img_norm.shape
        img_uint8 = img_norm.astype(np.uint8)
        
        flat_img = img_norm.reshape(-1, C)
        road_px = flat_img[road_idx]
        nonroad_px = flat_img[nonroad_idx]

        def compute_pointwise(pixels):
            if len(pixels) == 0:
                return np.zeros((0, 5), dtype=np.float32)
            N = len(pixels)
            px_uint8 = np.clip(pixels, 0, 255).astype(np.uint8).reshape(-1, 1, 3)
            hsv = cv2.cvtColor(px_uint8, cv2.COLOR_RGB2HSV).astype(np.float32).reshape(-1, 3)
            feat_H = hsv[:, 0]
            feat_S = hsv[:, 1]
            feat_V = hsv[:, 2]
            feat_brightness = pixels.mean(axis=1)
            R = pixels[:, 0]
            G = pixels[:, 1]
            B = pixels[:, 2] if C >= 3 else np.zeros(N, np.float32)
            feat_exg = 2.0 * G - R - B
            return np.stack([feat_H, feat_S, feat_V, feat_brightness, feat_exg], axis=-1)

        road_base = compute_pointwise(road_px)
        nonroad_base = compute_pointwise(nonroad_px)

        # Texture dihitung hanya pada koordinat piksel sampel
        gray = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2GRAY).astype(np.float32)

        def compute_texture(indices):
            texture = np.zeros(len(indices), dtype=np.float32)
            half = texture_ksize // 2
            rows = indices // W
            cols = indices % W
            for i in range(len(indices)):
                r = rows[i]
                c = cols[i]
                r0, r1 = max(0, r-half), min(H, r+half+1)
                c0, c1 = max(0, c-half), min(W, c+half+1)
                patch = gray[r0:r1, c0:c1]
                if patch.size > 0:
                    texture[i] = np.std(patch)
            return texture

        road_tex = compute_texture(road_idx)
        nonroad_tex = compute_texture(nonroad_idx)

        road_feat = np.column_stack([road_base, road_tex])
        nonroad_feat = np.column_stack([nonroad_base, nonroad_tex])

        feat_names = ["hue", "saturation", "value", "brightness", "exg", "texture"]
        return road_feat, nonroad_feat, feat_names

    # ══════════════════════════════════════════════════════════════════════════
    # FASE 4: ROAD FINGERPRINT
    # ══════════════════════════════════════════════════════════════════════════

    def compute_fingerprint(
        self,
        road_feat: np.ndarray,
        nonroad_feat: np.ndarray,
        feat_names: list,
    ) -> dict:
        """
        Hitung Road Fingerprint:
        - Statistik distribusi per fitur (mean, std, percentile)
        - Separability score tiap fitur (berguna untuk debugging)
        - Robust covariance matrix (MinCovDet) untuk Mahalanobis distance

        Returns:
            fingerprint dict (serializable ke JSON)
        """
        try:
            from sklearn.covariance import MinCovDet
        except ImportError:
            raise ImportError("scikit-learn diperlukan. Install: pip install scikit-learn")

        self._log("Fase 4: Menghitung Road Fingerprint ...")
        self._progress(65, "Menghitung distribusi fitur & covariance ...")

        fingerprint = {
            "features":    feat_names,
            "per_feature": {},
            "mahalanobis": {},
        }

        # ── Per-feature statistics ──────────────────────────────────────────
        for i, name in enumerate(feat_names):
            road_vals    = road_feat[:, i]
            nonroad_vals = nonroad_feat[:, i]

            sep = float(
                abs(np.mean(road_vals) - np.mean(nonroad_vals)) /
                (np.std(road_vals) + np.std(nonroad_vals) + 1e-6)
            )

            fingerprint["per_feature"][name] = {
                "road": {
                    "mean": float(np.mean(road_vals)),
                    "std":  float(np.std(road_vals)),
                    "p05":  float(np.percentile(road_vals, 5)),
                    "p25":  float(np.percentile(road_vals, 25)),
                    "p50":  float(np.percentile(road_vals, 50)),
                    "p75":  float(np.percentile(road_vals, 75)),
                    "p95":  float(np.percentile(road_vals, 95)),
                },
                "nonroad": {
                    "mean": float(np.mean(nonroad_vals)),
                    "std":  float(np.std(nonroad_vals)),
                },
                "separability": sep,
            }

            self._log(
                f"  {name:12s} | road_mean={np.mean(road_vals):7.2f} "
                f"| nonroad_mean={np.mean(nonroad_vals):7.2f} "
                f"| separability={sep:.3f}"
            )

        # ── Ringkasan separability ──────────────────────────────────────────
        seps = {k: v["separability"] for k, v in fingerprint["per_feature"].items()}
        best = max(seps, key=seps.get)
        self._log(f"  ★ Fitur paling diskriminatif: '{best}' (sep={seps[best]:.3f})")

        low_sep = [k for k, v in seps.items() if v < 0.3]
        if low_sep:
            self._log(f"  ⚠ Fitur separabilitas rendah (<0.3): {low_sep}")

        # ── Robust Mahalanobis covariance (tahan outlier) ──────────────────
        self._log("  Menghitung robust covariance matrix (MinCovDet) ...")
        self._progress(80, "Menghitung robust covariance matrix ...")

        try:
            mcd = MinCovDet(support_fraction=0.75, random_state=42)
            mcd.fit(road_feat)
            fingerprint["mahalanobis"] = {
                "location":         mcd.location_.tolist(),
                "precision_matrix": mcd.precision_.tolist(),
            }
            self._log("  ✓ Robust covariance berhasil (MinCovDet)")
        except Exception as ex:
            self._log(f"  ⚠ MinCovDet gagal ({ex}), fallback ke standard covariance")
            mean_vec = np.mean(road_feat, axis=0)
            cov      = np.cov(road_feat.T)
            prec     = np.linalg.pinv(cov)
            fingerprint["mahalanobis"] = {
                "location":         mean_vec.tolist(),
                "precision_matrix": prec.tolist(),
            }

        return fingerprint

    # ══════════════════════════════════════════════════════════════════════════
    # PIPELINE UTAMA: BUILD (Fase 1–4)
    # ══════════════════════════════════════════════════════════════════════════

    def build(
        self,
        raster_path: str,
        shp_path: str,
        output_path: str = "road_fingerprint.json",
    ) -> dict:
        """
        Jalankan Fase 1–4 secara lengkap dan simpan fingerprint ke JSON.

        Args:
            raster_path : Path citra referensi (GeoTIFF / ECW)
            shp_path    : Path SHP polygon jalan referensi
            output_path : Path output JSON (default: road_fingerprint.json)

        Returns:
            output dict (yang sama yang disimpan ke JSON)
        """
        self._log("=" * 55)
        self._log("  ROAD FINGERPRINT BUILDER")
        self._log("=" * 55)
        self._progress(5, "Memulai pembangunan Road Fingerprint ...")

        # Fase 1
        result = self.sample_pixels(raster_path, shp_path)
        if self.is_cancelled() or result[0] is None:
            self._log("Dibatalkan.")
            return {}
        road_idx, nonroad_idx, meta, img_raw, road_mask = result

        # Fase 2
        norm_params = self.compute_norm_params(img_raw)
        if self.is_cancelled():
            return {}
        img_norm = RoadFingerprintBuilder.normalize_image(img_raw, norm_params)

        # Fase 3
        self._log("Fase 3: Mengekstrak fitur visual secara efisien ...")
        self._progress(40, "Ekstraksi fitur (HSV, ExG, texture) ...")
        
        # Hemat memori: hanya ekstrak untuk piksel yang disampel
        road_feat, nonroad_feat, feat_names = self.extract_features_sampled(
            img_norm, road_idx, nonroad_idx, self.texture_ksize
        )
        self._log(f"  ✓ Fitur diekstrak: {feat_names}")

        if self.is_cancelled():
            return {}

        # Fase 4
        fingerprint = self.compute_fingerprint(road_feat, nonroad_feat, feat_names)
        if self.is_cancelled():
            return {}

        # Simpan JSON
        self._progress(90, "Menyimpan road_fingerprint.json ...")
        output = {
            "version":        "1.0",
            "description":    "Road Fingerprint — karakteristik visual jalan untuk deteksi otomatis",
            "source_raster":  str(raster_path),
            "source_shp":     str(shp_path),
            "raster_meta":    meta,
            "normalization":  norm_params,
            "fingerprint":    fingerprint,
        }

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(output, f, indent=2)

        self._log("=" * 55)
        self._log(f"  ✅ Fingerprint disimpan ke: {output_path}")
        self._log("=" * 55)
        self._progress(100, f"Fingerprint selesai: {Path(output_path).name}")

        return output

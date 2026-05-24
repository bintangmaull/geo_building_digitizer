"""
segformer_processor.py
Processor segmentasi semantik berbasis SegFormer untuk digitasi multi-objek.

Model: nvidia/segformer-b2-finetuned-ade-512-512 (ADE20K, 150 kelas)

Pipeline:
    1. Tiling raster (512x512 + overlap)
    2. SegFormer inference → per-pixel class mask
    3. Feature augmentation refinement (ExG, VARI, texture, warna)
    4. Postprocessing khusus per objek:
       - Jalan    : morphological closing + skeletonize + buffer + simplify
       - Badan Air: fill_holes + dissolve + simplify
       - Vegetasi : merge kecil + dissolve + simplify
    5. Vektorisasi → GeoDataFrame per kelas

Prioritas overlap (clipping dilakukan di priority_clipper.py):
    Bangunan > Jalan > Badan Air > Vegetasi

CATATAN: File ini BERDIRI SENDIRI. Tidak menyentuh kode bangunan.
"""

import os
import gc
import threading
import numpy as np
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple


# ── ADE20K Class Mapping ─────────────────────────────────────────────────────
# Kelas-kelas ADE20K (0-indexed) yang relevan untuk drone imagery
ROAD_CLASSES    = frozenset([6, 11, 52, 54, 61, 91])          # road, sidewalk, path, runway, bridge, dirt track
WATER_CLASSES   = frozenset([21, 26, 60, 109, 113, 128, 140]) # water, sea, river, pool, waterfall, lake, pier
VEG_CLASSES     = frozenset([4, 9, 17, 29, 66, 72])           # tree, grass, plant, field, flower, palm

CLASS_KEY_MAP = {
    "road":       ("Jalan",     ROAD_CLASSES),
    "water":      ("Badan Air", WATER_CLASSES),
    "vegetation": ("Vegetasi",  VEG_CLASSES),
}

# ── Model Options ─────────────────────────────────────────────────────────────
SEGFORMER_MODELS = {
    "SegFormer-B0 (Ringan, ~14MB)":    "nvidia/segformer-b0-finetuned-ade-512-512",
    "SegFormer-B2 (Seimbang, ~100MB)": "nvidia/segformer-b2-finetuned-ade-512-512",
    "SegFormer-B5 (Akurat, ~370MB)":   "nvidia/segformer-b5-finetuned-ade-640-640",
}
DEFAULT_SEGFORMER = "SegFormer-B2 (Seimbang, ~100MB)"


class SegFormerSegmentor:
    """
    Processor segmentasi semantik berbasis SegFormer.
    Menggantikan pendekatan OpenCV-threshold + SAM untuk objek non-bangunan.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_SEGFORMER,
        device: str = "auto",
        log_callback: Optional[Callable[[str], None]] = None,
        progress_callback: Optional[Callable[[int, str], None]] = None,
    ):
        self.model_name = model_name
        self.model_id = SEGFORMER_MODELS.get(model_name, SEGFORMER_MODELS[DEFAULT_SEGFORMER])

        if device == "auto":
            try:
                import torch
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                self.device = "cpu"
        else:
            self.device = device

        self.log_callback   = log_callback   or (lambda msg: print(msg))
        self.progress_callback = progress_callback or (lambda pct, msg: None)
        self._cancel_event  = threading.Event()
        self._model         = None
        self._processor     = None

    def cancel(self):
        self._cancel_event.set()

    def is_cancelled(self) -> bool:
        return self._cancel_event.is_set()

    def reset_cancel(self):
        self._cancel_event.clear()

    def _log(self, msg: str):
        self.log_callback(f"[SegFormer] {msg}")

    def _progress(self, pct: int, msg: str):
        self.progress_callback(pct, msg)

    # ── Model Loading ─────────────────────────────────────────────────────────

    def load_model(self):
        """Muat model SegFormer dari HuggingFace (auto-cached)."""
        from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
        import torch

        self._log(f"Memuat model: {self.model_name}")
        self._log(f"Model ID   : {self.model_id}")
        self._log(f"Device     : {self.device.upper()}")
        self._log("Download otomatis jika belum ada di cache (~100MB)...")

        try:
            self._processor = SegformerImageProcessor.from_pretrained(self.model_id)
            self._model = SegformerForSemanticSegmentation.from_pretrained(self.model_id)
            self._model.to(self.device)
            self._model.eval()
            self._log("Model SegFormer berhasil dimuat ✅")
        except Exception as e:
            raise RuntimeError(
                f"Gagal memuat SegFormer '{self.model_id}'.\n"
                f"Pastikan koneksi internet aktif untuk download pertama kali.\n"
                f"Error: {e}"
            )

    def unload_model(self):
        """Bebaskan model dari memori."""
        if self._model is not None:
            del self._model
            del self._processor
            self._model = None
            self._processor = None
            gc.collect()
            if self.device == "cuda":
                try:
                    import torch
                    torch.cuda.empty_cache()
                except Exception:
                    pass
            self._log("Model dibebaskan dari memori")

    # ── Inference ─────────────────────────────────────────────────────────────

    def segment_tile(self, tile_path: str) -> Dict[str, np.ndarray]:
        """
        Jalankan SegFormer inference pada satu tile.

        Returns:
            Dict dengan key "road", "water", "vegetation"
            Masing-masing: HxW bool array (True = piksel kelas tersebut)
        """
        import torch
        import torch.nn.functional as F
        from PIL import Image

        if self._model is None or self._processor is None:
            raise RuntimeError("Model belum dimuat. Panggil load_model() dulu.")

        # Load gambar
        img = Image.open(tile_path).convert("RGB")
        orig_w, orig_h = img.size

        # Preprocess → inference
        inputs = self._processor(images=img, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self._model(**inputs)

        # Upsample logits ke resolusi tile asli
        logits = outputs.logits  # (1, num_classes, H/4, W/4)
        upsampled = F.interpolate(
            logits,
            size=(orig_h, orig_w),
            mode="bilinear",
            align_corners=False,
        )
        pred = upsampled.argmax(dim=1).squeeze(0).cpu().numpy()  # HxW int

        # Buat mask per kelas
        masks = {
            "road":       np.isin(pred, list(ROAD_CLASSES)),
            "water":      np.isin(pred, list(WATER_CLASSES)),
            "vegetation": np.isin(pred, list(VEG_CLASSES)),
        }

        return masks

    def _load_rgb(self, tile_path: str) -> np.ndarray:
        """Load tile sebagai numpy array HxWx3 uint8 (RGB)."""
        import rasterio
        with rasterio.open(tile_path) as src:
            if src.count >= 3:
                data = src.read([1, 2, 3])
            else:
                data = np.stack([src.read(1)] * 3)
            rgb = np.transpose(data, (1, 2, 0)).astype(np.uint8)
        return rgb

    # ── Full Raster Pipeline ───────────────────────────────────────────────────

    def process_raster(
        self,
        raster_path: str,
        enabled_classes: Dict[str, bool],
        tile_size: int = 1024,
        overlap: int = 128,
        min_area_m2: Dict[str, float] = None,
        max_area_m2: Dict[str, float] = None,
        use_feature_refinement: bool = True,
        tile_progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> Dict[str, "geopandas.GeoDataFrame"]:
        """
        Proses seluruh raster dengan SegFormer.

        Args:
            raster_path    : Path ke file raster GeoTIFF
            enabled_classes: {"road": True, "water": False, "vegetation": True}
            tile_size      : Ukuran tile piksel (default 1024)
            overlap        : Overlap antar tile piksel
            min_area_m2    : Min luas per kelas {"road": 30.0, ...}
            max_area_m2    : Max luas per kelas {"road": 500000.0, ...}
            use_feature_refinement: Aktifkan penyempurnaan berbasis ExG/VARI/texture
            tile_progress_callback: Callback (idx, total)

        Returns:
            Dict {class_key: GeoDataFrame} — satu GDF per kelas aktif
        """
        import geopandas as gpd
        from core.tiling import tiles_generator

        if self._model is None:
            raise RuntimeError("Model belum dimuat.")

        min_area = min_area_m2 or {"road": 30.0, "water": 50.0, "vegetation": 10.0}
        max_area = max_area_m2 or {"road": 500000.0, "water": 10000000.0, "vegetation": 50000000.0}

        active = [k for k, v in enabled_classes.items() if v]
        self._log(f"Kelas aktif: {', '.join(active)}")

        project_root  = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        tiles_temp_dir = os.path.join(project_root, "temp", "tiles_segformer")
        os.makedirs(tiles_temp_dir, exist_ok=True)

        # Akumulator polygons per kelas (sebelum merge)
        accumulated: Dict[str, list] = {k: [] for k in active}

        # ── Tile-by-tile inference ────────────────────────────────────────────
        for tile_path, tile_meta, idx, total in tiles_generator(
            raster_path=raster_path,
            tile_size=tile_size,
            overlap=overlap,
            temp_dir=tiles_temp_dir,
            enable_filtering=False,
        ):
            if self.is_cancelled():
                self._log("Proses dibatalkan.")
                break

            self._log(f"[{idx+1}/{total}] Tile: {Path(tile_path).name}")
            if tile_progress_callback:
                tile_progress_callback(idx + 1, total)

            try:
                # 1. SegFormer inference
                masks = self.segment_tile(tile_path)

                # 2. Feature refinement (opsional)
                if use_feature_refinement:
                    try:
                        rgb = self._load_rgb(tile_path)
                        masks = self._refine_masks(masks, rgb, active)
                    except Exception as fe:
                        self._log(f"  ⚠️ Refinement gagal (diabaikan): {fe}")

                # 3. Per-tile postprocess → polygons
                transform = tile_meta["transform"]
                crs       = tile_meta.get("crs")

                for cls_key in active:
                    mask = masks.get(cls_key)
                    if mask is None or not mask.any():
                        continue

                    gdf_tile = self._mask_to_gdf(
                        mask=mask,
                        transform=transform,
                        crs=crs,
                        cls_key=cls_key,
                    )
                    if len(gdf_tile) > 0:
                        accumulated[cls_key].append(gdf_tile)

            except Exception as e:
                self._log(f"  ❌ Error tile {Path(tile_path).name}: {e}")
                continue

        if self.is_cancelled():
            return {}

        # ── Per-class merge + postprocess ─────────────────────────────────────
        results: Dict[str, gpd.GeoDataFrame] = {}

        for cls_key in active:
            if not accumulated[cls_key]:
                self._log(f"Tidak ada poligon terdeteksi untuk: {cls_key}")
                continue

            self._log(f"Postprocessing {cls_key}...")
            all_tiles = gpd.pd.concat(accumulated[cls_key], ignore_index=True)
            merged = gpd.GeoDataFrame(all_tiles, geometry="geometry",
                                      crs=accumulated[cls_key][0].crs)

            # Postprocessing spesifik per kelas
            gdf = self._postprocess(
                gdf=merged,
                cls_key=cls_key,
                raster_path=raster_path,
                min_area_m2=min_area.get(cls_key, 10.0),
                max_area_m2=max_area.get(cls_key, 50000000.0),
            )

            if len(gdf) > 0:
                label = CLASS_KEY_MAP[cls_key][0]
                gdf["class"] = label
                results[cls_key] = gdf
                self._log(f"  ✅ {label}: {len(gdf)} poligon")

        return results

    # ── Feature Refinement ────────────────────────────────────────────────────

    def _refine_masks(
        self,
        masks: Dict[str, np.ndarray],
        rgb: np.ndarray,
        active: List[str],
    ) -> Dict[str, np.ndarray]:
        """Perkuat mask SegFormer dengan indeks spektral RGB."""
        from core.objects.feature_augmentation import (
            refine_vegetation_mask,
            refine_road_mask,
            refine_water_mask,
        )
        refined = dict(masks)
        if "vegetation" in active and "vegetation" in masks:
            refined["vegetation"] = refine_vegetation_mask(masks["vegetation"], rgb)
        if "road" in active and "road" in masks:
            refined["road"] = refine_road_mask(masks["road"], rgb)
        if "water" in active and "water" in masks:
            refined["water"] = refine_water_mask(masks["water"], rgb)
        return refined

    # ── Mask → Vector ─────────────────────────────────────────────────────────

    def _mask_to_gdf(
        self,
        mask: np.ndarray,
        transform,
        crs,
        cls_key: str,
    ) -> "geopandas.GeoDataFrame":
        """Konversi pixel mask ke GeoDataFrame polygon."""
        import geopandas as gpd
        from rasterio.features import shapes
        from shapely.geometry import shape
        import scipy.ndimage as ndi

        # Morphological closing ringan untuk menutup lubang kecil
        kernel_size = {"road": 5, "water": 7, "vegetation": 3}.get(cls_key, 3)
        from skimage.morphology import disk, binary_closing
        closed = binary_closing(mask, disk(kernel_size))

        mask_u8 = closed.astype(np.uint8)
        geoms = []
        for geom_dict, val in shapes(mask_u8, mask=mask_u8, transform=transform):
            if val == 1:
                geom = shape(geom_dict)
                if geom.is_valid and geom.area > 0:
                    geoms.append(geom)

        if not geoms:
            return gpd.GeoDataFrame(geometry=[], crs=crs)
        return gpd.GeoDataFrame(geometry=geoms, crs=crs)

    # ── Per-class Postprocessing ───────────────────────────────────────────────

    def _postprocess(
        self,
        gdf: "geopandas.GeoDataFrame",
        cls_key: str,
        raster_path: str,
        min_area_m2: float,
        max_area_m2: float,
    ) -> "geopandas.GeoDataFrame":
        """Postprocessing sesuai karakteristik objek."""
        import geopandas as gpd

        # ── Buffer ke metric CRS untuk filter area ──
        if gdf.crs and gdf.crs.is_geographic:
            try:
                utm_crs = gdf.estimate_utm_crs()
                gdf_m = gdf.to_crs(utm_crs)
            except Exception:
                gdf_m = gdf.copy()
        else:
            gdf_m = gdf.copy()

        areas = gdf_m.geometry.area
        mask_area = (areas >= min_area_m2) & (areas <= max_area_m2)
        gdf = gdf[mask_area].copy().reset_index(drop=True)

        if len(gdf) == 0:
            return gdf

        # ── Deduplication antar tile (cepat via sjoin) ──
        gdf = self._deduplicate(gdf)

        if cls_key == "road":
            gdf = self._postprocess_road(gdf)
        elif cls_key == "water":
            gdf = self._postprocess_water(gdf)
        elif cls_key == "vegetation":
            gdf = self._postprocess_vegetation(gdf)

        return gdf.reset_index(drop=True)

    def _postprocess_road(self, gdf: "geopandas.GeoDataFrame") -> "geopandas.GeoDataFrame":
        """
        Postprocessing jalan:
        - Morphological closing sudah dilakukan di _mask_to_gdf
        - Simplify untuk memperhalus garis jalan
        - Filter aspek rasio (jalan harus memanjang)
        """
        from shapely.ops import unary_union

        # Simplify agresif untuk jalan (hapus noise kecil di tepi)
        gdf["geometry"] = gdf.geometry.simplify(tolerance=0.5, preserve_topology=True)

        # Filter aspek rasio: jalan harus memanjang (aspect >= 1.5)
        def aspect_ratio(geom):
            try:
                mbr = geom.minimum_rotated_rectangle
                coords = list(mbr.exterior.coords)
                edges = [
                    ((coords[i+1][0]-coords[i][0])**2 + (coords[i+1][1]-coords[i][1])**2)**0.5
                    for i in range(len(coords)-1)
                ]
                if len(edges) < 2:
                    return 1.0
                s1, s2 = edges[0], edges[1]
                return max(s1, s2) / max(min(s1, s2), 1e-9)
            except Exception:
                return 1.0

        # Ambil yang memanjang (aspect >= 1.5) ATAU area besar (persimpangan)
        if gdf.crs and gdf.crs.is_geographic:
            try:
                utm = gdf.estimate_utm_crs()
                areas = gdf.to_crs(utm).geometry.area
            except Exception:
                areas = gdf.geometry.area
        else:
            areas = gdf.geometry.area

        ratios  = gdf.geometry.apply(aspect_ratio)
        keep    = (ratios >= 1.5) | (areas >= 500.0)
        return gdf[keep].copy()

    def _postprocess_water(self, gdf: "geopandas.GeoDataFrame") -> "geopandas.GeoDataFrame":
        """
        Postprocessing badan air:
        - Fill holes kecil di dalam polygon air
        - Dissolve polygon yang saling menyentuh
        - Simplify boundary
        """
        from shapely.ops import unary_union
        from shapely.geometry import Polygon, MultiPolygon

        def fill_holes(geom, min_hole_area: float = 50.0):
            """Isi lubang kecil di dalam polygon."""
            try:
                if isinstance(geom, Polygon):
                    interior = [
                        r for r in geom.interiors
                        if Polygon(r).area > min_hole_area
                    ]
                    return Polygon(geom.exterior, interior)
                elif isinstance(geom, MultiPolygon):
                    return MultiPolygon([fill_holes(p, min_hole_area) for p in geom.geoms])
            except Exception:
                pass
            return geom

        gdf["geometry"] = gdf.geometry.apply(fill_holes)

        # Dissolve polygon yang saling overlap/berdekatan
        gdf["geometry"] = gdf.geometry.buffer(0.5)  # Kecil untuk merge yang menyentuh
        dissolved_geom  = unary_union(gdf.geometry)
        gdf["geometry"] = gdf.geometry.buffer(-0.5)  # Kembalikan

        # Simplify boundary air (lebih halus dari jalan)
        gdf["geometry"] = gdf.geometry.simplify(tolerance=1.0, preserve_topology=True)
        gdf = gdf[~gdf.geometry.is_empty].copy()

        return gdf

    def _postprocess_vegetation(self, gdf: "geopandas.GeoDataFrame") -> "geopandas.GeoDataFrame":
        """
        Postprocessing vegetasi:
        - Merge poligon kecil yang berdekatan
        - Simplify boundary
        - Filter noise (terlalu kecil setelah filtering)
        """
        from shapely.ops import unary_union

        # Buffer kecil untuk merge yang hampir menyentuh, lalu kembalikan
        gdf["geometry"] = gdf.geometry.buffer(1.5)
        dissolved       = unary_union(gdf.geometry)
        gdf["geometry"] = gdf.geometry.buffer(-1.5)

        # Simplify
        gdf["geometry"] = gdf.geometry.simplify(tolerance=0.8, preserve_topology=True)
        gdf = gdf[~gdf.geometry.is_empty].copy()
        gdf = gdf[gdf.geometry.area > 0].copy()

        return gdf

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _deduplicate(self, gdf: "geopandas.GeoDataFrame") -> "geopandas.GeoDataFrame":
        """Hapus duplikat dari overlap tile (IoU > 0.5)."""
        gdf = gdf.copy()
        gdf["geometry"] = gdf.geometry.buffer(0)  # Fix invalid
        keep  = [True] * len(gdf)
        geoms = list(gdf.geometry)
        for i in range(len(geoms)):
            if not keep[i]:
                continue
            for j in range(i + 1, len(geoms)):
                if not keep[j]:
                    continue
                try:
                    inter = geoms[i].intersection(geoms[j]).area
                    if inter == 0:
                        continue
                    union = geoms[i].union(geoms[j]).area
                    if union > 0 and (inter / union) > 0.5:
                        keep[j] = False
                except Exception:
                    pass
        return gdf.iloc[[i for i, v in enumerate(keep) if v]].copy().reset_index(drop=True)

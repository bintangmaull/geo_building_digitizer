"""
multi_object_runner.py
Orchestrator untuk digitasi multi-objek: Jalan, Badan Air, Vegetasi.

Mode Deteksi per Objek:
    - SegFormer (Rekomendasi): Semantic segmentation SegFormer (akurat, satu-pass per tile)
    - Pra-Deteksi Warna      : OpenCV color-based + SAM point prompts (fallback)
    - Otomatis SAM (Grid)    : SAM automatic generate() tanpa pre-deteksi

Prioritas Overlap (Bangunan datang dari pipeline terpisah di app.py):
    Bangunan > Jalan > Badan Air > Vegetasi

Output: Satu GeoDataFrame gabungan + satu Shapefile (.shp)

CATATAN: Pipeline bangunan (sam_processor.py + postprocess.py) TIDAK disentuh.
"""

import os
import gc
import shutil
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple


# ── Mode mapping ──────────────────────────────────────────────────────────────

def _map_mode(sidebar_value: str) -> str:
    """
    Konversi nilai dropdown sidebar → internal mode string.
    'SegFormer (Rekomendasi)' → 'segformer'
    'Pra-Deteksi Warna'       → 'predetect'
    'Otomatis SAM (Grid)'     → 'automatic'
    """
    v = sidebar_value.lower()
    if "segformer" in v:
        return "segformer"
    if "fingerprint" in v:
        return "fingerprint"
    if "otomatis" in v or "grid" in v:
        return "automatic"
    if "yolo" in v:
        return "yolo"
    return "predetect"


# ── Konfigurasi per objek ─────────────────────────────────────────────────────

def _get_objects_config(params: dict) -> dict:
    return {
        "road": {
            "label":       "Jalan",
            "emoji":       "🛣️",
            "masks_subdir": "masks_road",
            "min_area_m2": params.get("road_min_area_m2", 30.0),
            "max_area_m2": params.get("road_max_area_m2", 500000.0),
        },
        "water": {
            "label":       "Badan Air",
            "emoji":       "💧",
            "masks_subdir": "masks_water",
            "min_area_m2": params.get("water_min_area_m2", 50.0),
            "max_area_m2": params.get("water_max_area_m2", 10000000.0),
        },
        "vegetation": {
            "label":       "Vegetasi",
            "emoji":       "🌿",
            "masks_subdir": "masks_vegetation",
            "min_area_m2": params.get("veg_min_area_m2", 10.0),
            "max_area_m2": params.get("veg_max_area_m2", 50000000.0),
        },
    }


# ── Main Pipeline ──────────────────────────────────────────────────────────────

def run_multi_object_digitization(
    raster_path: str,
    output_dir: str,
    input_stem: str,
    sam_model_name: str,
    tile_size: int,
    enabled_objects: Dict[str, bool],
    params: dict,
    masks_base_dir: str,
    building_gdf: Optional["geopandas.GeoDataFrame"] = None,
    log_callback: Optional[Callable[[str], None]] = None,
    progress_callback: Optional[Callable[[int, str], None]] = None,
    cancel_check: Optional[Callable[[], bool]] = None,
) -> Optional["geopandas.GeoDataFrame"]:
    """
    Jalankan digitasi multi-objek (Jalan/Air/Vegetasi) lalu terapkan priority clipping.

    Args:
        raster_path    : Path ke file raster GeoTIFF
        output_dir     : Direktori output
        input_stem     : Nama dasar file input (tanpa ekstensi)
        sam_model_name : Nama model SAM (digunakan jika mode SAM)
        tile_size      : Ukuran tile dalam piksel
        enabled_objects: {"road": True, "water": False, "vegetation": True, ...}
        params         : Parameter lengkap dari sidebar
        masks_base_dir : Direktori untuk menyimpan mask temp
        building_gdf   : GeoDataFrame bangunan (untuk priority clipping)
                         Jika None, bangunan tidak diikutkan dalam clipping.
        log_callback   : Fungsi log
        progress_callback : Fungsi progress (pct: int, msg: str)
        cancel_check   : Fungsi yang mengembalikan True jika dibatalkan

    Returns:
        GeoDataFrame gabungan semua kelas (sudah di-clip prioritas), kolom 'class'.
        None jika semua kosong / dibatalkan.
    """
    import geopandas as gpd

    log          = log_callback or print
    progress     = progress_callback or (lambda pct, msg: None)
    is_cancelled = cancel_check or (lambda: False)

    os.makedirs(output_dir, exist_ok=True)

    objects_config  = _get_objects_config(params)
    object_modes    = params.get("object_modes", {})
    active_objects  = [k for k, v in enabled_objects.items() if v and k != "building"]

    if not active_objects:
        log("Tidak ada objek non-bangunan yang aktif.")
        return None

    log(f"\n{'='*55}")
    log(f"🗺️  DIGITASI MULTI-OBJEK")
    log(f"   Objek: {', '.join([objects_config[k]['label'] for k in active_objects])}")
    log(f"{'='*55}\n")

    # ── Deteksi mode: SegFormer vs per-object SAM ──────────────────────────────
    modes = {k: _map_mode(object_modes.get(k, "SegFormer (Rekomendasi)")) for k in active_objects}

    segformer_objects = [k for k in active_objects if modes[k] == "segformer"]
    other_objects     = [k for k in active_objects if modes[k] != "segformer"]

    gdfs_by_label: Dict[str, "geopandas.GeoDataFrame"] = {}

    # ── 1. SegFormer pipeline (satu-pass per tile, semua kelas sekaligus) ──────
    if segformer_objects:
        progress(35, "Memuat model SegFormer...")
        segformer_gdfs = _run_segformer_pipeline(
            raster_path=raster_path,
            active_keys=segformer_objects,
            objects_config=objects_config,
            params=params,
            tile_size=tile_size,
            log=log,
            progress=progress,
            is_cancelled=is_cancelled,
        )
        gdfs_by_label.update(segformer_gdfs)

    # ── 2. OpenCV/SAM pipeline (per-objek, mode lama) ─────────────────────────
    if other_objects and not is_cancelled():
        n_other   = len(other_objects)
        base_pct  = 40 if not segformer_objects else 70
        pct_each  = (50 if not segformer_objects else 20) // max(n_other, 1)

        for i, obj_key in enumerate(other_objects):
            if is_cancelled():
                break
            cfg   = objects_config[obj_key]
            label = cfg["label"]
            mode  = modes[obj_key]

            log(f"\n{cfg['emoji']}  {label} [{mode}]")
            progress(base_pct + i * pct_each, f"Memproses {label}...")

            masks_dir = os.path.join(masks_base_dir, cfg["masks_subdir"])
            if os.path.exists(masks_dir):
                shutil.rmtree(masks_dir, ignore_errors=True)
            os.makedirs(masks_dir, exist_ok=True)

            try:
                gdf = _process_single_object(
                    obj_key=obj_key,
                    label=label,
                    raster_path=raster_path,
                    masks_dir=masks_dir,
                    sam_model_name=sam_model_name,
                    tile_size=tile_size,
                    min_area_m2=cfg["min_area_m2"],
                    max_area_m2=cfg["max_area_m2"],
                    detection_mode=mode,
                    params=params,
                    log_callback=log,
                    progress_callback=lambda pct, msg, bp=base_pct+i*pct_each, pe=pct_each: progress(
                        bp + int(pct * pe / 100), msg
                    ),
                    cancel_check=is_cancelled,
                    building_gdf=building_gdf,
                )
                if gdf is not None and len(gdf) > 0:
                    gdf["class"] = label
                    gdfs_by_label[label] = gdf
                    log(f"  ✅ {label}: {len(gdf)} poligon")
                else:
                    log(f"  ⚠️ {label}: tidak ada poligon terdeteksi")
            except Exception as e:
                import traceback
                log(f"  ❌ Error {label}: {e}")
                log(traceback.format_exc())

    if is_cancelled() or not gdfs_by_label:
        if is_cancelled():
            log("🛑 Dibatalkan.")
        else:
            log("⚠️ Tidak ada objek yang berhasil didigitasi.")
        return None

    # ── 3. Priority Clipping ──────────────────────────────────────────────────
    progress(88, "Resolusi overlap prioritas...")
    log(f"\n🔗 Resolusi overlap (Bangunan > Jalan > Badan Air > Vegetasi)...")

    # Tambahkan bangunan jika ada (untuk clipping, bukan sebagai output baru)
    clip_input = dict(gdfs_by_label)
    if building_gdf is not None and len(building_gdf) > 0:
        building_copy = building_gdf.copy()
        building_copy["class"] = "Bangunan"
        clip_input["Bangunan"] = building_copy

    from core.objects.priority_clipper import apply_priority_clipping
    combined_gdf = apply_priority_clipping(
        gdfs_by_label=clip_input,
        log_callback=log,
    )

    if combined_gdf is None or len(combined_gdf) == 0:
        log("⚠️ Tidak ada poligon tersisa setelah clipping.")
        return None

    # Hapus bangunan dari output (bangunan sudah punya pipeline & output sendiri)
    if "class" in combined_gdf.columns:
        combined_gdf = combined_gdf[combined_gdf["class"] != "Bangunan"].copy()
        combined_gdf = combined_gdf.reset_index(drop=True)

    log(f"✅ Total poligon akhir: {len(combined_gdf)}")
    return combined_gdf


# ── SegFormer Pipeline ─────────────────────────────────────────────────────────

def _run_segformer_pipeline(
    raster_path: str,
    active_keys: List[str],
    objects_config: dict,
    params: dict,
    tile_size: int,
    log: Callable,
    progress: Callable,
    is_cancelled: Callable,
) -> Dict[str, "geopandas.GeoDataFrame"]:
    """
    Jalankan SegFormer untuk semua objek aktif dalam satu pass per tile.
    Lebih efisien dari menjalankan model 3x secara terpisah.
    """
    from core.objects.segformer_processor import SegFormerSegmentor, DEFAULT_SEGFORMER

    segformer_model_name = params.get("segformer_model", DEFAULT_SEGFORMER)

    segmentor = SegFormerSegmentor(
        model_name=segformer_model_name,
        log_callback=log,
        progress_callback=progress,
    )
    segmentor.reset_cancel()

    try:
        progress(38, "Memuat SegFormer...")
        segmentor.load_model()

        if is_cancelled():
            return {}

        enabled_classes = {k: True for k in active_keys}
        min_areas = {k: objects_config[k]["min_area_m2"] for k in active_keys}
        max_areas = {k: objects_config[k]["max_area_m2"] for k in active_keys}

        progress(42, "Segmentasi SegFormer tile-by-tile...")
        results_by_key = segmentor.process_raster(
            raster_path=raster_path,
            enabled_classes=enabled_classes,
            tile_size=tile_size,
            overlap=tile_size // 8,
            min_area_m2=min_areas,
            max_area_m2=max_areas,
            use_feature_refinement=params.get("use_feature_refinement", True),
            tile_progress_callback=None,
        )

        # Konversi key → label
        gdfs_by_label = {}
        for key, gdf in results_by_key.items():
            label = objects_config.get(key, {}).get("label", key)
            gdfs_by_label[label] = gdf

        return gdfs_by_label

    except Exception as e:
        import traceback
        log(f"❌ Error SegFormer pipeline: {e}")
        log(traceback.format_exc())
        return {}
    finally:
        try:
            segmentor.unload_model()
        except Exception:
            pass
        gc.collect()


# ── Per-Object SAM Pipeline (Fallback) ────────────────────────────────────────

def _process_single_object(
    obj_key: str,
    label: str,
    raster_path: str,
    masks_dir: str,
    sam_model_name: str,
    tile_size: int,
    min_area_m2: float,
    max_area_m2: float,
    detection_mode: str,
    params: dict,
    log_callback: Callable,
    progress_callback: Callable,
    cancel_check: Callable,
    building_gdf: Optional["geopandas.GeoDataFrame"] = None,
) -> Optional["geopandas.GeoDataFrame"]:
    """Jalankan satu processor objek (SAM-based: predetect, automatic, atau yolo)."""
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    models_dir   = os.path.join(project_root, "models")

    # Get specific YOLO model name for this object if YOLO mode is selected
    object_yolo_models = params.get("object_yolo_models", {})
    yolo_model_name = object_yolo_models.get(obj_key, "")
    if " (" in yolo_model_name:
        yolo_model_name = yolo_model_name.split(" (")[0].strip()

    if obj_key == "road":
        mode = params.get("road_config", {}).get("mode", "color_predetect")
        if mode == "color_predetect" or detection_mode != "fingerprint":
            raise NotImplementedError(
                "Mode 'Pra-Deteksi Warna (Lama)' untuk Jalan telah dihapus.\n"
                "Harap gunakan mode 'Road Fingerprint (Adaptif)'."
            )

        from core.objects.road_processor import RoadProcessor
        fp_path = params.get("road_config", {}).get("fingerprint_path", "")
        if not fp_path or not os.path.exists(fp_path):
            raise FileNotFoundError(
                f"File Road Fingerprint tidak ditemukan: {fp_path}\n"
                f"Silakan pilih mode '🔬 Buat Baru' dan klik BUAT FINGERPRINT terlebih dahulu."
            )

        processor = RoadProcessor(
            fingerprint_path=fp_path,
            log_callback=log_callback,
            progress_callback=progress_callback,
        )

        processor.reset_cancel()
        log_callback(f"  [1/3] Memuat fingerprint untuk {label}...")
        processor.load_fingerprint()
        
        if cancel_check():
            return None

        log_callback(f"  [2/3] Memproses tile-by-tile ({label})...")
        out_dir = params.get("output_dir", os.path.dirname(masks_dir))
        
        poly_gdf, line_gdf = processor.process_raster(
            raster_path=raster_path,
            output_dir=masks_dir,  # Hanya untuk temporary temp/masks_road
            tile_size=tile_size,
            overlap=tile_size // 8,
            min_area_m2=min_area_m2,
            building_gdf=building_gdf,
        )

        if cancel_check():
            return None

        # [3/3] Simpan centerline langsung (postprocess jalan sudah di-handle di process_raster)
        log_callback(f"  [3/3] Menyelesaikan output {label}...")
        if line_gdf is not None and len(line_gdf) > 0:
            stem = Path(raster_path).stem
            centerline_shp = os.path.join(out_dir, f"{stem}_jalan_centerline.shp")
            log_callback(f"  💾 Menyimpan Centerline Jalan ke: {centerline_shp}")
            try:
                line_gdf.to_file(centerline_shp)
            except Exception as e:
                log_callback(f"  ⚠️ Gagal menyimpan centerline: {e}")

        # Mengembalikan polygon GDF untuk priority clipping gabungan
        return poly_gdf

    elif obj_key == "water":
        from core.objects.water_processor import WaterProcessor
        processor = WaterProcessor(
            sam_model_name=sam_model_name, models_dir=models_dir, device="auto",
            detection_mode=detection_mode,
            log_callback=log_callback, progress_callback=progress_callback,
        )
    elif obj_key == "vegetation":
        from core.objects.vegetation_processor import VegetationProcessor
        processor = VegetationProcessor(
            sam_model_name=sam_model_name, models_dir=models_dir, device="auto",
            exg_threshold=params.get("veg_exg_threshold", 15.0),
            greenness_confirm=params.get("veg_greenness_confirm", 10.0),
            detection_mode=detection_mode,
            log_callback=log_callback, progress_callback=progress_callback,
        )
    else:
        log_callback(f"  ⚠️ Processor '{obj_key}' tidak dikenal.")
        return None

    processor.reset_cancel()

    try:
        if obj_key != "road":
            log_callback(f"  [1/3] Memuat model untuk {label}...")
            processor.load_model()

            if cancel_check():
                return None

            log_callback(f"  [2/3] Segmentasi tile-by-tile ({label})...")
            mask_results = processor.process_raster(
                raster_path=raster_path,
                masks_dir=masks_dir,
                tile_size=tile_size,
                overlap=tile_size // 8,
                min_area_m2=min_area_m2,
                max_area_m2=max_area_m2,
            )

            if cancel_check() or not mask_results:
                return None

            log_callback(f"  [3/3] Post-processing {label}...")
            gdf = processor.postprocess(
                mask_results=mask_results,
                original_raster_path=raster_path,
                min_area_m2=min_area_m2,
                max_area_m2=max_area_m2,
                log_callback=log_callback,
            )
            return gdf
        
    finally:
        try:
            processor.unload_model()
        except Exception:
            pass
        gc.collect()


# ── Legacy Export (untuk backward compat dengan app.py) ───────────────────────

def export_combined_shapefile(
    gdf: "geopandas.GeoDataFrame",
    output_path: str,
    source_raster_path: Optional[str] = None,
    log_callback: Optional[Callable[[str], None]] = None,
) -> str:
    """Ekspor GeoDataFrame ke Shapefile. Wrapper ke priority_clipper.export_final_shapefile."""
    from core.objects.priority_clipper import export_final_shapefile
    return export_final_shapefile(
        combined_gdf=gdf,
        output_path=output_path,
        source_raster_path=source_raster_path,
        log_callback=log_callback,
    )


def get_multi_object_summary(output_shp_path: str) -> dict:
    """Baca shapefile hasil dan kembalikan ringkasan per kelas."""
    try:
        import geopandas as gpd
        gdf = gpd.read_file(output_shp_path)
        summary = {
            "total": len(gdf),
            "by_class": {},
            "file_size_kb": os.path.getsize(output_shp_path) // 1024,
        }
        if "class" in gdf.columns:
            for cls in gdf["class"].unique():
                sub = gdf[gdf["class"] == cls]
                summary["by_class"][cls] = {
                    "count":         len(sub),
                    "total_area_m2": sub["Area_m2"].sum() if "Area_m2" in sub.columns else 0,
                }
        return summary
    except Exception:
        return {}

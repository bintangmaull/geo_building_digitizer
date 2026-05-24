"""
priority_clipper.py
Resolusi overlap antar poligon multi-kelas dengan sistem prioritas.

Urutan Prioritas (tertinggi ke terendah):
    1. Bangunan  — tidak pernah dipotong
    2. Jalan     — dipotong oleh Bangunan
    3. Badan Air — dipotong oleh Bangunan + Jalan
    4. Vegetasi  — dipotong oleh semua kelas di atasnya

Algoritma:
    - Iterasi dari prioritas tertinggi ke terendah
    - Setiap kelas mendapat potongan (difference) dari UNION semua kelas di atasnya
    - Poligon yang habis terpotong (area = 0 / empty) dihapus

Output: Satu GeoDataFrame gabungan dengan kolom 'class', tanpa overlap.
"""

import os
from typing import Dict, Optional, Callable

# Urutan prioritas (indeks kecil = prioritas lebih tinggi)
PRIORITY_ORDER = ["Bangunan", "Jalan", "Badan Air", "Vegetasi"]

# Mapping dari key internal ke label kelas
KEY_TO_LABEL = {
    "building":   "Bangunan",
    "road":       "Jalan",
    "water":      "Badan Air",
    "vegetation": "Vegetasi",
}
LABEL_TO_KEY = {v: k for k, v in KEY_TO_LABEL.items()}


def apply_priority_clipping(
    gdfs_by_label: Dict[str, "geopandas.GeoDataFrame"],
    log_callback: Optional[Callable[[str], None]] = None,
    min_remaining_area_m2: float = 5.0,
) -> Optional["geopandas.GeoDataFrame"]:
    """
    Terapkan clipping prioritas pada semua kelas dan gabungkan menjadi satu GeoDataFrame.

    Args:
        gdfs_by_label: Dict {label_kelas: GeoDataFrame}
                       Contoh: {"Bangunan": gdf1, "Jalan": gdf2, ...}
        log_callback : Fungsi log
        min_remaining_area_m2: Area minimum poligon setelah clipping (dalam m²)
                               Poligon yang lebih kecil dihapus.

    Returns:
        GeoDataFrame gabungan tanpa overlap, dengan kolom 'class'.
        None jika semua kelas kosong.
    """
    import geopandas as gpd
    from shapely.ops import unary_union
    from shapely.geometry import Polygon, MultiPolygon, GeometryCollection

    log = log_callback or print

    log("=== RESOLUSI OVERLAP PRIORITAS ===")
    log(f"   Urutan: {' > '.join(PRIORITY_ORDER)}")

    # Filter hanya kelas yang ada
    active_labels = [lbl for lbl in PRIORITY_ORDER if lbl in gdfs_by_label]
    if not active_labels:
        log("Tidak ada kelas untuk diproses.")
        return None

    log(f"   Kelas aktif: {', '.join(active_labels)}")

    # Tentukan CRS bersama dari kelas dengan prioritas tertinggi
    ref_crs = None
    for lbl in active_labels:
        gdf = gdfs_by_label[lbl]
        if len(gdf) > 0 and gdf.crs is not None:
            ref_crs = gdf.crs
            break

    if ref_crs is None:
        log("⚠️ CRS tidak ditemukan. Ekspor mungkin gagal.")

    # ── Hitung metric CRS untuk filter area ──────────────────────────────────
    utm_crs = None
    if ref_crs and ref_crs.is_geographic:
        try:
            import geopandas as _gpd
            dummy = _gpd.GeoDataFrame(geometry=[], crs=ref_crs)
            utm_crs = dummy.estimate_utm_crs()
        except Exception:
            pass

    # ── Akumulasikan union dari kelas yang sudah diproses ─────────────────────
    accumulated_union = None   # Union semua poligon kelas prioritas lebih tinggi
    result_gdfs = []

    for lbl in active_labels:
        gdf = gdfs_by_label[lbl].copy()

        if len(gdf) == 0:
            log(f"   [{lbl}] Dilewati (kosong)")
            continue

        # Pastikan CRS sama
        if ref_crs and gdf.crs and gdf.crs != ref_crs:
            try:
                gdf = gdf.to_crs(ref_crs)
            except Exception as e:
                log(f"   [{lbl}] ⚠️ Gagal reproyek CRS: {e}")

        # Fix invalid geometries
        gdf["geometry"] = gdf.geometry.buffer(0)

        n_before = len(gdf)

        if accumulated_union is not None and not accumulated_union.is_empty:
            log(f"   [{lbl}] Memotong overlap dengan kelas prioritas lebih tinggi...")

            def clip_geometry(geom):
                """Potong satu geometri dengan accumulated_union."""
                if geom is None or geom.is_empty:
                    return None
                try:
                    clipped = geom.difference(accumulated_union)
                    if clipped is None or clipped.is_empty:
                        return None
                    # Ekstrak hanya Polygon/MultiPolygon dari hasil
                    return _extract_polygons(clipped)
                except Exception:
                    return geom  # Fallback: kembalikan asli jika gagal

            gdf["geometry"] = gdf.geometry.apply(clip_geometry)

            # Hapus yang menjadi kosong atau None
            gdf = gdf[gdf.geometry.notna()].copy()
            gdf = gdf[~gdf.geometry.is_empty].copy()

            # Filter sisa poligon yang terlalu kecil setelah clipping
            if utm_crs:
                try:
                    gdf_m = gdf.to_crs(utm_crs)
                    area_mask = gdf_m.geometry.area >= min_remaining_area_m2
                    gdf = gdf[area_mask].copy()
                except Exception:
                    pass

            n_after = len(gdf)
            log(f"   [{lbl}] {n_before} → {n_after} poligon (setelah clipping)")
        else:
            log(f"   [{lbl}] {n_before} poligon (prioritas tertinggi, tidak dipotong)")

        if len(gdf) == 0:
            log(f"   [{lbl}] Tidak ada poligon tersisa setelah clipping.")
            continue

        # Pastikan kolom 'class' ada
        gdf["class"] = lbl

        result_gdfs.append(gdf)

        # Tambahkan kelas ini ke accumulated union untuk kelas berikutnya
        try:
            class_union = unary_union(gdf.geometry)
            if accumulated_union is None:
                accumulated_union = class_union
            else:
                accumulated_union = accumulated_union.union(class_union)
        except Exception as e:
            log(f"   [{lbl}] ⚠️ Gagal update accumulated union: {e}")

    # ── Gabungkan semua ───────────────────────────────────────────────────────
    if not result_gdfs:
        log("Tidak ada poligon tersisa setelah resolusi overlap.")
        return None

    import geopandas as gpd
    combined = gpd.pd.concat(result_gdfs, ignore_index=True)
    combined_gdf = gpd.GeoDataFrame(combined, geometry="geometry", crs=ref_crs)

    # Ringkasan per kelas
    log(f"\n=== HASIL AKHIR (tanpa overlap) ===")
    for lbl in active_labels:
        if "class" in combined_gdf.columns:
            n = len(combined_gdf[combined_gdf["class"] == lbl])
            log(f"   {lbl}: {n} poligon")
    log(f"   TOTAL: {len(combined_gdf)} poligon")
    log("=" * 40)

    return combined_gdf.reset_index(drop=True)


def export_final_shapefile(
    combined_gdf: "geopandas.GeoDataFrame",
    output_path: str,
    source_raster_path: Optional[str] = None,
    log_callback: Optional[Callable[[str], None]] = None,
) -> str:
    """
    Ekspor GeoDataFrame gabungan (sudah di-clip) ke satu file Shapefile.
    Menambahkan kolom: ID, class, Area_m2, Perim_m.

    Args:
        combined_gdf      : GeoDataFrame gabungan dari apply_priority_clipping()
        output_path       : Path ke .shp output
        source_raster_path: Untuk referensi CRS jika diperlukan
        log_callback      : Fungsi log

    Returns:
        str: Path file .shp yang berhasil ditulis
    """
    import geopandas as gpd
    from shapely.geometry import Polygon, MultiPolygon

    log = log_callback or print

    if combined_gdf is None or len(combined_gdf) == 0:
        raise ValueError("Tidak ada poligon untuk diekspor.")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    gdf = combined_gdf.copy().reset_index(drop=True)

    # Reproyek ke CRS raster sumber jika diperlukan
    if source_raster_path and os.path.isfile(source_raster_path):
        try:
            import rasterio
            with rasterio.open(source_raster_path) as src:
                target_crs = src.crs
            if target_crs and gdf.crs and gdf.crs != target_crs:
                log(f"Reproyek CRS: {gdf.crs} → {target_crs}")
                gdf = gdf.to_crs(target_crs)
        except Exception as e:
            log(f"⚠️ Gagal reproyek CRS: {e}")

    # ── Tambah kolom atribut ─────────────────────────────────────────────────
    gdf["ID"] = gdf.index + 1

    # Hitung area dan perimeter dalam satuan metrik
    if gdf.crs and gdf.crs.is_geographic:
        try:
            utm_crs  = gdf.estimate_utm_crs()
            gdf_m    = gdf.to_crs(utm_crs)
            gdf["Area_m2"] = gdf_m.geometry.area.round(2)
            gdf["Perim_m"] = gdf_m.geometry.length.round(2)
        except Exception:
            gdf["Area_m2"] = gdf.geometry.area.round(6)
            gdf["Perim_m"] = gdf.geometry.length.round(6)
    else:
        gdf["Area_m2"] = gdf.geometry.area.round(2)
        gdf["Perim_m"] = gdf.geometry.length.round(2)

    if "class" not in gdf.columns:
        gdf["class"] = "Unknown"

    # ── Ekspor ────────────────────────────────────────────────────────────────
    export_cols = ["ID", "class", "Area_m2", "Perim_m", "geometry"]
    available   = [c for c in export_cols if c in gdf.columns]
    export_gdf  = gdf[available].copy()

    # Bersihkan geometri: hanya Polygon/MultiPolygon
    export_gdf["geometry"] = export_gdf["geometry"].apply(_extract_polygons)
    export_gdf = export_gdf.dropna(subset=["geometry"])
    export_gdf = export_gdf[~export_gdf.geometry.is_empty]

    log(f"Mengekspor {len(export_gdf)} poligon ke: {output_path}")

    try:
        export_gdf.to_file(output_path, driver="ESRI Shapefile", encoding="UTF-8")
        size_kb = os.path.getsize(output_path) // 1024
        log(f"✅ Ekspor berhasil: {output_path} ({size_kb} KB)")
        log(f"   File pendamping: .dbf, .prj, .shx")
        return output_path
    except Exception as e:
        raise RuntimeError(f"Gagal mengekspor Shapefile: {e}")


def _extract_polygons(geom):
    """Ekstrak hanya Polygon/MultiPolygon dari suatu geometri."""
    from shapely.geometry import Polygon, MultiPolygon, GeometryCollection
    if geom is None or geom.is_empty:
        return None
    if not geom.is_valid:
        geom = geom.buffer(0)
    if isinstance(geom, (Polygon, MultiPolygon)):
        return geom
    elif isinstance(geom, GeometryCollection):
        polys = [g for g in geom.geoms if isinstance(g, (Polygon, MultiPolygon))]
        if not polys:
            return None
        if len(polys) == 1:
            return polys[0]
        return MultiPolygon(polys) if len(polys) > 1 else polys[0]
    return None

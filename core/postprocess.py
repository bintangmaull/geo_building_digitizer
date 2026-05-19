"""
postprocess.py
Post-processing pipeline for SAM mask outputs:
1. Merge masks from multiple tiles (with overlap deduplication via IoU NMS)
2. Filter by area (min/max building size)
3. Shadow detection and removal (dark pixel analysis)
4. Polygon regularization (orthogonalization / corner snapping to 90°)
5. Aspect ratio filter (remove very elongated objects = roads/rivers)
"""

import numpy as np
import os
from typing import Callable, List, Optional, Tuple
from pathlib import Path


# ──────────────────────────────────────────────
# STEP 1: Raster Mask → Vector Polygons
# ──────────────────────────────────────────────

def mask_to_polygons(
    mask_path: str,
    min_area_px: float = 2.0,
) -> "geopandas.GeoDataFrame":
    """
    Convert a binary raster mask to vector polygons using rasterio's vectorize.
    Returns a GeoDataFrame with polygon geometries and CRS set.
    """
    import rasterio
    from rasterio.features import shapes
    import geopandas as gpd
    from shapely.geometry import shape

    with rasterio.open(mask_path) as src:
        data = src.read(1)
        transform = src.transform
        crs = src.crs

    # Binary mask: non-zero = building
    binary = (data > 127).astype(np.uint8)

    geoms = []
    for geom_dict, val in shapes(binary, mask=binary, transform=transform):
        if val == 1:
            geom = shape(geom_dict)
            if geom.area >= min_area_px:  # area in projected units (CRS-dependent)
                geoms.append(geom)

    if not geoms:
        return gpd.GeoDataFrame(geometry=[], crs=crs)

    gdf = gpd.GeoDataFrame(geometry=geoms, crs=crs)
    return gdf


def merge_tile_masks(
    mask_paths_and_metas: List[Tuple[str, dict]],
    original_raster_path: str,
    min_area_px: float = 2.0,
    iou_threshold: float = 0.3,
    log_callback: Optional[Callable[[str], None]] = None,
) -> "geopandas.GeoDataFrame":
    """
    Merge polygons from all tiles into a single GeoDataFrame.
    Apply spatial NMS (dissolve overlapping polygons with IoU > threshold).
    """
    import geopandas as gpd
    from shapely.geometry import box
    import rasterio

    log = log_callback or (lambda x: None)

    all_gdfs = []
    log(f"DEBUG type: {type(mask_paths_and_metas)}, len: {len(mask_paths_and_metas)}")
    if len(mask_paths_and_metas) > 0:
        log(f"DEBUG elem 0 type: {type(mask_paths_and_metas[0])}, val: {repr(mask_paths_and_metas[0])[:100]}")
    for item in mask_paths_and_metas:
        if isinstance(item, tuple) and len(item) >= 2:
            mask_path, tile_meta = item[0], item[1]
        elif isinstance(item, str):
            mask_path = item
            tile_meta = {}
        else:
            log(f"DEBUG Invalid item: {repr(item)}")
            continue

        if not os.path.isfile(mask_path):
            continue
        try:
            gdf = mask_to_polygons(mask_path, min_area_px)
            if len(gdf) > 0:
                all_gdfs.append(gdf)
        except Exception as e:
            log(f"  Gagal vektorisasi {Path(mask_path).name}: {e}")

    if not all_gdfs:
        log("Tidak ada poligon yang berhasil diekstrak dari mask")
        # Return empty GDF with correct CRS
        with rasterio.open(original_raster_path) as src:
            return gpd.GeoDataFrame(geometry=[], crs=src.crs)

    log(f"Menggabungkan poligon dari {len(all_gdfs)} tile...")
    merged = gpd.pd.concat(all_gdfs, ignore_index=True)
    gdf = gpd.GeoDataFrame(merged, geometry="geometry", crs=all_gdfs[0].crs)

    log(f"Total poligon sebelum deduplication: {len(gdf)}")

    # Spatial NMS: hanya hapus duplikat dari area overlap antar tile
    # TIDAK melakukan unary_union karena bisa merge bangunan yang berdampingan
    gdf = gdf.reset_index(drop=True)

    try:
        from shapely.geometry import MultiPolygon, Polygon

        # Fix invalid geometries
        gdf["geometry"] = gdf.geometry.buffer(0)

        # IOU-based deduplication:
        # Hapus polygon B jika IoU(A,B) > threshold (artinya B adalah duplikat A dari tile overlap)
        keep = [True] * len(gdf)
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
                    iou = inter / union if union > 0 else 0
                    if iou > 0.5:  # >50% overlap = duplikat tile
                        keep[j] = False
                except Exception:
                    pass

        keep_indices = [k for k, v in enumerate(keep) if v]
        final_gdf = gdf.iloc[keep_indices].copy()
        final_gdf = final_gdf[final_gdf.geometry.area >= min_area_px]
        final_gdf = final_gdf.reset_index(drop=True)
        log(f"Poligon setelah deduplication: {len(final_gdf)}")
        return final_gdf
    except Exception as e:
        log(f"Deduplication gagal, menggunakan data mentah: {e}")
        return gdf


# ──────────────────────────────────────────────
# STEP 2: Area Filter
# ──────────────────────────────────────────────

def filter_by_area(
    gdf: "geopandas.GeoDataFrame",
    min_area_m2: float = 20.0,
    max_area_m2: float = 100000.0,
    log_callback: Optional[Callable[[str], None]] = None,
) -> "geopandas.GeoDataFrame":
    """
    Filter polygons by area in square meters.
    Handles both projected (meters) and geographic (degrees) CRS.
    """
    import geopandas as gpd

    log = log_callback or (lambda x: None)

    if len(gdf) == 0:
        return gdf

    # Project to a metric CRS for area calculation if geographic
    crs = gdf.crs
    if crs and crs.is_geographic:
        # Use UTM zone estimated from centroid
        centroid = gdf.unary_union.centroid
        utm_crs = gdf.estimate_utm_crs()
        gdf_metric = gdf.to_crs(utm_crs)
    else:
        gdf_metric = gdf.copy()

    areas = gdf_metric.geometry.area
    mask = (areas >= min_area_m2) & (areas <= max_area_m2)
    result = gdf[mask].copy()

    log(f"Filter area ({min_area_m2}-{max_area_m2} m²): {len(gdf)} -> {len(result)} poligon")
    return result.reset_index(drop=True)


# ──────────────────────────────────────────────
# STEP 3: Aspect Ratio Filter (Remove roads/rivers)
# ──────────────────────────────────────────────

def filter_by_aspect_ratio(
    gdf: "geopandas.GeoDataFrame",
    max_ratio: float = 8.0,
    log_callback: Optional[Callable[[str], None]] = None,
) -> "geopandas.GeoDataFrame":
    """
    Remove very elongated polygons (roads, rivers, fences).
    Uses minimum bounding rectangle aspect ratio.
    max_ratio: length/width threshold. Buildings typically < 5.
    """
    log = log_callback or (lambda x: None)

    if len(gdf) == 0:
        return gdf

    def get_aspect_ratio(geom):
        try:
            from shapely.geometry import MultiPolygon
            mbr = geom.minimum_rotated_rectangle
            if mbr is None or mbr.is_empty:
                return 1.0
            coords = list(mbr.exterior.coords)
            # Get edge lengths
            edges = []
            for i in range(len(coords) - 1):
                dx = coords[i+1][0] - coords[i][0]
                dy = coords[i+1][1] - coords[i][1]
                edges.append((dx**2 + dy**2) ** 0.5)
            if len(edges) < 2:
                return 1.0
            side1 = edges[0]
            side2 = edges[1]
            if min(side1, side2) == 0:
                return 999
            return max(side1, side2) / min(side1, side2)
        except Exception:
            return 1.0

    ratios = gdf.geometry.apply(get_aspect_ratio)
    mask = ratios <= max_ratio
    result = gdf[mask].copy()
    log(f"Filter rasio aspek (<={max_ratio}): {len(gdf)} -> {len(result)} poligon")
    return result.reset_index(drop=True)


# ──────────────────────────────────────────────
# STEP 3b: Compactness Filter (Remove blobs spanning multiple objects)
# ──────────────────────────────────────────────

def filter_by_compactness(
    gdf: "geopandas.GeoDataFrame",
    min_compactness: float = 0.05,
    log_callback: Optional[Callable[[str], None]] = None,
) -> "geopandas.GeoDataFrame":
    """
    Remove blobs that are highly irregular (spanning roads, blocks, or multiple objects).
    Uses Polsby-Popper compactness score = 4π × Area / Perimeter².
    Score 1.0 = perfect circle. Score near 0 = very irregular/elongated blob.
    Buildings typically score 0.05–0.8. Blobs spanning blocks score very near 0.
    """
    import math
    log = log_callback or (lambda x: None)

    if len(gdf) == 0:
        return gdf

    def compactness(geom):
        try:
            area = geom.area
            perim = geom.length
            if perim == 0:
                return 1.0
            return (4 * math.pi * area) / (perim ** 2)
        except Exception:
            return 1.0

    scores = gdf.geometry.apply(compactness)
    mask = scores >= min_compactness
    result = gdf[mask].copy()
    removed = len(gdf) - len(result)
    log(f"Filter compactness (>={min_compactness:.2f}): dihapus {removed} poligon blob tidak beraturan")
    return result.reset_index(drop=True)


# ──────────────────────────────────────────────
# STEP 4b: Vegetation / Greenness Filter
# ──────────────────────────────────────────────

def filter_vegetation(
    gdf: "geopandas.GeoDataFrame",
    raster_path: str,
    greenness_threshold: float = 15.0,
    log_callback: Optional[Callable[[str], None]] = None,
) -> "geopandas.GeoDataFrame":
    """
    Remove polygons whose pixels are predominantly green (vegetation: trees, grass, yards).
    
    Greenness index = mean(G channel) - mean(R channel)
    If greenness > threshold → classified as vegetation → removed.
    
    Buildings have red/orange/white/gray rooftops → greenness typically negative or near zero.
    Vegetation → greenness strongly positive (usually 20–80).
    """
    import rasterio
    from rasterio.mask import mask as rio_mask
    from shapely.geometry import mapping

    log = log_callback or (lambda x: None)

    if len(gdf) == 0:
        return gdf

    try:
        with rasterio.open(raster_path) as src:
            raster_crs = src.crs
            if gdf.crs and gdf.crs != raster_crs:
                gdf_raster = gdf.to_crs(raster_crs)
            else:
                gdf_raster = gdf

            keep_mask = []
            for geom in gdf_raster.geometry:
                try:
                    out_image, _ = rio_mask(src, [mapping(geom)], crop=True, nodata=0)
                    if out_image.shape[0] < 3:
                        keep_mask.append(True)
                        continue
                    r = out_image[0].astype(float)
                    g = out_image[1].astype(float)
                    b = out_image[2].astype(float)
                    # Only non-zero pixels
                    valid = r > 0
                    if valid.sum() == 0:
                        keep_mask.append(True)
                        continue
                    mean_r = r[valid].mean()
                    mean_g = g[valid].mean()
                    mean_b = b[valid].mean()
                    # Greenness: how much greener than the other channels
                    greenness = mean_g - max(mean_r, mean_b)
                    is_vegetation = greenness > greenness_threshold
                    keep_mask.append(not is_vegetation)
                except Exception:
                    keep_mask.append(True)

        result = gdf[keep_mask].copy()
        removed = len(gdf) - len(result)
        log(f"Filter vegetasi (greenness>{greenness_threshold}): dihapus {removed} poligon pohon/halaman")
        return result.reset_index(drop=True)
    except Exception as e:
        log(f"Filter vegetasi gagal: {e}. Melanjutkan tanpa filter vegetasi.")
        return gdf


# ──────────────────────────────────────────────
# STEP 5: Shadow Filter
# ──────────────────────────────────────────────

def filter_shadows(
    gdf: "geopandas.GeoDataFrame",
    raster_path: str,
    brightness_threshold: int = 55,
    log_callback: Optional[Callable[[str], None]] = None,
) -> "geopandas.GeoDataFrame":
    """
    Remove polygons that are likely building shadows (very dark areas).
    Calculates mean pixel brightness inside each polygon.
    Polygons with mean brightness < threshold are classified as shadows and removed.

    Args:
        brightness_threshold: 0-255. Pixels darker than this = shadow. Default 55.
    """
    import rasterio
    from rasterio.features import geometry_mask
    from rasterio.mask import mask as rio_mask
    from shapely.geometry import mapping

    log = log_callback or (lambda x: None)

    if len(gdf) == 0:
        return gdf

    try:
        with rasterio.open(raster_path) as src:
            raster_crs = src.crs
            # Reproject polygons to raster CRS if needed
            if gdf.crs and gdf.crs != raster_crs:
                gdf_raster = gdf.to_crs(raster_crs)
            else:
                gdf_raster = gdf

            keep_mask = []
            for idx, (geom, geom_src) in enumerate(zip(gdf_raster.geometry, gdf.geometry)):
                try:
                    out_image, _ = rio_mask(src, [mapping(geom)], crop=True, nodata=0)
                    # Calculate mean brightness across RGB bands
                    rgb = out_image[:3]  # First 3 bands
                    pixels = rgb[:, rgb[0] > 0]  # Ignore nodata
                    if pixels.size == 0:
                        keep_mask.append(True)
                        continue
                    brightness = np.mean(pixels)
                    is_shadow = brightness < brightness_threshold
                    keep_mask.append(not is_shadow)
                except Exception:
                    keep_mask.append(True)  # Keep if analysis fails

            result = gdf[keep_mask].copy()
            removed = len(gdf) - len(result)
            log(f"Filter bayangan (threshold={brightness_threshold}): dihapus {removed} poligon")
            return result.reset_index(drop=True)
    except Exception as e:
        log(f"Filter bayangan gagal: {e}. Melanjutkan tanpa filter bayangan.")
        return gdf


# ──────────────────────────────────────────────
# STEP 5: Polygon Regularization (Corner Snapping)
# ──────────────────────────────────────────────

def orthogonalize_polygon(geom, angle_tolerance_deg: float = 15.0, simplify_tol: float = 0.5):
    """
    Regularize a single polygon to have more orthogonal (90°) corners.
    
    Process:
    1. Simplify with Douglas-Peucker to remove micro-vertices
    2. Snap near-90° and near-45° corners to exact angles
    3. Fallback: minimum rotated bounding rectangle for very irregular shapes

    Args:
        geom: Shapely geometry
        angle_tolerance_deg: Corners within this range of 90°/45°/0° are snapped
        simplify_tol: Douglas-Peucker tolerance (in CRS units)
    """
    from shapely.geometry import Polygon, MultiPolygon
    from shapely.validation import make_valid
    import math

    if geom is None or geom.is_empty:
        return geom

    # Handle multi-polygons: process each part
    if isinstance(geom, MultiPolygon):
        parts = [orthogonalize_polygon(p, angle_tolerance_deg, simplify_tol) for p in geom.geoms]
        parts = [p for p in parts if p and not p.is_empty]
        if not parts:
            return geom
        if len(parts) == 1:
            return parts[0]
        return MultiPolygon(parts)

    try:
        # Step 1: Simplify (Douglas-Peucker)
        simplified = geom.simplify(simplify_tol, preserve_topology=True)
        if simplified.is_empty:
            simplified = geom

        coords = list(simplified.exterior.coords[:-1])  # Remove closing vertex
        n = len(coords)

        if n < 3:
            return geom

        # Step 2: Snap corners to 90°/45° angles
        def angle_between(p1, p2, p3):
            """Angle at p2 between vectors p1→p2 and p2→p3 (in degrees)."""
            v1 = (p1[0] - p2[0], p1[1] - p2[1])
            v2 = (p3[0] - p2[0], p3[1] - p2[1])
            dot = v1[0]*v2[0] + v1[1]*v2[1]
            mag1 = math.sqrt(v1[0]**2 + v1[1]**2)
            mag2 = math.sqrt(v2[0]**2 + v2[1]**2)
            if mag1 == 0 or mag2 == 0:
                return 90.0
            cos_val = max(-1.0, min(1.0, dot / (mag1 * mag2)))
            return math.degrees(math.acos(cos_val))

        # For each vertex, check if angle is close to 90°
        new_coords = list(coords)
        tol = angle_tolerance_deg

        for i in range(n):
            p_prev = coords[(i - 1) % n]
            p_curr = coords[i]
            p_next = coords[(i + 1) % n]
            angle = angle_between(p_prev, p_curr, p_next)

            # Check snap targets: 90° or 180° (straight)
            for target in [90.0, 180.0]:
                if abs(angle - target) <= tol:
                    # Snap: recalculate p_curr to make exact angle
                    # Simple approach: project p_curr onto perpendicular from p_prev to segment
                    # Direction of incoming edge
                    dx_in = p_curr[0] - p_prev[0]
                    dy_in = p_curr[1] - p_prev[1]
                    len_in = math.sqrt(dx_in**2 + dy_in**2)
                    if len_in == 0:
                        break
                    # Unit vector of incoming edge
                    ux, uy = dx_in / len_in, dy_in / len_in
                    # Perpendicular (90° snap): outgoing edge should be perpendicular
                    if target == 90.0:
                        # Project p_next onto perpendicular direction from p_curr
                        perp_x, perp_y = -uy, ux
                        dx_out = p_next[0] - p_curr[0]
                        dy_out = p_next[1] - p_curr[1]
                        proj = dx_out * perp_x + dy_out * perp_y
                        # Snapped next point
                        snapped_next = (p_curr[0] + proj * perp_x, p_curr[1] + proj * perp_y)
                        new_coords[(i + 1) % n] = snapped_next
                    break

        # Reconstruct polygon
        new_coords.append(new_coords[0])  # Close ring
        try:
            new_geom = Polygon(new_coords)
            new_geom = make_valid(new_geom)
            if new_geom.is_valid and not new_geom.is_empty and new_geom.area > 0:
                return new_geom
        except Exception:
            pass

        # Step 3 Fallback: minimum rotated rectangle
        try:
            mbr = geom.minimum_rotated_rectangle
            if mbr and not mbr.is_empty and mbr.area > 0:
                return mbr
        except Exception:
            pass

        return geom

    except Exception:
        return geom


def regularize_polygons(
    gdf: "geopandas.GeoDataFrame",
    enable: bool = True,
    angle_tolerance: float = 15.0,
    simplify_tolerance: float = 0.5,
    log_callback: Optional[Callable[[str], None]] = None,
) -> "geopandas.GeoDataFrame":
    """Apply orthogonalization to all polygons in a GeoDataFrame."""
    log = log_callback or (lambda x: None)

    if not enable or len(gdf) == 0:
        return gdf

    log(f"Regularisasi sudut poligon (toleransi={angle_tolerance}°)...")
    gdf = gdf.copy()
    gdf["geometry"] = gdf["geometry"].apply(
        lambda g: orthogonalize_polygon(g, angle_tolerance, simplify_tolerance)
    )
    # Remove invalid geometries after regularization
    gdf = gdf[gdf.geometry.is_valid & ~gdf.geometry.is_empty].reset_index(drop=True)
    log(f"Regularisasi selesai. Poligon valid: {len(gdf)}")
    return gdf


# ──────────────────────────────────────────────
# STEP 3c: Agricultural Field / Bare Soil Filter
# Removes large, uniform-color areas that are not buildings
# ──────────────────────────────────────────────

def filter_non_building_colors(
    gdf: "geopandas.GeoDataFrame",
    raster_path: str,
    max_field_area_m2: float = 500.0,
    log_callback=None,
) -> "geopandas.GeoDataFrame":
    """
    Remove polygons that look like agricultural fields or bare soil, not buildings.

    Strategy:
      - Small polygons (<= max_field_area_m2) always pass (preserve small houses).
      - Large polygons (> max_field_area_m2) are analyzed at pixel level:
          * If the interior pixels show very LOW color variance (std < 20) across
            all RGB channels AND low brightness variance (std < 22), it is classified
            as a flat, uniform field or bare soil patch and removed.
          * Building rooftops have visible texture, shadows, panels, chimneys, etc.,
            which always produce higher color variance even on uniform-colored roofs.
    """
    import rasterio
    from rasterio.mask import mask as rio_mask
    from shapely.geometry import mapping

    log = log_callback or (lambda x: None)

    if len(gdf) == 0:
        return gdf

    try:
        with rasterio.open(raster_path) as src:
            raster_crs = src.crs
            if gdf.crs and gdf.crs != raster_crs:
                gdf_raster = gdf.to_crs(raster_crs)
            else:
                gdf_raster = gdf

            # Compute areas in metric CRS for the area threshold
            if gdf.crs and gdf.crs.is_geographic:
                utm_crs = gdf.estimate_utm_crs()
                gdf_metric = gdf.to_crs(utm_crs)
            else:
                gdf_metric = gdf.copy()

            areas_m2 = gdf_metric.geometry.area
            keep_mask = []

            for i, (geom, area_m2) in enumerate(zip(gdf_raster.geometry, areas_m2)):
                # Small polygons → always keep (could be small buildings)
                if area_m2 <= max_field_area_m2:
                    keep_mask.append(True)
                    continue

                # Large polygons → apply pixel-level color variance analysis
                try:
                    out_image, _ = rio_mask(src, [mapping(geom)], crop=True, nodata=0)
                    if out_image.shape[0] < 3:
                        keep_mask.append(True)
                        continue

                    r = out_image[0].astype(float)
                    g = out_image[1].astype(float)
                    b = out_image[2].astype(float)

                    valid = r > 0
                    if valid.sum() < 10:
                        keep_mask.append(True)
                        continue

                    r_v, g_v, b_v = r[valid], g[valid], b[valid]

                    # Mean std across channels: low = uniform color = likely field/soil
                    mean_channel_std = (r_v.std() + g_v.std() + b_v.std()) / 3.0

                    # Brightness variance: low = homogeneous texture = likely field
                    brightness = (r_v + g_v + b_v) / 3.0
                    brightness_std = brightness.std()

                    # Threshold: fields/soil typically std < 20 across all channels
                    # Building rooftops have texture, edges → std typically > 25
                    is_flat_field = (mean_channel_std < 20.0) and (brightness_std < 22.0)
                    keep_mask.append(not is_flat_field)

                except Exception:
                    keep_mask.append(True)

        result = gdf[keep_mask].copy()
        removed = len(gdf) - len(result)
        log(f"Filter lahan seragam (area>{max_field_area_m2:.0f}m2, low-variance): dihapus {removed} poligon lahan/sawah")
        return result.reset_index(drop=True)

    except Exception as e:
        log(f"Filter lahan pertanian gagal: {e}. Melanjutkan tanpa filter.")
        return gdf


def resolve_overlaps(
    gdf: "geopandas.GeoDataFrame",
    log_callback: Optional[Callable[[str], None]] = None,
) -> "geopandas.GeoDataFrame":
    """
    Ensure no building polygons overlap with each other.
    Resolves overlaps by sorting from largest to smallest area, subtracting the
    accumulated processed union from each subsequent polygon.
    """
    import geopandas as gpd
    from shapely.geometry import Polygon, MultiPolygon
    from shapely.ops import unary_union

    log = log_callback or (lambda x: None)
    if len(gdf) == 0:
        return gdf

    log("Mengeliminasi poligon tumpang tindih (avoid overlaps)...")
    gdf = gdf.copy()
    gdf["_orig_area"] = gdf.geometry.area
    
    # Sort descending by area to preserve larger/higher-priority buildings
    gdf = gdf.sort_values(by="_orig_area", ascending=False).reset_index(drop=True)

    cleaned_geoms = []
    kept_indices = []
    processed_union = None

    for i, row in gdf.iterrows():
        geom = row.geometry
        if geom is None or geom.is_empty:
            continue

        if not geom.is_valid:
            geom = geom.buffer(0)

        if processed_union is None:
            cleaned_geoms.append(geom)
            kept_indices.append(i)
            processed_union = geom
        else:
            try:
                diff = geom.difference(processed_union)
                if diff.is_empty:
                    continue

                if isinstance(diff, (Polygon, MultiPolygon)):
                    clean_geom = diff
                elif hasattr(diff, "geoms"):
                    polys = [g for g in diff.geoms if isinstance(g, (Polygon, MultiPolygon))]
                    if not polys:
                        continue
                    elif len(polys) == 1:
                        clean_geom = polys[0]
                    else:
                        clean_geom = MultiPolygon(polys)
                else:
                    continue

                # Filter out extremely small slivers resulting from differences (e.g. < 2.0 m2)
                if clean_geom.area < 2.0:
                    continue

                cleaned_geoms.append(clean_geom)
                kept_indices.append(i)
                processed_union = unary_union([processed_union, clean_geom]).buffer(0)
            except Exception as e:
                # Fallback on topology errors
                cleaned_geoms.append(geom)
                kept_indices.append(i)
                processed_union = unary_union([processed_union, geom]).buffer(0)

    if not cleaned_geoms:
        return gpd.GeoDataFrame(geometry=[], crs=gdf.crs)

    final_gdf = gdf.iloc[kept_indices].copy()
    final_gdf.geometry = cleaned_geoms

    if "_orig_area" in final_gdf.columns:
        final_gdf = final_gdf.drop(columns=["_orig_area"])

    removed = len(gdf) - len(final_gdf)
    log(f"Eliminasi overlap selesai: Dibuat non-overlapping, membuang {removed} poligon tindih/sliver.")
    return final_gdf.reset_index(drop=True)


# ──────────────────────────────────────────────
# MAIN POST-PROCESS PIPELINE
# ──────────────────────────────────────────────

def run_postprocess_pipeline(
    mask_paths_and_metas: List[Tuple[str, dict]],
    original_raster_path: str,
    min_area_m2: float = 20.0,
    max_area_m2: float = 5000.0,
    max_aspect_ratio: float = 8.0,
    enable_shadow_filter: bool = True,
    shadow_threshold: int = 30,
    enable_vegetation_filter: bool = True,
    greenness_threshold: float = 15.0,
    enable_regularization: bool = True,
    angle_tolerance: float = 15.0,
    log_callback: Optional[Callable[[str], None]] = None,
    progress_callback: Optional[Callable[[int, str], None]] = None,
) -> "geopandas.GeoDataFrame":
    """
    Full post-processing pipeline from mask tiles to final building polygons.
    """
    log = log_callback or (lambda x: None)
    progress = progress_callback or (lambda pct, msg: None)

    log("=== MEMULAI POST-PROCESSING ===")

    # 1. Merge tile masks → polygons
    progress(70, "Menggabungkan masker dari semua tile...")
    gdf = merge_tile_masks(mask_paths_and_metas, original_raster_path, log_callback=log)
    log(f"Total poligon awal: {len(gdf)}")

    if len(gdf) == 0:
        log("PERINGATAN: Tidak ada poligon yang terdeteksi!")
        return gdf

    # 2. Area filter
    progress(74, "Filter berdasarkan luas bangunan...")
    gdf = filter_by_area(gdf, min_area_m2, max_area_m2, log_callback=log)

    # 3. Aspect ratio filter
    progress(77, "Filter rasio aspek (hapus jalan/sungai)...")
    gdf = filter_by_aspect_ratio(gdf, max_aspect_ratio, log_callback=log)

    # 3b. Compactness filter — hapus blob besar tidak beraturan
    progress(80, "Filter compactness (hapus blob tidak beraturan)...")
    gdf = filter_by_compactness(gdf, min_compactness=0.05, log_callback=log)

    # 3c. Agricultural field filter — hapus lahan pertanian seragam besar
    # Polygons > 500 m2 dengan warna piksel sangat seragam (low variance) = lahan sawah/kebun
    progress(83, "Filter lahan pertanian (hapus sawah/ladang besar)...")
    gdf = filter_non_building_colors(
        gdf, original_raster_path, max_field_area_m2=500.0, log_callback=log
    )

    # 4. Shadow filter
    if enable_shadow_filter:
        progress(86, "Mendeteksi dan menghapus bayangan...")
        gdf = filter_shadows(gdf, original_raster_path, shadow_threshold, log_callback=log)

    # 4b. Vegetation filter
    if enable_vegetation_filter:
        progress(88, "Mendeteksi dan menghapus vegetasi (pohon/halaman)...")
        gdf = filter_vegetation(gdf, original_raster_path, greenness_threshold, log_callback=log)

    # 5. Regularization
    if enable_regularization:
        progress(91, "Regularisasi sudut poligon bangunan...")
        gdf = regularize_polygons(gdf, enable=True, angle_tolerance=angle_tolerance, log_callback=log)

    # 6. Resolve overlaps (avoid overlap)
    progress(93, "Menghilangkan poligon tumpang tindih (avoid overlap)...")
    gdf = resolve_overlaps(gdf, log_callback=log)

    log(f"=== POST-PROCESSING SELESAI: {len(gdf)} bangunan terdeteksi ===")
    progress(95, f"Selesai: {len(gdf)} bangunan terdeteksi")

    return gdf

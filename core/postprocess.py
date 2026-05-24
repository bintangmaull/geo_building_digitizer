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
) -> "geopandas.GeoDataFrame":
    """
    Convert a binary or instance raster mask to vector polygons using rasterio's vectorize.
    Supports both binary masks and instance-aware (unique IDs) masks.
    Returns a GeoDataFrame with polygon geometries and CRS set.
    """
    import rasterio
    from rasterio.features import shapes
    import geopandas as gpd
    from shapely.geometry import shape
    import numpy as np

    with rasterio.open(mask_path) as src:
        data = src.read(1)
        transform = src.transform
        crs = src.crs

    unique_vals = np.unique(data)
    geoms = []
    ids = []

    for val in unique_vals:
        if val == 0:
            continue
        
        # Create a binary mask for this specific instance/value
        instance_mask = (data == val).astype(np.uint8)
        
        for geom_dict, v in shapes(instance_mask, mask=instance_mask, transform=transform):
            if v == 1:
                geom = shape(geom_dict)
                geoms.append(geom)
                ids.append(val)

    if not geoms:
        return gpd.GeoDataFrame(geometry=[], crs=crs)
        
    gdf = gpd.GeoDataFrame({"geometry": geoms, "building_id": ids}, crs=crs)
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
            gdf = mask_to_polygons(mask_path)
            if not gdf.empty:
                all_gdfs.append(gdf)
        except Exception as e:
            log(f"Gagal memproses {mask_path}: {e}")

    if not all_gdfs:
        log("Tidak ada poligon yang berhasil diekstrak dari mask")
        # Return empty GDF with correct CRS
        with rasterio.open(original_raster_path) as src:
            return gpd.GeoDataFrame(geometry=[], crs=src.crs)

    log(f"Menggabungkan poligon dari {len(all_gdfs)} tile...")
    merged = gpd.pd.concat(all_gdfs, ignore_index=True)
    gdf = gpd.GeoDataFrame(merged, geometry="geometry", crs=all_gdfs[0].crs)

    log(f"Total poligon sebelum deduplication: {len(gdf)}")

    # Group polygons by building_id (Global YOLO Box ID)
    # This perfectly merges halves of buildings cut by tile boundaries, WITHOUT merging distinct adjacent buildings!
    final_geoms = []
    
    if "building_id" in gdf.columns:
        for b_id, group in gdf.groupby("building_id"):
            union_geom = group.geometry.unary_union
            
            # Explode if necessary
            from shapely.geometry import MultiPolygon, Polygon, GeometryCollection
            if isinstance(union_geom, Polygon):
                final_geoms.append(union_geom)
            elif isinstance(union_geom, MultiPolygon):
                final_geoms.extend(list(union_geom.geoms))
            elif isinstance(union_geom, GeometryCollection):
                for geom in union_geom.geoms:
                    if isinstance(geom, Polygon):
                        final_geoms.append(geom)
                    elif isinstance(geom, MultiPolygon):
                        final_geoms.extend(list(geom.geoms))
    else:
        final_geoms = list(gdf.geometry)

    final_gdf = gpd.GeoDataFrame(geometry=final_geoms, crs=gdf.crs)
    final_gdf = final_gdf.reset_index(drop=True)
    log(f"Poligon setelah deduplication: {len(final_gdf)}")
    return final_gdf


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

def _rta_orthogonalize(geom, angle_tolerance_deg, actual_simplify_tol, mbr):
    """
    Internal helper: Rotate-to-Align Orthogonalization.
    Rotates the polygon to its dominant axis, simplifies, snaps orthogonal corners, rotates back.
    """
    from shapely.geometry import Polygon
    from shapely.validation import make_valid
    import math

    # A. Calculate the True Architectural Angle
    # MBR is highly unstable for L-shapes, T-shapes, and buildings with tree protrusions.
    # We simplify the mask to remove pixel staircases, then find the dominant angle using an edge-length histogram.
    angle = 0.0
    try:
        angle_simplify_tol = 0.2 if actual_simplify_tol > 0.1 else (0.2 / 111320)
        pre_simplified = geom.simplify(angle_simplify_tol, preserve_topology=True)
        coords = list(pre_simplified.exterior.coords)
        buckets = {}
        for i in range(len(coords) - 1):
            dx = coords[i+1][0] - coords[i][0]
            dy = coords[i+1][1] - coords[i][1]
            length = math.hypot(dx, dy)
            if length < angle_simplify_tol: continue # Ignore tiny artifact edges
            
            deg = math.degrees(math.atan2(dy, dx)) % 90.0
            
            # Penalize perfect H/V edges because they are highly likely to be artificial YOLO crop lines.
            # True H/V buildings will still easily win since 100% of their edges are H/V.
            is_perfect_hv = abs(dx) < 1e-8 or abs(dy) < 1e-8
            weight_multiplier = 0.5 if is_perfect_hv else 1.0
            
            # Distribute length across adjacent bins (smoothing) to prevent noisy straight edges
            # from splitting across multiple bins (e.g., 88, 89, 0, 1, 2) and losing to a single diagonal edge.
            for offset in range(-3, 4):
                bin_idx = int(round(deg + offset)) % 90
                # Give highest weight to the exact center, lower to the edges
                weight = length * (4 - abs(offset)) * weight_multiplier
                buckets[bin_idx] = buckets.get(bin_idx, 0) + weight
            
        if buckets:
            best_bin = 0
            max_val = -1
            for b_idx, val in buckets.items():
                if val > max_val:
                    max_val = val
                    best_bin = b_idx
                    
            hist_angle_deg = best_bin
            angle = math.radians(hist_angle_deg)
        else:
            raise ValueError("No valid edges found")
            
    except Exception as e:
        print(f"DEBUG RTA: Exception in angle calc: {e}")
        # Fallback to MBR if anything fails
        mbr_coords = list(mbr.exterior.coords)
        max_len = -1.0
        for i in range(4):
            dx = mbr_coords[i+1][0] - mbr_coords[i][0]
            dy = mbr_coords[i+1][1] - mbr_coords[i][1]
            length = math.hypot(dx, dy)
            if length > max_len:
                max_len = length
                angle = math.atan2(dy, dx)
                
    # Normalize angle to -45..45 degrees
    angle = angle % (math.pi / 2)
    if angle > math.pi / 4:
        angle -= math.pi / 2

    centroid = geom.centroid
    cx, cy = centroid.x, centroid.y

    def rotate_point(x, y, cx, cy, angle_rad):
        cos_a = math.cos(angle_rad)
        sin_a = math.sin(angle_rad)
        nx = cos_a * (x - cx) - sin_a * (y - cy) + cx
        ny = sin_a * (x - cx) + cos_a * (y - cy) + cy
        return nx, ny

    # B. Rotate to axis-aligned
    rotated_coords = [rotate_point(x, y, cx, cy, -angle) for x, y in geom.exterior.coords]
    rotated_geom = Polygon(rotated_coords)
    rotated_geom = make_valid(rotated_geom)

    # C. Simplify and regularize
    # Use a base tolerance to remove jagged noise without destroying architecture
    # In geographic degrees, actual_simplify_tol is very small (e.g., 2e-5)
    # The threshold for "strict_straight" (tree detected) was set to 2.5 meters.
    # We check if actual_simplify_tol corresponds to > 1.0 meters.
    is_strict = False
    if actual_simplify_tol > 1.0:
        is_strict = True
    elif actual_simplify_tol < 0.1 and actual_simplify_tol > (1.0 / 111320): 
        # Geographic coordinates check
        is_strict = True

    base_tol = actual_simplify_tol * 0.2 if is_strict else actual_simplify_tol
    simplified = rotated_geom.simplify(base_tol, preserve_topology=True)

    # NEW: Axis-Aligned Mitre Closing for tree bites
    # This mathematical trick swallows diagonal tree bites and restores perfect 90-degree corners,
    # while preserving large H/V architectural features (L-shapes).
    if is_strict and not simplified.is_empty:
        # Mitre radius: 3.5 meters (or equivalent in degrees)
        r_mitre = 3.5 if actual_simplify_tol > 0.1 else (3.5 / 111320)
        try:
            # join_style=2 is MITRE, cap_style=3 is SQUARE
            simplified = simplified.buffer(r_mitre, join_style=2).buffer(-r_mitre, join_style=2)
            simplified = make_valid(simplified)
            # Clean up redundant vertices after closing
            simplified = simplified.simplify(base_tol, preserve_topology=True)
        except Exception:
            pass

    if simplified.is_empty:
        simplified = rotated_geom

    coords = list(simplified.exterior.coords[:-1])
    n = len(coords)
    if n < 4:
        return None  # Signal caller to fall back to MBR

    # Trigonometric threshold for H/V classification
    # Use a wider tolerance to catch near-H/V edges 
    snap_tol_deg = min(angle_tolerance_deg, 35.0)
    rad_tol = math.radians(snap_tol_deg)
    tan_tol = math.tan(rad_tol)

    # D. Iterative edge forcing + corner snapping
    # Pass 1: Force every edge to be exactly H or V if it is "close enough"
    # This snaps diagonal edges caused by tree bites to the nearest axis.
    def force_orthogonal(coords_in):
        n_in = len(coords_in)
        out = list(coords_in)
        for i in range(n_in):
            p_prev = out[(i - 1) % n_in]
            p_curr = out[i]
            p_next = out[(i + 1) % n_in]

            dx_in  = p_curr[0] - p_prev[0]
            dy_in  = p_curr[1] - p_prev[1]
            dx_out = p_next[0] - p_curr[0]
            dy_out = p_next[1] - p_curr[1]

            in_horiz = abs(dy_in)  <= abs(dx_in)  * tan_tol if abs(dx_in)  > 1e-12 else False
            in_vert  = abs(dx_in)  <= abs(dy_in)  * tan_tol if abs(dy_in)  > 1e-12 else False
            out_horiz = abs(dy_out) <= abs(dx_out) * tan_tol if abs(dx_out) > 1e-12 else False
            out_vert  = abs(dx_out) <= abs(dy_out) * tan_tol if abs(dy_out) > 1e-12 else False

            # Snap corner: prefer to preserve the incoming direction, adjust current point
            if in_horiz and out_vert:
                # Incoming is horizontal → fix Y of current to match prev Y
                # Outgoing is vertical  → fix X of current to match next X
                out[i] = (p_next[0], p_prev[1])
            elif in_vert and out_horiz:
                # Incoming is vertical   → fix X of current to match prev X
                # Outgoing is horizontal → fix Y of current to match next Y
                out[i] = (p_prev[0], p_next[1])
            elif in_horiz and not out_vert:
                # Incoming horiz, outgoing is diagonal → project current onto horizontal
                out[i] = (p_curr[0], p_prev[1])
            elif in_vert and not out_horiz:
                # Incoming vert, outgoing is diagonal → project current onto vertical
                out[i] = (p_prev[0], p_curr[1])
        return out

    new_coords = coords
    # Iterate up to 8 times to allow snapping to propagate
    for _iteration in range(8):
        prev = list(new_coords)
        new_coords = force_orthogonal(new_coords)
        if new_coords == prev:
            break  # Converged

    # FINAL BRUTAL PASS: Force ALL remaining diagonal edges to nearest H or V axis.
    # No tolerance check — every edge that is not perfectly H or V gets snapped.
    # This guarantees true 90-degree corners on the output polygon.
    def force_all_to_hv(coords_in):
        n_in = len(coords_in)
        out = list(coords_in)
        changed = True
        passes = 0
        while changed and passes < 10:
            changed = False
            passes += 1
            for i in range(n_in):
                p_prev = out[(i - 1) % n_in]
                p_curr = out[i]
                p_next = out[(i + 1) % n_in]

                dx_in  = p_curr[0] - p_prev[0]
                dy_in  = p_curr[1] - p_prev[1]
                dx_out = p_next[0] - p_curr[0]
                dy_out = p_next[1] - p_curr[1]

                # Classify each edge: H if |dy|<|dx|, V if |dx|<|dy|
                in_horiz  = abs(dy_in)  <= abs(dx_in)
                in_vert   = abs(dx_in)  <  abs(dy_in)
                out_horiz = abs(dy_out) <= abs(dx_out)
                out_vert  = abs(dx_out) <  abs(dy_out)

                new_pt = out[i]
                if in_horiz and out_vert:
                    new_pt = (p_next[0], p_prev[1])
                elif in_vert and out_horiz:
                    new_pt = (p_prev[0], p_next[1])
                elif in_horiz and not out_vert:
                    # Both horizontal → share same Y, keep X as-is
                    new_pt = (p_curr[0], p_prev[1])
                elif in_vert and not out_horiz:
                    # Both vertical → share same X, keep Y as-is
                    new_pt = (p_prev[0], p_curr[1])

                if new_pt != out[i]:
                    out[i] = new_pt
                    changed = True
        return out

    new_coords = force_all_to_hv(new_coords)

    # Remove degenerate (duplicate) points
    clean = []
    for pt in new_coords:
        if not clean or (abs(pt[0] - clean[-1][0]) > 1e-10 or abs(pt[1] - clean[-1][1]) > 1e-10):
            clean.append(pt)
    new_coords = clean if len(clean) >= 4 else new_coords

    new_coords.append(new_coords[0])

    # MBR Healing: If the building is a rectangle that was cropped by YOLO,
    # its corners are missing (chamfered). But since we are now rotated to its true
    # architectural angle, its bounding box perfectly reconstructs the missing corners!
    orthogonalized_geom = Polygon(new_coords)
    if not orthogonalized_geom.is_empty and orthogonalized_geom.is_valid:
        bounds = orthogonalized_geom.bounds
        mbr_area = (bounds[2] - bounds[0]) * (bounds[3] - bounds[1])
        # A chamfered rectangle has area ~90% of its MBR. L-shapes are < 75%.
        if mbr_area > 0 and (orthogonalized_geom.area / mbr_area) > 0.82:
            new_coords = [
                (bounds[0], bounds[1]),
                (bounds[2], bounds[1]),
                (bounds[2], bounds[3]),
                (bounds[0], bounds[3]),
                (bounds[0], bounds[1])
            ]

    # E. Rotate back
    final_coords = [rotate_point(x, y, cx, cy, angle) for x, y in new_coords]
    final_geom = make_valid(Polygon(final_coords))

    if final_geom.is_valid and not final_geom.is_empty and final_geom.area > 0:
        return final_geom
    return None



def orthogonalize_polygon(geom, angle_tolerance_deg: float = 25.0, simplify_tol: float = 0.25,
                          closing_radius: float = 0.35, strict_straight: bool = False):
    """
    Regularize a building polygon to have orthogonal (90°) corners while
    preserving the actual shape (L/U/compound) and healing tree-caused indents.

    Pipeline:

    Step A – Morphological Closing (r=1.0m):
        Fills tree-caused concavities narrower than 2m.
        Preserves real architectural features wider than 2m (L-shape steps etc).

    Step B – Post-close MBR snap (strict):
        Only snap to MBR if the healed shape is already near-perfect rectangle
        (IoU >= 0.87 AND Solidity >= 0.93). L-shapes and compound buildings pass
        through to Step C.

    Step C – Simplify 0.4m + RTA Orthogonalization:
        Simplify removes SAM pixelation noise while keeping all
        architectural corners. RTA snaps corners within the tolerance angle to exact 90°.

    Fallback – Shape-preserving simplified geometry for complex buildings, or MBR for simple boxes.
    """
    from shapely.geometry import Polygon, MultiPolygon
    from shapely.validation import make_valid
    import math

    if geom is None or geom.is_empty:
        return geom

    # Handle multi-polygons
    if isinstance(geom, MultiPolygon):
        parts = [orthogonalize_polygon(p, angle_tolerance_deg, simplify_tol, closing_radius, strict_straight)
                 for p in geom.geoms]
        parts = [p for p in parts if p and not p.is_empty]
        if not parts:
            return geom
        return parts[0] if len(parts) == 1 else MultiPolygon(parts)

    try:
        # Step 0 – Geographic coordinate safeguard
        is_geographic = False
        try:
            c = geom.centroid
            if abs(c.x) <= 180.0 and abs(c.y) <= 90.0:
                minx, miny, maxx, maxy = geom.bounds
                if (maxx - minx) < 0.1 and (maxy - miny) < 0.1:
                    is_geographic = True
        except Exception:
            pass

        # Increase the baseline simplification to flatten out the 30cm-50cm pixel steps
        baseline_simplify = max(simplify_tol, 0.25)
        
        if strict_straight:
            # Force high tolerance to ignore tree indentations and "tabrak lurus"
            # We ONLY increase simplify tolerance. We DO NOT increase closing_radius 
            # because a large round buffer destroys 90-degree corners and skews the MBR angle!
            baseline_simplify = max(baseline_simplify, 2.5)
            
        actual_simplify_tol   = baseline_simplify
        actual_closing_radius = closing_radius
        if is_geographic:
            # 1 degree ≈ 111,320 m → scale meters to degrees
            actual_simplify_tol   = baseline_simplify   / 111_320
            actual_closing_radius = closing_radius  / 111_320

        # ── Step A: Morphological Closing ────────────────────────────────────────────
        working_geom = geom
        try:
            closed = (geom
                      .buffer(actual_closing_radius)
                      .buffer(-actual_closing_radius))
            closed = make_valid(closed)
            if closed and not closed.is_empty and closed.area > 0:
                # Allow up to 60% area growth (tree cover ≤ ~37% of building area)
                if closed.area / geom.area <= 1.6:
                    working_geom = closed
        except Exception:
            pass  # Keep raw geom if buffer fails

        # Compute MBR of the working (healed) geometry
        mbr = working_geom.minimum_rotated_rectangle
        if not mbr or mbr.is_empty or mbr.area == 0:
            mbr = geom.minimum_rotated_rectangle
            if not mbr or mbr.is_empty:
                return geom

        # ── Step B: Post-close MBR Snap ──────────────────────────────────────────────
        inter_area = working_geom.intersection(mbr).area
        union_area  = working_geom.union(mbr).area
        iou      = inter_area / union_area if union_area > 0 else 0
        solidity = (working_geom.area / working_geom.convex_hull.area
                    if working_geom.convex_hull.area > 0 else 0)

        # Strict threshold: only rectangularize near-perfect boxes,
        # let L/U/compound shapes proceed to RTA orthogonalization.
        if iou >= 0.87 and solidity >= 0.93:
            return mbr

        # ── Step C: Simplify + RTA Orthogonalization ────────────────────────────
        result = _rta_orthogonalize(working_geom, angle_tolerance_deg, actual_simplify_tol, mbr)
        if result is not None:
            return result

        # ── Shape-Preserving Fallback ──
        # If the building has complex geometry (L/U/T-shape or irregular) and RTA fails,
        # DO NOT squash it to MBR box. Return a simplified, cleaned version of the original shape.
        if iou < 0.85 or solidity < 0.90:
            try:
                simplified = working_geom.simplify(actual_simplify_tol * 0.8, preserve_topology=True)
                simplified = make_valid(simplified)
                if simplified and not simplified.is_empty and simplified.area > 0:
                    return simplified
            except Exception:
                pass

        # ── Fallback: MBR ─────────────────────────────────────────────────────────────
        return mbr

    except Exception:
        return geom


def regularize_polygons(
    gdf: "geopandas.GeoDataFrame",
    raster_path: str = None,
    greenness_threshold: float = 15.0,
    enable: bool = True,
    angle_tolerance: float = 20.0,
    simplify_tolerance: float = 0.25,
    log_callback: Optional[Callable[[str], None]] = None,
) -> "geopandas.GeoDataFrame":
    """Apply orthogonalization to all polygons in a GeoDataFrame."""
    log = log_callback or (lambda x: None)

    if not enable or len(gdf) == 0:
        return gdf

    log(f"Regularisasi sudut poligon (toleransi={angle_tolerance}°)...")
    gdf = gdf.copy()
    
    import rasterio
    from rasterio.mask import mask as rio_mask
    from shapely.geometry import mapping
    import numpy as np

    is_geographic = False
    if gdf.crs and gdf.crs.is_geographic:
        is_geographic = True
    buf_dist = 2.5 / 111320.0 if is_geographic else 2.5

    src = None
    gdf_raster = gdf
    if raster_path:
        try:
            src = rasterio.open(raster_path)
            raster_crs = src.crs
            if gdf.crs and gdf.crs != raster_crs:
                gdf_raster = gdf.to_crs(raster_crs)
        except Exception as e:
            log(f"Gagal membuka raster untuk deteksi pohon: {e}")
            src = None

    new_geoms = []
    for geom, raster_geom in zip(gdf.geometry, gdf_raster.geometry):
        strict_straight = False
        if src is not None:
            try:
                # Buffer to catch nearby trees overlapping or touching the boundary
                check_geom = raster_geom.buffer(buf_dist)
                out_image, _ = rio_mask(src, [mapping(check_geom)], crop=True, nodata=0)
                if out_image.shape[0] >= 3:
                    r = out_image[0].astype(float)
                    g = out_image[1].astype(float)
                    b = out_image[2].astype(float)
                    valid = r > 0
                    if valid.sum() > 0:
                        r_v = r[valid]
                        g_v = g[valid]
                        b_v = b[valid]
                        # Hitung greenness per piksel, bukan rata-rata
                        greenness = g_v - np.maximum(r_v, b_v)
                        
                        # Hitung berapa persentase area yang merupakan pohon
                        tree_pixels = (greenness > greenness_threshold).sum()
                        tree_ratio = tree_pixels / valid.sum()
                        
                        # Jika lebih dari 5% area buffer adalah pohon, paksa jadi kotak lurus
                        if tree_ratio > 0.05:
                            strict_straight = True
            except Exception:
                pass
        
        new_geoms.append(orthogonalize_polygon(
            geom, 
            angle_tolerance_deg=angle_tolerance, 
            simplify_tol=simplify_tolerance, 
            strict_straight=strict_straight
        ))

    if src is not None:
        src.close()

    gdf["geometry"] = new_geoms

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
    simplify_tolerance: float = 0.75,
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
        progress(91, "Regularisasi sudut poligon bangunan (dengan deteksi pohon)...")
        gdf = regularize_polygons(
            gdf, 
            raster_path=original_raster_path,
            greenness_threshold=greenness_threshold,
            enable=True, 
            angle_tolerance=angle_tolerance, 
            simplify_tolerance=simplify_tolerance,
            log_callback=log
        )

    # 6. Resolve overlaps (avoid overlap)
    progress(93, "Menghilangkan poligon tumpang tindih (avoid overlap)...")
    gdf = resolve_overlaps(gdf, log_callback=log)

    log(f"=== POST-PROCESSING SELESAI: {len(gdf)} bangunan terdeteksi ===")
    progress(95, f"Selesai: {len(gdf)} bangunan terdeteksi")

    return gdf

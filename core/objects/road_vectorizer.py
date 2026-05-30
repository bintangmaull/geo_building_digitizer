"""
road_vectorizer.py
Direct vectorization of road mask to polygons and centerlines.

Strategy (Opsi E — no skeleton needed):
    1. Binary mask → contour extraction (cv2.findContours)
    2. Simplify polygons (Douglas-Peucker)
    3. Extract centerline from final polygon (medial axis)

This produces road polygons with NATURAL variable width,
unlike the old skeleton+fixed-buffer approach.
"""

import numpy as np
from typing import List, Optional, Tuple, Callable


def mask_to_road_polygons(
    binary_mask: np.ndarray,
    transform,
    crs,
    min_area_m2: float = 30.0,
    simplify_tol_m: float = 0.8,
    smooth_iterations: int = 2,
    log_callback: Optional[Callable[[str], None]] = None,
) -> "geopandas.GeoDataFrame":
    """
    Convert binary road mask directly to vector polygons.
    
    No skeleton, no fixed-width buffer — the polygon follows the actual
    mask shape, preserving natural road width variation.
    
    Args:
        binary_mask: (H, W) uint8 mask (0=background, 255=road)
        transform: Rasterio affine transform for georeferencing
        crs: Coordinate reference system
        min_area_m2: Minimum polygon area to keep (filters noise)
        simplify_tol_m: Douglas-Peucker simplification tolerance in meters
        smooth_iterations: Number of morphological smoothing passes
        log_callback: Logging function
    
    Returns:
        GeoDataFrame with road polygon geometries
    """
    import cv2
    import geopandas as gpd
    from shapely.geometry import Polygon, MultiPolygon
    from shapely.validation import make_valid
    from shapely.ops import unary_union

    log = log_callback or (lambda x: None)

    if binary_mask is None or binary_mask.max() == 0:
        return gpd.GeoDataFrame(geometry=[], crs=crs)

    # ── Step 1: Morphological smoothing ──────────────────────────────────────
    # Smooth the mask edges to reduce jaggedness before vectorizing
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    smoothed = binary_mask.copy()
    for _ in range(smooth_iterations):
        smoothed = cv2.morphologyEx(smoothed, cv2.MORPH_CLOSE, kernel)
        smoothed = cv2.morphologyEx(smoothed, cv2.MORPH_OPEN, kernel)

    # ── Step 2: Find contours ────────────────────────────────────────────────
    contours, hierarchy = cv2.findContours(
        smoothed, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_TC89_L1
    )

    if not contours or hierarchy is None:
        return gpd.GeoDataFrame(geometry=[], crs=crs)

    log(f"  Contours found: {len(contours)}")

    # ── Step 3: Convert contours to geo-polygons ─────────────────────────────
    # Determine if CRS is geographic (degrees) or projected (meters)
    is_geographic = crs and crs.is_geographic
    pixel_size_m = abs(transform.a) * (111320.0 if is_geographic else 1.0)
    
    # Convert simplify tolerance from meters to CRS units
    if is_geographic:
        simplify_tol = simplify_tol_m / 111320.0
        min_area = min_area_m2 / (111320.0 ** 2)
    else:
        simplify_tol = simplify_tol_m
        min_area = min_area_m2

    polygons = []
    hierarchy = hierarchy[0]  # hierarchy shape: (N, 4)

    for i, contour in enumerate(contours):
        # Only process outer contours (hierarchy[i][3] == -1 means no parent)
        if hierarchy[i][3] != -1:
            continue

        if len(contour) < 4:
            continue

        # Convert pixel coordinates to geographic coordinates
        coords = []
        for pt in contour:
            px, py = pt[0]
            # Pixel to geo using affine transform
            geo_x = transform.c + px * transform.a + py * transform.b
            geo_y = transform.f + px * transform.d + py * transform.e
            coords.append((geo_x, geo_y))

        if len(coords) < 4:
            continue

        # Close the ring
        coords.append(coords[0])

        try:
            poly = Polygon(coords)
            if not poly.is_valid:
                poly = make_valid(poly)
            
            if poly.is_empty or poly.area < min_area:
                continue

            # Check for holes (child contours)
            holes = []
            child_idx = hierarchy[i][2]  # First child
            while child_idx != -1:
                child_contour = contours[child_idx]
                if len(child_contour) >= 4:
                    hole_coords = []
                    for pt in child_contour:
                        px, py = pt[0]
                        geo_x = transform.c + px * transform.a + py * transform.b
                        geo_y = transform.f + px * transform.d + py * transform.e
                        hole_coords.append((geo_x, geo_y))
                    hole_coords.append(hole_coords[0])
                    if len(hole_coords) >= 4:
                        holes.append(hole_coords)
                child_idx = hierarchy[child_idx][0]  # Next sibling

            # Rebuild polygon with holes
            if holes:
                poly = Polygon(coords, holes)
                if not poly.is_valid:
                    poly = make_valid(poly)

            if not poly.is_empty and poly.area >= min_area:
                polygons.append(poly)

        except Exception:
            continue

    if not polygons:
        return gpd.GeoDataFrame(geometry=[], crs=crs)

    # ── Step 4: Merge touching polygons ──────────────────────────────────────
    merged = unary_union(polygons)
    if isinstance(merged, Polygon):
        poly_list = [merged]
    elif isinstance(merged, MultiPolygon):
        poly_list = list(merged.geoms)
    else:
        poly_list = [g for g in merged.geoms if isinstance(g, (Polygon, MultiPolygon))]

    # ── Step 5: Simplify ─────────────────────────────────────────────────────
    simplified = []
    for poly in poly_list:
        s = poly.simplify(simplify_tol, preserve_topology=True)
        if not s.is_empty and s.area >= min_area:
            simplified.append(s)

    log(f"  Road polygons after simplify: {len(simplified)}")

    gdf = gpd.GeoDataFrame(geometry=simplified, crs=crs)
    gdf["class"] = "Jalan"
    return gdf


def polygons_to_centerlines(
    gdf_polygons: "geopandas.GeoDataFrame",
    simplify_tol_m: float = 1.0,
    min_length_m: float = 10.0,
    log_callback: Optional[Callable[[str], None]] = None,
) -> "geopandas.GeoDataFrame":
    """
    Extract centerlines from road polygons using Voronoi-based medial axis.
    
    This produces centerlines that follow the actual road shape,
    computed from the final clean polygon rather than from noisy mask.
    
    Args:
        gdf_polygons: GeoDataFrame with road polygons
        simplify_tol_m: Simplification tolerance for centerlines
        min_length_m: Minimum centerline length to keep
        log_callback: Logging function
    
    Returns:
        GeoDataFrame with LineString centerline geometries
    """
    import geopandas as gpd
    from shapely.geometry import LineString, MultiLineString, Polygon
    from shapely.ops import linemerge, unary_union

    log = log_callback or (lambda x: None)

    if len(gdf_polygons) == 0:
        return gpd.GeoDataFrame(geometry=[], crs=gdf_polygons.crs)

    is_geographic = gdf_polygons.crs and gdf_polygons.crs.is_geographic
    
    # Convert tolerance to CRS units
    if is_geographic:
        simplify_tol = simplify_tol_m / 111320.0
        min_length = min_length_m / 111320.0
    else:
        simplify_tol = simplify_tol_m
        min_length = min_length_m

    all_centerlines = []

    for poly in gdf_polygons.geometry:
        if poly is None or poly.is_empty:
            continue

        try:
            # Use Voronoi-based centerline extraction
            centerline = _voronoi_centerline(poly, simplify_tol)
            if centerline is not None:
                if isinstance(centerline, LineString):
                    if centerline.length >= min_length:
                        all_centerlines.append(centerline)
                elif isinstance(centerline, MultiLineString):
                    for line in centerline.geoms:
                        if line.length >= min_length:
                            all_centerlines.append(line)
        except Exception:
            # Fallback: use polygon boundary simplified
            try:
                boundary = poly.boundary
                if hasattr(boundary, 'geoms'):
                    for line in boundary.geoms:
                        simplified = line.simplify(simplify_tol)
                        if simplified.length >= min_length:
                            all_centerlines.append(simplified)
                elif boundary.length >= min_length:
                    all_centerlines.append(boundary.simplify(simplify_tol))
            except Exception:
                continue

    if not all_centerlines:
        return gpd.GeoDataFrame(geometry=[], crs=gdf_polygons.crs)

    # Merge connected lines
    try:
        merged = linemerge(all_centerlines)
        if isinstance(merged, LineString):
            final_lines = [merged]
        elif isinstance(merged, MultiLineString):
            final_lines = list(merged.geoms)
        else:
            final_lines = all_centerlines
    except Exception:
        final_lines = all_centerlines

    log(f"  Centerlines extracted: {len(final_lines)}")

    gdf = gpd.GeoDataFrame(geometry=final_lines, crs=gdf_polygons.crs)
    gdf["class"] = "Jalan"
    return gdf


def _voronoi_centerline(polygon, simplify_tol: float) -> Optional["LineString"]:
    """
    Extract centerline from a polygon using densified boundary + Voronoi diagram.
    
    Strategy:
        1. Densify polygon boundary (add points every ~1m)
        2. Compute Voronoi diagram of boundary points
        3. Keep only Voronoi edges that are INSIDE the polygon
        4. These internal edges approximate the medial axis / centerline
        5. Prune short dead-end branches
        6. Merge into continuous LineString(s)
    """
    from shapely.geometry import LineString, MultiLineString, Point, MultiPoint
    from shapely.ops import linemerge, unary_union
    import numpy as np

    if polygon is None or polygon.is_empty:
        return None

    # Densify boundary: add points at regular intervals
    boundary = polygon.boundary
    if hasattr(boundary, 'geoms'):
        # MultiLineString boundary (polygon with holes)
        boundary = boundary.geoms[0]  # Use outer ring

    # Densify: interpolate points along boundary
    total_length = boundary.length
    if total_length == 0:
        return None
    
    # Point spacing: ~2x simplify_tol for efficiency
    spacing = max(simplify_tol * 2, total_length / 500)
    n_points = max(int(total_length / spacing), 20)
    
    boundary_points = []
    for i in range(n_points):
        frac = i / n_points
        pt = boundary.interpolate(frac, normalized=True)
        boundary_points.append((pt.x, pt.y))

    if len(boundary_points) < 4:
        return None

    # Compute Voronoi diagram
    try:
        from scipy.spatial import Voronoi
        
        points_array = np.array(boundary_points)
        vor = Voronoi(points_array)
        
        # Extract Voronoi edges that are inside the polygon
        internal_lines = []
        
        for ridge_idx, (p1_idx, p2_idx) in enumerate(vor.ridge_vertices):
            if p1_idx < 0 or p2_idx < 0:
                continue  # Skip infinite ridges
            
            v1 = vor.vertices[p1_idx]
            v2 = vor.vertices[p2_idx]
            
            # Check if both vertices are inside the polygon
            pt1 = Point(v1[0], v1[1])
            pt2 = Point(v2[0], v2[1])
            
            if polygon.contains(pt1) and polygon.contains(pt2):
                line = LineString([v1, v2])
                # Also check midpoint is inside (for thin polygons)
                mid = line.interpolate(0.5, normalized=True)
                if polygon.contains(mid):
                    internal_lines.append(line)
        
        if not internal_lines:
            return None
        
        # Merge connected lines
        merged = linemerge(internal_lines)
        
        # Simplify
        if isinstance(merged, LineString):
            result = merged.simplify(simplify_tol, preserve_topology=True)
        elif isinstance(merged, MultiLineString):
            simplified_parts = []
            for line in merged.geoms:
                s = line.simplify(simplify_tol, preserve_topology=True)
                if s.length > simplify_tol * 3:
                    simplified_parts.append(s)
            if simplified_parts:
                result = linemerge(simplified_parts)
            else:
                return None
        else:
            return None
        
        return result
        
    except ImportError:
        # scipy not available — fallback to simple approach
        # Use polygon.representative_point chain
        return None
    except Exception:
        return None

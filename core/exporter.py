"""
exporter.py
Exports a GeoDataFrame of building polygons to Shapefile format.
Handles CRS reprojection to the original raster's CRS.
Adds metadata attributes (area, perimeter, ID) to each feature.
"""

import os
import math
from pathlib import Path
from typing import Callable, Optional


def export_to_shapefile(
    gdf: "geopandas.GeoDataFrame",
    output_path: str,
    source_raster_path: Optional[str] = None,
    log_callback: Optional[Callable[[str], None]] = None,
) -> str:
    """
    Export building polygons to Shapefile (.shp).

    Args:
        gdf: GeoDataFrame with building polygons
        output_path: Full path to output .shp file (directory will be created)
        source_raster_path: Optional path to original raster for CRS reference
        log_callback: Optional log function

    Returns:
        str: Path to the exported .shp file
    """
    import geopandas as gpd

    log = log_callback or (lambda x: None)

    if len(gdf) == 0:
        raise ValueError("Tidak ada bangunan yang terdeteksi untuk diekspor.")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Reproject to source raster CRS if provided
    if source_raster_path and os.path.isfile(source_raster_path):
        try:
            import rasterio
            with rasterio.open(source_raster_path) as src:
                target_crs = src.crs
            if target_crs and gdf.crs and gdf.crs != target_crs:
                log(f"Reproyek CRS: {gdf.crs} -> {target_crs}")
                gdf = gdf.to_crs(target_crs)
        except Exception as e:
            log(f"Peringatan: Gagal reproyek CRS: {e}")

    # Add attribute columns
    gdf = gdf.copy()
    gdf = gdf.reset_index(drop=True)
    gdf["ID"] = gdf.index + 1

    # Calculate area and perimeter
    # If CRS is geographic, project to metric for calculations
    if gdf.crs and gdf.crs.is_geographic:
        try:
            utm_crs = gdf.estimate_utm_crs()
            gdf_metric = gdf.to_crs(utm_crs)
            gdf["Area_m2"] = gdf_metric.geometry.area.round(2)
            gdf["Perim_m"] = gdf_metric.geometry.length.round(2)
        except Exception:
            gdf["Area_m2"] = gdf.geometry.area.round(6)
            gdf["Perim_m"] = gdf.geometry.length.round(6)
    else:
        gdf["Area_m2"] = gdf.geometry.area.round(2)
        gdf["Perim_m"] = gdf.geometry.length.round(2)

    # Shapefile column names max 10 chars
    export_gdf = gdf[["ID", "Area_m2", "Perim_m", "geometry"]].copy()

    # Validate geometries and ensure strictly Polygons or MultiPolygons are exported
    from shapely.geometry import Polygon, MultiPolygon, GeometryCollection
    
    def extract_polygons(geom):
        if geom is None or geom.is_empty:
            return None
        # Clean the geometry first
        if not geom.is_valid:
            geom = geom.buffer(0)
        
        if isinstance(geom, (Polygon, MultiPolygon)):
            return geom
        elif isinstance(geom, GeometryCollection):
            polys = [g for g in geom.geoms if isinstance(g, (Polygon, MultiPolygon))]
            if not polys:
                return None
            elif len(polys) == 1:
                return polys[0]
            else:
                return MultiPolygon(polys)
        else:
            return None

    export_gdf["geometry"] = export_gdf["geometry"].apply(extract_polygons)
    export_gdf = export_gdf.dropna(subset=["geometry"])
    export_gdf = export_gdf[export_gdf.geometry.apply(lambda g: isinstance(g, (Polygon, MultiPolygon)))]
    export_gdf = export_gdf[~export_gdf.geometry.is_empty]

    log(f"Mengekspor {len(export_gdf)} bangunan ke: {output_path}")

    try:
        export_gdf.to_file(output_path, driver="ESRI Shapefile", encoding="UTF-8")
        log(f"Ekspor berhasil: {output_path}")
        log(f"  File pendamping: .dbf, .prj, .shx juga telah dibuat")
        return output_path
    except Exception as e:
        raise RuntimeError(f"Gagal mengekspor Shapefile: {e}")


def get_output_summary(output_shp_path: str) -> dict:
    """Read the exported shapefile and return a summary."""
    try:
        import geopandas as gpd
        gdf = gpd.read_file(output_shp_path)
        return {
            "count": len(gdf),
            "total_area_m2": gdf["Area_m2"].sum() if "Area_m2" in gdf.columns else 0,
            "avg_area_m2": gdf["Area_m2"].mean() if "Area_m2" in gdf.columns else 0,
            "crs": str(gdf.crs),
            "file_size_kb": os.path.getsize(output_shp_path) // 1024,
        }
    except Exception:
        return {}

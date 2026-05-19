"""
diagnose_pipeline.py
Runs the full post-processing pipeline step by step on existing mask tiles
and reports how many polygons survive each filter stage.
Run this AFTER a digitization run has completed so mask files exist.
"""
import os
import sys
import glob

def run_diagnosis():
    project_root = os.path.abspath(".")
    sys.path.insert(0, project_root)

    raster_path = None
    # Try to find the last used raster automatically
    for candidate in ["contoh3.tif", "contoh2.tif", "contoh.tif"]:
        p = os.path.join(project_root, candidate)
        if os.path.exists(p):
            raster_path = p
            break
    if not raster_path:
        # Search output folder
        tifs = glob.glob(os.path.join(project_root, "*.tif"))
        tifs = [t for t in tifs if "mask" not in t and "tile" not in t]
        if tifs:
            raster_path = tifs[0]

    if not raster_path:
        print("[ERROR] Could not find input raster .tif file in project root!")
        return

    print(f"Raster: {raster_path}")

    # Find mask tiles
    masks_dir = os.path.join(project_root, "output", "masks")
    if not os.path.exists(masks_dir):
        masks_dir = os.path.join(project_root, "temp", "masks")
    masks = glob.glob(os.path.join(masks_dir, "mask_*.tif"))

    if not masks:
        print(f"[ERROR] No mask files found in {masks_dir}")
        print("Please run a digitization pass first, then run this script.")
        return

    print(f"Found {len(masks)} mask tiles in {masks_dir}")

    # Build (mask_path, tile_meta) list using rasterio for metadata
    import rasterio
    mask_and_meta = []
    for mask_path in sorted(masks):
        with rasterio.open(mask_path) as src:
            tile_meta = {
                "transform": src.transform,
                "crs": src.crs,
            }
        mask_and_meta.append((mask_path, tile_meta))

    from core.postprocess import (
        merge_tile_masks,
        filter_by_area,
        filter_by_aspect_ratio,
        filter_by_compactness,
        filter_non_building_colors,
        filter_shadows,
        filter_vegetation,
        regularize_polygons,
    )

    def log(msg):
        print(f"  {msg}")

    print("\n=== STEP-BY-STEP PIPELINE DIAGNOSIS ===\n")

    # Step 1: Merge
    print("STEP 1: Merging tile masks -> polygons...")
    gdf = merge_tile_masks(mask_and_meta, raster_path, log_callback=log)
    print(f"  >>> Polygons after merge: {len(gdf)}")
    if len(gdf) == 0:
        print("[STOP] No polygons generated. SAM mask tiles are empty (all zeros).")
        print("  Cause: The model is not detecting anything. Check model and SAM thresholds.")
        return

    # Step 2: Area filter (min=20, max=2000)
    print("\nSTEP 2: Area filter (min=20 m2, max=2000 m2)...")
    gdf2 = filter_by_area(gdf, min_area_m2=20.0, max_area_m2=2000.0, log_callback=log)
    print(f"  >>> Polygons after area filter: {len(gdf2)} (removed {len(gdf)-len(gdf2)})")

    # Step 3: Aspect ratio filter
    print("\nSTEP 3: Aspect ratio filter (max=8.0)...")
    gdf3 = filter_by_aspect_ratio(gdf2, max_ratio=8.0, log_callback=log)
    print(f"  >>> Polygons after aspect filter: {len(gdf3)} (removed {len(gdf2)-len(gdf3)})")

    # Step 3b: Compactness
    print("\nSTEP 3b: Compactness filter (min=0.20)...")
    gdf3b = filter_by_compactness(gdf3, min_compactness=0.20, log_callback=log)
    print(f"  >>> Polygons after compactness filter: {len(gdf3b)} (removed {len(gdf3)-len(gdf3b)})")

    # Step 3c: Agricultural field filter
    print("\nSTEP 3c: Agricultural field filter (>500m2 low-variance)...")
    gdf3c = filter_non_building_colors(gdf3b, raster_path, max_field_area_m2=500.0, log_callback=log)
    print(f"  >>> Polygons after field filter: {len(gdf3c)} (removed {len(gdf3b)-len(gdf3c)})")

    # Step 4: Shadow filter (threshold=55)
    print("\nSTEP 4: Shadow filter (brightness_threshold=55)...")
    gdf4 = filter_shadows(gdf3c, raster_path, brightness_threshold=55, log_callback=log)
    print(f"  >>> Polygons after shadow filter: {len(gdf4)} (removed {len(gdf3c)-len(gdf4)})")

    # Step 4b: Vegetation filter (greenness=15)
    print("\nSTEP 4b: Vegetation filter (greenness_threshold=15)...")
    gdf4b = filter_vegetation(gdf4, raster_path, greenness_threshold=15.0, log_callback=log)
    print(f"  >>> Polygons after vegetation filter: {len(gdf4b)} (removed {len(gdf4)-len(gdf4b)})")

    print(f"\n=== FINAL RESULT: {len(gdf4b)} polygons survived all filters ===")

    if len(gdf4b) == 0:
        print("\n[DIAGNOSIS] ALL polygons were removed! Check which step had the biggest drop.")
    else:
        # Print area stats
        import geopandas as gpd
        if gdf4b.crs and gdf4b.crs.is_geographic:
            gdf4b_metric = gdf4b.to_crs(gdf4b.estimate_utm_crs())
        else:
            gdf4b_metric = gdf4b
        areas = gdf4b_metric.geometry.area
        print(f"\nSurviving polygon area stats (m2):")
        print(f"  Min: {areas.min():.1f}, Max: {areas.max():.1f}, Mean: {areas.mean():.1f}")

if __name__ == "__main__":
    run_diagnosis()

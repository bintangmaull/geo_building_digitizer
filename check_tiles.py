import os
import glob
import numpy as np
import rasterio

def inspect_tiles():
    project_root = os.path.abspath(".")
    tiles_dir = os.path.join(project_root, "temp", "tiles")
    masks_dir = os.path.join(project_root, "output", "masks") # wait, let's look at masks too!
    
    print("=== INSPECTING GENERATED TILES ===")
    if not os.path.exists(tiles_dir):
        print(f"[ERROR] Tiles directory not found at: {tiles_dir}")
        return
        
    tiles = glob.glob(os.path.join(tiles_dir, "*.tif"))
    print(f"Found {len(tiles)} tiles in temp directory: {tiles_dir}")
    
    if not tiles:
        print("[WARNING] No tiles found. Have they been cleaned up?")
        return

    # Check first 3 tiles
    for tile_path in sorted(tiles)[:3]:
        name = os.path.basename(tile_path)
        try:
            with rasterio.open(tile_path) as src:
                data = src.read()
                print(f"\nTile: {name}")
                print(f"  Shape: {data.shape}")
                print(f"  Dtype: {src.dtypes}")
                print(f"  Min value: {data.min()}")
                print(f"  Max value: {data.max()}")
                print(f"  Mean value: {data.mean():.2f}")
                # check how many non-zero pixels
                non_zero = np.count_nonzero(data)
                pct = (non_zero / data.size) * 100
                print(f"  Non-zero pixel percentage: {pct:.2f}%")
        except Exception as e:
            print(f"  [ERROR] Reading tile failed: {e}")

    # Also inspect masks if any exist!
    print("\n=== INSPECTING GENERATED MASKS ===")
    masks = glob.glob(os.path.join(project_root, "output", "**", "mask_*.tif"), recursive=True)
    if not masks:
        # Check in the custom output folder if configured
        masks = glob.glob(os.path.join(project_root, "temp", "masks", "*.tif"))
        
    print(f"Found {len(masks)} masks.")
    for mask_path in sorted(masks)[:3]:
        name = os.path.basename(mask_path)
        try:
            with rasterio.open(mask_path) as src:
                data = src.read(1)
                print(f"\nMask: {name}")
                print(f"  Min: {data.min()}, Max: {data.max()}, Mean: {data.mean():.4f}")
                non_zero = np.count_nonzero(data > 127)
                print(f"  Pixels > 127: {non_zero}")
        except Exception as e:
            print(f"  [ERROR] Reading mask failed: {e}")

if __name__ == "__main__":
    inspect_tiles()

"""
road_dataset_generator.py
Generate training dataset for U-Net road segmentation.

Input:
    - GeoTIFF raster (drone imagery)
    - Shapefile with road polygons OR centerlines

Output:
    - tiles/images/  → RGB chips (512x512 PNG)
    - tiles/masks/   → Binary road masks (512x512 PNG, 0=background, 255=road)

If input SHP contains LineStrings (centerlines), they are buffered to create
polygon masks with realistic road width.
"""

import os
import math
import random
import numpy as np
from pathlib import Path
from typing import Callable, Optional, Tuple


def generate_road_training_data(
    geotiff_path: str,
    shp_path: str,
    output_dir: str,
    chip_size: int = 512,
    target_gsd: float = 0.15,
    buffer_width_m: float = 3.0,
    train_split: float = 0.85,
    augment_flip: bool = True,
    log_callback: Optional[Callable[[str], None]] = None,
    progress_callback: Optional[Callable[[float, str], None]] = None,
) -> bool:
    """
    Generate training chips and masks for road segmentation.
    
    Args:
        geotiff_path: Path to input GeoTIFF (drone imagery)
        shp_path: Path to Shapefile with road geometry (Polygon or LineString)
        output_dir: Output directory for dataset
        chip_size: Size of each training chip in pixels
        target_gsd: Target ground sample distance in meters
        buffer_width_m: Buffer width for LineString → Polygon conversion (meters)
        train_split: Fraction of data for training (rest = validation)
        augment_flip: Whether to add flipped versions for augmentation
        log_callback: Logging function
        progress_callback: Progress function(percent, message)
    
    Returns:
        True if successful
    """
    import rasterio
    import geopandas as gpd
    from rasterio.windows import Window
    from rasterio.features import rasterize
    from shapely.geometry import box, LineString, MultiLineString, Polygon, MultiPolygon
    from shapely.ops import unary_union
    from PIL import Image

    log = log_callback or print
    progress = progress_callback or (lambda p, m: None)

    log("=== GENERATING ROAD SEGMENTATION DATASET ===")
    log(f"Raster: {geotiff_path}")
    log(f"SHP: {shp_path}")
    log(f"Chip size: {chip_size}px, Target GSD: {target_gsd}m")

    # ── 1. Load and prepare road geometries ──────────────────────────────────
    progress(5, "Memuat geometri jalan...")
    
    gdf = gpd.read_file(shp_path)
    log(f"Geometri dimuat: {len(gdf)} fitur")

    with rasterio.open(geotiff_path) as src:
        raster_crs = src.crs
        raster_bounds = box(*src.bounds)
        pixel_size_x = abs(src.transform.a)
        pixel_size_y = abs(src.transform.e)
        raster_width = src.width
        raster_height = src.height
        raster_transform = src.transform

    # Reproject SHP to raster CRS if needed
    if gdf.crs and gdf.crs != raster_crs:
        gdf = gdf.to_crs(raster_crs)
        log(f"SHP diproyeksikan ke CRS raster: {raster_crs}")

    # Determine if geometries are lines or polygons
    geom_types = set(gdf.geometry.geom_type)
    log(f"Tipe geometri: {geom_types}")

    # If LineStrings, buffer them to create road polygons
    is_geographic = raster_crs and raster_crs.is_geographic
    
    if geom_types & {"LineString", "MultiLineString"}:
        if is_geographic:
            buf_deg = buffer_width_m / 111320.0
        else:
            buf_deg = buffer_width_m
        
        log(f"Buffering centerlines dengan lebar {buffer_width_m}m...")
        gdf["geometry"] = gdf.geometry.buffer(buf_deg, cap_style=2, join_style=1)
        log("Centerlines di-buffer menjadi polygon.")

    # Merge all road polygons into one unified geometry
    road_union = unary_union(gdf.geometry)
    log(f"Road union area: {road_union.area:.6f}")

    # ── 2. Calculate chip parameters ─────────────────────────────────────────
    progress(10, "Menghitung parameter chip...")

    # Current GSD
    if is_geographic:
        current_gsd = pixel_size_x * 111320.0
    else:
        current_gsd = pixel_size_x

    # Downsample factor to reach target GSD
    scale_factor = current_gsd / target_gsd
    if scale_factor > 1.5:
        log(f"⚠ Raster GSD ({current_gsd:.3f}m) lebih kasar dari target ({target_gsd}m). Menggunakan native resolution.")
        effective_chip_px = chip_size
    else:
        effective_chip_px = int(chip_size * (target_gsd / current_gsd))

    log(f"GSD raster: {current_gsd:.4f}m, effective chip: {effective_chip_px}px di raster")

    # ── 3. Generate chip positions ───────────────────────────────────────────
    progress(15, "Menentukan posisi chip...")

    # Strategy: grid-based with extra chips centered on road areas
    step = effective_chip_px  # Non-overlapping grid
    
    positions = []  # (col_off, row_off)
    
    # Grid positions
    for row_off in range(0, raster_height - effective_chip_px + 1, step):
        for col_off in range(0, raster_width - effective_chip_px + 1, step):
            positions.append((col_off, row_off))

    log(f"Grid positions: {len(positions)}")

    # ── 4. Extract chips and masks ───────────────────────────────────────────
    progress(20, "Mengekstrak chip dan mask...")

    # Create output directories
    img_train_dir = os.path.join(output_dir, "train", "images")
    msk_train_dir = os.path.join(output_dir, "train", "masks")
    img_val_dir = os.path.join(output_dir, "val", "images")
    msk_val_dir = os.path.join(output_dir, "val", "masks")
    
    for d in [img_train_dir, msk_train_dir, img_val_dir, msk_val_dir]:
        os.makedirs(d, exist_ok=True)

    # Shuffle positions for random train/val split
    random.shuffle(positions)
    split_idx = int(len(positions) * train_split)

    chips_saved = 0
    chips_with_road = 0
    total_positions = len(positions)

    with rasterio.open(geotiff_path) as src:
        for i, (col_off, row_off) in enumerate(positions):
            if i % 50 == 0:
                pct = 20 + int(70 * i / total_positions)
                progress(pct, f"Chip {i+1}/{total_positions}")

            # Read RGB tile
            window = Window(col_off, row_off, effective_chip_px, effective_chip_px)
            
            try:
                n_bands = min(src.count, 3)
                data = src.read(list(range(1, n_bands + 1)), window=window)
            except Exception:
                continue

            # Check if tile is valid (not all black/nodata)
            if data.max() == 0:
                continue

            # Convert to (H, W, 3) uint8
            rgb = np.clip(data, 0, 255).astype(np.uint8)
            if rgb.shape[0] < 3:
                rgb = np.repeat(rgb, 3, axis=0)
            rgb = np.transpose(rgb[:3], (1, 2, 0))  # (3, H, W) → (H, W, 3)

            # Resize to chip_size if needed
            if effective_chip_px != chip_size:
                from PIL import Image as PILImage
                rgb_pil = PILImage.fromarray(rgb)
                rgb_pil = rgb_pil.resize((chip_size, chip_size), PILImage.LANCZOS)
                rgb = np.array(rgb_pil)

            # ── Create road mask for this chip ──
            tile_transform = src.window_transform(window)
            
            # Rasterize road polygons into chip
            try:
                mask = rasterize(
                    [(road_union, 1)],
                    out_shape=(effective_chip_px, effective_chip_px),
                    transform=tile_transform,
                    fill=0,
                    dtype=np.uint8,
                )
            except Exception:
                mask = np.zeros((effective_chip_px, effective_chip_px), dtype=np.uint8)

            # Resize mask to chip_size if needed
            if effective_chip_px != chip_size:
                mask_pil = Image.fromarray(mask * 255)
                mask_pil = mask_pil.resize((chip_size, chip_size), Image.NEAREST)
                mask = np.array(mask_pil)
            else:
                mask = mask * 255

            # Track road coverage
            road_ratio = (mask > 0).sum() / mask.size
            has_road = road_ratio > 0.01  # At least 1% road pixels
            
            if has_road:
                chips_with_road += 1

            # Skip chips with zero road content (keep some for negative examples)
            # Keep ~20% of empty chips as negative examples
            if not has_road and random.random() > 0.2:
                continue

            # Determine train/val split
            is_train = i < split_idx
            img_dir = img_train_dir if is_train else img_val_dir
            msk_dir = msk_train_dir if is_train else msk_val_dir

            # Save chip
            chip_name = f"road_{chips_saved:05d}.png"
            Image.fromarray(rgb).save(os.path.join(img_dir, chip_name))
            Image.fromarray(mask).save(os.path.join(msk_dir, chip_name))
            chips_saved += 1

            # Augmentation: horizontal + vertical flip
            if augment_flip and has_road and is_train:
                # Horizontal flip
                rgb_h = np.fliplr(rgb).copy()
                mask_h = np.fliplr(mask).copy()
                chip_name_h = f"road_{chips_saved:05d}.png"
                Image.fromarray(rgb_h).save(os.path.join(img_dir, chip_name_h))
                Image.fromarray(mask_h).save(os.path.join(msk_dir, chip_name_h))
                chips_saved += 1

                # Vertical flip
                rgb_v = np.flipud(rgb).copy()
                mask_v = np.flipud(mask).copy()
                chip_name_v = f"road_{chips_saved:05d}.png"
                Image.fromarray(rgb_v).save(os.path.join(img_dir, chip_name_v))
                Image.fromarray(mask_v).save(os.path.join(msk_dir, chip_name_v))
                chips_saved += 1

    # ── 5. Summary ───────────────────────────────────────────────────────────
    progress(95, "Dataset selesai!")
    
    n_train = len(os.listdir(img_train_dir))
    n_val = len(os.listdir(img_val_dir))
    
    log(f"\n=== DATASET GENERATION COMPLETE ===")
    log(f"Total chips: {chips_saved}")
    log(f"  Train: {n_train}")
    log(f"  Val: {n_val}")
    log(f"  Chips with road: {chips_with_road}")
    log(f"Output: {output_dir}")
    
    progress(100, f"Dataset siap: {chips_saved} chips")
    return chips_saved > 0


# ══════════════════════════════════════════════════════════════════════════════
# PYTORCH DATASET CLASS
# ══════════════════════════════════════════════════════════════════════════════

class RoadSegmentationDataset:
    """
    PyTorch Dataset for road segmentation training.
    
    Loads image-mask pairs from directory structure:
        split/images/road_00001.png
        split/masks/road_00001.png
    
    Applies random augmentations during training:
        - Random rotation (0, 90, 180, 270)
        - Random brightness/contrast jitter
        - Random Gaussian noise
    """

    def __init__(self, data_dir: str, split: str = "train", augment: bool = True):
        """
        Args:
            data_dir: Root dataset directory
            split: 'train' or 'val'
            augment: Whether to apply random augmentations
        """
        self.img_dir = os.path.join(data_dir, split, "images")
        self.msk_dir = os.path.join(data_dir, split, "masks")
        self.augment = augment and (split == "train")
        
        self.filenames = sorted([
            f for f in os.listdir(self.img_dir)
            if f.endswith(".png")
        ])
        
        # Verify masks exist
        valid = []
        for f in self.filenames:
            if os.path.exists(os.path.join(self.msk_dir, f)):
                valid.append(f)
        self.filenames = valid

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        import torch
        from PIL import Image

        fname = self.filenames[idx]
        
        # Load image and mask
        img = np.array(Image.open(os.path.join(self.img_dir, fname)).convert("RGB"))
        msk = np.array(Image.open(os.path.join(self.msk_dir, fname)).convert("L"))

        # Augmentation
        if self.augment:
            img, msk = self._augment(img, msk)

        # Normalize image to [0, 1]
        img = img.astype(np.float32) / 255.0
        
        # Mask to binary [0, 1]
        msk = (msk > 127).astype(np.float32)

        # Convert to tensors: (H, W, C) → (C, H, W)
        img_tensor = torch.from_numpy(img).permute(2, 0, 1)  # (3, H, W)
        msk_tensor = torch.from_numpy(msk).unsqueeze(0)       # (1, H, W)

        return img_tensor, msk_tensor

    def _augment(self, img: np.ndarray, msk: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Apply random augmentations."""
        # Random 90-degree rotation
        k = random.randint(0, 3)
        if k > 0:
            img = np.rot90(img, k).copy()
            msk = np.rot90(msk, k).copy()

        # Random horizontal flip
        if random.random() > 0.5:
            img = np.fliplr(img).copy()
            msk = np.fliplr(msk).copy()

        # Random brightness jitter (±20%)
        if random.random() > 0.5:
            factor = random.uniform(0.8, 1.2)
            img = np.clip(img * factor, 0, 255).astype(np.uint8)

        # Random contrast jitter
        if random.random() > 0.5:
            factor = random.uniform(0.8, 1.2)
            mean = img.mean()
            img = np.clip((img - mean) * factor + mean, 0, 255).astype(np.uint8)

        # Random Gaussian noise
        if random.random() > 0.7:
            noise = np.random.normal(0, 5, img.shape).astype(np.float32)
            img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)

        return img, msk

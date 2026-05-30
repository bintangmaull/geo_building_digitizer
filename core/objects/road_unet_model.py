"""
road_unet_model.py
U-Net semantic segmentation model for road extraction.

Architecture: U-Net with ResNet-34 encoder (via segmentation_models_pytorch)
Loss: BCE + Dice + clDice (centerline Dice for connectivity)

clDice ensures the predicted road mask maintains topological connectivity —
roads don't break at intersections or under tree canopy.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# ══════════════════════════════════════════════════════════════════════════════
# MODEL
# ══════════════════════════════════════════════════════════════════════════════

def create_road_unet(encoder_name="resnet34", encoder_weights="imagenet", in_channels=3):
    """
    Create a U-Net model for binary road segmentation.
    
    Args:
        encoder_name: Backbone encoder (resnet34 is lightweight, fits 4GB VRAM)
        encoder_weights: Pre-trained weights (ImageNet for transfer learning)
        in_channels: Number of input channels (3 for RGB)
    
    Returns:
        smp.Unet model with sigmoid activation
    """
    import segmentation_models_pytorch as smp

    model = smp.Unet(
        encoder_name=encoder_name,
        encoder_weights=encoder_weights,
        in_channels=in_channels,
        classes=1,  # Binary: road vs not-road
        activation=None,  # We apply sigmoid in loss/inference
    )
    return model


# ══════════════════════════════════════════════════════════════════════════════
# LOSS FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def soft_skeletonize(mask, iters=10):
    """
    Differentiable soft skeletonization using iterative min-pooling.
    Approximates the medial axis / centerline of a binary mask.
    
    Args:
        mask: (B, 1, H, W) tensor, values in [0, 1]
        iters: Number of erosion iterations (more = thinner skeleton)
    
    Returns:
        Soft skeleton tensor (B, 1, H, W)
    """
    # Iterative erosion: subtract eroded version to get boundary
    # Then intersect with original to get skeleton approximation
    kernel_size = 3
    padding = kernel_size // 2
    
    skeleton = mask.clone()
    for _ in range(iters):
        # Min-pool = erosion for binary masks
        eroded = -F.max_pool2d(-skeleton, kernel_size, stride=1, padding=padding)
        # Skeleton = pixels that are in mask but would be removed by erosion
        # We accumulate the "center" pixels
        skeleton = torch.min(skeleton, eroded + (mask - eroded) * 0.5)
    
    # Final skeleton: thin center of the mask
    eroded_final = -F.max_pool2d(-mask, kernel_size, stride=1, padding=padding)
    skeleton = mask * (1.0 - eroded_final) + eroded_final * skeleton
    
    return skeleton


def cl_dice_loss(pred, target, iters=10, smooth=1.0):
    """
    Centerline Dice (clDice) loss for topological connectivity.
    
    Penalizes predictions where the road centerline is broken.
    This forces the model to maintain continuous road segments.
    
    Reference: "clDice - a Novel Topology-Preserving Loss Function for 
    Tubular Structure Segmentation" (Shit et al., 2021)
    
    Args:
        pred: (B, 1, H, W) predicted probabilities (after sigmoid)
        target: (B, 1, H, W) ground truth binary mask
        iters: Skeletonization iterations
        smooth: Smoothing factor to avoid division by zero
    
    Returns:
        1 - clDice (loss value, lower is better)
    """
    # Soft skeletonize both prediction and target
    skel_pred = soft_skeletonize(pred, iters=iters)
    skel_target = soft_skeletonize(target, iters=iters)
    
    # Topology Precision: how much of predicted skeleton is inside target mask
    tprec_num = (skel_pred * target).sum(dim=(2, 3)) + smooth
    tprec_den = skel_pred.sum(dim=(2, 3)) + smooth
    tprec = tprec_num / tprec_den
    
    # Topology Sensitivity: how much of target skeleton is inside predicted mask
    tsens_num = (skel_target * pred).sum(dim=(2, 3)) + smooth
    tsens_den = skel_target.sum(dim=(2, 3)) + smooth
    tsens = tsens_num / tsens_den
    
    # clDice = harmonic mean of tprec and tsens
    cl_dice = 2.0 * (tprec * tsens) / (tprec + tsens + 1e-7)
    
    return 1.0 - cl_dice.mean()


def dice_loss(pred, target, smooth=1.0):
    """Standard Dice loss."""
    pred_flat = pred.view(-1)
    target_flat = target.view(-1)
    
    intersection = (pred_flat * target_flat).sum()
    return 1.0 - (2.0 * intersection + smooth) / (pred_flat.sum() + target_flat.sum() + smooth)


class RoadSegmentationLoss(nn.Module):
    """
    Combined loss: BCE + Dice + clDice
    
    Weights:
        - BCE (0.5): pixel-level accuracy
        - Dice (0.3): global overlap
        - clDice (0.2): topological connectivity
    """
    
    def __init__(self, bce_weight=0.5, dice_weight=0.3, cldice_weight=0.2, cldice_iters=10):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.cldice_weight = cldice_weight
        self.cldice_iters = cldice_iters
        self.bce = nn.BCEWithLogitsLoss()
    
    def forward(self, logits, targets):
        """
        Args:
            logits: (B, 1, H, W) raw model output (before sigmoid)
            targets: (B, 1, H, W) ground truth binary mask
        """
        # BCE loss (operates on logits)
        loss_bce = self.bce(logits, targets)
        
        # Apply sigmoid for Dice and clDice
        pred = torch.sigmoid(logits)
        
        # Dice loss
        loss_dice = dice_loss(pred, targets)
        
        # clDice loss (only if target has road pixels — avoid NaN on empty masks)
        if targets.sum() > 10:
            loss_cldice = cl_dice_loss(pred, targets, iters=self.cldice_iters)
        else:
            loss_cldice = torch.tensor(0.0, device=logits.device)
        
        total = (self.bce_weight * loss_bce + 
                 self.dice_weight * loss_dice + 
                 self.cldice_weight * loss_cldice)
        
        return total, {
            "bce": loss_bce.item(),
            "dice": loss_dice.item(),
            "cldice": loss_cldice.item(),
            "total": total.item(),
        }


# ══════════════════════════════════════════════════════════════════════════════
# INFERENCE UTILITIES
# ══════════════════════════════════════════════════════════════════════════════

def predict_tile(model, tile_rgb: np.ndarray, device="cuda", threshold=0.5) -> np.ndarray:
    """
    Run inference on a single tile.
    
    Args:
        model: Trained U-Net model
        tile_rgb: (H, W, 3) uint8 RGB image
        device: 'cuda' or 'cpu'
        threshold: Probability threshold for binary mask
    
    Returns:
        (H, W) binary mask (uint8, 0 or 255)
    """
    model.eval()
    
    # Normalize to [0, 1]
    img = tile_rgb.astype(np.float32) / 255.0
    
    # (H, W, 3) → (1, 3, H, W)
    tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(device)
    
    with torch.no_grad():
        logits = model(tensor)
        prob = torch.sigmoid(logits)
    
    # (1, 1, H, W) → (H, W)
    prob_np = prob.squeeze().cpu().numpy()
    
    binary = (prob_np >= threshold).astype(np.uint8) * 255
    return binary


def predict_raster_center_crop(
    model,
    raster_path: str,
    tile_size: int = 1024,
    crop_size: int = 512,
    device: str = "cuda",
    threshold: float = 0.5,
    log_callback=None,
    progress_callback=None,
) -> np.ndarray:
    """
    Predict road mask for entire raster using center-crop stitching.
    
    Strategy:
        - Slide window of tile_size with step = crop_size (50% overlap)
        - Predict each tile
        - Only keep the center crop_size x crop_size from each prediction
        - Stitch center crops into full-resolution mask
    
    This eliminates tile-boundary artifacts completely.
    
    Args:
        model: Trained U-Net model
        raster_path: Path to input GeoTIFF
        tile_size: Size of each tile fed to model (e.g., 1024)
        crop_size: Size of center crop to keep (e.g., 512)
        device: 'cuda' or 'cpu'
        threshold: Binary threshold
        log_callback: Optional logging function
        progress_callback: Optional progress function(pct, msg)
    
    Returns:
        Full-resolution binary mask (H, W) uint8
    """
    import rasterio
    
    log = log_callback or (lambda x: None)
    progress = progress_callback or (lambda p, m: None)
    
    model.eval()
    
    with rasterio.open(raster_path) as src:
        full_h = src.height
        full_w = src.width
    
    # Output mask
    full_mask = np.zeros((full_h, full_w), dtype=np.uint8)
    
    # Calculate padding needed
    pad = (tile_size - crop_size) // 2  # e.g., (1024-512)//2 = 256
    
    # Step = crop_size (non-overlapping center crops tile the full image)
    step = crop_size
    
    # Calculate grid
    n_rows = (full_h + step - 1) // step
    n_cols = (full_w + step - 1) // step
    total_tiles = n_rows * n_cols
    
    log(f"Center-crop stitching: {n_rows}x{n_cols} = {total_tiles} tiles")
    log(f"Tile size: {tile_size}, crop size: {crop_size}, padding: {pad}")
    
    tile_idx = 0
    with rasterio.open(raster_path) as src:
        for row_i in range(n_rows):
            for col_i in range(n_cols):
                # Center crop position in full image
                crop_y = row_i * step
                crop_x = col_i * step
                
                # Tile read position (with padding around center crop)
                read_y = crop_y - pad
                read_x = crop_x - pad
                
                # Read tile (handle edges with zero-padding)
                tile_rgb = np.zeros((tile_size, tile_size, 3), dtype=np.uint8)
                
                # Calculate valid read region
                src_y0 = max(0, read_y)
                src_x0 = max(0, read_x)
                src_y1 = min(full_h, read_y + tile_size)
                src_x1 = min(full_w, read_x + tile_size)
                
                # Destination in tile array
                dst_y0 = src_y0 - read_y
                dst_x0 = src_x0 - read_x
                dst_y1 = dst_y0 + (src_y1 - src_y0)
                dst_x1 = dst_x0 + (src_x1 - src_x0)
                
                # Read from raster
                from rasterio.windows import Window
                window = Window(src_x0, src_y0, src_x1 - src_x0, src_y1 - src_y0)
                data = src.read([1, 2, 3], window=window)  # (3, H, W)
                data = np.clip(data, 0, 255).astype(np.uint8)
                tile_rgb[dst_y0:dst_y1, dst_x0:dst_x1, :] = np.transpose(data, (1, 2, 0))
                
                # Predict
                mask_tile = predict_tile(model, tile_rgb, device=device, threshold=threshold)
                
                # Extract center crop from prediction
                center_crop = mask_tile[pad:pad + crop_size, pad:pad + crop_size]
                
                # Place in full mask (handle edge cases)
                out_y1 = min(crop_y + crop_size, full_h)
                out_x1 = min(crop_x + crop_size, full_w)
                crop_h = out_y1 - crop_y
                crop_w = out_x1 - crop_x
                
                full_mask[crop_y:out_y1, crop_x:out_x1] = center_crop[:crop_h, :crop_w]
                
                tile_idx += 1
                if tile_idx % 10 == 0 or tile_idx == total_tiles:
                    pct = int(100 * tile_idx / total_tiles)
                    progress(pct, f"Prediksi tile {tile_idx}/{total_tiles}")
    
    log(f"Prediksi selesai. Road pixels: {(full_mask > 0).sum():,}")
    return full_mask

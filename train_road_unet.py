"""
train_road_unet.py
Training script for U-Net road segmentation model.

Usage:
    python train_road_unet.py --dataset dataset_road --epochs 30 --batch_size 4

Or called programmatically from the GUI via app.py.
"""

import os
import sys
import time
import argparse
import numpy as np
from pathlib import Path
from typing import Callable, Optional

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)


def train_road_unet(
    dataset_dir: str,
    epochs: int = 30,
    batch_size: int = 4,
    lr: float = 1e-4,
    encoder_name: str = "resnet34",
    output_model_path: str = None,
    log_callback: Optional[Callable[[str], None]] = None,
    progress_callback: Optional[Callable[[float, str], None]] = None,
) -> bool:
    """
    Train U-Net road segmentation model.
    
    Args:
        dataset_dir: Path to dataset (with train/val subdirs)
        epochs: Number of training epochs
        batch_size: Batch size (4 fits in 4GB VRAM with 512x512 tiles)
        lr: Learning rate
        encoder_name: Encoder backbone (resnet34 recommended for 4GB VRAM)
        output_model_path: Where to save the trained model
        log_callback: Logging function
        progress_callback: Progress function(percent, message)
    
    Returns:
        True if training succeeded
    """
    import torch
    from torch.utils.data import DataLoader
    from torch.optim import AdamW
    from torch.optim.lr_scheduler import CosineAnnealingLR

    from core.objects.road_unet_model import create_road_unet, RoadSegmentationLoss
    from core.objects.road_dataset_generator import RoadSegmentationDataset

    log = log_callback or print
    progress = progress_callback or (lambda p, m: None)

    # ── Setup ────────────────────────────────────────────────────────────────
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log(f"Device: {device}")
    if device == "cuda":
        gpu_name = torch.cuda.get_device_name(0)
        vram_mb = torch.cuda.get_device_properties(0).total_memory / 1024**2
        log(f"GPU: {gpu_name} ({vram_mb:.0f} MB VRAM)")

    if output_model_path is None:
        output_model_path = os.path.join(ROOT, "models", "road_unet.pth")
    os.makedirs(os.path.dirname(output_model_path), exist_ok=True)

    # ── Dataset ──────────────────────────────────────────────────────────────
    progress(5, "Memuat dataset...")
    
    train_dataset = RoadSegmentationDataset(dataset_dir, split="train", augment=True)
    val_dataset = RoadSegmentationDataset(dataset_dir, split="val", augment=False)

    log(f"Train samples: {len(train_dataset)}")
    log(f"Val samples: {len(val_dataset)}")

    if len(train_dataset) == 0:
        log("❌ Dataset kosong! Pastikan dataset sudah di-generate.")
        return False

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=2, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=2, pin_memory=True,
    )

    # ── Model ────────────────────────────────────────────────────────────────
    progress(10, "Membuat model U-Net...")
    
    model = create_road_unet(encoder_name=encoder_name)
    model = model.to(device)

    param_count = sum(p.numel() for p in model.parameters()) / 1e6
    log(f"Model: U-Net + {encoder_name} ({param_count:.1f}M parameters)")

    # ── Loss & Optimizer ─────────────────────────────────────────────────────
    criterion = RoadSegmentationLoss(
        bce_weight=0.5,
        dice_weight=0.3,
        cldice_weight=0.2,
        cldice_iters=8,
    )
    
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr * 0.01)

    # ── Training Loop ────────────────────────────────────────────────────────
    log(f"\n{'='*60}")
    log(f"TRAINING: {epochs} epochs, batch_size={batch_size}, lr={lr}")
    log(f"Loss: BCE(0.5) + Dice(0.3) + clDice(0.2)")
    log(f"{'='*60}\n")

    best_val_loss = float("inf")
    best_epoch = 0
    patience = 10
    patience_counter = 0

    for epoch in range(1, epochs + 1):
        epoch_start = time.time()
        
        # ── Train ────────────────────────────────────────────────────────
        model.train()
        train_losses = []
        train_metrics = {"bce": [], "dice": [], "cldice": []}

        for batch_idx, (images, masks) in enumerate(train_loader):
            images = images.to(device)
            masks = masks.to(device)

            optimizer.zero_grad()
            logits = model(images)
            loss, metrics = criterion(logits, masks)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_losses.append(loss.item())
            for k in train_metrics:
                train_metrics[k].append(metrics[k])

        scheduler.step()

        # ── Validate ─────────────────────────────────────────────────────
        model.eval()
        val_losses = []
        val_iou_scores = []

        with torch.no_grad():
            for images, masks in val_loader:
                images = images.to(device)
                masks = masks.to(device)

                logits = model(images)
                loss, _ = criterion(logits, masks)
                val_losses.append(loss.item())

                # Calculate IoU
                pred = (torch.sigmoid(logits) > 0.5).float()
                intersection = (pred * masks).sum(dim=(2, 3))
                union = pred.sum(dim=(2, 3)) + masks.sum(dim=(2, 3)) - intersection
                iou = (intersection + 1e-6) / (union + 1e-6)
                val_iou_scores.extend(iou.mean(dim=1).cpu().numpy().tolist())

        # ── Epoch Summary ────────────────────────────────────────────────
        train_loss = np.mean(train_losses)
        val_loss = np.mean(val_losses) if val_losses else 0
        val_iou = np.mean(val_iou_scores) if val_iou_scores else 0
        elapsed = time.time() - epoch_start
        current_lr = scheduler.get_last_lr()[0]

        log(
            f"Epoch {epoch:3d}/{epochs} | "
            f"Train: {train_loss:.4f} | "
            f"Val: {val_loss:.4f} | "
            f"IoU: {val_iou:.4f} | "
            f"LR: {current_lr:.2e} | "
            f"{elapsed:.1f}s"
        )

        # Progress
        pct = 10 + int(85 * epoch / epochs)
        progress(pct, f"Epoch {epoch}/{epochs} — Val IoU: {val_iou:.4f}")

        # ── Save Best Model ──────────────────────────────────────────────
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            patience_counter = 0
            
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
                "val_iou": val_iou,
                "encoder_name": encoder_name,
                "chip_size": 512,
            }, output_model_path)
            log(f"  💾 Model terbaik disimpan (val_loss={val_loss:.4f}, IoU={val_iou:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                log(f"\n⚠ Early stopping: val_loss tidak membaik selama {patience} epoch.")
                break

    # ── Final Summary ────────────────────────────────────────────────────────
    log(f"\n{'='*60}")
    log(f"TRAINING SELESAI")
    log(f"  Best epoch: {best_epoch}")
    log(f"  Best val_loss: {best_val_loss:.4f}")
    log(f"  Model disimpan: {output_model_path}")
    log(f"{'='*60}")

    progress(100, f"Training selesai! Best IoU: {val_iou:.4f}")
    return True


# ══════════════════════════════════════════════════════════════════════════════
# CLI ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train U-Net road segmentation model")
    parser.add_argument("--dataset", type=str, default="dataset_road",
                        help="Path to dataset directory")
    parser.add_argument("--epochs", type=int, default=30,
                        help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Batch size (4 for 4GB VRAM)")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Learning rate")
    parser.add_argument("--encoder", type=str, default="resnet34",
                        help="Encoder backbone (resnet34, resnet50, efficientnet-b0)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output model path")
    
    args = parser.parse_args()
    
    success = train_road_unet(
        dataset_dir=args.dataset,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        encoder_name=args.encoder,
        output_model_path=args.output,
    )
    
    sys.exit(0 if success else 1)

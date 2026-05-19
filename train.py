"""
train.py
Fine-tunes the Segment Anything (SAM) model decoder on a custom building dataset.
Designed to run blazingly fast on CUDA using your NVIDIA RTX 3050.
"""

import os
import glob
import cv2
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from typing import Callable, Optional

# ──────────────────────────────────────────────────────────────────────────────
# 1. PyTorch Dataset for SAM Fine-Tuning
# ──────────────────────────────────────────────────────────────────────────────

class SAMBuildingDataset(Dataset):
    """
    Dataset that loads 512x512 image chips and their binary masks.
    Automatically extracts random building pixels as positive point prompts
    to train SAM's decoder exactly how it is used during inference.
    """
    def __init__(self, dataset_dir: str):
        self.image_paths = sorted(glob.glob(os.path.join(dataset_dir, "images", "*.png")))
        self.mask_paths = sorted(glob.glob(os.path.join(dataset_dir, "masks", "*.png")))
        
    def __len__(self):
        return len(self.image_paths)
        
    def __getitem__(self, idx):
        # Load image (RGB) and mask (Grayscale)
        image = cv2.imread(self.image_paths[idx])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        mask = cv2.imread(self.mask_paths[idx], cv2.IMREAD_GRAYSCALE)
        
        # 1. Image preprocessing: Normalization as expected by SAM
        # Resize to 1024x1024 (SAM standard input size)
        img_1024 = cv2.resize(image, (1024, 1024), interpolation=cv2.INTER_LINEAR)
        mask_256 = cv2.resize(mask, (256, 256), interpolation=cv2.INTER_NEAREST)  # SAM low-res mask size
        
        # Convert image to float32, normalize, transpose to (C, H, W)
        mean = np.array([123.675, 116.28, 103.53], dtype=np.float32)
        std = np.array([58.395, 57.12, 57.375], dtype=np.float32)
        img_norm = (img_1024.astype(np.float32) - mean) / std
        img_tensor = torch.as_tensor(img_norm, dtype=torch.float32).permute(2, 0, 1)
        
        # Binary mask tensor (1 = building, 0 = background)
        mask_tensor = torch.as_tensor(mask_256 > 127).float().unsqueeze(0)
        
        # 2. Extract Point Prompt: Randomly pick a building centroid coordinate
        building_pixels = np.argwhere(mask == 255)
        if len(building_pixels) > 0:
            # Pick a random building pixel
            pt = random.choice(building_pixels)
            # Scale coordinates to SAM 1024x1024 space
            h_scale, w_scale = 1024.0 / mask.shape[0], 1024.0 / mask.shape[1]
            point_coords = np.array([[pt[1] * w_scale, pt[0] * h_scale]], dtype=np.float32)
            point_labels = np.array([1], dtype=np.int32)
        else:
            # Fallback if no building pixels (should be rare due to pre-filtering)
            point_coords = np.array([[512.0, 512.0]], dtype=np.float32)
            point_labels = np.array([0], dtype=np.int32)
            
        return img_tensor, mask_tensor, torch.as_tensor(point_coords), torch.as_tensor(point_labels)

# ──────────────────────────────────────────────────────────────────────────────
# 2. Hybrid BCE + Dice Loss Function
# ──────────────────────────────────────────────────────────────────────────────

class BCEDiceLoss(nn.Module):
    """
    Dice Loss + BCE Loss. Highly robust for segmenting buildings
    with sharp rectangular boundaries and preventing organic bleeding.
    """
    def __init__(self, weight_bce=1.0, weight_dice=1.0):
        super().__init__()
        self.weight_bce = weight_bce
        self.weight_dice = weight_dice
        
    def forward(self, pred, target):
        # BCE Loss
        bce = F.binary_cross_entropy_with_logits(pred, target)
        
        # Dice Loss
        pred_sig = torch.sigmoid(pred)
        smooth = 1e-5
        intersection = (pred_sig * target).sum()
        dice = 1.0 - (2.0 * intersection + smooth) / (pred_sig.sum() + target.sum() + smooth)
        
        return self.weight_bce * bce + self.weight_dice * dice

# ──────────────────────────────────────────────────────────────────────────────
# 3. Main Training Execution Function
# ──────────────────────────────────────────────────────────────────────────────

def train_model(
    dataset_dir: str,
    epochs: int = 15,
    batch_size: int = 2,
    lr: float = 1e-4,
    log_callback: Optional[Callable[[str], None]] = None,
    progress_callback: Optional[Callable[[float, str], None]] = None,
) -> bool:
    """
    Runs the full PyTorch fine-tuning loop on SAM's Mask Decoder.
    """
    log = log_callback or print
    progress = progress_callback or (lambda p, m: None)

    try:
        log("🧠 Menginisialisasi training model SAM kustom...")
        progress(72, "Menginisialisasi modul PyTorch...")

        # 1. Check GPU / CUDA
        device = "cuda" if torch.cuda.is_available() else "cpu"
        log(f"💻 Perangkat Latihan: {device.upper()}")
        if device == "cuda":
            log(f"🚀 GPU Terdeteksi: {torch.cuda.get_device_name(0)}")

        # 2. Load Dataset
        dataset = SAMBuildingDataset(dataset_dir)
        if len(dataset) == 0:
            log("❌ Error: Tidak ada potongan ubin training dalam dataset!")
            return False
            
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
        log(f"📚 Total ubin training: {len(dataset)} | Batch size: {batch_size}")

        # 3. Load SAM Model Checkpoint
        # We use vit_h (SAM-H) or vit_b (SAM-B) based on what is available in the models folder
        # Check existing checkpoints
        models_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
        os.makedirs(models_dir, exist_ok=True)
        
        checkpoint_path = None
        model_type = "vit_b"
        
        # 1. Look for custom pre-trained checkpoint first to continue training (fine-tune)
        custom_chpt = os.path.join(models_dir, "sam_bangunan_lokal.pth")
        if os.path.exists(custom_chpt):
            checkpoint_path = custom_chpt
            # Determine base type from standard downloaded weights
            model_type = "vit_b"
            if glob.glob(os.path.join(models_dir, "sam_vit_h*.pth")):
                model_type = "vit_h"
            log(f"🔄 Melanjutkan training (fine-tuning) dari model kustom SAM yang sudah ada: {os.path.basename(checkpoint_path)}")
        else:
            # 2. Look for downloaded base weights in models/
            chpts = glob.glob(os.path.join(models_dir, "sam_vit_b*.pth"))
            if chpts:
                checkpoint_path = chpts[0]
                model_type = "vit_b"
            else:
                chpts = glob.glob(os.path.join(models_dir, "sam_vit_h*.pth"))
                if chpts:
                    checkpoint_path = chpts[0]
                    model_type = "vit_h"
                else:
                    # Standard download paths
                    chpts = glob.glob(os.path.join(models_dir, "*.pth"))
                    if chpts:
                        checkpoint_path = chpts[0]
                        model_type = "vit_b" if "vit_b" in checkpoint_path else "vit_h"

        if not checkpoint_path or not os.path.exists(checkpoint_path):
            log("❌ Error: File bobot dasar SAM (.pth) tidak ditemukan di folder 'models'!")
            log("💡 Solusi: Silakan unduh model SAM-B atau SAM-H terlebih dahulu melalui GUI.")
            return False

        log(f"📦 Mengunduh bobot dasar SAM dari: {os.path.basename(checkpoint_path)} ({model_type})")
        from segment_anything import sam_model_registry
        sam_model = sam_model_registry[model_type](checkpoint=checkpoint_path)
        sam_model.to(device)

        # 4. Freeze Image Encoder & Prompt Encoder, Only Fine-Tune Mask Decoder
        # This prevents CUDA Out-Of-Memory (OOM) and ensures extremely fast training!
        for param in sam_model.image_encoder.parameters():
            param.requires_grad = False
        for param in sam_model.prompt_encoder.parameters():
            param.requires_grad = False
            
        # Keep mask decoder trainable
        for param in sam_model.mask_decoder.parameters():
            param.requires_grad = True

        optimizer = torch.optim.Adam(sam_model.mask_decoder.parameters(), lr=lr, weight_decay=1e-4)
        criterion = BCEDiceLoss()

        log("🔥 Memulai proses training pada GPU...")
        sam_model.train()

        total_batches = len(dataloader)
        
        for epoch in range(epochs):
            epoch_loss = 0.0
            for batch_idx, (imgs, masks, pt_coords, pt_labels) in enumerate(dataloader):
                imgs = imgs.to(device)
                masks = masks.to(device)
                pt_coords = pt_coords.to(device)
                pt_labels = pt_labels.to(device)

                batch_loss = torch.tensor(0.0, device=device, requires_grad=False)
                optimizer.zero_grad()

                # SAM's mask_decoder must receive ONE image at a time.
                # Internally it uses repeat_interleave which would double the batch dim
                # causing a size mismatch if we pass the full batch at once.
                current_batch_size = imgs.shape[0]
                for i in range(current_batch_size):
                    img_i        = imgs[i].unsqueeze(0)         # [1, C, H, W]
                    mask_i       = masks[i].unsqueeze(0)        # [1, 1, 256, 256]
                    coords_i     = pt_coords[i].unsqueeze(0)    # [1, 1, 2]
                    labels_i     = pt_labels[i].unsqueeze(0)    # [1, 1]

                    # Step 1: Forward Image Encoder (Freeze)
                    with torch.no_grad():
                        image_embedding_i = sam_model.image_encoder(img_i)  # [1, 256, 64, 64]

                    # Step 2: Forward Prompt Encoder (Freeze)
                    with torch.no_grad():
                        sparse_emb_i, dense_emb_i = sam_model.prompt_encoder(
                            points=(coords_i, labels_i),
                            boxes=None,
                            masks=None,
                        )

                    # Step 3: Forward Mask Decoder (Trainable) — safe with batch=1
                    low_res_mask_i, _ = sam_model.mask_decoder(
                        image_embeddings=image_embedding_i,
                        image_pe=sam_model.prompt_encoder.get_dense_pe(),
                        sparse_prompt_embeddings=sparse_emb_i,
                        dense_prompt_embeddings=dense_emb_i,
                        multimask_output=False,
                    )

                    # Accumulate loss across all samples in the batch
                    batch_loss = batch_loss + criterion(low_res_mask_i, mask_i)

                # Average loss over the batch and backpropagate once
                batch_loss = batch_loss / current_batch_size
                batch_loss.backward()
                optimizer.step()

                epoch_loss += batch_loss.item()

                # Dispatch progress status to GUI
                pct = 75.0 + ((epoch + (batch_idx + 1) / total_batches) / epochs) * 20.0
                if batch_idx % 2 == 0 or batch_idx == total_batches - 1:
                    progress(
                        pct,
                        f"Epoch {epoch+1}/{epochs} | Batch {batch_idx+1}/{total_batches} | Loss: {batch_loss.item():.4f}"
                    )
            
            avg_loss = epoch_loss / total_batches
            log(f"📈 Epoch {epoch+1:02d}/{epochs:02d} | Rata-rata Loss: {avg_loss:.4f}")

        # 5. Export Fine-Tuned Model Weights
        output_model_path = os.path.join(models_dir, "sam_bangunan_lokal.pth")
        log(f"💾 Menyimpan model kustom baru ke: {os.path.basename(output_model_path)}")
        torch.save(sam_model.state_dict(), output_model_path)
        
        # Free CUDA cache
        if device == "cuda":
            torch.cuda.empty_cache()
            
        log("🎉 Latihan selesai dengan sukses! Model kustom Anda siap digunakan.")
        return True
        
    except Exception as e:
        log(f"❌ Gagal selama proses training: {e}")
        import traceback
        log(traceback.format_exc())
        return False

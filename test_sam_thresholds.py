import os
import cv2
import torch
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

def test_thresholds():
    project_root = os.path.abspath(".")
    tile_path = os.path.join(project_root, "temp", "tiles", "tile_0004_896_896.tif")
    models_dir = os.path.join(project_root, "models")
    
    custom_path = os.path.join(models_dir, "sam_vit_b_01ec64.pth")
    original_path = os.path.join(models_dir, "sam_vit_b_01ec64.pth.asli")

    print("=== SAM THRESHOLD TESTING ===")
    
    if not os.path.exists(tile_path):
        print(f"[ERROR] Tile file not found at: {tile_path}")
        return
        
    image = cv2.imread(tile_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    print(f"Loaded image: {os.path.basename(tile_path)} with shape {image.shape}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Helper function to test a model
    def test_model(model_path, label):
        print(f"\n--- Testing {label} ---")
        try:
            model = sam_model_registry["vit_b"](checkpoint=model_path)
            model.to(device=device)
        except Exception as e:
            print(f"  [ERROR] Failed to load model: {e}")
            return

        # 1. Test Default strict thresholds
        print("  Evaluating with DEFAULT strict thresholds (iou=0.82, stability=0.88)...")
        generator_default = SamAutomaticMaskGenerator(
            model,
            points_per_side=32,
            pred_iou_thresh=0.82,
            stability_score_thresh=0.88,
            crop_n_layers=0,
            min_mask_region_area=100
        )
        try:
            masks_def = generator_default.generate(image)
            print(f"    -> Detected: {len(masks_def)} masks")
            if masks_def:
                mean_iou = sum(m["predicted_iou"] for m in masks_def) / len(masks_def)
                mean_stab = sum(m["stability_score"] for m in masks_def) / len(masks_def)
                print(f"    -> Mean IoU: {mean_iou:.4f}, Mean Stability: {mean_stab:.4f}")
        except Exception as e:
            print(f"    [ERROR] Generation failed: {e}")

        # 2. Test Relaxed thresholds
        print("  Evaluating with RELAXED thresholds (iou=0.50, stability=0.50)...")
        generator_relaxed = SamAutomaticMaskGenerator(
            model,
            points_per_side=32,
            pred_iou_thresh=0.50,
            stability_score_thresh=0.50,
            crop_n_layers=0,
            min_mask_region_area=100
        )
        try:
            masks_rel = generator_relaxed.generate(image)
            print(f"    -> Detected: {len(masks_rel)} masks")
            if masks_rel:
                mean_iou = sum(m["predicted_iou"] for m in masks_rel) / len(masks_rel)
                mean_stab = sum(m["stability_score"] for m in masks_rel) / len(masks_rel)
                print(f"    -> Mean IoU: {mean_iou:.4f}, Mean Stability: {mean_stab:.4f}")
        except Exception as e:
            print(f"    [ERROR] Generation failed: {e}")

    # Run the tests
    test_model(original_path, "ORIGINAL PRE-TRAINED MODEL")
    test_model(custom_path, "YOUR CUSTOM MODEL")

if __name__ == "__main__":
    test_thresholds()

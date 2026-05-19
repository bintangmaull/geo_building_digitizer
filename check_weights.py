import os
import torch

def inspect_weights():
    models_dir = os.path.abspath("models")
    custom_path = os.path.join(models_dir, "sam_vit_b_01ec64.pth")
    original_path = os.path.join(models_dir, "sam_vit_b_01ec64.pth.asli")

    print("=== INSPECTING MODEL WEIGHTS ===")
    
    if not os.path.exists(custom_path):
        print(f"[ERROR] Custom model not found at: {custom_path}")
        return
        
    if not os.path.exists(original_path):
        print(f"[ERROR] Original model not found at: {original_path}")
        return

    print("Loading models...")
    try:
        custom_state = torch.load(custom_path, map_location="cpu")
        original_state = torch.load(original_path, map_location="cpu")
    except Exception as e:
        print(f"[ERROR] Loading models failed: {e}")
        return

    # Check for NaN / Inf in custom state dict
    nan_found = False
    inf_found = False
    
    print("\nScanning custom model weights for anomalies (NaN, Inf)...")
    for key, val in custom_state.items():
        if isinstance(val, torch.Tensor):
            if torch.isnan(val).any():
                print(f"  [ANOMALY] NaN detected in key: {key}")
                nan_found = True
            if torch.isinf(val).any():
                print(f"  [ANOMALY] Inf detected in key: {key}")
                inf_found = True
                
    if not nan_found and not inf_found:
        print("  [SUCCESS] No NaN or Inf found in custom model weights. The tensor values are mathematically stable!")

    # Compare key weights in Mask Decoder to see if they actually changed
    print("\nComparing Mask Decoder weights between Original and Custom...")
    decoder_keys = [k for k in custom_state.keys() if "mask_decoder" in k and "weight" in k]
    
    if not decoder_keys:
        print("  [WARNING] No 'mask_decoder' keys found in state dict!")
        return

    changed_keys = 0
    unchanged_keys = 0
    
    for key in decoder_keys:
        if key in original_state:
            custom_val = custom_state[key]
            original_val = original_state[key]
            
            # Check if they are identical
            if torch.equal(custom_val, original_val):
                unchanged_keys += 1
            else:
                changed_keys += 1
                # Print some difference metrics for the first few keys
                if changed_keys <= 3:
                    diff = torch.abs(custom_val - original_val).mean().item()
                    print(f"  [CHANGED] Key '{key}' changed. Mean absolute difference: {diff:.6f}")
        else:
            print(f"  [ADDED] Key '{key}' only exists in custom model.")

    print(f"\nSummary of Mask Decoder Weights:")
    print(f"  - Keys changed (Fine-tuned): {changed_keys}")
    print(f"  - Keys unchanged (Identical to original): {unchanged_keys}")
    
    if changed_keys == 0:
        print("\n[WARNING] The custom model is 100% IDENTICAL to the original model! This means training was either never completed, never saved, or the file was simply copied without training.")
    else:
        print("\n[INFO] The custom model indeed contains active, modified fine-tuned weights!")

if __name__ == "__main__":
    inspect_weights()

"""
downsample_tiff.py
Memory-efficient downsampling utility for large GeoTIFF files.
Specifically designed to resample high-resolution drone rasters (e.g., 5cm GSD)
to optimized resolutions (e.g., 15cm GSD) to make deep learning workflows blazingly fast.
"""

import os
import sys
import argparse
import rasterio
from rasterio.enums import Resampling
from rasterio.windows import Window

def downsample_raster(input_path: str, output_path: str, target_gsd: float = 0.15):
    """
    Resamples a GeoTIFF to a target GSD using windowed block-by-block processing
    to ensure extremely low memory footprint even on massive 30GB+ rasters.
    """
    if not os.path.exists(input_path):
        print(f"❌ Error: Input file '{input_path}' not found!")
        return False

    print(f"📖 Opening source TIFF: {os.path.basename(input_path)}...")
    
    with rasterio.open(input_path) as src:
        # Determine source GSD
        res_x = abs(src.transform.a)
        res_y = abs(src.transform.e)
        source_gsd = (res_x + res_y) / 2.0
        
        print(f"📐 Original Size: {src.width:,} x {src.height:,} pixels")
        print(f"📐 Original GSD: {source_gsd*100:.2f} cm/pixel")
        
        if source_gsd >= target_gsd:
            print(f"⚠️ Warning: Target GSD ({target_gsd*100:.1f}cm) is larger than or equal to source GSD ({source_gsd*100:.1f}cm). No downsampling needed!")
            # Ask to proceed anyway or abort
            scale_factor = 1.0
        else:
            scale_factor = source_gsd / target_gsd
            
        # Compute new dimensions
        new_width = int(src.width * scale_factor)
        new_height = int(src.height * scale_factor)
        
        if new_width == 0 or new_height == 0:
            print("❌ Error: Target dimensions computed to 0. Target GSD might be too coarse.")
            return False
            
        print(f"🔄 Scale Factor: {scale_factor:.4f} (Shrinking to {scale_factor*100:.1f}% size)")
        print(f"📐 Target Size: {new_width:,} x {new_height:,} pixels")
        print(f"📐 Target GSD: {target_gsd*100:.1f} cm/pixel")

        # Update metadata for output
        new_transform = src.transform * src.transform.scale(
            (src.width / new_width),
            (src.height / new_height)
        )
        
        meta = src.meta.copy()
        meta.update({
            "driver": "GTiff",
            "height": new_height,
            "width": new_width,
            "transform": new_transform,
            # Use LZW compression with high-efficiency predictors for fast reading/writing and small file sizes
            "compress": "lzw",
            "predictor": 2, 
            "tiled": True,
            "blockxsize": 512,
            "blockysize": 512
        })

        print(f"💾 Saving compressed 15cm GeoTIFF to: {output_path}...")

        # Process block by block to keep memory usage under 100MB
        with rasterio.open(output_path, "w", **meta) as dst:
            # We process using destination blocks (512x512)
            block_w, block_h = 512, 512
            total_blocks_x = (new_width + block_w - 1) // block_w
            total_blocks_y = (new_height + block_h - 1) // block_h
            total_blocks = total_blocks_x * total_blocks_y
            
            block_count = 0
            
            for y_idx in range(total_blocks_y):
                dst_y = y_idx * block_h
                h_eff = min(block_h, new_height - dst_y)
                
                for x_idx in range(total_blocks_x):
                    dst_x = x_idx * block_w
                    w_eff = min(block_w, new_width - dst_x)
                    
                    # Define destination window
                    dst_window = Window(dst_x, dst_y, w_eff, h_eff)
                    
                    # Map back to source window coordinates
                    src_x = int(dst_x / scale_factor)
                    src_y = int(dst_y / scale_factor)
                    src_w = min(int(w_eff / scale_factor), src.width - src_x)
                    src_h = min(int(h_eff / scale_factor), src.height - src_y)
                    
                    src_window = Window(src_x, src_y, src_w, src_h)
                    
                    if src_w > 0 and src_h > 0:
                        # Read from source and resample on-the-fly to the exact destination block shape
                        data = src.read(
                            out_shape=(src.count, h_eff, w_eff),
                            window=src_window,
                            resampling=Resampling.bilinear
                        )
                        # Write directly to target
                        dst.write(data, window=dst_window)
                        
                    block_count += 1
                    if block_count % max(1, total_blocks // 10) == 0 or block_count == total_blocks:
                        pct = (block_count / total_blocks) * 100
                        print(f"   ⚡ Downsampling Progress: {pct:.1f}% completed ({block_count}/{total_blocks} blocks)")
                        
    print(f"🎉 Success! Optimized resampled image saved at: {output_path}")
    print(f"💡 File size is now significantly smaller and ready for lightning-fast training and digitization!")
    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Downsample high-resolution GeoTIFF drone rasters to optimized resolutions.")
    parser.add_argument("-i", "--input", required=True, help="Path to input high-resolution GeoTIFF file (.tif / .tiff)")
    parser.add_argument("-o", "--output", required=True, help="Path to save resampled output GeoTIFF file")
    parser.add_argument("-g", "--gsd", type=float, default=0.15, help="Target GSD in meters per pixel (default: 0.15 for 15cm)")
    
    args = parser.parse_args()
    downsample_raster(args.input, args.output, args.gsd)

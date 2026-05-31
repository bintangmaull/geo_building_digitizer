import os
import glob
from pathlib import Path

# ---------- KONFIGURASI ----------
# Masukkan path file raster (citra) asli yang Anda gunakan sebelumnya di sini:
RASTER_PATH = r"E:\Training SAM\TUBAN\Sawir\sawir15.tif"  # <-- OTOMATIS DITEMUKAN

# Path output shapefile yang diinginkan (pastikan QGIS ditutup!)
OUTPUT_SHP = r"E:\Project SAM\output\sawir15_bangunan.shp"
# ---------------------------------

def recover():
    print("Mempersiapkan pemulihan dari cache...")
    project_root = os.path.dirname(os.path.abspath(__file__))
    masks_dir = os.path.join(project_root, "temp", "masks")
    
    if not os.path.exists(masks_dir):
        print(f"Error: Folder cache tidak ditemukan: {masks_dir}")
        return
        
    mask_files = glob.glob(os.path.join(masks_dir, "mask_*.tif"))
    if not mask_files:
        print("Error: Tidak ada file mask di dalam folder cache.")
        return
        
    print(f"Ditemukan {len(mask_files)} tile cache. Memulai post-processing...")
    
    if not os.path.exists(RASTER_PATH):
        print(f"Peringatan: File raster '{RASTER_PATH}' tidak ditemukan!")
        print("Harap edit file 'recover_from_cache.py' dan ubah RASTER_PATH ke file .tif/.ecw yang benar.")
        return

    # Import pipeline dari core
    from core.postprocess import run_postprocess_pipeline
    from core.exporter import export_to_shapefile, get_output_summary
    
    try:
        print("\nMenjalankan Post-Processing (Ini memakan waktu sekitar 9 menit)...")
        # Parameter standar (bisa disesuaikan jika perlu)
        result_gdf = run_postprocess_pipeline(
            mask_paths_and_metas=mask_files,
            original_raster_path=RASTER_PATH,
            min_area_m2=20.0,
            max_area_m2=5000.0,
            max_aspect_ratio=8.0,
            enable_shadow_filter=True,       # Sesuaikan dengan pengaturan Anda
            enable_vegetation_filter=True,   # Sesuaikan dengan pengaturan Anda
            enable_regularization=True,
            simplify_tolerance=15.0,         # Sesuai dengan log Anda: toleransi=15.0
            log_callback=lambda msg: print(f"  [Log] {msg}")
        )
        
        if len(result_gdf) == 0:
            print("Tidak ada bangunan tersisa setelah post-processing.")
            return
            
        print(f"\nPost-processing selesai! {len(result_gdf)} bangunan terdeteksi.")
        print(f"Mengekspor ke: {OUTPUT_SHP}")
        
        # Ekspor ke shapefile
        export_to_shapefile(
            result_gdf,
            OUTPUT_SHP,
            source_raster_path=RASTER_PATH,
            log_callback=lambda msg: print(f"  [Export] {msg}")
        )
        
        summary = get_output_summary(OUTPUT_SHP)
        print("\n==================================================")
        print("✅ PEMULIHAN SUKSES!")
        print(f"   Bangunan diekspor: {summary.get('count', len(result_gdf))}")
        print(f"   Output tersimpan di: {OUTPUT_SHP}")
        print("==================================================")
        
    except Exception as e:
        import traceback
        print(f"\n❌ ERROR: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    recover()

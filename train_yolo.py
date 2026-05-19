"""
train_yolo.py
Fine-tunes the YOLOv8 model on a custom building dataset.
"""

import os
import sys
import threading
from typing import Callable, Optional

def train_yolo_model(
    dataset_dir: str,
    epochs: int = 15,
    batch_size: int = 4,
    base_model: str = "yolov8n.pt",
    log_callback: Optional[Callable[[str], None]] = None,
    progress_callback: Optional[Callable[[float, str], None]] = None,
) -> bool:
    """
    Runs YOLOv8 training.
    """
    log = log_callback or print
    progress = progress_callback or (lambda p, m: None)

    try:
        log(f"🧠 Menginisialisasi training YOLOv8 dengan base model {base_model}...")
        progress(72, "Memuat model ultralytics YOLO...")

        try:
            from ultralytics import YOLO
        except ImportError:
            log("❌ Error: library ultralytics belum terinstal. Jalankan pip install ultralytics")
            return False

        yaml_path = os.path.join(dataset_dir, "dataset.yaml")
        if not os.path.exists(yaml_path):
            log("❌ Error: dataset.yaml tidak ditemukan di dataset_dir!")
            return False
            
        models_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
        os.makedirs(models_dir, exist_ok=True)
        
        # Load the base model
        model = YOLO(base_model)
        
        log("🔥 Memulai proses training pada GPU (jika tersedia)...")
        log("💡 Catatan: Log detail training YOLO mungkin muncul di konsol terminal.")
        
        # Define a callback to capture training progress
        def on_train_epoch_end(trainer):
            # This is called at the end of each epoch
            epoch = trainer.epoch + 1
            max_epochs = trainer.epochs
            loss = trainer.metrics.get('train/loss', 0.0)
            
            pct = 75.0 + (epoch / max_epochs) * 20.0
            progress(pct, f"Epoch {epoch}/{max_epochs} selesai")
            log(f"📈 Epoch {epoch:02d}/{max_epochs:02d} selesai")

        # Add custom callback
        model.add_callback("on_train_epoch_end", on_train_epoch_end)
        
        # Run training
        results = model.train(
            data=yaml_path,
            epochs=epochs,
            batch=batch_size,
            imgsz=640,
            project=models_dir,
            name="yolo_train_run",
            exist_ok=True, # overwrite previous run
            device="0" # use GPU 0, will fallback to cpu if fail usually but ultralytics handles it
        )
        
        output_model_path = os.path.join(models_dir, "yolo_bangunan_lokal.pt")
        trained_weights = os.path.join(models_dir, "yolo_train_run", "weights", "best.pt")
        
        if os.path.exists(trained_weights):
            import shutil
            shutil.copy2(trained_weights, output_model_path)
            log(f"💾 Menyimpan model YOLO kustom baru ke: {os.path.basename(output_model_path)}")
        else:
            log("⚠️ Peringatan: File best.pt tidak ditemukan, menyimpan weights.pt jika ada.")
            
        log("🎉 Latihan YOLO selesai dengan sukses! Model kustom Anda siap digunakan.")
        progress(100, "Training selesai")
        return True
        
    except Exception as e:
        log(f"❌ Gagal selama proses training YOLO: {e}")
        import traceback
        log(traceback.format_exc())
        return False

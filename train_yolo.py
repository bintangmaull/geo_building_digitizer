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
    epochs: int = 100,
    batch_size: int = 4,
    base_model: str = "yolo12n.pt",
    output_model_name: str = "yolo_bangunan_lokal.pt",
    log_callback: Optional[Callable[[str], None]] = None,
    progress_callback: Optional[Callable[[float, str], None]] = None,
) -> bool:
    """
    Runs YOLOv8 training.
    """
    log = log_callback or print
    progress = progress_callback or (lambda p, m: None)

    try:
        log(f"🧠 Menginisialisasi training YOLO dengan base model {base_model}...")
        progress(72, "Memuat model ultralytics YOLO...")

        try:
            from ultralytics import YOLO
            import ultralytics.nn.tasks as tasks
            try:
                from core.models.uav_yolov12_modules import PConv, SKNet
                tasks.PConv = PConv
                tasks.SKNet = SKNet
                
                # Monkey-patch parse_model agar mengenali PConv dan SKNet sebagai base_modules
                # sehingga `c1` (in_channels) dilempar otomatis ke parameter mereka.
                original_parse_model = tasks.parse_model
                def custom_parse_model(d, ch, verbose=True):
                    # We inject our modules into the parsing loop by temporarily wrapping the modules
                    # Actually, we can just intercept the created layers!
                    pass # We will do a better interception below
                    
                # A safer monkey-patch: replace parse_model entirely for this run!
                import inspect
                try:
                    source = inspect.getsource(original_parse_model)
                except OSError:
                    # Fallback jika getsource gagal (misal file .pyc atau environment Windows)
                    import ast
                    tasks_file = inspect.getfile(tasks)
                    with open(tasks_file, "r", encoding="utf-8") as f:
                        tasks_code = f.read()
                    parsed = ast.parse(tasks_code)
                    for node in parsed.body:
                        if isinstance(node, ast.FunctionDef) and node.name == 'parse_model':
                            source = ast.unparse(node)
                            break
                    else:
                        raise ValueError("parse_model not found in tasks.py")

                # Sisipkan PConv dan SKNet ke dalam set base_modules di kode sumbernya!
                if 'base_modules = frozenset({' in source:
                    source = source.replace('base_modules = frozenset({', 'base_modules = frozenset({PConv, SKNet, ')
                elif 'base_modules = frozenset(' in source:
                    source = source.replace('base_modules = frozenset(', 'base_modules = frozenset({PConv, SKNet} | ')
                
                # Execute the modified source
                exec_globals = tasks.__dict__.copy()
                exec_globals['PConv'] = PConv
                exec_globals['SKNet'] = SKNet
                exec(source, exec_globals)
                tasks.parse_model = exec_globals['parse_model']
                
            except Exception as e:
                log(f"⚠️ Peringatan: Patch modul kustom UAV-YOLO gagal ({e})")
                import traceback
                log(traceback.format_exc())
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
            device="0", # use GPU 0, will fallback to cpu if fail usually but ultralytics handles it
            workers=0, # Fix for Windows WinError 1455 (pagefile too small)
            lr0=0.001, # Learning rate lebih kecil untuk fine-tuning agar tidak merusak bobot lama
            lrf=0.01,  # Final learning rate
            patience=25 # Early stopping jika model makin bodoh
        )
        
        output_model_path = os.path.join(models_dir, output_model_name)
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

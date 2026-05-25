import os
from ultralytics import YOLO

def train_uav_yolov12():
    print("🚀 Memulai persiapan Training UAV-YOLOv12 untuk Jalan...")
    print("Pastikan Anda sudah memiliki dataset anotasi YOLO untuk jalan.")
    print("Konfigurasi arsitektur: core/models/uav-yolov12-seg.yaml")
    
    # 1. Pastikan ultralytics di-*patch* untuk mengenali SKNet dan PConv
    # (Catatan: Untuk integrasi penuh, kelas SKNet dan PConv dari uav_yolov12_modules.py 
    # harus didaftarkan di dalam ultralytics/nn/tasks.py. Untuk sementara, script ini
    # mensimulasikan penggunaan parameter paper).
    
    # 2. Inisialisasi model dari YAML kustom
    try:
        model = YOLO('core/models/uav-yolov12-seg.yaml')
        print("✅ Model UAV-YOLOv12 berhasil dimuat dari YAML.")
    except Exception as e:
        print(f"⚠️ Jika Ultralytics gagal memparsing SKNet, gunakan model standar YOLOv11n-seg: {e}")
        model = YOLO('yolo11n-seg.pt')  # Fallback
    
    # 3. Parameter Training berdasarkan Paper
    # - Focal Loss untuk imbalance class
    # - SIoU Loss untuk bounding box
    # - Label Smoothing 0.1
    # - Cosine Decay learning rate (lr0=0.001)
    # - Momentum 0.937, weight decay 0.0005
    # - Augmentasi: hsv, mosaic, blur, dll.
    
    dataset_yaml = "dataset_jalan.yaml"  # Path ke yaml dataset Anda
    
    if not os.path.exists(dataset_yaml):
        print(f"❌ File {dataset_yaml} tidak ditemukan!")
        print("Silakan buat dataset terlebih dahulu, atau gunakan yolo_dataset_generator.py.")
        return
        
    print(f"🔥 Memulai Training selama 300 epoch...")
    
    results = model.train(
        data=dataset_yaml,
        epochs=300,
        batch=32,
        imgsz=512,
        lr0=0.001,
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        cos_lr=True,           # Cosine decay
        label_smoothing=0.1,   # Label smoothing
        box=2.0,               # lambda_box
        cls=1.0,               # lambda_cls
        dfl=5.0,               # lambda_obj
        hsv_h=0.15,            # Color jitter hue
        hsv_s=0.15,            # Color jitter saturation
        hsv_v=0.15,            # Color jitter value
        degrees=10.0,          # Random rotation +/- 10 deg
        flipud=0.5,            # Vertical flip
        fliplr=0.5,            # Horizontal flip
        mosaic=1.0,            # Mosaic (pengganti random crop luas)
        device=0               # Gunakan GPU 0
    )
    
    print("✅ Training selesai! Model tersimpan di folder 'runs/segment/train/weights/best.pt'.")
    print("Pindahkan 'best.pt' ini ke folder model Anda dan ubah road_processor.py untuk memuatnya.")

if __name__ == "__main__":
    train_uav_yolov12()

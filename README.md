# SAM-Geo Building Digitizer

Aplikasi desktop untuk **digitasi otomatis bangunan** dari citra raster (ECW / GeoTIFF)
menggunakan **Segment Anything Model (SAM)** via `segment-geospatial`.

## Fitur Utama
- 📂 Input ECW dan GeoTIFF
- 🤖 Model: MobileSAM, SAM-B, SAM-L, SAM-H
- ⚡ Akselerasi GPU NVIDIA (CUDA)
- 🧩 Tiling otomatis untuk citra besar
- 🔲 Regularisasi sudut bangunan ke 90°
- 🌑 Filter bayangan otomatis
- 💾 Ekspor ke Shapefile (.shp)

## Instalasi

```bat
install.bat
```

## Menjalankan Aplikasi

```bat
run.bat
```

## Struktur Proyek
```
Project SAM/
├── app.py              # Entry point
├── core/               # Backend pipeline
│   ├── ecw_handler.py  # Konversi ECW → GeoTIFF
│   ├── tiling.py       # Sliding window tiling
│   ├── sam_processor.py # SAM-Geo wrapper
│   ├── postprocess.py  # Filter + regularisasi
│   └── exporter.py     # Ekspor SHP
├── ui/                 # GUI components
│   ├── sidebar.py      # Panel kontrol
│   ├── preview_panel.py # Preview peta
│   └── log_panel.py    # Log + progress
├── models/             # SAM model checkpoints (auto-download)
├── temp/               # File sementara
├── output/             # Hasil digitasi (SHP)
├── install.bat         # Script instalasi
└── run.bat             # Script launcher
```

## Persyaratan Sistem
- Windows 10/11 64-bit
- Python 3.11
- GPU NVIDIA dengan CUDA 12.1 (disarankan untuk performa optimal)
- RAM minimal 8 GB (16 GB disarankan)
- VRAM minimal 4 GB (RTX 3050: 4 GB — gunakan MobileSAM atau SAM-B)

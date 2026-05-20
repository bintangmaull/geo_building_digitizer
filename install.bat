@echo off
title SAM-Geo Building Digitizer - Installer
color 0B

echo ========================================
echo   SAM-Geo Building Digitizer v1.2
echo   Installer / Setup Script
echo ========================================
echo.
echo   GPU  : NVIDIA GeForce RTX 3050 Laptop
echo   CUDA : 12.8 (PyTorch) / 13.2 (Driver)
echo ========================================
echo.

set "SCRIPT_DIR=%~dp0"
set "VENV_DIR=%SCRIPT_DIR%venv"
set "VENV_PYTHON=%VENV_DIR%\Scripts\python.exe"
set "VENV_PIP=%VENV_DIR%\Scripts\pip.exe"

REM ─────────────────────────────────────────
echo [1/5] Memeriksa virtual environment...
REM ─────────────────────────────────────────
if exist "%VENV_PYTHON%" (
    echo [INFO] Venv sudah ada di: %VENV_DIR%
) else (
    echo [INFO] Membuat virtual environment dengan Python 3.11...
    py -3.11 -m venv "%VENV_DIR%" --prompt SAM-Geo
    if %errorlevel% neq 0 (
        echo [ERROR] Gagal membuat venv. Pastikan Python 3.11 terinstal.
        echo         Download: https://www.python.org/downloads/
        pause
        exit /b 1
    )
    echo [OK] Venv berhasil dibuat.
)

REM ─────────────────────────────────────────
echo.
echo [2/5] Mengupdate pip dan setuptools...
REM ─────────────────────────────────────────
"%VENV_PYTHON%" -m pip install --upgrade pip setuptools wheel --quiet
echo [OK] pip diperbarui.

REM ─────────────────────────────────────────
echo.
echo [3/5] Menginstal PyTorch dengan dukungan CUDA 12.8 (GPU NVIDIA RTX)...
echo [INFO] Kompatibel dengan driver NVIDIA 596.49+ (CUDA Runtime 13.2)
echo [INFO] Ukuran file besar (~2.8 GB), harap tunggu...
REM ─────────────────────────────────────────
"%VENV_PIP%" install torch torchvision --index-url https://download.pytorch.org/whl/cu128 --upgrade
if %errorlevel% neq 0 (
    echo [WARN] PyTorch CUDA 12.8 gagal, mencoba CUDA 12.1...
    "%VENV_PIP%" install torch torchvision --index-url https://download.pytorch.org/whl/cu121 --upgrade
    if %errorlevel% neq 0 (
        echo [WARN] PyTorch CUDA gagal juga, menginstal versi CPU saja...
        "%VENV_PIP%" install torch torchvision
    )
)
echo [OK] PyTorch terinstal.

REM ─────────────────────────────────────────
echo.
echo [4/5] Menginstal library geospasial dan GUI...
REM ─────────────────────────────────────────
"%VENV_PIP%" install customtkinter rasterio geopandas shapely matplotlib Pillow "numpy>=1.24,<3"
if %errorlevel% neq 0 (
    echo [ERROR] Instalasi library gagal.
    pause
    exit /b 1
)
echo [OK] Library geospasial terinstal.

REM ─────────────────────────────────────────
echo.
echo [5/5] Menginstal segment-geospatial (samgeo)...
REM ─────────────────────────────────────────
"%VENV_PIP%" install segment-geospatial
if %errorlevel% neq 0 (
    echo [ERROR] Instalasi segment-geospatial gagal.
    pause
    exit /b 1
)
echo [OK] segment-geospatial terinstal.

REM ─────────────────────────────────────────
echo.
echo ========================================
echo   Instalasi SELESAI!
echo ========================================
echo.
echo Verifikasi instalasi:
"%VENV_PYTHON%" -c "import torch; import rasterio; import geopandas; import samgeo; import customtkinter; print('Semua library OK!'); print('PyTorch:', torch.__version__); print('CUDA Tersedia:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only')"
echo.
echo Jalankan aplikasi dengan: run.bat
echo.
pause

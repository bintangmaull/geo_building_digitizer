@echo off
title SAM-Geo Building Digitizer - Launcher
color 0A

echo ========================================
echo   SAM-Geo Building Digitizer v1.2
echo   Digitasi Bangunan Otomatis dengan SAM
echo ========================================
echo.

set "SCRIPT_DIR=%~dp0"
set "VENV_DIR=%SCRIPT_DIR%venv"
set "VENV_PYTHON=%VENV_DIR%\Scripts\python.exe"

REM Check if venv exists
if not exist "%VENV_PYTHON%" (
    echo [ERROR] Virtual environment tidak ditemukan!
    echo Lokasi yang diharapkan: %VENV_DIR%
    echo.
    echo Silakan jalankan install.bat terlebih dahulu.
    echo.
    pause
    exit /b 1
)

echo [INFO] Menggunakan Python dari: %VENV_PYTHON%
echo [INFO] Memulai aplikasi...
echo.

cd /d "%SCRIPT_DIR%"
"%VENV_PYTHON%" app.py

if %errorlevel% neq 0 (
    echo.
    echo [ERROR] Aplikasi berhenti dengan error (kode: %errorlevel%)
    echo Periksa pesan error di atas.
    pause
)

"""
feature_augmentation.py
Ekstraksi dan augmentasi fitur spektral dari citra RGB drone/satelit.

Fungsi utama:
    - compute_exg      : Excess Green Index (vegetasi)
    - compute_vari     : Visible Atmospherically Resistant Index (vegetasi)
    - compute_texture  : Laplacian + Sobel edge strength (jalan/infrastruktur)
    - compute_ndwi     : Approximasi NDWI dari RGB (badan air)
    - refine_veg_mask  : Konfirmasi mask vegetasi dengan ExG + VARI
    - refine_road_mask : Konfirmasi mask jalan dengan texture + non-green
    - refine_water_mask: Konfirmasi mask air dengan warna + kegelapan

Digunakan sebagai tahap refinement SETELAH SegFormer inference
untuk meningkatkan akurasi per-kelas.
"""

import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Index Computation
# ─────────────────────────────────────────────────────────────────────────────

def compute_exg(rgb: np.ndarray) -> np.ndarray:
    """
    Excess Green Index: ExG = 2G - R - B
    Nilai tinggi → vegetasi (hijau dominan).
    Input: HxWx3 uint8 atau float, channel order RGB.
    Output: HxW float.
    """
    r = rgb[:, :, 0].astype(float)
    g = rgb[:, :, 1].astype(float)
    b = rgb[:, :, 2].astype(float)
    return 2.0 * g - r - b


def compute_vari(rgb: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """
    Visible Atmospherically Resistant Index:
    VARI = (G - R) / (G + R - B + eps)
    Nilai > 0 → vegetasi, nilai < 0 → non-vegetasi.
    Output: HxW float, di-clip ke [-1, 1].
    """
    r = rgb[:, :, 0].astype(float)
    g = rgb[:, :, 1].astype(float)
    b = rgb[:, :, 2].astype(float)
    denom = g + r - b + eps
    vari = (g - r) / denom
    return np.clip(vari, -1.0, 1.0)


def compute_texture(rgb: np.ndarray) -> np.ndarray:
    """
    Texture / Edge Strength via Laplacian pada channel grayscale.
    Nilai tinggi → tepi tajam (jalan, bangunan, infrastruktur).
    Output: HxW float, dinormalisasi ke [0, 255].
    """
    import cv2
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    texture = np.abs(lap)
    # Normalize to 0-255
    t_max = texture.max()
    if t_max > 0:
        texture = texture / t_max * 255.0
    return texture


def compute_ndwi_rgb(rgb: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """
    Approximasi NDWI dari RGB:
    NDWI_approx = (G - R) / (G + R + eps)
    Nilai > 0 → kemungkinan ada air (hijau lebih terang dari merah).
    Tidak sempurna tanpa NIR, tapi berguna sebagai filter tambahan.
    Output: HxW float, di-clip ke [-1, 1].
    """
    r = rgb[:, :, 0].astype(float)
    g = rgb[:, :, 1].astype(float)
    ndwi = (g - r) / (g + r + eps)
    return np.clip(ndwi, -1.0, 1.0)


def compute_brightness(rgb: np.ndarray) -> np.ndarray:
    """Kecerahan rata-rata: mean(R, G, B). Output HxW float."""
    return rgb.astype(float).mean(axis=2)


# ─────────────────────────────────────────────────────────────────────────────
# Mask Refinement
# ─────────────────────────────────────────────────────────────────────────────

def refine_vegetation_mask(
    seg_mask: np.ndarray,
    rgb: np.ndarray,
    exg_threshold: float = 5.0,
    vari_threshold: float = -0.1,
) -> np.ndarray:
    """
    Perkuat/konfirmasi mask vegetasi dari SegFormer menggunakan ExG dan VARI.

    Piksel dipertahankan sebagai vegetasi jika:
      - SegFormer mendeteksinya sebagai vegetasi, DAN
      - ExG > exg_threshold ATAU VARI > vari_threshold

    Ini menghapus false positive seperti atap hijau, bangunan berwarna hijau, dll.

    Args:
        seg_mask     : HxW bool — mask vegetasi dari SegFormer
        rgb          : HxWx3 uint8 — citra RGB asli
        exg_threshold: Nilai ExG minimum untuk konfirmasi vegetasi
        vari_threshold: Nilai VARI minimum untuk konfirmasi vegetasi
    Returns:
        HxW bool — mask vegetasi yang telah disempurnakan
    """
    exg = compute_exg(rgb)
    vari = compute_vari(rgb)

    # Konfirmasi: setidaknya satu indikator spektral mendukung vegetasi
    spectral_confirm = (exg > exg_threshold) | (vari > vari_threshold)

    # Gabungkan: SegFormer ATAU (buffer tipis SegFormer DAN spectral)
    # Ini mempertahankan deteksi SegFormer yang sudah baik
    # dan menambah vegetasi yang mungkin terlewat
    import scipy.ndimage as ndi
    # Expand sedikit untuk menangkap tepi vegetasi
    seg_dilated = ndi.binary_dilation(seg_mask, iterations=2)
    expanded = seg_dilated & spectral_confirm

    # Gabungkan deteksi asli + ekspansi
    refined = seg_mask | expanded
    return refined.astype(bool)


def refine_road_mask(
    seg_mask: np.ndarray,
    rgb: np.ndarray,
    saturation_max: float = 60.0,
    veg_exg_threshold: float = 15.0,
) -> np.ndarray:
    """
    Perkuat mask jalan dari SegFormer menggunakan texture dan warna aspal.

    Piksel dipertahankan sebagai jalan jika:
      - SegFormer mendeteksinya sebagai jalan, DAN
      - Bukan vegetasi dominan (ExG rendah)
      - Saturation HSV rendah (abu-abu/netral = aspal/beton)

    Args:
        seg_mask         : HxW bool — mask jalan dari SegFormer
        rgb              : HxWx3 uint8 — citra RGB asli
        saturation_max   : Threshold saturation HSV maksimum untuk aspal
        veg_exg_threshold: Threshold ExG untuk mengidentifikasi vegetasi
    Returns:
        HxW bool — mask jalan yang telah disempurnakan
    """
    import cv2

    # Ubah RGB ke HSV (cv2 menggunakan BGR)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    saturation = hsv[:, :, 1].astype(float)

    exg = compute_exg(rgb)

    # Indikator non-vegetasi (aspal/beton)
    not_vegetation = exg < veg_exg_threshold
    low_saturation = saturation < saturation_max  # warna netral/abu-abu

    # Konfirmasi jalan: bukan vegetasi atau saturation rendah
    road_confirm = not_vegetation | low_saturation

    refined = seg_mask & road_confirm
    return refined.astype(bool)


def refine_water_mask(
    seg_mask: np.ndarray,
    rgb: np.ndarray,
    blue_boost_threshold: float = -10.0,
    brightness_max: float = 160.0,
) -> np.ndarray:
    """
    Perkuat mask badan air dari SegFormer menggunakan analisis warna.

    Cara kerja:
      - Pertahankan semua piksel yang sudah terdeteksi SegFormer sebagai air
      - Tambahkan piksel yang sangat gelap + blue-dominant (air dalam/keruh)
      - Hapus piksel yang sangat hijau (vegetasi di pinggir air)

    Args:
        seg_mask              : HxW bool — mask air dari SegFormer
        rgb                   : HxWx3 uint8 — citra RGB asli
        blue_boost_threshold  : (B - R) minimum untuk menganggap piksel sebagai air
        brightness_max        : Kecerahan maksimum untuk air yang gelap
    Returns:
        HxW bool — mask badan air yang telah disempurnakan
    """
    r = rgb[:, :, 0].astype(float)
    g = rgb[:, :, 1].astype(float)
    b = rgb[:, :, 2].astype(float)

    brightness = compute_brightness(rgb)
    exg = compute_exg(rgb)

    # Blue dominance: B >= R - threshold (air biru atau netral, bukan merah/kuning)
    blue_dominant = (b - r) >= blue_boost_threshold

    # Air gelap (air dalam, teduh): kecerahan rendah DAN blue dominant
    dark_water = (brightness < brightness_max) & blue_dominant

    # Bukan vegetasi (ExG rendah)
    not_veg = exg < 15.0

    # Ekspansi: tambahkan piksel air gelap yang berdekatan dengan deteksi SegFormer
    import scipy.ndimage as ndi
    seg_dilated = ndi.binary_dilation(seg_mask, iterations=3)
    extra_water = seg_dilated & dark_water & not_veg

    refined = seg_mask | extra_water
    # Hapus piksel yang sangat vegetatif dari mask air
    refined = refined & not_veg
    return refined.astype(bool)

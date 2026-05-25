"""
core/objects/__init__.py
Paket processor untuk digitasi objek geospasial selain bangunan.

Setiap modul berdiri sendiri dan tidak bergantung pada kode bangunan
(postprocess.py / sam_processor.py).

Modul:
    - road_processor      : Jalan & Infrastruktur (jembatan, rel kereta)
    - water_processor     : Badan Air (sungai, danau, kolam, kanal)
    - vegetation_processor: Vegetasi (hutan, taman, sawah, kebun)
"""

from .road_processor import RoadProcessor
from .road_fingerprint import RoadFingerprintBuilder
from .water_processor import WaterProcessor
from .vegetation_processor import VegetationProcessor

__all__ = [
    "RoadProcessor",
    "RoadFingerprintBuilder",
    "WaterProcessor",
    "VegetationProcessor",
]

import os
import sys
import sys
from pathlib import Path

# Fix print encoding on windows
sys.stdout.reconfigure(encoding='utf-8')
sys.path.append(os.path.abspath('.'))

from core.objects.gemini_processor import GeminiProcessor

def log_cb(msg):
    print(msg)

processor = GeminiProcessor(api_key=open(".gemini_key").read().strip(), log_callback=log_cb)
if not processor.is_ready():
    print("Gemini API tidak siap.")
    sys.exit(1)

tiles_dir = Path("temp/tiles")
tiles = list(tiles_dir.glob("*.tif"))
if not tiles:
    print("Tidak ada file tile tif.")
    sys.exit(1)

test_tile = str(tiles[0])
print(f"Menguji tile: {test_tile}")

boxes = processor.detect_boxes(test_tile)
print(f"Hasil boxes numpy:\n{boxes}")
print(f"Shape: {boxes.shape if hasattr(boxes, 'shape') else type(boxes)}")

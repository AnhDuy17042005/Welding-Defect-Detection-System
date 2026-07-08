"""Shared visualization settings for CLI and web inference."""

from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent

SHOW_MASKS = False
SHOW_BOXES = True
SHOW_LABELS = False
SHOW_CONFIDENCE = False

RIPPLE_COLOR = (0, 255, 0)
RIPPLE_IMAGE_WEIGHT = 0.65
RIPPLE_MASK_WEIGHT = 0.35

BOX_COLORS = (
    (0, 0, 255),
    (0, 165, 255),
    (255, 0, 255),
    (255, 128, 0),
    (0, 255, 255),
)

CLASS_COLORS = {
    0: (255, 180, 180),
    1: (0, 140, 255),
    2: (255, 0, 255),
    3: (80, 80, 180),
    4: (0, 0, 255),
    5: (255, 255, 0),
    6: (255, 120, 0),
    7: (255, 200, 0),
    8: (0, 255, 255),
    9: (0, 255, 0),
}

OVERLAY_IMAGE_EXTENSION = ".jpg"
OVERLAY_JPEG_QUALITY = 92
MASK_IMAGE_EXTENSION = ".png"

ANNOTATION_VISUALIZE_DIR = ROOT / "data" / "visualize"
ANNOTATION_DISPLAY_SIZE = (640, 640)
ANNOTATION_SAMPLE_COUNT = 50

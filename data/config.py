from __future__ import annotations

from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent

RAW_DIR = BASE_DIR / "dataset" / "raw"
PROCESSED_DIR = BASE_DIR / "dataset" / "processed"

SPLITS = ("train", "valid", "test")
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

CLASS_NAMES = {
    0: "crack",
    1: "porosity",
    2: "spatter",
    3: "welding_line",
}

CLASS_COLORS = {
    0: (0, 0, 255),        # red
    1: (0, 255, 255),      # yellow
    2: (200, 120, 240),    # pink/purple
    3: (0, 255, 0),        # green
}

DEFAULT_IMAGES_DIR      = PROCESSED_DIR / "train" / "images"
DEFAULT_LABELS_DIR      = PROCESSED_DIR / "train" / "labels"
DEFAULT_VISUALIZE_DIR   = BASE_DIR / "data" / "visualize"

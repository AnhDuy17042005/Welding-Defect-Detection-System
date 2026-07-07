from __future__ import annotations

"""Compatibility aliases for older data scripts.

New code should import directly from config.data and config.visualize.
"""

from configs.data import (
    CLASS_NAMES,
    DEFAULT_ANNOTATION_IMAGES,
    DEFAULT_ANNOTATION_LABELS,
    IMAGE_EXTENSIONS,
    RAW_DATASET,
    ROOT,
    SPLITS,
    YOLO_DATASET,
)
from configs.visualize import ANNOTATION_VISUALIZE_DIR, CLASS_COLORS


BASE_DIR = ROOT
RAW_DIR = RAW_DATASET
PROCESSED_DIR = YOLO_DATASET
IMAGE_EXTS = IMAGE_EXTENSIONS
DEFAULT_IMAGES_DIR = DEFAULT_ANNOTATION_IMAGES
DEFAULT_LABELS_DIR = DEFAULT_ANNOTATION_LABELS
DEFAULT_VISUALIZE_DIR = ANNOTATION_VISUALIZE_DIR

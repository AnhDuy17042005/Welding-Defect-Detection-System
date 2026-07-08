"""Shared project paths.

Keep this module limited to directory locations. Model, dataset, and runtime
defaults belong to their dedicated config modules.
"""

from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent

CONFIG_DIR          = ROOT / "configs"
SRC_DIR             = ROOT / "src"
BACKEND_DIR         = ROOT / "backend"
FRONTEND_DIR        = ROOT / "frontend"
ASSETS_DIR          = ROOT / "assets"
NOTEBOOKS_DIR       = ROOT / "notebooks"

DATASET_DIR         = ROOT / "dataset"
MODELS_DIR          = ROOT / "models"
OUTPUT_DIR          = ROOT / "output"
METRICS_DIR         = ROOT / "metrics"

YOLO_RUNS_DIR       = MODELS_DIR / "runs"
UNET_RUNS_DIR       = MODELS_DIR / "unet"

YOLO_OUTPUT_DIR     = OUTPUT_DIR / "yolo"
UNET_OUTPUT_DIR     = OUTPUT_DIR / "unet"
HYBRID_OUTPUT_DIR   = OUTPUT_DIR / "hybrid"

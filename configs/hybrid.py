"""Configuration shared by hybrid CLI and backend inference."""

from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent

HYBRID_OUTPUT = ROOT / "output" / "hybrid"
HYBRID_DEFAULT_IMAGE = ROOT / "dataset" / "require" / "9 (2).jpg"

ROI_MARGIN                  = 0.50
MIN_ROI_MARGIN              = 0.0
MAX_ROI_MARGIN              = 1.0

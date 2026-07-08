"""
    Dataset paths and annotation schema shared across the project.
"""

from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent

DATASET_DIR = ROOT / "dataset"
RAW_DATASET = DATASET_DIR / "raw"

SPLITS           = ("train", "valid", "test")
IMAGE_EXTENSIONS = frozenset({".jpg", ".jpeg", ".png", ".bmp", ".webp"})
MASK_EXTENSION   = ".png"

"""
    YOLO segmentation datasets
"""
YOLO_SOURCE_DATASET = DATASET_DIR  / "finaldata"
YOLO_DATASET        = DATASET_DIR  / "augmented"
YOLO_DATA_YAML      = YOLO_DATASET / "data.yaml"

"""
    U-Net ripple datasets
"""
RIPPLE_SOURCE_DATASET = DATASET_DIR / "ripple"
RIPPLE_SPLIT_DATASET  = DATASET_DIR / "ripple_split"
RIPPLE_ROI_DATASET    = DATASET_DIR / "ripple_roi"

"""
    Images used for manual inference and demonstrations
"""
REQUIRE_DATASET = DATASET_DIR / "require"
TEST_IMAGE_DIR  = DATASET_DIR / "test"

CLASS_NAMES = {
    0: "arc_strike",
    1: "continuous_undercut",
    2: "crack",
    3: "end_crater_pipe",
    4: "porosity",
    5: "root_concavity",
    6: "root_overlap",
    7: "slag_inclusion",
    8: "spatter",
    9: "welding_line",
}

WELDING_LINE_CLASS = "welding_line"
NON_DEFECT_CLASSES = frozenset(
    {
        "good_weld",
        "good-weld",
        "good weld",
        "welding_line",
        "welding-line",
        "welding line",
        "weld_line",
        "weld-line",
        "weld line",
    }
)

DEFAULT_ANNOTATION_IMAGES = YOLO_DATASET / "train" / "images"
DEFAULT_ANNOTATION_LABELS = YOLO_DATASET / "train" / "labels"

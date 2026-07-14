"""YOLO model registry and default train/inference settings."""

from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent

YOLO_RUNS_DIR       = ROOT / "models" / "runs"
YOLO_OUTPUT_DIR     = ROOT / "output" / "yolo"
YOLO_DATA_YAML      = ROOT / "dataset" / "augmented" / "data.yaml"
YOLO_DEFAULT_IMAGE  = ROOT / "dataset" / "require" / "5 (3).jpg"

YOLO_MODEL_ID       = "yolo_ver5"
YOLO_MODEL_VERSION  = 5
YOLO_MODEL          = YOLO_RUNS_DIR / f"train_ver{YOLO_MODEL_VERSION}" / "weights" / "best_openvino_model"

YOLO_MODELS = {
    f"yolo_ver{version}": YOLO_RUNS_DIR / f"train_ver{version}" / "weights" / "best_openvino_model"
    for version in range(1, 6)
}
YOLO_MODEL_LABELS = {
    model_id: f"YOLOv11 Ver {version}"
    for version, model_id in enumerate(YOLO_MODELS, start=1)
}

"""
    Inference configs
"""
YOLO_IMAGE_SIZE             = 960
YOLO_CONFIDENCE_THRESHOLD   = 0.25
YOLO_IOU_THRESHOLD          = 0.25
YOLO_MASK_THRESHOLD         = 0.5
YOLO_DEVICE                 = None

"""
    Training configs
"""
YOLO_BASE_MODEL             = "yolo11m-seg.pt"
YOLO_EPOCHS                 = 80
YOLO_BATCH_SIZE             = 2
YOLO_LEARNING_RATE          = 1e-4
YOLO_FINAL_LEARNING_RATE    = 0.01
YOLO_OPTIMIZER              = "AdamW"
YOLO_FREEZE_LAYERS          = 0
YOLO_CLASS_LOSS_GAIN        = 0.8
YOLO_PATIENCE               = 30
YOLO_SEED                   = 42
YOLO_WORKERS                = 2
YOLO_CACHE                  = False
YOLO_EXIST_OK               = True
YOLO_TASK                   = "segment"
YOLO_FRACTION               = 1.0

"""
    Augmentation configs
"""
YOLO_MOSAIC                 = 0.8
YOLO_COPY_PASTE             = 0.4
YOLO_DEGREES                = 15.0
YOLO_FLIP_LR                = 0.5
YOLO_FLIP_UD                = 0.3
YOLO_SCALE                  = 0.6
YOLO_SHEAR                  = 2.0
YOLO_PERSPECTIVE            = 0.0003
YOLO_HSV_H                  = 0.01
YOLO_HSV_S                  = 0.6
YOLO_HSV_V                  = 0.5

YOLO_TRAIN_DATA             = YOLO_DATA_YAML
YOLO_TRAIN_PROJECT          = YOLO_RUNS_DIR
YOLO_INFERENCE_OUTPUT       = YOLO_OUTPUT_DIR

"""
    Offline augmentation configs
"""
YOLO_AUGMENT_INPUT_IMAGES   = ROOT / "dataset" / "finaldata" / "train" / "images"
YOLO_AUGMENT_INPUT_LABELS   = ROOT / "dataset" / "finaldata" / "train" / "labels"
YOLO_AUGMENT_OUTPUT_IMAGES  = ROOT / "dataset" / "augmented" / "train" / "images"
YOLO_AUGMENT_OUTPUT_LABELS  = ROOT / "dataset" / "augmented" / "train" / "labels"
YOLO_AUGMENT_SCALE          = 7

YOLO_GEOMETRIC_AUGMENTATIONS = {
    "flip_horizontal": True,
    "flip_vertical": False,
    "rotation": True,
    "crop": False,
    "scale": True,
    "shear": False,
}

YOLO_PHOTOMETRIC_AUGMENTATIONS = {
    "blur": True,
    "brightness_contrast": True,
    "gaussian_noise": True,
    "hue_saturation": False,
}

YOLO_AUGMENT_PROBABILITIES = {
    "blur": 0.30,
    "brightness_contrast": 0.60,
    "gaussian_noise": 0.40,
    "hue_saturation": 0.0,
    "flip_horizontal": 0.50,
    "flip_vertical": 0.0,
    "rotation": 0.60,
    "crop": 0.0,
    "scale": 0.50,
    "shear": 0.0,
}

YOLO_AUGMENT_PARAMETERS = {
    "blur": 3,
    "brightness_limit": 0.22,
    "contrast_limit": 0.22,
    "hue_shift_limit": 5,
    "sat_shift_limit": 10,
    "val_shift_limit": 10,
    "std_range": (0.03, 0.08),
    "rotation": (-90, 90),
    "crop_ratio": (0.9, 1.0),
    "scale": (0.85, 1.15),
    "shear": (-0.05, 0.05),
}

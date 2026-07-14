"""
    U-Net model registry and default train/inference settings
"""

from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent

UNET_RUNS_DIR         = ROOT / "models" / "unet"
UNET_OUTPUT_DIR       = ROOT / "output" / "unet"
RIPPLE_SOURCE_DATASET = ROOT / "dataset" / "ripple"
RIPPLE_SPLIT_DATASET  = ROOT / "dataset" / "ripple_split"
RIPPLE_ROI_DATASET    = ROOT / "dataset" / "ripple_roi"
UNET_DEFAULT_IMAGE    = ROOT / "dataset" / "test" / "5 (3).jpg"

UNET_MODEL_ID       = "unet_ver3"
UNET_MODEL_VERSION  = 3
UNET_MODEL          = UNET_RUNS_DIR / f"train_ver{UNET_MODEL_VERSION}" / "best.xml"

UNET_MODELS = {
    f"unet_ver{version}": UNET_RUNS_DIR / f"train_ver{version}" / "best.xml"
    for version in range(1, 4)
}
UNET_MODEL_LABELS = {
    model_id: f"U-Net Ver {version}"
    for version, model_id in enumerate(UNET_MODELS, start=1)
}

"""
    Architecture and preprocessing defaults
"""
UNET_INPUT_CHANNELS = 3
UNET_NUM_CLASSES    = 1
UNET_BASE_CHANNELS  = 64
UNET_IMAGE_SIZE     = 512
UNET_THRESHOLD      = 0.7
UNET_METRIC_THRESHOLD = 0.5
UNET_DEVICE         = "auto"
IMAGENET_MEAN       = (0.485, 0.456, 0.406)
IMAGENET_STD        = (0.229, 0.224, 0.225)

"""
    Training defaults
"""
UNET_TRAIN_DATA     = RIPPLE_ROI_DATASET
UNET_EPOCHS         = 100
UNET_BATCH_SIZE     = 8
UNET_LEARNING_RATE  = 1e-3
UNET_MIN_LEARNING_RATE = 1e-6
UNET_WEIGHT_DECAY   = 1e-4
UNET_NUM_WORKERS    = 4
UNET_DICE_ALPHA     = 0.5
UNET_MAX_GRADIENT_NORM = 1.0
UNET_TRAIN_OUTPUT   = ROOT / "models"

"""
    Online augmentation defaults
"""
UNET_HORIZONTAL_FLIP_PROBABILITY = 0.3
UNET_VERTICAL_FLIP_PROBABILITY   = 0.5
UNET_ROTATION_LIMIT              = 15
UNET_ROTATION_PROBABILITY        = 0.5
UNET_CROP_RATIO                  = 0.9
UNET_CROP_PROBABILITY            = 0.5
UNET_BRIGHTNESS_LIMIT            = 0.2
UNET_CONTRAST_LIMIT              = 0.2
UNET_BRIGHTNESS_PROBABILITY      = 0.6
UNET_NOISE_STD_RANGE             = (0.04, 0.12)
UNET_NOISE_PROBABILITY           = 0.4
UNET_BLUR_LIMIT                  = (3, 5)
UNET_BLUR_PROBABILITY            = 0.3
UNET_CLAHE_CLIP_LIMIT            = 2.0
UNET_CLAHE_PROBABILITY           = 0.3

"""
    ROI dataset and mask post-processing defaults
"""
UNET_ROI_DATA_MARGIN             = 0.1
UNET_OPEN_KERNEL_SIZE            = 3
UNET_CLOSE_KERNEL_SIZE           = 7
UNET_MIN_AREA_PIXELS             = 64
UNET_MIN_AREA_RATIO              = 0.001
UNET_MIN_LARGEST_RATIO           = 0.03
UNET_FILL_HOLES                  = True

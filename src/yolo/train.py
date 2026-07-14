import sys
from pathlib import Path

import torch
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parents[2]

"""Support direct script run from the project root."""
if __package__ in (None, ""):
    sys.path.insert(0, str(PROJECT_ROOT))

from configs.yolo import (
    YOLO_BASE_MODEL,
    YOLO_BATCH_SIZE,
    YOLO_CACHE,
    YOLO_CLASS_LOSS_GAIN,
    YOLO_COPY_PASTE,
    YOLO_DEGREES,
    YOLO_DEVICE,
    YOLO_EPOCHS,
    YOLO_EXIST_OK,
    YOLO_FINAL_LEARNING_RATE,
    YOLO_FLIP_LR,
    YOLO_FLIP_UD,
    YOLO_FRACTION,
    YOLO_FREEZE_LAYERS,
    YOLO_HSV_H,
    YOLO_HSV_S,
    YOLO_HSV_V,
    YOLO_IMAGE_SIZE,
    YOLO_LEARNING_RATE,
    YOLO_MOSAIC,
    YOLO_OPTIMIZER,
    YOLO_PATIENCE,
    YOLO_PERSPECTIVE,
    YOLO_SCALE,
    YOLO_SEED,
    YOLO_SHEAR,
    YOLO_TASK,
    YOLO_TRAIN_DATA,
    YOLO_TRAIN_PROJECT,
    YOLO_WORKERS,
)

def main() -> None:

    device = YOLO_DEVICE or ("cuda" if torch.cuda.is_available() else "cpu")
    model = YOLO(YOLO_BASE_MODEL)

    model.train(
        data=YOLO_TRAIN_DATA,
        imgsz=YOLO_IMAGE_SIZE,
        epochs=YOLO_EPOCHS,
        batch=YOLO_BATCH_SIZE,
        device=device,

        project=YOLO_TRAIN_PROJECT,
        exist_ok=YOLO_EXIST_OK,

        task=YOLO_TASK,
        patience=YOLO_PATIENCE,
        seed=YOLO_SEED,
        workers=YOLO_WORKERS,
        cache=YOLO_CACHE,
        fraction=YOLO_FRACTION,

        optimizer=YOLO_OPTIMIZER,
        lr0=YOLO_LEARNING_RATE,
        lrf=YOLO_FINAL_LEARNING_RATE,

        freeze=YOLO_FREEZE_LAYERS,
        cls=YOLO_CLASS_LOSS_GAIN,

        mosaic=YOLO_MOSAIC,
        copy_paste=YOLO_COPY_PASTE,

        degrees=YOLO_DEGREES,
        scale=YOLO_SCALE,
        shear=YOLO_SHEAR,
        perspective=YOLO_PERSPECTIVE,

        fliplr=YOLO_FLIP_LR,
        flipud=YOLO_FLIP_UD,

        hsv_h=YOLO_HSV_H,
        hsv_s=YOLO_HSV_S,
        hsv_v=YOLO_HSV_V,
    )

if __name__ == "__main__":
    main()

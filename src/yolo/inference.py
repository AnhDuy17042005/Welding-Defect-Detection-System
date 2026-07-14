from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import cv2
import numpy as np
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parents[2]

"""Support direct script run from the project root."""
if __package__ in (None, ""):
    sys.path.insert(0, str(PROJECT_ROOT))

from configs.yolo import (
    YOLO_CONFIDENCE_THRESHOLD,
    YOLO_DEFAULT_IMAGE,
    YOLO_IMAGE_SIZE,
    YOLO_INFERENCE_OUTPUT,
    YOLO_IOU_THRESHOLD,
    YOLO_MODEL,
    YOLO_TASK,
)

def parse_args() -> argparse.Namespace:
    """
        Parse command line arguments.
    """

    parser = argparse.ArgumentParser(
        description="Predict welding_line mask and extract centerline."
    )

    parser.add_argument("--image", type=Path, default=YOLO_DEFAULT_IMAGE)
    parser.add_argument("--model", type=Path, default=YOLO_MODEL)
    parser.add_argument("--output-dir", type=Path, default=YOLO_INFERENCE_OUTPUT)

    parser.add_argument("--conf", type=float, default=YOLO_CONFIDENCE_THRESHOLD)
    parser.add_argument("--iou", type=float, default=YOLO_IOU_THRESHOLD)
    parser.add_argument("--imgsz", type=int, default=YOLO_IMAGE_SIZE)

    return parser.parse_args()
 

def load_image(image_path: Path) -> np.ndarray:
    """
        Load image with OpenCV.

        OpenCV loads image in BGR format.
    """

    image = cv2.imread(str(image_path))

    if image is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")

    return image


def predict(
    model_path: Path,
    image: np.ndarray,
    conf: float,
    iou: float,
    imgsz: int,
):
    """
        Run YOLO prediction on one image.
    """

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    model = YOLO(str(model_path), task=YOLO_TASK)

    results = model.predict(
        source=image,
        imgsz=imgsz,
        conf=conf,
        iou=iou,
        verbose=False,
    )

    return results[0]


def save_image(path: Path, image: np.ndarray) -> None:
    """
        Save image and raise error if saving fails.
    """

    success = cv2.imwrite(str(path), image)

    if not success:
        raise RuntimeError(f"Failed to save image: {path}")


def save_outputs(
    output_dir: Path,
    image_path: Path,
    result,
) -> None:
    """
        Save all debug outputs.
    """

    output_dir.mkdir(parents=True, exist_ok=True)

    stem = image_path.stem

    prediction_image = result.plot(
        masks=False,
        boxes=True,
        labels=True,
        conf=True,
    )

    save_image(output_dir / f"{stem}_prediction.jpg", prediction_image)


def main() -> None:
    args = parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Image: {args.image}")
    print(f"Model: {args.model}")

    image = load_image(args.image)
    result = predict(
        model_path=args.model,
        image=image,
        conf=args.conf,
        iou=args.iou,
        imgsz=args.imgsz,
    )

    save_outputs(
        output_dir=args.output_dir,
        image_path=args.image,
        result=result,
    )

    print(f"Saved outputs to: {args.output_dir}")


if __name__ == "__main__":
    main()

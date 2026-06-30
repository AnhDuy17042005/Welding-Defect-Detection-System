"""Evaluate a trained U-Net ripple segmentation model.

Run from the project root:

    python unet/evaluation.py

or:

    python -m unet.evaluation \
        --model models/unet/best.pth \
        --data dataset/ripple_split \
        --split test

The output directory contains aggregate metrics and comparison images showing
the original image, ground truth, and prediction side by side.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader


BASE_DIR = Path(__file__).resolve().parent.parent.parent

# Support both `python unet/evaluation.py` and `python -m unet.evaluation`.
if __package__ in (None, ""):
    sys.path.insert(0, str(BASE_DIR))
    from unet.augment import get_val_transforms
    from unet.dataset import RippleDataset
    from unet.inference import get_device, load_model
    from unet.losses import BCEDiceLoss
else:
    from .augment import get_val_transforms
    from .dataset import RippleDataset
    from .inference import get_device, load_model
    from .losses import BCEDiceLoss


DEFAULT_MODEL_PATH = BASE_DIR / "models" / "unet" / "best.pth"
DEFAULT_DATA_ROOT  = BASE_DIR / "dataset" / "ripple_split"
DEFAULT_OUTPUT_DIR = BASE_DIR / "evaluation" / "unet"

EPSILON = 1e-8


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate U-Net ripple segmentation.")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--split", choices=("train", "valid", "test"), default="test")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--img-size", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument(
        "--max-visualizations",
        type=int,
        default=20,
        help="Maximum number of comparison images to save; use 0 to disable.",
    )
    args = parser.parse_args()

    if not 0.0 <= args.threshold <= 1.0:
        parser.error("--threshold must be between 0 and 1")
    if args.batch_size < 1:
        parser.error("--batch-size must be at least 1")
    if args.num_workers < 0:
        parser.error("--num-workers cannot be negative")
    if args.max_visualizations < 0:
        parser.error("--max-visualizations cannot be negative")

    return args


def confusion_counts(
    prediction: torch.Tensor,
    target: torch.Tensor,
) -> tuple[int, int, int, int]:
    prediction = prediction.bool()
    target = target.bool()

    true_positive = int(torch.logical_and(prediction, target).sum().item())
    false_positive = int(torch.logical_and(prediction, ~target).sum().item())
    false_negative = int(torch.logical_and(~prediction, target).sum().item())
    true_negative = int(torch.logical_and(~prediction, ~target).sum().item())
    
    return true_positive, false_positive, false_negative, true_negative


def metrics_from_counts(tp: int, fp: int, fn: int, tn: int) -> dict[str, float]:
    total = tp + fp + fn + tn
    union = tp + fp + fn
    predicted_positive = tp + fp
    actual_positive = tp + fn
    actual_negative = tn + fp

    return {
        "iou": 1.0 if union == 0 else tp / union,
        "dice": 1.0 if (2 * tp + fp + fn) == 0 else (2 * tp) / (2 * tp + fp + fn),
        "precision": 1.0 if predicted_positive == 0 and actual_positive == 0 else tp / max(predicted_positive, EPSILON),
        "recall": 1.0 if actual_positive == 0 and predicted_positive == 0 else tp / max(actual_positive, EPSILON),
        "specificity": tn / max(actual_negative, EPSILON),
        "pixel_accuracy": (tp + tn) / max(total, EPSILON),
    }


def save_image(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), image):
        raise RuntimeError(f"Failed to save image: {path}")


def add_title(image: np.ndarray, title: str) -> np.ndarray:
    bar_height = 42
    panel = cv2.copyMakeBorder(
        image,
        bar_height,
        0,
        0,
        0,
        borderType=cv2.BORDER_CONSTANT,
        value=(28, 28, 28),
    )
    cv2.putText(
        panel,
        title,
        (12, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return panel


def colorize_mask(mask: np.ndarray, color: tuple[int, int, int]) -> np.ndarray:
    output = np.zeros((*mask.shape, 3), dtype=np.uint8)
    output[mask] = color
    return output


def save_visualization(
    output_dir: Path,
    image_path: Path,
    prediction: np.ndarray,
    target: np.ndarray,
    image_metrics: dict[str, float],
) -> None:
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Cannot read image for visualization: {image_path}")

    height, width = target.shape
    image = cv2.resize(image, (width, height), interpolation=cv2.INTER_LINEAR)

    ground_truth_panel = colorize_mask(target, (0, 255, 0))
    prediction_panel = colorize_mask(prediction, (0, 255, 255))

    panels = [
        add_title(image, "Original"),
        add_title(ground_truth_panel, "Ground truth"),
        add_title(prediction_panel, "Prediction"),
    ]
    comparison = np.concatenate(panels, axis=1)

    footer_height = 44
    comparison = cv2.copyMakeBorder(
        comparison,
        0,
        footer_height,
        0,
        0,
        borderType=cv2.BORDER_CONSTANT,
        value=(28, 28, 28),
    )
    summary = (
        f"IoU {image_metrics['iou']:.4f}   "
        f"Dice {image_metrics['dice']:.4f}   "
        f"Precision {image_metrics['precision']:.4f}   "
        f"Recall {image_metrics['recall']:.4f}"
    )
    cv2.putText(
        comparison,
        summary,
        (12, comparison.shape[0] - 14),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    save_image(
        output_dir / "visualizations" / f"{image_path.stem}_comparison.jpg",
        comparison,
    )


def save_reports(
    output_dir: Path,
    summary: dict[str, Any],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    with (output_dir / "metrics.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2, ensure_ascii=False)


@torch.inference_mode()
def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    device = get_device(args.device)
    model, img_size = load_model(args.model, device, args.img_size)

    split_dir = args.data / args.split
    dataset = RippleDataset(
        image_dir=split_dir / "images",
        mask_dir=split_dir / "masks",
        transform=get_val_transforms(img_size),
    )
    if len(dataset) == 0:
        raise ValueError(f"No images found in: {split_dir / 'images'}")

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )
    criterion = BCEDiceLoss(alpha=0.5)

    total_loss = 0.0
    total_tp = total_fp = total_fn = total_tn = 0
    image_index = 0

    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        logits = model(images)
        predictions = torch.sigmoid(logits) >= args.threshold

        for batch_index in range(images.shape[0]):
            image_loss = float(
                criterion(
                    logits[batch_index : batch_index + 1],
                    targets[batch_index : batch_index + 1],
                ).item()
            )
            tp, fp, fn, tn = confusion_counts(
                predictions[batch_index], targets[batch_index] >= 0.5
            )
            image_metrics = metrics_from_counts(tp, fp, fn, tn)
            image_path = dataset.images[image_index]

            prediction_mask = predictions[batch_index, 0].cpu().numpy()
            target_mask = (targets[batch_index, 0] >= 0.5).cpu().numpy()

            if image_index < args.max_visualizations:
                save_visualization(
                    output_dir=args.output_dir,
                    image_path=image_path,
                    prediction=prediction_mask,
                    target=target_mask,
                    image_metrics=image_metrics,
                )

            total_loss += image_loss
            total_tp += tp
            total_fp += fp
            total_fn += fn
            total_tn += tn
            image_index += 1

    aggregate_metrics = metrics_from_counts(total_tp, total_fp, total_fn, total_tn)
    summary: dict[str, Any] = {
        "model": str(args.model.resolve()),
        "data": str(args.data.resolve()),
        "split": args.split,
        "device": str(device),
        "samples": len(dataset),
        "img_size": img_size,
        "threshold": args.threshold,
        "loss": round(total_loss / len(dataset), 6),
        **{key: round(value, 6) for key, value in aggregate_metrics.items()},
        "confusion_counts": {
            "true_positive": total_tp,
            "false_positive": total_fp,
            "false_negative": total_fn,
            "true_negative": total_tn,
        },
    }
    save_reports(args.output_dir, summary)
    return summary


def main() -> None:
    args = parse_args()
    summary = evaluate(args)

    print("\nU-Net evaluation")
    print(f"Model:     {summary['model']}")
    print(f"Split:     {summary['split']} ({summary['samples']} samples)")
    print(f"Device:    {summary['device']}")
    print(f"Image size:{summary['img_size']}")
    print(f"Loss:      {summary['loss']:.4f}")
    print(f"IoU:       {summary['iou']:.4f}")
    print(f"Dice:      {summary['dice']:.4f}")
    print(f"Precision: {summary['precision']:.4f}")
    print(f"Recall:    {summary['recall']:.4f}")
    print(f"Saved to:  {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()

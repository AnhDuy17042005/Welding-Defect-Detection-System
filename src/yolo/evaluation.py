"""
    Evaluate a YOLO segmentation checkpoint and export report metrics
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np


"""Matplotlib Cache Directory"""
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")


"""YOLO Import"""
from ultralytics import YOLO


"""Project Root"""
PROJECT_ROOT = Path(__file__).resolve().parents[2]


"""Config Imports"""
from configs.path import METRICS_DIR
from configs.yolo import (
    YOLO_BATCH_SIZE,
    YOLO_IMAGE_SIZE,
    YOLO_MODEL,
    YOLO_TRAIN_DATA,
    YOLO_WORKERS,
)


"""Default Output Directory"""
DEFAULT_OUTPUT_DIR = METRICS_DIR / "yolo"


def parse_args() -> argparse.Namespace:
    """
        Parse command line arguments for YOLO evaluation.

        Args:
            --model      : path to YOLO checkpoint
            --data       : path to YOLO data YAML
            --split      : val / test split
            --imgsz      : inference image size
            --batch      : validation batch size
            --conf       : confidence threshold
            --iou        : IoU threshold for NMS / evaluation
            --device     : cpu / cuda / auto
            --workers    : number of dataloader workers
            --output-dir : directory to save metrics
            --plots      : save validation plots
    """

    parser = argparse.ArgumentParser(
        description="Evaluate YOLO segmentation on a validation or test split."
    )

    """Input Arguments"""
    parser.add_argument("--model", type=Path, default=YOLO_MODEL)
    parser.add_argument("--data", type=Path, default=YOLO_TRAIN_DATA)
    parser.add_argument("--split", choices=("val", "test"), default="test")

    """Validation Arguments"""
    parser.add_argument("--imgsz", type=int, default=YOLO_IMAGE_SIZE)
    parser.add_argument("--batch", type=int, default=YOLO_BATCH_SIZE)

    """Metric Arguments"""
    parser.add_argument(
        "--conf",
        type=float,
        default=0.001,
        help="Low confidence is recommended when calculating PR curves and mAP.",
    )
    parser.add_argument("--iou", type=float, default=0.7)

    """Runtime Arguments"""
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--workers", type=int, default=YOLO_WORKERS)

    """Output Arguments"""
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)

    """Plot Argument"""
    parser.add_argument(
        "--plots",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save confusion matrices and PR/F1 curves.",
    )

    args = parser.parse_args()

    """Validate Image Size"""
    if args.imgsz < 32:
        parser.error("--imgsz must be at least 32")

    """Validate Batch Size"""
    if args.batch < 1:
        parser.error("--batch must be at least 1")

    """Validate Confidence Threshold"""
    if not 0.0 <= args.conf <= 1.0:
        parser.error("--conf must be between 0 and 1")

    """Validate IoU Threshold"""
    if not 0.0 <= args.iou <= 1.0:
        parser.error("--iou must be between 0 and 1")

    return args


def to_builtin(value: Any) -> Any:
    """
        Convert values to JSON-compatible Python types.

        Args:
            value: any value from YOLO metrics

        Returns:
            value converted to built-in Python type
    """

    """Convert Dictionary"""
    if isinstance(value, dict):
        return {str(key): to_builtin(item) for key, item in value.items()}

    """Convert List Or Tuple"""
    if isinstance(value, (list, tuple)):
        return [to_builtin(item) for item in value]

    """Convert Numpy Array"""
    if isinstance(value, np.ndarray):
        return value.tolist()

    """Convert Numpy Scalar"""
    if isinstance(value, np.generic):
        return value.item()

    """Convert Path"""
    if isinstance(value, Path):
        return str(value)

    """Return Original Value"""
    return value


def metric_summary(metric: Any) -> dict[str, float]:
    """
        Build aggregate metrics for bbox or mask result.

        Metrics:
            precision : mean precision
            recall    : mean recall
            map50     : mAP at IoU 0.50
            map75     : mAP at IoU 0.75
            map50_95  : mAP from IoU 0.50 to 0.95
    """

    """Return Main Metrics"""
    return {
        "precision": float(metric.mp),
        "recall": float(metric.mr),
        "map50": float(metric.map50),
        "map75": float(metric.map75),
        "map50_95": float(metric.map),
    }


def build_per_class_rows(metrics: Any) -> list[dict[str, Any]]:
    """
        Build per-class metrics for YOLO segmentation.

        Args:
            metrics: YOLO validation metrics object

        Returns:
            list of per-class metric rows
    """

    """Map Class ID To Metric Position"""
    class_positions = {
        int(class_id): index
        for index, class_id in enumerate(metrics.ap_class_index)
    }

    """Per-class Rows"""
    rows: list[dict[str, Any]] = []

    """Loop Through All Dataset Classes"""
    for class_id, class_name in sorted(metrics.names.items()):
        """Find Class Position In Evaluated Classes"""
        position = class_positions.get(int(class_id))

        """Read Class Image And Instance Counts"""
        instances = int(metrics.nt_per_class[class_id])
        images = int(metrics.nt_per_image[class_id])

        """Base Class Information"""
        row: dict[str, Any] = {
            "class_id": int(class_id),
            "class_name": str(class_name),
            "images": images,
            "instances": instances,
            "evaluated": position is not None,
        }

        """Class Not Evaluated"""
        if position is None:
            row.update(
                {
                    "box_precision": None,
                    "box_recall": None,
                    "box_f1": None,
                    "box_map50": None,
                    "box_map50_95": None,
                    "mask_precision": None,
                    "mask_recall": None,
                    "mask_f1": None,
                    "mask_map50": None,
                    "mask_map50_95": None,
                }
            )

        else:
            """Get Box Metrics For Current Class"""
            box_precision, box_recall, box_map50, box_map = (
                metrics.box.class_result(position)
            )

            """Get Mask Metrics For Current Class"""
            mask_precision, mask_recall, mask_map50, mask_map = (
                metrics.seg.class_result(position)
            )

            """Add Box And Mask Metrics"""
            row.update(
                {
                    "box_precision": float(box_precision),
                    "box_recall": float(box_recall),
                    "box_f1": float(metrics.box.f1[position]),
                    "box_map50": float(box_map50),
                    "box_map50_95": float(box_map),
                    "mask_precision": float(mask_precision),
                    "mask_recall": float(mask_recall),
                    "mask_f1": float(metrics.seg.f1[position]),
                    "mask_map50": float(mask_map50),
                    "mask_map50_95": float(mask_map),
                }
            )

        """Save Class Row"""
        rows.append(row)

    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """
        Write metric rows to CSV file.

        Args:
            path: output CSV path
            rows: metric rows
    """

    """Skip Empty Rows"""
    if not rows:
        return

    """Write CSV"""
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    """
        Evaluate YOLO segmentation model on validation or test split.

        Pipeline:
            1. Parse arguments
            2. Check model and data YAML
            3. Load YOLO checkpoint
            4. Run YOLO validation
            5. Collect overall metrics
            6. Collect per-class metrics
            7. Save JSON and CSV reports
    """

    """Parse Arguments"""
    args = parse_args()

    """Resolve Paths"""
    model_path = args.model.resolve()
    data_path = args.data.resolve()
    output_dir = args.output_dir.resolve()

    """Check YOLO Checkpoint"""
    if not model_path.is_file():
        raise FileNotFoundError(f"YOLO checkpoint not found: {model_path}")

    """Check YOLO Data YAML"""
    if not data_path.is_file():
        raise FileNotFoundError(f"YOLO data YAML not found: {data_path}")

    """Create Output Directory"""
    output_dir.mkdir(parents=True, exist_ok=True)

    """Load YOLO Model"""
    model = YOLO(str(model_path))

    """Validation Arguments"""
    validation_args: dict[str, Any] = {
        "data": str(data_path),
        "split": args.split,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "conf": args.conf,
        "iou": args.iou,
        "workers": args.workers,
        "plots": args.plots,
        "project": str(output_dir.parent),
        "name": output_dir.name,
        "exist_ok": True,
        "verbose": True,
    }

    """Set Device If Provided"""
    if args.device:
        validation_args["device"] = args.device

    """Start Timer"""
    started = time.perf_counter()

    """Run YOLO Validation"""
    metrics = model.val(**validation_args)

    """Elapsed Time"""
    elapsed_seconds = time.perf_counter() - started

    """Build Per-class Metrics"""
    per_class = build_per_class_rows(metrics)

    """Build Evaluation Report"""
    report = {
        "model_type": "YOLO segmentation",
        "model": str(model_path),
        "data": str(data_path),
        "split": args.split,

        # Evaluation Settings
        "settings": {
            "imgsz": args.imgsz,
            "batch": args.batch,
            "conf": args.conf,
            "iou": args.iou,
            "device": args.device or "auto",
        },

        # Overall Metrics
        "overall": {
            "bbox": metric_summary(metrics.box),
            "mask": metric_summary(metrics.seg),
            "fitness": float(metrics.fitness),
        },

        # Speed Metrics
        "speed_ms_per_image": to_builtin(metrics.speed),
        "elapsed_seconds": round(elapsed_seconds, 3),

        # Per-class Metrics
        "per_class": per_class,
    }

    """Save JSON Report"""
    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(
        json.dumps(to_builtin(report), indent=2, ensure_ascii=True),
        encoding="utf-8",
    )

    """Save CSV Report"""
    write_csv(output_dir / "per_class_metrics.csv", per_class)

    """Print Summary"""
    print("\nYOLO evaluation complete")
    print(f"Model: {model_path}")
    print(f"Split: {args.split}")
    print(f"Metrics: {metrics_path}")
    print(
        "Mask: "
        f"P={metrics.seg.mp:.4f} R={metrics.seg.mr:.4f} "
        f"mAP50={metrics.seg.map50:.4f} mAP50-95={metrics.seg.map:.4f}"
    )


"""Run Main"""
if __name__ == "__main__":
    main()
"""
    Export YOLO segmentation checkpoint to ONNX.

    Default:
        models/runs/train_ver5/weights/best.pt

    Run:
        python -m src.onnx.yolo_export
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


"""Project Root"""
PROJECT_ROOT = Path(__file__).resolve().parents[2]

"""Support direct script run from the project root."""
if __package__ in (None, ""):
    sys.path.insert(0, str(PROJECT_ROOT))

"""Matplotlib Cache Directory"""
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

from ultralytics import YOLO
from configs.yolo import YOLO_IMAGE_SIZE, YOLO_MODEL


def parse_args() -> argparse.Namespace:
    """
        Parse command line arguments for YOLO ONNX export.

        Args:
            --model    : path to trained YOLO checkpoint
            --imgsz    : input image size for export
            --opset    : ONNX opset version
            --batch    : export batch size
            --device   : cpu / cuda device for export
            --output   : optional output ONNX path
            --simplify : simplify ONNX graph
            --dynamic  : enable dynamic input shape
    """

    parser = argparse.ArgumentParser(
        description="Export trained YOLOv11 segmentation model to ONNX."
    )

    """Input Argument"""
    parser.add_argument("--model", type=Path, default=YOLO_MODEL)

    """Export Arguments"""
    parser.add_argument("--imgsz", type=int, default=YOLO_IMAGE_SIZE)
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--batch", type=int, default=1)

    """Runtime Argument"""
    parser.add_argument("--device", type=str, default="cpu")

    """Output Argument"""
    parser.add_argument("--output", type=Path, default=None)

    """Optimization Arguments"""
    parser.add_argument("--simplify", action="store_true")
    parser.add_argument("--dynamic", action="store_true")

    args = parser.parse_args()

    """Validate Image Size"""
    if args.imgsz < 32:
        parser.error("--imgsz must be at least 32")

    """Validate Batch Size"""
    if args.batch < 1:
        parser.error("--batch must be at least 1")

    """Validate ONNX Opset"""
    if args.opset < 12:
        parser.error("--opset must be at least 12")

    return args


def export_yolo_to_onnx(
    model_path: Path,
    imgsz: int,
    opset: int,
    batch: int,
    device: str,
    simplify: bool,
    dynamic: bool,
    output_path: Path | None,
) -> Path:
    """
        Export trained YOLO segmentation checkpoint to ONNX format.

        Args:
            model_path : path to YOLO checkpoint
            imgsz      : input image size
            opset      : ONNX opset version
            batch      : export batch size
            device     : cpu / cuda device
            simplify   : simplify ONNX graph
            dynamic    : enable dynamic input shape
            output_path: optional custom output path

        Returns:
            exported_path: path to exported ONNX file
    """

    """Check YOLO Checkpoint"""
    if not model_path.is_file():
        raise FileNotFoundError(f"YOLO checkpoint not found: {model_path}")

    """Load YOLO Model"""
    model = YOLO(str(model_path))

    """Export YOLO To ONNX"""
    exported_path = Path(
        model.export(
            format="onnx",
            imgsz=imgsz,
            opset=opset,
            batch=batch,
            device=device,
            simplify=simplify,
            dynamic=dynamic,
        )
    )

    """Move Exported File To Custom Output Path"""
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        exported_path.replace(output_path)
        exported_path = output_path

    return exported_path


def main() -> None:
    """
        Main function for YOLO ONNX export.

        Pipeline:
            1. Parse arguments
            2. Check YOLO checkpoint
            3. Load YOLO model
            4. Export model to ONNX
            5. Optionally move ONNX file to custom output path
            6. Print export information
    """
    args = parse_args()

    """Export YOLO To ONNX"""
    output_path = export_yolo_to_onnx(
        model_path=args.model,
        imgsz=args.imgsz,
        opset=args.opset,
        batch=args.batch,
        device=args.device,
        simplify=args.simplify,
        dynamic=args.dynamic,
        output_path=args.output,
    )

    """Print Export Summary"""
    print(f"YOLO ONNX exported: {output_path}")

if __name__ == "__main__":
    main()
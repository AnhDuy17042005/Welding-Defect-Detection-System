"""
    Export U-Net ripple segmentation checkpoint to OpenVINO.

    Default:
        models/unet/train_ver3/best.pth

    Run:
        python -m src.openvino.unet_export
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import openvino as ov
import torch

"""Project Root"""
PROJECT_ROOT = Path(__file__).resolve().parents[2]

"""Support direct script run from the project root."""
if __package__ in (None, ""):
    sys.path.insert(0, str(PROJECT_ROOT))

from configs.unet import (
    UNET_DEVICE,
    UNET_IMAGE_SIZE,
    UNET_MODEL_VERSION,
    UNET_RUNS_DIR,
)
from src.unet.inference import get_device, load_model


def default_model_path() -> Path:
    """
        Build default U-Net PyTorch checkpoint path.
    """

    return UNET_RUNS_DIR / f"train_ver{UNET_MODEL_VERSION}" / "best.pth"


def default_output_path(model_path: Path) -> Path:
    """
        Build default OpenVINO IR output path from checkpoint path.

        Example:
            best.pth -> best.xml
    """

    return model_path.with_suffix(".xml")


def parse_args() -> argparse.Namespace:
    """
        Parse command line arguments for U-Net OpenVINO export.
    """

    parser = argparse.ArgumentParser(
        description="Export trained U-Net ripple segmentation model to OpenVINO."
    )

    """Input Arguments"""
    parser.add_argument("--model", type=Path, default=default_model_path())
    parser.add_argument("--output", type=Path, default=None)

    """Export Arguments"""
    parser.add_argument("--img-size", type=int, default=None)

    """Runtime Arguments"""
    parser.add_argument("--device", type=str, default=UNET_DEVICE)

    args = parser.parse_args()

    """Validate Image Size"""
    if args.img_size is not None and args.img_size < 32:
        parser.error("--img-size must be at least 32")

    return args


def export_unet_to_openvino(
    model_path: Path,
    output_path: Path | None,
    img_size: int | None,
    device_name: str,
) -> tuple[Path, int]:
    """
        Export trained U-Net checkpoint or ONNX model to OpenVINO IR format.

        Returns:
            output_path: exported .xml path
            input_size : model input size used for export
    """

    """Check Model Path"""
    if not model_path.is_file():
        raise FileNotFoundError(f"U-Net model not found: {model_path}")

    """Prepare Output Path"""
    output_path = output_path or default_output_path(model_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    """Convert ONNX Model Directly To OpenVINO IR"""
    if model_path.suffix.lower() == ".onnx":
        input_size = img_size or UNET_IMAGE_SIZE
        openvino_model = ov.convert_model(str(model_path))
        ov.save_model(openvino_model, output_path)

        return output_path, input_size

    """Load Device"""
    device = get_device(device_name)

    """Load U-Net Model"""
    model, input_size = load_model(model_path, device, img_size)

    """Set Evaluation Mode"""
    model.eval()

    """Create Example Input"""
    example_input = torch.randn(
        1,
        3,
        input_size,
        input_size,
        device=device,
        dtype=torch.float32,
    )

    """Convert Model To OpenVINO IR"""
    openvino_model = ov.convert_model(
        model,
        example_input=example_input,
    )

    """Save OpenVINO IR"""
    ov.save_model(openvino_model, output_path)

    return output_path, input_size


def main() -> None:
    """
        Main function for U-Net OpenVINO export.
    """

    args = parse_args()

    """Export U-Net To OpenVINO"""
    output_path, input_size = export_unet_to_openvino(
        model_path=args.model,
        output_path=args.output,
        img_size=args.img_size,
        device_name=args.device,
    )

    """Print Export Summary"""
    print(f"U-Net OpenVINO exported: {output_path}")
    print(f"Input size: {input_size}x{input_size}")


if __name__ == "__main__":
    main()

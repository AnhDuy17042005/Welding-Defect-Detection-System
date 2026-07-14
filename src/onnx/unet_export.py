"""
    Export U-Net ripple segmentation checkpoint to ONNX.

    Default:
        models/unet/train_ver3/best.pth

    Run:
        python -m src.onnx.unet_export
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

"""Project Root"""
PROJECT_ROOT = Path(__file__).resolve().parents[2]

"""Support direct script run from the project root."""
if __package__ in (None, ""):
    sys.path.insert(0, str(PROJECT_ROOT))

from configs.unet import UNET_DEVICE, UNET_MODEL
from src.unet.inference import get_device, load_model


def parse_args() -> argparse.Namespace:
    """
        Parse command line arguments for U-Net ONNX export.

        Args:
            --model    : path to trained U-Net checkpoint
            --output   : output ONNX file path
            --img-size : optional input image size
            --opset    : ONNX opset version
            --device   : cpu / cuda / auto
            --dynamic  : enable dynamic batch size
    """

    parser = argparse.ArgumentParser(
        description="Export trained U-Net ripple segmentation model to ONNX."
    )

    """Input Arguments"""
    parser.add_argument("--model", type=Path, default=UNET_MODEL)
    parser.add_argument("--output", type=Path, default=None)

    """Export Arguments"""
    parser.add_argument("--img-size", type=int, default=None)
    parser.add_argument("--opset", type=int, default=17)

    """Runtime Arguments"""
    parser.add_argument("--device", type=str, default=UNET_DEVICE)

    """Dynamic Batch Argument"""
    parser.add_argument("--dynamic", action="store_true")

    args = parser.parse_args()

    """Validate Image Size"""
    if args.img_size is not None and args.img_size < 32:
        parser.error("--img-size must be at least 32")

    """Validate ONNX Opset"""
    if args.opset < 12:
        parser.error("--opset must be at least 12")

    return args


def default_output_path(model_path: Path) -> Path:
    """
        Build default ONNX output path from checkpoint path.

        Example:
            best.pth -> best.onnx
    """

    """Replace Checkpoint Suffix With .onnx"""
    return model_path.with_suffix(".onnx")


def export_unet_to_onnx(
    model_path: Path,
    output_path: Path | None,
    img_size: int | None,
    opset: int,
    device_name: str,
    dynamic: bool,
) -> tuple[Path, int]:
    """
        Export trained U-Net checkpoint to ONNX format.

        Args:
            model_path : path to U-Net checkpoint
            output_path: output ONNX path
            img_size   : optional export input size
            opset      : ONNX opset version
            device_name: cpu / cuda / auto
            dynamic    : enable dynamic batch axis

        Returns:
            output_path: exported ONNX file path
            input_size : model input size used for export
    """

    """Check Model Checkpoint"""
    if not model_path.is_file():
        raise FileNotFoundError(f"U-Net checkpoint not found: {model_path}")

    """Load Device"""
    device = get_device(device_name)

    """Load U-Net Model"""
    model, input_size = load_model(model_path, device, img_size)

    """Set Evaluation Mode"""
    model.eval()

    """Prepare Output Path"""
    output_path = output_path or default_output_path(model_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    """Create Dummy Input"""
    dummy_input = torch.randn(
        1,
        3,
        input_size,
        input_size,
        device=device,
        dtype=torch.float32,
    )

    """Default Static Axes"""
    dynamic_axes = None

    """Enable Dynamic Batch Axis"""
    if dynamic:
        dynamic_axes = {
            "image": {0: "batch"},
            "logits": {0: "batch"},
        }

    """Export Model To ONNX"""
    torch.onnx.export(
        model,
        dummy_input,
        str(output_path),
        export_params=True,
        opset_version=opset,
        do_constant_folding=True,
        input_names=["image"],
        output_names=["logits"],
        dynamic_axes=dynamic_axes,
        dynamo=False,
    )

    return output_path, input_size


def main() -> None:
    """
        Main function for U-Net ONNX export.

        Pipeline:
            1. Parse arguments
            2. Load trained U-Net checkpoint
            3. Create dummy input
            4. Export model to ONNX
            5. Print export information
    """
    args = parse_args()

    """Export U-Net To ONNX"""
    output_path, input_size = export_unet_to_onnx(
        model_path=args.model,
        output_path=args.output,
        img_size=args.img_size,
        opset=args.opset,
        device_name=args.device,
        dynamic=args.dynamic,
    )

    """Print Export Summary"""
    print(f"U-Net ONNX exported: {output_path}")
    print(f"Input size: {input_size}x{input_size}")

if __name__ == "__main__":
    main()
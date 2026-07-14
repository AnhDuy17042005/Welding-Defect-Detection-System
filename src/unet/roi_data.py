"""
    Crop ripple dataset images to the foreground bounding box of each mask.

    Purpose:
        Convert full-size ripple segmentation data into ROI-based data.
        Each image and its mask are cropped by the foreground region in the mask.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]

"""Support direct script run from the project root."""
if __package__ in (None, ""):
    sys.path.insert(0, str(PROJECT_ROOT))

from configs.data import IMAGE_EXTENSIONS, RIPPLE_ROI_DATASET, RIPPLE_SPLIT_DATASET, SPLITS
from configs.unet import UNET_ROI_DATA_MARGIN


def parse_args() -> argparse.Namespace:
    """
        Parse command line arguments.

        Args:
            --input-dir:
                Root folder of the original ripple dataset.

            --output-dir:
                Root folder where cropped ROI dataset will be saved.

            --margin:
                Extra padding around the mask bounding box.
    """

    parser = argparse.ArgumentParser(
        description="Crop ripple images and masks using the mask foreground bounding box."
    )

    parser.add_argument("--input-dir", type=Path, default=RIPPLE_SPLIT_DATASET)
    parser.add_argument("--output-dir", type=Path, default=RIPPLE_ROI_DATASET)

    parser.add_argument(
        "--margin",
        type=float,
        default=UNET_ROI_DATA_MARGIN,
        help="BBox expansion on each side as a fraction of its width or height.",
    )

    args = parser.parse_args()

    """Validate crop margin"""
    if not 0.0 <= args.margin <= 1.0:
        parser.error("--margin must be between 0 and 1")

    """Prevent overwriting the original dataset"""
    if args.input_dir.resolve() == args.output_dir.resolve():
        parser.error("--output-dir must be different from --input-dir")

    return args


def find_images(image_dir: Path) -> list[Path]:
    """
        Find all valid image files in a directory.
    """

    """Check image directory"""
    if not image_dir.is_dir():
        raise FileNotFoundError(f"Image directory not found: {image_dir}")

    """Return sorted image paths with supported extensions"""
    return sorted(
        path
        for path in image_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def foreground_bbox(
    mask: np.ndarray,
    margin: float,
) -> tuple[int, int, int, int] | None:
    """
        Compute foreground bounding box from a binary mask.

        Foreground is defined as pixels with value > 0.

        Returns:
            (x1, y1, x2, y2) if foreground exists.
            None if mask is empty.
    """

    """Handle both grayscale mask and multi-channel mask"""
    if mask.ndim == 3:
        foreground = np.any(mask > 0, axis=2)
    else:
        foreground = mask > 0

    """Get foreground pixel coordinates"""
    ys, xs = np.where(foreground)

    """Skip empty mask"""
    if len(xs) == 0:
        return None

    """Compute tight bounding box"""
    x1 = int(xs.min())
    y1 = int(ys.min())
    x2 = int(xs.max()) + 1
    y2 = int(ys.max()) + 1

    """Compute margin size"""
    box_width = x2 - x1
    box_height = y2 - y1

    margin_x = int(round(box_width * margin))
    margin_y = int(round(box_height * margin))

    height, width = mask.shape[:2]

    """Expand bbox with margin and clamp to image boundary"""
    return (
        max(0, x1 - margin_x),
        max(0, y1 - margin_y),
        min(width, x2 + margin_x),
        min(height, y2 + margin_y),
    )


def save_image(path: Path, image: np.ndarray) -> None:
    """
        Save image to disk and raise error if saving fails.
    """

    """Create parent directory if needed"""
    path.parent.mkdir(parents=True, exist_ok=True)

    """Save image"""
    if not cv2.imwrite(str(path), image):
        raise RuntimeError(f"Failed to save: {path}")


def crop_split(
    input_root: Path,
    output_root: Path,
    split: str,
    margin: float,
) -> tuple[int, int]:
    """
        Crop all image-mask pairs in one dataset split.

        Expected input structure:
            input_root/split/images
            input_root/split/masks

        Output structure:
            output_root/split/images
            output_root/split/masks
    """

    """Input directories"""
    image_dir = input_root / split / "images"
    mask_dir = input_root / split / "masks"

    """Output directories"""
    output_image_dir = output_root / split / "images"
    output_mask_dir = output_root / split / "masks"

    """Check mask directory"""
    if not mask_dir.is_dir():
        raise FileNotFoundError(f"Mask directory not found: {mask_dir}")

    processed = 0
    skipped = 0

    """Process each image-mask pair"""
    for image_path in find_images(image_dir):
        mask_path = mask_dir / f"{image_path.stem}.png"

        """Each image must have a corresponding mask"""
        if not mask_path.exists():
            raise FileNotFoundError(f"Mask not found for {image_path.name}: {mask_path}")

        """Load image and mask"""
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)

        """Validate image and mask loading"""
        if image is None:
            raise ValueError(f"Cannot read image: {image_path}")

        if mask is None:
            raise ValueError(f"Cannot read mask: {mask_path}")

        """Image and mask must have the same spatial size"""
        if image.shape[:2] != mask.shape[:2]:
            raise ValueError(
                f"Image and mask size differ for {image_path.name}: "
                f"{image.shape[:2]} != {mask.shape[:2]}"
            )

        """Compute ROI bbox from mask foreground"""
        bbox = foreground_bbox(mask, margin)

        """Skip image if mask is empty"""
        if bbox is None:
            print(f"[skip] Empty mask: {mask_path}")
            skipped += 1
            continue

        """Crop image and mask by ROI bbox"""
        x1, y1, x2, y2 = bbox

        cropped_image = image[y1:y2, x1:x2]
        cropped_mask = mask[y1:y2, x1:x2]

        """Save cropped ROI pair"""
        save_image(output_image_dir / image_path.name, cropped_image)
        save_image(output_mask_dir / mask_path.name, cropped_mask)

        processed += 1

    return processed, skipped


def main() -> None:
    """
        Main function for cropping all dataset splits.
    """

    args = parse_args()

    total_processed = 0
    total_skipped = 0

    """Crop train, valid, and test splits"""
    for split in SPLITS:
        processed, skipped = crop_split(
            input_root=args.input_dir,
            output_root=args.output_dir,
            split=split,
            margin=args.margin,
        )

        total_processed += processed
        total_skipped += skipped

        print(f"{split}: cropped={processed}, skipped={skipped}")

    """Print final summary"""
    print(f"Output: {args.output_dir}")
    print(f"Total: cropped={total_processed}, skipped={total_skipped}")


if __name__ == "__main__":
    main()

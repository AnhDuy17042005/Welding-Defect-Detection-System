"""Crop ripple dataset images to the foreground bounding box of each mask."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


BASE_DIR = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_DIR = BASE_DIR / "dataset" / "ripple_split"
DEFAULT_OUTPUT_DIR = BASE_DIR / "dataset" / "ripple_roi"

SPLITS = ("train", "valid", "test")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Crop ripple images and masks using the mask foreground bounding box."
    )
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--margin",
        type=float,
        default=0.1,
        help="BBox expansion on each side as a fraction of its width or height.",
    )
    args = parser.parse_args()

    if not 0.0 <= args.margin <= 1.0:
        parser.error("--margin must be between 0 and 1")
    if args.input_dir.resolve() == args.output_dir.resolve():
        parser.error("--output-dir must be different from --input-dir")

    return args


def find_images(image_dir: Path) -> list[Path]:
    if not image_dir.is_dir():
        raise FileNotFoundError(f"Image directory not found: {image_dir}")

    return sorted(
        path
        for path in image_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def foreground_bbox(
    mask: np.ndarray,
    margin: float,
) -> tuple[int, int, int, int] | None:
    if mask.ndim == 3:
        foreground = np.any(mask > 0, axis=2)
    else:
        foreground = mask > 0

    ys, xs = np.where(foreground)
    if len(xs) == 0:
        return None

    x1 = int(xs.min())
    y1 = int(ys.min())
    x2 = int(xs.max()) + 1
    y2 = int(ys.max()) + 1

    box_width = x2 - x1
    box_height = y2 - y1
    margin_x = int(round(box_width * margin))
    margin_y = int(round(box_height * margin))
    height, width = mask.shape[:2]

    return (
        max(0, x1 - margin_x),
        max(0, y1 - margin_y),
        min(width, x2 + margin_x),
        min(height, y2 + margin_y),
    )


def save_image(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), image):
        raise RuntimeError(f"Failed to save: {path}")


def crop_split(
    input_root: Path,
    output_root: Path,
    split: str,
    margin: float,
) -> tuple[int, int]:
    image_dir = input_root / split / "images"
    mask_dir = input_root / split / "masks"
    output_image_dir = output_root / split / "images"
    output_mask_dir = output_root / split / "masks"

    if not mask_dir.is_dir():
        raise FileNotFoundError(f"Mask directory not found: {mask_dir}")

    processed = 0
    skipped = 0

    for image_path in find_images(image_dir):
        mask_path = mask_dir / f"{image_path.stem}.png"
        if not mask_path.exists():
            raise FileNotFoundError(f"Mask not found for {image_path.name}: {mask_path}")

        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        mask  = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
        if image is None:
            raise ValueError(f"Cannot read image: {image_path}")
        if mask is None:
            raise ValueError(f"Cannot read mask: {mask_path}")
        if image.shape[:2] != mask.shape[:2]:
            raise ValueError(
                f"Image and mask size differ for {image_path.name}: "
                f"{image.shape[:2]} != {mask.shape[:2]}"
            )

        bbox = foreground_bbox(mask, margin)
        if bbox is None:
            print(f"[skip] Empty mask: {mask_path}")
            skipped += 1
            continue

        x1, y1, x2, y2 = bbox
        save_image(output_image_dir / image_path.name, image[y1:y2, x1:x2])
        save_image(output_mask_dir / mask_path.name, mask[y1:y2, x1:x2])
        processed += 1

    return processed, skipped


def main() -> None:
    args = parse_args()
    total_processed = 0
    total_skipped = 0

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

    print(f"Output: {args.output_dir}")
    print(f"Total: cropped={total_processed}, skipped={total_skipped}")


if __name__ == "__main__":
    main()

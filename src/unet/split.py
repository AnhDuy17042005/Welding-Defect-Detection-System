"""
    Restructure ripple dataset for U-Net training.

    Purpose:
        Convert the original ripple dataset structure into U-Net format.

        Original structure:
            ripple/train/
                img001.jpg
                img001_mask.png

        New structure:
            ripple_split/train/
                images/img001.jpg
                masks/img001.png
"""

from pathlib import Path
import shutil
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
from configs.data import RIPPLE_SOURCE_DATASET, RIPPLE_SPLIT_DATASET


def copy_image_mask_pair(
    image_path: Path,
    mask_path: Path,
    output_image_dir: Path,
    output_mask_dir: Path,
):
    """
        Copy one image-mask pair to U-Net folder structure.

        Input:
            img001.jpg
            img001_mask.png

        Output:
            images/img001.jpg
            masks/img001.png
    """

    """Create output folders if they do not exist"""
    output_image_dir.mkdir(parents=True, exist_ok=True)
    output_mask_dir.mkdir(parents=True, exist_ok=True)

    """Keep original image name"""
    output_image_path = output_image_dir / image_path.name

    """Rename mask to match image stem"""
    output_mask_path = output_mask_dir / f"{image_path.stem}.png"

    """Copy image and mask files"""
    shutil.copy2(image_path, output_image_path)
    shutil.copy2(mask_path, output_mask_path)


def restructure_ripple_dataset(
    input_root: str | Path,
    output_root: str | Path,
    subsets=("train", "valid", "test"),
    image_exts=(".jpg", ".jpeg"),
    mask_suffix: str = "_mask",
    mask_ext: str = ".png",
):
    """
        Restructure ripple dataset into U-Net format.

        Input structure:
            ripple/train/
                img001.jpg
                img001_mask.png

        Output structure:
            ripple_split/train/
                images/img001.jpg
                masks/img001.png

        Args:
            input_root:
                Original dataset root folder.

            output_root:
                Output dataset root folder.

            subsets:
                Dataset subsets to process.

            image_exts:
                Supported image extensions.

            mask_suffix:
                Suffix used by mask files.

            mask_ext:
                Mask file extension.
    """

    """Convert input and output paths to Path objects"""
    input_root = Path(input_root)
    output_root = Path(output_root)

    """Check input dataset folder"""
    if not input_root.exists():
        raise FileNotFoundError(f"Input dataset folder not found: {input_root}")

    """Create output dataset folder"""
    output_root.mkdir(parents=True, exist_ok=True)

    """Process each dataset subset"""
    for subset in subsets:
        input_subset_dir = input_root / subset

        """Skip subset if folder does not exist"""
        if not input_subset_dir.exists():
            print(f"Skip {subset}: folder not found")
            continue

        """Output folders for images and masks"""
        output_image_dir = output_root / subset / "images"
        output_mask_dir = output_root / subset / "masks"

        image_paths = []

        """Collect image files by extension"""
        for ext in image_exts:
            image_paths.extend(input_subset_dir.glob(f"*{ext}"))

        """Sort image paths for stable processing order"""
        image_paths = sorted(image_paths)

        copied = 0
        missing = 0

        """Process each image and find corresponding mask"""
        for image_path in image_paths:
            mask_path = input_subset_dir / f"{image_path.stem}{mask_suffix}{mask_ext}"

            """Skip image if corresponding mask is missing"""
            if not mask_path.exists():
                print(f"Missing mask for: {image_path.name}")
                print(f"Expected mask : {mask_path.name}")
                missing += 1
                continue

            """Copy image-mask pair to U-Net structure"""
            copy_image_mask_pair(
                image_path=image_path,
                mask_path=mask_path,
                output_image_dir=output_image_dir,
                output_mask_dir=output_mask_dir,
            )

            copied += 1

        """Print subset summary"""
        print(f"{subset}: copied {copied} pairs, missing {missing} masks")

    """Print final summary"""
    print("Done restructuring dataset.")
    print(f"Output saved to: {output_root}")


if __name__ == "__main__":
    """
        Run dataset restructuring script.
    """

    restructure_ripple_dataset(
        input_root=RIPPLE_SOURCE_DATASET,
        output_root=RIPPLE_SPLIT_DATASET,
    )

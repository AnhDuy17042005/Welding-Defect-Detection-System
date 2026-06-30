from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parent.parent.parent

input_dataset  = ROOT / "dataset" / "ripple"
output_dataset = ROOT / "dataset" / "ripple_split"

def copy_image_mask_pair(
    image_path: Path,
    mask_path: Path,
    output_image_dir: Path,
    output_mask_dir: Path,
):
    """
    Copy image and mask to U-Net folder structure.

    Input:
        img001.jpg
        img001_mask.png

    Output:
        images/img001.jpg
        masks/img001.png
    """

    output_image_dir.mkdir(parents=True, exist_ok=True)
    output_mask_dir.mkdir(parents=True, exist_ok=True)

    output_image_path = output_image_dir / image_path.name

    # Rename mask to match image stem.
    output_mask_path = output_mask_dir / f"{image_path.stem}.png"

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
    Restructure dataset from:

        ripple/train/
            img001.jpg
            img001_mask.png

    To:

        ripple_unet/train/
            images/img001.jpg
            masks/img001.png
    """

    input_root = Path(input_root)
    output_root = Path(output_root)

    if not input_root.exists():
        raise FileNotFoundError(f"Input dataset folder not found: {input_root}")

    output_root.mkdir(parents=True, exist_ok=True)

    for subset in subsets:
        input_subset_dir = input_root / subset

        if not input_subset_dir.exists():
            print(f"Skip {subset}: folder not found")
            continue

        output_image_dir = output_root / subset / "images"
        output_mask_dir = output_root / subset / "masks"

        image_paths = []

        for ext in image_exts:
            image_paths.extend(input_subset_dir.glob(f"*{ext}"))

        image_paths = sorted(image_paths)

        copied = 0
        missing = 0

        for image_path in image_paths:
            mask_path = input_subset_dir / f"{image_path.stem}{mask_suffix}{mask_ext}"

            if not mask_path.exists():
                print(f"Missing mask for: {image_path.name}")
                print(f"Expected mask : {mask_path.name}")
                missing += 1
                continue

            copy_image_mask_pair(
                image_path=image_path,
                mask_path=mask_path,
                output_image_dir=output_image_dir,
                output_mask_dir=output_mask_dir,
            )

            copied += 1

        print(f"{subset}: copied {copied} pairs, missing {missing} masks")

    print("Done restructuring dataset.")
    print(f"Output saved to: {output_root}")


if __name__ == "__main__":

    restructure_ripple_dataset(
        input_root=input_dataset,
        output_root=output_dataset,
    )
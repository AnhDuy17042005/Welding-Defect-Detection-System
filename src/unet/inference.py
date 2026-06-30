from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

BASE_DIR = Path(__file__).resolve().parent.parent.parent
if __package__ in (None, ""):
    sys.path.insert(0, str(BASE_DIR))
    from src.unet.post_processing import post_process_mask
    from src.unet.unet import build
else:
    from .post_processing import post_process_mask
    from .unet import build

DEFAULT_IMAGE_PATH = BASE_DIR / "dataset" / "test" / "26.jpg"

DEFAULT_MODEL_PATH = BASE_DIR / "models" / "unet" / "train_ver2" / "best.pth"
DEFAULT_OUTPUT_DIR = BASE_DIR / "output" / "unet2"

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def parse_args() -> argparse.Namespace:
    """
        Parse command line arguments.
    """

    parser = argparse.ArgumentParser(description="Predict ripple mask with U-Net.")

    parser.add_argument("--image", type=Path, default=DEFAULT_IMAGE_PATH)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)

    parser.add_argument("--img-size", type=int, default=None)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--device", type=str, default="auto")

    return parser.parse_args()


def load_image(image_path: Path) -> np.ndarray:
    """
        Load image with OpenCV.
    """

    image = cv2.imread(str(image_path))

    if image is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")

    return image


def get_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    return torch.device(device)


def load_model(model_path: Path, device: torch.device, img_size: int | None):
    """
        Load U-Net checkpoint saved by unet/train.py.
    """

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    checkpoint = torch.load(model_path, map_location=device)

    if isinstance(checkpoint, dict) and "model" in checkpoint:
        state_dict = checkpoint["model"]
        train_args = checkpoint.get("args", {}) or {}
    else:
        state_dict = checkpoint
        train_args = {}

    first_conv = state_dict.get("enc1.block.0.weight")
    base_channel = int(first_conv.shape[0]) if first_conv is not None else 64
    input_size = img_size or int(train_args.get("img_size", 256))

    model = build(in_channels=3, num_classes=1, base_channels=base_channel)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    return model, input_size


def preprocess(image: np.ndarray, img_size: int, device: torch.device) -> torch.Tensor:
    """
        BGR OpenCV image to normalized NCHW tensor.
    """

    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (img_size, img_size), interpolation=cv2.INTER_LINEAR)
    normalized = (resized.astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
    tensor = torch.from_numpy(normalized.transpose(2, 0, 1)).unsqueeze(0)

    return tensor.to(device)


@torch.no_grad()
def predict(
    model: torch.nn.Module,
    image: np.ndarray,
    img_size: int,
    threshold: float,
    device: torch.device,
) -> np.ndarray:
    """
        Run U-Net prediction on one image and return a binary mask.
    """

    h, w = image.shape[:2]
    tensor = preprocess(image, img_size, device)

    logits = model(tensor)
    prob = torch.sigmoid(logits).squeeze().cpu().numpy()
    prob = cv2.resize(prob, (w, h), interpolation=cv2.INTER_LINEAR)

    return (prob >= threshold).astype(np.uint8) * 255


def make_overlay(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
        Overlay predicted mask on the original image.
    """

    color = np.zeros_like(image)
    color[mask > 0] = (0, 255, 0)

    overlay = image.copy()
    active = mask > 0
    overlay[active] = cv2.addWeighted(image, 0.65, color, 0.35, 0)[active]

    return overlay


def save_image(path: Path, image: np.ndarray) -> None:
    """
        Save image and raise error if saving fails.
    """

    success = cv2.imwrite(str(path), image)

    if not success:
        raise RuntimeError(f"Failed to save image: {path}")


def save_outputs(output_dir: Path, image_path: Path, image: np.ndarray, mask: np.ndarray) -> None:
    """
        Save predicted mask and overlay.
    """

    output_dir.mkdir(parents=True, exist_ok=True)

    stem = image_path.stem
    # save_image(output_dir / f"{stem}_mask.png", mask)
    save_image(output_dir / f"{stem}_prediction.jpg", make_overlay(image, mask))


def main() -> None:
    args = parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Image: {args.image}")
    print(f"Model: {args.model}")

    device = get_device(args.device)
    image = load_image(args.image)
    model, img_size = load_model(args.model, device, args.img_size)

    mask = predict(
        model=model,
        image=image,
        img_size=img_size,
        threshold=args.threshold,
        device=device,
    )
    mask = post_process_mask(mask)

    save_outputs(
        output_dir=args.output_dir,
        image_path=args.image,
        image=image,
        mask=mask,
    )

    print(f"Image size: {img_size}")
    print(f"Saved outputs to: {args.output_dir}")


if __name__ == "__main__":
    main()

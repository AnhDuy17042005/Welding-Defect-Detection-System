"""
    Hybrid inference pipeline: YOLO + U-Net.

    Purpose:
        1. Use YOLO to detect welding_line and defect boxes.
        2. Crop welding_line ROI from YOLO bounding box.
        3. Use U-Net to segment ripple inside each welding_line ROI.
        4. Post-process ripple mask and paste it back to the original image.
        5. Draw defect boxes and save final overlay result.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

"""Set matplotlib cache directory for environments like Colab/Linux server"""
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import cv2
import numpy as np
import torch
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parents[2]

from configs.data import WELDING_LINE_CLASS
from configs.hybrid import (
    HYBRID_DEFAULT_IMAGE,
    HYBRID_OUTPUT,
    MAX_ROI_MARGIN,
    MIN_ROI_MARGIN,
    ROI_MARGIN,
)
from configs.unet import (
    UNET_DEVICE,
    UNET_IMAGE_SIZE,
    UNET_MODEL,
    UNET_THRESHOLD,
)
from configs.visualize import BOX_COLORS, RIPPLE_COLOR
from configs.yolo import (
    YOLO_CONFIDENCE_THRESHOLD,
    YOLO_IMAGE_SIZE,
    YOLO_IOU_THRESHOLD,
    YOLO_MASK_THRESHOLD,
    YOLO_MODEL,
)
from src.unet.inference import get_device, load_model, predict as predict_unet
from src.unet.post_processing import post_process_mask


def parse_args() -> argparse.Namespace:
    """
        Parse command line arguments.

        Main arguments:
            --image:
                Input image path.

            --yolo-model:
                YOLO model checkpoint path.

            --unet-model:
                U-Net model checkpoint path.

            --output-dir:
                Directory to save hybrid inference result.

            --yolo-imgsz:
                Image size used for YOLO inference.

            --unet-img-size:
                Image size used for U-Net inference.

            --roi-margin:
                Extra context added around welding_line bbox.

            --conf:
                YOLO confidence threshold.

            --iou:
                YOLO NMS IoU threshold.

            --threshold:
                U-Net probability threshold.

            --device:
                cuda, cpu, or auto.
    """

    parser = argparse.ArgumentParser(
        description="Detect welding defects with YOLO and segment ripple inside welding_line ROIs."
    )

    """Input/output paths"""
    parser.add_argument("--image",          type=Path, default=HYBRID_DEFAULT_IMAGE)
    parser.add_argument("--yolo-model",     type=Path, default=YOLO_MODEL)
    parser.add_argument("--unet-model",     type=Path, default=UNET_MODEL)
    parser.add_argument("--output-dir",     type=Path, default=HYBRID_OUTPUT)

    """Inference image sizes"""
    parser.add_argument("--yolo-imgsz",     type=int, default=YOLO_IMAGE_SIZE)
    parser.add_argument("--unet-img-size",  type=int, default=UNET_IMAGE_SIZE)

    """ROI expansion around welding_line bbox"""
    parser.add_argument("--roi-margin", type=float, default=ROI_MARGIN)

    """YOLO and U-Net thresholds"""
    parser.add_argument("--conf", type=float, default=YOLO_CONFIDENCE_THRESHOLD)
    parser.add_argument("--iou", type=float, default=YOLO_IOU_THRESHOLD)
    parser.add_argument("--threshold", type=float, default=UNET_THRESHOLD)

    """Device setup"""
    parser.add_argument("--device", type=str, default=UNET_DEVICE)

    args = parser.parse_args()

    """Validate threshold values"""
    if not 0.0 <= args.conf <= 1.0:
        parser.error("--conf must be between 0 and 1")

    if not 0.0 <= args.iou <= 1.0:
        parser.error("--iou must be between 0 and 1")

    if not 0.0 <= args.threshold <= 1.0:
        parser.error("--threshold must be between 0 and 1")

    if not MIN_ROI_MARGIN <= args.roi_margin <= MAX_ROI_MARGIN:
        parser.error("--roi-margin must be between 0 and 1")

    """Validate image sizes"""
    if args.yolo_imgsz < 1 or args.unet_img_size < 1:
        parser.error("Image sizes must be positive")

    return args


def load_image(image_path: Path) -> np.ndarray:
    """
        Load input image with OpenCV.

        Returns:
            BGR image as numpy array.
    """

    """Read image from path"""
    image = cv2.imread(str(image_path))

    """Raise error if image cannot be loaded"""
    if image is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")

    return image


def load_yolo(model_path: Path) -> YOLO:
    """
        Load YOLO model from checkpoint path.
    """

    """Check YOLO checkpoint path"""
    if not model_path.exists():
        raise FileNotFoundError(f"YOLO model not found: {model_path}")

    return YOLO(str(model_path))


def normalize_class_name(class_name: str) -> str:
    """
        Normalize class name for safer comparison.

        Example:
            "welding-line" -> "welding_line"
            "Welding Line" -> "welding_line"
    """

    return class_name.strip().lower().replace("-", "_").replace(" ", "_")


def clipped_bbox(bbox: np.ndarray, width: int, height: int) -> tuple[int, int, int, int]:
    """
        Clip bounding box coordinates to image boundary.

        Args:
            bbox:
                Bounding box in xyxy format.

            width:
                Image width.

            height:
                Image height.

        Returns:
            Clipped integer bbox: (x1, y1, x2, y2).
    """

    """Convert bbox from numpy to coordinates"""
    x1, y1, x2, y2 = bbox.tolist()

    """Clip bbox to image boundary"""
    return (
        max(0, min(width, int(np.floor(x1)))),
        max(0, min(height, int(np.floor(y1)))),
        max(0, min(width, int(np.ceil(x2)))),
        max(0, min(height, int(np.ceil(y2)))),
    )


def expanded_bbox(
    bbox: np.ndarray,
    width: int,
    height: int,
    margin: float,
) -> tuple[int, int, int, int]:
    """
        Expand bounding box with margin and clip it to image boundary.

        This gives U-Net more surrounding context around the welding line.
    """

    """Get original bbox coordinates"""
    x1, y1, x2, y2 = bbox.tolist()

    """Compute bbox size"""
    box_width = x2 - x1
    box_height = y2 - y1

    """Expand bbox by margin ratio"""
    expanded = np.array(
        [
            x1 - box_width * margin,
            y1 - box_height * margin,
            x2 + box_width * margin,
            y2 + box_height * margin,
        ],
        dtype=np.float32,
    )

    """Clip expanded bbox to image boundary"""
    return clipped_bbox(expanded, width, height)


def overlay_ripple_mask(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
        Overlay ripple segmentation mask on the original image.
    """

    """Copy original image"""
    output = image.copy()

    """Find predicted ripple pixels"""
    active = mask > 0

    """Return original image if mask is empty"""
    if not np.any(active):
        return output

    """Create colored ripple mask"""
    color = np.zeros_like(image)
    color[active] = RIPPLE_COLOR

    """Blend original image and ripple color"""
    blended = cv2.addWeighted(image, 0.65, color, 0.35, 0)

    """Apply blending only on ripple pixels"""
    output[active] = blended[active]

    return output


def draw_label(
    image: np.ndarray,
    bbox: tuple[int, int, int, int],
    label: str,
    color: tuple[int, int, int],
) -> None:
    """
        Draw bounding box and label text on image.
    """

    """Draw detection box"""
    x1, y1, x2, y2 = bbox
    cv2.rectangle(image, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)

    """Text style"""
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.55
    thickness = 2

    """Compute label text size"""
    (text_width, text_height), baseline = cv2.getTextSize(
        label,
        font,
        font_scale,
        thickness
    )

    """Compute label background position"""
    label_y1 = max(0, y1 - text_height - baseline - 8)
    label_y2 = label_y1 + text_height + baseline + 8
    label_x2 = min(image.shape[1], x1 + text_width + 10)

    """Draw label background"""
    cv2.rectangle(image, (x1, label_y1), (label_x2, label_y2), color, -1)

    """Draw label text"""
    cv2.putText(
        image,
        label,
        (x1 + 5, label_y2 - baseline - 4),
        font,
        font_scale,
        (255, 255, 255),
        thickness,
        cv2.LINE_AA,
    )


def run_hybrid_inference(
    image: np.ndarray,
    yolo_model: YOLO,
    unet_model: torch.nn.Module,
    device: torch.device,
    yolo_imgsz: int,
    unet_img_size: int,
    conf: float,
    iou: float,
    threshold: float,
    roi_margin: float,
) -> tuple[np.ndarray, int, int]:
    """
        Run hybrid YOLO + U-Net inference.

        Workflow:
            1. YOLO detects objects on the full image.
            2. If detected class is welding_line:
                - expand bbox
                - crop ROI
                - run U-Net on ROI
                - post-process ROI mask
                - restrict ripple mask inside YOLO welding_line mask
                - paste ROI mask back to full image
            3. If detected class is defect:
                - store bbox for visualization
            4. Return final overlay result.
    """

    """Run YOLO prediction on full image"""
    result = yolo_model.predict(
        source=image,
        imgsz=yolo_imgsz,
        conf=conf,
        iou=iou,
        device=str(device),
        verbose=False,
    )[0]

    """Initialize full-size ripple mask"""
    height, width = image.shape[:2]
    ripple_mask = np.zeros((height, width), dtype=np.uint8)

    """Store non-welding_line defect boxes"""
    defect_boxes: list[tuple[tuple[int, int, int, int], str, float, int]] = []

    """Count welding_line ROIs processed by U-Net"""
    welding_line_count = 0

    """Process YOLO detections if available"""
    if result.boxes is not None:
        names = result.names

        """Get YOLO instance masks if model supports segmentation"""
        instance_masks = (
            result.masks.data.cpu().numpy()
            if result.masks is not None
            else None
        )

        """Loop through each YOLO detection"""
        for index, box in enumerate(result.boxes):

            class_id = int(box.cls.item())
            class_name = str(names[class_id])
            confidence = float(box.conf.item())

            """Get and clip YOLO bbox"""
            raw_bbox = box.xyxy[0].cpu().numpy()
            bbox = clipped_bbox(raw_bbox, width, height)
            x1, y1, x2, y2 = bbox

            """Skip invalid bbox"""
            if x2 <= x1 or y2 <= y1:
                continue

            """Process welding_line using U-Net"""
            if normalize_class_name(class_name) == WELDING_LINE_CLASS:
                
                """Require YOLO segmentation mask for welding_line filtering"""
                if instance_masks is None or index >= len(instance_masks):
                    continue

                welding_line_count += 1

                """Expand welding_line bbox and crop ROI"""
                bbox = expanded_bbox(raw_bbox, width, height, roi_margin)
                x1, y1, x2, y2 = bbox
                roi = image[y1:y2, x1:x2]

                """Predict ripple mask inside ROI using U-Net"""
                roi_mask = predict_unet(
                    model=unet_model,
                    image=roi,
                    img_size=unet_img_size,
                    threshold=threshold,
                    device=device,
                )

                """Clean U-Net mask"""
                roi_mask = post_process_mask(roi_mask)

                """Resize YOLO welding_line instance mask to original image size"""
                welding_mask = cv2.resize(
                    instance_masks[index],
                    (width, height),
                    interpolation=cv2.INTER_NEAREST,
                )

                """Limit U-Net ripple mask inside welding_line region only"""
                roi_welding_mask = (
                    welding_mask[y1:y2, x1:x2] > YOLO_MASK_THRESHOLD
                )
                roi_mask[~roi_welding_mask] = 0

                """Skip empty U-Net prediction"""
                if not np.any(roi_mask):
                    continue

                """Paste ROI ripple mask back to full-size image"""
                ripple_mask[y1:y2, x1:x2] = np.maximum(
                    ripple_mask[y1:y2, x1:x2],
                    roi_mask
                )

            else:
                """Save defect detection for drawing later"""
                defect_boxes.append((bbox, class_name, confidence, class_id))

    """Overlay ripple mask on original image"""
    output = overlay_ripple_mask(image, ripple_mask)

    """Draw defect boxes on output image"""
    for bbox, class_name, confidence, class_id in defect_boxes:
        color = BOX_COLORS[class_id % len(BOX_COLORS)]
        draw_label(output, bbox, f"{class_name} {confidence:.2f}", color)

    return output, welding_line_count, len(defect_boxes)


def save_image(path: Path, image: np.ndarray) -> None:
    """
        Save image to disk and raise error if saving fails.
    """

    """Create output directory if needed"""
    path.parent.mkdir(parents=True, exist_ok=True)

    """Save image"""
    if not cv2.imwrite(str(path), image):
        raise RuntimeError(f"Failed to save image: {path}")


def main() -> None:
    """
        Main function.

        Steps:
            1. Parse arguments
            2. Load device, image, YOLO model, and U-Net model
            3. Run hybrid inference
            4. Save final output image
            5. Print summary
    """

    args = parse_args()

    """Prepare device and input image"""
    device = get_device(args.device)
    image = load_image(args.image)

    """Print inference configuration"""
    print(f"Image: {args.image}")
    print(f"YOLO model: {args.yolo_model}")
    print(f"U-Net model: {args.unet_model}")
    print(f"Device: {device}")

    """Load YOLO model"""
    yolo_model = load_yolo(args.yolo_model)

    """Load U-Net model and checkpoint image size"""
    unet_model, checkpoint_img_size = load_model(
        args.unet_model,
        device,
        args.unet_img_size
    )

    """Run YOLO + U-Net hybrid inference"""
    output, welding_line_count, defect_count = run_hybrid_inference(
        image=image,
        yolo_model=yolo_model,
        unet_model=unet_model,
        device=device,
        yolo_imgsz=args.yolo_imgsz,
        unet_img_size=checkpoint_img_size,
        conf=args.conf,
        iou=args.iou,
        threshold=args.threshold,
        roi_margin=args.roi_margin,
    )

    """Save output image"""
    output_path = args.output_dir / f"{args.image.stem}_hybrid.jpg"
    save_image(output_path, output)

    """Print inference result summary"""
    print(f"Welding line ROIs: {welding_line_count}")
    print(f"ROI margin: {args.roi_margin:.0%}")
    print(f"Defect boxes: {defect_count}")
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()

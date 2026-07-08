"""
    YOLO + U-Net segmentation service.

    Purpose:
        1. Load YOLO segmentation models for welding defect detection.
        2. Load U-Net models for ripple segmentation.
        3. Run YOLO on uploaded image.
        4. Use YOLO welding_line ROI as input for U-Net.
        5. Overlay ripple mask and return encoded images for web app.
"""

from __future__ import annotations

import base64
import threading
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from ultralytics import YOLO

import torch

from configs.data import NON_DEFECT_CLASSES, WELDING_LINE_CLASS
from configs.hybrid import (
    MAX_ROI_MARGIN,
    MIN_ROI_MARGIN,
    ROI_MARGIN,
)
from configs.unet import (
    UNET_DEVICE,
    UNET_IMAGE_SIZE,
    UNET_MODEL_ID,
    UNET_THRESHOLD,
)
from configs.visualize import (
    OVERLAY_JPEG_QUALITY,
    RIPPLE_COLOR,
    RIPPLE_IMAGE_WEIGHT,
    RIPPLE_MASK_WEIGHT,
    SHOW_BOXES,
    SHOW_CONFIDENCE,
    SHOW_LABELS,
    SHOW_MASKS,
)
from configs.yolo import (
    YOLO_CONFIDENCE_THRESHOLD,
    YOLO_IMAGE_SIZE,
    YOLO_IOU_THRESHOLD,
    YOLO_MASK_THRESHOLD,
    YOLO_MODEL_ID,
    YOLO_TASK,
)
from src.unet.inference import (
    get_device,
    load_model as load_unet_model,
    predict as predict_unet,
)

from src.unet.post_processing import post_process_mask


class YoloSegmentationService:
    """
        Service for hybrid YOLO + U-Net inference.

        Main workflow:
            1. Decode uploaded image bytes.
            2. Run YOLO segmentation model.
            3. Extract YOLO predictions and masks.
            4. Crop welding_line ROI.
            5. Run U-Net inside welding_line ROI.
            6. Post-process U-Net ripple mask.
            7. Overlay results.
            8. Return JSON-ready prediction data.
    """

    def __init__(
        self,
        yolo_models: dict[str, Path],
        unet_models: dict[str, Path],
        default_yolo_model: str,
        default_unet_model: str,
        device: str | None = None,
        unet_device: str = UNET_DEVICE,
        unet_threshold: float = UNET_THRESHOLD,
        roi_margin: float = ROI_MARGIN,
    ) -> None:
        """
            Initialize YOLO and U-Net service.

            Args:
                yolo_models:
                    Dictionary of available YOLO model ids and paths.

                unet_models:
                    Dictionary of available U-Net model ids and paths.

                default_yolo_model:
                    YOLO model loaded at startup.

                default_unet_model:
                    U-Net model loaded at startup.

                device:
                    YOLO inference device.

                unet_device:
                    U-Net inference device.

                unet_threshold:
                    Default threshold for U-Net binary mask.

                roi_margin:
                    Extra margin around welding_line bbox before U-Net inference.
        """

        """Store registered model paths"""
        self.yolo_models = yolo_models
        self.unet_models = unet_models

        """Store YOLO device setting"""
        self.device = device

        """Lock model switching and inference for thread safety"""
        self._lock = threading.Lock()

        """Prepare U-Net device"""
        self.unet_device = get_device(unet_device)

        """Default U-Net prediction settings"""
        self.unet_threshold = unet_threshold
        self.roi_margin = roi_margin

        """Load default models"""
        self._load_yolo_model(default_yolo_model)
        self._load_unet_model(default_unet_model)

    @staticmethod
    def _registered_path(
        models: dict[str, Path],
        model_id: str,
        model_type: str,
    ) -> Path:
        """
            Get model path from registered model dictionary.

            Raises:
                ValueError:
                    If model id is not registered.

                FileNotFoundError:
                    If model checkpoint path does not exist.
        """

        """Find model path by model id"""
        path = models.get(model_id)

        """Raise error if model id is unknown"""
        if path is None:
            available = ", ".join(sorted(models))
            raise ValueError(
                f"Unknown {model_type} model '{model_id}'. Available: {available}"
            )

        """Check model checkpoint path"""
        if not path.exists():
            raise FileNotFoundError(f"{model_type} model not found: {path}")

        return path

    def _load_yolo_model(self, model_id: str) -> None:
        """
            Load YOLO model by model id.
        """

        """Get YOLO checkpoint path"""
        path = self._registered_path(self.yolo_models, model_id, "YOLO")

        """Load YOLO model"""
        self.model = YOLO(str(path), task=YOLO_TASK)

        """Store active YOLO model information"""
        self.model_path = path
        self.active_yolo_model = model_id

        """Normalize YOLO class names"""
        self.names = self._normalize_names(self.model.names)

    def _load_unet_model(self, model_id: str) -> None:
        """
            Load U-Net model by model id.
        """

        """Get U-Net checkpoint path"""
        path = self._registered_path(self.unet_models, model_id, "U-Net")

        """Remove previous U-Net model before loading new model"""
        if hasattr(self, "unet_model"):
            del self.unet_model

            """Clear CUDA cache if GPU is available"""
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        """Load U-Net model and checkpoint image size"""
        self.unet_model, self.unet_img_size = load_unet_model(
            path,
            self.unet_device,
            img_size=None,
        )

        """Store active U-Net model information"""
        self.unet_model_path = path
        self.active_unet_model = model_id

    def _select_models(self, yolo_model: str, unet_model: str) -> None:
        """
            Switch active models if requested model ids are different.
        """

        """Reload YOLO only when model id changes"""
        if yolo_model != self.active_yolo_model:
            self._load_yolo_model(yolo_model)

        """Reload U-Net only when model id changes"""
        if unet_model != self.active_unet_model:
            self._load_unet_model(unet_model)

    def predict(
        self,
        image_bytes: bytes,
        conf: float = YOLO_CONFIDENCE_THRESHOLD,
        iou: float = YOLO_IOU_THRESHOLD,
        imgsz: int = YOLO_IMAGE_SIZE,
        yolo_model: str = YOLO_MODEL_ID,
        unet_model: str = UNET_MODEL_ID,
        unet_imgsz: int | None = None,
        unet_threshold: float | None = None,
        roi_margin: float | None = None,
        show_masks: bool = SHOW_MASKS,
        show_boxes: bool = SHOW_BOXES,
        show_labels: bool = SHOW_LABELS,
        show_conf: bool = SHOW_CONFIDENCE,
    ) -> dict[str, Any]:
        """
            Run full YOLO + U-Net inference on uploaded image.

            Returns:
                Dictionary containing:
                    - image size
                    - model information
                    - inference parameters
                    - YOLO predictions
                    - YOLO mask image
                    - U-Net ripple mask image
                    - annotated output image
                    - inference time
                    - summary result
        """

        """Decode uploaded image bytes"""
        image = self._decode_image(image_bytes)

        """Get original image size"""
        h, w = image.shape[:2]

        """Validate and clamp YOLO parameters"""
        conf = self._clamp_float(conf, 0.01, 1.0)
        iou = self._clamp_float(iou, 0.01, 1.0)
        imgsz = self._clamp_int(imgsz, 320, 1280)

        """Validate and clamp U-Net threshold"""
        threshold = self._clamp_float(
            self.unet_threshold if unet_threshold is None else unet_threshold,
            0.01,
            1.0,
        )

        """Validate and clamp ROI margin"""
        margin = self._clamp_float(
            self.roi_margin if roi_margin is None else roi_margin,
            MIN_ROI_MARGIN,
            MAX_ROI_MARGIN,
        )

        """Prepare YOLO predict arguments"""
        predict_kwargs: dict[str, Any] = {
            "source": image,
            "imgsz": imgsz,
            "conf": conf,
            "iou": iou,
            "verbose": False,
        }

        """Use YOLO device if specified"""
        if self.device:
            predict_kwargs["device"] = self.device

        """Start inference timer"""
        started = time.perf_counter()

        """Lock inference to avoid model switching conflict"""
        with self._lock:
            """Select requested YOLO and U-Net models"""
            self._select_models(yolo_model, unet_model)

            """Select U-Net input image size"""
            active_unet_imgsz = self._clamp_int(
                self.unet_img_size if unet_imgsz is None else unet_imgsz,
                128,
                1024,
            )

            """Run YOLO prediction"""
            result = self.model.predict(**predict_kwargs)[0]

            """Run U-Net ripple segmentation inside welding_line ROI"""
            ripple_mask = self._predict_ripple(
                image,
                result,
                threshold,
                margin,
                active_unet_imgsz,
            )

        """Compute total inference time"""
        inference_ms = round((time.perf_counter() - started) * 1000, 2)

        """Extract YOLO predictions and YOLO mask image"""
        predictions, mask_image = self._extract_predictions_and_mask(
            result,
            image.shape
        )

        """Create YOLO annotated image"""
        annotated = result.plot(
            masks=show_masks,
            boxes=show_boxes,
            labels=show_labels,
            conf=show_conf,
        )

        """Overlay U-Net ripple mask on YOLO annotated image"""
        annotated = self._overlay_ripple(annotated, ripple_mask)

        """Return web-app friendly result dictionary"""
        return {
            "image_size": {"width": w, "height": h},

            "model": {
                "path": str(self.model_path),
                "classes": self.names,
            },

            "parameters": {
                "conf": conf,
                "iou": iou,
                "imgsz": imgsz,
                "yolo_model": self.active_yolo_model,
                "unet_model": self.active_unet_model,
                "unet_imgsz": active_unet_imgsz,
                "unet_threshold": threshold,
                "roi_margin": margin,
                "display": {
                    "masks": show_masks,
                    "boxes": show_boxes,
                    "labels": show_labels,
                    "conf": show_conf,
                },
            },

            # YOLO segmentation mask image.
            "mask_image": (
                self._encode_image(mask_image)
                if mask_image is not None
                else None
            ),

            # U-Net ripple binary mask image.
            "unet_mask_image": (
                self._encode_image(ripple_mask)
                if ripple_mask is not None
                else None
            ),

            "inference_ms": inference_ms,
            "summary": self._build_summary(predictions),
            "predictions": predictions,
            "annotated_image": self._encode_image(annotated),
        }

    @staticmethod
    def _decode_image(image_bytes: bytes) -> np.ndarray:
        """
            Decode uploaded image bytes to OpenCV BGR image.
        """

        """Check empty upload"""
        if not image_bytes:
            raise ValueError("Uploaded image is empty.")

        """Convert bytes to numpy buffer"""
        image_array = np.frombuffer(image_bytes, np.uint8)

        """Decode image buffer"""
        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

        """Check decoded image"""
        if image is None or image.size == 0:
            raise ValueError("Cannot decode uploaded image.")

        return image

    def _extract_predictions_and_mask(
        self,
        result: Any,
        image_shape: tuple[int, int, int],
    ) -> tuple[list[dict[str, Any]], np.ndarray | None]:
        """
            Extract YOLO prediction information and build colored mask image.

            Returns:
                predictions:
                    List of detection dictionaries.

                mask_canvas:
                    Colored YOLO segmentation mask image.
        """

        """Get YOLO boxes and masks"""
        boxes = result.boxes
        masks = result.masks

        """Normalize class names"""
        names = self._normalize_names(getattr(result, "names", self.names))

        predictions: list[dict[str, Any]] = []

        """Get original image size"""
        h, w = image_shape[:2]

        """Create empty mask canvas"""
        mask_canvas = np.zeros((h, w, 3), dtype=np.uint8)
        has_mask = False

        """Return empty prediction if YOLO detects nothing"""
        if boxes is None or len(boxes) == 0:
            return predictions, None

        """Extract boxes, class ids and confidences"""
        xyxy_list = boxes.xyxy.cpu().numpy()
        cls_list = boxes.cls.cpu().numpy().astype(int)
        conf_list = boxes.conf.cpu().numpy()

        """Extract YOLO polygon masks and bitmap masks"""
        mask_polygons = masks.xy if masks is not None else []
        mask_bitmaps = masks.data.cpu().numpy() if masks is not None else None

        """Process each YOLO prediction"""
        for index, bbox in enumerate(xyxy_list):
            class_id = int(cls_list[index])
            class_name = names.get(class_id, str(class_id))
            confidence = float(conf_list[index])

            """Convert bbox coordinates to rounded xyxy list"""
            x1, y1, x2, y2 = [
                round(float(value), 2)
                for value in bbox.tolist()
            ]

            """Extract polygon points if available"""
            polygon = None

            if index < len(mask_polygons) and mask_polygons[index] is not None:
                polygon = [
                    [round(float(point[0]), 2), round(float(point[1]), 2)]
                    for point in mask_polygons[index].tolist()
                ]

            """Compute mask area and draw colored mask"""
            mask_area_pixels = 0

            if mask_bitmaps is not None and index < len(mask_bitmaps):
                """Resize YOLO mask to original image size"""
                mask = cv2.resize(
                    mask_bitmaps[index],
                    (w, h),
                    interpolation=cv2.INTER_NEAREST,
                )

                """Find active mask pixels"""
                active_mask = mask > YOLO_MASK_THRESHOLD

                """Compute mask area in pixels"""
                mask_area_pixels = int(np.count_nonzero(active_mask))

                """Draw class color on mask canvas"""
                if mask_area_pixels > 0:
                    mask_canvas[active_mask] = self._color_for_class(class_id)
                    has_mask = True

            """Store one prediction result"""
            predictions.append(
                {
                    "class_id": class_id,
                    "class_name": class_name,
                    "confidence": round(confidence, 4),
                    "bbox": [x1, y1, x2, y2],
                    "polygon": polygon,
                    "mask_area_pixels": mask_area_pixels,
                }
            )

        """Sort predictions by confidence"""
        predictions.sort(key=lambda item: item["confidence"], reverse=True)

        return predictions, mask_canvas if has_mask else None

    def _predict_ripple(
        self,
        image: np.ndarray,
        result: Any,
        threshold: float,
        roi_margin: float,
        unet_imgsz: int,
    ) -> np.ndarray | None:
        """
            Predict ripple mask using U-Net inside YOLO welding_line ROIs.

            Workflow:
                1. Find YOLO detections with class welding_line.
                2. Expand welding_line bbox by roi_margin.
                3. Crop ROI from original image.
                4. Run U-Net on ROI.
                5. Post-process U-Net mask.
                6. Restrict U-Net mask inside YOLO welding_line mask.
                7. Paste ROI mask back to original image size.
        """

        """Get original image size"""
        height, width = image.shape[:2]

        """Create full-size empty ripple mask"""
        full_mask = np.zeros((height, width), dtype=np.uint8)

        """Require both YOLO boxes and YOLO masks"""
        if result.boxes is None or result.masks is None:
            return None

        """Normalize class names and get YOLO instance masks"""
        names = self._normalize_names(result.names)
        instance_masks = result.masks.data.cpu().numpy()

        """Loop through YOLO detections"""
        for index, box in enumerate(result.boxes):
            class_id = int(box.cls.item())
            class_name = self._normalize_class_name(names[class_id])

            """Only process welding_line class"""
            if class_name != WELDING_LINE_CLASS:
                continue

            """Skip if mask index is invalid"""
            if index >= len(instance_masks):
                continue

            """Get welding_line bbox"""
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()

            """Compute bbox size"""
            box_width = x2 - x1
            box_height = y2 - y1

            """Expand bbox with ROI margin and clip to image boundary"""
            x1 = max(0, int(x1 - box_width * roi_margin))
            y1 = max(0, int(y1 - box_height * roi_margin))
            x2 = min(width, int(x2 + box_width * roi_margin))
            y2 = min(height, int(y2 + box_height * roi_margin))

            """Skip invalid ROI"""
            if x2 <= x1 or y2 <= y1:
                continue

            """Crop welding_line ROI"""
            roi = image[y1:y2, x1:x2]

            """Predict ripple mask inside ROI using U-Net"""
            roi_mask = predict_unet(
                model=self.unet_model,
                image=roi,
                img_size=unet_imgsz,
                threshold=threshold,
                device=self.unet_device,
            )

            """Clean U-Net ripple mask"""
            roi_mask = post_process_mask(roi_mask)

            """Resize YOLO welding_line instance mask to original image size"""
            welding_mask = cv2.resize(
                instance_masks[index],
                (width, height),
                interpolation=cv2.INTER_NEAREST,
            )

            """Restrict ripple mask inside YOLO welding_line region"""
            roi_welding_mask = (
                welding_mask[y1:y2, x1:x2] > YOLO_MASK_THRESHOLD
            )
            roi_mask[~roi_welding_mask] = 0

            """Skip empty ripple prediction"""
            if not np.any(roi_mask):
                continue

            """Paste ROI ripple mask back to full-size image"""
            full_mask[y1:y2, x1:x2] = np.maximum(
                full_mask[y1:y2, x1:x2],
                roi_mask,
            )

        """Return None if no ripple was detected"""
        return full_mask if np.any(full_mask) else None

    @staticmethod
    def _overlay_ripple(
        image: np.ndarray,
        ripple_mask: np.ndarray | None,
    ) -> np.ndarray:
        """
            Overlay U-Net ripple mask on output image.
        """

        """Return original image if U-Net mask is empty"""
        if ripple_mask is None:
            return image

        """Find ripple pixels"""
        active = ripple_mask > 0

        """Create green color mask"""
        color = np.zeros_like(image)
        color[active] = RIPPLE_COLOR

        """Blend original image with ripple color"""
        output = image.copy()
        blended = cv2.addWeighted(
            image,
            RIPPLE_IMAGE_WEIGHT,
            color,
            RIPPLE_MASK_WEIGHT,
            0,
        )

        """Apply blending only on ripple pixels"""
        output[active] = blended[active]

        return output

    @staticmethod
    def _build_summary(predictions: list[dict[str, Any]]) -> dict[str, Any]:
        """
            Build short prediction summary for web app.

            Verdict:
                no_detection:
                    YOLO detects nothing.

                pass:
                    Only non-defect classes are detected.

                defect:
                    At least one defect class is detected.
        """

        """Handle no detection case"""
        if not predictions:
            return {
                "verdict": "no_detection",
                "primary_class": None,
                "confidence": None,
                "defect_count": 0,
                "object_count": 0,
            }

        """Filter defect predictions"""
        defect_predictions = [
            item
            for item in predictions
            if YoloSegmentationService._normalize_class_name(item["class_name"])
            not in NON_DEFECT_CLASSES
        ]

        """Select primary result"""
        primary = defect_predictions[0] if defect_predictions else predictions[0]

        """Build summary dictionary"""
        return {
            "verdict": "defect" if defect_predictions else "pass",
            "primary_class": primary["class_name"],
            "confidence": primary["confidence"],
            "defect_count": len(defect_predictions),
            "object_count": len(predictions),
        }

    @staticmethod
    def _encode_image(image: np.ndarray) -> str:
        """
            Encode OpenCV image to base64 JPEG data URL.

            This format can be displayed directly in web frontend.
        """

        """Encode image as JPEG"""
        success, buffer = cv2.imencode(
            ".jpg",
            image,
            [int(cv2.IMWRITE_JPEG_QUALITY), OVERLAY_JPEG_QUALITY]
        )

        """Check encoding result"""
        if not success:
            raise RuntimeError("Cannot encode prediction image.")

        """Convert JPEG bytes to base64 string"""
        encoded = base64.b64encode(buffer).decode("ascii")

        return f"data:image/jpeg;base64,{encoded}"

    @staticmethod
    def _normalize_names(names: Any) -> dict[int, str]:
        """
            Normalize class names from YOLO format to dictionary format.

            Supports:
                - dict from YOLO
                - list or tuple
        """

        """Handle dictionary names"""
        if isinstance(names, dict):
            return {int(key): str(value) for key, value in names.items()}

        """Handle list or tuple names"""
        if isinstance(names, (list, tuple)):
            return {index: str(value) for index, value in enumerate(names)}

        """Return empty dictionary if names format is unknown"""
        return {}

    @staticmethod
    def _normalize_class_name(class_name: str) -> str:
        """
            Normalize class name for safer comparison.
        """

        return class_name.strip().lower().replace(" ", "_")

    @staticmethod
    def _clamp_float(value: float, minimum: float, maximum: float) -> float:
        """
            Clamp float value into a valid range.
        """

        return max(minimum, min(maximum, float(value)))

    @staticmethod
    def _clamp_int(value: int, minimum: int, maximum: int) -> int:
        """
            Clamp integer value into a valid range.
        """

        return max(minimum, min(maximum, int(value)))

    @staticmethod
    def _color_for_class(class_id: int) -> tuple[int, int, int]:
        """
            Get visualization color for each class id.
        """

        """Color palette in BGR format"""
        palette = [
            (42, 157, 143),
            (231, 111, 81),
            (233, 196, 106),
            (38, 70, 83),
            (244, 162, 97),
            (80, 125, 188),
            (140, 82, 255),
        ]

        """Cycle color if class id is larger than palette size"""
        return palette[class_id % len(palette)]

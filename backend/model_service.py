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

from src.unet.inference import (
    get_device,
    load_model as load_unet_model,
    predict as predict_unet,
)

from src.unet.post_processing import post_process_mask


NON_DEFECT_CLASSES = {
    "good_weld",
    "good-weld",
    "good weld",
    "welding_line",
    "welding-line",
    "welding line",
    "weld_line",
    "weld-line",
    "weld line",
}

"""
    Visualize Config
"""
MASKS  = False
BOXES  = True
LABELS = False
CONF   = False
"""
    End Visualize Config
"""

class YoloSegmentationService:
    def __init__(
        self,
        yolo_models: dict[str, Path],
        unet_models: dict[str, Path],
        default_yolo_model: str,
        default_unet_model: str,
        device: str | None = None,
        unet_device: str = "auto",
        unet_threshold: float = 0.25,
        roi_margin: float = 0.50,
    ) -> None:
        self.yolo_models = yolo_models
        self.unet_models = unet_models
        self.device = device
        self._lock = threading.Lock()
        self.unet_device = get_device(unet_device)
        self.unet_threshold = unet_threshold
        self.roi_margin = roi_margin
        self._load_yolo_model(default_yolo_model)
        self._load_unet_model(default_unet_model)

    @staticmethod
    def _registered_path(
        models: dict[str, Path],
        model_id: str,
        model_type: str,
    ) -> Path:
        path = models.get(model_id)
        if path is None:
            available = ", ".join(sorted(models))
            raise ValueError(
                f"Unknown {model_type} model '{model_id}'. Available: {available}"
            )
        if not path.is_file():
            raise FileNotFoundError(f"{model_type} model not found: {path}")
        return path

    def _load_yolo_model(self, model_id: str) -> None:
        path = self._registered_path(self.yolo_models, model_id, "YOLO")
        self.model = YOLO(str(path))
        self.model_path = path
        self.active_yolo_model = model_id
        self.names = self._normalize_names(self.model.names)

    def _load_unet_model(self, model_id: str) -> None:
        path = self._registered_path(self.unet_models, model_id, "U-Net")
        if hasattr(self, "unet_model"):
            del self.unet_model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        self.unet_model, self.unet_img_size = load_unet_model(
            path,
            self.unet_device,
            img_size=None,
        )
        self.unet_model_path = path
        self.active_unet_model = model_id

    def _select_models(self, yolo_model: str, unet_model: str) -> None:
        if yolo_model != self.active_yolo_model:
            self._load_yolo_model(yolo_model)
        if unet_model != self.active_unet_model:
            self._load_unet_model(unet_model)

    def predict(
        self,
        image_bytes: bytes,
        conf: float = 0.25,
        iou: float = 0.25,
        imgsz: int = 960,
        yolo_model: str = "yolo_ver5",
        unet_model: str = "unet_ver3",
        unet_imgsz: int | None = None,
        unet_threshold: float | None = None,
        roi_margin: float | None = None,
        show_masks: bool = MASKS,
        show_boxes: bool = BOXES,
        show_labels: bool = LABELS,
        show_conf: bool = CONF,
    ) -> dict[str, Any]:

        image = self._decode_image(image_bytes)
        h, w = image.shape[:2]

        conf = self._clamp_float(conf, 0.01, 1.0)
        iou = self._clamp_float(iou, 0.01, 1.0)
        imgsz = self._clamp_int(imgsz, 320, 1280)

        threshold = self._clamp_float(
            self.unet_threshold if unet_threshold is None else unet_threshold,
            0.01,
            1.0,
        )
        margin = self._clamp_float(
            self.roi_margin if roi_margin is None else roi_margin,
            0.0,
            1.0,
        )

        predict_kwargs: dict[str, Any] = {
            "source": image,
            "imgsz": imgsz,
            "conf": conf,
            "iou": iou,
            "verbose": False,
        }
        if self.device:
            predict_kwargs["device"] = self.device

        started = time.perf_counter()
        with self._lock:
            self._select_models(yolo_model, unet_model)
            active_unet_imgsz = self._clamp_int(
                self.unet_img_size if unet_imgsz is None else unet_imgsz,
                128,
                1024,
            )
            result = self.model.predict(**predict_kwargs)[0]
            ripple_mask = self._predict_ripple(
                image,
                result,
                threshold,
                margin,
                active_unet_imgsz,
            )

        inference_ms = round((time.perf_counter() - started) * 1000, 2)

        predictions, mask_image = self._extract_predictions_and_mask(
            result, image.shape
        )

        annotated = result.plot(
            masks=show_masks,
            boxes=show_boxes,
            labels=show_labels,
            conf=show_conf,
        )
        annotated = self._overlay_ripple(annotated, ripple_mask)

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

            "mask_image": (
                self._encode_image(mask_image)
                if mask_image is not None
                else None
            ),
            "unet_mask_image": (
                self._encode_image(ripple_mask)
                if ripple_mask is not None
                else None
            ),

            "inference_ms": inference_ms,
            "summary": self._build_summary(predictions),
            "predictions": predictions,
            "annotated_image": self._encode_image(annotated),
            "mask_image": self._encode_image(mask_image) if mask_image is not None else None,
        }

    @staticmethod
    def _decode_image(image_bytes: bytes) -> np.ndarray:
        if not image_bytes:
            raise ValueError("Uploaded image is empty.")

        image_array = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

        if image is None or image.size == 0:
            raise ValueError("Cannot decode uploaded image.")

        return image

    def _extract_predictions_and_mask(
        self,
        result: Any,
        image_shape: tuple[int, int, int],
    ) -> tuple[list[dict[str, Any]], np.ndarray | None]:

        boxes = result.boxes
        masks = result.masks
        names = self._normalize_names(getattr(result, "names", self.names))

        predictions: list[dict[str, Any]] = []
        h, w = image_shape[:2]
        mask_canvas = np.zeros((h, w, 3), dtype=np.uint8)
        has_mask = False

        if boxes is None or len(boxes) == 0:
            return predictions, None

        xyxy_list = boxes.xyxy.cpu().numpy()
        cls_list = boxes.cls.cpu().numpy().astype(int)
        conf_list = boxes.conf.cpu().numpy()

        mask_polygons = masks.xy if masks is not None else []
        mask_bitmaps = masks.data.cpu().numpy() if masks is not None else None

        for index, bbox in enumerate(xyxy_list):
            class_id = int(cls_list[index])
            class_name = names.get(class_id, str(class_id))
            confidence = float(conf_list[index])
            x1, y1, x2, y2 = [round(float(value), 2) for value in bbox.tolist()]

            polygon = None
            if index < len(mask_polygons) and mask_polygons[index] is not None:
                polygon = [
                    [round(float(point[0]), 2), round(float(point[1]), 2)]
                    for point in mask_polygons[index].tolist()
                ]

            mask_area_pixels = 0
            if mask_bitmaps is not None and index < len(mask_bitmaps):
                mask = cv2.resize(
                    mask_bitmaps[index],
                    (w, h),
                    interpolation=cv2.INTER_NEAREST,
                )
                active_mask = mask > 0.5
                mask_area_pixels = int(np.count_nonzero(active_mask))
                if mask_area_pixels > 0:
                    mask_canvas[active_mask] = self._color_for_class(class_id)
                    has_mask = True

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
        height, width = image.shape[:2]
        full_mask = np.zeros((height, width), dtype=np.uint8)

        if result.boxes is None or result.masks is None:
            return None

        names = self._normalize_names(result.names)
        instance_masks = result.masks.data.cpu().numpy()

        for index, box in enumerate(result.boxes):
            class_id = int(box.cls.item())
            class_name = self._normalize_class_name(names[class_id])

            if class_name != "welding_line":
                continue
            if index >= len(instance_masks):
                continue

            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            box_width = x2 - x1
            box_height = y2 - y1

            x1 = max(0, int(x1 - box_width * roi_margin))
            y1 = max(0, int(y1 - box_height * roi_margin))
            x2 = min(width, int(x2 + box_width * roi_margin))
            y2 = min(height, int(y2 + box_height * roi_margin))

            if x2 <= x1 or y2 <= y1:
                continue

            roi = image[y1:y2, x1:x2]

            roi_mask = predict_unet(
                model=self.unet_model,
                image=roi,
                img_size=unet_imgsz,
                threshold=threshold,
                device=self.unet_device,
            )
            roi_mask = post_process_mask(roi_mask)

            welding_mask = cv2.resize(
                instance_masks[index],
                (width, height),
                interpolation=cv2.INTER_NEAREST,
            )
            roi_welding_mask = welding_mask[y1:y2, x1:x2] > 0.5
            roi_mask[~roi_welding_mask] = 0

            if not np.any(roi_mask):
                continue

            full_mask[y1:y2, x1:x2] = np.maximum(
                full_mask[y1:y2, x1:x2],
                roi_mask,
            )

        return full_mask if np.any(full_mask) else None


    @staticmethod
    def _overlay_ripple(
        image: np.ndarray,
        ripple_mask: np.ndarray | None,
    ) -> np.ndarray:
        if ripple_mask is None:
            return image

        active = ripple_mask > 0
        color = np.zeros_like(image)
        color[active] = (0, 255, 0)

        output = image.copy()
        blended = cv2.addWeighted(image, 0.65, color, 0.35, 0)
        output[active] = blended[active]

        return output

    @staticmethod
    def _build_summary(predictions: list[dict[str, Any]]) -> dict[str, Any]:
        if not predictions:
            return {
                "verdict": "no_detection",
                "primary_class": None,
                "confidence": None,
                "defect_count": 0,
                "object_count": 0,
            }

        defect_predictions = [
            item
            for item in predictions
            if YoloSegmentationService._normalize_class_name(item["class_name"])
            not in NON_DEFECT_CLASSES
        ]
        primary = defect_predictions[0] if defect_predictions else predictions[0]

        return {
            "verdict": "defect" if defect_predictions else "pass",
            "primary_class": primary["class_name"],
            "confidence": primary["confidence"],
            "defect_count": len(defect_predictions),
            "object_count": len(predictions),
        }

    @staticmethod
    def _encode_image(image: np.ndarray) -> str:
        success, buffer = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
        if not success:
            raise RuntimeError("Cannot encode prediction image.")

        encoded = base64.b64encode(buffer).decode("ascii")
        return f"data:image/jpeg;base64,{encoded}"

    @staticmethod
    def _normalize_names(names: Any) -> dict[int, str]:
        if isinstance(names, dict):
            return {int(key): str(value) for key, value in names.items()}

        if isinstance(names, (list, tuple)):
            return {index: str(value) for index, value in enumerate(names)}

        return {}

    @staticmethod
    def _normalize_class_name(class_name: str) -> str:
        return class_name.strip().lower().replace(" ", "_")

    @staticmethod
    def _clamp_float(value: float, minimum: float, maximum: float) -> float:
        return max(minimum, min(maximum, float(value)))

    @staticmethod
    def _clamp_int(value: int, minimum: int, maximum: int) -> int:
        return max(minimum, min(maximum, int(value)))

    @staticmethod
    def _color_for_class(class_id: int) -> tuple[int, int, int]:
        palette = [
            (42, 157, 143),
            (231, 111, 81),
            (233, 196, 106),
            (38, 70, 83),
            (244, 162, 97),
            (80, 125, 188),
            (140, 82, 255),
        ]
        return palette[class_id % len(palette)]

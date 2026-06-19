from __future__ import annotations

import base64
import threading
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from ultralytics import YOLO


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


class YoloSegmentationService:
    def __init__(self, model_path: Path, device: str | None = None) -> None:
        self.model_path = model_path
        self.device = device
        self._lock = threading.Lock()

        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {self.model_path}")

        self.model = YOLO(str(self.model_path))
        self.names = self._normalize_names(getattr(self.model, "names", {}))

    def predict(
        self,
        image_bytes: bytes,
        conf: float = 0.25,
        iou: float = 0.25,
        imgsz: int = 960,
    ) -> dict[str, Any]:
        image = self._decode_image(image_bytes)
        h, w = image.shape[:2]

        conf = self._clamp_float(conf, 0.01, 1.0)
        iou = self._clamp_float(iou, 0.01, 1.0)
        imgsz = self._clamp_int(imgsz, 320, 1280)

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
            result = self.model.predict(**predict_kwargs)[0]
        inference_ms = round((time.perf_counter() - started) * 1000, 2)

        predictions, mask_image = self._extract_predictions_and_mask(result, image.shape)
        annotated = result.plot(boxes=True, labels=True, conf=True)

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
            },
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

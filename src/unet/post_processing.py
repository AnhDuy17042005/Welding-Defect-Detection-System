"""Post-processing utilities for binary U-Net segmentation masks."""

from __future__ import annotations

import cv2
import numpy as np


def _ellipse_kernel(size: int) -> np.ndarray | None:
    if size <= 1:
        return None
    if size % 2 == 0:
        size += 1
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))


def post_process_mask(
    mask: np.ndarray,
    open_kernel_size: int = 3,
    close_kernel_size: int = 7,
    min_area_pixels: int = 64,
    min_area_ratio: float = 0.001,
    min_largest_ratio: float = 0.03,
    fill_holes: bool = True,
) -> np.ndarray:
    """Remove small islands and reconnect small gaps in a binary mask.

    Components are retained when they are large enough relative to both the
    image and the largest predicted component. This removes isolated noise
    without forcing the result to contain only one valid ripple region.
    """

    if mask.ndim == 3 and mask.shape[2] == 1:
        mask = mask[:, :, 0]
    if mask.ndim != 2:
        raise ValueError(f"Expected a 2D mask, got shape {mask.shape}")
    if min_area_pixels < 0:
        raise ValueError("min_area_pixels cannot be negative")
    if not 0.0 <= min_area_ratio <= 1.0:
        raise ValueError("min_area_ratio must be between 0 and 1")
    if not 0.0 <= min_largest_ratio <= 1.0:
        raise ValueError("min_largest_ratio must be between 0 and 1")

    binary = (mask > 0).astype(np.uint8) * 255
    if not np.any(binary):
        return binary

    open_kernel = _ellipse_kernel(open_kernel_size)
    if open_kernel is not None:
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, open_kernel)

    close_kernel = _ellipse_kernel(close_kernel_size)
    if close_kernel is not None:
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, close_kernel)

    component_count, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary,
        connectivity=8,
    )
    if component_count <= 1:
        return binary

    areas = stats[1:, cv2.CC_STAT_AREA]
    largest_area = int(areas.max())
    image_area = binary.shape[0] * binary.shape[1]
    minimum_area = max(
        min_area_pixels,
        int(round(image_area * min_area_ratio)),
        int(round(largest_area * min_largest_ratio)),
    )

    cleaned = np.zeros_like(binary)
    for component_id in range(1, component_count):
        area = int(stats[component_id, cv2.CC_STAT_AREA])
        if area >= minimum_area:
            cleaned[labels == component_id] = 255

    if fill_holes and np.any(cleaned):
        contours, _ = cv2.findContours(
            cleaned,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )
        cv2.drawContours(cleaned, contours, -1, 255, thickness=cv2.FILLED)

    return cleaned

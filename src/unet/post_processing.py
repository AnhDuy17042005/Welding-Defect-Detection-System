"""
    Post-processing utilities for binary U-Net segmentation masks.

    Purpose:
        Clean raw U-Net masks by removing small noise, reconnecting broken
        regions, filtering small components, and optionally filling holes.
"""

from __future__ import annotations

import cv2
import numpy as np

from configs.unet import (
    UNET_CLOSE_KERNEL_SIZE,
    UNET_FILL_HOLES,
    UNET_MIN_AREA_PIXELS,
    UNET_MIN_AREA_RATIO,
    UNET_MIN_LARGEST_RATIO,
    UNET_OPEN_KERNEL_SIZE,
)


def _ellipse_kernel(size: int) -> np.ndarray | None:
    """
        Create an elliptical morphology kernel.

        Odd kernel size is preferred because it has a clear center pixel.
        If size <= 1, no morphology operation will be applied.
    """
    if size <= 1:
        return None

    """Force kernel size to be odd"""
    if size % 2 == 0:
        size += 1

    return cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (size, size)
    )


def post_process_mask(
    mask: np.ndarray,
    open_kernel_size: int = UNET_OPEN_KERNEL_SIZE,
    close_kernel_size: int = UNET_CLOSE_KERNEL_SIZE,
    min_area_pixels: int = UNET_MIN_AREA_PIXELS,
    min_area_ratio: float = UNET_MIN_AREA_RATIO,
    min_largest_ratio: float = UNET_MIN_LARGEST_RATIO,
    fill_holes: bool = UNET_FILL_HOLES,
) -> np.ndarray:
    """
        Clean a binary segmentation mask predicted by U-Net.

        Processing steps:
            1. Convert mask to binary 0/255 format
            2. Remove small noise by morphological opening
            3. Reconnect small gaps by morphological closing
            4. Remove small connected components
            5. Fill holes inside valid mask regions
    """

    """Convert H x W x 1 mask to H x W"""
    if mask.ndim == 3 and mask.shape[2] == 1:
        mask = mask[:, :, 0]

    """Only 2D binary masks are supported"""
    if mask.ndim != 2:
        raise ValueError(f"Expected a 2D mask, got shape {mask.shape}")

    """Validate area filtering parameters"""
    if min_area_pixels < 0:
        raise ValueError("min_area_pixels cannot be negative")

    if not 0.0 <= min_area_ratio <= 1.0:
        raise ValueError("min_area_ratio must be between 0 and 1")

    if not 0.0 <= min_largest_ratio <= 1.0:
        raise ValueError("min_largest_ratio must be between 0 and 1")

    """Convert mask to binary format: background=0, object=255"""
    binary = (mask > 0).astype(np.uint8) * 255

    """Return early if the predicted mask is empty"""
    if not np.any(binary):
        return binary

    """Remove small isolated white noise"""
    open_kernel = _ellipse_kernel(open_kernel_size)

    if open_kernel is not None:
        binary = cv2.morphologyEx(
            binary,
            cv2.MORPH_OPEN,
            open_kernel
        )

    """Reconnect small gaps inside predicted regions"""
    close_kernel = _ellipse_kernel(close_kernel_size)

    if close_kernel is not None:
        binary = cv2.morphologyEx(
            binary,
            cv2.MORPH_CLOSE,
            close_kernel
        )

    """Find all connected mask components"""
    component_count, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary,
        connectivity=8,
    )

    """No foreground component found"""
    if component_count <= 1:
        return binary

    """Compute component area threshold"""
    areas = stats[1:, cv2.CC_STAT_AREA]
    largest_area = int(areas.max())
    image_area = binary.shape[0] * binary.shape[1]

    minimum_area = max(
        min_area_pixels,
        int(round(image_area * min_area_ratio)),
        int(round(largest_area * min_largest_ratio)),
    )

    """Keep only components large enough to be valid"""
    cleaned = np.zeros_like(binary)

    for component_id in range(1, component_count):
        area = int(stats[component_id, cv2.CC_STAT_AREA])

        if area >= minimum_area:
            cleaned[labels == component_id] = 255

    """Fill holes inside the remaining mask regions"""
    if fill_holes and np.any(cleaned):
        contours, _ = cv2.findContours(
            cleaned,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )

        cv2.drawContours(
            cleaned,
            contours,
            -1,
            255,
            thickness=cv2.FILLED
        )

    return cleaned

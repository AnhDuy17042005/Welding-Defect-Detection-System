"""
    Validation utilities for YOLO segmentation labels.

    Purpose:
        1. Find image and label files.
        2. Check missing labels and orphan labels.
        3. Validate YOLO segmentation label format.
        4. Read valid annotations and report invalid lines.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from configs.data import CLASS_NAMES, IMAGE_EXTENSIONS


@dataclass(frozen=True)
class YoloSegAnnotation:
    """
        Store one YOLO segmentation annotation.

        YOLO segmentation format:
            class_id x1 y1 x2 y2 x3 y3 ...

        Args:
            class_id:
                Object class id.

            coords:
                Flattened polygon coordinates normalized to [0, 1].

            raw_line:
                Original valid label line.
    """

    class_id: int
    coords: tuple[float, ...]
    raw_line: str


def find_image_paths(images_dir: Path, missing_ok: bool = False) -> list[Path]:
    """
        Find all image files in the image directory.
    """

    """Check image directory"""
    if not images_dir.exists():
        if missing_ok:
            return []

        raise FileNotFoundError(f"Images folder not found: {images_dir}")

    """Return supported image files sorted by path"""
    return sorted(
        path
        for path in images_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def find_label_paths(labels_dir: Path, missing_ok: bool = False) -> list[Path]:
    """
        Find all YOLO label files in the label directory.
    """

    """Check label directory"""
    if not labels_dir.exists():
        if missing_ok:
            return []

        raise FileNotFoundError(f"Labels folder not found: {labels_dir}")

    """Return all .txt label files"""
    return sorted(labels_dir.glob("*.txt"))


def find_missing_label_images(images: list[Path], labels: list[Path]) -> list[Path]:
    """
        Find images that do not have corresponding label files.
    """

    """Collect label file stem names"""
    label_stems = {path.stem for path in labels}

    """Return images whose stem does not exist in labels"""
    return [path for path in images if path.stem not in label_stems]


def find_orphan_labels(images: list[Path], labels: list[Path]) -> list[Path]:
    """
        Find label files that do not have corresponding image files.
    """

    """Collect image file stem names"""
    image_stems = {path.stem for path in images}

    """Return labels whose stem does not exist in images"""
    return [path for path in labels if path.stem not in image_stems]


def validate_yolo_seg_line(
    line: str,
    line_number: int,
) -> tuple[YoloSegAnnotation | None, str | None]:
    """
        Validate one YOLO segmentation label line.

        A valid line must contain:
            - class id
            - at least 3 polygon points
            - even number of polygon coordinates
            - coordinates normalized in range [0, 1]

        Returns:
            annotation:
                Parsed annotation if the line is valid.

            issue:
                Error message if the line is invalid.
    """

    """Remove leading and trailing spaces"""
    stripped = line.strip()

    """Ignore empty lines"""
    if not stripped:
        return None, None

    """Split line into class id and coordinates"""
    parts = stripped.split()

    """YOLO segmentation needs class id + at least 3 points"""
    if len(parts) < 7:
        return None, f"line {line_number}: segmentation label needs class + at least 3 points"

    try:
        """Parse class id and polygon coordinates"""
        class_id = int(float(parts[0]))
        coords = tuple(float(value) for value in parts[1:])

    except ValueError:
        return None, f"line {line_number}: non-numeric value"

    """Check class id is supported"""
    if class_id not in CLASS_NAMES:
        return None, f"line {line_number}: unsupported class id {class_id}"

    """Polygon coordinates must be pairs of x and y"""
    if len(coords) % 2 != 0:
        return None, f"line {line_number}: odd number of polygon coordinates"

    """A polygon must have at least 3 points"""
    if len(coords) < 6:
        return None, f"line {line_number}: polygon has fewer than 3 points"

    """YOLO normalized coordinates must be inside [0, 1]"""
    if any(value < 0.0 or value > 1.0 for value in coords):
        return None, f"line {line_number}: coordinates outside [0, 1]"

    """Return valid annotation"""
    return YoloSegAnnotation(
        class_id=class_id,
        coords=coords,
        raw_line=stripped,
    ), None


def read_yolo_seg_label(label_path: Path) -> tuple[list[YoloSegAnnotation], list[str], bool]:
    """
        Read and validate one YOLO segmentation label file.

        Returns:
            annotations:
                List of valid annotations.

            issues:
                List of invalid line messages.

            is_empty:
                True if the label file has no valid content.
    """

    annotations: list[YoloSegAnnotation] = []
    issues: list[str] = []
    has_content = False

    """Return empty result if label file does not exist"""
    if not label_path.exists():
        return annotations, issues, False

    """Read label file line by line"""
    lines = label_path.read_text(encoding="utf-8").splitlines()

    for line_number, line in enumerate(lines, start=1):
        """Skip blank lines"""
        if not line.strip():
            continue

        has_content = True

        """Validate one label line"""
        annotation, issue = validate_yolo_seg_line(line, line_number)

        """Store invalid line issue"""
        if issue is not None:
            issues.append(issue)
            continue

        """Store valid annotation"""
        if annotation is not None:
            annotations.append(annotation)

    """is_empty=True when file has no non-empty label line"""
    return annotations, issues, not has_content


def sanitize_yolo_seg_label(label_path: Path) -> tuple[list[str], list[str]]:
    """
        Keep only valid YOLO segmentation label lines.

        This function is useful for cleaning label files before training.

        Returns:
            valid_lines:
                Raw valid label lines.

            issues:
                Invalid line messages.
    """

    """Read and validate label file"""
    annotations, issues, _ = read_yolo_seg_label(label_path)

    """Extract original valid lines"""
    valid_lines = [annotation.raw_line for annotation in annotations]

    return valid_lines, issues

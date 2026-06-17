from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .config import CLASS_NAMES, IMAGE_EXTS


@dataclass(frozen=True)
class YoloSegAnnotation:
    class_id: int
    coords: tuple[float, ...]
    raw_line: str


def find_image_paths(images_dir: Path, missing_ok: bool = False) -> list[Path]:
    if not images_dir.exists():
        if missing_ok:
            return []
        raise FileNotFoundError(f"Images folder not found: {images_dir}")

    return sorted(
        path
        for path in images_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTS
    )


def find_label_paths(labels_dir: Path, missing_ok: bool = False) -> list[Path]:
    if not labels_dir.exists():
        if missing_ok:
            return []
        raise FileNotFoundError(f"Labels folder not found: {labels_dir}")

    return sorted(labels_dir.glob("*.txt"))


def find_missing_label_images(images: list[Path], labels: list[Path]) -> list[Path]:
    label_stems = {path.stem for path in labels}
    return [path for path in images if path.stem not in label_stems]


def find_orphan_labels(images: list[Path], labels: list[Path]) -> list[Path]:
    image_stems = {path.stem for path in images}
    return [path for path in labels if path.stem not in image_stems]


def validate_yolo_seg_line(
    line: str,
    line_number: int,
) -> tuple[YoloSegAnnotation | None, str | None]:
    stripped = line.strip()
    if not stripped:
        return None, None

    parts = stripped.split()
    if len(parts) < 7:
        return None, f"line {line_number}: segmentation label needs class + at least 3 points"

    try:
        class_id = int(float(parts[0]))
        coords = tuple(float(value) for value in parts[1:])
    except ValueError:
        return None, f"line {line_number}: non-numeric value"

    if class_id not in CLASS_NAMES:
        return None, f"line {line_number}: unsupported class id {class_id}"

    if len(coords) % 2 != 0:
        return None, f"line {line_number}: odd number of polygon coordinates"

    if len(coords) < 6:
        return None, f"line {line_number}: polygon has fewer than 3 points"

    if any(value < 0.0 or value > 1.0 for value in coords):
        return None, f"line {line_number}: coordinates outside [0, 1]"

    return YoloSegAnnotation(class_id=class_id, coords=coords, raw_line=stripped), None


def read_yolo_seg_label(label_path: Path) -> tuple[list[YoloSegAnnotation], list[str], bool]:
    annotations: list[YoloSegAnnotation] = []
    issues: list[str] = []
    has_content = False

    if not label_path.exists():
        return annotations, issues, False

    lines = label_path.read_text(encoding="utf-8").splitlines()
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue

        has_content = True
        annotation, issue = validate_yolo_seg_line(line, line_number)
        if issue is not None:
            issues.append(issue)
            continue

        if annotation is not None:
            annotations.append(annotation)

    return annotations, issues, not has_content


def sanitize_yolo_seg_label(label_path: Path) -> tuple[list[str], list[str]]:
    annotations, issues, _ = read_yolo_seg_label(label_path)
    valid_lines = [annotation.raw_line for annotation in annotations]
    return valid_lines, issues

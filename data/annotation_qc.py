from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from .config import CLASS_NAMES, DEFAULT_IMAGES_DIR, DEFAULT_LABELS_DIR
from .validate import (
    find_image_paths,
    find_label_paths,
    find_missing_label_images,
    find_orphan_labels,
    read_yolo_seg_label,
)

class AnnotationQC:
    def __init__(self, images_dir: Path, labels_dir: Path):
        self.images_dir = Path(images_dir)
        self.labels_dir = Path(labels_dir)

        self.class_counter = Counter()
        self.image_counter = Counter()
        self.bad_lines: list[str] = []
        self.missing_labels: list[Path] = []
        self.empty_labels: list[Path] = []
        self.orphan_labels: list[Path] = []

    def read_annotations(self, label_path: Path):
        annotations, issues, is_empty = read_yolo_seg_label(label_path)

        if is_empty:
            self.empty_labels.append(label_path)

        for issue in issues:
            self.bad_lines.append(f"{label_path} {issue}")

        classes_in_image = set()
        for annotation in annotations:
            self.class_counter[annotation.class_id] += 1
            classes_in_image.add(annotation.class_id)

        for class_id in classes_in_image:
            self.image_counter[class_id] += 1

        return annotations

    def scan_dataset(self):
        images = find_image_paths(self.images_dir)
        labels = find_label_paths(self.labels_dir)

        self.missing_labels = find_missing_label_images(images, labels)
        self.orphan_labels = find_orphan_labels(images, labels)

        for image_path in images:
            label_path = self.labels_dir / f"{image_path.stem}.txt"
            if label_path.exists():
                self.read_annotations(label_path)

        return images

    def print_summary(self, total_images: int):
        total_objects = sum(self.class_counter.values())

        print("\n========== Annotation QC Summary ==========")
        print(f"Total images     : {total_images}")
        print(f"Total objects    : {total_objects}")
        print(f"Missing labels   : {len(self.missing_labels)}")
        print(f"Empty labels     : {len(self.empty_labels)}")
        print(f"Orphan labels    : {len(self.orphan_labels)}")
        print(f"Bad label lines  : {len(self.bad_lines)}")

        print("\nClass distribution:")
        print(f"{'id':<4}{'class':<16}{'objects':>10}{'images':>10}")
        for class_id, class_name in CLASS_NAMES.items():
            print(
                f"{class_id:<4}"
                f"{class_name:<16}"
                f"{self.class_counter[class_id]:>10}"
                f"{self.image_counter[class_id]:>10}"
            )

        if self.missing_labels:
            print("\nMissing label examples:")
            for path in self.missing_labels[:10]:
                print(f"- {path}")

        if self.orphan_labels:
            print("\nOrphan label examples:")
            for path in self.orphan_labels[:10]:
                print(f"- {path}")

        if self.bad_lines:
            print("\nBad label line examples:")
            for item in self.bad_lines[:10]:
                print(f"- {item}")

    def run(self):
        images = self.scan_dataset()
        self.print_summary(total_images=len(images))
        return images


def parse_args():
    parser = argparse.ArgumentParser(description="QC YOLO segmentation labels.")
    parser.add_argument("--images", type=Path, default=DEFAULT_IMAGES_DIR)
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS_DIR)
    return parser.parse_args()


def main():
    args = parse_args()

    checker = AnnotationQC(
        images_dir=args.images,
        labels_dir=args.labels,
    )
    checker.run()


if __name__ == "__main__":
    main()

"""
    Annotation quality control for YOLO segmentation dataset.

    Purpose:
        Scan image-label pairs, check annotation issues, and summarize
        class distribution before training YOLO segmentation model.
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from configs.data import (
    CLASS_NAMES,
    DEFAULT_ANNOTATION_IMAGES,
    DEFAULT_ANNOTATION_LABELS,
)
from data.validate import (
    find_image_paths,
    find_label_paths,
    find_missing_label_images,
    find_orphan_labels,
    read_yolo_seg_label,
)


class AnnotationQC:
    """
        Quality checker for YOLO segmentation annotations.

        Main checks:
            - Missing label files
            - Empty label files
            - Orphan label files
            - Bad label lines
            - Class object distribution
            - Number of images containing each class
    """

    def __init__(self, images_dir: Path, labels_dir: Path):
        """
            Initialize annotation QC checker.
        """

        """Dataset directories"""
        self.images_dir = Path(images_dir)
        self.labels_dir = Path(labels_dir)

        """Count total objects per class"""
        self.class_counter = Counter()

        """Count number of images containing each class"""
        self.image_counter = Counter()

        """Store annotation issues"""
        self.bad_lines: list[str] = []
        self.missing_labels: list[Path] = []
        self.empty_labels: list[Path] = []
        self.orphan_labels: list[Path] = []

    def read_annotations(self, label_path: Path):
        """
            Read and validate one YOLO segmentation label file.

            This function updates:
                - empty label list
                - bad line list
                - class object counter
                - class image counter
        """

        """Read annotations and validation issues from label file"""
        annotations, issues, is_empty = read_yolo_seg_label(label_path)

        """Track empty label file"""
        if is_empty:
            self.empty_labels.append(label_path)

        """Track invalid label lines"""
        for issue in issues:
            self.bad_lines.append(f"{label_path} {issue}")

        """Track classes appearing in this image"""
        classes_in_image = set()

        """Count objects per class"""
        for annotation in annotations:
            self.class_counter[annotation.class_id] += 1
            classes_in_image.add(annotation.class_id)

        """Count images that contain each class"""
        for class_id in classes_in_image:
            self.image_counter[class_id] += 1

        return annotations

    def scan_dataset(self):
        """
            Scan the whole dataset and collect annotation statistics.
        """

        """Find all image and label files"""
        images = find_image_paths(self.images_dir)
        labels = find_label_paths(self.labels_dir)

        """Find image files without labels"""
        self.missing_labels = find_missing_label_images(images, labels)

        """Find label files without matching images"""
        self.orphan_labels = find_orphan_labels(images, labels)

        """Read existing labels for all images"""
        for image_path in images:
            label_path = self.labels_dir / f"{image_path.stem}.txt"

            if label_path.exists():
                self.read_annotations(label_path)

        return images

    def print_summary(self, total_images: int):
        """
            Print dataset quality control summary.
        """

        """Compute total number of annotated objects"""
        total_objects = sum(self.class_counter.values())

        """Print general dataset statistics"""
        print("\n========== Annotation QC Summary ==========")
        print(f"Total images     : {total_images}")
        print(f"Total objects    : {total_objects}")
        print(f"Missing labels   : {len(self.missing_labels)}")
        print(f"Empty labels     : {len(self.empty_labels)}")
        print(f"Orphan labels    : {len(self.orphan_labels)}")
        print(f"Bad label lines  : {len(self.bad_lines)}")

        """Print class distribution table"""
        print("\nClass distribution:")
        print(f"{'id':<4}{'class':<16}{'objects':>10}{'images':>10}")

        for class_id, class_name in CLASS_NAMES.items():
            print(
                f"{class_id:<4}"
                f"{class_name:<16}"
                f"{self.class_counter[class_id]:>10}"
                f"{self.image_counter[class_id]:>10}"
            )

        """Print missing label examples"""
        if self.missing_labels:
            print("\nMissing label examples:")

            for path in self.missing_labels[:10]:
                print(f"- {path}")

        """Print orphan label examples"""
        if self.orphan_labels:
            print("\nOrphan label examples:")

            for path in self.orphan_labels[:10]:
                print(f"- {path}")

        """Print bad label examples"""
        if self.bad_lines:
            print("\nBad label line examples:")

            for item in self.bad_lines[:10]:
                print(f"- {item}")

    def run(self):
        """
            Run full annotation QC process.
        """

        """Scan dataset and collect statistics"""
        images = self.scan_dataset()

        """Print final summary"""
        self.print_summary(total_images=len(images))

        return images


def parse_args():
    """
        Parse command line arguments.
    """

    parser = argparse.ArgumentParser(
        description="QC YOLO segmentation labels."
    )

    """Input dataset paths"""
    parser.add_argument("--images", type=Path, default=DEFAULT_ANNOTATION_IMAGES)
    parser.add_argument("--labels", type=Path, default=DEFAULT_ANNOTATION_LABELS)

    return parser.parse_args()


def main():
    """
        Main function for annotation quality control.
    """

    """Parse arguments"""
    args = parse_args()

    """Create annotation checker"""
    checker = AnnotationQC(
        images_dir=args.images,
        labels_dir=args.labels,
    )

    """Run quality check"""
    checker.run()


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import random
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parent.parent

DEFAULT_IMAGES_DIR    = BASE_DIR / "dataset" / "processed" / "train" / "images"
DEFAULT_LABELS_DIR    = BASE_DIR / "dataset" / "processed" / "train" / "labels"
DEFAULT_VISUALIZE_DIR = BASE_DIR / "data" / "visualize"

CLASS_NAMES = {
    0: "crack",
    1: "porosity",
    2: "spatter",
    3: "welding_line",
}

CLASS_COLORS = {
    0: (0, 0, 255),        # red
    1: (0, 255, 255),      # yellow
    2: (200, 120, 240),    # pink/purple
    3: (0, 255, 0),        # green
}

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


class AnnotationQC:
    def __init__(
        self,
        images_dir: Path,
        labels_dir: Path,
        visualize_dir: Path,
        n_samples: int = 50,
        display_size: tuple[int, int] = (960, 720),
        class_id: int = -1,
        show_text: bool = True,
        show_window: bool = True,
    ):
        self.images_dir = Path(images_dir)
        self.labels_dir = Path(labels_dir)
        self.visualize_dir = Path(visualize_dir)
        self.n_samples = n_samples
        self.display_size = display_size
        self.class_id = class_id
        self.show_text = show_text
        self.show_window = show_window

        self.class_counter = Counter()
        self.image_counter = Counter()
        self.bad_lines: list[str] = []
        self.missing_labels: list[Path] = []
        self.empty_labels: list[Path] = []
        self.orphan_labels: list[Path] = []

    def image_paths(self) -> list[Path]:
        if not self.images_dir.exists():
            raise FileNotFoundError(f"Images folder not found: {self.images_dir}")

        return sorted(
            path
            for path in self.images_dir.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_EXTS
        )

    def label_paths(self) -> list[Path]:
        if not self.labels_dir.exists():
            raise FileNotFoundError(f"Labels folder not found: {self.labels_dir}")

        return sorted(self.labels_dir.glob("*.txt"))

    def parse_label_line(self, line: str, label_path: Path, line_number: int):
        parts = line.strip().split()

        if len(parts) < 7:
            self.bad_lines.append(
                f"{label_path} line {line_number}: need class + at least 3 polygon points"
            )
            return None, None

        try:
            cls_id = int(float(parts[0]))
            coords = [float(value) for value in parts[1:]]
        except ValueError:
            self.bad_lines.append(f"{label_path} line {line_number}: non-numeric value")
            return None, None

        if cls_id not in CLASS_NAMES:
            self.bad_lines.append(f"{label_path} line {line_number}: unknown class id {cls_id}")
            return None, None

        if len(coords) % 2 != 0:
            self.bad_lines.append(f"{label_path} line {line_number}: odd coordinate count")
            return None, None

        if len(coords) < 6:
            self.bad_lines.append(f"{label_path} line {line_number}: polygon has fewer than 3 points")
            return None, None

        if any(value < 0.0 or value > 1.0 for value in coords):
            self.bad_lines.append(f"{label_path} line {line_number}: coordinate outside [0, 1]")
            return None, None

        polygon_01 = np.array(coords, dtype=np.float32).reshape(-1, 2)
        return cls_id, polygon_01

    def normalize_to_pixel(self, polygon_01: np.ndarray, width: int, height: int) -> np.ndarray:
        polygon = polygon_01.copy()
        polygon[:, 0] = np.clip(polygon[:, 0], 0.0, 1.0) * width
        polygon[:, 1] = np.clip(polygon[:, 1], 0.0, 1.0) * height
        return polygon.astype(np.int32)

    def overlay_polygon(
        self,
        image: np.ndarray,
        polygon_pixel: np.ndarray,
        color: tuple[int, int, int],
        mask_alpha: float = 0.35,
        line_alpha: float = 0.55,
        thickness: int = 1,
    ) -> np.ndarray:
        pts = polygon_pixel.reshape(-1, 1, 2)

        mask_layer = np.zeros_like(image, dtype=np.uint8)
        cv2.fillPoly(mask_layer, [pts], color)

        mask_binary = np.zeros(image.shape[:2], dtype=np.uint8)
        cv2.fillPoly(mask_binary, [pts], 255)

        colored = cv2.addWeighted(image, 1.0, mask_layer, mask_alpha, 0)
        blended = image.copy()
        blended[mask_binary > 0] = colored[mask_binary > 0]

        line_layer = blended.copy()
        cv2.polylines(
            line_layer,
            [pts],
            isClosed=True,
            color=color,
            thickness=thickness,
            lineType=cv2.LINE_AA,
        )

        return cv2.addWeighted(blended, 1.0 - line_alpha, line_layer, line_alpha, 0)

    def read_annotations(self, label_path: Path):
        if not label_path.exists():
            return []

        lines = [line for line in label_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if not lines:
            self.empty_labels.append(label_path)
            return []

        annotations = []
        classes_in_image = set()

        for line_number, line in enumerate(lines, start=1):
            cls_id, polygon_01 = self.parse_label_line(line, label_path, line_number)
            if polygon_01 is None:
                continue

            self.class_counter[cls_id] += 1
            classes_in_image.add(cls_id)
            annotations.append((cls_id, polygon_01))

        for cls_id in classes_in_image:
            self.image_counter[cls_id] += 1

        return annotations

    def scan_dataset(self):
        images = self.image_paths()
        labels = self.label_paths()

        image_stems = {path.stem for path in images}
        label_stems = {path.stem for path in labels}

        self.missing_labels = [path for path in images if path.stem not in label_stems]
        self.orphan_labels = [path for path in labels if path.stem not in image_stems]

        for image_path in images:
            label_path = self.labels_dir / f"{image_path.stem}.txt"
            if label_path.exists():
                self.read_annotations(label_path)

        return images

    def draw_sample(self, image_path: Path) -> np.ndarray | None:
        image = cv2.imread(str(image_path))
        if image is None:
            return None

        height, width = image.shape[:2]
        label_path = self.labels_dir / f"{image_path.stem}.txt"

        annotations = self.read_annotations(label_path)

        for cls_id, polygon_01 in annotations:
            if self.class_id != -1 and cls_id != self.class_id:
                continue

            polygon_pixel = self.normalize_to_pixel(polygon_01, width, height)
            color = CLASS_COLORS.get(cls_id, (255, 255, 255))
            class_name = CLASS_NAMES.get(cls_id, f"class_{cls_id}")

            image = self.overlay_polygon(
                image,
                polygon_pixel,
                color=color,
                mask_alpha=0.35,
                line_alpha=0.45,
                thickness=1,
            )

            if self.show_text:
                x, y = polygon_pixel[0]
                cv2.putText(
                    image,
                    class_name,
                    (int(x), int(y)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    color,
                    2,
                    cv2.LINE_AA,
                )

        return cv2.resize(image, self.display_size)

    def plot_class_distribution(self):
        self.visualize_dir.mkdir(parents=True, exist_ok=True)

        class_ids = sorted(CLASS_NAMES.keys())
        names = [CLASS_NAMES[class_id] for class_id in class_ids]
        object_counts = [self.class_counter[class_id] for class_id in class_ids]
        image_counts = [self.image_counter[class_id] for class_id in class_ids]

        x = np.arange(len(names))
        width = 0.38

        plt.figure(figsize=(10, 6))
        plt.bar(x - width / 2, object_counts, width, label="objects")
        plt.bar(x + width / 2, image_counts, width, label="images with class")

        plt.xticks(x, names, rotation=20, ha="right")
        plt.ylabel("Count")
        plt.title("YOLO Segmentation Class Distribution")
        plt.legend()
        plt.tight_layout()

        output_path = self.visualize_dir / "class_distribution.png"
        plt.savefig(output_path, dpi=160)
        plt.close()

        print(f"\nSaved chart: {output_path}")

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
        for cls_id, class_name in CLASS_NAMES.items():
            print(
                f"{cls_id:<4}"
                f"{class_name:<16}"
                f"{self.class_counter[cls_id]:>10}"
                f"{self.image_counter[cls_id]:>10}"
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

    def show_random_samples(self, images: list[Path]):
        if not self.show_window:
            return

        if not images:
            print("No images to display.")
            return

        samples = random.sample(images, min(self.n_samples, len(images)))

        for image_path in samples:
            image = self.draw_sample(image_path)
            if image is None:
                continue

            cv2.imshow("Annotation QC", image)
            key = cv2.waitKey(0) & 0xFF
            cv2.destroyWindow("Annotation QC")

            if key == 27 or key == ord("q"):
                break

        cv2.destroyAllWindows()

    def run(self):
        images = self.scan_dataset()
        self.print_summary(total_images=len(images))
        self.plot_class_distribution()
        self.show_random_samples(images)


def parse_args():
    parser = argparse.ArgumentParser(description="QC YOLO segmentation labels.")
    parser.add_argument("--images", type=Path, default=DEFAULT_IMAGES_DIR)
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS_DIR)
    parser.add_argument("--visualize-dir", type=Path, default=DEFAULT_VISUALIZE_DIR)
    parser.add_argument("--samples", type=int, default=50)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=640)
    parser.add_argument("--class-id", type=int, default=-1, help="-1 means show all classes.")
    parser.add_argument("--no-window", action="store_true", help="Only print summary and save chart.")
    parser.add_argument("--no-text", action="store_true", help="Hide class text on displayed images.")
    return parser.parse_args()


def main():
    args = parse_args()

    checker = AnnotationQC(
        images_dir=args.images,
        labels_dir=args.labels,
        visualize_dir=args.visualize_dir,
        n_samples=args.samples,
        display_size=(args.width, args.height),
        class_id=args.class_id,
        show_text=not args.no_text,
        show_window=not args.no_window,
    )
    checker.run()


if __name__ == "__main__":
    main()
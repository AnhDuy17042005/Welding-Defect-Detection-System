from __future__ import annotations

import argparse
import os
import random
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

os.environ.setdefault("MPLCONFIGDIR", str(Path("/tmp") / "matplotlib"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from annotation_qc import AnnotationQC
from config import (
    CLASS_COLORS,
    CLASS_NAMES,
    DEFAULT_IMAGES_DIR,
    DEFAULT_LABELS_DIR,
    DEFAULT_VISUALIZE_DIR,
)
from validate import read_yolo_seg_label


class AnnotationVisualizer:
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

    def normalize_to_pixel(self, coords: tuple[float, ...], width: int, height: int) -> np.ndarray:
        polygon = np.array(coords, dtype=np.float32).reshape(-1, 2)
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

    def draw_sample(self, image_path: Path) -> np.ndarray | None:
        image = cv2.imread(str(image_path))
        if image is None:
            return None

        height, width = image.shape[:2]
        label_path = self.labels_dir / f"{image_path.stem}.txt"
        annotations, _, _ = read_yolo_seg_label(label_path)

        for annotation in annotations:
            class_id = annotation.class_id
            if self.class_id != -1 and class_id != self.class_id:
                continue

            polygon_pixel = self.normalize_to_pixel(annotation.coords, width, height)
            color = CLASS_COLORS.get(class_id, (255, 255, 255))
            class_name = CLASS_NAMES.get(class_id, f"class_{class_id}")

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

    def plot_class_distribution(
        self,
        class_counter: Counter,
        image_counter: Counter,
    ) -> None:
        self.visualize_dir.mkdir(parents=True, exist_ok=True)

        class_ids = sorted(CLASS_NAMES.keys())
        names = [CLASS_NAMES[class_id] for class_id in class_ids]
        object_counts = [class_counter[class_id] for class_id in class_ids]
        image_counts = [image_counter[class_id] for class_id in class_ids]

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

    def show_random_samples(self, images: list[Path]) -> None:
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
        checker = AnnotationQC(images_dir=self.images_dir, labels_dir=self.labels_dir)
        images = checker.scan_dataset()
        checker.print_summary(total_images=len(images))

        self.plot_class_distribution(
            class_counter=checker.class_counter,
            image_counter=checker.image_counter,
        )
        self.show_random_samples(images)


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize YOLO segmentation labels.")
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

    visualizer = AnnotationVisualizer(
        images_dir=args.images,
        labels_dir=args.labels,
        visualize_dir=args.visualize_dir,
        n_samples=args.samples,
        display_size=(args.width, args.height),
        class_id=args.class_id,
        show_text=not args.no_text,
        show_window=not args.no_window,
    )
    visualizer.run()


if __name__ == "__main__":
    main()

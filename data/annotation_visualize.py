"""
    Visualize YOLO segmentation annotations.

    Purpose:
        1. Check annotation quality by reusing AnnotationQC.
        2. Draw YOLO polygon labels on images.
        3. Save random annotated samples for visual inspection.
        4. Plot class distribution chart.
"""

from __future__ import annotations

import argparse
import os
import random
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

"""Set matplotlib cache directory for Linux/Colab environment"""
os.environ.setdefault("MPLCONFIGDIR", str(Path("/tmp") / "matplotlib"))

"""Use non-GUI backend to save plots without opening window"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from configs.data import (
    CLASS_NAMES,
    DEFAULT_ANNOTATION_IMAGES,
    DEFAULT_ANNOTATION_LABELS,
)
from configs.visualize import (
    ANNOTATION_DISPLAY_SIZE,
    ANNOTATION_SAMPLE_COUNT,
    ANNOTATION_VISUALIZE_DIR,
    CLASS_COLORS,
)
from data.annotation_qc import AnnotationQC
from data.validate import read_yolo_seg_label


class AnnotationVisualizer:
    """
        Visualizer for YOLO segmentation labels.

        Main functions:
            - Overlay polygon masks on images
            - Draw class names
            - Save random visualization samples
            - Plot class distribution chart
    """

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
        """
            Initialize annotation visualizer.
        """

        """Dataset paths"""
        self.images_dir = Path(images_dir)
        self.labels_dir = Path(labels_dir)

        """Output visualization path"""
        self.visualize_dir = Path(visualize_dir)

        """Number of random samples to save"""
        self.n_samples = n_samples

        """Output image display size"""
        self.display_size = display_size

        """Class filter: -1 means visualize all classes"""
        self.class_id = class_id

        """Whether to draw class name on image"""
        self.show_text = show_text

        """Reserved flag for window display"""
        self.show_window = show_window

    def normalize_to_pixel(
        self,
        coords: tuple[float, ...],
        width: int,
        height: int,
    ) -> np.ndarray:
        """
            Convert normalized YOLO polygon coordinates to pixel coordinates.

            YOLO segmentation format:
                x, y in range [0, 1]

            Output:
                polygon points in pixel coordinates.
        """

        """Convert flat coordinate list to Nx2 polygon array"""
        polygon = np.array(coords, dtype=np.float32).reshape(-1, 2)

        """Convert normalized x coordinates to pixel x coordinates"""
        polygon[:, 0] = np.clip(polygon[:, 0], 0.0, 1.0) * width

        """Convert normalized y coordinates to pixel y coordinates"""
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
        """
            Overlay one polygon annotation on image.

            The polygon is drawn with:
                - semi-transparent filled mask
                - contour line around polygon
        """

        """Reshape polygon for OpenCV drawing functions"""
        pts = polygon_pixel.reshape(-1, 1, 2)

        """Create colored mask layer"""
        mask_layer = np.zeros_like(image, dtype=np.uint8)
        cv2.fillPoly(mask_layer, [pts], color)

        """Create binary mask for selecting polygon region"""
        mask_binary = np.zeros(image.shape[:2], dtype=np.uint8)
        cv2.fillPoly(mask_binary, [pts], 255)

        """Blend colored mask with original image"""
        colored = cv2.addWeighted(image, 1.0, mask_layer, mask_alpha, 0)

        """Apply blended color only inside polygon region"""
        blended = image.copy()
        blended[mask_binary > 0] = colored[mask_binary > 0]

        """Draw polygon contour line"""
        line_layer = blended.copy()
        cv2.polylines(
            line_layer,
            [pts],
            isClosed=True,
            color=color,
            thickness=thickness,
            lineType=cv2.LINE_AA,
        )

        """Blend contour line with masked image"""
        return cv2.addWeighted(
            blended,
            1.0 - line_alpha,
            line_layer,
            line_alpha,
            0
        )

    def draw_sample(self, image_path: Path) -> np.ndarray | None:
        """
            Draw annotations for one image.

            Returns:
                Resized annotated image.
                None if image cannot be loaded.
        """

        """Read image"""
        image = cv2.imread(str(image_path))

        if image is None:
            return None

        """Get image size"""
        height, width = image.shape[:2]

        """Read corresponding label file"""
        label_path = self.labels_dir / f"{image_path.stem}.txt"
        annotations, _, _ = read_yolo_seg_label(label_path)

        """Draw each annotation"""
        for annotation in annotations:
            class_id = annotation.class_id

            """Skip other classes if class filter is enabled"""
            if self.class_id != -1 and class_id != self.class_id:
                continue

            """Convert normalized polygon to pixel polygon"""
            polygon_pixel = self.normalize_to_pixel(
                annotation.coords,
                width,
                height
            )

            """Get class color and class name"""
            color = CLASS_COLORS.get(class_id, (255, 255, 255))
            class_name = CLASS_NAMES.get(class_id, f"class_{class_id}")

            """Overlay polygon mask"""
            image = self.overlay_polygon(
                image,
                polygon_pixel,
                color=color,
                mask_alpha=0.35,
                line_alpha=0.45,
                thickness=1,
            )

            """Draw class text on first polygon point"""
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

        """Resize visualization image"""
        return cv2.resize(image, self.display_size)

    def plot_class_distribution(
        self,
        class_counter: Counter,
        image_counter: Counter,
    ) -> None:
        """
            Plot class distribution chart.

            Chart shows:
                - number of annotated objects per class
                - number of images containing each class
        """

        """Create output directory"""
        self.visualize_dir.mkdir(parents=True, exist_ok=True)

        """Prepare class names and counters"""
        class_ids = sorted(CLASS_NAMES.keys())
        names = [CLASS_NAMES[class_id] for class_id in class_ids]
        object_counts = [class_counter[class_id] for class_id in class_ids]
        image_counts = [image_counter[class_id] for class_id in class_ids]

        """Prepare bar positions"""
        x = np.arange(len(names))
        width = 0.38

        """Create bar chart"""
        plt.figure(figsize=(10, 6))
        plt.bar(x - width / 2, object_counts, width, label="objects")
        plt.bar(x + width / 2, image_counts, width, label="images with class")

        """Chart formatting"""
        plt.xticks(x, names, rotation=20, ha="right")
        plt.ylabel("Count")
        plt.title("YOLO Segmentation Class Distribution")
        plt.legend()
        plt.tight_layout()

        """Save chart image"""
        output_path = self.visualize_dir / "class_distribution.png"
        plt.savefig(output_path, dpi=160)
        plt.close()

        print(f"\nSaved chart: {output_path}")

    def show_random_samples(self, images):
        """
            Randomly select images and save annotated visualization samples.
        """

        """Create output folders"""
        self.visualize_dir.mkdir(parents=True, exist_ok=True)

        samples_dir = self.visualize_dir / "samples"
        samples_dir.mkdir(parents=True, exist_ok=True)

        """Randomly sample images"""
        samples = random.sample(images, min(self.n_samples, len(images)))

        """Draw and save each sample"""
        for idx, img_path in enumerate(samples, start=1):
            image = self.draw_sample(img_path)

            """Skip unreadable image"""
            if image is None:
                print(f"Cannot read image or draw annotation: {img_path}")
                continue

            """Save visualization sample"""
            out_path = samples_dir / f"sample_{idx:03d}_{img_path.stem}.jpg"
            cv2.imwrite(str(out_path), image)

            print(f"Saved sample visualization: {out_path}")

    def run(self):
        """
            Run annotation visualization pipeline.

            Steps:
                1. Run annotation QC
                2. Print QC summary
                3. Save class distribution chart
                4. Save random annotation samples
        """

        """Run annotation quality control"""
        checker = AnnotationQC(
            images_dir=self.images_dir,
            labels_dir=self.labels_dir
        )

        images = checker.scan_dataset()

        """Print annotation QC summary"""
        checker.print_summary(total_images=len(images))

        """Save class distribution chart"""
        self.plot_class_distribution(
            class_counter=checker.class_counter,
            image_counter=checker.image_counter,
        )

        """Save random annotated samples"""
        self.show_random_samples(images)


def parse_args():
    """
        Parse command line arguments.
    """

    parser = argparse.ArgumentParser(
        description="Visualize YOLO segmentation labels."
    )

    """Input dataset paths"""
    parser.add_argument("--images", type=Path, default=DEFAULT_ANNOTATION_IMAGES)
    parser.add_argument("--labels", type=Path, default=DEFAULT_ANNOTATION_LABELS)

    """Output visualization path"""
    parser.add_argument(
        "--visualize-dir",
        type=Path,
        default=ANNOTATION_VISUALIZE_DIR,
    )

    """Visualization settings"""
    parser.add_argument("--samples", type=int, default=ANNOTATION_SAMPLE_COUNT)
    parser.add_argument("--width", type=int, default=ANNOTATION_DISPLAY_SIZE[0])
    parser.add_argument("--height", type=int, default=ANNOTATION_DISPLAY_SIZE[1])

    """Class filter: -1 means show all classes"""
    parser.add_argument(
        "--class-id",
        type=int,
        default=-1,
        help="-1 means show all classes."
    )

    """Disable window display"""
    parser.add_argument(
        "--no-window",
        action="store_true",
        help="Only print summary and save chart."
    )

    """Hide class text on visualization image"""
    parser.add_argument(
        "--no-text",
        action="store_true",
        help="Hide class text on displayed images."
    )

    return parser.parse_args()


def main():
    """
        Main function for annotation visualization.
    """

    """Parse arguments"""
    args = parse_args()

    """Create annotation visualizer"""
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

    """Run visualization"""
    visualizer.run()


if __name__ == "__main__":
    main()

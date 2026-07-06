import cv2
import numpy as np
import random
import albumentations as A
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

from configs.yolo import (
    YOLO_AUGMENT_INPUT_IMAGES as INPUT_IMAGES,
    YOLO_AUGMENT_INPUT_LABELS as INPUT_LABELS,
    YOLO_AUGMENT_OUTPUT_IMAGES as OUTPUT_IMAGES,
    YOLO_AUGMENT_OUTPUT_LABELS as OUTPUT_LABELS,
    YOLO_AUGMENT_PARAMETERS as PARAMS,
    YOLO_AUGMENT_PROBABILITIES as PROBABILITY,
    YOLO_AUGMENT_SCALE as SCALE_DATASET,
    YOLO_GEOMETRIC_AUGMENTATIONS as GEOMETRIC,
    YOLO_PHOTOMETRIC_AUGMENTATIONS as PHOTOMETRIC,
)

OUTPUT_IMAGES.mkdir(parents=True, exist_ok=True)
OUTPUT_LABELS.mkdir(parents=True, exist_ok=True)

class GeometryAugmentor:
    """
        Apply geometric augmentation to image and polygon labels.

        The polygon coordinates are normalized in YOLO segmentation format:
            x, y in range [0, 1]

        When image geometry changes, polygon points must be transformed
        consistently with the image.
    """

    def __init__(self, config, prob, params):
        """
            Store geometric augmentation configs.
        """

        self.cfg = config
        self.prob = prob
        self.param = params

    def flip_h(self, img, polys):
        """
            Flip image horizontally and update polygon x coordinates.

            Formula:
                new_x = 1 - old_x
                y stays unchanged
        """

        """Flip image left-right"""
        img = cv2.flip(img, 1)

        """Update polygon coordinates"""
        new_polys = [[[1 - x, y] for x, y in poly] for poly in polys]

        return img, new_polys

    def flip_v(self, img, polys):
        """
            Flip image vertically and update polygon y coordinates.

            Formula:
                x stays unchanged
                new_y = 1 - old_y
        """

        """Flip image top-bottom"""
        img = cv2.flip(img, 0)

        """Update polygon coordinates"""
        new_polys = [[[x, 1 - y] for x, y in poly] for poly in polys]

        return img, new_polys

    def rotate(self, img, polys, angle):
        """
            Rotate image around image center and update polygon coordinates.
        """

        """Get image size"""
        h, w = img.shape[:2]

        """Create rotation matrix"""
        M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1)

        """Rotate image"""
        img = cv2.warpAffine(img, M, (w, h))

        new_polys = []

        """Transform each polygon point"""
        for poly in polys:
            new_poly = []

            for x, y in poly:
                """Convert normalized coordinates to pixel coordinates"""
                px = x * w
                py = y * h

                """Apply affine rotation"""
                rx = M[0, 0] * px + M[0, 1] * py + M[0, 2]
                ry = M[1, 0] * px + M[1, 1] * py + M[1, 2]

                """Convert back to normalized coordinates and clip to [0, 1]"""
                new_poly.append([
                    np.clip(rx / w, 0, 1),
                    np.clip(ry / h, 0, 1),
                ])

            new_polys.append(new_poly)

        return img, new_polys

    def crop(self, img, polys, crop_ratio):
        """
            Randomly crop image and keep only polygon points inside crop area.

            Note:
                If a polygon has fewer than 3 valid points after crop,
                it will be discarded.
        """

        """Get image size"""
        h, w = img.shape[:2]

        """Random crop size"""
        cr = random.uniform(*crop_ratio)
        cw, ch = int(w * cr), int(h * cr)

        """Random crop position"""
        cx = random.randint(0, w - cw)
        cy = random.randint(0, h - ch)

        """Crop image"""
        cropped = img[cy:cy + ch, cx:cx + cw]
        new_polys = []

        """Transform polygon points into cropped image coordinate system"""
        for poly in polys:
            new_poly = []

            for x, y in poly:
                """Convert normalized point to pixel point"""
                px = x * w
                py = y * h

                """Keep point only if it is inside crop area"""
                if cx <= px <= cx + cw and cy <= py <= cy + ch:
                    nx = (px - cx) / cw
                    ny = (py - cy) / ch
                    new_poly.append([nx, ny])

            """Keep valid polygon only"""
            if len(new_poly) >= 3:
                new_polys.append(new_poly)

        """Resize cropped image back to original size"""
        cropped = cv2.resize(cropped, (w, h))

        return cropped, new_polys

    def scale(self, img, polys, scale_range):
        """
            Randomly scale image around the center and update polygons.
        """

        """Get image size"""
        h, w = img.shape[:2]

        """Random scale factors"""
        sx = random.uniform(*scale_range)
        sy = random.uniform(*scale_range)

        """Resize image"""
        new_w = int(w * sx)
        new_h = int(h * sy)
        img_scaled = cv2.resize(img, (new_w, new_h))

        new_polys = []

        """Scale polygon points around image center"""
        for poly in polys:
            new_poly = []

            for x, y in poly:
                nx = np.clip((x - 0.5) * sx + 0.5, 0, 1)
                ny = np.clip((y - 0.5) * sy + 0.5, 0, 1)
                new_poly.append([nx, ny])

            new_polys.append(new_poly)

        """Create empty canvas with original image size"""
        canvas = np.zeros_like(img)

        """Compute center offset"""
        dx = (new_w - w) // 2
        dy = (new_h - h) // 2

        """Paste scaled image into original-size canvas"""
        canvas[
            max(0, -dy):max(0, -dy) + min(h, new_h),
            max(0, -dx):max(0, -dx) + min(w, new_w),
        ] = img_scaled[
            max(0, dy):max(0, dy) + min(h, new_h),
            max(0, dx):max(0, dx) + min(w, new_w),
        ]

        return canvas, new_polys

    def shear(self, img, polys, shear_range):
        """
            Apply shear transform and update polygon coordinates.
        """

        """Get image size"""
        h, w = img.shape[:2]

        """Random shear factors"""
        shx = random.uniform(*shear_range)
        shy = random.uniform(*shear_range)

        """Image center"""
        cx, cy = w / 2, h / 2

        """Create shear matrix around image center"""
        M = np.float32([
            [1, shx, -shx * cy],
            [shy, 1, -shy * cx],
        ])

        """Apply shear to image"""
        img = cv2.warpAffine(img, M, (w, h))

        new_polys = []

        """Transform each polygon point"""
        for poly in polys:
            new_poly = []

            for x, y in poly:
                """Convert normalized coordinates to pixel coordinates"""
                px = x * w
                py = y * h

                """Apply shear transform"""
                sx = M[0, 0] * px + M[0, 1] * py + M[0, 2]
                sy = M[1, 0] * px + M[1, 1] * py + M[1, 2]

                """Convert back to normalized coordinates"""
                new_poly.append([
                    np.clip(sx / w, 0, 1),
                    np.clip(sy / h, 0, 1),
                ])

            new_polys.append(new_poly)

        return img, new_polys

    def apply(self, img, polys):
        """
            Apply enabled geometric augmentations by probability.
        """

        """Horizontal flip"""
        if self.cfg["flip_horizontal"] and random.random() < self.prob["flip_horizontal"]:
            img, polys = self.flip_h(img, polys)

        """Vertical flip"""
        if self.cfg["flip_vertical"] and random.random() < self.prob["flip_vertical"]:
            img, polys = self.flip_v(img, polys)

        """Rotation"""
        if self.cfg["rotation"] and random.random() < self.prob["rotation"]:
            angle = random.uniform(*self.param["rotation"])
            img, polys = self.rotate(img, polys, angle)

        """Random crop"""
        if self.cfg["crop"] and random.random() < self.prob["crop"]:
            img, polys = self.crop(img, polys, self.param["crop_ratio"])

        """Random scale"""
        if self.cfg["scale"] and random.random() < self.prob["scale"]:
            img, polys = self.scale(img, polys, self.param["scale"])

        """Random shear"""
        if self.cfg["shear"] and random.random() < self.prob["shear"]:
            img, polys = self.shear(img, polys, self.param["shear"])

        return img, polys


class PhotometricAugmentor:
    """
        Apply photometric augmentation to image only.

        These transforms change image appearance but do not change geometry,
        so polygon labels do not need to be updated.
    """

    def __init__(self, config, prob, params):
        """
            Build Albumentations photometric pipeline.
        """

        self.prob = prob
        self.params = params
        trans = []

        """Blur augmentation"""
        if config["blur"]:
            trans.append(
                A.Blur(
                    p=self.prob["blur"],
                    blur_limit=params["blur"],
                )
            )

        """Brightness and contrast augmentation"""
        if config["brightness_contrast"]:
            trans.append(
                A.RandomBrightnessContrast(
                    brightness_limit=self.params["brightness_limit"],
                    contrast_limit=self.params["contrast_limit"],
                    p=self.prob["brightness_contrast"],
                )
            )

        """Gaussian noise augmentation"""
        if config["gaussian_noise"]:
            trans.append(
                A.GaussNoise(
                    std_range=self.params["std_range"],
                    p=self.prob["gaussian_noise"],
                )
            )

        """Hue, saturation and value augmentation"""
        if config["hue_saturation"]:
            trans.append(
                A.HueSaturationValue(
                    hue_shift_limit=self.params["hue_shift_limit"],
                    sat_shift_limit=self.params["sat_shift_limit"],
                    val_shift_limit=self.params["val_shift_limit"],
                    p=self.prob["hue_saturation"],
                )
            )

        """Compose all enabled photometric transforms"""
        self.pipeline = A.Compose(trans)

    def apply(self, img):
        """
            Apply photometric pipeline to one image.
        """

        return self.pipeline(image=img)["image"]


class DatasetAugmentor:
    """
        Generate augmented dataset for YOLO segmentation.

        Input:
            images folder
            labels folder with YOLO polygon .txt files

        Output:
            augmented images
            augmented polygon labels
    """

    def __init__(
        self,
        input_img,
        input_lbl,
        out_img,
        out_lbl,
        scale,
        geo_cfg,
        photo_cfg,
        prob,
        params,
    ):
        """
            Initialize dataset augmentation pipeline.
        """

        """Input paths"""
        self.in_img = Path(input_img)
        self.in_lbl = Path(input_lbl)

        """Output paths"""
        self.out_img = Path(out_img)
        self.out_lbl = Path(out_lbl)

        """Number of output samples per original image"""
        self.scale = scale

        """Build augmentors"""
        self.geo = GeometryAugmentor(geo_cfg, prob, params)
        self.photo = PhotometricAugmentor(photo_cfg, prob, params)

        """Create output directories"""
        self.out_img.mkdir(parents=True, exist_ok=True)
        self.out_lbl.mkdir(parents=True, exist_ok=True)

    def read_polygons(self, path):
        """
            Read YOLO segmentation polygon label file.

            Label format:
                class_id x1 y1 x2 y2 x3 y3 ...

            Returns:
                polys:
                    List of polygon points.

                labels:
                    List of class ids.
        """

        polys, labels = [], []

        """Return empty labels if label file does not exist"""
        if not path.exists():
            return polys, labels

        """Read polygon labels line by line"""
        with open(path) as f:
            for line in f:
                parts = line.split()

                """Read class id"""
                cls = int(parts[0])

                """Read polygon coordinates"""
                coords = list(map(float, parts[1:]))

                """Convert flat coordinates to point list"""
                pts = [
                    [coords[i], coords[i + 1]]
                    for i in range(0, len(coords), 2)
                ]

                """Keep valid polygon only"""
                if len(pts) >= 3:
                    polys.append(pts)
                    labels.append(cls)

        return polys, labels

    def write_polygons(self, path, polys, labels):
        """
            Write polygon labels to YOLO segmentation format.
        """

        with open(path, "w") as f:
            for cls, poly in zip(labels, polys):
                flat = []

                """Flatten polygon points"""
                for x, y in poly:
                    flat.append(f"{x:.6f}")
                    flat.append(f"{y:.6f}")

                """Write one polygon object per line"""
                f.write(f"{cls} " + " ".join(flat) + "\n")

    def main(self):
        """
            Run augmentation for all images in the input dataset.
        """

        index = 0
        images = []

        """Collect image files"""
        for ext in ["*.jpg", "*.png", "*.jpeg"]:
            images.extend(self.in_img.glob(ext))

        images = sorted(images)

        print(f"Found {len(images)} images")

        """Process each image"""
        for img_path in images:
            lbl_path = self.in_lbl / f"{img_path.stem}.txt"

            """Read polygon labels"""
            polys, labels = self.read_polygons(lbl_path)

            """Read image"""
            img = cv2.imread(str(img_path))

            index += 1

            """Save original image and label into augmented dataset"""
            cv2.imwrite(
                str(self.out_img / f"{img_path.stem}_original.jpg"),
                img
            )

            self.write_polygons(
                self.out_lbl / f"{img_path.stem}_original.txt",
                polys,
                labels,
            )

            """Generate augmented samples"""
            for i in range(self.scale - 1):
                aug_img = img.copy()

                """Copy polygon labels before augmentation"""
                aug_poly = [p.copy() for p in polys]

                """Apply geometric transform to image and polygons"""
                aug_img, aug_poly = self.geo.apply(aug_img, aug_poly)

                """Apply photometric transform to image only"""
                aug_img = self.photo.apply(aug_img)

                """Save augmented image"""
                cv2.imwrite(
                    str(self.out_img / f"{img_path.stem}_augment{i}.jpg"),
                    aug_img,
                )

                """Save augmented polygon label"""
                self.write_polygons(
                    self.out_lbl / f"{img_path.stem}_augment{i}.txt",
                    aug_poly,
                    labels,
                )

                print(f"AUGMENT IMAGES: {index}")

        print("DONE AUGMENT")


if __name__ == "__main__":
    """
        Run dataset augmentation.
    """

    aug = DatasetAugmentor(
        INPUT_IMAGES,
        INPUT_LABELS,
        OUTPUT_IMAGES,
        OUTPUT_LABELS,
        SCALE_DATASET,
        GEOMETRIC,
        PHOTOMETRIC,
        PROBABILITY,
        PARAMS,
    )

    aug.main()

import cv2
import numpy as np
import random
import albumentations as A
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

INPUT_IMAGES  = ROOT / "dataset" / "finaldata" / "train" / "images"
INPUT_LABELS  = ROOT / "dataset" / "finaldata" / "train" / "labels"
OUTPUT_IMAGES = ROOT / "dataset" / "augmented" / "train" / "images"
OUTPUT_LABELS = ROOT / "dataset" / "augmented" / "train" / "labels"

# OUTPUT CONFIGS
OUTPUT_IMAGES.mkdir(parents=True, exist_ok=True)
OUTPUT_LABELS.mkdir(parents=True, exist_ok=True)

# DATASET SIZE CONFIGS
SCALE_DATASET = 7

# AUGMENT GROUP CONFIGS
GEOMETRIC = {
    "flip_horizontal":      True,
    "flip_vertical":        False,
    "rotation":             True,
    "crop":                 False,
    "scale":                True,
    "shear":                False,
}

PHOTOMETRIC = {
    "blur":                 True,
    "brightness_contrast":  True,
    "gaussian_noise":       True,
    "hue_saturation":       False,
}

# AUGMENT PROBABILITY CONFIGS
PROBABILITY = {
    "blur":                 0.30,
    "brightness_contrast":  0.60,
    "gaussian_noise":       0.4,
    "hue_saturation":       0.0,

    "flip_horizontal":      0.5,
    "flip_vertical":        0.0,
    "rotation":             0.60,
    "crop":                 0.0,
    "scale":                0.5,
    "shear":                0.0,
}

# AUGMENT PARAMS CONFIGS
PARAMS = {
    "blur":             3,

    "brightness_limit": 0.22,
    "contrast_limit":   0.22,

    "hue_shift_limit":  5,
    "sat_shift_limit":  10,
    "val_shift_limit":  10,

    "std_range":        (0.03, 0.08),

    "rotation":         (-90, 90),
    "crop_ratio":       (0.9, 1.0),
    "scale":            (0.85, 1.15),
    "shear":            (-0.05, 0.05)
}


class GeometryAugmentor:
    def __init__(self, config, prob, params):
        self.cfg   = config
        self.prob  = prob
        self.param = params

    # Flip
    def flip_h(self, img, polys):
        img = cv2.flip(img, 1)
        new_polys = [[[1-x, y] for x, y in poly] for poly in polys]
        return img, new_polys

    def flip_v(self, img, polys):
        img = cv2.flip(img, 0)
        new_polys = [[[x, 1-y] for x, y in poly] for poly in polys]
        return img, new_polys

    # Rotation
    def rotate(self, img, polys, angle):
        h, w = img.shape[:2]
        M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1)
        img = cv2.warpAffine(img, M, (w, h))

        new_polys = []
        for poly in polys:
            new_poly = []
            for x, y in poly:
                px = x * w
                py = y * h
                rx = M[0,0]*px + M[0,1]*py + M[0,2]
                ry = M[1,0]*px + M[1,1]*py + M[1,2]
                new_poly.append([np.clip(rx / w, 0, 1), np.clip(ry / h, 0, 1)])
            new_polys.append(new_poly)

        return img, new_polys

    # Crop 
    def crop(self, img, polys, crop_ratio):
        h, w = img.shape[:2]
        cr = random.uniform(*crop_ratio)
        cw, ch = int(w * cr), int(h * cr)
        cx = random.randint(0, w - cw)
        cy = random.randint(0, h - ch)

        cropped = img[cy:cy+ch, cx:cx+cw]
        new_polys = []

        for poly in polys:
            new_poly = []
            for x, y in poly:
                px = x * w
                py = y * h
                if cx <= px <= cx+cw and cy <= py <= cy+ch:
                    nx = (px - cx) / cw
                    ny = (py - cy) / ch
                    new_poly.append([nx, ny])
            if len(new_poly) >= 3:
                new_polys.append(new_poly)

        cropped = cv2.resize(cropped, (w, h))
        return cropped, new_polys

    # Scale 
    def scale(self, img, polys, scale_range):
        h, w = img.shape[:2]
        sx = random.uniform(*scale_range)
        sy = random.uniform(*scale_range)

        new_w = int(w * sx)
        new_h = int(h * sy)
        img_scaled = cv2.resize(img, (new_w, new_h))

        new_polys = []
        for poly in polys:
            new_poly = []
            for x, y in poly:
                nx = np.clip((x - 0.5) * sx + 0.5, 0, 1)
                ny = np.clip((y - 0.5) * sy + 0.5, 0, 1)
                new_poly.append([nx, ny])
            new_polys.append(new_poly)

        canvas = np.zeros_like(img)
        dx = (new_w - w) // 2
        dy = (new_h - h) // 2

        canvas[max(0,-dy):max(0,-dy)+min(h, new_h),
               max(0,-dx):max(0,-dx)+min(w, new_w)] = \
               img_scaled[max(0,dy):max(0,dy)+min(h, new_h),
                          max(0,dx):max(0,dx)+min(w, new_w)]

        return canvas, new_polys

    # Shear 
    def shear(self, img, polys, shear_range):
        h, w = img.shape[:2]

        shx = random.uniform(*shear_range)
        shy = random.uniform(*shear_range)

        cx, cy = w/2, h/2
        M = np.float32([
            [1, shx, -shx * cy],
            [shy, 1, -shy * cx]
        ])

        img = cv2.warpAffine(img, M, (w, h))

        new_polys = []
        for poly in polys:
            new_poly = []
            for x, y in poly:

                px = x * w
                py = y * h
                
                sx = M[0,0]*px + M[0,1]*py + M[0,2]
                sy = M[1,0]*px + M[1,1]*py + M[1,2]

                new_poly.append([np.clip(sx / w, 0, 1), np.clip(sy / h, 0, 1)])
            new_polys.append(new_poly)

        return img, new_polys

    def apply(self, img, polys):

        if self.cfg["flip_horizontal"] and random.random() < self.prob["flip_horizontal"]:
            img, polys = self.flip_h(img, polys)

        if self.cfg["flip_vertical"] and random.random() < self.prob["flip_vertical"]:
            img, polys = self.flip_v(img, polys)

        if self.cfg["rotation"] and random.random() < self.prob["rotation"]:
            angle = random.uniform(*self.param["rotation"])
            img, polys = self.rotate(img, polys, angle)

        if self.cfg["crop"] and random.random() < self.prob["crop"]:
            img, polys = self.crop(img, polys, self.param["crop_ratio"])

        if self.cfg["scale"] and random.random() < self.prob["scale"]:
            img, polys = self.scale(img, polys, self.param["scale"])

        if self.cfg["shear"] and random.random() < self.prob["shear"]:
            img, polys = self.shear(img, polys, self.param["shear"])

        return img, polys
    
class PhotometricAugmentor:
    def __init__(self, config, prob, params):
        self.prob = prob
        self.params = params
        trans = []

        if config["blur"]:
            trans.append(A.Blur(p=self.prob["blur"], blur_limit=params["blur"]))

        if config["brightness_contrast"]:
            trans.append(A.RandomBrightnessContrast(brightness_limit=self.params["brightness_limit"],
                                                    contrast_limit=self.params["contrast_limit"], 
                                                    p=self.prob["brightness_contrast"]))

        if config["gaussian_noise"]:
            trans.append(A.GaussNoise(std_range=self.params["std_range"], p=self.prob["gaussian_noise"]))

        if config["hue_saturation"]:
            trans.append(A.HueSaturationValue(hue_shift_limit=self.params["hue_shift_limit"],
                                              sat_shift_limit=self.params["sat_shift_limit"],
                                              val_shift_limit=self.params["val_shift_limit"], 
                                              p=self.prob["hue_saturation"]))

        self.pipeline = A.Compose(trans)

    def apply(self, img):
        return self.pipeline(image=img)["image"]
    
class DatasetAugmentor:
    def __init__(self, input_img, input_lbl, out_img, out_lbl, scale,
                 geo_cfg, photo_cfg, prob, params):

        self.in_img = Path(input_img)
        self.in_lbl = Path(input_lbl)
        self.out_img = Path(out_img)
        self.out_lbl = Path(out_lbl)
        self.scale = scale

        self.geo   = GeometryAugmentor(geo_cfg, prob, params)
        self.photo = PhotometricAugmentor(photo_cfg, prob, params)

        self.out_img.mkdir(parents=True, exist_ok=True)
        self.out_lbl.mkdir(parents=True, exist_ok=True)

    def read_polygons(self, path):
        polys, labels = [], []
        if not path.exists():
            return polys, labels

        with open(path) as f:
            for line in f:
                parts = line.split()
                cls = int(parts[0])
                coords = list(map(float, parts[1:]))
                pts = [[coords[i], coords[i+1]] for i in range(0, len(coords), 2)]
                if len(pts) >= 3:
                    polys.append(pts)
                    labels.append(cls)
        return polys, labels

    def write_polygons(self, path, polys, labels):
        with open(path, "w") as f:
            for cls, poly in zip(labels, polys):
                flat = []
                for x, y in poly:
                    flat.append(f"{x:.6f}")
                    flat.append(f"{y:.6f}")
                f.write(f"{cls} " + " ".join(flat) + "\n")

    def main(self):
        index = 0
        images = []
        for ext in ["*.jpg", "*.png", "*.jpeg"]:
            images.extend(self.in_img.glob(ext))
        images = sorted(images)
        print(f"Found {len(images)} images")

        for img_path in images:
            lbl_path = self.in_lbl / f"{img_path.stem}.txt"
            polys, labels = self.read_polygons(lbl_path)
            img = cv2.imread(str(img_path))

            index += 1
            
            cv2.imwrite(str(self.out_img / f"{img_path.stem}_original.jpg"), img)
            self.write_polygons(self.out_lbl / f"{img_path.stem}_original.txt", polys, labels)

            for i in range(self.scale - 1):
                aug_img = img.copy()
                aug_poly = [p.copy() for p in polys]

                # Geometric
                aug_img, aug_poly = self.geo.apply(aug_img, aug_poly)

                # Photometric
                aug_img = self.photo.apply(aug_img)

                # Save
                cv2.imwrite(str(self.out_img / f"{img_path.stem}_augment{i}.jpg"), aug_img)
                self.write_polygons(self.out_lbl / f"{img_path.stem}_augment{i}.txt", aug_poly, labels)

                print(f"AUGMENT IMAGES: {index}")

        print("DONE AUGMENT")


if __name__ == "__main__":
    aug = DatasetAugmentor(
        INPUT_IMAGES, INPUT_LABELS,
        OUTPUT_IMAGES, OUTPUT_LABELS,
        SCALE_DATASET, GEOMETRIC, PHOTOMETRIC,
        PROBABILITY, PARAMS
    )

    aug.main()

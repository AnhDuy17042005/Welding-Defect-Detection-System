from ultralytics import YOLO
from pathlib import Path
import torch

ROOT  = Path(__file__).resolve().parent.parent

MODEL     = "yolo11m-seg.pt"
DATA_YAML = ROOT / "dataset" / "augmented" / "data.yaml"
PROJECT   = ROOT / "models" / "runs"

img_size    = 960
epochs      = 80
batch_size  = 2
accumulate  = 8
lr0         = 1e-4
lrf         = 0.01
optimizer   = 'AdamW'
freeze      = 0
clss        = 0.8

device      = "cuda" if torch.cuda.is_available() else "cpu"
conf        = 0.25
patience    = 30

exist_ok    = True
seed        = 42
task        = "segment"
workers     = 2
cache       = False
fraction    = 1.0

dropout     = 0.1

mosaic      = 0.8
copy_paste  = 0.4

degrees     = 15.0
fliplr      = 0.5
flipud      = 0.3
scale       = 0.6
shear       = 2.0
perspective = 0.0003

hsv_h       = 0.01
hsv_s       = 0.6
hsv_v       = 0.5

def main() -> None:

    model = YOLO(MODEL)

    model.train(
        data=DATA_YAML,
        imgsz=img_size,
        epochs=epochs,
        batch=batch_size,
        device=device,

        project=PROJECT,
        exist_ok=exist_ok,

        task="segment",
        patience=patience,
        seed=seed,
        workers=workers,
        cache=cache,
        fraction=fraction,

        optimizer=optimizer,
        lr0=lr0,
        lrf=lrf,

        freeze=freeze,
        cls=clss,

        mosaic=mosaic,
        copy_paste=copy_paste,

        degrees=degrees,
        scale=scale,
        shear=shear,
        perspective=perspective,

        fliplr=fliplr,
        flipud=flipud,

        hsv_h=hsv_h,
        hsv_s=hsv_s,
        hsv_v=hsv_v,
    )

if __name__ == "__main__":
    main()
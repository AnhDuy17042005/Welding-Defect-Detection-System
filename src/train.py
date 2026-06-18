from ultralytics import YOLO
from pathlib import Path

BASE_DIR  = Path(__file__).resolve().parent.parent

MODEL     = "yolo11n-seg.pt"
DATA_YAML = BASE_DIR / "dataset" / "processed" / "data.yaml"
PROJECT   = BASE_DIR / "models" / "runs"

def main() -> None:

    model = YOLO(MODEL)
    model.train(
        data=DATA_YAML,
        imgsz=960,
        epochs=100,
        batch=8,
        device=0,
        patience=20,
        seed=42,
        workers=4,
        project=str(PROJECT),
        name="train_ver2"
    )

if __name__ == "__main__":
    main()
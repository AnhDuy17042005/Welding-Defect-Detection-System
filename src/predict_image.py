import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO
from get_roi import ROIExtractor

BASE_DIR = Path(__file__).resolve().parent.parent

IMG_PATH = BASE_DIR / "dataset" / "require" / "5 (2).jpg"
MODEL_PATH = BASE_DIR / "models" / "runs" / "train_ver2" / "weights" / "best.pt"

def load_image(img_path: Path) -> np.ndarray:
    img = cv2.imread(str(img_path))
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {img_path}")

    return img

def run_model(model_path: Path, img: np.ndarray):
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    if img.size == 0:
        raise ValueError("Cannot run prediction on an empty image.")

    model = YOLO(str(model_path))
    results = model.predict(
        source=img,
        imgsz=640,
        conf=0.25,
        iou=0.25,
        verbose=False,
    )

    return results[0]

def extract_predictions(result) -> list[dict]:
    predictions = []

    boxes = result.boxes
    masks = result.masks
    names = result.names

    if boxes is None or len(boxes) == 0:
        return predictions

    xyxy_list = boxes.xyxy.cpu().numpy()
    cls_list = boxes.cls.cpu().numpy().astype(int)
    conf_list = boxes.conf.cpu().numpy()

    # masks.xy là list polygon theo từng object, model là segmentation
    mask_polygons = masks.xy if masks is not None else [None] * len(xyxy_list)

    # masks.data là bitmap mask dạng tensor [N, H, W]
    mask_bitmaps = masks.data.cpu().numpy() if masks is not None else None

    for i, bbox in enumerate(xyxy_list):
        class_id = int(cls_list[i])
        class_name = names[class_id]
        confidence = float(conf_list[i])

        x1, y1, x2, y2 = bbox.tolist()

        polygon = None
        if mask_polygons[i] is not None:
            polygon = mask_polygons[i].tolist()

        mask_bitmap = None
        if mask_bitmaps is not None:
            mask_bitmap = mask_bitmaps[i]

        predictions.append(
            {
                "class_id": class_id,
                "class_name": class_name,
                "confidence": confidence,
                "bbox": [float(x1), float(y1), float(x2), float(y2)],
                "polygon": polygon,
                "mask": mask_bitmap,
            }
        )

    return predictions

def get_welding_line_mask(predictions: list[dict], img_shape: tuple) -> np.ndarray | None:
    h, w = img_shape[:2]
    combined_mask = np.zeros((h, w), dtype=np.uint8)

    for pred in predictions:
        class_name = pred["class_name"].lower().replace(" ", "_")

        if class_name != "welding_line":
            continue

        if pred["mask"] is None:
            continue

        mask = cv2.resize(
            pred["mask"],
            (w, h),
            interpolation=cv2.INTER_NEAREST,
        )

        mask = (mask > 0.5).astype(np.uint8) * 255
        combined_mask = cv2.bitwise_or(combined_mask, mask)

    if cv2.countNonZero(combined_mask) == 0:
        return None

    # White mask, black background
    return combined_mask


def main() -> None:
    img = load_image(IMG_PATH)
    print(f"Loaded image: {IMG_PATH}")
    print(f"Image size: {img.shape[1]}x{img.shape[0]}")

    # roi_extractor = ROIExtractor(
    #     window_name="Select ROI",
    #     display_size=(800, 800),
    # )

    # roi = roi_extractor.select_roi(img)

    # if roi is None:
    #     print("No ROI selected. Stop prediction.")
    #     return

    # roi_img = roi_extractor.crop(img, roi)
    # print(f"ROI: {roi}")
    # print(f"ROI size: {roi_img.shape[1]}x{roi_img.shape[0]}")
    # print(f"Loading model: {MODEL_PATH}")

    result = run_model(
        model_path=MODEL_PATH,
        img=img,
    )

    predictions = extract_predictions(result)
    print(f"Detected objects: {len(predictions)}")

    for pred in predictions:
        print(
            f"- {pred['class_name']}: "
            f"{pred['confidence']:.3f}, bbox={pred['bbox']}"
        )

    plot = result.plot(boxes=True, labels=True, conf=True)

    welding_mask = get_welding_line_mask(
        predictions=predictions,
        img_shape=img.shape,
    )

    if welding_mask is None:
        print("No welding_line mask found.")
    else:
        print("Found welding_line mask.")

    h, w = plot.shape[:2]

    display = cv2.resize(
        plot,
        (int(800), int(800)),
        interpolation=cv2.INTER_AREA,
    )

    cv2.imshow("ROI Prediction", display)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

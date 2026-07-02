from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from backend.model_service import BOXES, CONF, LABELS, MASKS, YoloSegmentationService


BASE_DIR = Path(__file__).resolve().parent.parent
FRONTEND_DIR = BASE_DIR / "frontend"
YOLO_MODELS = {
    f"yolo_ver{version}": (
        BASE_DIR / "models" / "runs" / f"train_ver{version}" / "weights" / "best.pt"
    )
    for version in range(1, 6)
}
UNET_MODELS = {
    f"unet_ver{version}": (
        BASE_DIR / "models" / "unet" / f"train_ver{version}" / "best.pth"
    )
    for version in range(1, 4)
}
YOLO_MODEL_LABELS = {
    model_id: f"YOLOv11 Ver {version}"
    for version, model_id in enumerate(YOLO_MODELS, start=1)
}
UNET_MODEL_LABELS = {
    model_id: f"U-Net Ver {version}"
    for version, model_id in enumerate(UNET_MODELS, start=1)
}


def get_max_upload_bytes() -> int:
    max_mb = float(os.getenv("MAX_UPLOAD_MB", "12"))
    return int(max_mb * 1024 * 1024)


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.yolo_service = YoloSegmentationService(
        yolo_models=YOLO_MODELS,
        unet_models=UNET_MODELS,
        default_yolo_model=os.getenv("YOLO_MODEL_ID", "yolo_ver5"),
        default_unet_model=os.getenv("UNET_MODEL_ID", "unet_ver3"),
        device=os.getenv("YOLO_DEVICE") or None,
        unet_device=os.getenv("UNET_DEVICE", "auto"),
        unet_threshold=float(os.getenv("UNET_THRESHOLD", "0.25")),
        roi_margin=float(os.getenv("ROI_MARGIN", "0.50")),
    )

    app.state.max_upload_bytes = get_max_upload_bytes()
    yield


app = FastAPI(
    title="Welding Defect Segmentation API",
    version="1.0.0",
    lifespan=lifespan,
)

app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")


def get_service(request: Request) -> YoloSegmentationService:
    service = getattr(request.app.state, "yolo_service", None)
    if service is None:
        raise HTTPException(status_code=503, detail="YOLO model is not loaded.")
    return service


@app.get("/")
def index() -> FileResponse:
    return FileResponse(FRONTEND_DIR / "index.html")


@app.get("/api/health")
def health(request: Request) -> dict[str, Any]:
    service = get_service(request)
    return {
        "model_name": "YOLOv11 + U-Net",
        "status": "ok",
        "model_path": str(service.model_path),
        "classes": service.names,
        "unet_threshold": service.unet_threshold,
        "roi_margin": service.roi_margin,
        "unet_imgsz": service.unet_img_size,
        "models": {
            "yolo": [
                {"id": model_id, "label": YOLO_MODEL_LABELS[model_id]}
                for model_id in YOLO_MODELS
            ],
            "unet": [
                {"id": model_id, "label": UNET_MODEL_LABELS[model_id]}
                for model_id in UNET_MODELS
            ],
            "selected_yolo": service.active_yolo_model,
            "selected_unet": service.active_unet_model,
        },
    }


@app.get("/api/classes")
def classes(request: Request) -> dict[str, Any]:
    service = get_service(request)
    return {"classes": service.names}


@app.post("/api/predict")
async def predict(
    request: Request,
    file: UploadFile = File(...),
    conf: float = Form(0.25),
    iou: float = Form(0.25),
    imgsz: int = Form(960),
    yolo_model: str = Form("yolo_ver5"),
    unet_model: str = Form("unet_ver3"),
    unet_imgsz: int = Form(512),
    unet_threshold: float = Form(0.25),
    roi_margin: float = Form(0.40),
    show_masks: bool = Form(MASKS),
    show_boxes: bool = Form(BOXES),
    show_labels: bool = Form(LABELS),
    show_conf: bool = Form(CONF),
) -> dict[str, Any]:
    if file.content_type and not file.content_type.startswith("image/"):
        raise HTTPException(status_code=415, detail="Only image uploads are supported.")

    max_upload_bytes = getattr(request.app.state, "max_upload_bytes", get_max_upload_bytes())
    image_bytes = await file.read(max_upload_bytes + 1)
    if len(image_bytes) > max_upload_bytes:
        raise HTTPException(status_code=413, detail="Uploaded image is too large.")

    try:
        service = get_service(request)
        result = service.predict(
            image_bytes=image_bytes,
            conf=conf,
            iou=iou,
            imgsz=imgsz,
            yolo_model=yolo_model,
            unet_model=unet_model,
            unet_imgsz=unet_imgsz,
            unet_threshold=unet_threshold,
            roi_margin=roi_margin,
            show_masks=show_masks,
            show_boxes=show_boxes,
            show_labels=show_labels,
            show_conf=show_conf,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}") from exc

    return result

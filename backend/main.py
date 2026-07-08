"""
    FastAPI backend for Welding Defect Segmentation web app.

    Purpose:
        1. Load YOLOv11 and U-Net models when the server starts.
        2. Serve frontend static files.
        3. Provide API endpoints for health check, class names, and prediction.
        4. Receive uploaded image from web app.
        5. Run YOLO + U-Net hybrid inference.
        6. Return prediction result and encoded output images to frontend.
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from typing import Any

"""Set matplotlib cache directory for Linux/server environment"""
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from backend.model_service import YoloSegmentationService
from configs.hybrid import ROI_MARGIN
from configs.path import FRONTEND_DIR
from configs.unet import (
    UNET_DEVICE,
    UNET_IMAGE_SIZE,
    UNET_MODEL_ID,
    UNET_MODELS,
    UNET_MODEL_LABELS,
    UNET_THRESHOLD,
)
from configs.visualize import (
    SHOW_BOXES,
    SHOW_CONFIDENCE,
    SHOW_LABELS,
    SHOW_MASKS,
)
from configs.yolo import (
    YOLO_CONFIDENCE_THRESHOLD,
    YOLO_DEVICE,
    YOLO_IMAGE_SIZE,
    YOLO_IOU_THRESHOLD,
    YOLO_MODEL_ID,
    YOLO_MODELS,
    YOLO_MODEL_LABELS,
)


def get_max_upload_bytes() -> int:
    """
        Get maximum upload file size from environment variable.

        Environment:
            MAX_UPLOAD_MB:
                Maximum uploaded image size in megabytes.

        Default:
            12 MB.
    """

    """Read max upload size in MB"""
    max_mb = float(os.getenv("MAX_UPLOAD_MB", "12"))

    """Convert MB to bytes"""
    return int(max_mb * 1024 * 1024)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
        FastAPI lifespan event.

        This function runs when the server starts and stops.

        Startup:
            - Load YOLO + U-Net service
            - Store service in app state
            - Store maximum upload size

        Shutdown:
            - FastAPI continues after yield
    """

    """Initialize YOLO + U-Net inference service"""
    app.state.yolo_service = YoloSegmentationService(
        yolo_models=YOLO_MODELS,
        unet_models=UNET_MODELS,
        default_yolo_model=os.getenv("YOLO_MODEL_ID", YOLO_MODEL_ID),
        default_unet_model=os.getenv("UNET_MODEL_ID", UNET_MODEL_ID),
        device=os.getenv("YOLO_DEVICE") or YOLO_DEVICE,
        unet_device=os.getenv("UNET_DEVICE", UNET_DEVICE),
        unet_threshold=float(os.getenv("UNET_THRESHOLD", str(UNET_THRESHOLD))),
        roi_margin=float(os.getenv("ROI_MARGIN", str(ROI_MARGIN))),
    )

    """Store upload limit in app state"""
    app.state.max_upload_bytes = get_max_upload_bytes()

    """Keep application running"""
    yield


"""Create FastAPI application"""
app = FastAPI(
    title="Welding Defect Segmentation API",
    version="1.0.0",
    lifespan=lifespan,
)

"""Mount frontend static directory"""
app.mount(
    "/static",
    StaticFiles(directory=FRONTEND_DIR),
    name="static"
)


def get_service(request: Request) -> YoloSegmentationService:
    """
        Get loaded YOLO + U-Net service from FastAPI app state.
    """

    """Read service from app state"""
    service = getattr(request.app.state, "yolo_service", None)

    """Return 503 if model service has not been loaded"""
    if service is None:
        raise HTTPException(
            status_code=503,
            detail="YOLO model is not loaded."
        )

    return service


@app.get("/")
def index() -> FileResponse:
    """
        Serve frontend homepage.
    """

    """Return index.html file"""
    return FileResponse(FRONTEND_DIR / "index.html")


@app.get("/api/health")
def health(request: Request) -> dict[str, Any]:
    """
        Health check endpoint.

        Returns:
            - API status
            - active model path
            - class names
            - U-Net settings
            - available YOLO and U-Net models
            - currently selected models
    """

    """Get loaded inference service"""
    service = get_service(request)

    """Return API and model status"""
    return {
        "model_name": "YOLOv11 + U-Net",
        "status": "ok",
        "model_path": str(service.model_path),
        "classes": service.names,
        "unet_threshold": service.unet_threshold,
        "roi_margin": service.roi_margin,
        "unet_imgsz": service.unet_img_size,
        "yolo_confidence": YOLO_CONFIDENCE_THRESHOLD,
        "yolo_iou": YOLO_IOU_THRESHOLD,
        "yolo_imgsz": YOLO_IMAGE_SIZE,
        "display": {
            "masks": SHOW_MASKS,
            "boxes": SHOW_BOXES,
            "labels": SHOW_LABELS,
            "confidence": SHOW_CONFIDENCE,
        },
        "models": {
            "yolo": [
                {
                    "id": model_id,
                    "label": YOLO_MODEL_LABELS[model_id],
                }
                for model_id in YOLO_MODELS
            ],
            "unet": [
                {
                    "id": model_id,
                    "label": UNET_MODEL_LABELS[model_id],
                }
                for model_id in UNET_MODELS
            ],
            "selected_yolo": service.active_yolo_model,
            "selected_unet": service.active_unet_model,
        },
    }


@app.get("/api/classes")
def classes(request: Request) -> dict[str, Any]:
    """
        Return YOLO class names.
    """

    """Get loaded inference service"""
    service = get_service(request)

    """Return class id to class name mapping"""
    return {"classes": service.names}


@app.post("/api/predict")
async def predict(
    request: Request,
    file: UploadFile = File(...),
    conf: float = Form(YOLO_CONFIDENCE_THRESHOLD),
    iou: float = Form(YOLO_IOU_THRESHOLD),
    imgsz: int = Form(YOLO_IMAGE_SIZE),
    yolo_model: str = Form(YOLO_MODEL_ID),
    unet_model: str = Form(UNET_MODEL_ID),
    unet_imgsz: int = Form(UNET_IMAGE_SIZE),
    unet_threshold: float = Form(UNET_THRESHOLD),
    roi_margin: float = Form(ROI_MARGIN),
    show_masks: bool = Form(SHOW_MASKS),
    show_boxes: bool = Form(SHOW_BOXES),
    show_labels: bool = Form(SHOW_LABELS),
    show_conf: bool = Form(SHOW_CONFIDENCE),
) -> dict[str, Any]:
    """
        Run prediction on uploaded image.

        Input:
            - image file
            - YOLO confidence threshold
            - YOLO IoU threshold
            - YOLO image size
            - selected YOLO model
            - selected U-Net model
            - U-Net image size
            - U-Net threshold
            - ROI margin
            - visualization options

        Returns:
            JSON result from YoloSegmentationService.
    """

    """Validate uploaded file type"""
    if file.content_type and not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=415,
            detail="Only image uploads are supported."
        )

    """Get maximum upload size"""
    max_upload_bytes = getattr(
        request.app.state,
        "max_upload_bytes",
        get_max_upload_bytes()
    )

    """Read uploaded image bytes with size limit"""
    image_bytes = await file.read(max_upload_bytes + 1)

    """Reject image if it exceeds upload limit"""
    if len(image_bytes) > max_upload_bytes:
        raise HTTPException(
            status_code=413,
            detail="Uploaded image is too large."
        )

    try:
        """Get loaded inference service"""
        service = get_service(request)

        """Run YOLO + U-Net prediction"""
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
        """Return bad request for invalid input parameters"""
        raise HTTPException(
            status_code=400,
            detail=str(exc)
        ) from exc

    except FileNotFoundError as exc:
        """Return server error if model file is missing"""
        raise HTTPException(
            status_code=500,
            detail=str(exc)
        ) from exc

    except Exception as exc:
        """Return server error for unexpected prediction failure"""
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {exc}"
        ) from exc

    """Return prediction result to frontend"""
    return result

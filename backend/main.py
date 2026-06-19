from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from backend.model_service import YoloSegmentationService


BASE_DIR = Path(__file__).resolve().parent.parent
FRONTEND_DIR = BASE_DIR / "frontend"
DEFAULT_MODEL_PATH = BASE_DIR / "models" / "runs" / "train_ver2" / "weights" / "best.pt"


def resolve_path(value: str | None, default: Path) -> Path:
    if not value:
        return default

    path = Path(value).expanduser()
    if path.is_absolute():
        return path

    return (BASE_DIR / path).resolve()


def get_max_upload_bytes() -> int:
    max_mb = float(os.getenv("MAX_UPLOAD_MB", "12"))
    return int(max_mb * 1024 * 1024)


@asynccontextmanager
async def lifespan(app: FastAPI):
    model_path = resolve_path(os.getenv("MODEL_PATH"), DEFAULT_MODEL_PATH)
    device = os.getenv("YOLO_DEVICE") or None
    app.state.yolo_service = YoloSegmentationService(model_path=model_path, device=device)
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
        "status": "ok",
        "model_path": str(service.model_path),
        "classes": service.names,
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
) -> dict[str, Any]:
    if file.content_type and not file.content_type.startswith("image/"):
        raise HTTPException(status_code=415, detail="Only image uploads are supported.")

    max_upload_bytes = getattr(request.app.state, "max_upload_bytes", get_max_upload_bytes())
    image_bytes = await file.read(max_upload_bytes + 1)
    if len(image_bytes) > max_upload_bytes:
        raise HTTPException(status_code=413, detail="Uploaded image is too large.")

    try:
        service = get_service(request)
        result = service.predict(image_bytes=image_bytes, conf=conf, iou=iou, imgsz=imgsz)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}") from exc

    return result

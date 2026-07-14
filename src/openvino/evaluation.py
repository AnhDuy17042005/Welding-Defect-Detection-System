"""
    Benchmark PyTorch and OpenVINO inference latency.

    Metric:
        Average latency in milliseconds per image.

    Run:
        python -m src.openvino.evaluation --target both
        python -m src.openvino.evaluation --target yolo --max-images 10
        python -m src.openvino.evaluation --target unet --repeat 5
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path
from statistics import mean
from typing import Callable

import cv2


PROJECT_ROOT = Path(__file__).resolve().parents[2]

"""Support direct script run from the project root."""
if __package__ in (None, ""):
    sys.path.insert(0, str(PROJECT_ROOT))

"""Matplotlib cache directory for Linux/server environments."""
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

from ultralytics import YOLO

from configs.data import IMAGE_EXTENSIONS, REQUIRE_DATASET
from configs.path import METRICS_DIR
from configs.unet import (
    RIPPLE_ROI_DATASET,
    UNET_IMAGE_SIZE,
    UNET_MODEL,
    UNET_MODEL_VERSION,
    UNET_RUNS_DIR,
)
from configs.yolo import (
    YOLO_IMAGE_SIZE,
    YOLO_MODEL,
    YOLO_MODEL_VERSION,
    YOLO_RUNS_DIR,
    YOLO_TASK,
)
from src.unet.inference import get_device, load_model, predict as predict_unet


DEFAULT_OUTPUT_DIR = METRICS_DIR / "openvino"


def default_yolo_pytorch_model() -> Path:
    """
        Build default YOLO PyTorch checkpoint path.
    """

    return (
        YOLO_RUNS_DIR
        / f"train_ver{YOLO_MODEL_VERSION}"
        / "weights"
        / "best.pt"
    )


def default_unet_pytorch_model() -> Path:
    """
        Build default U-Net PyTorch checkpoint path.
    """

    return UNET_RUNS_DIR / f"train_ver{UNET_MODEL_VERSION}" / "best.pth"


def parse_args() -> argparse.Namespace:
    """
        Parse command line arguments.
    """

    parser = argparse.ArgumentParser(
        description="Benchmark PyTorch vs OpenVINO average inference latency."
    )

    parser.add_argument(
        "--target",
        choices=("yolo", "unet", "both"),
        default="both",
        help="Which model family to benchmark.",
    )
    parser.add_argument(
        "--image-dir",
        type=Path,
        default=None,
        help=(
            "Optional image directory. If omitted, YOLO uses dataset/require "
            "and U-Net uses dataset/ripple_roi/test/images."
        ),
    )
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeat", type=int, default=3)

    parser.add_argument("--yolo-imgsz", type=int, default=YOLO_IMAGE_SIZE)
    parser.add_argument("--unet-img-size", type=int, default=UNET_IMAGE_SIZE)
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device for PyTorch baseline. CPU is recommended when comparing with OpenVINO.",
    )

    parser.add_argument(
        "--pytorch-yolo-model",
        type=Path,
        default=default_yolo_pytorch_model(),
    )
    parser.add_argument(
        "--openvino-yolo-model",
        type=Path,
        default=YOLO_MODEL,
    )
    parser.add_argument(
        "--pytorch-unet-model",
        type=Path,
        default=default_unet_pytorch_model(),
    )
    parser.add_argument(
        "--openvino-unet-model",
        type=Path,
        default=UNET_MODEL,
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)

    args = parser.parse_args()

    if args.max_images is not None and args.max_images < 1:
        parser.error("--max-images must be at least 1")

    if args.warmup < 0:
        parser.error("--warmup cannot be negative")

    if args.repeat < 1:
        parser.error("--repeat must be at least 1")

    if args.yolo_imgsz < 32:
        parser.error("--yolo-imgsz must be at least 32")

    if args.unet_img_size < 32:
        parser.error("--unet-img-size must be at least 32")

    return args


def check_model_path(path: Path, label: str) -> None:
    """
        Validate model checkpoint or model directory.
    """

    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")


def find_images(image_dir: Path, max_images: int | None) -> list[Path]:
    """
        Find benchmark images in a directory.
    """

    if not image_dir.is_dir():
        raise FileNotFoundError(f"Image directory not found: {image_dir}")

    images = sorted(
        path
        for path in image_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )

    if max_images is not None:
        images = images[:max_images]

    if not images:
        raise ValueError(f"No benchmark images found in: {image_dir}")

    return images


def load_images(image_paths: list[Path]) -> list[tuple[Path, object]]:
    """
        Load images once so benchmark time does not include disk I/O.
    """

    loaded = []

    for image_path in image_paths:
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)

        if image is None:
            raise FileNotFoundError(f"Cannot read image: {image_path}")

        loaded.append((image_path, image))

    return loaded


def benchmark_runner(
    target: str,
    backend: str,
    runner: Callable[[object], None],
    images: list[tuple[Path, object]],
    warmup: int,
    repeat: int,
) -> tuple[dict[str, float | int | str], list[dict[str, float | int | str]]]:
    """
        Benchmark one model backend.
    """

    for index in range(warmup):
        _, image = images[index % len(images)]
        runner(image)

    rows: list[dict[str, float | int | str]] = []
    latencies: list[float] = []

    for repeat_index in range(repeat):
        for image_path, image in images:
            started = time.perf_counter()
            runner(image)
            latency_ms = (time.perf_counter() - started) * 1000.0

            latencies.append(latency_ms)
            rows.append(
                {
                    "target": target,
                    "backend": backend,
                    "repeat": repeat_index + 1,
                    "image": str(image_path),
                    "latency_ms": round(latency_ms, 4),
                }
            )

    summary = {
        "target": target,
        "backend": backend,
        "image_count": len(images),
        "repeat": repeat,
        "sample_count": len(latencies),
        "average_latency_ms_per_image": round(mean(latencies), 4),
    }

    return summary, rows


def benchmark_yolo(
    args: argparse.Namespace,
) -> tuple[dict[str, object], list[dict[str, float | int | str]]]:
    """
        Benchmark YOLO PyTorch checkpoint against YOLO OpenVINO model.
    """

    check_model_path(args.pytorch_yolo_model, "YOLO PyTorch model")
    check_model_path(args.openvino_yolo_model, "YOLO OpenVINO model")

    image_dir = args.image_dir or REQUIRE_DATASET
    image_paths = find_images(image_dir, args.max_images)
    images = load_images(image_paths)

    pytorch_model = YOLO(str(args.pytorch_yolo_model), task=YOLO_TASK)
    openvino_model = YOLO(str(args.openvino_yolo_model), task=YOLO_TASK)

    def run_pytorch(image: object) -> None:
        pytorch_model.predict(
            source=image,
            imgsz=args.yolo_imgsz,
            device=args.device,
            verbose=False,
        )

    def run_openvino(image: object) -> None:
        openvino_model.predict(
            source=image,
            imgsz=args.yolo_imgsz,
            verbose=False,
        )

    pytorch_summary, pytorch_rows = benchmark_runner(
        target="yolo",
        backend="pytorch",
        runner=run_pytorch,
        images=images,
        warmup=args.warmup,
        repeat=args.repeat,
    )
    openvino_summary, openvino_rows = benchmark_runner(
        target="yolo",
        backend="openvino",
        runner=run_openvino,
        images=images,
        warmup=args.warmup,
        repeat=args.repeat,
    )

    speedup = (
        pytorch_summary["average_latency_ms_per_image"]
        / openvino_summary["average_latency_ms_per_image"]
    )

    return (
        {
            "metric": "average_latency_ms_per_image",
            "image_dir": str(image_dir),
            "pytorch": pytorch_summary,
            "openvino": openvino_summary,
            "speedup_ratio": round(float(speedup), 4),
        },
        pytorch_rows + openvino_rows,
    )


def benchmark_unet(
    args: argparse.Namespace,
) -> tuple[dict[str, object], list[dict[str, float | int | str]]]:
    """
        Benchmark U-Net PyTorch checkpoint against U-Net OpenVINO IR.
    """

    check_model_path(args.pytorch_unet_model, "U-Net PyTorch model")
    check_model_path(args.openvino_unet_model, "U-Net OpenVINO model")

    image_dir = args.image_dir or (RIPPLE_ROI_DATASET / "test" / "images")
    image_paths = find_images(image_dir, args.max_images)
    images = load_images(image_paths)

    device = get_device(args.device)
    pytorch_model, pytorch_img_size = load_model(
        args.pytorch_unet_model,
        device,
        args.unet_img_size,
    )
    openvino_model, openvino_img_size = load_model(
        args.openvino_unet_model,
        device,
        args.unet_img_size,
    )

    def run_pytorch(image: object) -> None:
        predict_unet(
            model=pytorch_model,
            image=image,
            img_size=pytorch_img_size,
            threshold=0.5,
            device=device,
        )

    def run_openvino(image: object) -> None:
        predict_unet(
            model=openvino_model,
            image=image,
            img_size=openvino_img_size,
            threshold=0.5,
            device=device,
        )

    pytorch_summary, pytorch_rows = benchmark_runner(
        target="unet",
        backend="pytorch",
        runner=run_pytorch,
        images=images,
        warmup=args.warmup,
        repeat=args.repeat,
    )
    openvino_summary, openvino_rows = benchmark_runner(
        target="unet",
        backend="openvino",
        runner=run_openvino,
        images=images,
        warmup=args.warmup,
        repeat=args.repeat,
    )

    speedup = (
        pytorch_summary["average_latency_ms_per_image"]
        / openvino_summary["average_latency_ms_per_image"]
    )

    return (
        {
            "metric": "average_latency_ms_per_image",
            "image_dir": str(image_dir),
            "pytorch": pytorch_summary,
            "openvino": openvino_summary,
            "speedup_ratio": round(float(speedup), 4),
        },
        pytorch_rows + openvino_rows,
    )


def write_csv(path: Path, rows: list[dict[str, float | int | str]]) -> None:
    """
        Write per-image benchmark rows.
    """

    if not rows:
        return

    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    """
        Run selected benchmark target and save results.
    """

    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    summary: dict[str, object] = {}
    rows: list[dict[str, float | int | str]] = []

    if args.target in ("yolo", "both"):
        yolo_summary, yolo_rows = benchmark_yolo(args)
        summary["yolo"] = yolo_summary
        rows.extend(yolo_rows)

    if args.target in ("unet", "both"):
        unet_summary, unet_rows = benchmark_unet(args)
        summary["unet"] = unet_summary
        rows.extend(unet_rows)

    summary_path = args.output_dir / "latency_summary.json"
    rows_path = args.output_dir / "latency_rows.csv"

    with summary_path.open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)

    write_csv(rows_path, rows)

    print(f"Latency summary saved: {summary_path}")
    print(f"Per-image latency rows saved: {rows_path}")

    for target, target_summary in summary.items():
        speedup = target_summary["speedup_ratio"]
        pytorch_latency = target_summary["pytorch"]["average_latency_ms_per_image"]
        openvino_latency = target_summary["openvino"]["average_latency_ms_per_image"]

        print(
            f"{target}: PyTorch={pytorch_latency:.2f} ms/image | "
            f"OpenVINO={openvino_latency:.2f} ms/image | "
            f"Speedup={speedup:.2f}x"
        )


if __name__ == "__main__":
    main()

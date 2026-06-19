# Welding Defect Segmentation System

A web-based AI Vision project for welding defect inspection using YOLO segmentation.
The system aims to classify welding surface defects and visualize defect regions with segmentation masks.

## Project Overview

This project focuses on building an end-to-end welding defect inspection system. Users can upload a welding image, and the system will return:

* Predicted defect class
* Segmentation mask overlay
* Confidence score
* Processed output image
* Basic inspection result

## Current Supported Classes

Due to limited public annotated datasets, the current implementation focuses on 4 available classes:

| ID | Class        |
| -: | ------------ |
|  0 | Crack        |
|  1 | Porosity     |
|  2 | Spatter      |
|  3 | Welding line |

The system is designed to be expandable to the full 13 customer-required welding defect classes when reliable annotated data becomes available.

## Tech Stack

* Python
* YOLO11-seg
* OpenCV
* FastAPI
* Docker

## Web Inference App

Run the local API and upload UI:

```bash
.venv/bin/pip install -r requirements.txt
.venv/bin/uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload
```

Open `http://127.0.0.1:8000`.

Docker:

```bash
docker compose up --build
```

The default model path is `models/runs/train_ver2/weights/best.pt`. Override it with `MODEL_PATH` when deploying another YOLO segmentation model.

## Status

This project is under active development.
First commit includes the initial repository structure and project documentation.

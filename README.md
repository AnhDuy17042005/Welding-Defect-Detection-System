# Welding Defect Detection System

Hệ thống thị giác máy tính hỗ trợ phát hiện và phân đoạn khuyết tật mối hàn bằng pipeline kết hợp **YOLOv11-seg** và **U-Net**. Project cung cấp cả script suy luận độc lập và web app FastAPI để tải ảnh, điều chỉnh tham số và xem kết quả trực quan.

> Đây là project nghiên cứu và thử nghiệm. Kết quả chưa được chứng nhận để thay thế quy trình kiểm tra chất lượng công nghiệp.

## Ý tưởng chính

YOLOv11 và U-Net đảm nhiệm hai vai trò bổ sung cho nhau:

- **YOLOv11-seg** phát hiện các loại khuyết tật, đồng thời xác định bbox và instance mask của `welding_line`.
- **U-Net** thực hiện binary segmentation để tìm vùng ripple bên trong ROI đường hàn.
- Kết quả U-Net được hậu xử lý, đưa về kích thước ảnh gốc và giới hạn bởi instance mask `welding_line` của YOLO.
- Ảnh cuối kết hợp bbox lỗi của YOLO với ripple mask của U-Net.

Ripple là tín hiệu mô tả cấu trúc bề mặt đường hàn, không thay thế các class khuyết tật do YOLO dự đoán.

## Kiến trúc U-Net

U-Net sử dụng encoder-decoder và skip connection để giữ lại thông tin không gian khi khôi phục segmentation mask. Model nhận ảnh RGB và trả về logits một kênh cho bài toán `ripple/background`.

<p align="center">
  <img src="assets/architecture.png" alt="U-Net architecture" width="780">
</p>

## Pipeline YOLOv11 + U-Net

<p align="center">
  <img src="assets/pipeline.png" alt="YOLOv11 and U-Net hybrid pipeline" width="900">
</p>

Luồng inference hiện tại:

1. Ảnh gốc được đưa vào YOLOv11-seg.
2. Các lỗi được giữ lại để vẽ bbox hoặc mask theo cấu hình hiển thị.
3. Mỗi bbox `welding_line` được mở rộng bằng ROI margin và crop khỏi ảnh gốc.
4. ROI được resize theo U-Net Image Size rồi đưa vào U-Net.
5. Binary mask được làm sạch bằng morphology và lọc connected components nhỏ.
6. Mask U-Net được giao với YOLO `welding_line` mask để loại vùng nằm ngoài đường hàn.
7. Mask được ghép về ảnh gốc và overlay cùng kết quả YOLO.

Nếu YOLO không phát hiện `welding_line`, U-Net không được chạy trên ảnh đó.

## Web app

Web app hỗ trợ:

- Upload hoặc kéo thả ảnh mối hàn.
- Chọn checkpoint YOLO và U-Net đã đăng ký.
- Chọn image size riêng cho từng model.
- Điều chỉnh confidence, IoU, U-Net threshold và ROI margin.
- Bật/tắt YOLO masks, boxes, labels và confidence khi render.
- Xem Overlay, Mask YOLO và Mask U-Net.
- Zoom, pan và lưu ảnh kết quả.

Backend cung cấp các endpoint chính:

| Endpoint | Mô tả |
|---|---|
| `GET /api/health` | Trạng thái service và danh sách model |
| `GET /api/classes` | Danh sách class của YOLO đang hoạt động |
| `POST /api/predict` | Nhận ảnh và trả kết quả hybrid |

## Cấu trúc project

```text
backend/
  main.py                 FastAPI routes và model registry
  model_service.py        Load model và phục vụ hybrid inference
frontend/
  index.html              Giao diện web
  app.js                  Upload, config, render và lưu kết quả
  styles.css              Bố cục và responsive styling
src/
  hybrid_inference.py     Hybrid inference bằng command line
  yolo/                   Train và inference YOLO
  unet/                   Dataset, augmentation, train, inference và hậu xử lý U-Net
assets/                   Hình kiến trúc và pipeline
dataset/                  Dữ liệu YOLO và ripple segmentation
models/                   Checkpoint YOLO và U-Net
output/                   Kết quả inference
```

Checkpoint được quản lý trong `models/` và đăng ký tại `backend/main.py`. Docker Compose mount thư mục này vào container ở chế độ read-only, vì vậy model không được đóng gói trực tiếp vào Docker image.

## Cài đặt

Project sử dụng Python 3.12.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Nếu sử dụng GPU NVIDIA, có thể cài bản PyTorch phù hợp với CUDA của máy trước khi cài các dependency còn lại.

## Chạy web app

### Chạy trực tiếp

```bash
source .venv/bin/activate
uvicorn backend.main:app --host 127.0.0.1 --port 8000
```

Mở `http://127.0.0.1:8000` trong trình duyệt.

### Chạy bằng Docker

```bash
docker compose up --build
```

Các model được mount từ `./models` theo cấu hình trong `docker-compose.yml`.

## Chạy hybrid inference bằng CLI

```bash
source .venv/bin/activate
python src/hybrid_inference.py \
  --image dataset/require/example.jpg \
  --yolo-model models/runs/train/weights/best.pt \
  --unet-model models/unet/train/best.pth \
  --output-dir output/hybrid
```

Các tham số thường dùng:

| Tham số | Ý nghĩa |
|---|---|
| `--yolo-imgsz` | Kích thước đầu vào YOLO |
| `--unet-img-size` | Kích thước ROI đầu vào U-Net |
| `--conf` | Confidence threshold của YOLO |
| `--iou` | IoU threshold cho NMS |
| `--threshold` | Ngưỡng nhị phân hóa U-Net |
| `--roi-margin` | Tỷ lệ context thêm vào mỗi phía bbox |

Ví dụ `roi-margin=0.5` làm chiều rộng và chiều cao ROI tăng tối đa thành hai lần bbox ban đầu trước khi bị giới hạn bởi biên ảnh.

## Huấn luyện

Các entrypoint chính:

```bash
# U-Net
python -m src.unet.train --data dataset/ripple_roi

# YOLOv11
python src/yolo/train.py
```

U-Net checkpoint lưu cả model state và cấu hình train như image size, giúp inference tự đọc lại kích thước mặc định. Tham số `--pretrained` dùng để fine-tune trọng số model, còn `--resume` tiếp tục đầy đủ model, optimizer và scheduler từ checkpoint.

## Dữ liệu

Hai nhóm dữ liệu chính:

- `dataset/augmented`: ảnh và polygon label cho YOLO segmentation.
- `dataset/ripple_split`: ảnh và binary mask ripple theo các split train/valid/test.
- `dataset/ripple_roi`: dữ liệu ripple được crop theo vùng ROI để huấn luyện U-Net.

Ảnh và mask U-Net phải cùng stem, ví dụ:

```text
train/images/sample.jpg
train/masks/sample.png
```

Mask sử dụng `0` cho background và giá trị lớn hơn `0` cho ripple.

## Cấu hình triển khai

Các biến môi trường thường dùng:

| Biến | Mô tả |
|---|---|
| `YOLO_MODEL_ID` | Model YOLO mặc định trong registry |
| `UNET_MODEL_ID` | Model U-Net mặc định trong registry |
| `YOLO_DEVICE` | Device chạy YOLO |
| `UNET_DEVICE` | Device chạy U-Net, mặc định `auto` |
| `UNET_THRESHOLD` | U-Net threshold mặc định |
| `ROI_MARGIN` | ROI margin mặc định |
| `MAX_UPLOAD_MB` | Kích thước file upload tối đa |

Model được load một lần khi service khởi động. Khi người dùng chọn checkpoint khác, service thay model đang hoạt động trước lần dự đoán tiếp theo.

## Hạn chế hiện tại

- Chất lượng phụ thuộc mạnh vào khả năng YOLO tìm đúng `welding_line`.
- Dataset còn nhỏ và mất cân bằng giữa các class lỗi.
- U-Net được huấn luyện với binary ripple mask và chưa mô hình hóa trường hợp negative ROI đầy đủ.
- ROI margin, image size và threshold cần được chọn bằng validation metrics thay vì chỉ quan sát overlay.
- Pipeline phục vụ nghiên cứu; cần đánh giá thêm trước khi sử dụng trong sản xuất.

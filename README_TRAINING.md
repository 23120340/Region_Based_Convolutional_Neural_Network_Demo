# Hướng dẫn chi tiết Train/Test YOLOv8 cho Earbud Detection

## 1. Mục tiêu dự án

Dự án này dùng YOLOv8 để phát hiện:
- vỏ hộp sạc earbud
- tay người
- tai nghe trái/phải
- khe trống trong hộp

Mục tiêu chính là nhận diện đúng vị trí và trạng thái của từng đối tượng trong video hoặc ảnh.

## 2. Cấu trúc thư mục dataset

```text
C:\RNN\
├── data.yaml
├── train/
│   ├── images/
│   └── labels/
├── valid/
│   ├── images/
│   └── labels/
├── test/
│   ├── images/
│   └── labels/
├── train_local.py
├── quick_test.py
├── test_model.py
├── run_inference_improved.py
├── run_inference_video.py
├── merge_slots.py
├── cleanup.py
├── quick_start.py
├── README.md
├── README_TRAINING.md
└── runs/
```

Các class chuẩn của dữ liệu:
- Case
- Hand
- Left_Earbud
- Left_Slot
- Right_Earbud
- Right_Slot

## 3. Cài đặt môi trường

Từ thư mục dự án:

```bash
pip install ultralytics opencv-python pandas numpy
```

Nếu dùng GPU NVIDIA, hãy đảm bảo driver và CUDA đã được cài đặt đúng cách.

## 4. Huấn luyện model

Chạy:

```bash
python train_local.py
```

Một số tham số được định nghĩa trong file `train_local.py`, ví dụ:
- `epochs=100`
- `imgsz=640`
- `batch=16`
- `model='yolov8n.pt'`

Bạn có thể đổi sang model lớn hơn:
```python
model = YOLO('yolov8s.pt')
# hoặc yolov8m.pt, yolov8l.pt
```

Kết quả training được lưu trong:
```text
runs/train/
└── ...
```

Các file quan trọng:
- `weights/best.pt`
- `weights/last.pt`
- `results.png`
- `confusion_matrix.png`

## 5. Test nhanh

```bash
python quick_test.py
```

Hoặc chỉ định model cụ thể:

```bash
python quick_test.py runs/train/earbud_detection/weights/best.pt
```

## 6. Đánh giá chi tiết trên dataset test

```bash
python test_model.py --model runs/train/earbud_detection/weights/best.pt --test-set
```

Chỉ số hiển thị gồm:
- mAP50
- mAP50-95
- Precision
- Recall
- metrics theo từng class

## 7. Chạy inference trên 1 ảnh

```bash
python test_model.py --model runs/train/earbud_detection/weights/best.pt --source test/images/example.jpg
```

## 8. Chạy inference trên video

Phương án 1: dùng script local:
```bash
python run_inference_improved.py
```

Phương án 2: dùng script cũ / tùy chỉnh:
```bash
python run_inference_video.py
```

Nếu cần chỉ định model và video bằng command line, hãy kiểm tra nội dung script để chỉnh đường dẫn phù hợp.

## 9. Gợi ý tối ưu hiệu suất

### Tăng độ chính xác
- tăng `epochs`
- dùng model lớn hơn (`yolov8s.pt`, `yolov8m.pt`)
- tăng chất lượng dataset và annotation

### Giảm RAM / VRAM
- giảm `batch`
- giảm `imgsz`
- dùng `yolov8n.pt`

## 10. Xử lý sự cố

### Out of Memory
- giảm `batch` từ 16 xuống 8 hoặc 4
- giảm `imgsz` từ 640 xuống 416

### Training chậm
- dùng GPU nếu có
- giảm số worker trong script
- tắt các biểu đồ không cần thiết nếu cần tối ưu tốc độ

### Model sai nhầm lớp
- kiểm tra `data.yaml`
- kiểm tra `labels` và annotation
- xem `confusion_matrix.png`

## 11. Mẹo thực tế

- Luôn kiểm tra dataset bằng `train/labels` và `valid/labels`
- Nên bắt đầu với `yolov8n.pt` để debug pipeline trước
- Sau khi model ổn, chuyển sang model lớn hơn để cải thiện mAP
- Lưu lại `best.pt` để dùng cho inference và deploy

## 12. Tài liệu tham khảo nhanh

- `README.md` - tóm tắt dự án
- `data.yaml` - cấu hình dataset và label names
- `train_local.py` - script huấn luyện chính
- `quick_test.py` - kiểm tra nhanh
- `test_model.py` - đánh giá và inference
- `run_inference_improved.py` - inference video

## 13. Kết luận

Đây là một project YOLOv8 hoàn chỉnh cho bài toán phát hiện earbud trong hộp sạc, phù hợp để train local, test nhanh, và chạy inference trên ảnh/video mà không cần phụ thuộc vào API bên ngoài.


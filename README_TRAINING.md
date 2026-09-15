# Hướng dẫn chi tiết train và test mô hình YOLOv8

## 1. Mục tiêu dự án

Dự án này dùng YOLOv8 để phát hiện:
- vỏ hộp sạc tai nghe
- tay người
- tai nghe trái và phải
- vị trí khe trống trong hộp

Mục tiêu là nhận diện đúng vị trí và trạng thái của từng đối tượng trên ảnh hoặc video.

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

Các lớp dữ liệu chuẩn:
- Case
- Hand
- Left_Earbud
- Left_Slot
- Right_Earbud
- Right_Slot

## 3. Cài đặt môi trường

Từ thư mục dự án, chạy:

```bash
pip install ultralytics opencv-python pandas numpy
```

Nếu máy có GPU NVIDIA, hãy chắc chắn driver và CUDA đã được cài đặt đúng.

## 4. Huấn luyện mô hình

Chạy lệnh:

```bash
python train_local.py
```

Một số tham số quan trọng trong file `train_local.py` như:
- `epochs=100`
- `imgsz=640`
- `batch=16`
- `model='yolov8n.pt'`

Bạn có thể đổi sang model lớn hơn:
```python
model = YOLO('yolov8s.pt')
# hoặc yolov8m.pt, yolov8l.pt
```

Kết quả huấn luyện sẽ được lưu trong thư mục:
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

## 6. Đánh giá chi tiết trên tập test

```bash
python test_model.py --model runs/train/earbud_detection/weights/best.pt --test-set
```

Các chỉ số hiển thị gồm:
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

Cách 1: dùng script local
```bash
python run_inference_improved.py
```

Cách 2: dùng script cũ hoặc tùy chỉnh
```bash
python run_inference_video.py
```

Nếu muốn truyền đường dẫn model và video bằng lệnh, hãy kiểm tra nội dung script để sửa theo nhu cầu.

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

### Lỗi thiếu bộ nhớ (Out of Memory)
- giảm `batch` từ 16 xuống 8 hoặc 4
- giảm `imgsz` từ 640 xuống 416

### Huấn luyện chậm
- dùng GPU nếu có
- giảm số worker trong script
- tắt các biểu đồ không cần thiết nếu cần tối ưu tốc độ

### Model nhầm lớp
- kiểm tra `data.yaml`
- kiểm tra file annotation trong `labels`
- xem `confusion_matrix.png`

## 11. Mẹo thực tế

- Luôn kiểm tra dữ liệu trong `train/labels` và `valid/labels`
- Nên bắt đầu với `yolov8n.pt` trước để debug pipeline
- Sau khi pipeline ổn, chuyển sang model lớn hơn để cải thiện mAP
- Lưu lại `best.pt` để dùng cho inference và triển khai

## 12. Tài liệu tham khảo nhanh

- `README.md` - tóm tắt dự án
- `data.yaml` - cấu hình dataset và tên lớp
- `train_local.py` - script huấn luyện chính
- `quick_test.py` - kiểm tra nhanh
- `test_model.py` - đánh giá và inference
- `run_inference_improved.py` - inference video

## 13. Kết luận

Đây là một dự án YOLOv8 hoàn chỉnh cho bài toán phát hiện tai nghe trong hộp sạc, phù hợp để huấn luyện local, test nhanh và chạy inference trên ảnh/video mà không cần phụ thuộc API bên ngoài.


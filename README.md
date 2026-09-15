# Dự án phát hiện Earbud bằng YOLOv8

Dự án này huấn luyện và chạy mô hình YOLOv8 để phát hiện tai nghe trong hộp sạc, bao gồm thân hộp, tay người, tai nghe trái/phải và các khe trống trong hộp.

## Tổng quan

Mô hình được thiết kế để:
- phát hiện vỏ hộp sạc
- phát hiện tay người
- nhận diện tai nghe trái và phải
- nhận diện khe trống trong hộp
- chạy inference trên video local

## Các lớp đối tượng hỗ trợ
- Case
- Hand
- Left_Earbud
- Left_Slot
- Right_Earbud
- Right_Slot

## Trạng thái dự án
- Dòng mô hình: YOLOv8
- Backbone mặc định: `yolov8n.pt`
- mAP50 tốt nhất ghi nhận: khoảng `0.830`
- Quy trình huấn luyện: huấn luyện local bằng Ultralytics YOLOv8

## Cấu trúc thư mục
```text
RNN/
├── README.md
├── README_TRAINING.md
├── data.yaml
├── train_local.py
├── quick_test.py
├── test_model.py
├── run_inference_improved.py
├── run_inference_video.py
├── merge_slots.py
├── cleanup.py
├── quick_start.py
├── train/
│   ├── images/
│   └── labels/
├── valid/
│   ├── images/
│   └── labels/
├── test/
│   ├── images/
│   └── labels/
├── runs/
│   └── detect/
├── yolov8n.pt
└── .gitignore
```

## Yêu cầu môi trường

Cài đặt các thư viện cần thiết:
```bash
pip install ultralytics opencv-python pandas numpy
```

Nếu dùng GPU NVIDIA, hãy đảm bảo driver và CUDA đã được cài đặt đúng cách.

## Bắt đầu nhanh

### 1) Huấn luyện mô hình
```bash
python train_local.py
```

### 2) Chạy test nhanh
```bash
python quick_test.py
```

### 3) Chạy inference trên video
```bash
python run_inference_improved.py
```

### 4) Đánh giá mô hình
```bash
python test_model.py --model runs/train/earbud_detection/weights/best.pt --test-set
```

## Định dạng dataset
Dataset tuân theo định dạng YOLO:
```text
train/
  images/
  labels/
valid/
  images/
  labels/
test/
  images/
  labels/
```

File annotation là file `.txt` theo chuẩn YOLO với tọa độ chuẩn hóa.

## Các script quan trọng
- `train_local.py`: điểm vào chính để huấn luyện local
- `quick_test.py`: kiểm tra nhanh trên một vài ảnh mẫu
- `test_model.py`: đánh giá toàn bộ và test ảnh/video
- `run_inference_improved.py`: chạy nhận diện trên video đầu vào
- `merge_slots.py`: gom hoặc chuẩn hóa dữ liệu khe trống
- `cleanup.py`: công cụ dọn dữ liệu và chuẩn bị dataset

## Ghi chú khi sử dụng
- Cập nhật `data.yaml` nếu thay đổi đường dẫn dataset hoặc tên lớp.
- Nếu muốn model lớn hơn và chính xác hơn, hãy chuyển từ `yolov8n.pt` sang `yolov8s.pt`, `yolov8m.pt` hoặc lớn hơn.
- Nếu thiếu VRAM, hãy giảm `batch` hoặc kích thước ảnh trong file training.

## Lịch sử huấn luyện
- v1: baseline ban đầu
- v2_fast: tối ưu tốc độ huấn luyện
- v3_fixed: cải thiện phát hiện khe trống và độ chính xác

## Tham khảo thêm
Xem [README_TRAINING.md](README_TRAINING.md) để có hướng dẫn từng bước, cách sửa lỗi và câu lệnh thực tế.

## Giấy phép
Mã nguồn dự án được cung cấp theo dạng “as-is” cho mục đích nghiên cứu và demo. Hãy kiểm tra giấy phép của dataset trước khi triển khai công khai hoặc thương mại hóa.


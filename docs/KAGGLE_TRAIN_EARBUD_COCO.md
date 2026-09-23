# Train dataset Earbud Detect COCO trên Kaggle

Cách đơn giản nhất là import notebook
[`Kaggle_Training_Earbud.ipynb`](../Kaggle_Training_Earbud.ipynb). Notebook đã chứa toàn bộ cell cần thiết; bạn chỉ sửa danh sách trong CELL 1:

```python
DATASET_ROOTS = [
    r'/kaggle/input/earbud-detect/thu-muc-coco',
    r'/kaggle/input/left-right-earbud/thu-muc-coco',
]
```

Có thể để một hoặc nhiều đường dẫn COCO. Sau đó bật GPU và bấm **Run All**. Không cần upload hoặc gọi thêm script Python. Notebook remap class theo **tên lớp**, thêm tiền tố cho tên ảnh và bỏ ảnh trùng theo SHA-256 trước khi train.

File `scripts/train_earbud_coco_kaggle.py` là phương án dòng lệnh tương đương nếu bạn không dùng notebook.

Script sử dụng:

```text
scripts/train_earbud_coco_kaggle.py
```

Script tự thực hiện toàn bộ các bước:

1. Tìm hoặc nhận đường dẫn Roboflow COCO export.
2. Đọc `train/valid/test/_annotations.coco.json`.
3. Bỏ category metadata không có bounding box, ví dụ `earbud-detect`.
4. Remap `category_id` COCO thành class ID YOLO liên tục từ 0.
5. Chuyển bounding box COCO sang YOLO và tạo `data.yaml`.
6. Train YOLO bằng GPU Kaggle.
7. Đánh giá trên test, hoặc validation nếu không có test.
8. Xuất `best_earbud_detector.pt` và `earbud_training_results.zip`.

## 1. Chuẩn bị Kaggle

1. Nén dataset COCO thành ZIP rồi tạo một Kaggle Dataset, hoặc upload trực tiếp thư mục dataset.
2. Import `Kaggle_Training_Earbud.ipynb` vào Kaggle.
3. Chọn **Add Input** để gắn dataset vào notebook.
4. Trong **Settings**, chọn **Accelerator → GPU** và bật Internet để tải `yolo11s.pt`.
5. Sửa duy nhất `DATASET_ROOTS` ở CELL 1 rồi chọn **Run All**.

Cấu trúc COCO được hỗ trợ:

```text
<dataset-root>/
├── train/
│   ├── _annotations.coco.json
│   └── *.jpg
├── valid/
│   ├── _annotations.coco.json
│   └── *.jpg
└── test/
    ├── _annotations.coco.json
    └── *.jpg
```

`val` có thể được dùng thay cho `valid`. Tập `test` không bắt buộc.

## 2. Notebook tự chạy những gì

Các cell đã được viết sẵn theo thứ tự:

1. Cấu hình duy nhất cần sửa là `DATASET_ROOTS`.
2. Cài Ultralytics.
3. Kiểm tra đường dẫn và bắt buộc GPU hoạt động.
4. Đọc COCO, kiểm tra nhãn, chuyển sang YOLO và tạo `data.yaml`.
5. Train YOLO11s trong 80 epoch.
6. Đánh giá `best.pt` trên test hoặc validation.
7. Đóng gói checkpoint và báo cáo.
8. Hiển thị biểu đồ kết quả.

Các lệnh bên dưới chỉ dành cho trường hợp bạn muốn chạy script thủ công thay vì notebook.

## 3. Chạy script thủ công (tùy chọn)

Cell cài thư viện:

```python
!pip install -q -U ultralytics
```

Cell kiểm tra GPU:

```python
import torch
print(torch.__version__)
print("CUDA:", torch.cuda.is_available())
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NONE")
```

Cell tìm đường dẫn dataset:

```python
from pathlib import Path

for json_file in Path("/kaggle/input").rglob("_annotations.coco.json"):
    print(json_file)
```

Nếu Kaggle chỉ có một COCO export, script có thể tự tìm:

```python
!python /kaggle/working/train_earbud_coco_kaggle.py \
    --model yolo11s.pt \
    --epochs 80 \
    --imgsz 640 \
    --batch 16
```

Nếu có nhiều dataset, truyền chính xác thư mục chứa `train/valid/test`:

```python
!python /kaggle/working/train_earbud_coco_kaggle.py \
    --dataset-root "/kaggle/input/earbud-detect/earbud detect.coco" \
    --model yolo11s.pt \
    --epochs 80 \
    --imgsz 640 \
    --batch 16
```

Nếu GPU báo hết VRAM, đổi `--batch 16` thành `--batch 8`. Muốn thử nhanh pipeline trước, dùng `--epochs 3`.

## 4. Kiểm tra dataset trước khi tốn thời gian train

Chạy convert và validate nhưng chưa train:

```python
!python /kaggle/working/train_earbud_coco_kaggle.py \
    --dataset-root "/kaggle/input/earbud-detect/earbud detect.coco" \
    --prepare-only
```

Nếu dataset đã được gán lại đúng hướng mới, bắt buộc kiểm tra sáu lớp geometry:

```python
!python /kaggle/working/train_earbud_coco_kaggle.py \
    --dataset-root "/kaggle/input/earbud-detect/earbud detect.coco" \
    --require-geometry-v2 \
    --epochs 80
```

Sáu lớp bắt buộc của geometry v2:

```text
open_case
close_case
left_earbud
right_earbud
empty_left
empty_right
```

Script đặt `fliplr=0.0` và `flipud=0.0`. Không bật lật ảnh tự động khi class phân biệt tai trái/phải, vì ảnh bị lật nhưng class ID không tự hoán đổi sẽ làm model học sai.

Nếu COCO hiện vẫn là `Earphone_Case`, `Empty_Slot`, `Hand`, `Left_Earbud`, `Right_Earbud`, script vẫn có thể train detector baseline. Tuy nhiên checkpoint đó chưa phân biệt hộp mở/đóng và khe trái/phải, nên chưa đủ để tự kiểm tra chính xác toàn bộ quy trình mới.

Đặc biệt, để báo violation khi tháo từng tai nghe, model phải nhận diện được `empty_left` và `empty_right` xuất hiện trở lại. Class `Empty_Slot` chung của dataset cũ không đủ tin cậy để xác định đầy đủ chuyển trạng thái trái/phải. Không cần tạo class YOLO `remove`; runtime suy ra việc tháo từ trạng thái trước và sau.

## 5. Tải kết quả về máy

Sau khi train, mở panel **Output** của Kaggle và tải:

```text
/kaggle/working/best_earbud_detector.pt
/kaggle/working/earbud_training_results.zip
```

Với dataset geometry v2, chép checkpoint vào:

```text
G:\Internship\RBCNN_Demo\artifacts\training\earbud_geometry_detector\weights\best.pt
```

Sau đó chạy camera:

```powershell
cd "G:\Internship\RBCNN_Demo"
.\.venv\Scripts\python.exe scripts\run_earbud.py --mode camera --source 0 --auto-advance
```

Không chép checkpoint schema cũ vào đường dẫn geometry v2. Tên class trong checkpoint phải khớp cấu hình `configs/camera_earbud_config.json`.

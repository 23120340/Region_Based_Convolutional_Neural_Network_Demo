# Camera thời gian thực: nhận diện linh kiện bút và nối với quy trình

## 1. Cái gì chạy được ngay?

`scripts/run_camera.py` mở webcam, dùng YOLO-World với các text prompt trong `configs/camera_config.json`, khoanh vùng năm lớp:

- `barrel`: thân dưới;
- `refill`: ruột mực;
- `spring`: lò xo;
- `cap`: nắp/thân trên;
- `assembled_pen`: bút đã lắp.

YOLO-World là open-vocabulary detector: có thể thử các lớp mới bằng `set_classes()` mà chưa cần train. Webcam hoặc OpenCV frame đều là nguồn inference được Ultralytics hỗ trợ. Tài liệu chính thức: [YOLO-World](https://docs.ultralytics.com/models/yolo-world/), [Predict mode](https://docs.ultralytics.com/modes/predict/).

## 2. Vì sao vẫn có phím Space?

Bounding box chỉ xác nhận linh kiện xuất hiện. Ba ví dụ dễ gây sai:

- thấy lò xo trên tay không có nghĩa lò xo đã được luồn vào ruột;
- thấy nắp trong vùng lắp không có nghĩa nắp đã được vặn chặt;
- thấy cây bút hoàn chỉnh không chứng minh người dùng vừa bấm thử.

Do đó chế độ mặc định dùng detector để **gợi ý** bước khi linh kiện đúng nằm ổn định trong WORK ZONE, sau đó dùng Space để xác nhận. `--auto-advance` có sẵn để thử nghiệm bốn bước đầu, nhưng chưa phải chế độ nghiệm thu.

## 3. Cách bố trí và chạy

1. Dán giấy nền ít hoa văn, ánh sáng đều.
2. Đặt bốn khay ở phía trên hoặc hai bên, bên ngoài khung WORK ZONE màu vàng.
3. Để camera laptop nhìn nghiêng xuống bàn; nếu có tripod thì top-down tốt hơn.
4. Tháo rời bút và đặt mỗi linh kiện cách nhau đủ xa.
5. Chạy:

```powershell
python scripts/run_camera.py --source 0
```

Nếu camera mặc định không đúng, thử `--source 1`. Có thể chỉnh ROI chuẩn hóa, confidence, prompt và tần suất inference tại `configs/camera_config.json`.

Ngưỡng `confidence` zero-shot mặc định thấp (`0.12`) để thăm dò. Nếu xuất hiện nhiều box sai, tăng dần lên `0.20`–`0.30`. Nếu bỏ sót, giảm nhẹ hoặc đưa camera gần hơn. Lò xo là đối tượng khó nhất vì rất nhỏ.

## 4. Có cần dataset và train không?

Không bắt buộc để chạy bản thử zero-shot, nhưng **cần** nếu mục tiêu là demo ổn định hoặc đánh giá khoa học. Dataset detector phải là ảnh của chính bút/góc quay dự kiến và có bounding box cho từng linh kiện.

Mốc khởi đầu thực dụng:

- 200–400 ảnh đã gán box, gồm tay che một phần, nền và ánh sáng khác nhau;
- ưu tiên nhiều ảnh cận cảnh cho `spring` và `refill`;
- có ảnh âm tính: bàn trống, tay, điện thoại, kéo, bút khác;
- chia train/val/test theo buổi quay hoặc người, không chia các frame liền nhau ngẫu nhiên.

Thu ảnh:

```powershell
python scripts/capture_detection_images.py --camera 0
```

Nhấn Space để lưu vào `datasets/pen_parts/raw/`, sau đó gán bounding box bằng CVAT hoặc Label Studio và xuất YOLO detection format:

```text
datasets/pen_parts/
├── images/train, images/val, images/test
├── labels/train, labels/val, labels/test
└── data.yaml
```

Mỗi file label có các dòng `class_id x_center y_center width height`, tọa độ chuẩn hóa từ 0 đến 1. Cấu trúc dataset và API huấn luyện tuân theo [Ultralytics detection datasets](https://docs.ultralytics.com/datasets/detect/).

Train baseline:

```powershell
python scripts/train_detector.py --data datasets/pen_parts/data.yaml --epochs 60
```

Checkpoint sẽ nằm dưới `artifacts/training/pen_parts_detector/weights/best.pt`. Chạy nó bằng:

```powershell
python scripts/run_camera.py --source 0 --model artifacts/training/pen_parts_detector/weights/best.pt
```

Script camera tự chọn backend: checkpoint có `world` trong tên dùng custom prompt YOLO-World; checkpoint fine-tune đóng lớp dùng tên class `barrel/refill/spring/cap/assembled_pen` từ `data.yaml`.

## 5. Để tự động hoàn toàn cần thêm dữ liệu gì?

Detector linh kiện và action recognizer là hai bài toán khác nhau:

| Model | Nhãn | Trả lời câu hỏi |
|---|---|---|
| Detector YOLO | Bounding box `barrel/refill/spring/cap/pen` | Linh kiện nào đang ở đâu? |
| ViT + LSTM | Đoạn video `pick/insert/screw/test` | Người dùng đang thực hiện hành động nào? |
| FSM | State và luật chuyển | Hành động có đúng thứ tự không? |

Muốn bỏ Space, cần video gán nhãn theo thời gian cho năm action, ngoài ảnh bounding box. Khi đó đầu ra ViT + LSTM đi qua debouncer rồi vào FSM hiện có.

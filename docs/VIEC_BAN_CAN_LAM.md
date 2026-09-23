# Việc bạn cần làm tiếp theo

Cập nhật: 22/09/2026. Bản chính: `G:\Internship\RBCNN_Demo`.

## Phần tôi đã chuẩn bị

- Đã hợp nhất repo theo hướng earbud; loại tên và đường dẫn mặc định của dự án cũ.
- Action v2 có sáu nhãn: `idle`, `open_case`, `insert_first_earbud`, `insert_second_earbud`, `close_case`, `remove_earbud`.
- Backbone v2 là `facebook/dinov2-small`, embedding 384 chiều.
- GUI, FSM, Fusion và project profile v2 đã thống nhất tên action.
- Script annotation, split, kiểm tra pipeline, train và evaluation đã được nâng cấp.
- Camera có thể liệt kê/chuyển thiết bị; phím `S` lưu ảnh có overlay.
- Dataset/model baseline được giữ để đối chiếu nhưng không dùng lẫn với v2.

## Trạng thái còn thiếu

- `annotations_v2.csv` đang rỗng ngoài dòng header.
- Chưa có feature DINOv2 trong `data/earbud_actions/features_v2/`.
- Chưa có `artifacts/action_model_v2/best.pt`.
- Chưa có dataset sáu lớp geometry trái/phải hoàn chỉnh tại `datasets/earbud_geometry/`.
- Chưa có `artifacts/training/earbud_geometry_detector/weights/best.pt`.
- PyTorch đang cài là bản CPU; CUDA hiện trả về `False`.

## A. Việc bạn phải làm cho YOLO

### A1. Thu ảnh

```powershell
cd "G:\Internship\RBCNN_Demo"
python scripts/capture_detection_images.py `
  --camera 0 `
  --output datasets/earbud_geometry/raw/person01/session01
```

Chụp các trạng thái: hộp mở hai khe trống; một tai trong hộp; hai tai trong hộp; hộp đóng; tai nằm ngoài/gần hộp; tay che một phần; nhiều góc xoay và ánh sáng.

Để bắt lỗi tháo tai nghe, phải chụp đủ cả hai hướng tháo:

- từ hai tai trong hộp sang chỉ còn tai trái, `empty_right` xuất hiện lại;
- từ hai tai trong hộp sang chỉ còn tai phải, `empty_left` xuất hiện lại;
- từ một tai trong hộp sang không còn tai nào, cả hai khe trống xuất hiện lại;
- tay đang che lúc tháo và 3–5 frame sau khi tay rời hộp để detector thấy rõ trạng thái cuối.

Đây vẫn là ảnh detection bình thường, không tạo class YOLO tên `remove`. Violation được suy ra từ chuyển trạng thái số tai trong hộp và khe trống xuất hiện trở lại.

Không lấy hàng loạt frame giống nhau. Mỗi group nên có ngày/người/góc quay riêng để chia train/val/test.

### A2. Gán bounding box

Dùng Roboflow, CVAT hoặc Label Studio với đúng thứ tự sáu lớp:

```text
0 open_case
1 close_case
2 left_earbud
3 right_earbud
4 empty_left
5 empty_right
```

Không tự đổi dataset 3 hoặc 6 lớp hiện có sang schema này vì lớp hộp mở/đóng và khe trái/phải chưa có đủ thông tin.

### A3. Export, kiểm tra và train

Export YOLO, đặt tại `datasets/earbud_geometry/`, rồi chạy:

```powershell
python scripts/validate_detection_dataset.py `
  --data datasets/earbud_geometry/data.yaml

python scripts/train_detector.py `
  --data datasets/earbud_geometry/data.yaml `
  --model yolo11n.pt `
  --epochs 100 `
  --device 0 `
  --name earbud_geometry_detector
```

## B. Việc bạn phải làm cho DINOv2 + BiLSTM

Bạn đã có video ở session 01 và 02 nhưng cần gán lại theo sáu action. Làm đúng năm bước sau; chi tiết phím và cách review nằm trong [LSTM_Training_Guide.md](LSTM_Training_Guide.md).

### Bước 1 — Annotation

```powershell
python scripts/annotate_actions.py `
  --session 01 `
  --session 02 `
  --output data/earbud_actions/annotations_v2.csv `
  --config configs/action_earbud_v2_config.json
```

Gán hai đoạn lắp riêng: lần 0→1 là `insert_first_earbud`, lần 1→2 là `insert_second_earbud`. Đoạn nhấc bất kỳ tai nào ra khỏi hộp là `remove_earbud`. Không đổi tự động nhãn `insert_earbud` cũ.

### Bước 2 — Chia tập

Với pilot chỉ có cùng người/session:

```powershell
python scripts/split_annotations.py `
  --annotations data/earbud_actions/annotations_v2.csv `
  --output data/earbud_actions/annotations_v2_split.csv `
  --config configs/action_earbud_v2_config.json `
  --mode video-ratio `
  --allow-same-session `
  --train 0.70 --val 0.15 --test 0.15
```

Kết quả chính thức phải quay thêm ít nhất ba group person/session và dùng `--mode group-ratio`.

### Bước 3 — Trích DINOv2 feature

```powershell
python scripts/extract_spatial_features.py `
  --videos-dir data/earbud_actions/raw_videos `
  --output-dir data/earbud_actions/features_v2 `
  --config configs/action_earbud_v2_config.json `
  --overwrite
```

Lần đầu cần tải `facebook/dinov2-small`; sau đó weights nằm trong cache Hugging Face.

### Bước 4 — Kiểm tra và train BiLSTM

```powershell
python scripts/verify_pipeline.py `
  --annotations data/earbud_actions/annotations_v2_split.csv `
  --features-dir data/earbud_actions/features_v2 `
  --config configs/action_earbud_v2_config.json `
  --allow-same-session

python scripts/train_action_model.py `
  --annotations data/earbud_actions/annotations_v2_split.csv `
  --features-dir data/earbud_actions/features_v2 `
  --config configs/action_earbud_v2_config.json `
  --output artifacts/action_model_v2 `
  --allow-same-session
```

### Bước 5 — Evaluation

```powershell
python scripts/evaluate_action_model.py `
  --checkpoint artifacts/action_model_v2/best.pt `
  --annotations data/earbud_actions/annotations_v2_split.csv `
  --features-dir data/earbud_actions/features_v2 `
  --config configs/action_earbud_v2_config.json `
  --split test `
  --allow-same-session `
  --output artifacts/action_model_v2/evaluation_test.json
```

## C. Chạy hệ thống

Chỉ YOLO geometry:

```powershell
python scripts/run_earbud.py --mode camera --source 0 --auto-advance --device 0
```

Với `--auto-advance`, việc tháo tai được báo tự động:

```text
2 tai -> 1 tai: remove_earbud_to_one  -> VIOLATION -> quay lại bước lắp tai thứ hai
1 tai -> 0 tai: remove_earbud_to_zero -> VIOLATION -> quay lại bước lắp tai thứ nhất
2 tai -> 0 tai: remove_earbud_to_zero -> VIOLATION -> quay lại bước lắp tai thứ nhất
```

Hệ thống chỉ xác nhận thay đổi sau `dwell_frames=3` kết quả YOLO liên tiếp để tay che hoặc một frame detect hụt không gây báo lỗi giả.

Hybrid v2 sau khi có đủ hai checkpoint:

```powershell
python scripts/run_hybrid.py `
  --project configs/projects/earbud_v2.json `
  --source 0 `
  --device 0 `
  --allow-download
```

## D. Dữ liệu nên quay thêm

- Ít nhất 3 person/session độc lập; toàn bộ group chỉ thuộc một split.
- Mỗi group: 15–20 chu trình đúng.
- Mỗi lỗi chính: ít nhất 10 clip, gồm đóng khi 0 tai, đóng khi 1 tai, đặt tai ngoài hộp, tháo tai đã lắp, che khuất và sai thứ tự.
- Có `idle` thật: tay đi ngang, chỉnh camera, cầm vật nhưng không thực hiện bước.

## E. Tiêu chí trước khi báo cáo kết quả

- Không còn video xuất hiện ở nhiều split.
- Mỗi split có đủ năm action hoặc nêu rõ lớp thiếu.
- YOLO có metric từng lớp và ảnh false positive/false negative.
- Action model có confusion matrix 5×5 và macro-F1 test.
- Camera chạy 10 chu trình đúng liên tiếp; thử từng lỗi ít nhất 5 lần.
- Không dùng kết quả `--allow-same-session` như bằng chứng tổng quát hóa cuối.

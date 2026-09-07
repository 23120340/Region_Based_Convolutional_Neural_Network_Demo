# Tóm tắt thay đổi theo bản nhận xét và đánh giá repo

## 1. Phạm vi

Các thay đổi này xử lý những điểm có thể sửa bằng code ngay, đồng thời không giả lập kết quả ML khi chưa có nhãn thật. Dữ liệu ảnh/video của người dùng được giữ nguyên; không tự ý xóa, khôi phục hoặc chia lại.

## 2. Các lỗi ưu tiên đã sửa

### 2.1. BiLSTM aggregation

Trước đây classifier dùng:

```python
sequence[:, -1, :]
```

Với BiLSTM, backward component tại vị trí cuối không đại diện đầy đủ cho toàn chuỗi. Model mới lấy final hidden state của forward và backward ở layer cuối rồi ghép lại:

```python
forward = hidden[-2]
backward = hidden[-1]
context = torch.cat((forward, backward), dim=1)
```

Implementation mới: `src/pen_assembly/models/action_net.py`.

### 2.2. Không lọc wrong action trước FSM

`ComponentDwellGate` cũ chỉ giữ action thuộc `expected_actions`. Vì vậy khi FSM chờ `insert_refill` nhưng camera thấy `cap`, evidence `screw_cap` bị bỏ và FSM không thể báo lỗi.

Gate mới:

- theo dõi mọi linh kiện hợp lệ đi vào WORK ZONE;
- đánh dấu suggestion là expected hoặc unexpected nhưng không loại bỏ;
- chống phát lại linh kiện đang giữ nguyên trong vùng;
- cho phép linh kiện mới phát event dù linh kiện cũ vẫn còn thấy;
- để FSM là nơi duy nhất quyết định PASS/VIOLATION.

### 2.3. Bỏ hard-code embedding dimension khỏi model

Đã thêm `configs/action_model_config.json`, tách riêng:

- backbone và `embedding_dim`;
- FPS lấy mẫu;
- sequence length và stride;
- hidden size, layer, bidirectional, dropout;
- batch size, epoch, learning rate và seed;
- danh sách sáu action.

Khi đổi từ ViT-Base 768 chiều sang backbone khác, sửa config và cache lại feature thay vì sửa code LSTM.

## 3. Pipeline ViT + LSTM đã bổ sung

### 3.1. Quay video temporal

Thêm `scripts/record_assembly_videos.py`:

- quay một chu trình/clip bằng Space;
- bắt buộc khai báo person, session và scenario;
- tự đặt tên video duy nhất;
- tự ghi `recording_log.csv`.

### 3.2. Spatial encoder thật

Thêm `src/pen_assembly/models/spatial_encoder.py`:

- tải Hugging Face ViT theo config;
- preprocess ảnh;
- lấy CLS embedding từng frame;
- freeze backbone cho giai đoạn cache feature.

### 3.3. Feature caching

Thêm `scripts/extract_spatial_features.py`:

- đọc video đệ quy theo person/session;
- lấy mẫu theo FPS trong config;
- trích embedding một lần;
- lưu `<video_id>.npy` và metadata `<video_id>.json`.

### 3.4. Temporal window dataset

Thêm `src/pen_assembly/action_dataset.py`:

- đọc annotation theo thời gian giây;
- chỉ lấy đúng split train/val/test;
- tạo sliding window 16 frame;
- padding đoạn ngắn;
- kiểm tra embedding dimension;
- cache tối đa bốn video trong RAM và không giữ mmap khóa file trên Windows.

### 3.5. Train và evaluate

Thêm:

- `scripts/train_action_model.py` — AdamW, CrossEntropy, CosineAnnealing, lưu checkpoint tốt nhất theo validation Macro-F1.
- `scripts/evaluate_action_model.py` — Macro-F1, confusion matrix và classification report trên val/test.
- `data/pen_actions/annotations_template.csv` — template annotation thống nhất.
- `src/pen_assembly/models/vit_lstm_recognizer.py` — adapter trả `Prediction(action, confidence)` cho runtime sau khi có checkpoint.

Dataset temporal tự từ chối nếu cùng một `video_id` hoặc cùng cặp `person/session` xuất hiện ở nhiều split. Đây là hàng rào chống data leakage, không chỉ là lời nhắc trong tài liệu.

## 4. Bảo vệ pipeline detector

Kiểm tra repo ngày 03/09/2026 cho thấy:

```text
61 images
61 label files
0 non-empty label files
0 bounding boxes
```

Đã thêm:

- `src/pen_assembly/detection_dataset.py` để kiểm tra ảnh, label, tọa độ và số box theo lớp.
- `scripts/validate_detection_dataset.py` để người dùng chạy trước khi train.
- `scripts/train_detector.py` tự gọi validator và dừng nếu dataset không train được.

Vì vậy lệnh train hiện dừng có chủ đích cho tới khi có bounding box thật; hệ thống không còn âm thầm train 61 ảnh như background.

## 5. Tổ chức package và storage

- Neural model thật được chuyển vào `src/pen_assembly/models/`.
- `models/pen_action_net.py` được giữ làm compatibility import để không phá code cũ.
- `.gitignore` bỏ qua raw video, feature cache, ảnh dataset lớn, cache Ultralytics và model `.h5`.
- Annotation/config/code vẫn có thể lưu bằng Git.
- `scripts/split_dataset.py` không còn mặc định tạo label rỗng; tùy chọn này chỉ bật rõ ràng bằng `--create-empty-labels` cho ảnh background thật. Script cũng cảnh báo rằng random split chỉ dùng smoke test.

Không đổi tên thư mục repo trong lần sửa này vì thao tác đó có thể phá remote, đường dẫn tài liệu và môi trường đang dùng. Việc đổi tên repo nên thực hiện riêng khi người dùng xác nhận.

## 6. Model ImageNet-1K hiện có

`models/tf_model.h5` đã được kiểm tra ở chế độ read-only:

- TensorFlow/Keras weights;
- ViT-Base, patch 16×16;
- 12 encoder layer;
- embedding dimension 768;
- classifier 1.000 lớp ImageNet.

Model này có ích làm pretrained spatial backbone nhưng không phải object detector và không trả bounding box. Pipeline PyTorch hiện dùng model name trong `action_model_config.json`. Việc chuyển/tích hợp trực tiếp `.h5` cần đúng Hugging Face ViT config và image processor, nên chưa ép file này vào YOLO hoặc LSTM sai mục đích.

## 7. Những việc cố ý chưa làm

- Không train YOLO vì tất cả label hiện rỗng.
- Không train LSTM vì chưa có temporal videos và annotations.
- Không công bố accuracy/F1 giả.
- Không xóa hoặc chia lại 61 ảnh của người dùng.
- Không bật tự động action recognition khi chưa có checkpoint được đánh giá.
- Không đổi tên repo hoặc Git remote.

## 8. Cách kiểm thử

Các nhóm test bao gồm:

- FSM và violation.
- Temporal debouncer.
- Camera config/switching.
- Component dwell và wrong-action passthrough.
- BiLSTM forward/backward hidden aggregation.
- Action config và tensor shape.
- Temporal window padding.
- Detection dataset validation.

Kết quả kiểm thử cập nhật ngày 07/09/2026:

```text
Ran 34 tests
OK
```

Ngoài ra, `python -m compileall -q src scripts tests models` hoàn tất với exit code `0`. Validator detector dừng đúng chủ đích vì 61 label hiện có chưa chứa bounding box; các lệnh train/evaluate action cũng dừng bằng thông báo hướng dẫn khi chưa có annotation, feature hoặc checkpoint.

## 9. Điểm bắt đầu tiếp theo

Người dùng thực hiện theo `docs/VIEC_BAN_CAN_LAM.md`. Theo lần kiểm tra ngày 07/09/2026, repo có 317 ảnh raw trong hai session của `person01`, nhưng các split train/val/test đang trống và chưa có bounding box. Bước gần nhất là chọn 20–30 ảnh sạch từ `session02`, gán nhãn thử, sau đó thu thêm session độc lập trước khi train.

## 10. Cập nhật công cụ capture detection

`scripts/capture_detection_images.py` đã được chỉnh để ảnh JPG lưu ra không chứa dòng chữ điều khiển của cửa sổ preview. Script hỗ trợ thêm `--width`, `--height`, `--no-mirror`, hiển thị độ phân giải camera thực tế và dùng timestamp microsecond để tránh ghi đè ảnh khi chạy nhiều lần. Hướng dẫn chụp lại theo từng person/session nằm tại mục 4 của `docs/VIEC_BAN_CAN_LAM.md`.

## 11. Kiểm tra trước khi đưa lên GitHub

Ngày 07/09/2026:

- 34/34 unit test đạt và toàn bộ `src`, `scripts`, `tests`, `models` compile thành công;
- không phát hiện file không bị ignore nào lớn hơn 5 MB;
- raw images, model weights, artifacts, feature cache và 61 label rỗng đều không được stage;
- `docs/plan.md` chỉ được chuyển từ UTF-16 sang UTF-8, nội dung không thay đổi;
- remote hiện vẫn là `23120340/Region_Based_Convolutional_Neural_Network_Demo`.

Checkpoint thử `pen_parts_detector-3` có precision, recall và mAP bằng 0 nên không được phát hành như model hợp lệ.

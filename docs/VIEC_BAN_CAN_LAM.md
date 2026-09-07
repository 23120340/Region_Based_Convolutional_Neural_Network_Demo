# Việc bạn cần làm để hoàn thiện hệ thống nhận diện và kiểm tra lắp ráp bút

## 1. Trạng thái dữ liệu hiện tại

Ngày kiểm tra gần nhất: **07/09/2026**.

Repo hiện có:

| Vị trí | Số ảnh | Trạng thái |
|---|---:|---|
| `raw/person01/session01` | 192 | Ảnh cũ có chữ giao diện trên ảnh; không nên dùng làm dữ liệu chính |
| `raw/person01/session02` | 125 | Ảnh sạch hơn, có thể chọn lọc để gán nhãn |
| `images/train` | 0 | Chưa tạo split chính thức |
| `images/val` | 0 | Chưa tạo split chính thức |
| `images/test` | 0 | Chưa tạo split chính thức |
| Tổng ảnh raw | 317 | Chưa có bounding box |

Thư mục `labels/` còn 61 file `.txt` rỗng từ lần chia thử trước. Chúng không phải annotation hoàn chỉnh và đã được thêm vào `.gitignore` để không bị đưa nhầm lên GitHub. **Không chạy train detector trước khi hoàn thành mục 3.** Script train đã được chặn để tránh tạo model sai.

Checkpoint `artifacts/training/pen_parts_detector-3/weights/best.pt` là kết quả thử cũ với precision/recall/mAP bằng 0 và không được dùng làm model cuối. Thư mục `artifacts/` cũng không được push lên GitHub.

## 2. Chuẩn bị bàn và camera

Bạn cần chuẩn bị:

- 2–3 cây bút bấm cùng mẫu và cùng màu trong vòng thử đầu.
- Bốn khay: `barrel`, `refill`, `spring`, `cap`.
- Một WORK ZONE ở giữa, không chồng lên các khay.
- Nền trơn tương phản với ruột và lò xo.
- Camera cố định; ưu tiên top-down, nếu dùng camera laptop thì nghiêng khoảng 45°.
- Ánh sáng đều từ hai phía nếu có thể.

Kiểm tra camera:

```powershell
cd "E:\Professional documents\Internship\RBCNN_Demo"
python scripts/run_camera.py --list-cameras
python scripts/run_camera.py --source 0
```

Nếu gắn webcam ngoài, chọn index được liệt kê hoặc nhấn `C` trong cửa sổ camera để chuyển thiết bị.

Không thay đổi vị trí camera giữa các clip của cùng một `session`. Nếu camera, nền hoặc ánh sáng thay đổi đáng kể, tăng session: `session01`, `session02`, ...

## 3. Việc ưu tiên số 1: chọn ảnh raw và gán bounding box

### 3.1. Tạo project annotation

Dùng CVAT hoặc Label Studio, tạo đúng năm lớp theo đúng ID:

| ID | Tên lớp | Khi nào dùng |
|---:|---|---|
| 0 | `barrel` | Thân dưới đang tháo rời |
| 1 | `refill` | Ruột/mực bút |
| 2 | `spring` | Lò xo |
| 3 | `cap` | Nắp hoặc thân trên |
| 4 | `assembled_pen` | Cây bút đã hoàn chỉnh |

Trước mắt chọn khoảng 20–30 ảnh rõ từ `session02` để làm pilot và upload lên CVAT:

```text
datasets/pen_parts/raw/person01/session02/
```

Không upload toàn bộ `session01` làm dataset chính vì ảnh ở session này có chữ điều khiển nằm trong hình. Không cần xóa ảnh; chỉ không chọn chúng khi tạo split cuối.

### 3.2. Quy tắc khoanh box

1. Box ôm sát phần nhìn thấy của vật thể.
2. Không đưa cả bàn tay vào box linh kiện.
3. Một ảnh có nhiều linh kiện thì gán đủ tất cả linh kiện nhìn thấy rõ.
4. Vật bị che một phần nhưng vẫn xác định được thì vẫn gán.
5. Không gán `assembled_pen` khi bút chưa đủ ruột, lò xo và nắp.
6. Không gán các bộ phận đã nằm hoàn toàn bên trong bút và không còn nhìn thấy.
7. Ảnh thật sự không có linh kiện mới được để label rỗng.

### 3.3. Xuất và đặt file nhãn

Xuất theo **YOLO detection format**. Mỗi ảnh phải có file `.txt` cùng tên:

```text
images/train/pen_001.jpg
labels/train/pen_001.txt
```

Mỗi dòng label:

```text
class_id x_center y_center width height
```

Ví dụ:

```text
0 0.512500 0.608333 0.312500 0.120833
2 0.421875 0.554167 0.043750 0.062500
```

Tất cả tọa độ được chuẩn hóa trong `[0, 1]`.

### 3.4. Kiểm tra trước khi train

```powershell
python scripts/validate_detection_dataset.py
```

Chỉ được đi tiếp khi cuối báo cáo là:

```text
Total bounding boxes: lớn hơn 0
Trainable: YES
```

Kiểm tra thêm rằng cả năm lớp đều có box. Nếu `spring: 0` hoặc lớp nào đó bằng 0, phải gán/thu thêm ảnh lớp đó.

## 4. Thu thêm ảnh detection đúng cách

317 ảnh raw hiện có mới đến từ một người và hai session. Sau khi loại ảnh trùng, ảnh có chữ và ảnh quá mờ, mục tiêu vòng đầu vẫn nên có khoảng **250–400 ảnh đã gán nhãn**, ưu tiên nhiều instance của `spring` và `refill`.

### 4.1. Hiểu đúng bước capture

`capture_detection_images.py` chỉ chụp và lưu ảnh `.jpg`; bước này chưa tạo bounding box. Sau khi chụp xong, bạn upload ảnh lên CVAT, vẽ Rectangle rồi xuất nhãn YOLO `.txt`.

Luồng đúng:

```text
Camera -> chụp JPG theo từng session -> kiểm tra và loại ảnh hỏng
       -> gắn bounding box bằng CVAT -> xuất YOLO
       -> chia theo session thành train/val/test -> chạy validator -> train
```

Không đưa ảnh mới chưa gán nhãn thẳng vào `images/train` rồi tạo file `.txt` rỗng. YOLO sẽ hiểu nhầm đó là ảnh background.

### 4.2. Chuẩn bị trước khi chụp

1. Lau ống kính và cố định camera; không cầm camera bằng tay khi chụp.
2. Ưu tiên camera nhìn từ trên xuống. Nếu chỉ dùng camera laptop, nghiêng khoảng 45° và giữ nguyên góc trong cả session.
3. Dùng nền trơn tương phản: không dùng nền trắng nếu ruột/lò xo quá khó nhìn.
4. Bật ánh sáng đều, tránh bóng đổ mạnh và phản chiếu trên lò xo.
5. Đặt linh kiện trong vùng camera nhìn rõ nhưng không để chúng chạm mép ảnh.
6. Chuẩn bị ít nhất: thân bút, ruột, lò xo, nắp/thân trên và một bút hoàn chỉnh.

Mỗi khi đổi người, ngày chụp, camera, góc máy, nền hoặc điều kiện ánh sáng đáng kể, hãy tạo `session` mới. Không trộn ảnh các session vào một thư mục.

### 4.3. Kiểm tra camera và chọn đúng index

Liệt kê camera đang kết nối:

```powershell
cd "E:\Professional documents\Internship\RBCNN_Demo"
python scripts/run_camera.py --list-cameras
```

Nếu kết quả là `Camera khả dụng: 0, 1`, thường `0` là camera laptop và `1` là webcam gắn ngoài. Hãy thử đúng index trước khi bắt đầu session.

### 4.4. Lệnh chụp một session

Ví dụ chụp bằng camera laptop:

```powershell
python scripts/capture_detection_images.py `
  --camera 0 `
  --output datasets/pen_parts/raw/person01/session01 `
  --width 1280 `
  --height 720
```

Ví dụ dùng webcam ngoài và không lật gương:

```powershell
python scripts/capture_detection_images.py `
  --camera 1 `
  --output datasets/pen_parts/raw/person01/session02 `
  --width 1280 `
  --height 720 `
  --no-mirror
```

Phím điều khiển:

| Phím | Chức năng |
|---|---|
| `Space` | Lưu một ảnh JPG |
| `Q` hoặc `Esc` | Kết thúc session |

Dòng chữ `SPACE save...` chỉ nằm trên cửa sổ xem trước, không được ghi vào ảnh dataset. Tên ảnh chứa thời gian đến microsecond nên chạy lại script không ghi đè ảnh cũ.

Nếu camera không hỗ trợ đúng `1280×720`, cửa sổ sẽ hiển thị độ phân giải thực tế mà camera trả về. Chất lượng và độ nét quan trọng hơn việc ép đúng một độ phân giải.

### 4.5. Cách chụp trong một session

Một session nên có khoảng 50–80 ảnh. Trước mỗi lần nhấn `Space`, thay đổi ít nhất một yếu tố: vị trí, hướng xoay, khoảng cách, số linh kiện hoặc mức tay che.

Thứ tự đề xuất:

1. **Linh kiện riêng lẻ — 20 ảnh:** mỗi lớp `barrel`, `refill`, `spring`, `cap` khoảng 5 ảnh; xoay ngang, dọc và chéo.
2. **Nhiều linh kiện — 15 ảnh:** đặt 2–4 linh kiện trong cùng ảnh, tách nhau đủ để nhìn rõ.
3. **Tay cầm linh kiện — 15 ảnh:** tay che khoảng 20%, 35% và tối đa khoảng 50%; vật thể vẫn phải nhận dạng được.
4. **Bút hoàn chỉnh — 10 ảnh:** thay đổi góc xoay và vị trí cho lớp `assembled_pen`.
5. **Ảnh âm tính — 5 đến 10 ảnh:** bàn trống, chỉ có tay hoặc các vật gây nhầm như kẹp giấy, bút chì và dây nhỏ. Các ảnh này sau này mới được để label rỗng.

Khi chụp lò xo và ruột bút:

- Không đặt quá xa camera khiến vật chỉ còn vài pixel.
- Dùng nền tương phản và kiểm tra ảnh không bị nhòe.
- Chụp cả khi nằm riêng, trong lòng bàn tay và cạnh thân bút.
- Không chụp hàng chục ảnh liên tiếp khi đồ vật gần như đứng yên.

### 4.6. Kế hoạch tối thiểu cho năm session

| Session | Nội dung chính | Split dự kiến |
|---|---|---|
| `person01/session01` | Ánh sáng và góc máy chuẩn | Train |
| `person01/session02` | Thay nhẹ vị trí/góc camera | Train |
| `person01/session03` | Nhiều tình huống tay che và vật gây nhầm | Train |
| `person01/session04` | Quay riêng, không lấy frame gần session train | Validation |
| `person02/session01` | Người hoặc ngày khác, giữ kín để đánh giá | Test |

Nếu hiện chỉ có một người, hãy chụp test vào ngày khác và thay đổi nhẹ nền/ánh sáng. Không xem trước kết quả test liên tục để điều chỉnh model.

### 4.7. Kiểm tra ảnh ngay sau khi chụp

Đếm và mở thư mục vừa chụp:

```powershell
Get-ChildItem datasets/pen_parts/raw/person01/session01 -Filter *.jpg | Measure-Object
Invoke-Item datasets/pen_parts/raw/person01/session01
```

Kiểm tra nhanh toàn bộ ảnh và chỉ giữ ảnh đáp ứng các điều kiện:

- Đúng linh kiện, đúng session và không có chữ giao diện trên ảnh.
- Ảnh đủ nét để nhìn ra lò xo/ruột.
- Không quá tối, cháy sáng hoặc bị bàn tay che hoàn toàn.
- Không có quá nhiều ảnh gần như giống hệt nhau.
- Mỗi lớp đều xuất hiện đủ trong session validation và test.

Sau khi kiểm tra, upload từng session lên CVAT để gắn bounding box. Ghi lại bảng `person/session -> train/val/test` và không chia ngẫu nhiên các ảnh liên tiếp.

### 4.8. Các tình huống bắt buộc phải có trong toàn bộ dataset

- Mỗi linh kiện nằm ngang, dọc, chéo.
- Linh kiện trong khay và trong WORK ZONE.
- Tay đang cầm, che khoảng 20–50%.
- Nhiều linh kiện xuất hiện cùng lúc.
- Bút hoàn chỉnh ở nhiều góc.
- Ảnh âm tính: bàn trống, chỉ có tay, bút chì, kẹp giấy, dây nhỏ hoặc vật giống lò xo.
- Ánh sáng sáng hơn/tối hơn một chút nhưng vật thể vẫn nhìn rõ.

Không nhấn Space liên tục khi cảnh gần như không thay đổi. Mỗi ảnh nên thay đổi ít nhất một yếu tố: vị trí, góc xoay, tay che, khoảng cách hoặc nền sáng.

### 4.9. Mục tiêu số lượng instance sau khi gắn nhãn

Mục tiêu instance tối thiểu để bắt đầu thử:

| Lớp | Số instance mong muốn |
|---|---:|
| `barrel` | ≥ 150 |
| `refill` | ≥ 200 |
| `spring` | ≥ 250 |
| `cap` | ≥ 150 |
| `assembled_pen` | ≥ 150 |

Đây là mốc khởi đầu, không phải bảo đảm chất lượng. Sau lần train đầu, thu thêm đúng các tình huống model đang sai.

## 5. Chia dataset detection không bị rò rỉ

Không chia các ảnh liên tiếp cùng session ngẫu nhiên sang cả ba tập. Chia theo buổi quay hoặc người:

```text
session01, session02, session03 -> train
session04                    -> validation
session05                    -> test
```

Nếu chỉ có một người, thay đổi session theo ngày/góc camera/ánh sáng. Tốt hơn là test có ít nhất một người chưa xuất hiện trong train.

Dataset hiện tại chỉ có `person01/session01` và `person01/session02`, nên chưa thể tạo test độc lập đáng tin cậy. Khi thu đủ session mới, tạo split theo session và giữ nguyên test cho tới khi model đã chốt.

Không dùng `scripts/split_dataset.py` với cách chia ngẫu nhiên để tạo kết quả báo cáo cuối. Script đó chỉ phù hợp smoke test. Không tạo label rỗng cho ảnh có linh kiện.

## 6. Train và đánh giá detector

Sau khi validator báo `Trainable: YES`:

```powershell
python scripts/train_detector.py --data datasets/pen_parts/data.yaml --epochs 60
```

Checkpoint:

```text
artifacts/training/pen_parts_detector/weights/best.pt
```

Chạy camera bằng model đã train:

```powershell
python scripts/run_camera.py --source 0 --model artifacts/training/pen_parts_detector/weights/best.pt
```

Ghi riêng kết quả validation và test:

- Precision, recall và mAP cho từng lớp.
- False positive khi chỉ có tay/bàn trống.
- Tỷ lệ bỏ sót lò xo và ruột.
- Tốc độ inference trên laptop.

Không chọn confidence threshold bằng test set. Thử các threshold trên validation, chốt một ngưỡng rồi mới chạy test một lần.

## 7. Quay dataset video hành động

Detector chỉ biết vật gì đang ở đâu. Để hệ thống biết bạn đang **cắm**, **vặn** hay **bấm**, cần video temporal riêng.

### 7.1. Quay một session

Ví dụ người 1, buổi 1, quy trình đúng:

```powershell
python scripts/record_assembly_videos.py `
  --source 0 `
  --person person01 `
  --session session01 `
  --scenario correct
```

Trong cửa sổ:

1. Chuẩn bị bút ở trạng thái tháo rời.
2. Nhấn `Space` để bắt đầu quay.
3. Đứng yên khoảng 1 giây (`idle`).
4. Thực hiện đủ chu trình.
5. Đứng yên khoảng 1 giây sau khi bấm thử.
6. Nhấn `Space` để kết thúc clip.
7. Chuẩn bị lại linh kiện rồi quay lượt tiếp theo.

Mỗi clip chỉ chứa một chu trình. Script tự tạo `recording_log.csv`.

### 7.2. Số lượt cần quay

Với mỗi người:

- 10 lượt đúng tốc độ bình thường.
- 5 lượt đúng chậm.
- 5 lượt đúng nhanh.
- 5–10 lượt cho mỗi lỗi được giao.

Tổng mục tiêu:

- 4–5 người.
- 15–20 chu trình đúng/người.
- 15–30 clip cho mỗi loại lỗi.

Các lệnh lỗi:

```powershell
python scripts/record_assembly_videos.py --source 0 --person person01 --session session02 --scenario missing_refill
python scripts/record_assembly_videos.py --source 0 --person person01 --session session03 --scenario missing_spring
python scripts/record_assembly_videos.py --source 0 --person person01 --session session04 --scenario wrong_order
python scripts/record_assembly_videos.py --source 0 --person person01 --session session05 --scenario premature_test
```

## 8. Gán nhãn thời gian cho video

Sao chép template:

```powershell
Copy-Item data/pen_actions/annotations_template.csv data/pen_actions/annotations.csv
```

Mỗi dòng dùng thời gian giây:

```csv
video_id,person_id,session_id,split,start_time_s,end_time_s,action_name,is_anomaly
person01_session01_correct_20260903_101500_001,person01,session01,train,0.000,1.000,idle,0
person01_session01_correct_20260903_101500_001,person01,session01,train,1.000,2.300,pick_barrel,0
```

Sáu action hợp lệ:

```text
idle
pick_barrel
insert_refill
insert_spring
screw_cap
test_click
```

Quy tắc:

- `start_time_s`: lúc tay bắt đầu hành động có mục đích.
- `end_time_s`: lúc vật đã được đặt/lắp xong hoặc tay rời thao tác.
- Gán hết các đoạn rõ ràng, không gán cả clip thành một action.
- Đoạn mơ hồ ở biên có thể bỏ ra thay vì ép nhãn sai.
- Cột `split` phải giống nhau cho toàn bộ đoạn của cùng `video_id`.
- Không để video cùng người/session xuất hiện ở nhiều split.

Dataset loader đã kiểm tra tự động hai quy tắc cuối và sẽ dừng nếu phát hiện rò rỉ split.

## 9. Trích ViT feature, train và đánh giá BiLSTM

Cài dependency nếu chưa có:

```powershell
python -m pip install -r requirements-ml.txt
```

Trích feature một lần:

```powershell
python scripts/extract_spatial_features.py
```

Kết quả mỗi video:

```text
data/pen_actions/features/<video_id>.npy   # shape (N, 768)
data/pen_actions/features/<video_id>.json  # fps, frame count, backbone
```

Train BiLSTM:

```powershell
python scripts/train_action_model.py
```

Đánh giá test sau khi đã chốt model bằng validation:

```powershell
python scripts/evaluate_action_model.py `
  --checkpoint artifacts/action_model/best.pt `
  --split test
```

Không dùng file `models/tf_model.h5` làm YOLO detector. File đó là ViT ImageNet-1K TensorFlow weights. Pipeline mới dùng backbone được khai báo trong `configs/action_model_config.json`; embedding dimension không còn hard-code trong model LSTM.

## 10. Khi nào mới bật tự động kiểm tra hoàn toàn?

Chỉ tích hợp checkpoint action vào realtime khi:

- Detector nhận đủ năm lớp trên validation/test độc lập.
- Action model có Macro-F1 mục tiêu ban đầu ≥ 0,85.
- Recall của `insert_spring`, `screw_cap`, `test_click` đạt mức chấp nhận.
- Thử nghiệm đúng/sai ngoài đời không có cảnh báo giả thường xuyên.
- Confidence và majority vote đã được chọn trên validation.

Camera hiện đã được sửa để không lọc mất hành động sai. Nếu camera thấy `cap` khi FSM đang chờ `refill`, evidence `screw_cap` vẫn được gửi tới FSM để tạo `VIOLATION`.

## 11. Checklist bàn giao cho mình

Khi hoàn thành một giai đoạn, bạn chỉ cần báo và cung cấp đúng phần sau:

### Detector

- [ ] `images/train`, `images/val`, `images/test`.
- [ ] `labels/train`, `labels/val`, `labels/test`.
- [ ] Validator báo `Trainable: YES`.
- [ ] Ghi rõ người/session nào thuộc split nào.

### Action recognition

- [ ] Video nằm trong `data/pen_actions/raw_videos/<person>/<session>/`.
- [ ] `recording_log.csv`.
- [ ] `annotations.csv` có timestamp và split.
- [ ] Ít nhất ba người hoặc nhiều session độc lập.

## 12. Việc bạn nên làm ngay hôm nay

1. Chọn 20–30 ảnh rõ, không trùng nhau từ `person01/session02`.
2. Gán bounding box trên CVAT và kiểm tra đúng thứ tự năm class.
3. Thu thêm `person01/session03`, `person01/session04` và ít nhất một session của `person02` hoặc ngày quay khác.
4. Chia ảnh đã gán nhãn theo session vào `images/` và `labels/` của train/val/test.
5. Chạy `python scripts/validate_detection_dataset.py`.
6. Chỉ train khi validator nhận được box của đủ năm lớp và báo `Trainable: YES`.
7. Chưa dùng checkpoint `pen_parts_detector-3` vì toàn bộ metric của lần train đó bằng 0.

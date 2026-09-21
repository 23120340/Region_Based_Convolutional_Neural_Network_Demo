# Kế hoạch làm lại hệ thống Hybrid giám sát lắp tai nghe

> Cập nhật geometry-only mới nhất (`open_case → tai 1 → tai 2 → close_case`) đã được chuyển vào [VIEC_BAN_CAN_LAM_GEOMETRY.md](VIEC_BAN_CAN_LAM_GEOMETRY.md). Tài liệu hiện tại vẫn được giữ nguyên để bảo toàn kế hoạch Hybrid/ViT–BiLSTM và các ghi chú dataset trước đó.

Cập nhật và đối chiếu với repository: **21/09/2026**.

Tài liệu này là kế hoạch thực hiện lại dự án từ baseline hiện có. Các số liệu trong phần “Hiện trạng” được đọc trực tiếp từ code, dataset và artifact đang có; các con số trong phần “Tiêu chí nghiệm thu” là **mục tiêu đề xuất**, chưa phải kết quả đã đạt.

## 1. Mục tiêu và phạm vi

Hệ thống quan sát một chu trình:

```text
đặt hộp mở nắp
→ lắp tai nghe thứ nhất
→ lắp tai nghe thứ hai
→ đóng nắp
```

Hệ thống cần trả lời ba câu hỏi khác nhau:

1. **Trong ảnh có gì và ở đâu?** — YOLO nhận diện hộp, tai nghe, khe trống, tay và trạng thái nắp.
2. **Người dùng đang làm gì?** — ViT tạo embedding từng frame, BiLSTM phân loại hành động theo cửa sổ thời gian.
3. **Thao tác có làm thay đổi trạng thái vật lý đúng như mong đợi không?** — Fusion Engine đối chiếu hành động với detection, sau đó FSM kiểm tra thứ tự.

Đầu ra mong muốn cho mỗi sự kiện:

- `PASS`: hoàn thành đúng một bước.
- `VIOLATION`: có bằng chứng đủ rõ về thao tác sai.
- `WAITING/UNCERTAIN`: chưa đủ bằng chứng; không tự suy diễn thành PASS hay lỗi.
- Log có timestamp, cycle ID, state trước/sau, dự đoán action, confidence, trạng thái vật thể và lý do quyết định.

Đây vẫn là prototype nghiên cứu. Bounding box 2D không chứng minh được tiếp xúc điện hoặc chất lượng sạc, vì vậy chưa được dùng như thiết bị kiểm định sản xuất.

## 2. Kiến trúc mục tiêu

```text
Camera/video
  ├─ YOLO
  │    ├─ Case_Open / Case_Closed
  │    ├─ Left_Earbud / Right_Earbud
  │    ├─ Empty_Slot_Left / Empty_Slot_Right
  │    └─ Hand
  │
  ├─ ViT encoder → embedding cache → BiLSTM → action + confidence
  │
  └─ Fusion Engine
       ├─ ổn định detection qua thời gian
       ├─ kiểm tra containment/occupancy/chuyển trạng thái
       ├─ kết hợp action evidence
       └─ phát event đã xác minh
              ↓
             FSM → PASS / VIOLATION / hướng dẫn bước kế tiếp
              ↓
       JSONL + ảnh/video bằng chứng + báo cáo metric
```

Nguyên tắc quan trọng: model chỉ tạo **bằng chứng**; Fusion và FSM mới quyết định một bước có hoàn tất hay vi phạm hay không.

## 3. Hiện trạng đã kiểm chứng

### 3.1. Code và runtime

| Hạng mục | Hiện trạng | Ghi chú |
|---|---|---|
| FSM cấu hình được | Có | `src/assembly/fsm.py` và `configs/earbud_two_step_fsm_config.json` |
| YOLO adapter | Có | Hỗ trợ closed-set YOLO và YOLO-World |
| ViT–BiLSTM | Có | Runtime cache embedding, không encode lại 16 frame cũ |
| Fusion hai tai nghe | Có | Kết hợp containment, số khe trống và action evidence |
| Project profile | Có | `configs/projects/earbud.json` |
| Runtime Hybrid | Có | `scripts/run_hybrid.py` |
| Log JSONL | Có nhưng thiếu chi tiết | Hiện chủ yếu ghi kết quả FSM; chưa ghi đầy đủ detection/fusion/confidence |
| Theo dõi ID qua frame | Chưa có | Chưa dùng ByteTrack hoặc tracker tương đương |
| Phát hiện lấy tai nghe ra | Có | Khi khe trống xuất hiện trở lại ổn định, Fusion phát violation và FSM lùi về trạng thái 1 tai hoặc 0 tai |
| Xác nhận nắp đã đóng bằng hình ảnh | Chưa có | Detector hiện chưa có `Case_Open/Case_Closed` |
| Test camera end-to-end tự động | Chưa có | Mới có unit test cho từng khối logic |

Hành vi thực tế của Fusion hiện tại:

- `pick_case` được phát khi hộp xuất hiện ổn định; không cần action model xác nhận `pick_case`.
- `insert_first_earbud` và `insert_second_earbud` cần đồng thời `insert_earbud` đủ confidence và occupancy tăng.
- `remove_earbud_to_one`/`remove_earbud_to_zero` không cần thêm class action: YOLO suy ra thao tác tháo khi occupancy giảm và `Empty_Slot` xuất hiện trở lại ổn định.
- Sau violation tháo tai, FSM lùi về state vật lý tương ứng; vì vậy đóng nắp ngay sau đó không thể PASS sai.
- `close_case` cần action model dự đoán đóng nắp, nhưng chưa kiểm tra được trạng thái nắp bằng YOLO.
- Đóng nắp sớm vẫn được chuyển tới FSM để FSM trả `VIOLATION`.
- `stable_frames=3` là ba lần Fusion nhận kết quả YOLO ổn định, không nhất thiết là ba frame camera liên tiếp khi `--yolo-every > 1`.

### 3.2. Detection dataset hiện tại

Dataset local: `datasets/earbud_merged`.

| Split | Ảnh | Label rỗng | Bounding box |
|---|---:|---:|---:|
| Train | 464 | 46 | 1.742 |
| Validation | 58 | 6 | 198 |
| Test | 59 | 5 | 231 |
| **Tổng** | **581** | **57** | **2.171** |

| ID | Class | Số box |
|---:|---|---:|
| 0 | `Earphone_Case` | 526 |
| 1 | `Earbud` | 144 |
| 2 | `Empty_Slot` | 336 |
| 3 | `Left_Earbud` | 358 |
| 4 | `Right_Earbud` | 350 |
| 5 | `Hand` | 457 |

Kiểm tra cấu trúc hiện tại:

- Thiếu label: 0.
- Dòng label sai định dạng: 0.
- Label rỗng/background: 57.
- Cấu trúc đủ để train, nhưng chưa chứng minh split độc lập theo video/session.

Checkpoint YOLO hiện có 60 epoch. Kết quả validation tốt nhất trong `results.csv` ở epoch 48:

| Precision | Recall | mAP50 | mAP50–95 |
|---:|---:|---:|---:|
| 0,578 | 0,565 | 0,582 | 0,493 |

Kết luận: detector hiện là baseline, chưa đủ ổn định cho một hệ thống kiểm lỗi. Cần xem metric theo từng class và đặc biệt ưu tiên recall của hộp, tai nghe, khe trống và trạng thái nắp.

### 3.3. Action dataset hiện tại

- Có 35 video gốc: 18 video ở `per1/session 01`, 17 video ở `per1/session 02`.
- Chỉ 17 video của `per1/session 02` đã có annotation và feature.
- `annotations.csv` có 66 segment.
- Train: 11 video, 42 segment.
- Validation: 2 video, 8 segment.
- Test: 4 video, 16 segment.
- Tất cả train/validation/test đều thuộc cùng một người và cùng session `per1/02`.

Nhãn hiện tại:

```text
idle
pick_case
insert_earbud
close_case
```

Checkpoint action model đạt validation Macro-F1 = 1,0 từ epoch 2; file evaluation hiện cũng chỉ báo cáo trên split `val` với 66 temporal window. Kết quả này **không được xem là kết quả tổng quát hóa**, vì:

- train/val/test cùng người, cùng ngày/session và cùng bối cảnh;
- validation chỉ gồm hai video;
- các temporal window từ cùng một video có độ tương quan rất cao;
- model có thể học nền, góc camera, tay hoặc nhịp thao tác thay vì học hành động.

`CachedActionWindowDataset` mặc định đã chặn person/session xuất hiện ở nhiều split. Model cũ chỉ có thể train trên split hiện tại nếu dùng `--allow-same-session`; không dùng tùy chọn này cho kết quả cuối.

### 3.4. Test và môi trường

- Repository hiện có 45 test method.
- Trong môi trường kiểm tra ngày 21/09/2026, 30 test nhẹ đã đạt.
- Chưa chạy lại được toàn bộ suite vì Python hiện tại thiếu `numpy`, `torch`, `PyYAML` và OpenCV.
- Các config và checkpoint mà profile earbud tham chiếu đều đang tồn tại.

Không ghi “45/45 đạt” cho đến khi chạy toàn bộ suite trong đúng virtual environment.

## 4. Những blocker phải xử lý trước khi train lại

### P0 — Split action bị rò rỉ theo person/session

Không tiếp tục dùng metric 1,0 làm kết quả báo cáo. Cần thu ít nhất ba group độc lập và chia **trước khi train**:

```text
train: person/session A, B, ...
val:   person/session chưa có trong train
test:  person/session chưa có trong train và val
```

Không random frame hoặc random segment vào các split khác nhau. Toàn bộ video của cùng một `person_id + session_id` phải nằm trong đúng một split.

### P0 — Chưa biết detection split có rò rỉ hay không

Nếu các frame gần nhau của cùng video xuất hiện ở cả train và test thì metric sẽ cao giả. Cần lưu `source_video`, `person_id`, `session_id`, `frame_index` trong manifest và chia theo source group trước khi augmentation.

### P0 — Chưa xác nhận vật lý cho bước đóng nắp

Thêm `Case_Open` và `Case_Closed`, hoặc một classifier trạng thái nắp riêng. Chỉ phát PASS cho `close_case` khi có cả:

```text
action = close_case
AND đủ hai tai nghe đã xác nhận
AND trạng thái ổn định chuyển Case_Open → Case_Closed
```

Đóng nắp sớm vẫn phải được phát thành event để FSM báo lỗi, nhưng cần log rằng lỗi được xác định từ action hay từ trạng thái nắp.

### P0 — Removal/rework đã có baseline

Hệ thống đã phát `remove_earbud_to_one` hoặc `remove_earbud_to_zero` khi khe trống xuất hiện trở lại ổn định. Kết quả là `VIOLATION` và FSM rollback về bước vật lý tương ứng.

Phần còn cần nâng cấp khi dataset có class `Hand` tin cậy: chỉ chốt removal sau khi tay rời vùng hộp. Baseline hiện tại dùng bằng chứng dương tính từ khe trống và `stable_frames`/`dwell_frames`, không coi việc mất bbox tai trong một frame là removal.

### P1 — Log chưa đủ để debug

Mỗi quyết định cần ghi thêm:

- `frame_index`, timestamp và thời gian monotonic;
- action label/confidence;
- danh sách detection label/confidence/box;
- số tai nghe trong hộp, số khe trống, occupancy;
- fusion event và reason;
- FSM state trước/sau;
- thời gian YOLO, ViT và tổng latency;
- đường dẫn ảnh/video bằng chứng khi có VIOLATION.

## 5. Quy trình làm lại dự án

### Giai đoạn 0 — Đóng băng baseline

Mục tiêu: giữ kết quả cũ để so sánh, không ghi đè artifact.

**Trạng thái 21/09/2026: ĐÃ HOÀN THÀNH.** Baseline nằm tại `artifacts/baselines/earbud_v1`; manifest, thông tin máy và checksum nằm tại `reports/earbud_v1`. Bộ config mới `configs/projects/earbud_v2.json` đã trỏ sang checkpoint/log v2 riêng.

Đã thực hiện:

1. [x] Ghi commit/hash code, ngày, cấu hình và máy chạy.
2. [x] Sao lưu `artifacts/action_model`, `artifacts/training/earbud_merged_detector` và `annotations.csv` dưới tên có version.
3. [x] Tạo config `earbud_v2` trỏ sang đường dẫn checkpoint/log mới thay vì ghi đè `best.pt` cũ.
4. [x] Lưu seed và thông tin môi trường hiện có; đánh dấu rõ các phiên bản của môi trường train cũ không thể truy xuất.

Đầu ra:

```text
artifacts/baselines/earbud_v1/
reports/earbud_v1/README.md
reports/earbud_v1/baseline_manifest.json
reports/earbud_v1/checksums.sha256
configs/projects/earbud_v2.json
```

Hoàn thành khi có thể xác định chính xác checkpoint nào được tạo bởi dataset/config nào.

### Giai đoạn 1 — Chốt SOP và schema nhãn

Viết một SOP ngắn, quan sát được bằng camera:

| Bước | Trạng thái trước | Hành động | Bằng chứng hoàn tất | Lỗi cần bắt |
|---|---|---|---|---|
| 1 | Chưa có hộp | Đặt hộp mở | `Case_Open` ổn định trong work zone | Hộp đóng, hộp ngoài zone |
| 2 | Hai khe trống | Lắp tai 1 | Occupancy 0→1 và action insert | Đặt gần khe, đặt ngoài hộp |
| 3 | Một khe trống | Lắp tai 2 | Occupancy 1→2 và action insert | Lắp sai/không vào khe |
| 4 | Đủ hai tai | Đóng nắp | `Case_Open→Case_Closed` và action close | Đóng sớm |
| Rework | Đã có tai | Lấy tai ra | Occupancy giảm ổn định | Removal không cho phép |

Quyết định trước khi gán nhãn:

- Có cần phân biệt tai trái/phải không?
- Có cho phép lắp tai phải trước tai trái không?
- Có cho phép tháo ra và lắp lại không?
- Một clip chứa cả chu trình hay một action?
- `idle` bao gồm những khoảng nào; khoảng chuyển tiếp khó xác định có dùng `uncertain` không?

Khuyến nghị class detection v2:

```text
Case_Open
Case_Closed
Left_Earbud
Right_Earbud
Empty_Slot_Left
Empty_Slot_Right
Hand
```

Không dùng đồng thời box `Earbud` generic và box trái/phải cho cùng một vật thể trong dataset v2.

### Giai đoạn 2 — Chuẩn hóa môi trường

Windows PowerShell:

```powershell
cd "E:\Professional documents\Internship\RBCNN_Demo"
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements-camera.txt
python -m pip install -r requirements-ml.txt
python -m pip install -e .
python -m unittest discover -s tests -v
```

Nếu dùng NVIDIA GPU, cài đúng PyTorch/CUDA theo máy trước rồi kiểm tra:

```powershell
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

Gate hoàn thành:

- Import được `cv2`, `torch`, `transformers`, `ultralytics`, `yaml`.
- Toàn bộ unit test đạt.
- Camera mở được và ghi thử một video không bị tua nhanh.

### Giai đoạn 3 — Thu dữ liệu có thiết kế

#### 3.1. Ma trận dữ liệu tối thiểu đề xuất

- Ít nhất 3 người; nếu chỉ có một người thì dùng ít nhất 3 ngày/session với thay đổi có kiểm soát.
- Mỗi group quay 15–20 chu trình đúng.
- Mỗi lỗi quan trọng có ít nhất 10–15 clip, phân bố qua nhiều group.
- Có thay đổi ánh sáng, tay áo/găng, vị trí hộp và góc xoay; không thay đổi tất cả yếu tố cùng lúc.

Các scenario bắt buộc:

| Scenario | Kết quả mong đợi |
|---|---|
| `correct` | Hoàn tất đủ bốn bước |
| `close_empty` | Báo đóng khi chưa có tai |
| `close_one_earbud` | Báo đóng khi mới có một tai |
| `near_slot_not_inserted` | Không PASS insertion |
| `outside_case` | Không PASS insertion |
| `remove_after_insert` | Báo removal hoặc vào rework |
| `hand_occlusion` | Không tạo event giả khi che tạm thời |
| `case_removed` | Reset/abort chu trình theo SOP |
| `wrong_order` | Báo lỗi nếu SOP cấm thứ tự đó |

Quay action video:

```powershell
python scripts/record_assembly_videos.py `
  --project earbud `
  --person person02 `
  --session session01 `
  --scenario correct `
  --source 0
```

Mỗi clip phải được xem lại ngay: đúng FPS, không mất đầu/cuối thao tác, nhìn rõ hộp và hai khe.

#### 3.2. Chia split trước khi trích frame/feature

Tạo bảng group cố định, ví dụ:

| Group | Split |
|---|---|
| person01/session01 | train |
| person02/session01 | train |
| person03/session01 | validation |
| person04/session01 | test |

Nếu chưa có bốn người, dùng session/ngày độc lập nhưng phải ghi rõ hạn chế. Không dùng `--allow-same-session` cho model báo cáo cuối.

Script `split_annotations.py` hiện chia theo video, chưa đảm bảo group split. Với dataset v2, gán split theo manifest person/session hoặc nâng cấp script trước khi dùng.

### Giai đoạn 4 — Làm lại detection dataset

Quy trình:

1. Chốt class list và guideline bbox bằng hình minh họa.
2. Chia source video/session thành train/val/test.
3. Sau đó mới lấy frame từ từng split.
4. Loại frame gần như trùng nhau; không lấy hàng chục frame liên tiếp của cùng cảnh.
5. Gán nhãn, review chéo và sửa toàn bộ missing/wrong class/bbox quá rộng.
6. Xác nhận 57 label rỗng cũ; ảnh nào có vật thể mục tiêu thì không được để rỗng.
7. Tạo `data.yaml` portable, không chứa đường dẫn tuyệt đối của một máy.

Kiểm tra:

```powershell
python scripts/validate_detection_dataset.py `
  --data datasets/earbud_v2/data.yaml
```

Train baseline:

```powershell
python scripts/train_detector.py `
  --data datasets/earbud_v2/data.yaml `
  --model yolo26n.pt `
  --epochs 100 `
  --image-size 640 `
  --batch 8 `
  --device 0 `
  --name earbud_v2_detector
```

Phải báo cáo:

- Precision, recall, AP50 và AP50–95 từng class.
- Confusion matrix.
- Ảnh false positive/false negative.
- Kết quả riêng cho che khuất, ánh sáng yếu, hộp xoay và vật thể nhỏ.

Không chọn model chỉ theo mAP trung bình. Với class dùng làm điều kiện bắt buộc, recall thấp sẽ làm hệ thống bỏ sót bước.

### Giai đoạn 5 — Làm lại action dataset và BiLSTM

Gán nhãn video:

```powershell
python scripts/annotate_actions.py `
  --videos-dir data/earbud_actions_v2/raw_videos `
  --output data/earbud_actions_v2/annotations.csv
```

Quy tắc annotation:

- Ranh giới action dựa trên chuyển động quan sát được, không dựa trên kết quả YOLO.
- Không để hai segment khác nhãn chồng thời gian nếu code chưa hỗ trợ multi-label.
- Khoảng mơ hồ phải được xử lý nhất quán: loại khỏi annotation, hoặc thêm nhãn `uncertain` vào config; không tự ghi `uncertain` khi config vẫn chỉ có bốn nhãn hiện tại.
- Review ít nhất 10% clip bởi người thứ hai.
- Kiểm tra mọi `video_id`, `person_id`, `session_id`, `scenario` và `split`.

Trích feature:

```powershell
python scripts/extract_spatial_features.py `
  --videos-dir data/earbud_actions_v2/raw_videos `
  --output-dir data/earbud_actions_v2/features `
  --config configs/action_earbud_config.json
```

Kiểm tra pipeline:

```powershell
python scripts/verify_pipeline.py `
  --annotations data/earbud_actions_v2/annotations.csv `
  --features-dir data/earbud_actions_v2/features `
  --config configs/action_earbud_config.json
```

Ngoài validator hiện có, phải xác nhận không có `(person_id, session_id)` ở nhiều split.

Train:

```powershell
python scripts/train_action_model.py `
  --annotations data/earbud_actions_v2/annotations.csv `
  --features-dir data/earbud_actions_v2/features `
  --config configs/action_earbud_config.json `
  --output artifacts/action_model_v2 `
  --device cuda
```

Đánh giá cả validation và test:

```powershell
python scripts/evaluate_action_model.py `
  --checkpoint artifacts/action_model_v2/best.pt `
  --annotations data/earbud_actions_v2/annotations.csv `
  --features-dir data/earbud_actions_v2/features `
  --config configs/action_earbud_config.json `
  --split test `
  --output artifacts/action_model_v2/evaluation_test.json
```

Không điều chỉnh threshold hoặc kiến trúc dựa trên test. Test chỉ chạy sau khi đã khóa lựa chọn bằng validation.

### Giai đoạn 6 — Nâng cấp Fusion và FSM

Thứ tự triển khai:

1. Dùng `Case_Open/Case_Closed` cho bước đặt hộp và đóng nắp.
2. Giữ occupancy ổn định theo thời gian; phân biệt mất detection với removal thật.
3. Kiểm thử và tinh chỉnh baseline `remove_earbud_to_one`/`remove_earbud_to_zero`; bổ sung điều kiện `Hand` rời khỏi hộp nếu dataset v2 giữ class tay.
4. Thêm tracker ID nếu detection trái/phải vẫn nhảy nhãn qua frame.
5. Log toàn bộ evidence thay vì chỉ log outcome FSM.
6. Viết replay runner để chạy lại video cố định mà không cần camera trực tiếp.
7. Tạo `configs/projects/earbud_v2.json` trỏ tới config và checkpoint v2; giữ profile `earbud.json` để tái hiện baseline cũ.

Các unit/integration test bắt buộc:

- Geometry đứng yên không được tự tạo insertion.
- Đưa tai nghe gần khe nhưng ngoài hộp không được PASS.
- Action insert nhưng occupancy không tăng không được PASS.
- Occupancy tăng nhưng không có action evidence không được PASS.
- Đóng nắp khi 0/1 tai phải là VIOLATION.
- Đóng nắp khi đủ hai tai nhưng chưa thấy `Case_Closed` chưa được PASS.
- Che khuất ngắn không rollback.
- Removal ổn định phải báo lỗi hoặc vào rework.
- Reset phải xóa buffer, latch và cycle state.

### Giai đoạn 7 — Đánh giá end-to-end

Tạo manifest video test độc lập với ground truth event. Với mỗi clip, lưu:

```text
video_id
scenario
ground-truth event sequence
predicted event sequence
false PASS
missed violation
false violation
delay của từng event
```

Metric cần báo cáo:

- Completion accuracy toàn chu trình.
- Precision/recall/F1 cho từng event.
- Violation recall theo từng scenario lỗi.
- False PASS rate.
- False alarm trên mỗi chu trình hoặc mỗi giờ.
- Độ trễ trung vị và P95 của event.
- FPS và thời gian YOLO/ViT–LSTM trên đúng máy demo.

Mục tiêu nghiệm thu đề xuất cho prototype v2:

| Chỉ số | Mục tiêu |
|---|---:|
| Completion accuracy trên test độc lập | ≥ 90% |
| Recall lỗi đóng nắp sớm | ≥ 95% |
| Recall các violation quan trọng | ≥ 90% |
| False PASS | ≤ 5% chu trình |
| False violation | ≤ 0,1/chu trình |
| Event delay P95 | ≤ 1,5 giây |

Các ngưỡng này chỉ có ý nghĩa khi test độc lập theo person/session và có đủ số clip lỗi.

### Giai đoạn 8 — Chạy demo và viết báo cáo

Chạy camera GPU:

```powershell
python scripts/run_hybrid.py `
  --project configs/projects/earbud_v2.json `
  --source 0 `
  --device 0 `
  --sample-fps 10 `
  --yolo-every 1
```

Chạy CPU để kiểm tra chức năng:

```powershell
python scripts/run_hybrid.py `
  --project configs/projects/earbud_v2.json `
  --source 0 `
  --device cpu `
  --sample-fps 3 `
  --yolo-every 4
```

Giảm `sample-fps` làm thay đổi độ dài thời gian thực của cửa sổ 16 embedding: 1,6 giây ở 10 FPS nhưng khoảng 5,3 giây ở 3 FPS. Nếu deploy CPU ở FPS thấp, nên train/evaluate lại với sampling tương ứng.

Nếu Hugging Face cache chưa có backbone, chạy một lần khi có mạng:

```powershell
python scripts/run_hybrid.py --project configs/projects/earbud_v2.json --source 0 --allow-download
```

Sau khi cache xong, bỏ `--allow-download`; inference camera chạy local.

## 6. Thứ tự công việc thực tế

Không bắt đầu bằng việc train lại ngay. Thứ tự nên là:

1. Chốt SOP và class/action schema.
2. Dựng môi trường sạch, chạy đủ test.
3. Sửa group split và manifest dữ liệu.
4. Quay thêm người/session và scenario lỗi.
5. Làm sạch detection dataset, train và phân tích lỗi.
6. Gán action, trích feature, train và test độc lập.
7. Bổ sung trạng thái nắp, removal/rework và logging.
8. Chạy replay end-to-end trước khi chạy camera live.
9. Khóa threshold bằng validation.
10. Chạy test đúng một lần để lập báo cáo cuối.

## 7. Checklist bàn giao

### Dữ liệu

- [ ] Có manifest nguồn và quy tắc đặt tên.
- [ ] Không có person/session hoặc source video ở nhiều split.
- [ ] Label rỗng đã được kiểm tra thủ công.
- [ ] Có đủ correct và violation scenario.
- [ ] Có data card ghi nguồn, giấy phép, cách split và giới hạn.

### Model

- [ ] Báo cáo YOLO theo từng class trên test độc lập.
- [ ] Báo cáo action confusion matrix và macro-F1 trên test độc lập.
- [ ] Checkpoint đi kèm config, class order và phiên bản thư viện.
- [ ] Không dùng kết quả `--allow-same-session` làm metric cuối.

### Logic hệ thống

- [ ] Insertion cần action và thay đổi occupancy.
- [ ] Close cần action và `Case_Open→Case_Closed`.
- [ ] Removal/rework đã có quy tắc rõ ràng.
- [ ] Che khuất ngắn không tạo event giả.
- [ ] Event log có đủ evidence để điều tra lỗi.

### Kiểm thử và demo

- [ ] Toàn bộ unit test đạt trong environment đã khóa version.
- [ ] Replay test đạt trước camera live.
- [ ] Có ít nhất 10 chu trình đúng liên tiếp không false alarm.
- [ ] Mỗi scenario lỗi quan trọng được demo ít nhất 5 lần.
- [ ] Báo cáo FPS, latency, phần cứng và giới hạn sử dụng.

## 8. Việc nên làm ngay trong vòng tiếp theo

1. Không dùng Macro-F1 = 1,0 hiện tại làm kết luận.
2. Chọn thêm ít nhất hai person/session độc lập và lập bảng split trước khi quay.
3. Quyết định schema detection v2 có `Case_Open/Case_Closed` và khe trái/phải.
4. Kiểm tra thủ công 57 ảnh label rỗng và nguồn của từng split detection.
5. Chạy toàn bộ test trong `.venv`; lưu kết quả và phiên bản môi trường.
6. Quay pilot nhỏ: mỗi group 3 correct + 1 clip cho từng lỗi chính.
7. Review pilot trước khi quay hàng loạt.
8. Chỉ train v2 sau khi manifest, split và guideline đã được khóa.

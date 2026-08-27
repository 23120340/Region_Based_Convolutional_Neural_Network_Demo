# Hướng dẫn từng bước thu thập dữ liệu cho hệ thống giám sát lắp ráp bút bi

## 1. Mục tiêu

Tài liệu này hướng dẫn chuẩn bị dữ liệu để hệ thống có thể:

1. Tự phát hiện và khoanh vùng từng linh kiện bút trên camera.
2. Tự nhận diện hành động người dùng đang thực hiện.
3. Tự kiểm tra hành động có đúng thứ tự lắp ráp hay không.
4. Cảnh báo các lỗi như quên ruột, quên lò xo, vặn nắp quá sớm hoặc bấm thử khi chưa hoàn thành.

Hệ thống gồm ba phần khác nhau:

```text
Ảnh có bounding box
        ↓
Object Detector: linh kiện nào đang ở đâu?
        ↓
Video có nhãn thời gian
        ↓
ViT + LSTM: người dùng đang thực hiện hành động nào?
        ↓
FSM: hành động có đúng thứ tự quy trình không?
```

Vì vậy cần thu hai loại dữ liệu riêng:

- **Dataset ảnh detection**: dùng để nhận diện thân, ruột, lò xo, nắp và bút hoàn chỉnh.
- **Dataset video action**: dùng để nhận diện các hành động lắp ráp theo thời gian.

---

## 2. Danh sách dụng cụ cần chuẩn bị

### 2.1. Dụng cụ bắt buộc

- Một laptop có camera hoặc webcam USB.
- Tối thiểu 2–3 cây bút bi bấm cùng loại.
- Một mặt bàn tương đối phẳng.
- Một tờ giấy A3/A4 lớn hoặc tấm nền một màu.
- Băng dính màu để đánh dấu các vùng linh kiện.
- Đèn bàn nếu khu vực quay thiếu sáng.

### 2.2. Dụng cụ nên có

- Tripod hoặc giá đỡ camera.
- Một đèn đặt bên trái và một đèn đặt bên phải để giảm bóng tay.
- Các khay nhỏ đựng riêng từng linh kiện.
- Một hoặc hai loại áo tay dài/ngắn để tạo khác biệt khi quay.

### 2.3. Quy định về loại bút

Trong đợt dữ liệu đầu tiên chỉ sử dụng **một mẫu bút cố định**. Không trộn nhiều loại bút có cấu trúc hoặc màu sắc quá khác nhau.

Ghi thông tin mẫu bút vào nhật ký:

```text
Tên/nhãn hiệu:
Màu thân:
Chiều dài:
Số lượng bút:
Ngày bắt đầu thu dữ liệu:
```

---

## 3. Bố trí camera và bàn lắp ráp

### 3.1. Bố trí các vùng

Đánh dấu năm vùng cố định:

```text
+----------------------------------------------------------------+
| [Khay thân] [Khay ruột] [Khay lò xo] [Khay nắp]                |
|                                                                |
|                 +--------------------------+                   |
|                 |      VÙNG LẮP RÁP        |                   |
|                 |       WORK ZONE          |                   |
|                 +--------------------------+                   |
+----------------------------------------------------------------+
```

Yêu cầu:

- Các khay không được chồng lấn nhau.
- WORK ZONE phải đủ rộng để hai tay thao tác.
- Khay linh kiện nên nằm ngoài WORK ZONE.
- Nền không nên có hoa văn giống ruột bút hoặc lò xo.
- Có thể dùng băng dính màu vàng để đánh dấu WORK ZONE.

### 3.2. Vị trí camera

Ưu tiên theo thứ tự:

1. Camera top-down nhìn thẳng từ trên xuống.
2. Camera nghiêng khoảng 45 độ từ phía trước.
3. Camera laptop nhìn nghiêng xuống bàn.

Không được thay đổi vị trí camera trong cùng một buổi quay. Nếu camera bị dịch chuyển, coi đó là một buổi quay mới và ghi lại trong nhật ký.

### 3.3. Cấu hình hình ảnh đề xuất

- Độ phân giải: `1280 × 720` hoặc `1920 × 1080`.
- Tốc độ: `30 FPS`.
- Linh kiện phải đủ sáng và không bị cháy sáng.
- Camera phải lấy nét được ruột bút và lò xo.
- Hạn chế để bóng tay che toàn bộ WORK ZONE.

### 3.4. Kiểm tra trước khi quay

- [ ] Camera nhìn thấy đủ bốn khay và WORK ZONE.
- [ ] Lò xo nhìn thấy rõ khi đặt trong WORK ZONE.
- [ ] Ruột bút không bị trùng màu với nền.
- [ ] Không có vật dụng không liên quan nằm trên bàn.
- [ ] Camera không rung khi người dùng thao tác.
- [ ] Ánh sáng không thay đổi liên tục.

---

## 4. Kiểm tra camera zero-shot trước khi thu dataset

Mở PowerShell tại thư mục dự án:

```powershell
cd "E:\Professional documents\Internship\RBCNN_Demo"
python scripts/run_camera.py --source 0
```

Nếu không mở đúng camera, thử:

```powershell
python scripts/run_camera.py --source 1
```

Lần lượt đưa từng vật thể vào giữa camera:

1. Thân bút.
2. Ruột bút.
3. Lò xo.
4. Nắp bút.
5. Bút đã lắp hoàn chỉnh.

Với mỗi vật thể, thử ít nhất:

- 5 góc xoay khác nhau.
- 3 vị trí khác nhau trong khung hình.
- 2 khoảng cách camera khác nhau.
- Tay che một phần vật thể.

Ghi kết quả vào bảng:

| Linh kiện | Nhận đúng | Nhận sai | Không nhận ra | Ghi chú |
|---|---:|---:|---:|---|
| `barrel` | | | | |
| `refill` | | | | |
| `spring` | | | | |
| `cap` | | | | |
| `assembled_pen` | | | | |

Nếu zero-shot nhận sai nhiều, vẫn tiếp tục thu dataset. Không cần cố điều chỉnh model quá lâu trước khi có dữ liệu thật.

---

## 5. Thu dataset ảnh để nhận diện linh kiện

### 5.1. Các lớp cần gán nhãn

Sử dụng đúng năm tên lớp sau:

| Class ID | Tên lớp | Nội dung |
|---:|---|---|
| 0 | `barrel` | Thân dưới của bút |
| 1 | `refill` | Ruột/mực bút |
| 2 | `spring` | Lò xo |
| 3 | `cap` | Nắp hoặc thân trên |
| 4 | `assembled_pen` | Cây bút đã lắp hoàn chỉnh |

Không tự ý đổi tên lớp giữa các buổi gán nhãn.

### 5.2. Mở công cụ thu ảnh

```powershell
python scripts/capture_detection_images.py --camera 0
```

Điều khiển:

- `Space`: lưu ảnh hiện tại.
- `Q` hoặc `Esc`: kết thúc.

Ảnh được lưu mặc định tại:

```text
datasets/pen_parts/raw/
```

### 5.3. Số lượng ảnh đề xuất

Mục tiêu vòng đầu là 200–400 ảnh có chất lượng.

| Nhóm ảnh | Số lượng đề xuất |
|---|---:|
| Mỗi linh kiện nằm riêng | 50–80 ảnh |
| Nhiều linh kiện xuất hiện cùng lúc | 60–100 ảnh |
| Tay đang cầm hoặc che một phần | 80–120 ảnh |
| Bút hoàn chỉnh | 50–80 ảnh |
| Ảnh âm tính không có linh kiện | 50–100 ảnh |

Một ảnh có thể chứa nhiều linh kiện, nên tổng số instance sẽ lớn hơn tổng số ảnh.

Ưu tiên thu nhiều dữ liệu cho `spring` và `refill` vì hai vật thể này nhỏ và dễ bị bỏ sót.

### 5.4. Danh sách góc chụp bắt buộc

Với mỗi loại linh kiện:

- [ ] Nằm ngang.
- [ ] Nằm dọc.
- [ ] Nằm chéo.
- [ ] Ở giữa khung hình.
- [ ] Gần cạnh trái/phải.
- [ ] Nằm trong khay.
- [ ] Nằm trong WORK ZONE.
- [ ] Được cầm trên tay.
- [ ] Bị một ngón tay che một phần.
- [ ] Nằm cạnh các linh kiện khác.
- [ ] Có bóng tay hoặc bóng linh kiện.

### 5.5. Ảnh âm tính cần thu

Ảnh âm tính giúp giảm nhận diện nhầm. Cần chụp:

- Bàn trống.
- Chỉ có bàn tay.
- Bút chì hoặc que nhỏ không phải ruột bút.
- Kẹp giấy hoặc vật kim loại không phải lò xo.
- Điện thoại, kéo, tua vít hoặc vật dụng văn phòng.
- Khay rỗng.

Ảnh âm tính không có file nhãn hoặc có file nhãn rỗng, tùy công cụ xuất dữ liệu.

### 5.6. Những ảnh phải loại bỏ

- Ảnh bị rung hoặc nhòe quá mức.
- Ảnh quá tối hoặc cháy sáng.
- Lò xo nhỏ tới mức gần như không nhìn thấy.
- Camera bị che gần hết.
- Hai ảnh hoàn toàn giống nhau.
- Ảnh có thông tin cá nhân không cần thiết.

---

## 6. Gán bounding box cho dataset detection

### 6.1. Công cụ

Có thể sử dụng:

- CVAT.
- Label Studio.
- Công cụ annotation khác có thể xuất YOLO detection format.

### 6.2. Quy tắc khoanh vùng

1. Bounding box phải ôm sát vật thể.
2. Không khoanh cả bàn tay vào box linh kiện.
3. Mọi linh kiện thuộc năm lớp xuất hiện rõ trong ảnh đều phải được gán.
4. Vật thể bị tay che nhưng vẫn nhận biết được vẫn phải gán nhãn.
5. Nếu vật thể chỉ còn một phần quá nhỏ và không thể xác định chắc chắn thì bỏ qua.
6. Không gán `assembled_pen` cho một cụm bút chưa hoàn chỉnh.
7. Không gán đồng thời `assembled_pen` và các bộ phận đã nằm bên trong mà không còn nhìn thấy.

### 6.3. Ví dụ YOLO label

Mỗi dòng có định dạng:

```text
class_id x_center y_center width height
```

Ví dụ:

```text
0 0.420312 0.618750 0.251562 0.093056
1 0.603125 0.441667 0.178125 0.036111
```

Các tọa độ phải được chuẩn hóa về khoảng từ 0 đến 1.

### 6.4. Kiểm tra chất lượng annotation

Sau mỗi 50 ảnh:

- [ ] Kiểm tra có bỏ sót linh kiện không.
- [ ] Kiểm tra tên class có đúng không.
- [ ] Kiểm tra box có ôm sát vật thể không.
- [ ] Kiểm tra lò xo có bị gán nhầm thành vật kim loại khác không.
- [ ] Kiểm tra bút hoàn chỉnh có bị gán thành `barrel` không.

Nên để một người gán nhãn và một người khác kiểm tra lại ít nhất 10–20% số ảnh.

---

## 7. Chia train, validation và test

### 7.1. Tỷ lệ đề xuất

```text
Train:      70%
Validation: 15%
Test:       15%
```

### 7.2. Không chia ngẫu nhiên các frame liền nhau

Không được lấy các frame gần giống nhau từ cùng một đoạn quay rồi chia vào cả train và test. Điều này làm kết quả đánh giá cao giả tạo.

Nên chia theo buổi quay:

```text
Buổi 1, 2, 3, 4  -> train
Buổi 5           -> validation
Buổi 6           -> test
```

Nếu có nhiều người:

```text
Người A, B, C -> train
Người D       -> validation
Người E       -> test
```

### 7.3. Cấu trúc thư mục sau khi chia

```text
datasets/pen_parts/
├── images/
│   ├── train/
│   ├── val/
│   └── test/
├── labels/
│   ├── train/
│   ├── val/
│   └── test/
├── raw/
└── data.yaml
```

Tên file ảnh và file label phải giống nhau:

```text
images/train/pen_0001.jpg
labels/train/pen_0001.txt
```

---

## 8. Train và kiểm tra detector

### 8.1. Train

```powershell
python scripts/train_detector.py `
  --data datasets/pen_parts/data.yaml `
  --epochs 60
```

Checkpoint tốt nhất dự kiến nằm tại:

```text
artifacts/training/pen_parts_detector/weights/best.pt
```

### 8.2. Chạy checkpoint đã train

```powershell
python scripts/run_camera.py `
  --source 0 `
  --model artifacts/training/pen_parts_detector/weights/best.pt
```

### 8.3. Tiêu chí chấp nhận detector đề xuất

- [ ] Recall mỗi lớp đạt khoảng 90% trở lên trên test set.
- [ ] Lò xo được nhận diện ổn định trong WORK ZONE.
- [ ] Không nhận bàn tay thành linh kiện.
- [ ] Không nhận bút chì thành ruột bút quá thường xuyên.
- [ ] Bounding box không nhảy liên tục giữa các lớp.
- [ ] Camera chạy đủ nhanh cho thao tác thực tế.

Nếu một lớp có kết quả thấp, thu thêm dữ liệu có chủ đích cho chính trường hợp model đang sai, sau đó train lại.

---

## 9. Thu dataset video nhận diện hành động

Detector chỉ trả lời vật thể nào đang xuất hiện. Để hệ thống biết người dùng **đang cắm**, **đang vặn** hay **đang bấm thử**, cần thêm video action.

### 9.1. Bộ nhãn hành động

| Action ID | Action name | Nội dung |
|---:|---|---|
| 0 | `idle` | Chưa thao tác hoặc tay nghỉ |
| 1 | `pick_barrel` | Đưa thân bút vào WORK ZONE |
| 2 | `insert_refill` | Đưa ruột vào thân bút |
| 3 | `insert_spring` | Lắp lò xo vào đầu ruột |
| 4 | `screw_cap` | Vặn nắp/thân trên |
| 5 | `test_click` | Bấm thử cơ cấu ngòi |

### 9.2. Số lượng video đề xuất

| Kịch bản | Số lượt tối thiểu |
|---|---:|
| Chu trình đúng | 60–100 lượt |
| Quên ruột | 15–30 lượt |
| Quên lò xo | 15–30 lượt |
| Lắp lò xo trước ruột | 15–30 lượt |
| Vặn nắp quá sớm | 15–30 lượt |
| Bấm thử quá sớm | 15–30 lượt |

Nên có 3–5 người thực hiện. Nếu chưa đủ người, bắt đầu với 2 người nhưng phải thay đổi tốc độ, hướng tay và áo.

### 9.3. Cách quay chu trình đúng

Thực hiện đúng thứ tự:

```text
idle
  -> pick_barrel
  -> insert_refill
  -> insert_spring
  -> screw_cap
  -> test_click
  -> idle
```

Mỗi người thực hiện:

- 10 lượt tốc độ bình thường.
- 5 lượt chậm.
- 5 lượt nhanh.
- Một số lượt dừng ngắn giữa các bước.

Không diễn quá máy móc; dữ liệu cần giống thao tác thực tế.

### 9.4. Cách quay tình huống sai

Mỗi video lỗi chỉ nên có một lỗi chính rõ ràng.

Ví dụ quên lò xo:

```text
pick_barrel -> insert_refill -> screw_cap
```

Ví dụ sai thứ tự:

```text
pick_barrel -> insert_spring
```

Ví dụ bấm thử quá sớm:

```text
pick_barrel -> insert_refill -> insert_spring -> test_click
```

Không làm hỏng linh kiện hoặc cố tạo thao tác nguy hiểm.

---

## 10. Gán nhãn thời gian cho video action

Mỗi video cần xác định frame bắt đầu và kết thúc của từng hành động.

Định dạng đề xuất:

```csv
video_id,start_frame,end_frame,action_name,is_anomaly
video_001,0,29,idle,0
video_001,30,62,pick_barrel,0
video_001,63,118,insert_refill,0
video_001,119,166,insert_spring,0
video_001,167,245,screw_cap,0
video_001,246,291,test_click,0
```

Đối với video lỗi:

```csv
video_id,start_frame,end_frame,action_name,is_anomaly
video_error_001,0,30,idle,0
video_error_001,31,66,pick_barrel,0
video_error_001,67,121,insert_refill,0
video_error_001,122,193,screw_cap,1
```

Quy tắc:

1. Nhãn bắt đầu khi tay bắt đầu thực hiện mục đích của hành động.
2. Nhãn kết thúc khi vật thể đã được đặt/lắp xong hoặc tay rời khỏi thao tác.
3. Khoảng chuyển tiếp không rõ có thể gán `idle` hoặc bỏ khỏi tập train, nhưng phải dùng một quy tắc thống nhất.
4. Không gán toàn bộ video bằng một action duy nhất.

---

## 11. Train action model ViT + LSTM

Chỉ bắt đầu bước này khi video và annotation đã được kiểm tra.

Pipeline dự kiến:

```text
Video
  -> lấy mẫu 8–10 FPS
  -> ViT trích embedding từng frame
  -> cửa sổ 16 frame
  -> BiLSTM
  -> xác suất 6 hành động
  -> temporal debouncer
  -> FSM
```

Tiêu chí đề xuất:

- Macro-F1 trên test set đạt ít nhất `0.85`.
- Không chỉ báo cáo accuracy vì lớp `idle` có thể chiếm nhiều dữ liệu.
- Kiểm tra riêng recall của `insert_spring`, `screw_cap` và `test_click`.
- Test set phải chứa người hoặc buổi quay không có trong train.

Các script trích ViT feature và train LSTM sẽ được hoàn thiện sau khi dataset video thật đã sẵn sàng.

---

## 12. Điều kiện để hệ thống tự động xác nhận một bước

Không nên cho FSM chuyển bước chỉ vì detector thấy một linh kiện.

Một bước chỉ được tự động chấp nhận khi:

```text
Action model dự đoán đúng hành động
                  +
Detector thấy đúng linh kiện trong đúng vùng
                  +
Dự đoán ổn định tối thiểu 3/5 cửa sổ
                  +
Hành động hợp lệ với state hiện tại
                  =
FSM xác nhận hoàn thành bước
```

Ví dụ `insert_spring`:

- State hiện tại phải là `S2_REFILL_INSERTED`.
- Detector phải thấy `spring` đi vào WORK ZONE.
- Action model phải dự đoán `insert_spring` ổn định.
- Sau khi cả hai điều kiện đồng ý, FSM mới chuyển sang `S3_SPRING_INSERTED`.

Riêng `test_click` cần action model nhận diện chuyển động bấm; bounding box của bút không đủ để xác nhận bước này.

---

## 13. Kịch bản kiểm thử end-to-end

Sau khi tích hợp detector và action model, thực hiện:

| Kịch bản | Số lần | Kết quả mong đợi |
|---|---:|---|
| Lắp đúng đầy đủ | 30 | Hoàn thành không cảnh báo |
| Quên ruột | 10 | Cảnh báo và không chuyển state |
| Quên lò xo | 10 | Cảnh báo và không chuyển state |
| Lắp sai thứ tự | 10 | Cảnh báo sai thứ tự |
| Bấm thử quá sớm | 10 | Cảnh báo chưa vặn nắp |
| Tay che một phần | 10 | Không chuyển nhầm bước |
| Thao tác chậm | 10 | Vẫn nhận đúng |
| Thao tác nhanh | 10 | Không bỏ bước |

Ghi lại:

- Số lần nhận đúng.
- Số cảnh báo đúng.
- Số cảnh báo sai.
- Số bước bị bỏ sót.
- Độ trễ từ thao tác tới cảnh báo.

---

## 14. Thứ tự công việc cần thực hiện

### Giai đoạn 1 — Có thể làm ngay

- [ ] Chọn một loại bút cố định.
- [ ] Bố trí bốn khay và WORK ZONE.
- [ ] Cố định camera và ánh sáng.
- [ ] Chạy camera zero-shot.
- [ ] Ghi lại các trường hợp nhận đúng và sai.

### Giai đoạn 2 — Dataset detector

- [ ] Thu 200–400 ảnh.
- [ ] Thu đủ ảnh `spring` và `refill`.
- [ ] Thu ảnh âm tính.
- [ ] Gán bounding box năm lớp.
- [ ] Kiểm tra lại ít nhất 10–20% annotation.
- [ ] Chia train/val/test theo buổi quay.
- [ ] Train detector.
- [ ] Test detector trực tiếp trên camera.

### Giai đoạn 3 — Dataset hành động

- [ ] Quay 60–100 chu trình đúng.
- [ ] Quay từng loại lỗi có chủ đích.
- [ ] Có nhiều người và nhiều tốc độ thao tác.
- [ ] Gán nhãn start/end frame cho từng action.
- [ ] Chia dữ liệu theo người hoặc buổi quay.

### Giai đoạn 4 — Tự động hóa hoàn toàn

- [ ] Trích ViT embedding.
- [ ] Train LSTM.
- [ ] Đánh giá Macro-F1 và confusion matrix.
- [ ] Ghép detector + action model + debouncer + FSM.
- [ ] Chạy bộ kiểm thử end-to-end.
- [ ] Chỉ bỏ phím Space khi đạt tiêu chí nghiệm thu.

---

## 15. Dữ liệu cần bàn giao để tiếp tục phát triển model

Khi hoàn thành thu thập, cần chuẩn bị:

```text
datasets/
├── pen_parts/
│   ├── images/
│   ├── labels/
│   └── data.yaml
└── pen_actions/
    ├── raw_videos/
    ├── annotations.csv
    └── recording_log.csv
```

`recording_log.csv` nên có:

```csv
session_id,video_id,person_id,pen_type,camera_angle,lighting,scenario,notes
session_01,video_001,person_A,pen_01,45_degree,normal,correct,
session_01,video_002,person_A,pen_01,45_degree,normal,missing_spring,
```

Không cần đưa checkpoint ngẫu nhiên hoặc video chưa gán nhãn vào bước huấn luyện chính thức.

---

## 16. Công việc nên bắt đầu đầu tiên

Trong buổi làm việc đầu tiên chỉ cần hoàn thành:

1. Chuẩn bị 2–3 cây bút cùng loại.
2. Đánh dấu các khay và WORK ZONE.
3. Chạy `python scripts/run_camera.py --source 0` để kiểm tra góc camera.
4. Chạy `python scripts/capture_detection_images.py --camera 0`.
5. Thu khoảng 50 ảnh thử nghiệm gồm đủ năm lớp và ảnh bàn trống.
6. Kiểm tra ảnh có rõ lò xo và ruột bút không.
7. Nếu chất lượng đạt, tiếp tục thu đủ 200–400 ảnh.

Không nên quay toàn bộ dataset trước khi kiểm tra 50 ảnh đầu tiên, vì một góc camera hoặc ánh sáng không phù hợp sẽ làm hỏng toàn bộ dữ liệu còn lại.

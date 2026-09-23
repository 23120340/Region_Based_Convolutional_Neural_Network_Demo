# Huấn luyện ViT–BiLSTM v2: gán nhãn riêng từng lần lắp tai nghe

Cập nhật: 22/09/2026. Làm lần lượt đúng **5 bước** dưới đây: Annotation → chia tập → trích xuất ViT → train BiLSTM → đánh giá.

Hướng dẫn này chỉ dùng video và mốc thời gian hành động trong `data/earbud_actions/`; dataset ảnh bounding box của YOLO là pipeline độc lập.

| Thành phần | Dữ liệu / cấu hình dùng trong hướng dẫn |
|---|---|
| Video | `data/earbud_actions/raw_videos/per1/02/` hoặc toàn bộ `raw_videos/` |
| Nhãn action v2 | `configs/action_earbud_v2_config.json` |
| Annotation gốc | `data/earbud_actions/annotations_v2.csv` |
| Annotation sau chia tập | `data/earbud_actions/annotations_v2_split.csv` |
| Feature ViT | `data/earbud_actions/features_v2/` |
| BiLSTM v2 | `artifacts/action_model_v2/best.pt` |

Trong hướng dẫn này, tên `annotations_v2.csv` là phiên bản mới của `annotations.csv` bạn nhắc đến. Tất cả lệnh phải dùng cùng bộ đường dẫn v2; config `action_earbud_config.json` cũ vẫn chỉ có bốn lớp.

Mở PowerShell tại thư mục dự án:

```powershell
cd "G:\Internship\RBCNN_Demo"
```

Đặt video trong đúng thư mục. Ví dụ:

```text
data/earbud_actions/raw_videos/
  per1/01/per1_01_correct_001.mp4
  per1/02/per1_02_correct_001.mp4
  per2/01/per2_01_correct_001.mp4
```

Tên file phải có `person_session_...`, duy nhất trên toàn bộ dataset. Script lấy person/session từ tên file, không tự suy ra từ tên thư mục. Không đổi tên session để giả lập một buổi quay mới. CSV mẫu chỉ minh họa định dạng, không phải nhãn thật của video bạn.

## Bước 1: Gán nhãn hành động (Annotation)

Video mới quay chưa có mốc thời gian. Bạn xem từng video và đánh dấu lúc bắt đầu/kết thúc thao tác, sau đó chọn nhãn trong terminal.

### Cách thực hiện: dùng GUI có sẵn

Gán nhãn các video của `per1/02`:

```powershell
python scripts/annotate_actions.py `
  --videos-dir data/earbud_actions/raw_videos/per1/02 `
  --output data/earbud_actions/annotations_v2.csv `
  --config configs/action_earbud_v2_config.json
```

Có thể thêm `--list-videos` để kiểm tra menu sáu nhãn và danh sách video trước; chế độ đó không mở GUI và không tạo CSV.

Nếu muốn chọn đồng thời session 01 và 02 ở mọi người, thay bằng:

```powershell
python scripts/annotate_actions.py `
  --videos-dir data/earbud_actions/raw_videos `
  --session 01 --session 02 `
  --output data/earbud_actions/annotations_v2.csv `
  --config configs/action_earbud_v2_config.json
```

Chỉ chọn một lệnh để bắt đầu. Nếu CSV đã tồn tại, thêm `--resume` để bỏ qua video đã có ít nhất một dòng nhãn. Khi thêm video của session khác, vẫn ghi vào cùng CSV gốc với `--resume`.

**Nếu CSV đang chứa `insert_earbud` cũ:** giữ file đó làm bản lưu, chọn tên CSV mới và gán lại hai đoạn lắp từ video. Không tự đổi tất cả `insert_earbud` thành nhãn thứ nhất, và không dùng `--resume` với bộ nhãn cũ.

### Menu nhãn mới

| Số | Nhãn | Đoạn cần gán |
|---:|---|---|
| 0 | `idle` | Đứng chờ hoặc không thao tác, trước/giữa/sau các bước |
| 1 | `open_case` | Mở hộp và đưa vào vùng thao tác, đến lúc hộp ổn định |
| 2 | `insert_first_earbud` | Lấy tai nghe, đưa vào hộp đang trống: trạng thái 0 → 1 tai |
| 3 | `insert_second_earbud` | Lấy tai còn lại, đưa vào hộp đã có một tai: trạng thái 1 → 2 tai |
| 4 | `close_case` | Tác động lên nắp đến khi nắp đã đóng và tay dừng/rời hộp |
| 5 | `remove_earbud` | Nhấc một tai nghe ra khỏi khe/hộp; không phân biệt trái/phải hoặc lần tháo thứ mấy |

Đó là **sáu lớp phân loại**. Tai thứ nhất có thể là tai trái hoặc tai phải. Không gán action theo bên trái/phải; YOLO + Fusion chịu trách nhiệm xác định đúng khe và số tai còn lại.

### Phím tắt và cách đặt mốc

- `Space`: tạm dừng / tiếp tục.
- `s`: đặt START và tạm dừng. Nhấn `Space` để chạy tiếp tới cuối thao tác.
- `e`: đặt END, rồi chuyển sang terminal nhập số 0–5 và Enter. Tool trở lại trạng thái tạm dừng.
- `b` / `f`: lùi / tiến 5 giây.
- `,` / `.`: lùi / tiến một frame để chỉnh biên thao tác.
- `u`: bỏ đoạn vừa gán trong video hiện tại nếu chọn sai.
- `q`: lưu các đoạn của video này và chuyển video tiếp theo.
- `Esc`: lưu các đoạn đã đánh dấu trong video hiện tại và thoát.

Click vào cửa sổ video trước khi dùng phím tắt. Chỉ nhập số nhãn khi terminal yêu cầu. END bao gồm frame đang hiển thị; sau khi gán xong, tiến một frame bằng `.` trước khi đặt START cho đoạn kế tiếp để tránh chồng nhãn.

`--resume` bỏ qua **cả video** nếu CSV đã có nhãn của video đó, kể cả gán dở. Nên dùng `q` sau khi hoàn thành toàn bộ một video rồi mới thoát. Nếu đã thoát giữa video, sao lưu CSV và xóa riêng các dòng của video gán dở để gán lại, hoặc làm video đó vào CSV riêng rồi review/ghép có kiểm soát.

### Ví dụ một clip dài 15 giây

| Mốc minh họa | Nhãn |
|---|---|
| 0.0–1.0 s | `idle` |
| 1.0–3.0 s | `open_case` |
| 3.0–6.0 s | `insert_first_earbud` |
| 6.0–9.0 s | `insert_second_earbud` |
| 9.0–12.0 s | `close_case` |
| 12.0–15.0 s | `idle` |

**Các mốc trên chỉ là ví dụ.** Hãy theo hình ảnh thực tế, không sao chép mốc sang clip khác. Bắt đầu insertion khi tay lấy tai để lắp; kết thúc khi tai nằm trong khe và tay rời tai. Hai lần lắp phải là hai đoạn riêng, khoảng chờ ở giữa mang nhãn idle.

Clip đóng nắp sớm vẫn gán `close_case`; không thêm một insertion chưa xảy ra. Với clip đã có sẵn một tai, thao tác lắp tai còn lại là `insert_second_earbud`. Đoạn nhấc tai khỏi khe gán `remove_earbud`: START khi tay bắt đầu nhấc tai và END khi tai đã ra khỏi hộp. Thử lắp không thành hoặc thao tác mơ hồ không được gán thành insert/remove; loại đoạn đó hoặc ghi chú riêng để review.

Kết quả là CSV có các cột:

```csv
video_id,person_id,session_id,split,start_time_s,end_time_s,action_name
per1_02_correct_001,per1,02,train,3.0000,6.0000,insert_first_earbud
per1_02_correct_001,per1,02,train,6.0000,9.0000,insert_second_earbud
```

Đây là hai dòng minh họa. Review CSV thật: START < END, không trùng/chồng đoạn, không còn `insert_earbud`, không vượt thời lượng clip. Một clip lỗi có thể thiếu bước; toàn bộ mỗi split cần có đủ sáu lớp để đánh giá.

## Bước 2: Phân chia tập dữ liệu (Train / Val / Test Split)

**Khi có ít nhất ba nhóm `(person_id, session_id)` quay độc lập**, dùng cách chính:

```powershell
python scripts/split_annotations.py `
  --annotations data/earbud_actions/annotations_v2.csv `
  --output data/earbud_actions/annotations_v2_split.csv `
  --config configs/action_earbud_v2_config.json `
  --mode group-ratio `
  --train 0.70 --val 0.15 --test 0.15 --seed 42
```

Script giữ nguyên mỗi người/session trong một tập. Ví dụ `per1/01`, `per1/02`, `per2/01` là ba nhóm; nếu có đúng ba nhóm thì mỗi tập nhận một nhóm, chưa thể gần 70/15/15. Tỷ lệ tính theo nhóm nên không hứa số video chính xác là 12/3/2. Muốn kiểm tra người mới, nên dành toàn bộ video của người đó cho test; chia theo session chỉ đảm bảo phiên quay độc lập.

**Nếu hiện chỉ có 17 video trong `per1/02` và muốn thử pipeline ngay**, dùng phương án pilot:

```powershell
python scripts/split_annotations.py `
  --annotations data/earbud_actions/annotations_v2.csv `
  --output data/earbud_actions/annotations_v2_split.csv `
  --config configs/action_earbud_v2_config.json `
  --mode video-ratio `
  --train 0.70 --val 0.15 --test 0.15 --seed 42 `
  --allow-same-session
```

Với đúng 17 video, cách làm tròn hiện tại cho 12 train, 3 val, 2 test. Toàn bộ đoạn của một video ở cùng một tập; các tập vẫn cùng người/ánh sáng/buổi quay, nên kết quả pilot chưa đánh giá được khả năng làm việc với session mới. Không đổi seed nhiều lần để chọn điểm test cao.

File gốc `annotations_v2.csv` được giữ nguyên; các bước sau đọc `annotations_v2_split.csv`. Chỉ chạy một phương án chia cho một lần thí nghiệm. Đọc thống kê để xác nhận mỗi split có đủ cả sáu lớp. Khi thêm video, annotation vào CSV gốc, chia lại trước lần train mới và ghi rõ phiên bản dataset.

## Bước 3: Trích xuất đặc trưng hình ảnh với ViT (Spatial Feature Extraction)

Config v2 dùng **DINOv2-Small đóng băng**, embedding **384 chiều**, lấy mẫu tối đa **10 FPS**.

Trích xuất từ thư mục gốc để có feature cho mọi video đã annotation ở các session:

```powershell
python scripts/extract_spatial_features.py `
  --videos-dir data/earbud_actions/raw_videos `
  --output-dir data/earbud_actions/features_v2 `
  --config configs/action_earbud_v2_config.json `
  --batch-size 16
```

Nếu chỉ có `per1/02`, bạn có thể chỉ định thư mục đó, nhưng khi thêm session phải trích thêm feature tương ứng. Kết quả:

```text
data/earbud_actions/features_v2/
  per1_02_correct_001.npy
  per1_02_correct_001.json
```

`.npy` chứa chuỗi vector của video, `.json` lưu backbone/FPS/số frame. Script bỏ qua cache đã tồn tại; nếu đổi video, backbone hoặc sample FPS, cần tạo cache mới hoặc trích lại với `--overwrite`. Không cần trích lại chỉ vì sửa mốc annotation nếu video và cách trích không đổi.

Các lệnh không ép device: code tự chọn CUDA nếu PyTorch hỗ trợ, còn lại dùng CPU. Kiểm tra:

```powershell
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

Có card NVIDIA chưa đủ; PyTorch bản CPU vẫn trả `False`. Có thể thêm `--device cuda:0` khi kiểm tra trả `True`. Không có thời gian chạy cố định 3–5 phút; tốc độ tùy máy/video. Backbone có thể tải vào Hugging Face cache ở lần đầu, việc xử lý frame diễn ra trên máy.

Kiểm tra trước khi train:

```powershell
python scripts/verify_pipeline.py `
  --annotations data/earbud_actions/annotations_v2_split.csv `
  --features-dir data/earbud_actions/features_v2 `
  --config configs/action_earbud_v2_config.json
```

**Nếu chọn pilot ở Bước 2, thêm `--allow-same-session` vào lệnh kiểm tra trên.** Validator vẫn từ chối cùng video ở nhiều tập. Nó cũng kiểm tra đủ sáu lớp, thời gian hợp lệ, feature/backbone/FPS và các đoạn chồng nhau. Sửa các ERROR rồi mới làm bước 4.

## Bước 4: Huấn luyện mô hình Temporal BiLSTM

BiLSTM v2 có **2 lớp, hidden_dim 256, cửa sổ 16 embedding, 30 epoch**, đầu ra **6 lớp**. Với 10 FPS, cửa sổ bao phủ khoảng 1,6 giây. Đoạn ngắn được lặp frame cuối để đủ cửa sổ; không kéo dài biên giả sang thao tác khác để đủ 16 frame.

```powershell
python scripts/train_action_model.py `
  --annotations data/earbud_actions/annotations_v2_split.csv `
  --features-dir data/earbud_actions/features_v2 `
  --config configs/action_earbud_v2_config.json `
  --output artifacts/action_model_v2
```

Nếu chọn pilot ở Bước 2:

```powershell
python scripts/train_action_model.py `
  --annotations data/earbud_actions/annotations_v2_split.csv `
  --features-dir data/earbud_actions/features_v2 `
  --config configs/action_earbud_v2_config.json `
  --output artifacts/action_model_v2 `
  --allow-same-session
```

Chỉ chạy một lệnh phù hợp split. `--allow-same-session` chỉ nới kiểm tra session, không cho chia một video ra nhiều tập.

Kết quả:

- `artifacts/action_model_v2/best.pt`: trọng số tại epoch có validation Macro-F1 tốt nhất.
- `artifacts/action_model_v2/history.json`: loss và Macro-F1 train/validation từng epoch.

LSTM mới được train trên sáu nhãn; checkpoint bốn hoặc năm lớp cũ không dùng được. Các lệnh train dùng lại cùng thư mục output có thể ghi đè kết quả lần trước, vì vậy dùng output có version khi so sánh thí nghiệm. ViT đang frozen: bước này huấn luyện BiLSTM và đầu phân loại.

## Bước 5: Đánh giá mô hình (Evaluation)

Xem validation để kiểm tra lỗi và chọn cấu hình:

```powershell
python scripts/evaluate_action_model.py `
  --checkpoint artifacts/action_model_v2/best.pt `
  --annotations data/earbud_actions/annotations_v2_split.csv `
  --features-dir data/earbud_actions/features_v2 `
  --config configs/action_earbud_v2_config.json `
  --split val `
  --output artifacts/action_model_v2/evaluation_val.json
```

Sau khi chốt lựa chọn, đánh giá test:

```powershell
python scripts/evaluate_action_model.py `
  --checkpoint artifacts/action_model_v2/best.pt `
  --annotations data/earbud_actions/annotations_v2_split.csv `
  --features-dir data/earbud_actions/features_v2 `
  --config configs/action_earbud_v2_config.json `
  --split test `
  --output artifacts/action_model_v2/evaluation_test.json
```

**Nếu chọn pilot ở Bước 2, thêm `--allow-same-session` vào cả hai lệnh đánh giá.** Không dùng test để chọn epoch/threshold hoặc chỉnh nhãn nhằm tăng điểm.

Terminal và file JSON cung cấp:

- Macro F1-score trên sáu lớp.
- Confusion matrix **6 × 6**, hàng là nhãn thật, cột là nhãn dự đoán.
- Precision, recall, F1 và số mẫu của từng lớp.
- Số temporal windows; đây không phải số video/chu trình độc lập.

Thứ tự hàng/cột: `idle, open_case, insert_first_earbud, insert_second_earbud, close_case, remove_earbud`.

Hai insertion có chuyển động gần giống nhau. LSTM chỉ thấy cửa sổ gần nhất, nên có thể nhầm nếu hộp/khe bị tay che hoặc cửa sổ không chứa đủ ngữ cảnh. Điểm offline cao cũng chưa chứng minh kiểm tra quy trình đúng trên camera: YOLO và Fusion vẫn cần xác nhận occupancy 0 → 1 → 2, FSM kiểm tra trình tự. Khi test thực tế, theo dõi riêng nhầm lẫn giữa hai lớp insertion và các lần PASS sai.

Sau năm bước, xem hướng dẫn runtime trong README. Profile `configs/projects/earbud_v2.json` cần cả checkpoint action mới và checkpoint YOLO tương ứng; các lệnh trên chỉ tạo checkpoint action.

### Những việc bạn cần thực hiện

1. Video thật đã có trong `raw_videos`; kiểm tra lại tên person/session và chất lượng từng clip.
2. Gán hai lần lắp thành hai đoạn riêng, xem lại CSV.
3. Chọn group split khi có đủ session, hoặc pilot video split để thử ngay.
4. Chạy trích feature, kiểm tra, train, đánh giá theo đúng đường dẫn v2.
5. Quay thêm session/người độc lập để kiểm tra model ngoài điều kiện đã học.

`annotations_v2.csv` hiện mới có dòng header. Bạn phải gán mốc thời gian thật bằng GUI trước khi split, trích feature hoặc train.

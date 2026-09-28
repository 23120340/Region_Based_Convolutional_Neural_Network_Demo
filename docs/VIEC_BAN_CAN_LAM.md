# Việc bạn cần làm tiếp theo

## Tách nhánh main / insert_earbud — 28/09/2026

Main giữ năm nhãn first/second và các sửa UI/runtime; `feature/merge-insert-earbud-action` dùng bốn nhãn với insert chung. Hai model đã tách thư mục. Đọc [hướng dẫn hai nhánh](HUONG_DAN_HAI_NHANH_EARBUD.md) trước khi train hoặc đổi nhánh. Main được tái train vì weights năm lớp cũ đã bị ghi đè; bản bốn lớp đang có được giữ nguyên.

Bạn cần thử trên feature: phải trước/trái trước, giữ tai đứng yên sau PASS, sai khe, tháo/lắp lại và đóng sớm. LSTM nhận insert chung; FSM vẫn xác nhận từng tai. Chưa có bằng chứng camera thực tế rằng lỗi idle hoặc lắp phải trước đã được xử lý hoàn toàn.

## Cập nhật 28/09/2026: kiểm tra xác nhận trên giao diện

Code đã bổ sung ô xác nhận lớn, trạng thái từng bước và lịch sử PASS/VIOLATION. Không đổi checkpoint YOLO hoặc LSTM. Những mục lịch sử phía dưới không phải trạng thái mới nhất của model/dataset.

1. Thoát cửa sổ đang chạy bằng Q/Esc; mở PowerShell tại `G:\Internship\RBCNN_Demo` và chạy lại:

   ```powershell
   .\.venv\Scripts\python.exe scripts/run_hybrid.py --project configs/projects/earbud_v2.json --source 0 --no-mirror --fullscreen
   ```

2. Nhấn R; đưa hộp **mở rỗng** vào vùng quan sát, cho thấy hai khe. Chờ **ĐÃ XÁC NHẬN: Mở hộp** trước khi lắp tai thứ nhất. Dòng hướng dẫn sẽ cho biết số lần thấy khe trái/phải hoặc đang chờ LSTM `open_case`.
3. Lắp một tai đúng bên, đưa tay ra để bbox rõ. Chờ bước “tai thứ nhất” chuyển **ĐÃ XÁC NHẬN** rồi lắp tai còn lại. Chỉ đóng nắp khi cả hai bước lắp đã được xác nhận.
4. Khi đóng nắp được YOLO + LSTM + FSM chấp nhận, ô chính hiện **CHU TRÌNH ĐÃ HOÀN TẤT**. Nếu chỉ thấy nhãn LSTM/bbox mà bước chưa xanh thì chưa phải PASS; đọc ô **VIỆC CẦN LÀM / ĐIỀU KIỆN ĐANG CHỜ**.
5. Thử lấy một tai đã lắp ra: phải báo VI PHẠM, bước bị lùi hiện **CẦN LÀM LẠI**, không giữ trạng thái hoàn tất. Lắp lại tai đúng bên và đóng nắp lại sau khi được chấp nhận. R xóa lịch sử trên giao diện, nhưng log JSONL vẫn được lưu.
6. Nếu còn đứng bước, gửi ảnh **toàn giao diện** và đoạn log PASS/VIOLATION tương ứng. Nhớ ghi bên lắp trước, lớp bbox/độ tin cậy và dòng LSTM. Hiện PyTorch CPU-only; mục tiêu lấy mẫu 10 FPS không đảm bảo CPU đạt 10 FPS thực tế.

Ngưỡng hành động hybrid vẫn **>0,5**. Fusion dùng bbox **≥0,35**, cùng ngưỡng YOLO hiển thị; ngưỡng độ phủ khe 40% và ổn định lắp/tháo ba lần YOLO vẫn giữ. Chỉ khởi tạo vị trí khe được phép gom ba quan sát trống dương tính trong năm lần YOLO gần nhất, để chịu được mất bbox ngắn; không suy luận khe trống từ bbox bị mất.

YOLO mặc định kiểm tra mỗi frame theo camera config, không tự giảm xuống mỗi ba frame trên CPU. Nếu máy chậm, có thể thử `--yolo-every 2`, nhưng phải kiểm tra lại các bước nhanh; không giảm nhịp chỉ để tăng FPS hiển thị rồi coi kết quả xác nhận là tương đương.

## Cập nhật 27/09/2026: thử kiểm tra lắp/tháo theo từng khe

- Đã có checkpoint YOLO sáu lớp và BiLSTM pilot năm lớp trên máy. Các mục lịch sử phía dưới có thể mô tả trạng thái cũ.
- Chạy `.\run_hybrid.ps1 -Source 0 -Fullscreen` trong `G:\Internship\RBCNN_Demo`.
- Đọc các bước ở [HYBRID_SLOT_CHECK_20260927.md](HYBRID_SLOT_CHECK_20260927.md).
- Việc bạn cần thử: lắp phải trước/trái trước; đặt sai bên; lấy một/cả hai tai ra; đóng nắp khi thiếu tai; lắp lại sau lỗi; tay che và giữ tai lơ lửng trên khe.
- Đầu lượt phải cho thấy hai khe trống để hệ thống ghi nhớ vị trí. Nhấn R nếu đổi hộp/xoay mạnh.
- Sau khi tháo, đợi dòng yêu cầu “lắp lại tai trái/phải”; chỉ đóng nắp sau khi số tai đã xác nhận trở lại 2/2.
- Ghi lại tình huống báo sai và bổ sung ảnh/clip tương ứng để fine-tune. Chưa coi test mô phỏng là bằng chứng chính xác trên camera.

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

Dataset `RNN/LR_Earbud` đã được kiểm tra: 190/190 ảnh đã có sẵn trong
`RNN/train`, nên không được ghép lặp. Bản sạch dùng để train nằm tại
`datasets/earbud_rnn_merged` và đã đạt `Trainable: YES`. Xem
[DANH_GIA_DATASET_RNN.md](DANH_GIA_DATASET_RNN.md).

Để train detector trái/phải, upload `earbud_rnn_merged`, import
`Kaggle_Training_Earbud_LR.ipynb`, sửa `DATASET_ROOT` rồi Run All.

Phần detection cần bổ sung:

- Ảnh từ phiên quay độc lập, không phải frame liền nhau của video hiện tại.
- Thêm nhiều kiểu tai nghe nếu mục tiêu là nhận diện tổng quát.
- Thêm `other_earbud` nếu hệ thống phải từ chối tai nghe không thuộc đúng bộ.
- Tăng mạnh `empty_left` và `empty_right`; hiện toàn dataset chỉ có 32 và 25 box.

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

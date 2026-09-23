# Việc bạn cần làm – hệ thống kiểm tra đủ hai tai nghe

Cập nhật: 21/09/2026

## Phần code và cấu hình đã hoàn thành

Hệ thống hiện dùng quy trình:

```text
open_case
  → insert_earbud_1 (Earbud ở trong hộp, còn 1 khe trống)
  → insert_earbud_2 (đủ 2 Earbud ở trong hộp, còn 0 khe trống)
  → close_case
  → SUCCESS
```

Các luật đã được tích hợp:

- Không còn bước `pick_case` mơ hồ. `open_case` bắt đầu chu trình, `close_case` kết thúc chu trình.
- Mở nắp hộp không thể làm PASS bước lắp tai.
- `insert_earbud_1` chỉ PASS khi có một box `left_earbud` hoặc `right_earbud` nằm đúng bên trong `open_case` và chỉ còn khe đối diện trống.
- `insert_earbud_2` chỉ PASS khi có đủ `left_earbud` và `right_earbud` nằm trong `open_case`, không còn khe trống.
- Nếu `left_earbud` xuất hiện nhưng `empty_left` vẫn còn, hoặc `right_earbud` xuất hiện nhưng `empty_right` vẫn còn, Fusion phát `wrong_earbud_side` dưới dạng `VIOLATION`.
- Việc “nằm trong” dùng `Detection.is_inside()` với ngưỡng mặc định 60% diện tích box tai nghe/khe nằm trong box hộp.
- Đóng hộp khi còn thiếu tai sẽ phát `close_case` và FSM báo lỗi đỏ, không chuyển trạng thái.
- Nếu tai nghe đã PASS nhưng bị lấy ra, `empty_left`/`empty_right` xuất hiện trở lại ổn định sẽ phát `remove_earbud_to_one` hoặc `remove_earbud_to_zero` dưới dạng `VIOLATION`.
- FSM tự lùi về bước thực tế: còn 1 tai thì phải lắp lại tai thứ hai; còn 0 tai thì phải lắp lại từ tai thứ nhất.
- Luật tháo tai phải ổn định đủ `dwell_frames` (mặc định 3 kết quả inference), giảm báo nhầm khi tay che hoặc detector chớp tắt.
- Nếu tháo lần lượt cả hai tai, hệ thống phát hai violation liên tiếp: `2 → 1`, sau đó `1 → 0`; không được đóng hộp sau bất kỳ lần tháo nào.
- Không cần thêm nhãn hành động `remove` cho BiLSTM ở bản đầu. YOLO dùng bằng chứng dương là khe trống xuất hiện trở lại và số box tai trong hộp giảm, nên không nhầm động tác tay rút ra với động tác insert có hình dáng gần giống nhau.
- Màn hình hiển thị `inside=N/2` và `empty=N/2` để bạn biết máy đang dựa vào bằng chứng nào.

## Ba bước bạn cần thực hiện

### Bước 1 – Gán lại nhãn và train trên Kaggle

Dataset mới phải có đúng sáu tên lớp sau (notebook giữ nguyên class ID/thứ tự từ `data.yaml` của dataset):

```text
open_case
close_case
left_earbud
right_earbud
empty_left
empty_right
```

Quy tắc gán nhãn:

- `open_case`: một box bao toàn bộ hộp đang mở, đủ rộng để chứa vùng hai khe và hai tai nghe khi đã lắp.
- `close_case`: một box bao hộp đã đóng hoàn toàn.
- `left_earbud`: chỉ tai nghe vật lý bên trái, cả khi ở ngoài lẫn khi đã nằm trong hộp.
- `right_earbud`: chỉ tai nghe vật lý bên phải, cả khi ở ngoài lẫn khi đã nằm trong hộp.
- `empty_left`: chỉ vẽ khi khe trái đang trống và nhìn thấy được.
- `empty_right`: chỉ vẽ khi khe phải đang trống và nhìn thấy được.
- Khi tai đã che một khe, không được giữ nhãn `empty_left`/`empty_right` trên khe đó.
- Không cần lớp `Hand`; tay người là che khuất/background trong phiên bản detector này.

Mỗi trạng thái cần có đủ ảnh:

1. Hộp mở, hai khe trống, không có tai trong hộp.
2. Một tai trong hộp, một khe còn trống.
3. Hai tai trong hộp, không còn khe trống.
4. Hộp đã đóng.
5. Ảnh lỗi: tai ở sát hộp nhưng còn nằm ngoài, tai đặt trượt, tay che một phần, hộp đóng khi thiếu tai.
6. Ảnh sau khi tháo tai trái, sau khi tháo tai phải và sau khi tháo lần lượt cả hai tai; phải thấy rõ `empty_left`/`empty_right` tương ứng.

Lưu ý: dataset cũ không thể tự đổi chính xác sang sáu lớp mới. Nhãn `Earphone_Case` cũ không chứa thông tin mở/đóng; `Empty_Slot` cũ không chứa thông tin trái/phải. Có thể giữ annotation `Left_Earbud`/`Right_Earbud` đã đúng, nhưng phải sửa hai nhóm nhãn còn lại trên Roboflow/CVAT rồi export COCO hoặc YOLOv8.

Nếu export COCO, hãy import `Kaggle_Training_Earbud.ipynb`, Add Input dataset,
sửa duy nhất `DATASET_ROOTS` ở CELL 1 rồi bấm **Run All**. Bạn có thể đưa cả
dataset chính và dataset tai trái/phải vào cùng danh sách. Nếu muốn notebook
dừng ngay khi nhãn chưa đúng hướng mới, đặt `STRICT_GEOMETRY_V2 = True`.
Hướng dẫn và phương án chạy script thủ công nằm trong
[KAGGLE_TRAIN_EARBUD_COCO.md](KAGGLE_TRAIN_EARBUD_COCO.md).

Notebook hiện nhận trực tiếp Roboflow COCO. Nếu dataset đang ở YOLOv8, hãy export thêm một bản COCO từ Roboflow. Dù dùng định dạng nào cũng không được chỉ đổi tên schema cũ thành năm tên mới.

### Bước 2 – Đưa checkpoint mới vào dự án

Tải `best_earbud_detector.pt` hoặc `earbud_training_results.zip` từ Output của Kaggle rồi chép checkpoint vào:

```text
artifacts/training/earbud_geometry_detector/weights/best.pt
```

Checkpoint schema cũ không tương thích với cấu hình camera sáu lớp mới. Đặt `best.pt` vừa train vào đúng đường dẫn geometry ở trên.

### Bước 3 – Bật camera và thử đúng/sai

PowerShell:

```powershell
cd "G:\Internship\RBCNN_Demo"

.\.venv\Scripts\python.exe scripts\run_earbud.py `
  --mode camera `
  --source 0 `
  --auto-advance
```

Thử đúng một lượt:

1. Đưa hộp mở và hai khe trống vào WORK ZONE: `open_case` PASS.
2. Đặt tai thứ nhất vào hộp: màn hình phải chuyển từ `empty=2/2` sang `empty=1/2`, `inside=1/2` rồi PASS.
3. Đặt tai thứ hai: màn hình phải chuyển sang `empty=0/2`, `inside=2/2` rồi PASS.
4. Đóng hộp: `close_case` PASS và FSM báo `SUCCESS`.

Thử lỗi:

- Đặt tai bên cạnh hộp: không được PASS insert và phải báo tai chưa nằm trong hộp.
- Đóng hộp khi chưa có tai: báo đóng quá sớm.
- Đóng sau khi mới lắp một tai: báo còn thiếu 1/2 tai.
- Làm mất detection khe nhưng model chưa thấy đủ hai tai trong hộp: không được PASS tai thứ hai.
- Sau khi lắp tai thứ nhất, lấy tai đó ra: khi thấy lại hai khe trống, phải báo `VIOLATION` và quay về bước lắp tai thứ nhất.
- Sau khi lắp đủ hai tai, lấy một tai ra: khi thấy lại một khe trống, phải báo `VIOLATION` và yêu cầu lắp lại tai thứ hai.

## Nếu camera chưa PASS dù thao tác đúng

Kiểm tra dòng `Geometry` trên cửa sổ:

- Không thấy `open_case`: bổ sung ảnh hộp mở ở đúng góc camera vào dataset.
- `inside=0/2` dù tai đã lắp: box `open_case` khi gán nhãn có thể quá nhỏ, hoặc box `left_earbud`/`right_earbud` bị lệch.
- Số khe trống chập chờn: bổ sung ảnh `empty_left` và `empty_right`, nhất là lúc tay che một phần.
- Box hơi lệch ở camera thật: giảm `containment_threshold` trong `configs/camera_earbud_config.json` từ `0.6` xuống `0.5`. Không nên giảm thấp hơn trước khi kiểm tra lại annotation.

Không dùng phím số để đánh giá độ chính xác model vì phím số bỏ qua bằng chứng hình học. Chỉ dùng chúng để kiểm thử riêng FSM.

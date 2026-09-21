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
- `insert_earbud_1` chỉ PASS khi có ít nhất một box `earbud` nằm trong box `open_case` và chỉ còn một khe trống.
- `insert_earbud_2` chỉ PASS khi có hai box `earbud` nằm trong box `open_case` và không còn khe trống.
- Việc “nằm trong” dùng `Detection.is_inside()` với ngưỡng mặc định 60% diện tích box tai nghe/khe nằm trong box hộp.
- Đóng hộp khi còn thiếu tai sẽ phát `close_case` và FSM báo lỗi đỏ, không chuyển trạng thái.
- Nếu tai nghe đã PASS nhưng bị lấy ra, `empty_left`/`empty_right` xuất hiện trở lại ổn định sẽ phát `remove_earbud_to_one` hoặc `remove_earbud_to_zero` dưới dạng `VIOLATION`.
- FSM tự lùi về bước thực tế: còn 1 tai thì phải lắp lại tai thứ hai; còn 0 tai thì phải lắp lại từ tai thứ nhất.
- Luật tháo tai phải ổn định đủ `dwell_frames` (mặc định 3 kết quả inference), giảm báo nhầm khi tay che hoặc detector chớp tắt.
- Màn hình hiển thị `inside=N/2` và `empty=N/2` để bạn biết máy đang dựa vào bằng chứng nào.

## Ba bước bạn cần thực hiện

### Bước 1 – Gán lại nhãn và train trên Kaggle

Dataset mới phải có đúng năm tên lớp sau (notebook giữ nguyên class ID/thứ tự từ `data.yaml` của dataset):

```text
open_case
close_case
earbud
empty_left
empty_right
```

Quy tắc gán nhãn:

- `open_case`: một box bao toàn bộ hộp đang mở, đủ rộng để chứa vùng hai khe và hai tai nghe khi đã lắp.
- `close_case`: một box bao hộp đã đóng hoàn toàn.
- `earbud`: mỗi tai nghe vật lý là một box riêng, cả khi ở ngoài lẫn khi đã nằm trong hộp.
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

Lưu ý: dataset sáu lớp cũ không thể tự đổi chính xác sang năm lớp mới. Nhãn `Earphone_Case` cũ không chứa thông tin mở/đóng; `Empty_Slot` cũ không chứa thông tin trái/phải. Bạn phải sửa annotation trên Roboflow/CVAT rồi export lại định dạng YOLOv8.

Sau khi Add Data trên Kaggle:

1. Mở `Kaggle_Training_Earbud.ipynb`.
2. Sửa `DATASET_ROOT` nếu đường dẫn dataset khác.
3. Bật GPU và chọn **Run All**.

Notebook sẽ kiểm tra đúng bộ năm tên lớp trước khi train. Nếu vẫn là schema sáu lớp cũ, notebook chủ động dừng để tránh tạo checkpoint sai.

### Bước 2 – Đưa checkpoint mới vào dự án

Tải `training_results.zip` từ Output của Kaggle, giải nén và chép đè:

```text
artifacts/training/earbud_merged_detector/weights/best.pt
```

Checkpoint đang có trong dự án được train theo sáu lớp cũ nên không tương thích với cấu hình camera năm lớp mới. Bắt buộc thay bằng `best.pt` vừa train.

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
- `inside=0/2` dù tai đã lắp: box `open_case` khi gán nhãn có thể quá nhỏ, hoặc box `earbud` bị lệch.
- Số khe trống chập chờn: bổ sung ảnh `empty_left` và `empty_right`, nhất là lúc tay che một phần.
- Box hơi lệch ở camera thật: giảm `containment_threshold` trong `configs/camera_earbud_config.json` từ `0.6` xuống `0.5`. Không nên giảm thấp hơn trước khi kiểm tra lại annotation.

Không dùng phím số để đánh giá độ chính xác model vì phím số bỏ qua bằng chứng hình học. Chỉ dùng chúng để kiểm thử riêng FSM.

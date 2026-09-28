# Ngưỡng empty, độ phủ khe và camera điện thoại

Cập nhật 28/09/2026, tại `G:\Internship\RBCNN_Demo`. Lần sửa này trên nhánh **main** đang mở, không phải nhánh gộp insert. Chưa thay hoặc train lại model.

## 1. Có dùng nhầm model không?

YOLO hiện chọn `artifacts/training/earbud_geometry_detector/weights/best.pt`, checkpoint fine-tune từ YOLO11s với sáu tên:

```text
open_case, close_case, left_earbud, right_earbud, empty_left, empty_right
```

Tên trong checkpoint khớp config; adapter tra tên lớp, không ép thứ tự class ID của config lên kết quả. SHA-256 kiểm tra:

```text
7c7a2d1a6610737c57b2dddddf6bde0f987ae9d764be6bc876d3da1fed757843
```

Không có bằng chứng ánh xạ sai tên lớp trong runtime; điều này không bảo đảm detector nhận đúng mọi ảnh. Nếu bạn có best.pt mới hơn từ Kaggle, cần so sánh file cụ thể trước khi thay.

- `main`: `artifacts/action_model_pilot/best.pt`; năm nhãn idle/open/insert_first/insert_second/close.
- `feature/merge-insert-earbud-action`: `artifacts/action_model_insert_earbud/best.pt`; bốn nhãn idle/open/insert/close.

Terminal giờ in checkpoint YOLO, checkpoint LSTM và actions thực tế của config. Runtime từ chối checkpoint có nhãn khác config. Hai bước lắp bên phải UI là bước FSM; nhánh insert chung vẫn cần đếm đủ hai tai.

## 2. Ngưỡng mới

Trong `configs/camera_earbud_config.json`, mỗi lớp empty có `"min_confidence": 0.5`. Trong `configs/projects/earbud_v2.json`, Fusion có `"min_empty_confidence": 0.5`.

| Bằng chứng | Điều kiện |
|---|---|
| empty_left / empty_right | confidence **> 0,5**, đúng 0,5 bị loại |
| Tai nghe / hộp YOLO | confidence ≥ 0,35 như trước |
| Hành động LSTM | confidence > 0,5 |
| Tai chiếm khe | phủ ≥ 40%, đúng bên, ổn định 3 quan sát YOLO và LSTM phù hợp |

Empty dưới ngưỡng không được dùng hiệu chuẩn hoặc suy ra tháo tai. Nó cũng không xuất hiện trên overlay. Không thay ngưỡng action hay biến mất empty thành PASS.

Ảnh bạn gửi đã có `insert_earbud` 95–96,5%; vấn đề là thiếu bbox tai và empty sai 0,69–0,90. Các empty sai đó vẫn vượt 0,5. Muốn giải quyết gốc cần ảnh rỗng/chỉ trái/chỉ phải/đủ hai tai từ camera mới, gán đúng nhãn và fine-tune YOLO. Không gắn empty lên khe đã chứa tai.

## 3. Độ phủ giờ hiển thị gì?

`coverage = diện tích giao(bbox tai, bbox khe) / diện tích bbox khe`. Đây không phải IoU, confidence empty hoặc mức sạc.

- Có cả bbox khe và tai trong hộp: hiện phần trăm, kể cả dưới 40%; 27% chưa đủ để xác nhận lắp.
- Thiếu bbox tai: hiện **Phủ khe: thiếu bbox tai**, không trình bày như phép đo 0%.
- Thiếu bbox khe/anchor: hiện **Phủ khe: thiếu bbox khe**.
- Hộp đóng: không đo độ phủ khe.
- Dòng `Empty ...` thể hiện confidence của detection empty hiện tại; dấu — nghĩa là chưa có empty đạt điều kiện.
- Box tham chiếu trên camera ghi `slot_left/right cover:...`, có thể là vị trí khe đã ghi nhớ, không nhất thiết là empty đang được YOLO nhận.

Các quy tắc PASS/VIOLATION vẫn giữ nguyên. Chỉ bổ sung chẩn đoán, không tạo bbox tai giả.

## 4. Nối điện thoại bằng USB

**Không phải cứ cắm dây rồi đặt source 3 là chạy.** Windows phải nhìn thấy điện thoại như webcam trước. Số source được cấp theo thiết bị/backend, không cố định.

### Điện thoại hỗ trợ webcam USB trực tiếp

Dùng cáp truyền dữ liệu; trên điện thoại chọn chế độ USB **Webcam** nếu có. Pixel có hướng dẫn chính thức: [Use your Pixel phone as a webcam](https://support.google.com/pixelcamera/answer/14274129?hl=en). Không mặc định mọi Android/iPhone có tùy chọn này.

### Nếu không có chế độ webcam: DroidCam

1. Cài app DroidCam trên điện thoại và [PC client chính thức](https://droidcam.app/windows/) trên Windows.
2. Android: bật USB Debugging, dùng cáp dữ liệu và chấp nhận yêu cầu cho phép từ PC của bạn; cài driver hãng nếu cần.
3. iPhone: chấp nhận Trust Computer và bảo đảm driver Apple USB đã cài.
4. Mở app điện thoại, kết nối thiết bị trong PC client qua USB và xác nhận preview đã có hình.
5. Dùng output webcam `DroidCam Video` mà PC client cung cấp.

Chi tiết Android/iOS và driver: [DroidCam USB Setup](https://droidcam.app/help/#usb). Không tự cài phần mềm/driver hoặc bật debug từ repository. Chỉ cho phép máy tính tin cậy; tắt USB Debugging sau khi không dùng.

## 5. Chọn đúng source trong dự án

Thoát realtime cũ bằng Q/ESC trước để tránh chiếm camera. Mở PowerShell:

```powershell
cd "G:\Internship\RBCNN_Demo"
.\.venv\Scripts\python.exe scripts/run_hybrid.py --list-cameras --max-camera-index 8
```

Lệnh chỉ dò camera, không nạp YOLO/DINO/LSTM. Danh sách có số 3 không tự chứng minh đó là điện thoại; thử source đó và xem hình. Nếu đúng điện thoại ở số 3:

```powershell
.\run_hybrid.ps1 -Source 3 -Fullscreen
```

Hoặc:

```powershell
.\.venv\Scripts\python.exe scripts/run_hybrid.py --project configs/projects/earbud_v2.json --source 3 --no-mirror --fullscreen
```

Nếu không thấy camera: kiểm tra preview PC client, driver, quyền camera cho desktop apps, ứng dụng khác đang chiếm camera và chỉ số dò tối đa. Không đổi model để chữa lỗi kết nối camera.

Đặt điện thoại cố định, dùng góc camera/ánh sáng tương tự dữ liệu; đầu lượt mở hộp rỗng và cho thấy rõ cả hai khe. Khi thay camera/vị trí hộp, bắt đầu lại bằng R. Camera rõ hơn có thể giúp chất lượng đầu vào, nhưng không thay thế việc sửa nhãn/train detector.

## 6. Kiểm thử và giới hạn

140 tests đạt trên bản staging và bản thật ở ổ G, gồm lọc ngưỡng, hiệu chuẩn/tháo, hiển thị độ phủ và chế độ dò camera. Không tự mở camera của bạn để kiểm thử và chưa xác nhận điện thoại là source 3.

Lần sửa này chưa tự áp dụng lên nhánh feature; không đổi nhánh trên worktree đang có sửa dở. Kết quả tests và các file thay đổi xem `TOM_TAT_HOP_NHAT_REPO_20260922.md`.

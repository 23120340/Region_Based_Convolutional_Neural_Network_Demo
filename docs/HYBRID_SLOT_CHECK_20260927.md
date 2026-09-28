# Kiểm tra lắp/tháo tai nghe theo từng khe — 27/09/2026

Bản chạy: `G:\Internship\RBCNN_Demo`. Điểm vào là `scripts/run_hybrid.py`; script LSTM riêng không chạy YOLO và không áp dụng luật bbox này.

## Chạy camera

Mở PowerShell trong thư mục dự án:

```powershell
.\run_hybrid.ps1 -Source 0 -Fullscreen
```

Camera USB có thể là `-Source 1`. Launcher mặc định không mirror. Thêm `-Mirror` chỉ khi cần lật hình cho khớp dữ liệu quay.

Hoặc chạy trực tiếp:

```powershell
.\.venv\Scripts\python.exe scripts/run_hybrid.py --project configs/projects/earbud_v2.json --source 0 --no-mirror --fullscreen
```

Không cần train lại BiLSTM cho thay đổi logic. Profile đã trỏ đúng cặp `action_earbud_pilot_config.json` và `artifacts/action_model_pilot/best.pt` (5 lớp). YOLO dùng `artifacts/training/earbud_geometry_detector/weights/best.pt`.

## Cách thử đúng

1. Đặt một hộp mở nắp trong vùng thao tác, lấy cả hai tai ra. Cho camera thấy rõ hai khe và thực hiện động tác mở hộp để LSTM xác nhận `open_case`.
2. Đợi hai khe được ghi nhớ, FSM báo hộp sẵn sàng. Nếu bỏ lỡ động tác mở hộp, đóng rồi mở lại; nhấn R để bắt đầu lại khi cần.
3. Lắp tai trái hoặc phải trước đều được. YOLO phải thấy tai đúng bên phủ khe; LSTM phải nhận `insert_first_earbud`.
4. Lắp tai còn lại; chờ `insert_second_earbud` được xác nhận.
5. Thử lấy một tai ra. Đợi khe tương ứng được phát hiện trống ổn định. Giao diện báo `VIOLATION: remove_earbud_to_one`, yêu cầu lắp lại đúng tai bị thiếu; quy trình lùi về đã lắp một tai.
6. Đóng nắp lúc này phải bị báo lỗi. Mở lại, lắp tai bị thiếu, đợi xác nhận rồi mới đóng.
7. Nếu lấy cả hai tai ra, lỗi là `remove_earbud_to_zero`; phải làm lại từ bước lắp tai thứ nhất.
8. Nhấn R khi thay hộp hoặc bắt đầu một lượt mới. Phím F bật/tắt toàn màn hình; Q/Esc thoát.

Camera phải nhìn thấy khe khi thử tháo. Chỉ mất bbox tai nghe do tay che chưa đủ bằng chứng để báo tháo.

## Luật kết hợp

Profile dùng `assembly.slot_fusion:PairedEarbudFusionEngine`.

| Bằng chứng | Kết quả |
|---|---|
| `right_earbud` phủ đúng vùng `empty_right` đã ghi nhớ; ổn định; LSTM đúng bước | Xác nhận lắp tai phải |
| `left_earbud` phủ đúng vùng `empty_left` đã ghi nhớ; ổn định; LSTM đúng bước | Xác nhận lắp tai trái |
| Tai đúng khe nhưng LSTM chưa chắc/chưa đúng bước | Chờ hoặc báo sai thứ tự; chưa tăng số tai đã xác nhận |
| Tai phải phủ khe trái hoặc ngược lại | `wrong_earbud_side`, yêu cầu sửa bên |
| Khe đã xác nhận có tai nay trống ổn định, không còn tai phủ khe | Báo tháo; lùi FSM; yêu cầu lắp lại |
| Khe/tai bị mất detection, không có bằng chứng khe trống | Chưa rõ; giữ tiến độ đã xác nhận |
| YOLO thấy hộp đóng ổn định + LSTM nhận đóng nắp | FSM cho PASS khi đã xác nhận hai tai; nếu thiếu thì VIOLATION |

Vùng khe trống được ghi nhớ theo tọa độ tương đối trong hộp, nên vẫn đối chiếu được khi bbox `empty_right` biến mất sau khi lắp tai. Không suy ra “đã lắp” chỉ vì mất bbox khe.

Độ phủ = diện tích giao nhau của bbox tai và bbox khe / diện tích bbox khe. Đây không phải IoU. Ngưỡng ban đầu là 0,4; bbox tai phải nằm phần lớn trong hộp. Một bbox mơ hồ phủ ngang cả hai khe không được tính thành hai tai.

Nếu YOLO cùng lúc vẽ tai đúng khe và khe trống, bằng chứng tai phủ đúng khe được ưu tiên. Do đó hệ thống không lặp lắp/tháo chỉ vì hai bbox còn chồng nhau.

## Ngưỡng có thể chỉnh

Trong `configs/projects/earbud_v2.json`:

```json
"stable_frames": 3,
"min_action_confidence": 0.5,
"min_detection_confidence": 0.35,
"containment_threshold": 0.6,
"slot_overlap_threshold": 0.4,
"require_closed_case": true
```

`stable_frames` tính theo số lần YOLO chạy, không phải số frame hiển thị. Hybrid chỉ nhận LSTM khi confidence **lớn hơn** `min_action_confidence` (0,5). Có thể ghi đè bằng `--action-confidence 0.65`. Confidence YOLO và confidence LSTM là hai ngưỡng khác nhau. Ngưỡng trong project profile ưu tiên hơn action config khi chạy hybrid; script LSTM riêng vẫn đọc action config.

Cập nhật 28/09: Fusion dùng cùng ngưỡng bbox 0,35 với YOLO hiển thị. Riêng khởi tạo vị trí khe rỗng gom ba quan sát dương tính trong năm lần YOLO gần nhất, chịu được mất bbox ngắn; bằng chứng lắp/tháo vẫn cần ba lần liên tiếp. Ô hướng dẫn cho biết đang chờ khe, YOLO ổn định hoặc LSTM đúng bước.

LSTM không bắt buộc phải có lớp `remove_earbud`: khe trống xuất hiện trở lại là bằng chứng tháo. Nếu sau này train lớp tháo, dự đoán ấy là thông tin hỗ trợ.

## Giao diện và giới hạn

- Canvas 1440×900, cửa sổ kéo giãn được hoặc toàn màn hình; ảnh camera giữ tỉ lệ.
- Cột bên phải hiển thị các bước, trạng thái mỗi khe, phần trăm phủ và tiến độ ổn định.
- Ô lớn dưới camera giữ **ĐÃ XÁC NHẬN** sau PASS hoặc **CHU TRÌNH ĐÃ HOÀN TẤT** sau đủ quy trình. Lịch sử ba sự kiện gần nhất nằm bên phải; khi tháo tai, bước bị lùi hiện **CẦN LÀM LẠI**. Nhãn LSTM/bbox đơn lẻ không được coi là xác nhận.
- Hướng dẫn tiếng Việt bên dưới chỉ rõ phải lắp lại bên nào.
- Giao diện được vẽ sau suy luận, không đưa chữ và overlay vào DINOv2/YOLO.
- Khi không thấy hộp, hệ thống tạm dừng kiểm tra, không tự xóa tiến độ hoặc kết luận tháo.
- Bbox 2D không chứng minh tai đã khớp cơ khí/đang sạc; tay giữ tai lơ lửng đúng trên khe vẫn có thể gây nhầm. Hãy thử tình huống này bằng camera thực tế.
- Tọa độ khe được ghi nhớ phù hợp với hộp giữ hướng tương đối ổn định; nếu đổi hộp/xoay mạnh hãy nhấn R rồi hiệu chuẩn lại bằng hai khe trống.
- Python trong `.venv` hiện là PyTorch CPU-only; logic này không tự bật CUDA.

Các test mô phỏng kiểm tra logic và trạng thái; không thay thế đánh giá trên camera/dataset thật.

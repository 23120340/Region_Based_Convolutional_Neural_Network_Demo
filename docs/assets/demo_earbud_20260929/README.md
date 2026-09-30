# Ảnh minh chứng demo Earbud ngày 29/09/2026

Thư mục này chứa các khung hình được trích từ hai video holdout trong `tests/`. Các ảnh dùng để chèn vào báo cáo hoặc slide bảo vệ; không đưa hai video nguồn hay các ảnh này vào tập huấn luyện YOLO/LSTM.

| Tệp ảnh | Video nguồn | Mốc thời gian | Nội dung minh chứng |
|---|---|---:|---|
| `01_correct_wait_for_open.jpg` | `2026-09-29 10-17-04.mp4` | 00:15 | Hộp đóng, hệ thống ở `WAIT_FOR_OPEN`, không báo vi phạm |
| `02_correct_open_case_pass.jpg` | `2026-09-29 10-17-04.mp4` | 00:20 | Xác nhận mở hộp và chuyển sang `S1_CASE_READY` |
| `03_correct_first_earbud_pass.jpg` | `2026-09-29 10-17-04.mp4` | 00:45 | Xác nhận tai thứ nhất, tiến độ 1/2 |
| `04_correct_second_earbud_pass.jpg` | `2026-09-29 10-17-04.mp4` | 01:35 | Xác nhận tai thứ hai, tiến độ 2/2 |
| `05_correct_cycle_completed.jpg` | `2026-09-29 10-17-04.mp4` | 01:40 | Xác nhận đóng hộp, FSM hoàn tất chu trình |
| `06_fault_remove_to_zero_violation.jpg` | `2026-09-29 15-43-35.mp4` | 00:27 | Hệ thống ghi nhận `VIOLATION · remove_earbud_to_zero` và yêu cầu lắp lại |
| `07_fault_remove_to_one_violation.jpg` | `2026-09-29 15-43-35.mp4` | 00:36 | Sau khi đã đạt 2/2, tháo một tai làm FSM lùi về 1/2 và phát cảnh báo |
| `08_fault_corrective_first_reinsert.jpg` | `2026-09-29 15-43-35.mp4` | 00:54 | Thao tác sửa sai được chấp nhận, FSM xác nhận lại tai thứ nhất |
| `09_fault_corrective_second_reinsert.jpg` | `2026-09-29 15-43-35.mp4` | 01:06 | Hai tai được xác nhận đúng khe, tiến độ trở lại 2/2 |
| `10_fault_completed_after_correction.jpg` | `2026-09-29 15-43-35.mp4` | 01:09 | Đóng hộp sau sửa sai, FSM đạt `S4_COMPLETED` |

## Ghi chú sử dụng

- Với báo cáo Word, nên chèn ảnh ở chiều rộng khoảng 14–16 cm và giữ nguyên tỷ lệ.
- Khi trình bày video thứ hai, nên gọi đây là **kịch bản lỗi tháo tai và sửa sai**. Dù tên video được mô tả là “lắp sai”, các mã sự kiện nhìn thấy rõ trên UI là `remove_earbud_to_zero` và `remove_earbud_to_one`; video chưa đủ để khẳng định riêng độ chính xác của nhánh `wrong_side`.
- Để kiểm chứng `wrong_side`, cần thêm clip giữ tai trái trong khe phải hoặc tai phải trong khe trái tối thiểu 0,5 giây, không che khuất vùng khe.

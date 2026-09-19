# Dự án Giám sát Lắp ráp Hộp Tai nghe (Earbud Assembly Monitor)

Dự án này là hệ thống giám sát quy trình lắp ráp hộp tai nghe (Earbud) theo thời gian thực sử dụng camera và mô hình YOLO đã tinh chỉnh (fine-tuned). Hệ thống giúp theo dõi từng bước lắp ráp, cảnh báo thao tác lỗi, sai quy trình hoặc bỏ bước.

## Tính năng

- Theo dõi quy trình 3 bước: Đặt hộp sạc -> Lắp tai nghe vào hộp -> Đóng nắp hộp.
- Phát hiện các bất thường như quên lắp tai nghe mà đã đóng nắp, hoặc đóng nắp sớm.
- Giao diện Live Camera kết hợp Bounding Box và dự đoán hành động theo thời gian thực.
- Theo dõi trạng thái thông qua FSM (Finite State Machine) chống nhiễu (debouncer).

## Hướng dẫn sử dụng

### 1. Chạy giám sát bằng Live Camera (Khuyên dùng)
Bạn có thể chạy camera mặc định (ví dụ camera laptop hoặc camera USB) để nhận diện thời gian thực.

```powershell
python scripts/run_earbud.py --mode camera --source 0
```
- Nếu có nhiều camera, chạy lệnh `python scripts/run_earbud.py --list-cameras` để xem ID. Thay `--source 0` bằng `--source 1` tương ứng.
- Có thể chạy kèm `--auto-advance` nếu muốn hệ thống tự chuyển bước sau một khoảng thời gian chờ (dwell time).

### 2. Chạy giao diện mô phỏng (GUI)
Nếu bạn không có camera hoặc chỉ muốn kiểm tra logic của FSM, hãy dùng giao diện giả lập:

```powershell
python scripts/run_earbud.py --mode gui
```

### 3. Inference trên ảnh tĩnh
Để chạy thử trên một thư mục ảnh hoặc một ảnh tĩnh:

```powershell
python scripts/run_earbud.py --mode infer --source đường_dẫn_ảnh
```

## Các phím tắt trong chế độ Camera

| Phím | Chức năng |
| :--- | :--- |
| **`Space`** | Xác nhận hành động hiện tại |
| **`1`-`3`** | Chuyển thủ công từng bước (pick_case / insert_earbud / close_case) |
| **`C`** | Đổi qua lại giữa các camera |
| **`R`** | Đặt lại chu trình FSM |
| **`S`** | Chụp màn hình (Lưu tại `artifacts/screenshots`) |
| **`Q`/`ESC`** | Thoát |

## Cấu trúc thư mục (Earbud)

- `configs/`
  - `camera_earbud_config.json`: Cấu hình lớp (classes) và vùng hoạt động của YOLO.
  - `earbud_fsm_config.json`: Cấu hình các trạng thái State Machine.
  - `action_earbud_config.json`: Cấu hình cho model nhận diện hành động (Action model).
- `src/assembly/`: Engine xử lý FSM, debouncer, vision, camera stream.
- `artifacts/training/earbud_merged_detector/`: Nơi chứa trọng số model YOLO đã được fine-tune (`best.pt`).

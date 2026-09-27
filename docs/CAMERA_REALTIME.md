# Camera realtime và thu ảnh YOLO cho tai nghe

Chạy lệnh từ `G:\Internship\RBCNN_Demo`.

## 1. Kiểm tra camera

```powershell
python scripts/run_earbud.py --list-cameras
```

Camera laptop thường là `0`; camera USB thường là `1` hoặc `2`. Khi đang chạy, nhấn `C` để chuyển qua camera khả dụng tiếp theo.

## 2. Chụp ảnh làm dataset detection

```powershell
python scripts/capture_detection_images.py --camera 0
```

Mặc định ảnh được lưu vào `datasets/earbud_geometry/raw/`. Đổi camera hoặc thư mục:

```powershell
python scripts/capture_detection_images.py `
  --camera 1 `
  --output datasets/earbud_geometry/raw/person02/session01
```

Phím: `SPACE` lưu ảnh, `Q`/`Esc` thoát. Mỗi ảnh nên thay đổi vị trí hộp, góc xoay, tay che, ánh sáng và trạng thái lắp; không giữ hàng chục ảnh gần như giống nhau.

Khoanh đúng sáu lớp:

```text
open_case
close_case
left_earbud
right_earbud
empty_left
empty_right
```

`open_case` phải bao vùng chứa hai tai/khe. Gán đúng tai vật lý bằng `left_earbud` hoặc `right_earbud`; không vẽ thêm box `earbud` chung trên cùng vật thể. Chỉ vẽ `empty_left` hoặc `empty_right` khi khe tương ứng thật sự trống và nhìn thấy.

## 3. Kiểm tra và train YOLO

Nếu export từ Roboflow đã có `train/images`, `valid/images`, `test/images`, dùng thẳng file YAML; không chạy split lại.

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

Checkpoint mục tiêu:

```text
artifacts/training/earbud_geometry_detector/weights/best.pt
```

Dataset 3 lớp `Case/Earbud/Empty_Slot` và dataset baseline cũ không thể tự đổi chính xác sang sáu lớp geometry trái/phải; phải sửa annotation trước.

## 4. Chạy realtime

```powershell
python scripts/run_earbud.py `
  --mode camera `
  --source 0 `
  --auto-advance `
  --device 0
```

Phím điều khiển:

| Phím | Chức năng |
|---|---|
| `C` | Đổi camera |
| `S` | Lưu frame có bounding box/overlay vào `artifacts/screenshots/` |
| `R` | Reset chu trình |
| `SPACE` | Xác nhận gợi ý hiện tại bằng tay |
| `1`–`9` | Mô phỏng bước FSM; chỉ dùng debug |
| `Q` / `Esc` | Thoát |

Không dùng phím số hoặc `SPACE` khi đánh giá độ chính xác tự động vì chúng bỏ qua một phần bằng chứng model.

## 5. Chạy checkpoint BiLSTM pilot bằng camera

Để hình không bị lật gương và cửa sổ mở ở 1280×720:

```powershell
python scripts/run_earbud_lstm.py `
  --source 0 `
  --model artifacts/action_model_pilot/best.pt `
  --action-config configs/action_earbud_pilot_config.json `
  --fsm-config configs/earbud_v2_fsm_config.json `
  --device cpu `
  --sample-fps 4 `
  --no-mirror `
  --window-width 1280 `
  --window-height 720
```

Thêm `--fullscreen` nếu muốn mở toàn màn hình ngay từ đầu. Trong lúc chạy, nhấn `F` để chuyển qua lại giữa cửa sổ và toàn màn hình; `R` reset chu trình; `Q` hoặc `Esc` để thoát.

`--no-mirror` chỉ tác động khi `--source` là camera. Video file luôn được hiển thị đúng chiều gốc.

Ngưỡng nhận hành động nằm trong `configs/action_earbud_pilot_config.json`:

```json
"inference": {
  "min_confidence": 0.5
}
```

Runtime chỉ gửi hành động vào FSM khi `confidence > 0.5`. Giá trị bằng hoặc thấp hơn 0,5 được hiển thị là `uncertain` và không làm chuyển bước.

## 6. Hybrid v2

Bản kiểm tra từng khe và giao diện lớn: xem [HYBRID_SLOT_CHECK_20260927.md](HYBRID_SLOT_CHECK_20260927.md).

```powershell
.\run_hybrid.ps1 -Source 0 -Fullscreen
```

Launcher dùng checkpoint LSTM pilot đang có, không mirror mặc định. Bắt đầu bằng hộp mở rỗng với hai khe nhìn thấy; R để bắt đầu lượt mới.

Sau khi có cả YOLO v2 và action model v2:

```powershell
python scripts/run_hybrid.py `
  --project configs/projects/earbud_v2.json `
  --source 0 `
  --device 0
```

Lần đầu dùng DINOv2 khi cache chưa có cần mạng và cờ `--allow-download`. Sau khi model đã nằm trong cache Hugging Face, runtime chạy local.

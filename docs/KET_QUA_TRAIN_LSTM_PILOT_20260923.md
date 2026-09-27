# Kết quả train thử BiLSTM — 23/09/2026

## Mục tiêu lần chạy

Kiểm tra toàn bộ pipeline `annotation → split → DINOv2 → BiLSTM → evaluation` bằng 35 video hiện có.

Đây là **pilot 5 hành động**, chưa phải mô hình production. Nhãn `remove_earbud` tạm thời không được đưa vào vì `annotations_v2.csv` chưa có đoạn video nào mang nhãn này.

Các hành động đã train:

1. `idle`
2. `open_case`
3. `insert_first_earbud`
4. `insert_second_earbud`
5. `close_case`

## Dữ liệu và cấu hình

- Config: `configs/action_earbud_pilot_config.json`
- Annotation đã chia: `data/earbud_actions/annotations_v2_pilot_split.csv`
- DINOv2 backbone: `facebook/dinov2-small`, embedding 384 chiều
- Sequence length: 16 frame; stride: 4
- BiLSTM: 2 lớp, hidden dimension 256, bidirectional
- Epoch: 15; batch size: 16
- Split theo video: 25 train, 5 validation, 5 test
- Temporal windows: 295 train, 67 validation, 57 test

Do hiện chỉ có hai session của cùng `per1`, lệnh split phải dùng `--allow-same-session`. Vì vậy kết quả dưới đây chỉ chứng minh pipeline hoạt động trên dữ liệu pilot, chưa chứng minh khả năng tổng quát sang người, camera hoặc bối cảnh mới.

## Kết quả

Checkpoint tốt nhất xuất hiện ở epoch 4:

- Validation macro-F1: **0,9644**
- Validation accuracy: **97,01%**
- Test macro-F1: **0,9040**
- Test accuracy: **92,98%**

Test F1 theo lớp:

| Hành động | F1 | Số window |
|---|---:|---:|
| `idle` | 0,9756 | 21 |
| `open_case` | 0,7273 | 7 |
| `insert_first_earbud` | 0,8696 | 10 |
| `insert_second_earbud` | 1,0000 | 10 |
| `close_case` | 0,9474 | 9 |

Lỗi đáng chú ý: 3/7 window `open_case` bị dự đoán thành `insert_first_earbud`. Đây là lớp cần quay và gán nhãn thêm, đặc biệt quanh ranh giới giữa lúc vừa mở hộp và lúc bắt đầu cầm tai nghe.

## File kết quả

- Checkpoint: `artifacts/action_model_pilot/best.pt`
- Lịch sử train: `artifacts/action_model_pilot/history.json`
- Báo cáo validation: `artifacts/action_model_pilot/evaluation_val.json`
- Báo cáo test: `artifacts/action_model_pilot/evaluation_test.json`
- Feature cache: `data/earbud_actions/features_v2_pilot/`

`artifacts/` và feature cache được Git ignore vì có thể tái tạo và có dung lượng lớn. Hãy sao lưu `best.pt` riêng nếu cần giữ checkpoint.

## Lệnh tái tạo

```powershell
python scripts/split_annotations.py `
  --annotations data/earbud_actions/annotations_v2.csv `
  --output data/earbud_actions/annotations_v2_pilot_split.csv `
  --config configs/action_earbud_pilot_config.json `
  --mode video-ratio --train 0.70 --val 0.15 --test 0.15 `
  --seed 42 --allow-same-session

python scripts/extract_spatial_features.py `
  --videos-dir data/earbud_actions/raw_videos `
  --output-dir data/earbud_actions/features_v2_pilot `
  --config configs/action_earbud_pilot_config.json `
  --device cpu --batch-size 16

python scripts/train_action_model.py `
  --annotations data/earbud_actions/annotations_v2_pilot_split.csv `
  --features-dir data/earbud_actions/features_v2_pilot `
  --config configs/action_earbud_pilot_config.json `
  --output artifacts/action_model_pilot `
  --device cpu --allow-same-session

python scripts/evaluate_action_model.py `
  --checkpoint artifacts/action_model_pilot/best.pt `
  --annotations data/earbud_actions/annotations_v2_pilot_split.csv `
  --features-dir data/earbud_actions/features_v2_pilot `
  --config configs/action_earbud_pilot_config.json `
  --split test `
  --output artifacts/action_model_pilot/evaluation_test.json `
  --device cpu --allow-same-session
```

## Việc cần làm trước khi train bản 6 hành động

1. Quay và gán nhãn `remove_earbud` cho cả tháo tai trái, tai phải và tháo lần lượt hai tai.
2. Thu ít nhất một session độc lập từ người hoặc ngày quay khác để dành riêng cho test.
3. Bổ sung mẫu `open_case`, nhất là các frame chuyển tiếp sang `insert_first_earbud`.
4. Cài PyTorch có CUDA trong môi trường riêng để RTX 3050 xử lý DINOv2; Python hiện tại đang dùng `torch 2.13.0+cpu`.
5. Sau khi annotation có đủ sáu lớp, quay lại `configs/action_earbud_v2_config.json`, trích xuất feature và train checkpoint production mới.

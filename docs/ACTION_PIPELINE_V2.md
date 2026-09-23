# Pipeline action v2

Hướng dẫn đầy đủ theo đúng **Bước 1–5** đã được thống nhất tại [LSTM_Training_Guide.md](LSTM_Training_Guide.md):

1. Gán nhãn hành động bằng GUI, menu 0–5 và mốc thời gian từng thao tác.
2. Chia train/validation/test theo session; có tùy chọn pilot chia theo video.
3. Trích đặc trưng DINOv2-Small 384 chiều và kiểm tra cache.
4. Huấn luyện BiLSTM đầu ra sáu lớp.
5. Đánh giá validation/test, Macro-F1 và confusion matrix 6 × 6.

Dùng config `configs/action_earbud_v2_config.json` và các đường dẫn:

```text
data/earbud_actions/annotations_v2.csv
data/earbud_actions/annotations_v2_split.csv
data/earbud_actions/features_v2/
artifacts/action_model_v2/
```

Sáu nhãn: `idle`, `open_case`, `insert_first_earbud`, `insert_second_earbud`, `close_case`, `remove_earbud`.

`first/second` tương ứng lắp từ 0 → 1 và 1 → 2 tai trong hộp; không cố định bên trái/phải. `remove_earbud` là chuyển động nhấc một tai ra, còn Fusion dùng YOLO để suy ra occupancy giảm 2→1 hay 1→0. Cần annotation lại từ video, không thể tự biến nhãn `insert_earbud` cũ thành hai mốc thời gian đúng.

Lệnh bắt đầu cho session bạn đang dùng:

```powershell
python scripts/annotate_actions.py --videos-dir data/earbud_actions/raw_videos/per1/02 --output data/earbud_actions/annotations_v2.csv --config configs/action_earbud_v2_config.json
```

Nếu CSV có nhãn cũ, chọn file mới. `--resume` chỉ dùng với CSV đã cùng bộ nhãn, và bỏ qua cả video đã có nhãn.

Chi tiết các thay đổi: [TOM_TAT_HOP_NHAT_REPO_20260922.md](TOM_TAT_HOP_NHAT_REPO_20260922.md).

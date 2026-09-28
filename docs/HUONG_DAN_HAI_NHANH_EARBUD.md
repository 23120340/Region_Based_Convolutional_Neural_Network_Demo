# Hai nhánh Earbud: bản tham chiếu và action chung

| | `main` | `feature/merge-insert-earbud-action` |
|---|---|---|
| LSTM | `idle`, `open_case`, `insert_first_earbud`, `insert_second_earbud`, `close_case` | `idle`, `open_case`, `insert_earbud`, `close_case` |
| Lần lắp | LSTM đúng lần + geometry đúng khe | LSTM nhận lắp; Fusion đếm khe đã xác nhận |
| CSV hai lần lắp | Hai nhãn khác nhau | Hai đoạn riêng, cùng nhãn insert |
| Checkpoint | `artifacts/action_model_pilot/best.pt` | `artifacts/action_model_insert_earbud/best.pt` |
| Log | `artifacts/events/earbud_v2.jsonl` | `artifacts/events/earbud_insert_earbud.jsonl` |

**Chỉ gộp nhãn LSTM, không gộp FSM.** Quy trình cả hai nhánh vẫn là mở hộp → tai thứ nhất → tai thứ hai → đóng nắp. YOLO/Fusion kiểm tra trái/phải, đúng khe, tháo tai và đóng sớm.

## Những gì đã xử lý

- Giữ lịch sử nhánh feature từ `af8c788`; đưa sửa main vào bằng merge, không reset/rebase.
- Khôi phục main về first/second theo CSV Git cũ, giữ nguyên video ID, mốc thời gian và split.
- Pilot best.pt đã bị train đè bốn lớp. Weights bị Git ignore nên đổi nhánh không khôi phục được weights. Đã bảo toàn bản bốn lớp ở thư mục riêng, train lại bản năm lớp từ cache/split cũ. Bản năm lớp này **không phải khôi phục byte-for-byte** checkpoint cũ.
- Không thay YOLO, video hoặc cache DINOv2. Launcher lấy đường dẫn model từ project profile; kiểm tra taxonomy/backbone trước khi khởi tạo encoder để báo nhầm model sớm.
- Giữ các sửa trước đây trên cả hai nhánh: ô xác nhận/lịch sử, rollback, embedding liên tục, lấy mẫu theo config train, YOLO theo camera config, bbox ≥0,35, action >0,5.
- Trên feature, insert chung trước bước mở được chuyển thành sự kiện lắp thứ nhất để FSM báo sai thứ tự, không phải lỗi nhãn không tồn tại.

## Đổi nhánh và chạy

Thoát camera đang chạy; mở PowerShell tại `G:\Internship\RBCNN_Demo`:

```powershell
git status --short
git switch main
.\run_hybrid.ps1 -Source 0 -Fullscreen
```

Thử action chung:

```powershell
git switch feature/merge-insert-earbud-action
.\run_hybrid.ps1 -Source 0 -Fullscreen
```

Lệnh Python cũ dùng được trên cả hai nhánh, profile cùng tên nhưng chọn weights riêng:

```powershell
.\.venv\Scripts\python.exe scripts/run_hybrid.py --project configs/projects/earbud_v2.json --source 0 --no-mirror --fullscreen
```

Nếu Git báo thay đổi cục bộ ngăn switch, commit/stash thay đổi mới của bạn; không dùng reset --hard/checkout -- để ép chuyển. Bản working tree lẫn taxonomy trước khi tách được giữ trong stash tên `safety: before separating earbud branches`; **không pop toàn bộ lên main** vì sẽ đưa nhãn chung trở lại.

## Năm bước train nếu sau này thêm dữ liệu

1. **Annotation:** main chọn riêng first/second. Feature có menu 0 idle, 1 open_case, 2 insert_earbud, 3 close_case; mỗi lần lắp vẫn gán một đoạn, cả hai chọn số 2. Không gộp cả thao tác lắp và chờ đứng yên thành đoạn dài. CSV hiện có 185 đoạn/35 video nên không cần gán lại chỉ vì tách nhánh.

   ```powershell
   .\.venv\Scripts\python.exe scripts/annotate_actions.py --session 01 --session 02 --output data/earbud_actions/annotations_v2.csv --config configs/action_earbud_pilot_config.json --resume
   ```

2. **Split:** giữ `annotations_v2_pilot_split.csv` (25 train, 5 val, 5 test) để so sánh. Video/session mới cần chia riêng, không chia cửa sổ cùng video qua nhiều split.
3. **DINOv2:** cache `features_v2_pilot/` dùng chung, backbone `facebook/dinov2-small`, 384 chiều, 10 FPS. Đổi nhãn không đổi vector; chỉ trích thêm video mới hoặc trích lại khi đổi backbone/nhịp.
4. **Train:** chọn đúng thư mục theo nhánh; tuyệt đối không train bốn lớp đè lên `action_model_pilot` nữa.

   ```powershell
   # main
   $actionOutput = 'artifacts/action_model_pilot'
   # trên feature thay dòng trên bằng:
   # $actionOutput = 'artifacts/action_model_insert_earbud'
   .\.venv\Scripts\python.exe scripts/train_action_model.py --annotations data/earbud_actions/annotations_v2_pilot_split.csv --features-dir data/earbud_actions/features_v2_pilot --config configs/action_earbud_pilot_config.json --output $actionOutput --allow-same-session
   ```

5. **Evaluate:** dùng cùng `$actionOutput` và CSV/config thuộc nhánh đang mở.

   ```powershell
   .\.venv\Scripts\python.exe scripts/evaluate_action_model.py --checkpoint "$actionOutput/best.pt" --annotations data/earbud_actions/annotations_v2_pilot_split.csv --features-dir data/earbud_actions/features_v2_pilot --config configs/action_earbud_pilot_config.json --split test --output "$actionOutput/evaluation_test.json" --allow-same-session
   ```

Không dùng `artifacts/action_model/best.pt` baseline Google ViT/pick_case cho profile DINOv2. `--allow-same-session` chỉ cho pilot, không phải đánh giá tổng quát hóa.

## Kết quả xác minh

| Model | Val macro-F1 (67 windows) | Test macro-F1 (57 windows) | Test accuracy |
|---|---|---|---|
| Năm lớp tái train | 0,9675 | 0,9040 | 92,98% |
| Bốn lớp đã train, đánh giá lại | 1,0000 | 0,9243 | 94,74% |

Main đạt 129 tests, feature 135 tests trong staging. Hai taxonomy khác nhau nên chênh lệch F1 không chứng minh phương án nào tốt hơn tổng thể. Đây là action model trên cùng dữ liệu pilot, không phải độ chính xác YOLO/Fusion/FSM hoặc camera người mới.

## Việc bạn cần thử bằng camera

- Trên feature, thử phải trước/trái sau, rồi lượt mới trái trước/phải sau. LSTM đều là insert chung; bảng quy trình vẫn xác nhận lần thứ nhất/thứ hai.
- Giữ một tai đứng yên sau PASS: không được tự tính thành hai tai dù cửa sổ LSTM còn frame cũ.
- Thử sai khe, đóng sớm, tháo một/cả hai tai và lắp lại. Phải báo lỗi/lùi bước, chỉ hoàn tất khi đủ tai.
- Tay che hoặc mất bbox không phải bằng chứng tháo. Tai lơ lửng trên khe vẫn có thể gây nhầm với camera 2D.
- Gộp nhãn **không tự sửa idle**. Cần clip đứng yên, tay di chuyển nhưng không lắp, mở hộp, thứ tự lắp đa dạng và session/người mới để đánh giá.

Hiện PyTorch CPU-only: 10 FPS là mục tiêu, không bảo đảm FPS thực tế. Không tự push GitHub trong lần tách nhánh này.

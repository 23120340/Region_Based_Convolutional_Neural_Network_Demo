# Reliable Generic Insert — Báo cáo triển khai

## Kiến trúc sau tái cấu trúc

```text
Camera/video
  ├─ Scheduler deadline tích lũy 10 FPS → DINOv2-small → BiLSTM 4 nhãn
  └─ YOLO 6 lớp geometry ───────────────┐
                                        ↓
                          Temporal Evidence Fusion
                         0.25s insert / 0.5s wrong
                         0.8s removal + khe nhìn rõ
                                        ↓
                     FSM: WAIT_FOR_OPEN → 1 tai → 2 tai → DONE
                         UNKNOWN/INSUFFICIENT ≠ VIOLATION
```

## Quyền quyết định

- LSTM: chỉ nhận diện cử động chung `idle/open_case/insert_earbud/close_case`.
- YOLO + Fusion: nhận dạng tai vật lý trái/phải, khe tương ứng, đúng/sai bên và occupancy 0/1/2.
- FSM: chuyển `insert_first_earbud`/`insert_second_earbud` dựa trên số slot đã được Fusion xác nhận; đây không phải nhãn LSTM.

## Ngưỡng vận hành

Nguồn chuẩn là `configs/earbud_temporal_thresholds.json`: display >0.5, occupied ≥0.5, wrong-side ≥0.58, empty >0.5; insert 0.25 giây, wrong-side 0.5 giây, removal 0.8 giây. Mất bbox hoặc tay che chỉ tạo `UNKNOWN`; không rollback FSM.

Khởi động khi hộp đóng tạo `WAIT_FOR_OPEN`, không tạo violation. `--test-start-open` cho phép lấy một hộp mở làm baseline bằng sự kiện INFO `initialize_open_case`; production mặc định vẫn bắt buộc quan sát `open_case`.

## Kiểm thử tự động

- Unit scheduler nguồn 12.49, 15 và 30.0008 FPS: cadence embedding gần 10 FPS, deadline không trôi pha.
- Integration đa FPS: cùng một timeline vật lý ở cả ba source FPS phải sinh đúng bốn PASS theo cùng thứ tự.
- Fusion: ngưỡng confidence, insert/wrong/removal theo thời gian, occlusion → UNKNOWN, test baseline, tracking hộp dịch chuyển.
- Integration: mở hộp → lắp hai tai → che/mất bbox → tháo thật → lắp lại → đóng; kiểm tra JSONL PASS/VIOLATION.
- Holdout: manifest khóa tên và SHA-256 của hai clip hiện tại; tool annotation/extract/split/train từ chối dữ liệu holdout.

## Kết quả holdout với checkpoint hiện có

Chạy CPU, YOLO mỗi frame, DINO scheduler 10 FPS và LSTM generic hiện có:

| Clip | Sự kiện được ghi nhận | Kết luận |
|---|---|---|
| `insert_test_1.mp4` | PASS `open_case`; không có insert/violation giả | Logic mới yên hơn nhưng YOLO geometry chưa giữ đủ bằng chứng tai phủ khe trong clip này. Cần fine-tune bằng failure case cùng camera/góc/ánh sáng. |
| `insert_test_wrong.mp4` | PASS `open_case`, first insert, second insert, close | Clip này trước đây được xác định là bản quay lỗi, nên không dùng làm ground-truth “sai khe”. Nó chỉ chứng minh pipeline hoàn thành được một chu trình mà không rollback giả. |

Không điều chỉnh ngưỡng để ép `insert_test_1` PASS vì như vậy sẽ làm mất ý nghĩa holdout. Bước sửa đúng là bổ sung dữ liệu YOLO, đặc biệt cảnh tai đã nằm trong khe nhưng detector vẫn trả `empty_left/empty_right`, rồi đánh giá lại trên chính hai clip không tham gia train.

Môi trường `.venv` hiện dùng PyTorch CPU-only; `--device 0` báo `Torch not compiled with CUDA enabled`. Muốn đo FPS GPU phải cài wheel PyTorch CUDA phù hợp driver rồi xác nhận `torch.cuda.is_available() == True`.

## Chạy

```powershell
.\.venv\Scripts\python.exe scripts/run_hybrid.py `
  --project configs/projects/earbud_v2.json --source 0 --no-mirror --fullscreen
```

Chỉ dùng với video test bắt đầu khi hộp đã mở:

```powershell
.\.venv\Scripts\python.exe scripts/run_hybrid.py `
  --project configs/projects/earbud_v2.json --source tests/insert_test_1.mp4 `
  --test-start-open --headless
```

Hai clip hiện tại chỉ là regression suite. Cần bổ sung một clip đúng hoàn chỉnh và một clip tháo một tai thật vào manifest trước khi dùng kết quả làm tiêu chí phát hành.

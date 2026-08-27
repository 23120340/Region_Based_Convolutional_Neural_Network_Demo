# Pen Assembly Monitor — MVP trước Hybrid ViT + LSTM

Đây là mô hình thu nhỏ để kiểm chứng **logic giám sát quy trình lắp ráp** trước khi có video, camera cố định và thiết bị PCB. MVP không giả vờ rằng một LSTM chưa huấn luyện có thể nhận diện hành động: đầu vào hiện tại là thao tác mô phỏng trên giao diện; đầu ra đi qua đúng pipeline `prediction -> debouncer -> FSM -> event log` sẽ dùng cho model thật sau này.

## MVP đã làm được gì?

- Theo dõi 5 bước: đặt thân bút → lắp ruột → lắp lò xo → vặn nắp → bấm thử.
- Cảnh báo khi bỏ bước hoặc làm sai thứ tự; lỗi không làm FSM nhảy sang trạng thái mới.
- Lọc dự đoán chập chờn theo cửa sổ 5 mẫu và chỉ phát một sự kiện cho một hành động ổn định.
- Hiển thị checklist, trạng thái, bước tiếp theo và lịch sử sự kiện trên GUI.
- Ghi bằng chứng dạng JSONL tại `artifacts/events.jsonl`.
- Toàn bộ quy trình nằm trong `configs/pen_fsm_config.json`, không hard-code riêng cho bút trong engine.

## Chạy nhanh

Yêu cầu duy nhất cho MVP là Python 3.10+; không cần cài PyTorch.

```powershell
python scripts/run_demo.py
```

Trên giao diện, bấm từng hành động để giả lập kết quả nhận diện từ camera. Có thể chạy sẵn các kịch bản **Đúng quy trình**, **Quên lò xo**, **Quên ruột** hoặc **Bấm thử quá sớm**.

Chạy không cần giao diện:

```powershell
python scripts/simulate.py --scenario correct
python scripts/simulate.py --scenario missing_spring
python scripts/simulate.py --scenario missing_refill
python scripts/simulate.py --scenario premature_test
```

Chạy kiểm thử:

```powershell
python -m unittest discover -s tests -v
```

## Chạy nhận diện trực tiếp bằng camera laptop

Cài backend camera (trên máy hiện tại đã cài):

```powershell
python -m pip install -r requirements-camera.txt
```

Chạy camera số 0:

```powershell
python scripts/run_camera.py --source 0
```

Lần chạy đầu, chương trình tải checkpoint `yolov8s-worldv2.pt`. Đặt bốn khay linh kiện ngoài khung vàng và đưa linh kiện đang thao tác vào **WORK ZONE**. Các phím điều khiển:

- `Space`: xác nhận hành động được detector gợi ý.
- `1`–`5`: xác nhận thủ công từng bước từ đặt thân đến bấm thử.
- `R`: reset chu trình; `S`: chụp ảnh bằng chứng; `Q`: thoát.

Có thể thử tự chuyển bốn bước dựa trên sự hiện diện của linh kiện:

```powershell
python scripts/run_camera.py --source 0 --auto-advance
```

`--auto-advance` chỉ là baseline thử nghiệm. Detector biết **vật gì đang hiện diện**, nhưng không thể chứng minh thao tác “đã cắm”, “đã vặn” hay “đã bấm”; vì vậy chế độ có Space là mặc định. Xem [docs/CAMERA_REALTIME.md](docs/CAMERA_REALTIME.md) để hiệu chỉnh, thu dataset và train model riêng.

## Bố trí mô hình vật lý nhỏ

Đặt camera top-down và đánh dấu 5 vùng trên một tờ A4:

```text
[Thân bút] [Ruột bút] [Lò xo] [Nắp bút]

             [Vùng lắp ráp]
```

Mỗi lượt quay cần giữ nguyên góc máy. Giai đoạn đầu nên quay 40 lượt đúng và ít nhất 20 lượt sai có chủ đích, chia train/validation/test **theo người hoặc video**, không chia ngẫu nhiên từng frame.

## Ranh giới giữa MVP và Hybrid ViT + LSTM

```text
MVP mô phỏng
Nút mô phỏng -> TemporalDebouncer -> ConfigurableAssemblyTracker -> JSONL/UI

Camera zero-shot hiện tại
Webcam -> YOLO-World boxes -> work-zone dwell -> xác nhận -> cùng FSM/log

Khi có dữ liệu
Camera -> ViT embedding -> LSTM action classifier -> TemporalDebouncer -> cùng FSM/UI
```

`src/pen_assembly/model_contract.py` là hợp đồng tích hợp action model. Model tương lai chỉ cần trả `Prediction(action, confidence)`; engine, cấu hình, UI và log không phải viết lại.

Chỉ chuyển sang huấn luyện Hybrid ViT + LSTM sau khi:

1. FSM vượt toàn bộ test kịch bản.
2. Quy trình lắp bút và bộ nhãn đã được chốt.
3. Có video đủ đa dạng và nhãn theo đoạn thời gian.
4. Baseline trên tập test tách theo người đạt ngưỡng đã thống nhất (đề xuất Macro-F1 ≥ 0,85).

Xem [docs/MVP_PEN_ASSEMBLY.md](docs/MVP_PEN_ASSEMBLY.md) để biết tiêu chí nghiệm thu và lộ trình chuyển sang camera/model thật.

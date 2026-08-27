# Phạm vi triển khai MVP giám sát lắp ráp bút bi

## 1. Mục tiêu của vòng kiểm thử đầu tiên

Vòng này trả lời câu hỏi: **logic giám sát quy trình có đúng và có thể tái sử dụng không?** Nó chưa trả lời câu hỏi độ chính xác của ViT + LSTM vì hiện chưa có tập video gán nhãn để huấn luyện và đánh giá.

Đầu vào kiểm thử logic là các nút/kịch bản mô phỏng dự đoán hành động. Chế độ camera zero-shot hiện đã được bổ sung để khoanh vùng linh kiện và đề xuất bước theo WORK ZONE; xem `docs/CAMERA_REALTIME.md`.

## 2. Tiêu chí nghiệm thu MVP

| Mã | Tiêu chí | Cách kiểm tra |
|---|---|---|
| MVP-01 | Chuỗi đúng đi qua đủ 5 trạng thái và kết thúc ở `S5_TEST_CLICKED` | Kịch bản `correct` |
| MVP-02 | Quên ruột bị cảnh báo, FSM không đổi state | Kịch bản `missing_refill` |
| MVP-03 | Quên lò xo bị cảnh báo, FSM không đổi state | Kịch bản `missing_spring` |
| MVP-04 | Bấm thử trước khi vặn nắp bị cảnh báo | Kịch bản `premature_test` |
| MVP-05 | Một hành động kéo dài chỉ sinh một event | Unit test debouncer |
| MVP-06 | Có thể bắt đầu cây bút tiếp theo mà không khởi động lại app | Unit test cycle 2 |
| MVP-07 | Mỗi PASS/VIOLATION được lưu dưới dạng JSONL | Unit test monitor |

## 3. Cấu trúc hiện tại

```text
configs/pen_fsm_config.json       Quy trình và thông báo lỗi
src/pen_assembly/config.py        Nạp và kiểm tra cấu hình
src/pen_assembly/smoother.py      Majority vote + event latch
src/pen_assembly/fsm.py           Engine kiểm tra thứ tự
src/pen_assembly/monitor.py       Ghép pipeline và ghi JSONL
src/pen_assembly/gui.py           Dashboard mô phỏng
src/pen_assembly/model_contract.py Hợp đồng cho model tương lai
models/pen_action_net.py          Khung BiLSTM tùy chọn cho giai đoạn 2
scripts/simulate.py               Kiểm thử kịch bản headless
scripts/run_demo.py               Chạy GUI
tests/                            Unit/integration tests
```

## 4. Bước phát triển tiếp theo khi có camera và bút

1. Khóa cứng góc camera, ánh sáng, ROI và vị trí bốn khay linh kiện.
2. Quay lượt đúng và sai; gán nhãn đoạn `[video, start_frame, end_frame, action]`.
3. Tách dữ liệu theo người/video để tránh rò rỉ dữ liệu.
4. Trích ViT embedding theo frame và cache ra đĩa.
5. Huấn luyện LSTM, báo cáo confusion matrix và Macro-F1 trên test set độc lập.
6. Viết adapter thực thi `ActionRecognizer.predict()` rồi đưa `Prediction` vào `AssemblyMonitor`.
7. So sánh lỗi end-to-end với lỗi riêng của model; tinh chỉnh ngưỡng debouncer bằng validation set, không bằng test set.

Không được bật camera với trọng số LSTM ngẫu nhiên và gọi đó là kết quả nhận diện. Nếu chưa có checkpoint đã đánh giá, GUI phải tiếp tục ghi rõ là chế độ mô phỏng.

## 5. Điều kiện chuyển sang PCB hoặc thiết bị khác

Chỉ cần giữ nguyên engine và thay ba phần:

- cấu hình action/state/violation;
- tập video và taxonomy nhãn;
- checkpoint của action recognizer.

Nên chuyển bài toán sau khi pen MVP đạt toàn bộ tiêu chí trên, model nhận diện đạt ngưỡng đã chốt trên người chưa xuất hiện trong train, và kiểm thử trực tiếp cho thấy cảnh báo sai ở mức chấp nhận được.

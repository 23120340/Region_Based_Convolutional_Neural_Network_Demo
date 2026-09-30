# Đánh giá hai video kiểm thử tai nghe — cập nhật 29/09/2026

## Kết luận ngắn

`insert_test_1.mp4` là video quy trình đúng và thay thế hoàn toàn `insert_test_0.mp4` trong lần đánh giá cũ. `insert_test_wrong.mp4` là video lắp sai bên.

Hệ thống hiện tại trên nhánh `main` **chưa vượt qua video chuẩn**: chỉ xác nhận bước mở hộp, không xác nhận được hai lần lắp và không hoàn tất chu trình. Nguyên nhân chính gồm:

1. BiLSTM 5 nhãn gần như dự đoán `insert_second_earbud` trong toàn bộ đoạn lắp, kể cả khi FSM đang chờ lần lắp thứ nhất.
2. YOLO đổi nhãn tai trái/phải và có lúc nhận khe đang chứa tai thành `empty_left`/`empty_right`.
3. Fusion tin vào ba frame empty liên tiếp để rollback tai đã lắp, nên false-positive của YOLO biến thành VIOLATION tháo tai.
4. Runtime chỉ lấy được khoảng 7,5 embedding/giây từ video 30 FPS thay vì mục tiêu 10 FPS do cách cập nhật deadline.
5. Một hộp đóng ở đầu video bị coi là hành động đóng nắp sai quy trình, dù đây chỉ là trạng thái khởi đầu trước khi mở hộp.

Phép đối chứng bằng checkpoint LSTM 4 nhãn `insert_earbud` chung đã xác nhận được lần lắp thứ nhất trên video chuẩn. Tuy nhiên nó vẫn thất bại sau đó vì YOLO/fusion báo khe trống giả. Vì vậy hướng đúng là **dùng LSTM 4 nhãn chung và đồng thời sửa perception/fusion**, không chỉ đổi checkpoint.

## 1. Phạm vi đánh giá mới

- Repo: `G:\Internship\RBCNN_Demo`.
- Nhánh đang chạy: `main`, revision `5a8112d`.
- Trạng thái nhánh lúc kiểm thử: `main` ahead 2, behind 1 so với `origin/main`; chưa pull, merge, commit hay push.
- Video chuẩn: `tests/insert_test_1.mp4`, 873 frame, 30.0004 FPS, 29.10 giây, 720 × 1280.
- Video sai bên: `tests/insert_test_wrong.mp4`, 974 frame, 30.0008 FPS, 32.47 giây, 720 × 1280.
- Chạy đúng pipeline của `scripts/run_hybrid.py` với YOLO + DINOv2 + BiLSTM + fusion + FSM ở chế độ headless, CPU, offline và không mirror.
- Không sửa dự đoán bằng dữ liệu giả. Instrumentation chỉ ghi lại frame, bbox, LSTM, trạng thái slot và sự kiện.
- Hai video chưa có annotation theo từng mốc thời gian. “Đúng/sai” là nhãn tình huống do người quay cung cấp, nên báo cáo không tính accuracy/F1 từ hai clip này.

### Model đang được `main` sử dụng

| Thành phần | Cấu hình |
| --- | --- |
| YOLO | `artifacts/training/earbud_geometry_detector/weights/best.pt`; 6 lớp; input 512 |
| DINOv2 | `facebook/dinov2-small`; embedding 384 |
| BiLSTM | `artifacts/action_model_pilot/best.pt` |
| 5 nhãn LSTM | `idle`, `open_case`, `insert_first_earbud`, `insert_second_earbud`, `close_case` |
| Điều kiện fusion | action > 0.5; empty > 0.5; coverage >= 40%; ổn định 3 frame YOLO |

SHA256:

```text
YOLO:      7c7a2d1a6610737c57b2dddddf6bde0f987ae9d764be6bc876d3da1fed757843
LSTM 5:    2eaa1a6e8122d6646217a044e5bf05eb7b8ad2d272db7725649a0911f24c1166
LSTM 4:    abcc74831f3b00fa3002b8c8fd60aa5f2f2abac7fbed225a2b1efc6ad09a5ee8
```

## 2. Kết quả chính với model 5 nhãn của `main`

| Video | PASS | VIOLATION | Trạng thái cuối |
| --- | --- | --- | --- |
| `insert_test_1` — chuẩn | `open_case` | `close_case` đầu clip, `wrong_earbud_side`, `insert_second_earbud` sai thứ tự, `close_case` cuối clip | `S1_CASE_READY` |
| `insert_test_wrong` — sai bên | `open_case` | 12 lần `insert_second_earbud` sai thứ tự, 3 lần `wrong_earbud_side`, 1 lần `close_case` | `S1_CASE_READY` |

Không video nào đạt `S4_COMPLETED`. Số lượng VIOLATION ở clip sai bên không phải 16 lỗi độc lập: phần lớn là cùng một dự đoán `insert_second_earbud` bị phát lặp lại khi geometry dao động.

### Video chuẩn `insert_test_1.mp4`

| Thời gian video | Kết quả | Đánh giá |
| --- | --- | --- |
| 3.867 s | VIOLATION `close_case` | Hộp đang đóng ở trạng thái ban đầu. Đây nên là trạng thái setup/chờ mở, không nên quy thành người dùng đóng nắp sai. |
| 5.067 s | PASS `open_case` | Đúng: YOLO thấy hai khe rỗng và LSTM nhận mở hộp. |
| 12.267 s | VIOLATION `wrong_earbud_side` | Người dùng xác nhận video làm đúng; YOLO gọi tai trong khe trái là `right_earbud` chỉ với confidence khoảng 0.44. Đây là lỗi nhận dạng trái/phải, không phải bằng chứng chắc chắn người dùng lắp sai. |
| 22.233 s | VIOLATION `insert_second_earbud` | Geometry thấy tai đúng khe nhưng FSM chờ lần thứ nhất; LSTM lại trả `insert_second_earbud` khoảng 98.8%. |
| 24.533 s | VIOLATION `close_case` | Bị coi đóng quá sớm vì cả hai lần lắp chưa từng được xác nhận. |

Phân bố 204 lần phân loại sau khi đủ 16 embedding:

```text
idle=21, open_case=21, insert_first_earbud=20,
insert_second_earbud=113, close_case=29
```

Model trả `insert_second_earbud` kéo dài trong các cảnh 12–24 giây, bao gồm cả khi mới lắp tai đầu tiên, khi đã có hai tai và trước lúc đóng nắp. Confidence cao không làm dự đoán này đúng hơn.

YOLO cũng không duy trì trạng thái occupied: tại khoảng 21 và 24 giây, hình có tai trong hộp nhưng các bbox lại chủ yếu là `empty_left`/`empty_right`. Đây là lý do geometry không thể cung cấp đủ hai lần xác nhận ổn định.

### Video sai bên `insert_test_wrong.mp4`

- PASS đúng bước mở hộp ở 3.467 giây.
- Từ 10.5 giây, geometry nhiều lần thấy tai phủ khe, nhưng LSTM gần như luôn trả `insert_second_earbud`; FSM vẫn chờ `insert_first_earbud`, nên phát cảnh báo sai thứ tự lặp lại.
- YOLO có phát hiện sai bên ở 20.599, 21.533 và 22.866 giây, nhưng xen giữa lại đổi nhãn sang occupied đúng bên. Điều này chứng minh nhãn trái/phải chưa ổn định đủ để quyết định PASS/VIOLATION sau chỉ ba frame.
- Model không hề dự đoán `insert_first_earbud` trong clip này:

```text
idle=27, open_case=18, insert_first_earbud=0,
insert_second_earbud=177, close_case=7
```

FSM bắt được tình huống sai bên, nhưng hiện chưa thể coi kết quả là đáng tin cậy vì cùng một cảnh liên tục đổi giữa đúng bên và sai bên.

## 3. Phép đối chứng LSTM 4 nhãn trên video chuẩn

Phép đối chứng giữ nguyên code `main`, YOLO, fusion và FSM; chỉ thay action config/checkpoint bằng model:

```text
idle, open_case, insert_earbud, close_case
```

| Model | PASS trên video chuẩn | Kết quả cuối |
| --- | --- | --- |
| LSTM 5 nhãn | `open_case` | Không xác nhận tai nào |
| LSTM 4 nhãn | `open_case`, `insert_first_earbud` | Sau đó bị rollback về 0 do empty giả |

Timeline đối chứng đáng chú ý:

| Thời gian | Sự kiện |
| --- | --- |
| 4.933 s | PASS mở hộp |
| 12.267 s | YOLO báo sai bên giả; LSTM 4 nhãn vẫn nhận `insert_earbud` 93.7% |
| 22.233 s | PASS lần lắp thứ nhất; LSTM `insert_earbud` 94.1%, tai phủ khe phải khoảng 76% |
| 22.800 s | YOLO chỉ còn hai bbox empty; fusion báo tháo tai và rollback từ 1 về 0, dù người dùng không thực hiện bước tháo |
| 24.400 s | Đóng nắp bị từ chối vì trạng thái đã rollback |

Kết luận của phép đối chứng:

- Gộp `insert_first_earbud` và `insert_second_earbud` thành `insert_earbud` loại bỏ đúng nút thắt “LSTM phải biết đây là lần thứ mấy”.
- YOLO/fusion vẫn là nút thắt còn lại. Không nên nói rằng chỉ chuyển sang model 4 nhãn là hệ thống đã chạy đúng.
- “Tai thứ nhất/thứ hai” vẫn được hiển thị trên UI/FSM như **tiến độ đã xác nhận**, nhưng không còn là hai lớp cần LSTM phân biệt.

## 4. Nguyên nhân kỹ thuật

### 4.1. Thiết kế 5 nhãn đang học tương quan sai

Các video huấn luyện đều là scenario đúng và có thứ tự thao tác gần như cố định. `insert_first_earbud`/`insert_second_earbud` vì thế dễ bị học thành “hình dáng/vị trí tai bên nào” hoặc “hộp đang có bao nhiêu tai”, thay vì hành động lắp lần thứ nhất/thứ hai. Video mới chỉ cần đổi bên lắp trước là mô hình suy ra sai thứ tự.

LSTM nên trả lời câu hỏi: **người dùng có đang thực hiện thao tác lắp tai không?** YOLO/fusion nên trả lời: **tai nào, vào khe nào, đây là lần xác nhận thứ mấy?**

### 4.2. Nhãn YOLO trái/phải và empty chưa ổn định

- Cùng một tai có thể đổi giữa `left_earbud` và `right_earbud` theo góc nghiêng.
- Khe đã chứa tai vẫn có thể được dự đoán là empty trên 0.5.
- Điện thoại trên bàn có lúc bị gọi là `open_case`, cho thấy cần khóa/tracking hộp mục tiêu thay vì luôn chọn bbox case confidence cao nhất.
- Ngưỡng confidence 0.5 đã hoạt động đúng theo config, nhưng tăng ngưỡng không giải quyết được dự đoán sai có confidence cao.

### 4.3. Fusion biến nhiễu ba frame thành sự kiện thật

Hiện tại ba frame `empty` liên tiếp đủ để kết luận tháo tai. Ba frame ở nguồn 30 FPS chỉ khoảng 0.1 giây. Sai nhãn ngắn do tay che, nghiêng hộp hoặc blur vì thế có thể rollback FSM.

Tương tự, `wrong_side` chưa được latch theo một “episode” ổn định; khi nhãn YOLO đổi, hệ thống vừa có thể báo sai bên vừa quay lại chờ/PASS, gây thông báo mâu thuẫn và lặp.

### 4.4. Cadence runtime lệch cadence train

Cả hai clip gần 30 FPS chỉ đạt khoảng 7.5 embedding/giây dù mục tiêu là 10. `scripts/run_hybrid.py` đang dùng:

```python
next_sample_time = now + sample_interval
```

Trong khi extractor khi train cộng dồn deadline. Với video 30.0004 FPS, 3 frame hơi ngắn hơn 0.1 giây nên runtime luôn đợi frame thứ tư. Cửa sổ 16 embedding kéo dài gần 2 giây thay vì khoảng 1.5 giây giữa frame đầu/cuối.

### 4.5. FSM đang trộn “trạng thái” với “hành động”

Một bbox `close_case` ổn định chỉ chứng minh hộp đang đóng, không chứng minh người dùng vừa thực hiện hành động đóng. Khi chương trình bắt đầu và hộp vốn đã đóng, hệ thống phải hiển thị “đang chờ mở hộp”, không phát VIOLATION đóng sớm.

## 5. Định hướng sửa đề xuất

### Giai đoạn 1 — sửa nền tảng, chưa cần train lại

1. Chuyển nhánh phát triển sang LSTM 4 nhãn với `insert_earbud` chung; geometry/FSM sinh `insert_first_earbud` hoặc `insert_second_earbud` theo số slot đã xác nhận.
2. Sửa scheduler thành deadline cộng dồn và bổ sung test cho nguồn 12.49, 15, 29.9986 và 30.0008 FPS.
3. Thêm trạng thái setup `WAIT_FOR_OPEN`: hộp đóng trước khi từng thấy hộp mở là INFO, không phải VIOLATION.
4. Debounce sự kiện theo episode: một lỗi sai thứ tự/sai bên chỉ phát một lần cho tới khi scene thay đổi rõ ràng.
5. Giữ trạng thái `UNKNOWN/OCCLUDED` riêng. Mất bbox tai hoặc có bbox empty trong lúc tay che không được suy thành tháo tai.

### Giai đoạn 2 — làm chắc fusion

1. Track đúng một bbox hộp sau khi khởi tạo; bỏ bbox case giả ở điện thoại/nền.
2. Khi bbox tai và empty xung đột cùng khe, ưu tiên trạng thái `UNKNOWN` cho đến khi bằng chứng ổn định; không cập nhật anchor bằng frame bị che.
3. Không rollback chỉ từ ba frame empty. Removal nên cần đồng thời:
   - slot nhìn rõ, tay đã rời vùng;
   - empty confidence cao và tồn tại theo thời gian, không chỉ theo số frame;
   - không có bbox tai phủ slot;
   - có chuyển đổi từ occupied sang empty sau một thao tác/motion hợp lệ.
4. Sau `wrong_side`, giữ latch cho đến khi thấy tai bị lấy ra rồi xuất hiện lại đúng khe ổn định. Không tự xóa lỗi chỉ vì class YOLO đổi trong vài frame.

### Giai đoạn 3 — sửa và bổ sung dataset YOLO

Thu thêm từ đúng camera, góc quay và hộp hiện tại:

- hộp rỗng, chỉ tai trái, chỉ tai phải, đủ hai tai;
- mỗi tai đúng khe và sai khe;
- tay che, hộp nghiêng, motion blur, xa/gần, ánh sáng mạnh/yếu;
- các hard negative: điện thoại, hộp đóng, mặt bàn, dây trang trí;
- chuỗi tháo một tai thật sau khi đã lắp đủ.

Quy tắc nhãn:

- left/right theo **tai vật lý**, không theo vị trí trái/phải trên màn hình;
- chỉ gán `empty_left/right` khi khe thực sự trống;
- không gán đồng thời một vùng vừa occupied vừa empty;
- rà soát bbox quá lớn, bbox trùng và bbox 1–2 pixel trước khi train;
- chia train/val/test theo buổi quay hoặc video, không chia các frame gần nhau sang nhiều tập.

Nếu hai tai quá giống nhau khiến YOLO khó phân biệt ổn định, bản prototype nên dán marker màu nhỏ khác nhau cho trái/phải. Đây là cách kiểm chứng logic hệ thống trước khi đầu tư dataset lớn.

### Giai đoạn 4 — regression test trước khi quay lại camera realtime

Dùng ít nhất bốn video giữ riêng, không đưa vào train:

1. Quy trình đúng, lắp trái trước.
2. Quy trình đúng, lắp phải trước.
3. Lắp sai bên rồi sửa lại.
4. Lắp đủ rồi tháo một tai.

Tiêu chí tối thiểu:

- Hai video đúng đều có đúng 4 PASS và 0 VIOLATION.
- Video sai bên có một violation cho mỗi episode, không PASS slot sai.
- Video tháo tai chỉ rollback sau khi tai thực sự rời khe.
- Không có `open_case` giả từ điện thoại và không phát lặp cùng một lỗi mỗi vài frame.

## 6. Quyết định cần thống nhất

Hướng mình khuyến nghị là:

> Dùng nhánh `feature/merge-insert-earbud-action` làm nền cho LSTM 4 nhãn, đưa các sửa lỗi scheduler/setup/fusion vào đó, rồi chỉ merge về `main` sau khi `insert_test_1` và `insert_test_wrong` đạt tiêu chí regression.

Không nên tiếp tục fine-tune LSTM 5 nhãn trên cùng dữ liệu thứ tự cố định. Nếu vẫn muốn giữ 5 nhãn, bắt buộc phải quay cân bằng cả hai thứ tự trái-trước/phải-trước và nhiều người/góc; nhưng nhiệm vụ “lần thứ nhất/lần thứ hai” vốn đã được FSM biết, nên đây là độ phức tạp không cần thiết.

Thứ tự thực hiện hợp lý:

1. Sửa code nền tảng và chạy lại hai video hiện có.
2. Xác định các frame YOLO sai còn lại và sửa annotation/dataset.
3. Fine-tune YOLO.
4. Chạy bộ bốn video regression.
5. Cuối cùng mới đánh giá realtime bằng camera laptop/điện thoại.

## 7. Bằng chứng và khả năng tái hiện

- [Timeline, bbox, hash và outcome của hai video với model 5 nhãn](../artifacts/evaluations/user_test_20260928/current_user_tests_evidence.json)
- [Ảnh bbox video chuẩn](../artifacts/evaluations/user_test_20260928/current_detections_insert_test_1.jpg)
- [Ảnh bbox video sai bên](../artifacts/evaluations/user_test_20260928/current_detections_insert_test_wrong.jpg)
- [Kết quả đối chứng model 4 nhãn](../artifacts/evaluations/user_test_20260928/generic_runtime_summary_insert_test_1.json)

Lệnh tái hiện model hiện tại trên `main`:

```powershell
$env:OMP_NUM_THREADS = '2'
$env:MKL_NUM_THREADS = '2'
$env:HF_HUB_OFFLINE = '1'
$env:TRANSFORMERS_OFFLINE = '1'

.\.venv\Scripts\python.exe scripts/run_hybrid.py --project configs/projects/earbud_v2.json --source tests/insert_test_1.mp4 --headless --no-mirror --device cpu --event-log artifacts/evaluations/manual_correct_run02.jsonl

.\.venv\Scripts\python.exe scripts/run_hybrid.py --project configs/projects/earbud_v2.json --source tests/insert_test_wrong.mp4 --headless --no-mirror --device cpu --event-log artifacts/evaluations/manual_wrong_run02.jsonl
```

Logger JSONL ghi nối, vì vậy mỗi lần chạy nên dùng tên log mới.

## 8. Thay đổi trong lần đánh giá này

- Thay video chuẩn cũ `insert_test_0` bằng `insert_test_1` trong kết luận.
- Chạy lại cả hai video bằng đúng checkpoint 5 nhãn của `main`.
- Chạy thêm đối chứng LSTM 4 nhãn trên video chuẩn.
- Cập nhật báo cáo này và thêm bằng chứng mới trong `artifacts/evaluations/user_test_20260928`.
- Không sửa source/config/model/annotation/video; không retrain; không commit, pull, merge hoặc push.

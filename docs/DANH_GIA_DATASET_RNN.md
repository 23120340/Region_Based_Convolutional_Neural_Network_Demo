# Đánh giá và ghép dataset RNN trái/phải

Cập nhật: 23/09/2026

## Kết luận chính

Thư mục `RNN/LR_Earbud` không cung cấp thêm ảnh mới cho `RNN`: cả **190/190 ảnh** có tên và SHA-256 giống ảnh đã tồn tại trong `RNN/train`. Label của hai nơi chỉ khác class ID vì dùng hai taxonomy khác nhau; sau khi ánh xạ tên lớp, nội dung annotation tương đương.

Không được copy thêm 190 ảnh này lần thứ hai. Làm vậy chỉ tăng trọng số của các frame trùng, có thể làm confidence trên cùng bối cảnh tăng giả tạo nhưng không giúp model nhận tai nghe mới tốt hơn.

## Dataset đã chuẩn hóa

Chạy:

```powershell
python scripts/merge_rnn_earbud_datasets.py
```

Output nằm tại `datasets/earbud_rnn_merged/`. Script ánh xạ bảy class về tên runtime, chuyển polygon thành bounding box, bỏ annotation suy biến, khử trùng SHA-256 và dùng hardlink để không nhân đôi ảnh trên ổ G.

Kết quả:

```text
Train: 436 ảnh, 776 boxes
Val:    75 ảnh, 197 boxes
Test:   33 ảnh,  86 boxes
Tổng:  544 ảnh, 1.059 boxes
Invalid label rows: 0
Trainable: YES
```

Phân bố toàn dataset:

```text
open_case:     261
close_case:     96
left_earbud:   216
right_earbud:  151
empty_left:     32
empty_right:    25
hand:          278
```

Hai lớp khe trống còn ít; test chỉ có 12 box tai trái và 5 box tai phải. Metric test từng phía vì vậy còn dao động mạnh.

## Train trên Kaggle

Import [`Kaggle_Training_Earbud_LR.ipynb`](../Kaggle_Training_Earbud_LR.ipynb), Add Input thư mục `earbud_rnn_merged`, sửa duy nhất:

```python
DATASET_ROOT = r'/kaggle/input/ten-dataset/earbud_rnn_merged'
```

Bật GPU, bật Internet và chọn **Run All**. Notebook sẽ xuất:

```text
/kaggle/working/best_earbud_lr_detector.pt
/kaggle/working/earbud_lr_training_results.zip
```

Chép checkpoint về `artifacts/training/earbud_geometry_detector/weights/best.pt`. Cấu hình camera hiện tại đã nhận `left_earbud`, `right_earbud` và kiểm tra đúng khe trái/phải.

## Có nhận được hai tai nghe khác không?

Không thể đảm bảo. Dataset hiện chủ yếu là cùng một mẫu tai nghe màu xanh, cùng mặt bàn và các frame gần nhau. YOLO closed-set không có khái niệm “không biết”; một tai nghe lạ vẫn có thể bị ép thành `left_earbud` hoặc `right_earbud` với confidence khó tin cậy.

Nếu muốn nhận nhiều loại tai nghe trái/phải:

- thu ít nhất 5–10 mẫu tai nghe khác nhau;
- chụp nhiều góc xoay, khoảng cách, ánh sáng và nền;
- chụp từng bên, cả hai bên cùng lúc, trong tay và trong hộp;
- chia train/val/test theo phiên quay hoặc mẫu tai nghe, không chia ngẫu nhiên frame cùng video;
- bổ sung mạnh `empty_left`, `empty_right` và trạng thái bị tay che.

Nếu chỉ chấp nhận đúng cặp tai nghe của hệ thống, cần thêm class `other_earbud` bằng các tai nghe khác làm hard negative, hoặc thêm tầng xác minh danh tính/embedding. Chỉ tăng ngưỡng confidence không đủ để loại vật thể ngoài dataset.

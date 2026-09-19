# Hướng Dẫn Huấn Luyện Mô Hình LSTM Nhận Diện Hành Động

Tài liệu này hướng dẫn bạn từng bước từ việc đưa video quay ngoài vào dự án, gán nhãn hành động (annotate), trích xuất đặc trưng không gian (spatial features) và cuối cùng là huấn luyện mô hình LSTM.

## 1. Nạp video gốc (Raw Videos)

Đưa tất cả các video bạn vừa quay vào thư mục mặc định của dự án để chuẩn bị gán nhãn:

- **Thư mục lưu trữ**: `data/earbud_actions/raw_videos`
- **Lưu ý tên file**: Đặt tên file theo cấu trúc `nguoi_phantu_...` (ví dụ: `per1_01_video1.mp4`) để hệ thống tự động nhận diện `person_id` (người thực hiện) và `session_id` (phiên thực hiện).

---

## 2. Gán nhãn video (Annotation)

Sau khi đã có video trong thư mục, bạn cần gán nhãn cho từng hành động (thời gian bắt đầu, thời gian kết thúc, loại hành động).

Chạy lệnh sau trong terminal:
```bash
python scripts/annotate_actions.py
```

**Cách thao tác trên cửa sổ video (OpenCV):**
- `SPACE`: Tạm dừng / Tiếp tục phát video.
- `s`: Đặt điểm **Bắt đầu** (Start) của một đoạn hành động.
- `e`: Đặt điểm **Kết thúc** (End) của đoạn hành động. (Sau khi bấm `e`, quay lại terminal để chọn nhãn tương ứng).
- `b` / `f`: Tua lại (back) 5 giây / Tua tới (forward) 5 giây.
- `q`: Bỏ qua video hiện tại và chuyển sang video tiếp theo.
- `ESC`: Thoát và lưu các nhãn đã gán.

> [!TIP]
> Kết quả gán nhãn sẽ được tự động lưu vào file `data/earbud_actions/annotations.csv`. Nếu sau này có video mới, bạn có thể chạy lại lệnh với cờ `--resume` để bỏ qua các video đã gán nhãn.

---

## 3. Phân chia tập dữ liệu (Train / Val / Test Split)

Bước tiếp theo là chia danh sách các đoạn hành động vừa gán nhãn thành 3 tập dữ liệu (huấn luyện, xác thực và kiểm thử).

Chạy lệnh sau:
```bash
python scripts/split_annotations.py
```

**Một số tùy chọn chia dữ liệu:**
- **Mặc định (Leave-one-out)**: Lệnh trên mặc định luân phiên để lại 1 video làm tập validation (val) và 1 video làm tập test, số video còn lại làm train. Cực kỳ hiệu quả nếu bạn có ít video (<6 video).
- **Tỷ lệ cố định (Ratio)**: Nếu số lượng video lớn, bạn có thể chia theo tỷ lệ (ví dụ: 70% train, 15% val, 15% test).
  ```bash
  python scripts/split_annotations.py --mode ratio --train 0.7 --val 0.15 --test 0.15
  ```

Lệnh này sẽ cập nhật trực tiếp cột `split` trong file `annotations.csv`.

---

## 4. Trích xuất đặc trưng (Feature Extraction)

Để mô hình LSTM có thể học hiệu quả, thay vì huấn luyện trực tiếp trên hình ảnh, hệ thống sẽ đi qua từng frame video và sử dụng mô hình ViT (Vision Transformer) để trích xuất ra các véc-tơ đặc trưng (embedding).

Chạy lệnh:
```bash
python scripts/extract_spatial_features.py
```

Quá trình này có thể mất một lúc tùy vào độ dài của video và sức mạnh của máy tính/GPU. 
> [!NOTE]
> Các file đặc trưng (`.npy` và `.json`) sẽ được lưu vào thư mục `data/earbud_actions/features/`.

---

## 5. Huấn luyện mô hình (Train LSTM)

Khi đã có dữ liệu nhãn (annotations) được chia tập rõ ràng và các file đặc trưng (features) đã được trích xuất, bạn có thể bắt đầu quá trình huấn luyện mô hình Action LSTM.

Chạy lệnh:
```bash
python scripts/train_action_model.py
```

**Trong quá trình này:**
- Hệ thống sẽ đọc dữ liệu từ `annotations.csv` và các đặc trưng ở `features/`.
- Quá trình training diễn ra, log lỗi (loss) và độ chính xác (F1-score) cho mỗi epoch sẽ được in ra ở terminal.
- Nếu bạn có GPU, mô hình sẽ tự động sử dụng CUDA để tăng tốc độ.
- Trọng số mô hình tốt nhất (best model) sẽ được tự động lưu lại vào file: `artifacts/action_model/best.pt`.

**Tùy chọn phụ**:
Nếu quá trình split dữ liệu ở Bước 3 làm các đoạn cắt trong cùng một video (session) nằm xen kẽ ở cả tập train và val/test, bạn có thể chạy với cờ `--allow-same-session` để bỏ qua lỗi cảnh báo rò rỉ dữ liệu (data leakage) của script huấn luyện.
```bash
python scripts/train_action_model.py --allow-same-session
```

---

## Tổng kết Pipeline

```mermaid
flowchart TD
    A[Video gốc (.mp4)] --> B{annotate_actions.py}
    B --> C(annotations.csv)
    C --> D{split_annotations.py}
    D --> E(annotations.csv đã cập nhật split)
    
    A --> F{extract_spatial_features.py}
    F --> G(Thư mục features/ chứa .npy)
    
    E --> H{train_action_model.py}
    G --> H
    H --> I(Trọng số mô hình: best.pt)
```

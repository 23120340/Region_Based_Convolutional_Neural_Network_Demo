# Chuẩn dữ liệu Earbud Reliable Pipeline

## Nguyên tắc không được phá vỡ

- Hai video `tests/insert_test_1.mp4` và `tests/insert_test_wrong.mp4` là **regression holdout**. Không gán nhãn để train, không trích DINO feature và không đưa frame của chúng vào YOLO/LSTM.
- Chia train/val/test theo **buổi quay**. Nếu chưa đủ session thì tối thiểu phải chia theo nguyên video; không chia ngẫu nhiên từng frame.
- Tai trái/phải được gán theo đặc tính vật lý của tai nghe, không theo bên trái/phải trên màn hình.
- Chỉ gán `empty_left`/`empty_right` khi lòng khe thật sự trống và nhìn rõ. Tay che, chói sáng hoặc bbox mâu thuẫn phải để không nhãn khe, không đoán `empty`.

## Taxonomy YOLO bắt buộc

`open_case`, `close_case`, `left_earbud`, `right_earbud`, `empty_left`, `empty_right`.

Mỗi session giữ cố định loại tai nghe, camera, tiêu cự và setup ánh sáng sản xuất. Giữa các session nên chủ động thay đổi nhẹ vị trí, độ nghiêng và người thao tác để đo khả năng tổng quát.

## Shot list cho mỗi session

Quay video liên tục, sau đó lấy frame 3–5 FPS; không chụp hàng chục frame gần như giống hệt nhau.

1. Hộp mở và hai khe trống, nhiều góc nghiêng hợp lệ.
2. Chỉ tai trái đúng khe trái; chỉ tai phải đúng khe phải.
3. Đủ hai tai đúng vị trí.
4. Tai trái cắm khe phải và tai phải cắm khe trái.
5. Tay che một phần lúc đưa tai vào/lấy tai ra.
6. Hộp đóng nắp.
7. Negative: bàn trống, điện thoại, găng tay, dây sạc và dụng cụ; ảnh negative có file label rỗng.

Mỗi trạng thái chính nên có ít nhất 100–200 ảnh **khác biệt thực sự** qua tối thiểu 3 session trước khi xem kết quả test là đáng tin. Nếu một lớp có recall thấp hoặc sai theo góc, bổ sung đúng failure case đó thay vì chỉ tăng ảnh nền.

## Manifest chống data leakage

Tạo CSV:

```csv
image_id,video_id,session_id,split
session01_clip03_f0042,session01_clip03,session01,train
session02_clip01_f0012,session02_clip01,session02,val
session03_clip02_f0090,session03_clip02,session03,test
```

Kiểm tra trước khi train:

```powershell
python scripts/validate_detection_dataset.py --data datasets/earbud/data.yaml
python scripts/validate_detection_split.py --manifest datasets/earbud/capture_manifest.csv
```

## LSTM bốn nhãn chung

LSTM chỉ học `idle`, `open_case`, `insert_earbud`, `close_case`. Không dùng `insert_first_earbud` hoặc `insert_second_earbud` làm target. Chuyển annotation cũ sang file mới, không ghi đè nguồn:

```powershell
python scripts/build_generic_action_annotations.py `
  --input data/earbud_actions/annotations.csv `
  --output data/earbud_actions/annotations_generic.csv
```

Sau đó split theo session, trích feature và train với `configs/action_earbud_generic_config.json`. Các script sẽ dừng nếu thấy ID hoặc SHA-256 của video holdout.

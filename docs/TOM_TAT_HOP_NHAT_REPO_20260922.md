# Tóm tắt hợp nhất repository — 22/09/2026

## Bản chính

Repository duy nhất tiếp tục sử dụng:

```text
G:\Internship\RBCNN_Demo
```

Đây là bản đầy đủ nhất và đúng hướng hiện tại: YOLO cho vật thể, DINOv2-Small + BiLSTM cho năm hành động, Fusion Engine và FSM cho quy trình hai tai nghe.

## Kết quả so sánh các thư mục

| Thư mục | Kết luận |
|---|---|
| `RBCNN_Demo` | Bản chính; có Git, video, dataset, artifact, code và test mới nhất |
| `Region_Based_Convolutional_Neural_Network_Demo-main` | Bản sao cũ; không có file riêng, đã chuyển vào Recycle Bin |
| `RBCNN_Demo_cleanup_backup_20260922_151804` | Backup tạm do quá trình dọn repo tạo ra; đã chuyển vào Recycle Bin |

So sánh theo đường dẫn tương đối cho thấy bản tên dài có 0 file riêng; bản chính có thêm 1.269 file. Các file cùng tên nhưng khác hash chủ yếu là code/config đã nâng cấp, metadata feature và video đã sửa timestamp. Bản chính cũng giữ backup video trước sửa tốc độ, vì vậy không mất dữ liệu gốc.

`AOI_PCB` và `Outside_fol_sp` là dữ liệu/dự án PCB riêng, không phải bản sao của earbud nên không được trộn hoặc xóa.

## Những phần đã hợp nhất và sửa

- Chuẩn hóa đường dẫn làm việc sang ổ G.
- Bỏ mặc định dataset/action của dự án cũ.
- Đổi class model dùng chung thành `AssemblyActionNet`.
- GUI và scenario lấy bước trực tiếp từ FSM, không hard-code sản phẩm.
- Action v2 ban đầu thống nhất năm nhãn, sau đó được mở rộng thành sáu nhãn:
  `idle`, `open_case`, `insert_first_earbud`,
  `insert_second_earbud`, `close_case`, `remove_earbud`.
- Chuyển backbone v2 sang `facebook/dinov2-small`, embedding 384 chiều.
- Đồng bộ project profile, FSM và Fusion cho `open_case` và hai lần lắp.
- Nâng cấp annotation, split, validation, train và evaluation; thêm kiểm tra pipeline sáu lớp.
- Tách checkpoint geometry mục tiêu khỏi detector baseline 6 lớp để tránh nạp nhầm.
- Xóa `camera_earbud_v2_config.json` cũ bị trùng; profile v2 dùng trực tiếp cấu hình geometry sáu lớp trái/phải.
- Viết lại README và tài liệu camera/action bằng link tương đối.

## Những file tài liệu đã loại

- `ASSEMBLY_TRACKER_PLAN.md`: kế hoạch chung cũ, nội dung đã được cô đọng trong tài liệu Hybrid.
- `docs/plan.md`: bản kế hoạch dài trùng mục tiêu và không còn phản ánh cấu hình hiện tại.
- `docs/TOM_TAT_THAY_DOI_THEO_DANH_GIA.md`: nhật ký cũ lỗi encoding và còn nội dung dự án không liên quan.
- `configs/camera_earbud_v2_config.json`: cấu hình sáu lớp trùng baseline, không đúng schema geometry v2.

Hai thư mục dư được chuyển vào Recycle Bin sau khi test đạt, nên vẫn có thể phục hồi cho đến khi Recycle Bin bị làm rỗng. Git cũng giữ lịch sử của file đã track.

## Phần người dùng còn phải tạo

- Annotation v2 thật.
- Feature DINOv2 v2.
- Checkpoint BiLSTM v2.
- Dataset geometry sáu lớp trái/phải.
- Checkpoint YOLO geometry v2.

Checklist và lệnh nằm trong [VIEC_BAN_CAN_LAM.md](VIEC_BAN_CAN_LAM.md).

## Bổ sung ngày 23/09/2026 — chuyển hoàn toàn từ E sang G

Đã đối chiếu nội dung chuẩn hóa giữa:

```text
E:\Professional documents\Internship\RBCNN_Demo
G:\Internship\RBCNN_Demo
```

Bản E không có dataset hoặc checkpoint riêng. Pipeline annotation/split/train/evaluate đã được hợp nhất vào G từ trước; các file chỉ còn ở E đều là kế hoạch cũ, cấu hình sáu lớp đã loại hoặc sample `pick_case` không còn đúng v2. Hai dòng dữ liệu trong backup E cũng đều đã có trong G.

Sau khi kiểm thử, ba thư mục RBCNN ở E được chuyển vào Recycle Bin. Dự án `AOI_PCB` trên E không bị xóa vì là dự án PCB độc lập.

Repo G được dọn thêm:

- cache Python/pytest;
- feature ViT v1 có thể tạo lại và không tương thích DINOv2-Small;
- log, ảnh inference tạm;
- bản sao baseline trùng hash với checkpoint đang giữ;
- hai ZIP dataset đã xác minh toàn bộ entry đều có trong thư mục giải nén;
- wrapper runtime cũ và CSV sample không còn đúng taxonomy v2.

Tổng cộng khoảng **370,46 MB** được chuyển vào Recycle Bin trong lần dọn này. Muốn dung lượng ổ đĩa được giải phóng thực tế, cần làm rỗng Recycle Bin sau khi đã chắc chắn không cần khôi phục.

### Trạng thái kho Git sau khi dọn

Đã chạy `git gc --prune=now`: toàn bộ object rời đã được gom thành một pack, object rác bằng 0. Thư mục `.git` vẫn khoảng **563,38 MB** vì các video, ZIP dataset và artifact lớn trong các commit/nhánh lịch sử vẫn là object hợp lệ. Không rewrite lịch sử và không force-push để tránh làm hỏng các clone hoặc nhánh đang tồn tại trên GitHub.

Dung lượng hiện tại của bản chính:

- Toàn repo: khoảng **1.076,70 MB**.
- `.git`: khoảng **563,38 MB**.
- Working tree: khoảng **513,32 MB**, trong đó có dataset, video và tài liệu dự án đang được giữ lại.

File `docs/Dự án nhận diện hộp tai nghe_RNN.docx` khoảng 45,9 MB được giữ vì là tài liệu trực tiếp của đề tài, không phải file rác.

## Bổ sung ngày 23/09/2026 — train COCO trên Kaggle

- Thêm `scripts/train_earbud_coco_kaggle.py`, chạy độc lập trên Kaggle.
- Script tự tìm Roboflow COCO export, bỏ category metadata không có box, remap ID, chuyển bounding box sang YOLO, train, đánh giá và đóng gói kết quả.
- Thêm kiểm tra tùy chọn `--require-geometry-v2` để không vô tình dùng checkpoint schema cũ cho quy trình mới.
- Thêm `docs/KAGGLE_TRAIN_EARBUD_COCO.md` với từng cell cần chạy và vị trí chép `best.pt` về dự án.
- Đã thử prepare trực tiếp trên dataset `10-30-Auto Label.coco`: 186 ảnh, 975 box, không có box lỗi; ba test chuyển đổi COCO đều đạt.
- Viết lại `Kaggle_Training_Earbud.ipynb` thành notebook COCO tự chứa: người dùng chỉ sửa `DATASET_ROOT` rồi Run All; xóa `scripts/kaggle_finetune_cells.py` cũ vì trùng chức năng và chỉ chứa các đoạn code dạng chuỗi.

## Bổ sung kiểm tra tháo tai nghe

- Xác nhận runtime đã phát `remove_earbud_to_one` khi occupancy giảm từ 2 xuống 1 và `remove_earbud_to_zero` khi giảm về 0.
- Hai sự kiện đều là `VIOLATION`; FSM lùi về trạng thái vật lý thực tế và không cho phép đóng hộp cho đến khi lắp lại đủ tai.
- Thêm test chuỗi tháo lần lượt cả hai tai `2 → 1 → 0` và bổ sung checklist ảnh cần thu cho hai phía trái/phải.

## Bổ sung taxonomy trái/phải và action remove

- Action model v2 có sáu nhãn; `remove_earbud` được thêm vào config và menu annotation ở vị trí số 5.
- Detector geometry dùng `left_earbud`, `right_earbud`, `empty_left`, `empty_right` thay cho `earbud` chung.
- Fusion/FSM phát `wrong_earbud_side` khi tai trái/phải nằm nhầm khe; action này là luật hệ thống, không phải nhãn cần train cho LSTM.
- YOLO geometry vẫn là điều kiện bắt buộc cho removal; dự đoán `remove_earbud` từ BiLSTM chỉ là bằng chứng hỗ trợ.

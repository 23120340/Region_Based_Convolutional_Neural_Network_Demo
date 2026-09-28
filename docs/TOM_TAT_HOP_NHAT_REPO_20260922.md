# Tóm tắt hợp nhất repository — 22/09/2026

## Bổ sung 28/09/2026 — ngưỡng empty, độ phủ và camera USB

- Sửa trên `main` hiện tại tại `G:\Internship\RBCNN_Demo`; không chuyển nhánh, thay/train trọng số, đổi nhãn action hay push GitHub.
- `camera_config.py`: hỗ trợ `classes[].min_confidence` tùy chọn với phép so sánh nghiêm ngặt >; config cũ không có trường này vẫn dùng ngưỡng chung.
- `yolo_world_detector.py` và config camera: lọc `empty_left/right` > 0,5 trước cả overlay/Fusion, giữ ngưỡng các lớp khác 0,35.
- `slot_fusion.py` và project profile: thêm `min_empty_confidence=0.5` cho bằng chứng hiệu chuẩn/tháo; không suy ra lắp từ việc mất bbox.
- Sửa độ phủ chẩn đoán: vẫn tính phần giao dưới ngưỡng lắp 40%, thêm cờ đủ bbox để tính và confidence empty. Luật xác nhận đúng bên/ổn định/LSTM không bị nới.
- Dashboard: ghi rõ thiếu bbox tai/khe hoặc nắp đóng; hiển thị độ phủ và confidence empty riêng. Bbox tham chiếu đổi chú thích `empty_* 0%` thành `slot_* cover:...`, tránh nhầm anchor với detection empty.
- `run_hybrid.py`: in checkpoint YOLO/LSTM + danh sách actions; thêm `--list-cameras --max-camera-index N` dùng helper dò camera sẵn có, thoát trước khi nạp model.
- Bổ sung tests biên 0,5/0,5001, cấu hình cũ, lọc output YOLO, không tháo bởi empty dưới ngưỡng, độ phủ thấp/thiếu bbox và dò camera không nạp model. 140 tests đạt trên bản staging và bản thật ở ổ G (14,9 giây); không tự mở camera để kiểm thử.
- Hướng dẫn: `docs/NGUONG_EMPTY_VA_CAMERA_DIEN_THOAI.md`; cập nhật `docs/VIEC_BAN_CAN_LAM.md`. Giữ nguyên hai thay đổi riêng có sẵn của người dùng.

## Tách hai nhánh ngày 28/09/2026

- Main khôi phục taxonomy năm lớp và annotation first/second từ Git cũ, giữ thời gian/split; đưa các sửa UI/runtime trước đây vào commit main.
- Giữ lịch sử feature từ `af8c788`; merge main vào rồi hoàn thiện bốn lớp, test annotation/runtime/geometry và checkpoint riêng.
- Pilot đã bị ghi đè bốn lớp: bảo toàn bản này ở `artifacts/action_model_insert_earbud/`; tái train năm lớp ở `artifacts/action_model_pilot/` từ cache/split cũ, không dùng baseline Google ViT làm bản khôi phục.
- Launcher đọc model từ profile; kiểm tra action/backbone trước spatial encoder để báo nhầm model sớm.
- Test staging: main 129 tests, feature 135 tests đạt. Năm lớp val-F1 0,9675/test-F1 0,9040; bốn lớp val-F1 1,0000/test-F1 0,9243. Pilot cùng người/session, không phải độ chính xác hệ thống hoặc xác minh camera.
- Không tự push/reset/rebase/xóa thay đổi riêng của bạn. File capture đang đổi sang pen và việc xóa `configs/camera_earbud_hybrid_config.json` được giữ ngoài commit của lần tách này.
- Chi tiết: `docs/HUONG_DAN_HAI_NHANH_EARBUD.md`.

## Bổ sung 28/09/2026 — xác nhận quy trình không bị mất trên giao diện

- `hybrid_dashboard.py`: thêm ô xác nhận PASS/hoàn tất/vi phạm lớn; tách hướng dẫn đang chờ khỏi kết quả đã chấp nhận; trạng thái từng bước rõ ràng và ba sự kiện gần nhất. Nguồn xác nhận là FSM, không phải nhãn LSTM. Bước bị lùi sau tháo tai hiển thị “CẦN LÀM LẠI”.
- `run_hybrid.py`: truyền lịch sử kết quả sang UI, xóa lịch sử khi R; giữ chuỗi đặc trưng liên tục sau PASS/VIOLATION nhưng vô hiệu hóa quyết định cũ; lấy mẫu mặc định theo config train thay vì tự giảm xuống 4 FPS trên CPU.
- YOLO mặc định theo camera config, không tự chuyển sang mỗi ba frame trên CPU. Replay cùng video thật 12,49 FPS: chạy mỗi ba frame bỏ lỡ bước lắp thứ hai và báo đóng sớm; chạy mỗi frame xác nhận đủ bốn bước. Đây là kiểm thử một clip, không phải đánh giá độ chính xác toàn bộ hệ thống.
- `slot_fusion.py`: khởi tạo khe rỗng bằng ba quan sát dương tính trong cửa sổ năm lần YOLO để chịu mất bbox ngắn; giữ kiểm tra lắp/tháo ba lần liên tiếp, kiểm tra đúng bên và rollback. Hướng dẫn chỉ rõ đang chờ khe, YOLO ổn định hay LSTM đúng bước.
- `earbud_v2.json`: thống nhất ngưỡng bbox Fusion với ngưỡng hiển thị YOLO 0,35; confidence hành động vẫn >0,5. Không thay model/checkpoint hoặc sửa taxonomy người dùng.
- Bổ sung test xác nhận từ runtime tới dashboard, chuỗi đặc trưng không bị xóa sau PASS, lịch sử/reset, hoàn tất/rollback và khởi tạo khe khi detection chập chờn. Cập nhật hướng dẫn camera, việc người dùng cần thử và báo cáo tổng quan.
- Các thay đổi được kiểm tra ở bản staging rồi chuyển đúng sang repo ổ G; không chỉnh bản repo ổ E.
- Kiểm chứng: 124 unit/integration tests đạt bằng Python `.venv` hiện tại. Replay toàn bộ `per1_02_correct_20260918_144106_017.mp4` với checkpoint thật, cache offline và YOLO mỗi frame ghi đủ bốn PASS, kết thúc `S4_COMPLETED`. Đã xem bản render UI xác nhận/vi phạm; chưa trực tiếp kiểm thử camera laptop của người dùng.

## Bổ sung 27/09/2026 — kiểm tra bbox từng khe và giao diện hybrid

- Thêm `PairedEarbudFusionEngine`: ghi nhớ vùng khe theo hộp; đối chiếu tai trái/phải với khe tương ứng; xác nhận khi độ phủ >= 40%, ổn định 3 lần YOLO và LSTM đúng bước có confidence > 0,5.
- Chỉ mất detection không được tính thành lắp/tháo. Tai và khe trống cùng được detect ở cùng vị trí không tạo vòng lặp insertion/removal.
- Tháo tai đã xác nhận tạo VIOLATION, lùi FSM, yêu cầu lắp lại đúng bên; xử lý cả tháo sau khi đã đóng hộp hoàn tất.
- Profile hybrid trỏ đúng config/checkpoint pilot 5 lớp; kiểm tra tên lớp YOLO trước khi mở camera.
- Giao diện 1440×900 có tiếng Việt, tiến độ từng khe, trạng thái quy trình, yêu cầu sửa lỗi; F toàn màn hình, R reset. Launcher không mirror mặc định.
- Replay video lấy mẫu theo thời gian video; thêm `--headless` để chạy kiểm tra không mở cửa sổ.
- Hướng dẫn và việc người dùng cần thử: `docs/HYBRID_SLOT_CHECK_20260927.md` và phần đầu `docs/VIEC_BAN_CAN_LAM.md`.
- Kiểm tra hoàn tất ngày 28/09: 107 unit/integration tests đạt. Test pipeline sáu lớp dùng fixture riêng; test config hiện tại khớp năm lớp mà người dùng đang dùng, không đổi taxonomy hiện có.
- Đã chạy 100 frame video bằng checkpoint YOLO + DINOv2/BiLSTM thật ở chế độ headless (CPU, cache local). Đây là smoke test chạy chung; chưa đo độ chính xác hành động trên camera thực tế.

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

## Bổ sung ngày 23/09/2026 — dataset RNN trái/phải

- Kiểm tra `RNN` và `RNN/LR_Earbud`; xác nhận 190/190 ảnh của bộ LR đã có trong `RNN/train`, không ghép trùng lần hai.
- Thêm `scripts/merge_rnn_earbud_datasets.py` để chuẩn hóa bảy class, đổi polygon thành bbox, khử trùng SHA-256 và dùng hardlink tiết kiệm dung lượng.
- Tạo `datasets/earbud_rnn_merged`: 544 ảnh, 1.059 box, không có label lỗi và `Trainable: YES`.
- Chuẩn hóa class thành `open_case`, `close_case`, `left_earbud`, `right_earbud`, `empty_left`, `empty_right`, `hand`, khớp runtime trái/phải hiện tại.
- Thêm đánh giá tại `docs/DANH_GIA_DATASET_RNN.md`; thêm `/RNN/` vào `.gitignore` để không đẩy hơn 800 MB dữ liệu thô lên GitHub.

## Bổ sung kiểm tra tháo tai nghe

- Xác nhận runtime đã phát `remove_earbud_to_one` khi occupancy giảm từ 2 xuống 1 và `remove_earbud_to_zero` khi giảm về 0.
- Hai sự kiện đều là `VIOLATION`; FSM lùi về trạng thái vật lý thực tế và không cho phép đóng hộp cho đến khi lắp lại đủ tai.
- Thêm test chuỗi tháo lần lượt cả hai tai `2 → 1 → 0` và bổ sung checklist ảnh cần thu cho hai phía trái/phải.

## Bổ sung taxonomy trái/phải và action remove

- Action model v2 có sáu nhãn; `remove_earbud` được thêm vào config và menu annotation ở vị trí số 5.
- Detector geometry dùng `left_earbud`, `right_earbud`, `empty_left`, `empty_right` thay cho `earbud` chung.
- Fusion/FSM phát `wrong_earbud_side` khi tai trái/phải nằm nhầm khe; action này là luật hệ thống, không phải nhãn cần train cho LSTM.
- YOLO geometry vẫn là điều kiện bắt buộc cho removal; dự đoán `remove_earbud` từ BiLSTM chỉ là bằng chứng hỗ trợ.

## Bổ sung ngày 23/09/2026 — train/test BiLSTM pilot

- Tạo `configs/action_earbud_pilot_config.json` gồm năm lớp đã có dữ liệu; chưa dùng `remove_earbud` vì annotation hiện có 0 mẫu.
- Chia 35 video thành 25 train, 5 validation và 5 test tại `data/earbud_actions/annotations_v2_pilot_split.csv`.
- Trích xuất DINOv2-Small feature 384 chiều cho đủ 35 video vào `data/earbud_actions/features_v2_pilot`.
- Train BiLSTM 15 epoch; checkpoint tốt nhất tại epoch 4 có validation macro-F1 0,9644.
- Test đạt accuracy 92,98% và macro-F1 0,9040 trên 57 temporal window; lỗi chính là `open_case` bị nhầm sang `insert_first_earbud`.
- Lưu checkpoint và báo cáo trong `artifacts/action_model_pilot`; ghi kết quả chi tiết tại `docs/KET_QUA_TRAIN_LSTM_PILOT_20260923.md`.
- Đây là pilot cùng người/session, chưa phải phép đánh giá tổng quát. Bản production vẫn cần dữ liệu `remove_earbud` và một session test độc lập.

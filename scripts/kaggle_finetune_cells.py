"""
=================================================================
HƯỚNG DẪN DÙNG TRÊN KAGGLE
=================================================================
Tạo Notebook mới trên Kaggle, copy từng phần (CELL 1, CELL 2...)
vào từng ô code riêng biệt, rồi Run All.

Dataset path của bạn trên Kaggle:
  /kaggle/input/datasets/shintheanother/earbud-dtsv1/earbud_merged
=================================================================
"""

# ============================================================
# CELL 1 — Cài thư viện
# ============================================================
"""
!pip install -q ultralytics
"""

# ============================================================
# CELL 2 — Chuẩn bị dataset & tạo data.yaml chuẩn
# ============================================================
"""
import os, glob

# ─── SỬA ĐƯỜNG DẪN NÀY NẾU CẦN ─────────────────────────────
# Đây là đường dẫn dataset của bạn trên Kaggle
DATASET_ROOT = '/kaggle/input/datasets/shintheanother/earbud-dtsv1/earbud_merged'

# Kiểm tra thư mục tồn tại
assert os.path.isdir(DATASET_ROOT), f"Không tìm thấy dataset tại: {DATASET_ROOT}"
print(f"Dataset root: {DATASET_ROOT}")
print(f"Nội dung: {sorted(os.listdir(DATASET_ROOT))}")

# Kiểm tra số ảnh
for split in ['train', 'valid', 'test']:
    imgs = glob.glob(os.path.join(DATASET_ROOT, split, 'images', '*'))
    lbls = glob.glob(os.path.join(DATASET_ROOT, split, 'labels', '*'))
    print(f"  {split:5s}: {len(imgs)} ảnh, {len(lbls)} labels")

# ─── Tạo data.yaml mới với đường dẫn TUYỆT ĐỐI ─────────────
# KHÔNG đọc lại file yaml cũ (tránh lỗi \r\n trên Windows)
yaml_path = '/kaggle/working/data_earbud.yaml'

yaml_content = f\"\"\"path: {DATASET_ROOT}
train: train/images
val: valid/images
test: test/images

nc: 6
names: ['Earphone_Case', 'Earbud', 'Empty_Slot', 'Left_Earbud', 'Right_Earbud', 'Hand']
\"\"\"

with open(yaml_path, 'w', encoding='utf-8') as f:
    f.write(yaml_content)

print(f"\\n✅ Đã tạo data.yaml tại: {yaml_path}")
print("Nội dung:")
print(yaml_content)
"""

# ============================================================
# CELL 3 — Load pretrained model (yolov8n.pt)
# ============================================================
"""
from ultralytics import YOLO

# File yolov8n.pt có trong dataset zip, hoặc Ultralytics tự tải
pt_in_dataset = os.path.join(DATASET_ROOT, 'yolov8n.pt')
if os.path.exists(pt_in_dataset):
    print(f"Dùng yolov8n.pt từ dataset: {pt_in_dataset}")
    model = YOLO(pt_in_dataset)
else:
    print("Tải yolov8n.pt từ Ultralytics Hub...")
    model = YOLO('yolov8n.pt')

print("Model loaded OK")
"""

# ============================================================
# CELL 4 — Fine-tune
# ============================================================
"""
results = model.train(
    data=yaml_path,          # file data.yaml vừa tạo ở CELL 2
    epochs=60,
    imgsz=640,
    batch=16,                # Giảm xuống 8 nếu GPU bị OOM
    project='/kaggle/working/training',
    name='earbud_merged_detector',
    exist_ok=True,
    pretrained=True,
    optimizer='AdamW',
    lr0=0.001,
    lrf=0.01,
    patience=15,
    save=True,
    save_period=10,
    plots=True,
    verbose=True,
)
print("\\n✅ Fine-tune xong!")
"""

# ============================================================
# CELL 5 — Đánh giá trên tập test
# ============================================================
"""
best_pt = '/kaggle/working/training/earbud_merged_detector/weights/best.pt'
best_model = YOLO(best_pt)
metrics = best_model.val(data=yaml_path, split='test', plots=True, verbose=True)

print('\\n════════ KẾT QUẢ TEST ════════')
print(f'  mAP50    : {metrics.box.map50:.4f}')
print(f'  mAP50-95 : {metrics.box.map:.4f}')
print(f'  Precision: {metrics.box.mp:.4f}')
print(f'  Recall   : {metrics.box.mr:.4f}')
print('═════════════════════════════')
"""

# ============================================================
# CELL 6 — Nén kết quả để tải về
# ============================================================
"""
import os

!zip -r -q /kaggle/working/weights_only.zip /kaggle/working/training/earbud_merged_detector/weights/

size_mb = os.path.getsize('/kaggle/working/weights_only.zip') / 1024 / 1024
print(f"\\n✅ Đã nén: weights_only.zip ({size_mb:.1f} MB)")
print("\\n📥 Cách lấy file best.pt về máy:")
print("  1. Panel phải → mục Output → tải weights_only.zip")
print("  2. Giải nén, lấy file best.pt")
print("  3. Copy đè vào: artifacts/training/earbud_merged_detector/weights/best.pt")
print("  4. Chạy: python scripts/run_earbud.py  ← camera sẽ tự dùng model mới")
"""

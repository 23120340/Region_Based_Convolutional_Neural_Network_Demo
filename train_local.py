"""
Script train YOLOv8 model local
"""
from ultralytics import YOLO
import torch

def main():
    # Kiểm tra GPU
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Đang sử dụng device: {device}")

    # Load pretrained model (YOLOv8n - nhỏ nhất, nhanh)
    # Bạn có thể đổi thành: yolov8s.pt, yolov8m.pt, yolov8l.pt, yolov8x.pt (lớn hơn, chính xác hơn)
    model = YOLO('yolov8n.pt')

    # Train model
    results = model.train(
        data='data.yaml',           # File cấu hình dataset
        epochs=100,                 # Số epoch (có thể điều chỉnh)
        imgsz=640,                  # Kích thước ảnh
        batch=16,                   # Batch size (giảm nếu thiếu RAM/VRAM)
        device=device,              # CPU hoặc GPU
        project='runs/train',       # Thư mục lưu kết quả
        name='earbud_detection',    # Tên experiment
        patience=20,                # Early stopping patience
        save=True,                  # Lưu checkpoint
        save_period=10,             # Lưu mỗi 10 epochs
        cache=False,                # Cache images (True nếu đủ RAM)
        workers=4,                  # Số worker threads
        optimizer='auto',           # SGD, Adam, AdamW, auto
        lr0=0.01,                   # Learning rate ban đầu
        lrf=0.01,                   # Learning rate cuối cùng
        momentum=0.937,             # Momentum
        weight_decay=0.0005,        # Weight decay
        warmup_epochs=3,            # Warmup epochs
        warmup_momentum=0.8,        # Warmup momentum
        box=7.5,                    # Box loss weight
        cls=0.5,                    # Class loss weight
        dfl=1.5,                    # DFL loss weight
        plots=True,                 # Tạo plots
        verbose=True                # Hiển thị chi tiết
    )

    # Đánh giá model trên validation set
    print("\n" + "="*50)
    print("Đánh giá model trên validation set:")
    print("="*50)
    metrics = model.val()

    print(f"\nmAP50: {metrics.box.map50:.4f}")
    print(f"mAP50-95: {metrics.box.map:.4f}")

    # Export model sang ONNX (optional, để deploy)
    print("\n" + "="*50)
    print("Export model sang ONNX format (để deploy):")
    print("="*50)
    try:
        model.export(format='onnx', opset=12, simplify=True)
        print("✓ Đã export thành công!")
    except Exception as e:
        print(f"✗ Lỗi khi export: {e}")

    print("\n" + "="*50)
    print("Training hoàn tất!")
    print(f"Model tốt nhất được lưu tại: runs/train/earbud_detection/weights/best.pt")
    print("="*50)

if __name__ == '__main__':
    main()

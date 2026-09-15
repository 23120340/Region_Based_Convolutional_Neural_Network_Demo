"""
Script nhanh để test model trên 1 vài ảnh từ test set
"""
from ultralytics import YOLO
import os
import random

def quick_test(model_path='runs/train/earbud_detection/weights/best.pt', num_samples=5):
    """
    Test nhanh model trên một vài ảnh ngẫu nhiên từ test set
    """

    if not os.path.exists(model_path):
        print(f"✗ Không tìm thấy model: {model_path}")
        print("Vui lòng train model trước bằng: python train_local.py")
        return

    # Load model
    print(f"Đang load model từ: {model_path}")
    model = YOLO(model_path)

    # Lấy random ảnh từ test set
    test_images_dir = 'test/images'
    if not os.path.exists(test_images_dir):
        print(f"✗ Không tìm thấy thư mục: {test_images_dir}")
        return

    test_images = [f for f in os.listdir(test_images_dir)
                   if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    if len(test_images) == 0:
        print(f"✗ Không tìm thấy ảnh trong {test_images_dir}")
        return

    # Random chọn một vài ảnh
    sample_images = random.sample(test_images, min(num_samples, len(test_images)))

    print(f"\nĐang test trên {len(sample_images)} ảnh ngẫu nhiên...")
    print("="*50)

    # Predict
    for img_name in sample_images:
        img_path = os.path.join(test_images_dir, img_name)
        print(f"\n📷 {img_name}")

        results = model.predict(
            source=img_path,
            conf=0.5,
            save=True,
            project='runs/quick_test',
            name='results',
            verbose=False
        )

        # Hiển thị kết quả
        for r in results:
            boxes = r.boxes
            if len(boxes) == 0:
                print("  ➜ Không phát hiện object nào")
            else:
                print(f"  ➜ Phát hiện {len(boxes)} objects:")
                for box in boxes:
                    cls = int(box.cls[0])
                    conf = float(box.conf[0])
                    class_name = model.names[cls]
                    print(f"     • {class_name}: {conf:.2%}")

    print("\n" + "="*50)
    print(f"✓ Kết quả được lưu tại: runs/quick_test/results/")
    print("="*50)

if __name__ == '__main__':
    import sys

    if len(sys.argv) > 1:
        model_path = sys.argv[1]
        quick_test(model_path)
    else:
        # Sử dụng model mặc định
        quick_test()

"""
Script test model đã train trên ảnh/video
"""
from ultralytics import YOLO
import cv2
import os

def test_on_image(model_path, image_path, conf_threshold=0.5):
    """Test model trên 1 ảnh"""
    model = YOLO(model_path)

    # Predict
    results = model.predict(
        source=image_path,
        conf=conf_threshold,
        save=True,
        project='runs/predict',
        name='test_images'
    )

    print(f"✓ Kết quả được lưu tại: runs/predict/test_images/")

    # In ra thông tin detections
    for r in results:
        boxes = r.boxes
        print(f"\nĐã phát hiện {len(boxes)} objects:")
        for box in boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            class_name = model.names[cls]
            print(f"  - {class_name}: {conf:.2f}")

def test_on_video(model_path, video_path, conf_threshold=0.5):
    """Test model trên video"""
    model = YOLO(model_path)

    # Predict trên video
    results = model.predict(
        source=video_path,
        conf=conf_threshold,
        save=True,
        project='runs/predict',
        name='test_video',
        stream=True  # Stream để xử lý từng frame
    )

    # Process results
    for i, r in enumerate(results):
        if i % 30 == 0:  # Print mỗi 30 frames
            boxes = r.boxes
            print(f"Frame {i}: phát hiện {len(boxes)} objects")

    print(f"\n✓ Video kết quả được lưu tại: runs/predict/test_video/")

def test_on_test_set(model_path):
    """Đánh giá model trên test set"""
    model = YOLO(model_path)

    # Validate trên test set
    metrics = model.val(
        data='data.yaml',
        split='test',
        conf=0.001,  # Confidence threshold thấp để tính đầy đủ metrics
        iou=0.6,     # IOU threshold
        plots=True,
        save_json=True,
        project='runs/test',
        name='evaluation'
    )

    print("\n" + "="*50)
    print("KẾT QUẢ ĐÁNH GIÁ TRÊN TEST SET:")
    print("="*50)
    print(f"mAP50: {metrics.box.map50:.4f}")
    print(f"mAP50-95: {metrics.box.map:.4f}")
    print(f"Precision: {metrics.box.mp:.4f}")
    print(f"Recall: {metrics.box.mr:.4f}")

    # Per-class metrics
    print("\nKẾT QUẢ TỪNG CLASS:")
    for i, name in enumerate(model.names.values()):
        print(f"  {name}:")
        print(f"    - AP50: {metrics.box.ap50[i]:.4f}")
        print(f"    - AP: {metrics.box.ap[i]:.4f}")

    print(f"\nBiểu đồ được lưu tại: runs/test/evaluation/")

def main():
    import argparse

    parser = argparse.ArgumentParser(description='Test YOLOv8 model')
    parser.add_argument('--model', type=str, required=True,
                        help='Đường dẫn đến model (ví dụ: runs/train/earbud_detection/weights/best.pt)')
    parser.add_argument('--source', type=str,
                        help='Đường dẫn ảnh/video để test')
    parser.add_argument('--test-set', action='store_true',
                        help='Đánh giá trên test set')
    parser.add_argument('--conf', type=float, default=0.5,
                        help='Confidence threshold (default: 0.5)')

    args = parser.parse_args()

    if not os.path.exists(args.model):
        print(f"✗ Không tìm thấy model: {args.model}")
        return

    if args.test_set:
        # Đánh giá trên test set
        test_on_test_set(args.model)
    elif args.source:
        # Test trên ảnh hoặc video
        if args.source.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
            test_on_image(args.model, args.source, args.conf)
        elif args.source.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
            test_on_video(args.model, args.source, args.conf)
        else:
            print("✗ Format không được hỗ trợ!")
    else:
        print("Vui lòng chỉ định --source (ảnh/video) hoặc --test-set")
        print("\nVí dụ:")
        print("  python test_model.py --model runs/train/earbud_detection/weights/best.pt --test-set")
        print("  python test_model.py --model runs/train/earbud_detection/weights/best.pt --source test_image.jpg")
        print("  python test_model.py --model runs/train/earbud_detection/weights/best.pt --source video.mp4 --conf 0.6")

if __name__ == '__main__':
    main()

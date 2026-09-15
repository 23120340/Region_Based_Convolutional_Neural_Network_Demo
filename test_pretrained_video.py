"""
Script test model ngay tren video ma khong can train truoc
Chi de kiem tra dataset va xem model pretrained detect duoc khong
"""
from ultralytics import YOLO
import cv2
import os

# Cau hinh
VIDEO_PATH = r"C:\Users\Admin\Videos\VidTest_RNN\1788841100925_210920418793840293_5943654704285097780.mp4"
OUTPUT_PATH = r"C:\RNN\test_pretrained_video.mp4"
MODEL = "yolov8n.pt"  # Pretrained model - chua train tren dataset cua ban
CONFIDENCE_THRESHOLD = 0.3

# Class colors
COLORS = {
    "Case": (0, 0, 255),
    "Hand": (255, 255, 0),
    "Left_Earbud": (255, 0, 0),
    "Empty_Slot": (255, 128, 0),
    "Right_Earbud": (0, 255, 0)
}

def main():
    print("="*60)
    print("TEST PRETRAINED MODEL ON VIDEO (DEMO PURPOSE)")
    print("This uses pretrained YOLO - NOT trained on your dataset")
    print("="*60)

    # Check video
    if not os.path.exists(VIDEO_PATH):
        print(f"X Video not found: {VIDEO_PATH}")
        return

    # Load pretrained model
    print(f"\nLoading pretrained model: {MODEL}")
    model = YOLO(MODEL)
    print("Model loaded successfully!")

    # Open video
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"X Cannot open video: {VIDEO_PATH}")
        return

    # Get video info
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"\nVideo info: {width}x{height} @ {fps} FPS, {total_frames} frames")
    print(f"Output: {OUTPUT_PATH}")

    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (width, height))

    frame_count = 0
    print("\nProcessing video...")
    print("="*60)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        # Run inference
        results = model.predict(
            source=frame,
            conf=CONFIDENCE_THRESHOLD,
            verbose=False
        )

        # Draw predictions
        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                cls = int(box.cls[0])
                conf = float(box.conf[0])
                class_name = model.names[cls]

                # Random color for pretrained classes
                color = (0, 255, 0)  # Green for all detections

                # Draw box
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                # Draw label
                label = f"{class_name}: {conf:.2f}"
                (label_width, label_height), _ = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2
                )

                cv2.rectangle(
                    frame,
                    (x1, y1 - label_height - 10),
                    (x1 + label_width, y1),
                    color,
                    -1
                )

                cv2.putText(
                    frame, label,
                    (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 0),
                    2
                )

        out.write(frame)

        if frame_count % 30 == 0:
            progress = (frame_count / total_frames) * 100
            print(f"Processed: {frame_count}/{total_frames} frames ({progress:.1f}%)")

    cap.release()
    out.release()
    cv2.destroyAllWindows()

    print("\n" + "="*60)
    print(f"COMPLETED! Video saved to: {OUTPUT_PATH}")
    print(f"Total frames processed: {frame_count}")
    print("\nNOTE: This is PRETRAINED model detection (COCO classes)")
    print("To detect your custom classes (Case, Earbud, Empty_Slot):")
    print("  1. Run: python train_local.py")
    print("  2. Then: python run_inference_video_local.py")
    print("="*60)

if __name__ == '__main__':
    main()

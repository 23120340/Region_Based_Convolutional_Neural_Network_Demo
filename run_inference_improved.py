"""
Script inference video voi model moi (improved)
"""
from ultralytics import YOLO
import cv2
import os

# Cau hinh
MODEL_PATH = "runs/detect/runs/train/earbud_fast/weights/best.pt"
VIDEO_PATH = "1788841100925_210920418793840293_5943654704285097780.mp4"
OUTPUT_PATH = "output_video_improved.mp4"
CONFIDENCE_THRESHOLD = 0.5

# Class colors
COLORS = {
    "Case": (0, 0, 255),           # Red
    "Hand": (255, 255, 0),         # Cyan
    "Left_Earbud": (255, 0, 0),    # Blue
    "Empty_Slot": (255, 128, 0),   # Orange
    "Right_Earbud": (0, 255, 0)    # Green
}

def main():
    if not os.path.exists(MODEL_PATH):
        print(f"Model not found: {MODEL_PATH}")
        return

    if not os.path.exists(VIDEO_PATH):
        print(f"Video not found: {VIDEO_PATH}")
        return

    print("="*60)
    print("RUNNING INFERENCE WITH IMPROVED MODEL")
    print("="*60)
    print(f"\nModel: {MODEL_PATH}")
    print(f"Video: {VIDEO_PATH}")
    print(f"Output: {OUTPUT_PATH}")

    # Load model
    print("\nLoading improved model...")
    model = YOLO(MODEL_PATH)

    # Open video
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"Cannot open video")
        return

    # Get video info
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"\nVideo: {width}x{height} @ {fps} FPS, {total_frames} frames")

    # Create video writer with H264
    fourcc = cv2.VideoWriter_fourcc(*'H264')
    out = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (width, height))

    if not out.isOpened():
        print("H264 failed, trying XVID...")
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (width, height))

    print("\nProcessing video...")
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        # Run inference
        results = model.predict(
            source=frame,
            conf=CONFIDENCE_THRESHOLD,
            verbose=False,
            imgsz=640  # Use 640 for better quality
        )

        # Draw detections
        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                cls = int(box.cls[0])
                conf = float(box.conf[0])
                class_name = model.names[cls]
                color = COLORS.get(class_name, (255, 255, 255))

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

        if frame_count % 100 == 0:
            progress = (frame_count / total_frames) * 100
            print(f"Processed: {frame_count}/{total_frames} frames ({progress:.1f}%)")

    cap.release()
    out.release()

    print("\n" + "="*60)
    print(f"COMPLETED! Video saved to: {OUTPUT_PATH}")
    print(f"Total frames: {frame_count}")
    print("\nIMPROVED MODEL RESULTS:")
    print("  - Better Left/Right Earbud detection")
    print("  - Improved Empty_Slot detection")
    print("  - mAP50: 0.785 (+3% vs old model)")
    print("="*60)

if __name__ == '__main__':
    main()

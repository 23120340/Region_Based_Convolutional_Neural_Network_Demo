import cv2
from inference_sdk import InferenceHTTPClient
import os
from dotenv import load_dotenv

# Load API key từ file .env
load_dotenv()
api_key = os.getenv("ROBOFLOW_API_KEY")

if not api_key or api_key == "YOUR_API_KEY_HERE":
    print("ERROR: Bạn cần thêm ROBOFLOW_API_KEY vào file .env")
    print("1. Truy cập: https://app.roboflow.com/settings/api")
    print("2. Copy Private API Key")
    print("3. Mở file .env và paste API key vào")
    exit()

# Khởi tạo client
CLIENT = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",
    api_key=api_key
)

# Cấu hình
MODEL_ID = "traigautchin1983-gmail-com/case-earbud-empty-slot-3-rfdetr-small-t1"
VIDEO_PATH = r"C:\Users\Admin\Videos\VidTest_RNN\1788841100925_210920418793840293_5943654704285097780.mp4"
OUTPUT_PATH = r"C:\RNN\output_video.mp4"

# Mở video
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    print(f"Không thể mở video: {VIDEO_PATH}")
    exit()

# Lấy thông tin video
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

print(f"Video info: {width}x{height} @ {fps} FPS, {total_frames} frames")

# Tạo video writer
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (width, height))

# Class colors
COLORS = {
    "Left_Earbud": (255, 0, 0),    # Blue
    "Right_Earbud": (0, 255, 0),   # Green
    "Case": (0, 0, 255),           # Red
    "Hand": (255, 255, 0)          # Cyan
}

frame_count = 0
print("Bắt đầu xử lý video...")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1

    # Chạy inference
    try:
        result = CLIENT.infer(frame, model_id=MODEL_ID)

        # Vẽ predictions lên frame
        if 'predictions' in result:
            for pred in result['predictions']:
                x = int(pred['x'] - pred['width'] / 2)
                y = int(pred['y'] - pred['height'] / 2)
                w = int(pred['width'])
                h = int(pred['height'])

                class_name = pred['class']
                confidence = pred['confidence']
                color = COLORS.get(class_name, (255, 255, 255))

                # Vẽ bounding box
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

                # Vẽ label
                label = f"{class_name}: {confidence:.2f}"
                cv2.putText(frame, label, (x, y - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    except Exception as e:
        print(f"Error processing frame {frame_count}: {e}")

    # Ghi frame
    out.write(frame)

    # Hiển thị tiến trình
    if frame_count % 30 == 0:
        progress = (frame_count / total_frames) * 100
        print(f"Đã xử lý: {frame_count}/{total_frames} frames ({progress:.1f}%)")

# Dọn dẹp
cap.release()
out.release()
cv2.destroyAllWindows()

print(f"\nHoàn thành! Video đã được lưu tại: {OUTPUT_PATH}")
print(f"Tổng số frames đã xử lý: {frame_count}")

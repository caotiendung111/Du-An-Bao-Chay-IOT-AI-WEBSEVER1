from ultralytics import YOLO
import cv2
import math

# --- CẤU HÌNH ---
# 1. Đường dẫn đến file model (Nếu bạn để trong thư mục khác thì sửa lại)
MODEL_PATH = 'best.pt'

# 2. Ngưỡng tự tin (Confidence Threshold)
# Chỉ hiện khung nếu AI chắc chắn trên 50%
CONF_THRESHOLD = 0.5

# --- CHƯƠNG TRÌNH CHÍNH ---
# Load model
print("Đang tải model AI...")
try:
    model = YOLO(MODEL_PATH)
except Exception as e:
    print(f"Lỗi: Không tìm thấy file '{MODEL_PATH}'. Hãy copy nó ra cùng chỗ với file code này nhé!")
    exit()

# Mở Webcam (Số 0 là camera mặc định của laptop)
cap = cv2.VideoCapture(0)
cap.set(3, 640)  # Chiều rộng
cap.set(4, 480)  # Chiều cao

print("--- ĐANG BẬT CAMERA ---")
print("Bấm nút 'q' để thoát chương trình")

while True:
    success, img = cap.read()
    if not success:
        print("Không đọc được camera!")
        break

    # Đưa ảnh cho AI nhận diện
    # stream=True giúp chạy mượt hơn trên video
    results = model(img, stream=True, conf=CONF_THRESHOLD)

    # Xử lý kết quả trả về
    for r in results:
        boxes = r.boxes
        for box in boxes:
            # 1. Lấy tọa độ khung chữ nhật (x1, y1, x2, y2)
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            # 2. Tính độ tự tin (vd: 0.85 -> 85%)
            conf = math.ceil((box.conf[0] * 100)) / 100

            # 3. Lấy tên class (vd: 'fire')
            cls = int(box.cls[0])
            try:
                # Lấy tên lớp từ model đã train
                class_name = model.names[cls]
            except:
                class_name = "LUA"

            label = f'{class_name} {conf}'

            # 4. Vẽ lên màn hình
            # Vẽ khung màu Đỏ (BGR: 0, 0, 255)
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 3)

            # Vẽ nền chữ cho dễ đọc
            t_size = cv2.getTextSize(label, 0, fontScale=1, thickness=2)[0]
            c2 = x1 + t_size[0], y1 - t_size[1] - 3
            cv2.rectangle(img, (x1, y1), c2, (0, 0, 255), -1, cv2.LINE_AA)

            # Viết chữ trắng
            cv2.putText(img, label, (x1, y1 - 2), 0, 1, (255, 255, 255), thickness=2, lineType=cv2.LINE_AA)

    # Hiển thị
    cv2.imshow("TEST YOLOv8 - Phat Hien Lua", img)

    # Bấm phím 'q' để thoát
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
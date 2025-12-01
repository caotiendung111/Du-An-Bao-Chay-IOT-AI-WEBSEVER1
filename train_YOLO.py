from ultralytics import YOLO
import torch

# Dòng này BẮT BUỘC PHẢI CÓ trên Windows để không bị lỗi văng chương trình
if __name__ == '__main__':

    # 1. Kiểm tra xem máy có Card rời (GPU) không
    device = '0' if torch.cuda.is_available() else 'cpu'
    print(f"--- Đang sử dụng thiết bị: {device.upper()} ---")

    # 2. Load model YOLOv8 Nano
    # Bản 'n' (nano) là bản nhẹ nhất, train nhanh, rất hợp cho dự án IoT/Raspberry Pi của bạn
    model = YOLO('yolov8n.pt')

    # 3. Bắt đầu train
    print(">>> Bắt đầu huấn luyện AI...")

    try:
        results = model.train(
            data='data.yaml',  # File cấu hình đường dẫn
            epochs=10,  # Học 50 lần (bạn có thể tăng lên 100 nếu muốn giỏi hơn)
            imgsz=320,  # Kích thước ảnh chuẩn
            batch=8,  # Số lượng ảnh học 1 lúc (nếu máy yếu thì giảm xuống 8)
            device=device,  # Chạy bằng CPU hoặc GPU
            name='fire_model'  # Tên folder lưu kết quả
        )

        print("\n" + "=" * 40)
        print("CHÚC MỪNG! ĐÃ TRAIN XONG.")
        print("File model xịn nhất nằm ở đường dẫn này:")
        print("runs/detect/fire_model/weights/best.pt")
        print("=" * 40)

    except Exception as e:
        print(f"\n[LỖI RỒI]: {e}")
        print("Mẹo: Kiểm tra lại file data.yaml xem đường dẫn 'Val' hoặc 'train' có đúng không?")
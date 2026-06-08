from ultralytics import YOLO
import torch

# Mandatory entry point check on Windows to prevent multi-processing spawn errors
if __name__ == '__main__':

    # 1. Detect active GPU hardware acceleration (CUDA)
    device = '0' if torch.cuda.is_available() else 'cpu'
    print(f"--- Computing device selected: {device.upper()} ---")

    # 2. Instantiate pre-trained YOLOv8 Nano model
    # YOLOv8n (nano) is the lightest configuration, ideal for edge resource-constrained microcontrollers
    model = YOLO('yolov8n.pt')

    # 3. Execute training process
    print(">>> Beginning training execution...")

    try:
        results = model.train(
            data='data.yaml',  # Project dataset configuration
            epochs=10,  # Number of training epochs (increase to 50-100 for optimal convergence)
            imgsz=320,  # Image resolution size
            batch=8,  # Batch size (decrease if system encounters Out-Of-Memory limits)
            device=device,  # Target hardware runtime device
            name='fire_model'  # Output storage subdirectory name
        )

        print("\n" + "=" * 40)
        print("CONGRATULATIONS! TRAINING COMPLETE.")
        print("Best performing model weights saved to:")
        print("runs/detect/fire_model/weights/best.pt")
        print("=" * 40)

    except Exception as e:
        print(f"\n[TRAINING FAILURE]: {e}")
        print("Troubleshooting: Verify dataset directory paths in data.yaml.")
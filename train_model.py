# train_model.py
# Simple training script for car color detection

import torch
from ultralytics import YOLO

def main():
    # Check device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Load model
    model = YOLO('yolo11n.pt')
    
    # Training parameters
    if device == 'cpu':
        batch_size = 4
        workers = 2
        epochs = 50  # Reduced for CPU
    else:
        batch_size = 16
        workers = 8
        epochs = 100
    
    # Train
    results = model.train(
        data=r"C:\Users\Lutifah\Desktop\INTERSHIP\Car colour detection Model\YOLOv11/data.yaml",
        epochs=epochs,
        imgsz=640,
        device=device,
        batch=batch_size,
        workers=workers,
        patience=30,
        save=True,
        project='car_color_detection',
        name='training_run'
    )
    
    print("Training completed!")
    print(f"Best model saved at: runs/detect/training_run/weights/best.pt")

if __name__ == "__main__":
    main()

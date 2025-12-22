import os
from ultralytics import YOLO

def train_custom_model():
    # 1. Select the dataset configuration file
    # Ensure 'data.yaml' exists, which specifies the paths for training and validation images.
    dataset_yaml_path = "tennis_dataset/data.yaml"

    if not os.path.exists(dataset_yaml_path):
        print(f"Error: {dataset_yaml_path} not found.")
        return
    
    # 2. Model Definition
    print("Loading YOLO11 Nano model...")
    model = YOLO("yolo11n.pt")

    # 3. Training
    print("Starting training...")
    model.train(
        data=dataset_yaml_path,
        epochs=100,
        imgsz=640,
        device=0,  # Use GPU (device=0)
        name="tennis_dataset_yolo11n_100epochs"
    )

    print("Training complete. The model is saved in the './runs/detect/train/' directory.")

if __name__ == "__main__":
    train_custom_model()
import os
from ultralytics import YOLO

def evaluate_custom_model():
    # 1. Data Loading
    dataset_yaml_path = "./data.yaml"

    if not os.path.exists(dataset_yaml_path):
        print(f"Error: {dataset_yaml_path} not found.")
        return

    # 2. Model Definition
    model = YOLO("yolov11n.pt")

    # 3. Evaluation
    model.val(data=dataset_yaml_path, imgsz=640, device=0)

if __name__ == "__main__":
    evaluate_custom_model()
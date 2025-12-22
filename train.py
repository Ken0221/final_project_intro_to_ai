import os
from ultralytics import YOLO

def train_custom_model():
    """
    Finetune YOLO11n on merged dataset (tennis balls, plastic bottles, people)

    Strategy: Light finetuning to improve tennis ball detection while preserving
    YOLO's strong pretrained capabilities for person and bottle detection.
    """
    # 1. Select the dataset configuration file
    dataset_yaml_path = "merged_dataset/data.yaml"

    if not os.path.exists(dataset_yaml_path):
        print(f"Error: {dataset_yaml_path} not found.")
        print("Please run 'python merge_datasets.py' first to create the merged dataset.")
        return

    # 2. Load pretrained model
    print("Loading YOLO11 Nano pretrained model...")
    model = YOLO("yolo11n.pt")

    # 3. Finetune training with lighter settings
    # Lower epochs and learning rate for finetuning (not training from scratch)
    print("Starting finetuning...")
    print(f"Dataset: {dataset_yaml_path}")
    print("Classes: 0=tennisball, 1=plastic-bottle, 2=person")

    model.train(
        data=dataset_yaml_path,
        epochs=50,              # Reduced from 100 (finetune needs fewer epochs)
        imgsz=640,
        batch=16,               # Batch size (adjust based on GPU memory)
        lr0=0.001,              # Lower initial learning rate for finetuning
        lrf=0.01,               # Lower final learning rate
        device=0,               # Use GPU (device=0), or 'cpu' for CPU
        name="multiclass_finetune_50epochs",
        patience=10,            # Early stopping patience
        save=True,              # Save checkpoints
        plots=True,             # Generate training plots
        pretrained=True,        # Keep pretrained weights (important!)
        optimizer='AdamW',      # AdamW optimizer (good for finetuning)
        verbose=True            # Show detailed output
    )

    print("\n" + "="*70)
    print("Training complete!")
    print("="*70)
    print(f"Best model saved at: runs/detect/multiclass_finetune_50epochs/weights/best.pt")
    print(f"Last model saved at: runs/detect/multiclass_finetune_50epochs/weights/last.pt")
    print("\nTo use the model:")
    print("  from ultralytics import YOLO")
    print("  model = YOLO('runs/detect/multiclass_finetune_50epochs/weights/best.pt')")
    print("  results = model('path/to/image.jpg')")

if __name__ == "__main__":
    train_custom_model()
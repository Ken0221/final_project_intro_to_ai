from ultralytics import YOLO
import os
import torch

def train_optimized():
    """
    Train YOLO11n with high resolution (1280) and balanced dataset.
    Optimized for Windows and stability.
    """
    # 1. Dataset config
    dataset_yaml = "merged_v5_dataset/data.yaml"
    
    if not os.path.exists(dataset_yaml):
        print(f"Error: {dataset_yaml} not found.")
        return

    # Check CUDA availability
    device = '0' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    if device == 'cpu':
        print("Warning: CUDA not found. Training on CPU will be extremely slow.")

    # 2. Load model
    print("Loading YOLO11n...")
    model = YOLO("yolo11n.pt")

    # 3. Train with optimized settings for SMALL OBJECTS
    print("Starting Optimized Training (1280px)...")
    
    model.train(
        data=dataset_yaml,
        epochs=50,
        imgsz=1280,             # High resolution for small objects (Crucial)
        
        # Hardware & Stability Settings
        batch=16,                # Safe for 6-8GB VRAM. If you have 16GB+, try 8 or 16.
        device=device,
        workers=2,              # CRITICAL for Windows: avoid multiprocessing errors
        verbose=False,          # CRITICAL: Prevent RecursionError in deepcopy/FLOPs check
        
        # Augmentation adjustments
        mosaic=1.0,             # Strong spatial augmentation
        mixup=0.15,             # Mixup to help with generalization
        scale=0.5,              # EXTREME SCALING: 0.9 means images can be sized 0.1x to 1.9x (creates tiny balls)
        
        # Loss adjustments
        box=7.5,                # High box loss gain (focus on accurate bounding boxes)
        cls=0.5,
        
        # Training mechanics
        lr0=0.001,
        lrf=0.01,
        patience=10,
        name="yolo11n_highres_balanced_0.5scale",
        save=True,
        exist_ok=True,
        pretrained=True
    )
    
    print("Training Complete. Model saved in runs/detect/yolo11n_highres_balanced/weights/best.pt")

if __name__ == "__main__":
    # Fix for MP on Windows
    # from multiprocessing import freeze_support
    # freeze_support() 
    train_optimized()

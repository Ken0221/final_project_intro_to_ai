"""
Finetune YOLO on COCO-mapped dataset.

This preserves all 80 COCO classes while enhancing detection for:
- sports ball (class 32) - trained with tennisball data
- bottle (class 39) - trained with plastic-bottle data  
- person (class 0) - trained with person data
"""

import os
from ultralytics import YOLO


def train_coco_finetune():
    """
    Finetune YOLO on COCO-mapped dataset.
    
    Key difference from regular finetuning:
    - Dataset labels use COCO class IDs (0-79)
    - Model retains all 80 class detection capability
    - Only specified classes get enhanced from your data
    """
    dataset_yaml_path = "balanced_coco_dataset/data.yaml"
    
    if not os.path.exists(dataset_yaml_path):
        print(f"Error: {dataset_yaml_path} not found.")
        print("Please run 'python remap_labels_to_coco.py' first.")
        return
    
    # Use YOLO11 Large for better accuracy (or change to yolo11n.pt for speed)
    print("Loading YOLO11 Large pretrained model...")
    model = YOLO("yolo11n.pt")  # Will auto-download if not present
    
    print("Starting COCO-compatible finetuning...")
    print(f"Dataset: {dataset_yaml_path}")
    print("Enhanced classes:")
    print("  - sports ball (32) <- tennisball data")
    print("  - bottle (39) <- plastic-bottle data")
    print("  - person (0) <- person data")
    print("All other 77 COCO classes remain from pretrained weights.")
    
    model.train(
        data=dataset_yaml_path,
        epochs=1,
        imgsz=640,
        batch=32,                # Adjust based on GPU memory (16GB VRAM -> try 8-16)
        lr0=0.0005,             # Low learning rate to preserve pretrained weights
        lrf=0.01,
        device=0,
        name="coco_balanced_enhanced",
        patience=15,
        save=True,
        plots=True,
        pretrained=True,
        optimizer='AdamW',
        amp=True,               # Mixed precision for speed
        verbose=True,
        
        # Important: freeze early layers to preserve pretrained features
        freeze=10,              # Freeze first 10 layers
    )
    
    print("\n" + "="*70)
    print("Training complete!")
    print("="*70)
    print(f"Best model: runs/detect/coco_finetune_enhanced/weights/best.pt")
    print(f"\nThis model can detect ALL 80 COCO classes!")
    print(f"Enhanced classes: sports ball, bottle, person")
    print(f"\nUsage:")
    print(f"  model = YOLO('runs/detect/coco_finetune_enhanced/weights/best.pt')")
    print(f"  results = model('image.jpg')")


if __name__ == "__main__":
    train_coco_finetune()

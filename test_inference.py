from ultralytics import YOLO
import sys
import glob
import os

def test_model(model_path, source="test_images"):
    """
    Test the trained model on images or video.
    """
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        print("Please run training first.")
        return

    model = YOLO(model_path)
    
    # Run inference with a lower confidence threshold for small objects
    # and a higher IoU threshold to avoid duplicates
    print(f"Testing model: {model_path} on {source}")
    
    results = model.predict(
        source=source,
        save=True,
        imgsz=1280,      # Use same high resolution as training
        conf=0.25,       # Confidence threshold
        iou=0.45,        # NMS IoU threshold
        device=0
    )
    
    print(f"Results saved to {results[0].save_dir}")

if __name__ == "__main__":
    # Default to best model from optimized training
    model_path = "runs/detect/yolo11n_highres_balanced/weights/best.pt"
    
    if len(sys.argv) > 1:
        source = sys.argv[1]
    else:
        print("Usage: python test_inference.py <path_to_image_or_video>")
        print("Using default test image if exists...")
        source = "test_sample.jpg" # Placeholder
        
    test_model(model_path, source)

import os
import cv2
import numpy as np
import glob
from tqdm import tqdm
import math

def visualize_dataset(dataset_path, output_dir, grid_size=4, tile_size=640, max_grids=20):
    """
    Visualizes images and labels from the dataset in a grid format.
    
    Args:
        dataset_path (str): Path to the dataset root (e.g., 'merged_v6_dataset_scaled').
        output_dir (str): Directory to save the grid images.
        grid_size (int): Number of images per row/col (default 4x4).
        tile_size (int): Size to resize each image for the grid.
        max_grids (int): Maximum number of grids to generate to save time.
    """
    
    # Try to load class names from data.yaml
    data_yaml_path = os.path.join(dataset_path, 'data.yaml')
    class_names = ['Tennis', 'Bottle', 'Person'] # Default
    if os.path.exists(data_yaml_path):
        try:
            with open(data_yaml_path, 'r') as f:
                import yaml
                data = yaml.safe_load(f)
                if 'names' in data:
                    class_names = data['names']
                    print(f"Loaded class names from data.yaml: {class_names}")
        except Exception as e:
            print(f"Could not load data.yaml: {e}")

    colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255)] # BGR defaults
    # Extend colors if needed
    while len(colors) < len(class_names):
        colors.append((np.random.randint(0,255), np.random.randint(0,255), np.random.randint(0,255)))
    
    image_files = sorted(glob.glob(os.path.join(dataset_path, 'train', 'images', '*.*')))
    image_files = [f for f in image_files if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
    
    if not image_files:
        print("No images found.")
        return
        
    os.makedirs(output_dir, exist_ok=True)
    print(f"Found {len(image_files)} images. Generating visualizations...")
    
    batch_size = grid_size * grid_size
    num_batches = math.ceil(len(image_files) / batch_size)
    
    # Limit processing
    if max_grids:
        num_batches = min(num_batches, max_grids)
        print(f"Limiting to {num_batches} grids ({num_batches * batch_size} images).")

    for i in tqdm(range(num_batches), desc="Creating Grids"):
        batch_files = image_files[i*batch_size : (i+1)*batch_size]
        
        processed_images = []
        
        for img_path in batch_files:
            # Read Image
            img = cv2.imread(img_path)
            if img is None:
                # Create placeholder for failed load
                img = np.zeros((tile_size, tile_size, 3), dtype=np.uint8)
                cv2.putText(img, "Load Error", (10, tile_size//2), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
            else:
                # Handle EXIF via PIL if needed, but cv2.imread usually ignores it (raw pixels).
                # If we want to visualize CORRECTLY as the user sees it, we should use PIL to rotate.
                # However, labels are usually relative to the RAW image dimensions in YOLO unless rotated.
                # BUT, the user's issue was rotation. 
                # If we use PIL to rotate, we MUST also rotate labels if they were relative to the unrotated image.
                # In this specific dataset case, the labels MATCH the EXIF-rotated image (upright).
                try:
                    pil_img = Image.open(img_path)
                    pil_img = ImageOps.exif_transpose(pil_img)
                    img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
                except:
                    pass

            h, w = img.shape[:2]
            
            # Read Labels
            label_path = os.path.join(dataset_path, 'train', 'labels', os.path.basename(img_path).rsplit('.', 1)[0] + '.txt')
            if os.path.exists(label_path):
                with open(label_path, 'r') as f:
                    lines = f.readlines()
                    
                for line in lines:
                    parts = line.strip().split()
                    if not parts: continue
                    
                    try:
                        cls_id = int(parts[0])
                        
                        # Handle Polygon (Segmentation) vs BBox
                        if len(parts) > 5:
                            # Polygon: class x1 y1 x2 y2 ...
                            coords = list(map(float, parts[1:]))
                            xs = coords[0::2]
                            ys = coords[1::2]
                            min_x, max_x = min(xs), max(xs)
                            min_y, max_y = min(ys), max(ys)
                            
                            cx = (min_x + max_x) / 2
                            cy = (min_y + max_y) / 2
                            bw = max_x - min_x
                            bh = max_y - min_y
                        else:
                            # BBox: class cx cy w h
                            cx, cy, bw, bh = map(float, parts[1:5])
                        
                        # Denormalize
                        x1 = int((cx - bw/2) * w)
                        y1 = int((cy - bh/2) * h)
                        x2 = int((cx + bw/2) * w)
                        y2 = int((cy + bh/2) * h)
                        
                        color = colors[cls_id] if 0 <= cls_id < len(colors) else (255, 255, 255)
                        label_name = class_names[cls_id] if 0 <= cls_id < len(class_names) else str(cls_id)
                        
                        # Draw Box
                        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                        
                        # Draw Text
                        (text_w, text_h), baseline = cv2.getTextSize(label_name, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                        if y1 - text_h - baseline < 0:
                            # Draw below if not enough space on top
                            cv2.rectangle(img, (x1, y1), (x1 + text_w, y1 + text_h + baseline), color, -1)
                            cv2.putText(img, label_name, (x1, y1 + text_h), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1)
                        else:
                            cv2.rectangle(img, (x1, y1 - text_h - baseline), (x1 + text_w, y1), color, -1)
                            cv2.putText(img, label_name, (x1, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1)
                        
                    except ValueError:
                        pass
            
            # Resize for grid
            img_resized = cv2.resize(img, (tile_size, tile_size))
            processed_images.append(img_resized)
            
        # Fill remaining spots in batch with black
        while len(processed_images) < batch_size:
            processed_images.append(np.zeros((tile_size, tile_size, 3), dtype=np.uint8))
            
        # Create Grid
        # Stack rows
        rows = []
        for r in range(grid_size):
            row_imgs = processed_images[r*grid_size : (r+1)*grid_size]
            rows.append(np.hstack(row_imgs))
            
        grid_img = np.vstack(rows)
        
        # Save Grid
        output_filename = os.path.join(output_dir, f"grid_{i:03d}.jpg")
        cv2.imwrite(output_filename, grid_img)

    print(f"\nSaved {num_batches} grids to '{output_dir}'.")

if __name__ == "__main__":
    visualize_dataset("merged_v6_dataset_scaled", "check_merged_v6_dataset_scaled", max_grids=1000) # Generate up to 50 grids (800 images)

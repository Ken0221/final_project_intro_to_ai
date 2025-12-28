import os
import shutil
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import yaml

def filter_dataset(input_dir, output_dir, min_area=400):
    """
    Filters the dataset by removing bounding boxes with area < min_area (in pixels).
    """
    input_base = Path(input_dir)
    output_base = Path(output_dir)
    
    if output_base.exists():
        print(f"Removing existing output directory: {output_dir}")
        shutil.rmtree(output_base)
    
    splits = ['train', 'valid', 'test']
    
    total_boxes = 0
    kept_boxes = 0
    
    for split in splits:
        print(f"\nProcessing {split} set...")
        input_images_dir = input_base / split / 'images'
        input_labels_dir = input_base / split / 'labels'
        
        output_images_dir = output_base / split / 'images'
        output_labels_dir = output_base / split / 'labels'
        
        output_images_dir.mkdir(parents=True, exist_ok=True)
        output_labels_dir.mkdir(parents=True, exist_ok=True)
        
        if not input_images_dir.exists():
            print(f"Skipping {split} (images dir not found)")
            continue
            
        image_files = list(input_images_dir.glob("*.jpg")) + \
                      list(input_images_dir.glob("*.jpeg")) + \
                      list(input_images_dir.glob("*.png"))
        
        for img_path in tqdm(image_files, desc=f"Filtering {split}"):
            # Copy image
            shutil.copy2(img_path, output_images_dir / img_path.name)
            
            # Process label
            label_path = input_labels_dir / f"{img_path.stem}.txt"
            
            if not label_path.exists():
                # If no label file, strictly speaking it's a background image or unlabeled
                # We create an empty label file to be consistent
                (output_labels_dir / label_path.name).touch()
                continue
            
            # Read Image for dimensions
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            
            h_img, w_img = img.shape[:2]
            
            with open(label_path, 'r') as f:
                lines = f.readlines()
                
            new_lines = []
            for line in lines:
                parts = line.strip().split()
                if len(parts) < 5: continue
                
                # Format: class x y w h
                # We need w and h
                try:
                    w_norm = float(parts[3])
                    h_norm = float(parts[4])
                    
                    area_pixels = (w_norm * w_img) * (h_norm * h_img)
                    
                    total_boxes += 1
                    if area_pixels >= min_area:
                        new_lines.append(line)
                        kept_boxes += 1
                except ValueError:
                    continue
            
            # Write new label file
            with open(output_labels_dir / label_path.name, 'w') as f:
                f.writelines(new_lines)
                
    # Create data.yaml
    data_yaml = {
        'path': str(output_base.absolute()),
        'train': 'train/images',
        'val': 'valid/images',
        'test': 'test/images',
        'nc': 1,
        'names': ['human'] 
    }
    
    # Try to copy names from original if possible
    orig_yaml = input_base / 'data.yaml'
    if orig_yaml.exists():
        try:
            with open(orig_yaml, 'r') as f:
                orig_data = yaml.safe_load(f)
                if 'names' in orig_data:
                    data_yaml['names'] = orig_data['names']
                if 'nc' in orig_data:
                    data_yaml['nc'] = orig_data['nc']
        except Exception as e:
            print(f"Warning: Could not read original data.yaml: {e}")

    with open(output_base / 'data.yaml', 'w') as f:
        yaml.dump(data_yaml, f, sort_keys=False)

    print(f"\nFiltering Complete.")
    print(f"Total Boxes: {total_boxes}")
    print(f"Kept Boxes: {kept_boxes}")
    print(f"Removed: {total_boxes - kept_boxes} ({((total_boxes - kept_boxes)/total_boxes)*100:.2f}%)")
    print(f"Saved to {output_dir}")

if __name__ == "__main__":
    filter_dataset("human_dataset", "human_large_dataset", min_area=10000)

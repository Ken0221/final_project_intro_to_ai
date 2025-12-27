import os
import random
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import shutil

def scale_image(img, scale):
    """
    Scales the image by the given factor and pastes it onto a gray canvas of the original size.
    Returns the new image and the offset (x, y) for label adjustment.
    """
    h, w = img.shape[:2]
    new_w = int(w * scale)
    new_h = int(h * scale)
    
    # Resize image
    resized_img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    
    # Create canvas (gray background to simulate environment/noise, or black)
    # Using gray (114, 114, 114) as it's common in YOLO preprocessing
    canvas = np.full((h, w, 3), 114, dtype=np.uint8)
    
    # Random paste position
    x_offset = random.randint(0, w - new_w)
    y_offset = random.randint(0, h - new_h)
    
    canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized_img
    
    return canvas, x_offset, y_offset, new_w, new_h

def process_dataset(input_dir, output_dir, target_count=2000):
    """
    Applies scaling and class remapping to the dataset, balancing the count of each class.
    """
    input_base = Path(input_dir)
    output_base = Path(output_dir)
    
    # Define scaling ranges
    SCALE_RANGES = {
        0: (0.1, 0.7),  # Tennis (Original) -> Small (Simulate far)
        1: (0.1, 0.7),  # Bottle (Original) -> Small
        2: (0.8, 1.0),  # Person -> keep mostly original
        3: (0.7, 1.0),  # Bottle (Custom) -> Mild scaling
        4: (0.7, 1.0)   # Tennis (Custom) -> Mild scaling
    }
    
    # Define Class Remapping
    # Map input Class ID -> Output Class ID
    CLASS_REMAP = {
        0: 0, # Tennis -> Tennis
        1: 1, # Bottle -> Bottle
        2: 2, # Person -> Person
        3: 1, # Bottle (Custom) -> Bottle
        4: 0  # Tennis (Custom) -> Tennis
    }

    # Group inputs by TARGET class for balancing
    # We will store (image_path, label_path, source_class_id) tuples in lists
    # Key: Target Class ID (0, 1, 2)
    source_buckets = {0: [], 1: [], 2: []}
    
    splits = ['train', 'valid']
    
    for split in splits:
        print(f"\nScanning {split} set...")
        images_dir = input_base / split / 'images'
        labels_dir = input_base / split / 'labels'
        
        if not images_dir.exists():
            continue
            
        image_files = list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.jpeg")) + list(images_dir.glob("*.png"))
        
        for img_file in tqdm(image_files, desc=f"Loading {split}"):
            label_file = labels_dir / f"{img_file.stem}.txt"
            if not label_file.exists():
                continue
                
            # Determine source class ID from label file
            # Heuristic: Find specific classes. 
            with open(label_file, 'r') as f:
                lines = f.readlines()
            
            class_ids = [int(line.split()[0]) for line in lines]
            if not class_ids: continue

            # Determine primary source class and target bucket
            src_id = 2 # Default Person
            target_id = 2
            
            # Prioritize finding the objects
            if 0 in class_ids: 
                src_id = 0
                target_id = 0
            elif 4 in class_ids:
                src_id = 4
                target_id = 0
            elif 1 in class_ids:
                src_id = 1
                target_id = 1
            elif 3 in class_ids:
                src_id = 3
                target_id = 1
            elif 2 in class_ids:
                src_id = 2
                target_id = 2
            
            source_buckets[target_id].append((img_file, label_file, src_id))



    for split in splits:
        # Re-scan for this specific split
        split_sources = {0: [], 1: [], 2: []}
        
        images_dir = input_base / split / 'images'
        labels_dir = input_base / split / 'labels'
        if not images_dir.exists(): continue
            
        image_files = list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.jpeg")) + list(images_dir.glob("*.png"))
        
        for img_file in image_files:
            label_file = labels_dir / f"{img_file.stem}.txt"
            if not label_file.exists(): continue
            with open(label_file, 'r') as f: lines = f.readlines()
            class_ids = [int(line.split()[0]) for line in lines]
            if not class_ids: continue
            
            src_id, target_id = 2, 2
            if 0 in class_ids: src_id, target_id = 0, 0
            elif 4 in class_ids: src_id, target_id = 4, 0
            elif 1 in class_ids: src_id, target_id = 1, 1
            elif 3 in class_ids: src_id, target_id = 3, 1
            elif 2 in class_ids: src_id, target_id = 2, 2
            
            split_sources[target_id].append((img_file, label_file, src_id))

        out_images_dir = output_base / split / 'images'
        out_labels_dir = output_base / split / 'labels'
        out_images_dir.mkdir(parents=True, exist_ok=True)
        out_labels_dir.mkdir(parents=True, exist_ok=True)

        loop_count_target = target_count if split == 'train' else 0 # 0 means process all exactly once
        
        if split == 'train':
            print(f"\nGeneratng TRAINING Data (Scaled & Balanced)...")
            
            # Handle target_count input
            if isinstance(target_count, int):
                target_counts = {0: target_count, 1: target_count, 2: target_count}
            else:
                target_counts = target_count

            for tid in [0, 1, 2]:
                sources = split_sources[tid]
                t_count = target_counts.get(tid, 2000)
                
                if not sources: 
                    print(f"  Warning: No sources for class {tid}")
                    continue
                    
                print(f"  Class {tid}: Generating {t_count} samples...")
                for i in tqdm(range(t_count), desc=f"Class {tid}"):
                    src = random.choice(sources) # Random w/ replacement
                    process_single_image(src, i, out_images_dir, out_labels_dir, SCALE_RANGES, CLASS_REMAP, True)
        else:
            print(f"\nProcessing {split.upper()} Data (Copy & Remap only)...")
            for tid in [0, 1, 2]:
                sources = split_sources[tid]
                if not sources: continue
                # Unique original filenames are sufficient for Valid/Test (no sampling with replacement)
                for src_tuple in tqdm(sources, desc=f"Class {tid}"):
                    src_img_path, src_lbl_path, src_cls_id = src_tuple
                    
                    # 1. Copy Image (Preserve original)
                    dest_img_path = out_images_dir / src_img_path.name
                    shutil.copy2(src_img_path, dest_img_path)
                    
                    # 2. Remap Label
                    if not src_lbl_path.exists(): continue
                    
                    with open(src_lbl_path, 'r') as f: 
                        lines = f.readlines()
                    
                    new_lines = []
                    for line in lines:
                        parts = line.strip().split()
                        if not parts: continue
                        
                        try:
                            cls_id = int(parts[0])
                            
                            # Remap class (e.g. 3->1, 4->0)
                            new_cls_id = CLASS_REMAP.get(cls_id, cls_id)
                            
                            # Reconstruct line with new class ID
                            # Keep original coordinates
                            new_line = f"{new_cls_id} " + " ".join(parts[1:]) + "\n"
                            new_lines.append(new_line)
                        except ValueError:
                            continue
                        
                    with open(out_labels_dir / src_lbl_path.name, 'w') as f:
                        f.writelines(new_lines)

    # Generate data.yaml
    data_yaml_content = f"""train: ../train/images
val: ../valid/images

nc: 3
names: ['tennisball', 'plastic-bottle', 'person']
"""
    with open(output_base / "data.yaml", 'w') as f:
        f.write(data_yaml_content)

    print(f"\nProcessing Complete. Saved to {output_dir}")


def process_single_image(source_tuple, index, out_img_dir, out_lbl_dir, ranges, remap, apply_scale):
    src_img_path, src_lbl_path, src_cls_id = source_tuple
    
    img = cv2.imread(str(src_img_path))
    if img is None: return

    with open(src_lbl_path, 'r') as f: lines = f.readlines()

    scale_range = ranges.get(src_cls_id, (1.0, 1.0))
    scale = random.uniform(scale_range[0], scale_range[1]) if apply_scale else 1.0
    
    new_img, x_off, y_off, new_w, new_h = scale_image(img, scale)
    
    # Save Name
    # train_0_0001_origName.jpg
    # Avoid collisions in Valid set if we just enumerate
    name_prefix = f"{src_cls_id}_{index}"
    out_name = f"{name_prefix}_{src_img_path.name}"
    
    cv2.imwrite(str(out_img_dir / out_name), new_img)
    
    # Remap Labels
    new_lines = []
    h_orig, w_orig = img.shape[:2]
    
    for line in lines:
        parts = line.strip().split()
        cls_id = int(parts[0])
        
        # Handle YOLO Segmentation (Polygon) -> Convert to BBox
        if len(parts) > 5:
            # Format: class x1 y1 x2 y2 ...
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
            # Format: class cx cy w h
            cx, cy, bw, bh = map(float, parts[1:])
        
        new_cls_id = remap.get(cls_id, cls_id)
        
        # Transform
        abs_cx, abs_cy = cx * w_orig, cy * h_orig
        abs_bw, abs_bh = bw * w_orig, bh * h_orig
        
        scaled_cx, scaled_cy = abs_cx * scale, abs_cy * scale
        scaled_bw, scaled_bh = abs_bw * scale, abs_bh * scale
        
        final_cx = (scaled_cx + x_off) / w_orig
        final_cy = (scaled_cy + y_off) / h_orig
        final_bw = scaled_bw / w_orig
        final_bh = scaled_bh / h_orig
        
        new_lines.append(f"{new_cls_id} {final_cx:.6f} {final_cy:.6f} {final_bw:.6f} {final_bh:.6f}\n")
        
    with open(out_lbl_dir / f"{name_prefix}_{src_img_path.stem}.txt", 'w') as f:
        f.writelines(new_lines)




if __name__ == "__main__":
    if not os.path.exists("merged_v6_dataset"):
        print("Error: merged_v6_dataset not found. Please run merge_datasets.py first.")
    else:
        # We will output to merged_v6_dataset_scaled to keep original for comparison/debugging
        # or we can overwrite. Let's create a new one to be safe.
        # Example usage with custom counts
        target_counts = {
            0: 2000, # Tennis
            1: 2000, # Bottle
            2: 2000  # Person
        }
        process_dataset("merged_v6_dataset", "merged_v6_dataset_scaled", target_count=target_counts)

import os
import shutil
import random
from pathlib import Path
from tqdm import tqdm

def balance_dataset(target_count=4000):
    """
    Balances the merged datasets by upsampling (duplicating) rare classes.
    """
    base_dir = Path("merged_v2_dataset")
    train_dir = base_dir / "train"
    
    if not train_dir.exists():
        print(f"Error: {train_dir} not found. Please run merge_datasets.py first.")
        return

    print(" Analyzing class distribution...")
    
    print(" Analyzing class distribution...")
    
    # Analyze current distribution
    class_counts = {0: 0, 1: 0, 2: 0} # 0: tennisball, 1: bottle, 2: person
    image_files = list((train_dir / "images").glob("*.jpg")) + list((train_dir / "images").glob("*.jpeg")) + list((train_dir / "images").glob("*.png"))
    
    # Map file stem to its main class (heuristic: majority class in file)
    file_class_map = {}
    
    for img_file in tqdm(image_files, desc="Scanning labels"):
        label_file = train_dir / "labels" / f"{img_file.stem}.txt"
        if not label_file.exists(): 
            continue
            
        with open(label_file, 'r') as f:
            lines = f.readlines()
            
        if not lines:
            continue
            
        # Determine the 'primary' class of this image. 
        # If it contains ANY rare class (0 or 1), treat it as a rare class sample.
        # Otherwise it's a person image.
        classes_in_img = [int(line.split()[0]) for line in lines]
        
        primary_class = 2 # Default to person
        if 0 in classes_in_img:
            primary_class = 0
            class_counts[0] += 1
        elif 1 in classes_in_img:
            primary_class = 1
            class_counts[1] += 1
        else:
            class_counts[2] += 1 # Only count as person if NO rare object is present
            
        file_class_map[img_file] = primary_class

    print("\n Current Distribution:")
    print(f"  Class 0 (Tennis Ball): {class_counts[0]}")
    print(f"  Class 1 (Bottle):      {class_counts[1]}")
    print(f"  Class 2 (Person):      {class_counts[2]}")
    
    target_classes = [0, 1, 2] # Process ALL classes
    
    for cls in target_classes:
        current_count = class_counts[cls]
        
        if current_count == target_count:
            print(f"Class {cls} count ({current_count}) is perfect.")
            continue
            
        if current_count < target_count:
            # UPSAMPLE
            needed = target_count - current_count
            print(f"\n Upsampling Class {cls}: Need {needed} more images...")
            
            candidates = [img for img, c in file_class_map.items() if c == cls]
            
            if not candidates:
                print(f"  Warning: No candidates found for class {cls}!")
                continue
                
            for i in tqdm(range(needed), desc=f"Duplicating Class {cls}"):
                src_img = random.choice(candidates)
                src_label = train_dir / "labels" / f"{src_img.stem}.txt"
                
                new_name = f"{src_img.stem}_dup_{i}"
                dst_img = train_dir / "images" / f"{new_name}{src_img.suffix}"
                dst_label = train_dir / "labels" / f"{new_name}.txt"
                
                shutil.copy2(src_img, dst_img)
                if src_label.exists():
                    shutil.copy2(src_label, dst_label)
                    
        elif current_count > target_count:
            # DOWNSAMPLE
            needed_to_remove = current_count - target_count
            print(f"\n Downsampling Class {cls}: Removing {needed_to_remove} images...")
            
            candidates = [img for img, c in file_class_map.items() if c == cls]
            to_remove = random.sample(candidates, needed_to_remove)
            
            for img in tqdm(to_remove, desc=f"Removing Class {cls}"):
                # Remove image
                img.unlink()
                # Remove label
                lbl = train_dir / "labels" / f"{img.stem}.txt"
                if lbl.exists():
                    lbl.unlink()

    print("\nBalancing Complete!")
    
if __name__ == "__main__":
    balance_dataset(target_count=4000)

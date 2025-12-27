"""
Merge tennis_dataset and tennis_dataset2 into a single dataset.

Label check:
- tennis_dataset: 'tennisball' (class 0)
- tennis_dataset2: 'tennis ball' (class 0) - has space

Both use class ID 0, so labels are compatible. Output uses 'tennisball'.
"""

import os
import shutil
from pathlib import Path
from tqdm import tqdm


def merge_tennis_datasets():
    # Paths
    dataset1 = Path(r"c:\Users\User\Downloads\tennis_dataset")
    dataset2 = Path(r"c:\Users\User\Downloads\tennis_dataset2")
    output = Path(r"d:\nycu\balling\final_project_intro_to_ai\merged_tennis_dataset")
    
    # Create output directories
    for split in ['train', 'valid', 'test']:
        (output / split / 'images').mkdir(parents=True, exist_ok=True)
        (output / split / 'labels').mkdir(parents=True, exist_ok=True)
    
    stats = {'train': 0, 'valid': 0, 'test': 0}
    
    # Process both datasets
    for dataset_name, dataset_path in [('ds1', dataset1), ('ds2', dataset2)]:
        print(f"\nProcessing {dataset_path.name}...")
        
        for split in ['train', 'valid', 'test']:
            src_images = dataset_path / split / 'images'
            src_labels = dataset_path / split / 'labels'
            
            if not src_images.exists():
                print(f"  {split}: not found, skipping")
                continue
            
            image_files = list(src_images.glob('*'))
            print(f"  {split}: {len(image_files)} images")
            
            for img_file in tqdm(image_files, desc=f"  Copying {split}"):
                # Add prefix to avoid filename conflicts
                new_name = f"{dataset_name}_{img_file.name}"
                
                # Copy image
                dst_img = output / split / 'images' / new_name
                if not dst_img.exists():
                    shutil.copy2(img_file, dst_img)
                
                # Copy label (same name but .txt extension)
                label_name = img_file.stem + '.txt'
                src_label = src_labels / label_name
                if src_label.exists():
                    new_label_name = f"{dataset_name}_{label_name}"
                    dst_label = output / split / 'labels' / new_label_name
                    if not dst_label.exists():
                        shutil.copy2(src_label, dst_label)
                
                stats[split] += 1
    
    # Create data.yaml
    yaml_content = """train: train/images
val: valid/images
test: test/images

nc: 1
names: ['tennisball']

# Merged from:
# - tennis_dataset: 80 train, 27 valid, 17 test (label: 'tennisball')
# - tennis_dataset2: 1865 train, 100 valid, 50 test (label: 'tennis ball')
# Both use class ID 0, labels are compatible.
"""
    
    with open(output / 'data.yaml', 'w') as f:
        f.write(yaml_content)
    
    print(f"\n{'='*60}")
    print("Merge complete!")
    print(f"{'='*60}")
    print(f"Output: {output}")
    print(f"\nTotal images:")
    print(f"  train: {stats['train']}")
    print(f"  valid: {stats['valid']}")
    print(f"  test:  {stats['test']}")
    print(f"\nLabel name: 'tennisball' (class 0)")
    print(f"\nNote: Both datasets used class ID 0, so labels are directly compatible.")


if __name__ == "__main__":
    merge_tennis_datasets()

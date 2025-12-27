"""
Remap dataset labels to COCO class IDs for finetuning on pretrained YOLO.

Mapping:
- tennisball (0) -> sports ball (32)
- plastic-bottle (1) -> bottle (39)
- person (2) -> person (0)

This allows finetuning to enhance detection of these 3 classes
while preserving all 80 COCO class detection capabilities.
"""

import os
import shutil
from pathlib import Path
from tqdm import tqdm

# COCO class mapping
# Your dataset class ID -> COCO class ID
LABEL_MAPPING = {
    0: 32,  # tennisball -> sports ball
    1: 39,  # plastic-bottle -> bottle
    2: 0,   # person -> person
}

# Full COCO class names (80 classes)
COCO_NAMES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
    'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat',
    'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
    'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
    'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
    'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
    'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator',
    'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]


def remap_label_file(src_path: Path, dst_path: Path):
    """Remap a single label file to COCO class IDs."""
    with open(src_path, 'r') as f:
        lines = f.readlines()
    
    remapped_lines = []
    for line in lines:
        parts = line.strip().split()
        if len(parts) >= 5:  # class_id, x, y, w, h
            old_class_id = int(parts[0])
            if old_class_id in LABEL_MAPPING:
                new_class_id = LABEL_MAPPING[old_class_id]
                parts[0] = str(new_class_id)
                remapped_lines.append(' '.join(parts) + '\n')
            else:
                print(f"Warning: Unknown class ID {old_class_id} in {src_path}")
    
    with open(dst_path, 'w') as f:
        f.writelines(remapped_lines)


def create_coco_data_yaml(output_dir: Path):
    """Create data.yaml with all 80 COCO class names."""
    yaml_content = f"""# COCO-mapped dataset for finetuning
# Original classes mapped to COCO IDs:
#   tennisball -> sports ball (32)
#   plastic-bottle -> bottle (39)
#   person -> person (0)

train: train/images
val: valid/images

nc: 80
names: {COCO_NAMES}
"""
    yaml_path = output_dir / 'data.yaml'
    with open(yaml_path, 'w') as f:
        f.write(yaml_content)
    print(f"Created {yaml_path}")


def remap_dataset(src_dir: str, dst_dir: str):
    """
    Remap entire dataset to COCO class IDs.
    
    Args:
        src_dir: Source dataset directory (merged_dataset)
        dst_dir: Destination directory for remapped dataset
    """
    src_path = Path(src_dir)
    dst_path = Path(dst_dir)
    
    # Create output directory
    dst_path.mkdir(parents=True, exist_ok=True)
    
    for split in ['train', 'valid']:
        src_images = src_path / split / 'images'
        src_labels = src_path / split / 'labels'
        dst_images = dst_path / split / 'images'
        dst_labels = dst_path / split / 'labels'
        
        if not src_images.exists() or not src_labels.exists():
            print(f"Skipping {split}: directories not found")
            continue
        
        # Create output directories
        dst_images.mkdir(parents=True, exist_ok=True)
        dst_labels.mkdir(parents=True, exist_ok=True)
        
        # Copy images (symlink to save space, or copy)
        print(f"\nProcessing {split} images...")
        image_files = list(src_images.glob('*'))
        for img_file in tqdm(image_files, desc=f"Copying {split} images"):
            dst_img = dst_images / img_file.name
            if not dst_img.exists():
                shutil.copy2(img_file, dst_img)
        
        # Remap labels
        print(f"Remapping {split} labels...")
        label_files = list(src_labels.glob('*.txt'))
        for label_file in tqdm(label_files, desc=f"Remapping {split} labels"):
            dst_label = dst_labels / label_file.name
            remap_label_file(label_file, dst_label)
    
    # Create data.yaml
    create_coco_data_yaml(dst_path)
    
    print(f"\n{'='*60}")
    print("Label remapping complete!")
    print(f"{'='*60}")
    print(f"Output directory: {dst_path}")
    print(f"\nClass mapping applied:")
    print(f"  tennisball (0) -> sports ball (32)")
    print(f"  plastic-bottle (1) -> bottle (39)")
    print(f"  person (2) -> person (0)")
    print(f"\nNext step: Run training with the remapped dataset:")
    print(f"  python train_coco_finetune.py")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Remap dataset labels to COCO class IDs")
    parser.add_argument('--src', type=str, default='merged_dataset',
                        help='Source dataset directory')
    parser.add_argument('--dst', type=str, default='coco_mapped_dataset',
                        help='Destination directory for remapped dataset')
    
    args = parser.parse_args()
    
    remap_dataset(args.src, args.dst)

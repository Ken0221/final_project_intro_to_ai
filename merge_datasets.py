#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Merge three datasets: Tennis Balls, Plastic Bottles, People
"""
import os
import shutil
import sys
from pathlib import Path

# Fix encoding for Windows
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# 定义合并后的类别映射
# 新的类别ID: 0=tennisball, 1=plastic-bottle, 2=person
CLASS_MAPPING = {
    'tennis_dataset': {
        'original_classes': ['tennisball'],
        'new_ids': [0]  # tennisball -> 0
    },
    'plastic_bottles': {
        'original_classes': ['plastic-bottle'],
        'new_ids': [1]  # plastic-bottle -> 1
    },
    'people': {
        'original_classes': ['Human', 'Human '],  # 注意有两个类别,都映射为 person
        'new_ids': [2, 2]  # Human -> 2, Human  -> 2
    }
}

def merge_datasets():
    """合并数据集"""

    # 创建合并后的数据集目录
    output_dir = Path('merged_dataset')

    for split in ['train', 'valid']:
        (output_dir / split / 'images').mkdir(parents=True, exist_ok=True)
        (output_dir / split / 'labels').mkdir(parents=True, exist_ok=True)

    print("="*70)
    print("Merging datasets...")
    print("="*70)

    stats = {
        'train': {'images': 0, 'labels': 0},
        'valid': {'images': 0, 'labels': 0}
    }

    # Process each dataset
    for dataset_name, mapping in CLASS_MAPPING.items():
        print(f"\nProcessing {dataset_name}...")

        dataset_path = Path(dataset_name)

        for split in ['train', 'valid']:
            images_src = dataset_path / split / 'images'
            labels_src = dataset_path / split / 'labels'

            images_dst = output_dir / split / 'images'
            labels_dst = output_dir / split / 'labels'

            if not images_src.exists():
                print(f"  Warning: {images_src} does not exist, skipping")
                continue

            # 复制图片和标签
            image_files = list(images_src.glob('*'))

            for img_file in image_files:
                # 复制图片
                dst_img = images_dst / f"{dataset_name}_{img_file.name}"
                shutil.copy2(img_file, dst_img)
                stats[split]['images'] += 1

                # 处理对应的标签文件
                label_file = labels_src / (img_file.stem + '.txt')

                if label_file.exists():
                    dst_label = labels_dst / f"{dataset_name}_{img_file.stem}.txt"

                    # 读取并重新映射类别ID
                    with open(label_file, 'r') as f:
                        lines = f.readlines()

                    new_lines = []
                    for line in lines:
                        parts = line.strip().split()
                        if len(parts) >= 5:  # class_id x y w h
                            old_class_id = int(parts[0])

                            # 映射到新的类别ID
                            if old_class_id < len(mapping['new_ids']):
                                new_class_id = mapping['new_ids'][old_class_id]
                                new_line = f"{new_class_id} " + " ".join(parts[1:]) + "\n"
                                new_lines.append(new_line)

                    # 写入新的标签文件
                    with open(dst_label, 'w') as f:
                        f.writelines(new_lines)

                    stats[split]['labels'] += 1

            print(f"  {split}: {len(image_files)} images")

    print("\n" + "="*70)
    print("Merge completed!")
    print("="*70)
    print(f"\nTrain set: {stats['train']['images']} images, {stats['train']['labels']} labels")
    print(f"Valid set: {stats['valid']['images']} images, {stats['valid']['labels']} labels")

    # 创建 data.yaml
    data_yaml = output_dir / 'data.yaml'
    with open(data_yaml, 'w') as f:
        f.write(f"""train: train/images
val: valid/images

nc: 3
names: ['tennisball', 'plastic-bottle', 'person']

# 数据集来源
# - Tennis balls: 80 train, 27 valid
# - Plastic bottles: 1000 train, 247 valid
# - People: 10933 train, 1570 valid
""")

    print(f"\nDataset config saved to: {data_yaml}")
    print("\nNew class mapping:")
    print("  0: tennisball")
    print("  1: plastic-bottle")
    print("  2: person")

    return output_dir

if __name__ == '__main__':
    merged_dir = merge_datasets()
    print(f"\nMerged dataset location: {merged_dir.absolute()}")
    print("\nNext step: Run 'python train.py' to start training")

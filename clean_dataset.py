import os
import glob

def clean_dataset(dataset_path):
    splits = ['train', 'valid']
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.dng', '.webp', '.mpo']
    
    for split in splits:
        label_dir = os.path.join(dataset_path, split, 'labels')
        image_dir = os.path.join(dataset_path, split, 'images')
        
        if not os.path.exists(label_dir):
            print(f"Label directory not found: {label_dir}")
            continue
            
        if not os.path.exists(image_dir):
            print(f"Image directory not found: {image_dir}")
            continue
            
        print(f"Cleaning {split} set...")
        label_files = glob.glob(os.path.join(label_dir, '*.txt'))
        
        deleted_count = 0
        for label_path in label_files:
            filename = os.path.basename(label_path)
            stem = os.path.splitext(filename)[0]
            
            # Check if corresponding image exists
            image_exists = False
            for ext in image_extensions:
                image_path = os.path.join(image_dir, stem + ext)
                if os.path.exists(image_path):
                    image_exists = True
                    break
            
            if not image_exists:
                print(f"Removing orphan label: {label_path}")
                try:
                    os.remove(label_path)
                    deleted_count += 1
                except OSError as e:
                    print(f"Error deleting {label_path}: {e}")
        
        print(f"Finished cleaning {split} set. Removed {deleted_count} orphan label files.")

if __name__ == "__main__":
    dataset_path = 'tennis_custom_2_dataset'
    if os.path.exists(dataset_path):
        clean_dataset(dataset_path)
    else:
        print(f"Dataset directory '{dataset_path}' not found.")

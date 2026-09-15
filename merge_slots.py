"""
Script để gộp Left_Slot (class 3) và Right_Slot (class 5) thành Empty_Slot
"""
import os
import shutil
from pathlib import Path

def remap_labels(label_file, class_mapping):
    """
    Đọc file label và remap class indices

    Args:
        label_file: Đường dẫn file label
        class_mapping: Dict mapping old_class_id -> new_class_id
    """
    with open(label_file, 'r') as f:
        lines = f.readlines()

    new_lines = []
    for line in lines:
        parts = line.strip().split()
        if len(parts) >= 5:  # YOLO format: class x y w h
            old_class = int(parts[0])
            new_class = class_mapping.get(old_class, old_class)
            parts[0] = str(new_class)
            new_lines.append(' '.join(parts) + '\n')

    return new_lines

def process_dataset():
    """
    Xử lý toàn bộ dataset để merge classes
    """
    # Class mapping:
    # 0: Case -> 0
    # 1: Hand -> 1
    # 2: Left_Earbud -> 2
    # 3: Left_Slot -> 3 (Empty_Slot)
    # 4: Right_Earbud -> 4
    # 5: Right_Slot -> 3 (Empty_Slot) <- MERGE VÀO CLASS 3

    class_mapping = {
        0: 0,  # Case
        1: 1,  # Hand
        2: 2,  # Left_Earbud
        3: 3,  # Left_Slot -> Empty_Slot
        4: 4,  # Right_Earbud
        5: 3,  # Right_Slot -> Empty_Slot (MERGE)
    }

    # Backup original dataset
    backup_dir = Path('dataset_backup')
    if not backup_dir.exists():
        print("Backing up original dataset...")
        backup_dir.mkdir()
        for split in ['train', 'valid', 'test']:
            split_backup = backup_dir / split
            if Path(split).exists():
                shutil.copytree(split, split_backup)
        print(f"Backup saved to: {backup_dir}")
    else:
        print(f"Backup already exists: {backup_dir}")

    # Process each split
    total_files = 0
    total_modified = 0

    for split in ['train', 'valid', 'test']:
        labels_dir = Path(split) / 'labels'
        if not labels_dir.exists():
            print(f"Warning: Not found: {labels_dir}")
            continue

        print(f"\nProcessing {split} set...")
        label_files = list(labels_dir.glob('*.txt'))

        for label_file in label_files:
            total_files += 1
            new_lines = remap_labels(label_file, class_mapping)

            # Overwrite file with new labels
            with open(label_file, 'w') as f:
                f.writelines(new_lines)

            total_modified += 1

            if total_modified % 50 == 0:
                print(f"  Processed {total_modified} files...")

        print(f"Done {split}: {len(label_files)} files")

    print(f"\n{'='*50}")
    print(f"Processed all {total_files} label files")
    print(f"Left_Slot and Right_Slot merged into Empty_Slot (class 3)")
    print(f"{'='*50}")

def create_new_data_yaml():
    """
    Tạo file data.yaml mới với class structure mới
    """
    yaml_content = """train: ../train/images
val: ../valid/images
test: ../test/images

nc: 5
names: ['Case', 'Hand', 'Left_Earbud', 'Empty_Slot', 'Right_Earbud']

roboflow:
  workspace: traigautchin1983-gmail-com
  project: case-earbud-empty-slot
  version: 3
  license: CC BY 4.0
  url: https://universe.roboflow.com/traigautchin1983-gmail-com/case-earbud-empty-slot/dataset/3
  note: Modified - Left_Slot and Right_Slot merged into Empty_Slot
"""

    # Backup original data.yaml
    if os.path.exists('data.yaml') and not os.path.exists('data.yaml.backup'):
        shutil.copy('data.yaml', 'data.yaml.backup')
        print("\nBackup original data.yaml -> data.yaml.backup")

    # Write new data.yaml
    with open('data.yaml', 'w') as f:
        f.write(yaml_content)

    print("Created new data.yaml with 5 classes")

def main():
    print("="*50)
    print("MERGE LEFT_SLOT & RIGHT_SLOT -> EMPTY_SLOT")
    print("="*50)

    # Process labels
    process_dataset()

    # Create new data.yaml
    create_new_data_yaml()

    print("\n" + "="*50)
    print("COMPLETED!")
    print("="*50)
    print("\nNew class structure:")
    print("  0: Case")
    print("  1: Hand")
    print("  2: Left_Earbud")
    print("  3: Empty_Slot (Left_Slot + Right_Slot)")
    print("  4: Right_Earbud")
    print("\nYou can train the model with: python train_local.py")
    print("="*50)

if __name__ == '__main__':
    main()

import json
import os
import shutil
import sys
import zipfile
from pathlib import Path

def fix_dataset_labels(zip_path: str):
    zip_file = Path(zip_path)
    if not zip_file.is_file():
        print(f"Error: Not found {zip_path}")
        return

    print(f"Processing {zip_file.name}...")
    temp_dir = zip_file.parent / (zip_file.stem + "_temp_extract")
    fixed_zip_path = zip_file.parent / (zip_file.stem.replace("_fixed", "") + "_final.zip")
    
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
    temp_dir.mkdir(parents=True)
    
    print(f"Extracting to {temp_dir.name}...")
    with zipfile.ZipFile(zip_file, 'r') as zf:
        zf.extractall(temp_dir)
        
    print("Fixing labels in annotations...")
    
    # Valid geometry classes mapped to lowercase
    VALID_CLASSES = {
        'open_case', 'close_case', 'left_earbud', 'right_earbud',
        'empty_left', 'empty_right'
    }

    for root, dirs, files in os.walk(temp_dir):
        for file in files:
            if file == "_annotations.coco.json":
                json_path = Path(root) / file
                with open(json_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # 1. Fix categories
                new_categories = []
                valid_category_ids = set()
                
                for cat in data.get("categories", []):
                    old_name = cat.get("name", "").strip()
                    new_name = old_name.lower() # Convert to lowercase
                    
                    if new_name in VALID_CLASSES:
                        cat["name"] = new_name
                        new_categories.append(cat)
                        valid_category_ids.add(cat["id"])
                    else:
                        print(f"  Removing invalid category: {old_name} (ID: {cat['id']})")
                
                data["categories"] = new_categories
                
                # 2. Filter annotations
                old_ann_count = len(data.get("annotations", []))
                new_annotations = [
                    ann for ann in data.get("annotations", [])
                    if ann["category_id"] in valid_category_ids
                ]
                data["annotations"] = new_annotations
                
                print(f"  Removed {old_ann_count - len(new_annotations)} invalid annotations in {json_path.relative_to(temp_dir)}")

                with open(json_path, 'w', encoding='utf-8') as f:
                    json.dump(data, f, ensure_ascii=False)

    # Zip it back up
    print(f"Creating new zip: {fixed_zip_path.name}...")
    if fixed_zip_path.exists():
        fixed_zip_path.unlink()
        
    shutil.make_archive(str(fixed_zip_path.with_suffix('')), 'zip', root_dir=temp_dir)
    shutil.rmtree(temp_dir)
    print(f"\nDone! You can now use: {fixed_zip_path.name}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python fix_dataset_labels.py <path_to_zip>")
    else:
        fix_dataset_labels(sys.argv[1])

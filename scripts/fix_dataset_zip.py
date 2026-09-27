import json
import os
import shutil
import sys
import zipfile
from pathlib import Path

def fix_dataset_zip(zip_path: str):
    zip_file = Path(zip_path)
    if not zip_file.is_file():
        print(f"Error: Not found {zip_path}")
        return

    print(f"Processing {zip_file.name}...")
    temp_dir = zip_file.parent / (zip_file.stem + "_temp_extract")
    fixed_zip_path = zip_file.parent / (zip_file.stem + "_fixed.zip")
    
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
    temp_dir.mkdir(parents=True)
    
    # 1. Extract the zip
    print(f"Extracting to {temp_dir.name}...")
    with zipfile.ZipFile(zip_file, 'r') as zf:
        zf.extractall(temp_dir)
        
    # 2. Rename files with '&' and update json
    print("Fixing filenames and annotations...")
    for root, dirs, files in os.walk(temp_dir):
        # Fix json first if it exists in this folder
        for file in files:
            if file == "_annotations.coco.json":
                json_path = Path(root) / file
                with open(json_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                changed = False
                for img in data.get("images", []):
                    old_name = img.get("file_name", "")
                    if "&" in old_name:
                        new_name = old_name.replace("&", "and")
                        img["file_name"] = new_name
                        changed = True
                
                if changed:
                    with open(json_path, 'w', encoding='utf-8') as f:
                        json.dump(data, f, ensure_ascii=False)
                    print(f"Updated JSON: {json_path.relative_to(temp_dir)}")

        # Rename files that contain '&'
        for file in files:
            if "&" in file:
                old_path = Path(root) / file
                new_name = file.replace("&", "and")
                new_path = Path(root) / new_name
                old_path.rename(new_path)
                print(f"Renamed: {file} -> {new_name}")
                
    # 3. Zip it back up
    print(f"Creating new zip: {fixed_zip_path.name}...")
    if fixed_zip_path.exists():
        fixed_zip_path.unlink()
        
    shutil.make_archive(str(fixed_zip_path.with_suffix('')), 'zip', root_dir=temp_dir)
    
    # 4. Clean up temp dir
    shutil.rmtree(temp_dir)
    print(f"\nDone! You can now use: {fixed_zip_path.name}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python fix_dataset_zip.py <path_to_zip>")
    else:
        fix_dataset_zip(sys.argv[1])

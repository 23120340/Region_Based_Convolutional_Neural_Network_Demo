"""
Script don dep files khong can thiet
"""
import os
import shutil

def cleanup():
    """Xoa cac file khong can thiet"""

    print("="*60)
    print("CLEANING UP UNNECESSARY FILES")
    print("="*60)

    # Danh sach thu muc va file can xoa
    to_delete = [
        # Video trung gian
        "output_video.mp4",
        "output_video_local.mp4",
        "output_video_fixed.mp4",
        "output_result.avi",
        # Chi giu lai video cuoi cung
        # "output_video_improved.mp4" - GIU LAI

        # Training runs cu (chi giu v3_fixed - model tot nhat)
        "runs/detect/runs/train/earbud_detection",
        "runs/detect/runs/train/earbud_detection_v2",
        "runs/detect/runs/train/earbud_fast",
        # "runs/detect/runs/train/earbud_empty_slot_fix" - GIU LAI

        # Analysis/validation runs
        "runs/analysis",
        "runs/predict_video",
        "runs/detect/val",
        "runs/detect/val-2",
        "runs/detect/val-3",
        "runs/detect/val-4",

        # Backup dataset (da merge)
        "dataset_backup",

        # Scripts tam thoi
        "analyze_errors.py",
        "analyze_model.py",
        "compare_models.py",
        "create_video_avi.py",
        "debug_empty_slot.py",
        "monitor_training.py",
        "run_inference_video_local.py",
        "run_inference_yolo.py",
        "test_models_comparison.py",
        "train_improved.py",
        "train_fix_empty_slot.py",
        "train_fast.py",
    ]

    deleted_count = 0
    freed_space = 0

    for item in to_delete:
        if not os.path.exists(item):
            continue

        try:
            if os.path.isfile(item):
                size = os.path.getsize(item)
                os.remove(item)
                print(f"[OK] Deleted file: {item} ({size/1024/1024:.1f} MB)")
                deleted_count += 1
                freed_space += size
            elif os.path.isdir(item):
                # Tinh dung luong truoc khi xoa
                size = sum(
                    os.path.getsize(os.path.join(dirpath, filename))
                    for dirpath, dirnames, filenames in os.walk(item)
                    for filename in filenames
                )
                shutil.rmtree(item)
                print(f"[OK] Deleted directory: {item} ({size/1024/1024:.1f} MB)")
                deleted_count += 1
                freed_space += size
        except Exception as e:
            print(f"[FAIL] Failed to delete {item}: {e}")

    print("\n" + "="*60)
    print(f"CLEANUP SUMMARY")
    print("="*60)
    print(f"Deleted items: {deleted_count}")
    print(f"Freed space: {freed_space/1024/1024:.1f} MB")

    print("\n" + "="*60)
    print("KEPT FILES:")
    print("="*60)
    kept_files = [
        "runs/detect/runs/train/earbud_empty_slot_fix/weights/best.pt",
        "output_video_improved.mp4",
        "train/", "valid/", "test/",
        "data.yaml",
        "merge_slots.py",
        "run_inference_improved.py"
    ]
    for f in kept_files:
        if os.path.exists(f):
            print(f"[KEPT] {f}")

if __name__ == '__main__':
    cleanup()

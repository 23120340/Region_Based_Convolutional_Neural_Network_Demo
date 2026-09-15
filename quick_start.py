"""
Quick start script - Train va test model tren video
"""
import subprocess
import sys
import os

def run_command(cmd, description):
    """Run a command and handle errors"""
    print("\n" + "="*60)
    print(f"STEP: {description}")
    print("="*60)
    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0:
        print(f"\nError in: {description}")
        return False
    return True

def main():
    print("="*60)
    print("QUICK START: TRAIN AND TEST ON VIDEO")
    print("="*60)
    print("\nThis script will:")
    print("  1. Train YOLOv8 model on your dataset")
    print("  2. Run inference on video")
    print("\nEstimated time: 30-60 minutes (depending on your hardware)")
    print("\nPress Ctrl+C to cancel...")

    try:
        input("\nPress Enter to continue...")
    except KeyboardInterrupt:
        print("\n\nCancelled by user.")
        return

    # Step 1: Train model
    if not run_command("python train_local.py", "Training model"):
        return

    # Step 2: Check if model exists
    model_path = "runs/train/earbud_detection/weights/best.pt"
    if not os.path.exists(model_path):
        print(f"\nError: Model not found at {model_path}")
        print("Training may have failed. Check the output above.")
        return

    print(f"\n✓ Model trained successfully: {model_path}")

    # Step 3: Run inference on video
    if not run_command("python run_inference_video_local.py", "Running inference on video"):
        return

    print("\n" + "="*60)
    print("✓ ALL DONE!")
    print("="*60)
    print("\nResults:")
    print(f"  - Trained model: {model_path}")
    print(f"  - Training plots: runs/train/earbud_detection/")
    print(f"  - Output video: C:\\RNN\\output_video_local.mp4")
    print("\nNext steps:")
    print("  - Review training plots in runs/train/earbud_detection/")
    print("  - Watch output video to check detection quality")
    print("  - Run 'python test_model.py --model {model_path} --test-set' for detailed metrics")
    print("="*60)

if __name__ == '__main__':
    main()

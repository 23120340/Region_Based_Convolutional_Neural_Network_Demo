# Earbud Detection with YOLOv8

This project trains and runs a custom YOLOv8 object detection model for detecting earbuds inside a charging case, including the case body, hands, earbuds, and empty slots.

## Overview

The model is designed for:
- detecting the charging case
- detecting the human hand
- identifying left and right earbuds
- identifying empty slots in the case
- running inference on local video files

## Supported classes
- Case
- Hand
- Left_Earbud
- Left_Slot
- Right_Earbud
- Right_Slot

## Project status
- Model family: YOLOv8
- Default backbone: `yolov8n.pt`
- Best reported mAP50: around `0.830`
- Training workflow: local training with Ultralytics YOLOv8

## Repository structure
```text
RNN/
├── README.md
├── README_TRAINING.md
├── data.yaml
├── train_local.py
├── quick_test.py
├── test_model.py
├── run_inference_improved.py
├── run_inference_video.py
├── merge_slots.py
├── cleanup.py
├── quick_start.py
├── train/
│   ├── images/
│   └── labels/
├── valid/
│   ├── images/
│   └── labels/
├── test/
│   ├── images/
│   └── labels/
├── runs/
│   └── detect/
├── yolov8n.pt
└── .gitignore
```

## Requirements

Install dependencies:
```bash
pip install ultralytics opencv-python pandas numpy
```

If you want to use a GPU-enabled environment, make sure CUDA is installed and available to PyTorch.

## Quick start

### 1) Train model
```bash
python train_local.py
```

### 2) Run a quick test
```bash
python quick_test.py
```

### 3) Run inference on a video
```bash
python run_inference_improved.py
```

### 4) Evaluate the model
```bash
python test_model.py --model runs/train/earbud_detection/weights/best.pt --test-set
```

## Dataset format
The dataset follows YOLO format structure:
```text
train/
  images/
  labels/
valid/
  images/
  labels/
test/
  images/
  labels/
```

The annotations are standard YOLO `.txt` label files with normalized coordinates.

## Key scripts
- `train_local.py`: local training entry point
- `quick_test.py`: fast validation on sample images
- `test_model.py`: full evaluation and image/video testing
- `run_inference_improved.py`: detection on a supplied video
- `merge_slots.py`: merges or normalizes slot annotations
- `cleanup.py`: cleanup utilities for dataset preparation

## Usage notes
- Update `data.yaml` if your dataset paths or class names are changed.
- For larger or more accurate models, switch from `yolov8n.pt` to `yolov8s.pt`, `yolov8m.pt`, or larger variants.
- If you run out of VRAM, reduce `batch` size or image resolution in the training script.

## Training history
- v1: initial model baseline
- v2_fast: faster training optimization
- v3_fixed: improved empty-slot and detection behavior

## More details
For a full training checklist, troubleshooting guide, and example commands, see [README_TRAINING.md](README_TRAINING.md).

## License
The project code is provided as-is for research and demo usage. Please check the dataset source license before any public deployment or commercial use.


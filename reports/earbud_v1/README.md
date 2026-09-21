# Earbud v1 baseline

Baseline được đóng băng lúc **2026-09-21T03:25:43-04:00**.

## Nhận dạng code

- Branch: `main`.
- Git HEAD: `18347cb18dca53e640a29b50931579056effecb0`.
- Working tree tại thời điểm chụp: **không sạch**, có thay đổi staged và unstaged.
- Source snapshot SHA-256: `001df3fe5040df7f48ccc184481a87cce766c7943cc00f0cf00d83279c66f32b`.

Không được dùng riêng Git HEAD để khẳng định đã tái tạo chính xác baseline này. Xem `baseline_manifest.json` và `checksums.sha256`.

## Nội dung sao lưu

```text
artifacts/baselines/earbud_v1/
├── action_model/
├── earbud_merged_detector/
├── annotations.csv
├── configs/
├── pyproject.toml
├── requirements-camera.txt
└── requirements-ml.txt
```

Tổng cộng 32 file, 26.392.402 byte. Checksum của checkpoint và annotation nguồn trùng với bản sao.

## Seed và cấu hình chính

| Thành phần | Seed | Cấu hình |
|---|---:|---|
| Action model | 42 | ViT Base, 10 FPS, sequence 16, BiLSTM 2 lớp, 30 epoch |
| YOLO detector | 0 | `yolo26n.pt`, 640 px, batch 8, 60 epoch, deterministic |

## Máy kiểm tra hiện tại

- Host: `debian`.
- OS: Debian GNU/Linux, kernel `6.12.101+deb13-amd64`.
- CPU: Intel Core i7-11850HE, 8 core/16 logical CPU.
- RAM: khoảng 15 GiB.
- Display adapter nhìn thấy: Intel TigerLake-H GT1 UHD Graphics.
- NVIDIA GPU/CUDA: không phát hiện trong môi trường hiện tại.
- Python: 3.13.5.
- Torch, Ultralytics, Transformers, OpenCV, NumPy, scikit-learn, PyYAML và Pillow: chưa cài trong Python hiện tại.

Đây là máy/môi trường dùng để **kiểm tra và đóng băng baseline**, chưa chắc là môi trường Windows đã dùng để train checkpoint cũ. Artifact cũ không lưu phiên bản thư viện, CUDA hoặc GPU gốc, nên các giá trị đó được ghi là chưa biết thay vì suy đoán.

## Version v2

Các lần train mới phải dùng:

- Action output: `artifacts/action_model_v2/`.
- Detector output/name: `artifacts/training/earbud_v2_detector/`.
- Event log: `artifacts/events/earbud_v2.jsonl`.
- Project profile: `configs/projects/earbud_v2.json`.

Profile v1 và toàn bộ bản sao trong `artifacts/baselines/earbud_v1` không được ghi đè.

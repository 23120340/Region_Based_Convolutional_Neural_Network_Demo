$ErrorActionPreference = "Stop"

Set-Location (Split-Path -Parent $PSScriptRoot)

if (Get-Command py -ErrorAction SilentlyContinue) {
    py -3.11 -m venv .venv
} elseif (Get-Command python -ErrorAction SilentlyContinue) {
    python -m venv .venv
} else {
    throw "Khong tim thay Python. Hay cai Python 3.10 tro len."
}

$Python = Join-Path $PWD ".venv\Scripts\python.exe"
& $Python -m pip install --upgrade pip
& $Python -m pip install -r requirements-camera.txt

& $Python -c "import cv2, torch, ultralytics; print('OpenCV:', cv2.__version__); print('Ultralytics:', ultralytics.__version__); print('PyTorch:', torch.__version__); print('CUDA:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"

Write-Host "`nCai dat xong. Chay camera bang:"
Write-Host ".\.venv\Scripts\python.exe scripts\run_earbud.py --mode camera --source 0"

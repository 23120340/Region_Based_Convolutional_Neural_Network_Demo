# ============================================================
# run_hybrid.ps1 - Khoi dong he thong YOLO + ViT-BiLSTM
# Chay tu thu muc goc du an:
#   cd "G:\Internship\RBCNN_Demo"
#   .\run_hybrid.ps1
#   .\run_hybrid.ps1 -Source phone   # chi dung camera dien thoai qua QR
# ============================================================

param(
    [string]$Source = "0",
    [string]$Device = "",
    [switch]$Fullscreen,
    [switch]$Mirror,
    [switch]$AllowDownload
)

$PROJECT_ROOT = Split-Path -Parent $MyInvocation.MyCommand.Path
$VENV_PYTHON  = Join-Path $PROJECT_ROOT ".venv\Scripts\python.exe"
$SCRIPT       = Join-Path $PROJECT_ROOT "scripts\run_hybrid.py"
$PROJECT_JSON = Join-Path $PROJECT_ROOT "configs\projects\earbud_v2.json"
$YOLO_WEIGHTS = Join-Path $PROJECT_ROOT "artifacts\training\earbud_geometry_detector\weights\best.pt"
$LSTM_WEIGHTS = $null
if (Test-Path -LiteralPath $PROJECT_JSON) {
    $profile = Get-Content -LiteralPath $PROJECT_JSON -Raw -Encoding UTF8 | ConvertFrom-Json
    # Ignored weights are shared when switching branches; the profile selects
    # the branch-specific checkpoint instead of a hard-coded pilot filename.
    $LSTM_WEIGHTS = Join-Path $PROJECT_ROOT $profile.action_model
}

Write-Host ""
Write-Host "======================================================" -ForegroundColor Cyan
Write-Host "  Earbud Assembly Monitor - YOLO + ViT-BiLSTM Hybrid " -ForegroundColor Cyan
Write-Host "======================================================" -ForegroundColor Cyan
Write-Host ""

$ok = $true

if (-not (Test-Path $VENV_PYTHON)) {
    Write-Host "[ERROR] Khong tim thay Python venv: $VENV_PYTHON" -ForegroundColor Red
    Write-Host "        Chay truoc: python -m venv .venv" -ForegroundColor Yellow
    Write-Host "        Sau do:     .venv\Scripts\pip install -r requirements-camera.txt" -ForegroundColor Yellow
    $ok = $false
}

if (-not (Test-Path $YOLO_WEIGHTS)) {
    Write-Host "[ERROR] Khong tim thay YOLO checkpoint:" -ForegroundColor Red
    Write-Host "        $YOLO_WEIGHTS" -ForegroundColor Red
    Write-Host "        Hay tai best_earbud_detector.pt tu Kaggle va copy vao duong dan tren." -ForegroundColor Yellow
    $ok = $false
} else {
    Write-Host "[OK] YOLO weights  : $YOLO_WEIGHTS" -ForegroundColor Green
}

if (-not $LSTM_WEIGHTS -or -not (Test-Path -LiteralPath $LSTM_WEIGHTS)) {
    Write-Host "[ERROR] Khong tim thay BiLSTM checkpoint:" -ForegroundColor Red
    Write-Host "        $LSTM_WEIGHTS" -ForegroundColor Red
    $ok = $false
} else {
    Write-Host "[OK] LSTM weights  : $LSTM_WEIGHTS" -ForegroundColor Green
}

if (-not (Test-Path $PROJECT_JSON)) {
    Write-Host "[ERROR] Khong tim thay project config:" -ForegroundColor Red
    Write-Host "        $PROJECT_JSON" -ForegroundColor Red
    $ok = $false
} else {
    Write-Host "[OK] Project config: $PROJECT_JSON" -ForegroundColor Green
}

if (-not $ok) {
    Write-Host ""
    Write-Host "Pre-flight FAILED. Sua cac loi tren truoc khi chay lai." -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "Phim dieu khien trong cua so camera:" -ForegroundColor Cyan
Write-Host "  Q / ESC  - Thoat"
Write-Host "  R        - Reset FSM + Fusion + LSTM buffer"
Write-Host "  F        - Bat/tat toan man hinh"
Write-Host "  P        - Ket noi camera dien thoai (hien QR); N: QR moi; D: ngat"
Write-Host "  Bat dau voi hop mo rong va hai khe nhin ro."
Write-Host ""
Write-Host "Dang khoi dong..." -ForegroundColor Yellow
Write-Host ""

$runtimeArgs = @("--project", $PROJECT_JSON, "--source", $Source)
if (-not $Mirror) { $runtimeArgs += "--no-mirror" }
if ($Fullscreen) { $runtimeArgs += "--fullscreen" }
if ($Device) { $runtimeArgs += @("--device", $Device) }
if ($AllowDownload) { $runtimeArgs += "--allow-download" }
& $VENV_PYTHON $SCRIPT @runtimeArgs

if ($LASTEXITCODE -ne 0) {
    Write-Host ""
    Write-Host "[WARN] Chuong trinh thoat voi ma loi: $LASTEXITCODE" -ForegroundColor Yellow
}
exit $LASTEXITCODE

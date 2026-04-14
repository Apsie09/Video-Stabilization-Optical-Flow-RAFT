#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${1:-raft-gpu}"

if ! command -v conda >/dev/null 2>&1; then
  echo "conda is not available in PATH."
  exit 1
fi

eval "$(conda shell.bash hook)"

echo "[1/5] Creating conda env: ${ENV_NAME}"
conda create -n "${ENV_NAME}" python=3.10 pip -y

echo "[2/5] Activating env: ${ENV_NAME}"
conda activate "${ENV_NAME}"

echo "[3/5] Installing PyTorch with CUDA 12.1 (pip wheels)"
python -m pip install --upgrade pip
python -m pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121

echo "[4/5] Installing project dependencies"
python -m pip install -r requirements-gpu.txt

echo "[5/5] Sanity check"
echo
echo "Environment ready: ${ENV_NAME}"
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda runtime:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))
PY

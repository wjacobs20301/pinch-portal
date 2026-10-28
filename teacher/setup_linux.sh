#!/usr/bin/env bash
# One-time setup on the Linux (Bazzite) desktop. Run from the teacher/ folder:  ./setup_linux.sh
# Everything installs into your home folder / this folder — nothing system-wide, so it works on Bazzite's
# read-only (immutable) system image without layering packages.
set -euo pipefail
cd "$(dirname "$0")"

# uv: fast Python + package manager, installs to ~/.local/bin
if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi

uv venv --python 3.12 .venv

# Pick the PyTorch build for this GPU
if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1; then
  echo "NVIDIA GPU detected"; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
  TORCH_INDEX="https://download.pytorch.org/whl/cu128"
elif ls /dev/kfd >/dev/null 2>&1; then
  echo "AMD GPU detected (ROCm)"
  TORCH_INDEX="https://download.pytorch.org/whl/rocm6.4"
else
  echo "No supported GPU found — this would run on CPU and be far too slow."; exit 1
fi

uv pip install --python .venv/bin/python torch torchvision --index-url "$TORCH_INDEX"
uv pip install --python .venv/bin/python diffusers transformers accelerate safetensors pillow \
  opencv-python-headless controlnet_aux peft remotezip onnx onnxscript

.venv/bin/python - <<'EOF'
import torch
print("torch", torch.__version__, "| GPU available:", torch.cuda.is_available(),
      "|", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")
EOF

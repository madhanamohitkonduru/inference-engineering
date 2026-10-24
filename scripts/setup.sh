#!/usr/bin/env bash
# One-command environment rebuild for a fresh RunPod PyTorch (CUDA 12.8) pod.
# Usage:  bash scripts/setup.sh
# Puts the venv and the Hugging Face cache on /workspace (persistent volume) when it exists.
set -euo pipefail

VLLM_VERSION="${VLLM_VERSION:-0.32.0}"
if [ -d /workspace ] && [ -w /workspace ]; then BASE=/workspace; else BASE="$HOME"; fi
VENV="${VENV:-$BASE/venv}"
export HF_HOME="${HF_HOME:-$BASE/hf-cache}"
echo "BASE=$BASE  VENV=$VENV  HF_HOME=$HF_HOME"

# --- system packages ---
apt-get update -qq
apt-get install -y -qq tmux git curl zip >/dev/null

# --- python env + vLLM ---
[ -d "$VENV" ] || python3 -m venv "$VENV"
# shellcheck disable=SC1091
source "$VENV/bin/activate"
pip install -q uv
uv pip install "vllm==${VLLM_VERSION}" pandas matplotlib
python -c "import vllm; print('vllm', vllm.__version__)"

# --- Nsight Systems CLI (nsys) ---
if ! command -v nsys >/dev/null 2>&1; then
  for pkg in cuda-nsight-systems-12-8 nsight-systems-cli; do
    if apt-get install -y -qq "$pkg" >/dev/null 2>&1; then echo "installed $pkg"; break; fi
  done
  NSYS_BIN="$(find /opt /usr/local -name nsys -type f -perm -u+x 2>/dev/null | head -1 || true)"
  if [ -n "${NSYS_BIN:-}" ] && ! command -v nsys >/dev/null 2>&1; then ln -sf "$NSYS_BIN" /usr/local/bin/nsys; fi
fi
if command -v nsys >/dev/null 2>&1; then nsys --version
else echo "WARNING: nsys not found. Install Nsight Systems manually (see day-04 notes) before the Nsight block."; fi

# --- env file for new shells ---
cat > "$BASE/env.sh" <<ENVEOF
source "$VENV/bin/activate"
export HF_HOME="$HF_HOME"
ENVEOF
echo "Done. In every new terminal run:  source $BASE/env.sh"

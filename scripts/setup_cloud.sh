#!/usr/bin/env bash
#
# Cloud GPU Setup Script for SynthInspect
# Run this on a fresh Vast.ai, RunPod, or Lambda Cloud instance
#
# Usage:
#   chmod +x scripts/setup_cloud.sh
#   ./scripts/setup_cloud.sh
#
set -e

REPO_URL="https://github.com/MohitGoyal09/severstal-steel-defect-detection.git"
PROJECT_DIR="$HOME/severstal-steel-defect-detection"
DATA_DIR="$HOME/data"

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  SynthInspect — Cloud GPU Setup                              ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

# ─── Detect GPU ───
echo "► Detecting GPU..."
if command -v nvidia-smi &>/dev/null; then
	GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
	GPU_MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader | head -1)
	echo "  GPU: $GPU_NAME ($GPU_MEM)"
else
	echo "  WARNING: nvidia-smi not found. GPU may not be available."
fi

# ─── Detect CUDA ───
echo "► Checking CUDA..."
if command -v nvcc &>/dev/null; then
	CUDA_VERSION=$(nvcc --version | grep "release" | sed 's/.*release //' | sed 's/,.*//')
	echo "  CUDA: $CUDA_VERSION"
else
	echo "  WARNING: nvcc not found. PyTorch may use CPU."
fi

# ─── Install system dependencies ───
echo "► Installing system packages..."
apt-get update -qq
apt-get install -y -qq \
	git \
	wget \
	curl \
	unzip \
	build-essential \
	libgl1-mesa-glx \
	libglib2.0-0 \
	libsm6 \
	libxext6 \
	libxrender-dev \
	libgomp1 \
	htop \
	tree \
	>/dev/null 2>&1
echo "  Done."

# ─── Setup Python environment ───
echo "► Setting up Python..."
if ! command -v python3 &>/dev/null; then
	apt-get install -y -qq python3 python3-pip python3-venv
fi

PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
echo "  Python: $PYTHON_VERSION"

# Create virtual environment
VENV_DIR="$HOME/.venvs/severstal"
mkdir -p "$HOME/.venvs"
python3 -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"

# Upgrade pip
pip install -q --upgrade pip setuptools wheel

# ─── Install PyTorch with CUDA ───
echo "► Installing PyTorch (auto-detecting CUDA version)..."

if command -v nvcc &>/dev/null; then
	CUDA_MAJOR=$(nvcc --version | grep "release" | sed 's/.*release //' | sed 's/,.*//' | cut -d. -f1)
	CUDA_MINOR=$(nvcc --version | grep "release" | sed 's/.*release //' | sed 's/,.*//' | cut -d. -f2)

	if [ "$CUDA_MAJOR" = "12" ]; then
		echo "  Installing PyTorch for CUDA 12.x"
		pip install -q torch torchvision torchaudio
	elif [ "$CUDA_MAJOR" = "11" ]; then
		echo "  Installing PyTorch for CUDA 11.8"
		pip install -q torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
	else
		echo "  Unknown CUDA version, installing latest PyTorch"
		pip install -q torch torchvision torchaudio
	fi
else
	echo "  No CUDA detected, installing CPU PyTorch"
	pip install -q torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
fi

# Verify PyTorch sees GPU
python3 -c "import torch; print(f'  PyTorch CUDA available: {torch.cuda.is_available()}'); print(f'  Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU\"}')"

# ─── Clone repository ───
echo "► Cloning repository..."
if [ -d "$PROJECT_DIR" ]; then
	echo "  Repo exists, pulling latest..."
	cd "$PROJECT_DIR"
	git pull
else
	git clone "$REPO_URL" "$PROJECT_DIR"
	cd "$PROJECT_DIR"
fi
echo "  Done."

# ─── Install project dependencies ───
echo "► Installing project dependencies..."
pip install -q \
	segmentation-models-pytorch \
	efficientnet_pytorch \
	albumentations \
	opencv-python \
	pandas \
	numpy \
	scipy \
	matplotlib \
	tensorboard \
	tqdm \
	pyyaml \
	click \
	scikit-learn \
	pillow \
	jupyter

pip install -q -e .
echo "  Done."

# ─── Verify installation ───
echo "► Verifying installation..."
python3 -c "
import torch
import segmentation_models_pytorch as smp
import albumentations
import cv2
import pandas
print('  ✓ PyTorch:', torch.__version__)
print('  ✓ CUDA:', torch.cuda.is_available())
print('  ✓ SMP:', smp.__version__)
print('  ✓ Albumentations:', albumentations.__version__)
print('  ✓ OpenCV:', cv2.__version__)
print('  ✓ Pandas:', pandas.__version__)
"

# ─── Setup data directory ───
echo "► Setting up data directory..."
mkdir -p "$DATA_DIR"
if [ -d "$DATA_DIR/train_images" ] && [ -f "$DATA_DIR/train.csv" ]; then
	echo "  Data already exists at $DATA_DIR"
else
	echo "  Data directory is empty."
	echo "  You need to upload or download the Severstal dataset."
	echo "  Expected structure:"
	echo "    $DATA_DIR/"
	echo "    ├── train_images/"
	echo "    ├── test_images/"
	echo "    └── train.csv"
fi

# ─── Create helper scripts ───
echo "► Creating helper scripts..."

cat >"$HOME/run_baseline.sh" <<'EOF'
#!/usr/bin/env bash
# Run baseline detector
set -e
source "$HOME/.venvs/severstal/bin/activate"
cd "$HOME/severstal-steel-defect-detection"
 sever train -c experiments/unet-b5-colab.yml
EOF
chmod +x "$HOME/run_baseline.sh"

cat >"$HOME/run_gan.sh" <<'EOF'
#!/usr/bin/env bash
# Train GAN
set -e
source "$HOME/.venvs/severstal/bin/activate"
cd "$HOME/severstal-steel-defect-detection"
python -m gan.train_wgan --config gan/config.yml
EOF
chmod +x "$HOME/run_gan.sh"

cat >"$HOME/generate_synthetic.sh" <<'EOF'
#!/usr/bin/env bash
# Generate synthetic data
set -e
source "$HOME/.venvs/severstal/bin/activate"
cd "$HOME/severstal-steel-defect-detection"
python -m gan.generate_synthetic \
    --generator saved/gan/G_final.pth \
    --critic saved/gan/D_final.pth \
    --output synthetic/ \
    --n_samples 2000
EOF
chmod +x "$HOME/generate_synthetic.sh"

cat >"$HOME/run_tensorboard.sh" <<'EOF'
#!/usr/bin/env bash
# Launch TensorBoard
source "$HOME/.venvs/severstal/bin/activate"
cd "$HOME/severstal-steel-defect-detection"
tensorboard --logdir saved/ --bind_all --port 6006
EOF
chmod +x "$HOME/run_tensorboard.sh"

echo "  Done."

# ─── Summary ───
echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  Setup Complete!                                             ║"
echo "╠══════════════════════════════════════════════════════════════╣"
echo "║  Project: $PROJECT_DIR"
echo "║  Data:    $DATA_DIR"
echo "╠══════════════════════════════════════════════════════════════╣"
echo "║  Next Steps:                                                 ║"
echo "║                                                              ║"
echo "║  1. Upload/download Severstal data to:                       ║"
echo "║     $DATA_DIR/                                              ║"
echo "║                                                              ║"
echo "║  2. Run baseline:                                            ║"
echo "║     ~/run_baseline.sh                                       ║"
echo "║                                                              ║"
echo "║  3. Train GAN:                                               ║"
echo "║     ~/run_gan.sh                                            ║"
echo "║                                                              ║"
echo "║  4. Generate synthetic data:                                 ║"
echo "║     ~/generate_synthetic.sh                                 ║"
echo "║                                                              ║"
echo "║  5. Launch TensorBoard:                                      ║"
echo "║     ~/run_tensorboard.sh                                    ║"
echo "║     Then open: http://<instance-ip>:6006                    ║"
echo "╚══════════════════════════════════════════════════════════════╝"

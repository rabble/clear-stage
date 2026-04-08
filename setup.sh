#!/bin/bash
# Setup script for RunPod A100 instance
#
# Model weights are stored on a RunPod Network Volume so they persist
# across sessions. First run downloads ~20GB; subsequent runs skip.
#
# Usage:
#   1. Create a RunPod Network Volume (30GB)
#   2. Launch a pod with the volume mounted at /workspace
#   3. git clone --recursive https://github.com/rabble/clear-stage.git /workspace/clear-stage
#   4. cd /workspace/clear-stage && bash setup.sh
#
# On subsequent runs, just: cd /workspace/clear-stage && bash setup.sh
# (skips downloads, only installs pip deps if needed)

set -e

echo "=== Clear Stage Setup ==="

# Where to store large model weights (persists on Network Volume)
WEIGHTS_DIR="${WEIGHTS_DIR:-/workspace/models}"
mkdir -p "$WEIGHTS_DIR"

# Install system deps
apt-get update && apt-get install -y ffmpeg git-lfs 2>/dev/null || true

# Clone with submodules if not already done
if [ ! -d "void-model/videox_fun" ]; then
    git submodule update --init --recursive
fi

# Create venv (on the persistent volume so pip installs survive too)
if [ ! -d ".venv" ]; then
    python3 -m venv .venv
fi
source .venv/bin/activate

# Install deps (skip if already done)
if ! python -c "import videox_fun" 2>/dev/null; then
    pip install -r void-model/requirements.txt
    pip install git+https://github.com/facebookresearch/segment-anything-2.git
    pip install groundingdino-py || pip install git+https://github.com/IDEA-Research/GroundingDINO.git
    pip install -r requirements.txt
fi

# Download model checkpoints to persistent volume
echo "=== Checking model checkpoints in $WEIGHTS_DIR ==="

# VOID Pass 1
if [ ! -f "$WEIGHTS_DIR/void_pass1.safetensors" ]; then
    echo "Downloading VOID Pass 1..."
    huggingface-cli download netflix/void-model void_pass1.safetensors --local-dir "$WEIGHTS_DIR"
fi

# Base CogVideoX model
if [ ! -d "$WEIGHTS_DIR/CogVideoX-Fun-V1.5-5b-InP" ]; then
    echo "Downloading CogVideoX base model..."
    huggingface-cli download alibaba-pai/CogVideoX-Fun-V1.5-5b-InP \
        --local-dir "$WEIGHTS_DIR/CogVideoX-Fun-V1.5-5b-InP"
fi

# SAM2 checkpoint
if [ ! -f "$WEIGHTS_DIR/sam2_hiera_large.pt" ]; then
    echo "Downloading SAM2 checkpoint..."
    wget -q -O "$WEIGHTS_DIR/sam2_hiera_large.pt" \
        https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_large.pt
fi

# GroundingDINO weights
if [ ! -f "$WEIGHTS_DIR/groundingdino_swint_ogc.pth" ]; then
    echo "Downloading GroundingDINO weights..."
    wget -q -O "$WEIGHTS_DIR/groundingdino_swint_ogc.pth" \
        https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth
fi

# Symlink weights into expected locations
ln -sf "$WEIGHTS_DIR/void_pass1.safetensors" void-model/void_pass1.safetensors
ln -sf "$WEIGHTS_DIR/sam2_hiera_large.pt" void-model/sam2_hiera_large.pt
ln -sf "$WEIGHTS_DIR/groundingdino_swint_ogc.pth" groundingdino_swint_ogc.pth
if [ -d "$WEIGHTS_DIR/CogVideoX-Fun-V1.5-5b-InP" ]; then
    ln -sfn "$WEIGHTS_DIR/CogVideoX-Fun-V1.5-5b-InP" void-model/CogVideoX-Fun-V1.5-5b-InP
fi

# Set up ffmpeg if needed
if ! command -v ffmpeg &> /dev/null; then
    ln -sf $(python -c "import imageio_ffmpeg; print(imageio_ffmpeg.get_ffmpeg_exe())") ~/.local/bin/ffmpeg
fi

echo ""
echo "=== Setup complete! ==="
echo "Weights stored in: $WEIGHTS_DIR (persistent across sessions)"
echo ""
echo "To run:"
echo "  source .venv/bin/activate"
echo "  export GEMINI_API_KEY=your_key_here"
echo "  python -m clear_stage.run_pipeline \\"
echo "    --video sample_videos/IMG_2745.mov \\"
echo "    --output output/result.mp4 \\"
echo "    --prompt 'A pole dance studio with mirrors and wooden floor'"

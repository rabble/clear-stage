#!/bin/bash
# Setup script for RunPod A100 instance
# Run this once after cloning the repo on a GPU machine
set -e

echo "=== Clear Stage Setup ==="

# Install system deps
apt-get update && apt-get install -y ffmpeg git-lfs 2>/dev/null || true

# Clone with submodules if not already done
if [ ! -d "void-model/videox_fun" ]; then
    git submodule update --init --recursive
fi

# Create venv
python3 -m venv .venv
source .venv/bin/activate

# Install VOID dependencies
pip install -r void-model/requirements.txt

# Install SAM2
pip install git+https://github.com/facebookresearch/segment-anything-2.git

# Install GroundingDINO
pip install groundingdino-py || pip install git+https://github.com/IDEA-Research/GroundingDINO.git

# Install our deps
pip install -r requirements.txt

# Download model checkpoints
echo "=== Downloading model checkpoints (~20GB) ==="

cd void-model

# VOID Pass 1 checkpoint
if [ ! -f "void_pass1.safetensors" ]; then
    echo "Downloading VOID Pass 1..."
    huggingface-cli download netflix/void-model void_pass1.safetensors --local-dir .
fi

# Base CogVideoX model
if [ ! -d "CogVideoX-Fun-V1.5-5b-InP" ]; then
    echo "Downloading CogVideoX base model..."
    huggingface-cli download alibaba-pai/CogVideoX-Fun-V1.5-5b-InP --local-dir ./CogVideoX-Fun-V1.5-5b-InP
fi

# SAM2 checkpoint
if [ ! -f "sam2_hiera_large.pt" ]; then
    echo "Downloading SAM2 checkpoint..."
    wget -q https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_large.pt
fi

cd ..

# GroundingDINO weights
if [ ! -f "groundingdino_swint_ogc.pth" ]; then
    echo "Downloading GroundingDINO weights..."
    wget -q https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth
fi

# Set up ffmpeg symlink if needed
if ! command -v ffmpeg &> /dev/null; then
    echo "Setting up ffmpeg from imageio..."
    ln -sf $(python -c "import imageio_ffmpeg; print(imageio_ffmpeg.get_ffmpeg_exe())") ~/.local/bin/ffmpeg
fi

echo ""
echo "=== Setup complete! ==="
echo ""
echo "To run:"
echo "  source .venv/bin/activate"
echo "  export GEMINI_API_KEY=your_key_here"
echo "  python -m clear_stage.run_pipeline \\"
echo "    --video sample_videos/IMG_2745.mov \\"
echo "    --output output/result.mp4 \\"
echo "    --prompt 'A pole dance studio with mirrors and wooden floor'"

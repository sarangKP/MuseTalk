MuseTalk 1.5 — Optimized for NVIDIA L4 (Lightning AI)

This repository contains a modified setup of MuseTalk 1.5, specifically tuned for NVIDIA L4 (24GB VRAM) and RTX 3080 architectures. It bypasses common "Dependency Hell" issues found in the original repo when running on modern Ubuntu/CUDA 12.x environments.
🚀 Key Optimizations

    Architecture Support: Native compatibility for Ada Lovelace (L4) and Ampere (3080).

    VRAM Efficiency: Configured for batch_size 8 to 12 on L4, cutting inference time by ~3x.

    Dependency Fixes: Resolved pkg_resources and chumpy build errors common in 2026 Python environments.

    Lightning AI Ready: Designed for system-level installation in Lightning Studios (No Conda required).

🛠️ Quick Start (Lightning AI Studio)
1. Environment Preparation

Ensure your Studio is set to Python 3.10.
Bash

# Fix system headers and audio codecs
sudo apt-get update && sudo apt-get install -y libglib2.0-0 libsm6 libxext6 libxrender1 ffmpeg build-essential python3-dev libsndfile1

# Fix legacy build dependencies
pip install "setuptools<80" wheel

2. Core Installation
Bash

# Install PyTorch (CUDA 11.8 build for MMCV stability)
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118

# Install MMLab Stack
pip install --no-cache-dir -U openmim
mim install mmengine==0.10.4
mim install "mmcv==2.0.1"
mim install "mmdet==3.1.0"
mim install "mmpose==1.1.0"

# Install MuseTalk Deps (including legacy chumpy fix)
pip install -r requirements.txt
pip install chumpy --no-build-isolation
pip install librosa==0.10.1 huggingface_hub==0.24.0

3. Model Weights

Run the patched download script:
Bash

bash download_weights.sh
# Note: Ensure models/musetalkV15/config.json is symlinked to musetalk.json

⚡ Inference (L4 Optimized)

To process at maximum speed on an L4 GPU:
Bash

python scripts/inference.py \
    --inference_config configs/inference/test.yaml \
    --batch_size 8 \
    --use_float16 \
    --version v15

Parameter	Value	Reason
batch_size	8	Optimized for 24GB VRAM
use_float16	True	Triggers L4 Tensor Core acceleration
bbox_shift	5	Default for natural mouth openness
🐛 Troubleshooting
Error	Fix
ModuleNotFoundError: No module named 'pkg_resources'	pip install "setuptools<80"
AttributeError: cached_download	pip install huggingface_hub==0.24.0
FileNotFoundError: config.json	ln -s musetalk.json config.json inside model folder
📜 Acknowledgements

Original MuseTalk by TMElyralab.

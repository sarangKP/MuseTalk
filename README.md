# MuseTalk 1.5 — Optimized for NVIDIA L4 (Lightning AI)

This repository contains an optimized deployment of MuseTalk 1.5, specifically tuned for **NVIDIA L4 (24GB VRAM)** and **RTX 3080/4080** architectures. It resolves the "Dependency Hell" common in 2026 Python environments and modern CUDA 12.x drivers.

---

## 🚀 Performance Optimizations

* **Architecture Support:** Native compatibility for Ada Lovelace (`sm_89`) and Ampere (`sm_86`).
* **High Throughput:** Configured for `batch_size 8` to `12` on L4, utilizing 24GB VRAM for 3x faster inference.
* **Precision:** Fully tested with `--use_float16` for Tensor Core acceleration.
* **Modern OS Support:** Patched for Ubuntu 22.04+ and Python 3.10+ (Lightning AI Studios).

---

## 🛠️ Installation (Lightning AI / Ubuntu 22.04)

### 1. System Dependencies
Run these once to handle audio codecs and C++ build headers.
```bash
sudo apt-get update && sudo apt-get install -y \
  libglib2.0-0 libsm6 libxext6 libxrender1 ffmpeg \
  build-essential python3-dev libsndfile1

```

### 2. The "Modern Python" Fixes

In 2026, `setuptools` has removed `pkg_resources`. You must pin an older version to build legacy dependencies like `chumpy` and `whisper`.

```bash
pip install "setuptools<80" wheel
pip install huggingface_hub==0.24.0  # Supports legacy 'cached_download'

```

### 3. PyTorch & MMLab Stack

We use CUDA 11.8 binaries for PyTorch to ensure 1:1 compatibility with pre-built `mmcv` wheels, even on CUDA 12.x drivers.

```bash
# PyTorch
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url [https://download.pytorch.org/whl/cu118](https://download.pytorch.org/whl/cu118)

# MMLab
pip install --no-cache-dir -U openmim
mim install mmengine==0.10.4
mim install "mmcv==2.0.1"
mim install "mmdet==3.1.0"
mim install "mmpose==1.1.0"

```

### 4. MuseTalk Core & Audio

```bash
pip install -r requirements.txt
pip install chumpy --no-build-isolation
pip install librosa==0.10.1 soundfile audiomentations

```

---

## 📦 Model Weights & Structure

1. **Run the patched download script:**

```bash
bash download_weights.sh

```

2. **Crucial Step:** Symlink the V1.5 config so the code detects it correctly.

```bash
ln -s ~/MuseTalk/models/musetalkV15/musetalk.json ~/MuseTalk/models/musetalkV15/config.json
ln -s ~/MuseTalk/models/sd-vae ~/MuseTalk/models/sd-vae-ft-mse

```

---

## ⚡ Inference Command (L4 Mode)

Process videos at maximum speed using these flags:

```bash
python scripts/inference.py \
    --inference_config configs/inference/test.yaml \
    --batch_size 8 \
    --use_float16 \
    --version v15

```

---

## 🛠️ Troubleshooting Matrix

|                 Error                |         Root Cause       |            Resolution                 |

| `ModuleNotFoundError: pkg_resources` | `setuptools` too new     | `pip install "setuptools<80"`         |
| `ImportError: cached_download`       | `huggingface_hub` > 0.25 | `pip install huggingface_hub==0.24.0` |
| `FileNotFoundError: config.json`     | Weight naming mismatch   | `ln -s musetalk.json config.json`     |
| `ModuleNotFoundError: librosa`       | Missing audio backend    | `pip install librosa==0.10.1`         |

---

## 📜 Credits

Original implementation by TMElyralab.
This fork maintained by sarangKP.

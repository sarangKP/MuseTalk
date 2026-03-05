import os
import torch
import mmcv
from mmcv.ops import get_compiling_cuda_version, get_compiler_version

def check_env():
    print("--- 🛠️ SYSTEM CHECK ---")
    print(f"L4 Detected: {torch.cuda.get_device_name(0)}")
    print(f"PyTorch Version: {torch.version.cuda} (Expected: 11.8)")
    print(f"MMCV Version: {mmcv.__version__}")
    print(f"MMCV CUDA Compiled: {get_compiling_cuda_version()}")
    
    # Weights Verification
    paths = [
        "models/musetalkV15/unet.pth",
        "models/musetalkV15/musetalk.json",
        "models/dwpose/dw-ll_ucoco_384.pth",
        "models/whisper/pytorch_model.bin",
        "models/sd-vae/diffusion_pytorch_model.bin"
    ]
    
    print("\n--- 📦 WEIGHTS CHECK ---")
    for p in paths:
        status = "✅ FOUND" if os.path.exists(p) else "❌ MISSING"
        print(f"{status}: {p}")

check_env()
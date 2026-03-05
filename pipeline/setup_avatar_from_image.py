#!/usr/bin/env python3
"""
setup_avatar_from_image.py
===========================
Converts a single face image → looping avatar video → prepared avatar data.
Run this once, then start the live server.

Usage (run from your MuseTalk root):
    cd /teamspace/studios/this_studio/MuseTalk
    python pipeline/setup_avatar_from_image.py \
        --image face_1.png \
        --avatar_id face_1

What it does:
    1. Validates the image and detects a face
    2. Creates a 5-second looping "idle" video from the image (25fps = 125 frames)
    3. Runs the full avatar preparation pipeline:
       - Extracts frames
       - Detects landmarks & bboxes
       - Pre-encodes VAE latents
       - Pre-computes blending masks
    4. Prints the exact server command to run next
"""

import argparse
import os
import sys
import time
import shutil

import cv2
import numpy as np


# ── Step 1: Validate image + detect face ────────────────────────────────────

def check_image(image_path: str) -> np.ndarray:
    img = cv2.imread(image_path)
    if img is None:
        print(f"❌ Could not read image: {image_path}")
        sys.exit(1)
    h, w = img.shape[:2]
    print(f"✓ Image loaded: {w}x{h} px")

    # Quick face check via cascade (no heavy deps)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    face_cascade  = cv2.CascadeClassifier(cascade_path)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)
    if len(faces) == 0:
        print("⚠️  No face detected with quick check — continuing anyway (MuseTalk will detect properly)")
    else:
        x, y, fw, fh = faces[0]
        print(f"✓ Face detected at [{x},{y},{x+fw},{y+fh}]")
    return img


# ── Step 2: Build looping idle video ─────────────────────────────────────────

def make_idle_video(img: np.ndarray, output_path: str, fps: int = 25, duration_sec: int = 5):
    """
    Write a static looping video from a single frame.
    25fps × 5s = 125 identical frames — enough for the avatar cycle.
    MuseTalk cycles these during inference so any length works.
    """
    n_frames = fps * duration_sec
    h, w     = img.shape[:2]
    fourcc   = cv2.VideoWriter_fourcc(*"mp4v")
    writer   = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
    for _ in range(n_frames):
        writer.write(img)
    writer.release()
    print(f"✓ Idle video created: {output_path} ({n_frames} frames @ {fps}fps)")


# ── Step 3: Create realtime.yaml config ──────────────────────────────────────

YAML_TEMPLATE = """\
{avatar_id}:
  preparation: true
  video_path: {video_path}
  bbox_shift: 0
  audio_clips:
    0: ./data/audio/silence.wav
"""

def make_realtime_yaml(avatar_id: str, video_path: str, yaml_out: str):
    content = YAML_TEMPLATE.format(avatar_id=avatar_id, video_path=video_path)
    with open(yaml_out, "w") as f:
        f.write(content)
    print(f"✓ Config written: {yaml_out}")


def make_silence_wav(path: str, duration_sec: float = 1.0, sr: int = 16000):
    """Create a short silence WAV for the preparation dummy run."""
    import wave, struct
    os.makedirs(os.path.dirname(path), exist_ok=True)
    n = int(sr * duration_sec)
    with wave.open(path, "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(struct.pack(f"<{n}h", *([0] * n)))
    print(f"✓ Silence WAV created: {path}")


# ── Step 4: Run preparation via realtime_inference.py ────────────────────────

def run_preparation(avatar_id: str, yaml_path: str, version: str = "v15"):
    """
    Calls scripts/realtime_inference.py in preparation-only mode.
    The preparation flag in the YAML triggers face detection + latent encoding.
    """
    unet_config = f"./models/musetalkV15/musetalk.json" if version == "v15" else "./models/musetalk/musetalk.json"
    unet_model  = f"./models/musetalkV15/unet.pth"      if version == "v15" else "./models/musetalk/pytorch_model.bin"

    cmd = (
        f"python scripts/realtime_inference.py "
        f"--version {version} "
        f"--unet_config {unet_config} "
        f"--unet_model_path {unet_model} "
        f"--inference_config {yaml_path} "
        f"--batch_size 4 "
        f"--skip_save_images "
    )
    print(f"\n🔧 Running preparation:\n   {cmd}\n")
    ret = os.system(cmd)
    if ret != 0:
        print("❌ Preparation failed. Check output above.")
        sys.exit(1)
    print("✓ Preparation complete!")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Set up live avatar from a single image")
    parser.add_argument("--image",       required=True,  help="Path to face image (PNG/JPG)")
    parser.add_argument("--avatar_id",   default="face_1", help="Avatar ID (used for directory naming)")
    parser.add_argument("--version",     default="v15",  choices=["v1", "v15"])
    parser.add_argument("--fps",         type=int, default=25)
    parser.add_argument("--duration",    type=int, default=5,  help="Idle video duration in seconds")
    parser.add_argument("--skip_prep",   action="store_true",  help="Skip preparation (just create video/yaml)")
    a = parser.parse_args()

    print("\n" + "="*55)
    print("  Live Avatar Setup")
    print("="*55)

    # Paths
    musetalk_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(musetalk_root)
    print(f"Working dir: {musetalk_root}\n")

    assets_dir  = "./data/video"
    audio_dir   = "./data/audio"
    config_dir  = "./configs/inference"
    os.makedirs(assets_dir,  exist_ok=True)
    os.makedirs(audio_dir,   exist_ok=True)
    os.makedirs(config_dir,  exist_ok=True)

    video_path  = f"{assets_dir}/{a.avatar_id}_idle.mp4"
    yaml_path   = f"{config_dir}/{a.avatar_id}_live.yaml"
    silence_path = f"{audio_dir}/silence.wav"

    # 1. Check image
    print("Step 1/4  Validating image...")
    img = check_image(a.image)

    # 2. Create idle video
    print("\nStep 2/4  Creating idle video...")
    make_idle_video(img, video_path, fps=a.fps, duration_sec=a.duration)

    # 3. Create config + silence
    print("\nStep 3/4  Creating config files...")
    make_realtime_yaml(a.avatar_id, video_path, yaml_path)
    make_silence_wav(silence_path)

    # 4. Run preparation
    if not a.skip_prep:
        print("\nStep 4/4  Running MuseTalk avatar preparation...")
        run_preparation(a.avatar_id, yaml_path, version=a.version)
    else:
        print("\nStep 4/4  [Skipped — run preparation manually]")

    # ── Determine output path ─────────────────────────────────────────────
    if a.version == "v15":
        avatar_out = f"./results/v15/avatars/{a.avatar_id}"
    else:
        avatar_out = f"./results/avatars/{a.avatar_id}"

    # ── Final instructions ────────────────────────────────────────────────
    print("\n" + "="*55)
    print("  ✅ Setup Complete!")
    print("="*55)
    print(f"""
Avatar prepared at:
  {avatar_out}

─── Next Steps ───────────────────────────────────

1. Start the live server:

   python pipeline/live_avatar_server.py \\
       --port 8765 \\
       --batch_size 2 \\
       --unet_model_path ./models/musetalkV15/unet.pth \\
       --unet_config ./models/musetalkV15/musetalk.json \\
       --whisper_dir ./models/whisper

2. Load the avatar (in a new terminal or HTTP client):

   curl -X POST http://localhost:8765/avatar/load \\
     -H "Content-Type: application/json" \\
     -d '{{"avatar_id": "{a.avatar_id}", "avatar_path": "{avatar_out}"}}'

3. Stream audio to it:

   python pipeline/tts_chunker.py \\
       --avatar_id {a.avatar_id} \\
       --text "Hello! I am your live avatar." \\
       --tts edge_tts

4. Open pipeline/viewer.html in your browser,
   connect to ws://localhost:8765, avatar ID: {a.avatar_id}

─────────────────────────────────────────────────
""")


if __name__ == "__main__":
    main()

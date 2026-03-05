"""
Avatar Preparation Script
==========================
Pre-processes an avatar video ONCE so that:
  - Full frames are extracted and stored as PNG
  - Face bounding boxes + landmarks are detected and pickled
  - VAE latents for all frames are pre-encoded and saved to disk
  - Blending masks are pre-computed

After running this, the live server loads these artefacts into RAM
and can begin streaming within seconds.

Usage:
    python prepare_avatar.py \
        --avatar_id my_avatar \
        --video_path ./assets/my_face.mp4 \
        --version v15
"""

import argparse
import os
import pickle
import sys
import time
import json

import cv2
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from musetalk.utils.utils import load_all_model
from musetalk.utils.preprocessing import get_landmark_and_bbox, read_imgs
from musetalk.utils.blending import get_image_prepare_material
from musetalk.utils.face_parsing import FaceParsing


def video_to_frames(video_path: str, save_path: str, max_frames: int = 10000):
    os.makedirs(save_path, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    count = 0
    while count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imwrite(f"{save_path}/{count:08d}.png", frame)
        count += 1
    cap.release()
    print(f"Extracted {count} frames from {video_path}")
    return count


@torch.no_grad()
def prepare_avatar(
    avatar_id:   str,
    video_path:  str,
    version:     str = "v15",
    bbox_shift:  int = 0,
    extra_margin: int = 10,
    device:      str = "cuda",
    use_float16: bool = True,
    unet_config:      str = "./models/musetalkV15/musetalk.json",
    unet_model_path:  str = "./models/musetalkV15/unet.pth",
    left_cheek_width:  int = 0,
    right_cheek_width: int = 0,
):
    # ── Paths ──────────────────────────────────────────────────────────────
    if version == "v15":
        base = f"./results/v15/avatars/{avatar_id}"
    else:
        base = f"./results/avatars/{avatar_id}"

    full_imgs_path   = f"{base}/full_imgs"
    coords_path      = f"{base}/coords.pkl"
    latents_path     = f"{base}/latents.pt"
    mask_path        = f"{base}/mask"
    mask_coords_path = f"{base}/mask_coords.pkl"
    info_path        = f"{base}/avator_info.json"

    os.makedirs(base, exist_ok=True)
    os.makedirs(full_imgs_path, exist_ok=True)
    os.makedirs(mask_path, exist_ok=True)

    # ── Step 1: Extract frames ─────────────────────────────────────────────
    print("Step 1/4: Extracting frames...")
    n_frames = video_to_frames(video_path, full_imgs_path)

    # ── Step 2: Detect landmarks & compute bboxes ──────────────────────────
    print("Step 2/4: Detecting face landmarks and bounding boxes...")
    t0 = time.time()
    input_img_list = sorted([
        os.path.join(full_imgs_path, f)
        for f in os.listdir(full_imgs_path)
        if f.endswith(".png")
    ])

    coord_list, frame_list = get_landmark_and_bbox(
        input_img_list,
        bbox_shift=bbox_shift if version == "v1" else 0,
        extra_margin=extra_margin,
    )
    print(f"  Landmark detection: {time.time()-t0:.1f}s for {len(coord_list)} frames")

    with open(coords_path, "wb") as f:
        pickle.dump(coord_list, f)

    # ── Step 3: Pre-encode VAE latents ─────────────────────────────────────
    print("Step 3/4: Pre-encoding VAE latents...")
    t0 = time.time()

    device_obj = torch.device(device if torch.cuda.is_available() else "cpu")
    vae, unet, pe = load_all_model(
        unet_model_path=unet_model_path,
        vae_type="sd-vae",
        unet_config=unet_config,
        device=device_obj,
    )

    if use_float16:
        vae.vae = vae.vae.half().to(device_obj)
    else:
        vae.vae = vae.vae.to(device_obj)

    input_latent_list = []
    for i, (bbox, frame) in enumerate(zip(coord_list, frame_list)):
        if bbox == (0, 0, 0, 0):
            # No face detected — use previous latent or zeros
            if input_latent_list:
                input_latent_list.append(input_latent_list[-1])
            else:
                input_latent_list.append(torch.zeros(1, 4, 8, 8, dtype=vae.vae.dtype))
            continue

        x1, y1, x2, y2 = bbox
        crop = frame[y1:y2, x1:x2]
        crop_resized = cv2.resize(crop, (256, 256))

        # Normalize to [-1, 1]
        crop_tensor = torch.from_numpy(crop_resized).float() / 127.5 - 1.0
        crop_tensor = crop_tensor.permute(2, 0, 1).unsqueeze(0)
        if use_float16:
            crop_tensor = crop_tensor.half()
        crop_tensor = crop_tensor.to(device_obj)

        latent = vae.get_latents_for_unet(crop_tensor)
        input_latent_list.append(latent.cpu())

        if i % 50 == 0:
            print(f"  Encoded {i}/{len(frame_list)} frames...")

    torch.save(input_latent_list, latents_path)
    print(f"  VAE encoding: {time.time()-t0:.1f}s for {len(input_latent_list)} frames")

    # ── Step 4: Pre-compute blending masks ────────────────────────────────
    print("Step 4/4: Pre-computing blending masks...")
    t0 = time.time()

    if version == "v15":
        fp = FaceParsing(
            left_cheek_width=left_cheek_width,
            right_cheek_width=right_cheek_width,
        )
    else:
        fp = FaceParsing()

    mask_list     = []
    mask_coord_list = []
    for i, (bbox, frame) in enumerate(zip(coord_list, frame_list)):
        if bbox == (0, 0, 0, 0):
            mask_list.append(None)
            mask_coord_list.append(None)
            continue

        mask, mask_coord = get_image_prepare_material(frame, bbox, fp)
        mask_list.append(mask)
        mask_coord_list.append(mask_coord)

        # Save mask image
        if mask is not None:
            cv2.imwrite(f"{mask_path}/{i:08d}.png", mask)

        if i % 50 == 0:
            print(f"  Mask {i}/{len(frame_list)}...")

    with open(mask_coords_path, "wb") as f:
        pickle.dump(mask_coord_list, f)

    print(f"  Mask computation: {time.time()-t0:.1f}s")

    # ── Save avatar metadata ───────────────────────────────────────────────
    info = {
        "avatar_id":    avatar_id,
        "video_path":   video_path,
        "version":      version,
        "bbox_shift":   bbox_shift,
        "n_frames":     n_frames,
        "prepared_at":  time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(info_path, "w") as f:
        json.dump(info, f, indent=2)

    print(f"\n✅ Avatar '{avatar_id}' prepared at: {base}")
    print(f"   Frames: {n_frames} | Latents: {len(input_latent_list)}")
    print(f"\nNow load it in the server:\n"
          f"  POST /avatar/load  {{\"avatar_id\": \"{avatar_id}\", \"avatar_path\": \"{base}\"}}")

    return base


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--avatar_id",         type=str,  required=True)
    parser.add_argument("--video_path",         type=str,  required=True)
    parser.add_argument("--version",            type=str,  default="v15", choices=["v1", "v15"])
    parser.add_argument("--bbox_shift",         type=int,  default=0)
    parser.add_argument("--extra_margin",       type=int,  default=10)
    parser.add_argument("--device",             type=str,  default="cuda")
    parser.add_argument("--no_float16",         action="store_true")
    parser.add_argument("--unet_config",        type=str,  default="./models/musetalkV15/musetalk.json")
    parser.add_argument("--unet_model_path",    type=str,  default="./models/musetalkV15/unet.pth")
    parser.add_argument("--left_cheek_width",   type=int,  default=0)
    parser.add_argument("--right_cheek_width",  type=int,  default=0)
    a = parser.parse_args()

    prepare_avatar(
        avatar_id=a.avatar_id,
        video_path=a.video_path,
        version=a.version,
        bbox_shift=a.bbox_shift,
        extra_margin=a.extra_margin,
        device=a.device,
        use_float16=not a.no_float16,
        unet_config=a.unet_config,
        unet_model_path=a.unet_model_path,
        left_cheek_width=a.left_cheek_width,
        right_cheek_width=a.right_cheek_width,
    )

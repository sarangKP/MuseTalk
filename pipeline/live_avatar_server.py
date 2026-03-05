"""
Live Avatar Pipeline - Ultra Low-Latency MuseTalk Server
=========================================================
Architecture:
  TTS Audio → Whisper Chunks → UNet Inference → VAE Decode → WebSocket Frame Push

Latency budget targets:
  - Audio chunk size     : 80ms  (2 frames @ 25fps)
  - Whisper encoding     : ~15ms (tiny model, GPU)
  - UNet forward pass    : ~20ms (half precision, batch=2)
  - VAE decode           : ~10ms
  - Blend + encode JPEG  : ~5ms
  - WebSocket push       : ~2ms
  ─────────────────────────────
  Target end-to-end      : <150ms per chunk
"""

import asyncio
import json
import time
import queue
import threading
import logging
import base64
import io
import os
import sys
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# MuseTalk imports (adjust paths to your repo layout)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from musetalk.utils.utils import load_all_model, datagen
from musetalk.utils.blending import get_image_blending
from musetalk.utils.audio_processor import AudioProcessor
from transformers import WhisperModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
log = logging.getLogger("live_avatar")

# ──────────────────────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class PipelineConfig:
    # Model paths
    unet_model_path: str = "./models/musetalkV15/unet.pth"
    unet_config:     str = "./models/musetalkV15/musetalk.json"
    vae_type:        str = "sd-vae"
    whisper_dir:     str = "./models/whisper"

    # Inference
    device:          str = "cuda"
    use_float16:     bool = True
    batch_size:      int = 2          # Keep tiny for low latency (2 frames = 80ms)
    fps:             int = 25

    # Audio windowing
    audio_pad_left:  int = 2
    audio_pad_right: int = 2

    # Streaming
    jpeg_quality:    int = 85         # Lower = faster encode, higher latency reduction
    max_frame_queue: int = 8          # Frames buffered ahead in output queue
    chunk_audio_ms:  int = 80         # Audio chunk size fed to Whisper


CFG = PipelineConfig()

# ──────────────────────────────────────────────────────────────────────────────
# Global model state (loaded once at startup)
# ──────────────────────────────────────────────────────────────────────────────

class ModelState:
    vae = None
    unet = None
    pe = None
    whisper = None
    audio_processor = None
    device = None
    weight_dtype = None
    timesteps = None

    @classmethod
    def load(cls, cfg: PipelineConfig):
        log.info("Loading MuseTalk models...")
        t0 = time.time()

        cls.device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
        cls.vae, cls.unet, cls.pe = load_all_model(
            unet_model_path=cfg.unet_model_path,
            vae_type=cfg.vae_type,
            unet_config=cfg.unet_config,
            device=cls.device,
        )

        if cfg.use_float16:
            cls.weight_dtype = torch.float16
            cls.pe      = cls.pe.half().to(cls.device)
            cls.vae.vae = cls.vae.vae.half().to(cls.device)
            cls.unet.model = cls.unet.model.half().to(cls.device)
        else:
            cls.weight_dtype = torch.float32
            cls.pe      = cls.pe.to(cls.device)
            cls.vae.vae = cls.vae.vae.to(cls.device)
            cls.unet.model = cls.unet.model.to(cls.device)

        cls.audio_processor = AudioProcessor(feature_extractor_path=cfg.whisper_dir)
        cls.whisper = WhisperModel.from_pretrained(cfg.whisper_dir)
        cls.whisper = cls.whisper.to(device=cls.device, dtype=cls.weight_dtype).eval()
        cls.whisper.requires_grad_(False)

        cls.timesteps = torch.tensor([0], device=cls.device)

        # Warmup pass to JIT-compile kernels
        log.info("Warming up GPU kernels...")
        _warmup(cfg)

        log.info(f"Models loaded in {time.time()-t0:.1f}s on {cls.device}")

    @classmethod
    def ready(cls) -> bool:
        return cls.unet is not None


def _warmup(cfg: PipelineConfig):
    """Run a dummy forward pass so CUDA kernels are compiled before first request."""
    try:
        dummy_latent = torch.zeros(
            cfg.batch_size, 8, 8, 8,
            dtype=ModelState.weight_dtype, device=ModelState.device
        )
        dummy_audio = torch.zeros(
            cfg.batch_size, 10, 384,
            dtype=ModelState.weight_dtype, device=ModelState.device
        )
        with torch.no_grad():
            audio_feat = ModelState.pe(dummy_audio)
            ModelState.unet.model(dummy_latent, ModelState.timesteps, encoder_hidden_states=audio_feat)
        log.info("Warmup complete.")
    except Exception as e:
        log.warning(f"Warmup skipped: {e}")


# ──────────────────────────────────────────────────────────────────────────────
# Avatar Session  (pre-processed face frames stored in memory)
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class AvatarSession:
    avatar_id: str
    full_frames: list        # BGR frames (full resolution)
    input_latents: list      # Pre-encoded VAE latents
    coords: list             # Face bbox coords per frame
    mask_list: list          # Blending masks
    mask_coords: list        # Mask coords
    frame_count: int = 0     # Cycling index


# In-memory registry of loaded avatars
_avatars: dict[str, AvatarSession] = {}
_avatar_lock = threading.Lock()


def load_avatar(avatar_id: str, avatar_path: str) -> AvatarSession:
    """Load pre-processed avatar data from disk into memory."""
    import pickle
    coords_path      = f"{avatar_path}/coords.pkl"
    latents_path     = f"{avatar_path}/latents.pt"
    mask_coords_path = f"{avatar_path}/mask_coords.pkl"
    full_imgs_path   = f"{avatar_path}/full_imgs"
    mask_path        = f"{avatar_path}/mask"

    if not os.path.exists(coords_path):
        raise FileNotFoundError(f"Avatar not prepared at: {avatar_path}. Run preparation first.")

    with open(coords_path, "rb") as f:
        coords = pickle.load(f)
    with open(mask_coords_path, "rb") as f:
        mask_coords = pickle.load(f)

    input_latent_list = torch.load(latents_path)

    # Load frames into RAM for fast access
    img_files = sorted([
        f for f in os.listdir(full_imgs_path)
        if f.endswith((".png", ".jpg"))
    ])
    full_frames = [cv2.imread(os.path.join(full_imgs_path, f)) for f in img_files]

    mask_files = sorted([
        f for f in os.listdir(mask_path)
        if f.endswith((".png", ".jpg"))
    ])
    mask_list = [cv2.imread(os.path.join(mask_path, f)) for f in mask_files]

    session = AvatarSession(
        avatar_id=avatar_id,
        full_frames=full_frames,
        input_latents=input_latent_list,
        coords=coords,
        mask_list=mask_list,
        mask_coords=mask_coords,
    )
    log.info(f"Avatar '{avatar_id}' loaded: {len(full_frames)} frames, {len(input_latent_list)} latents")
    return session


# ──────────────────────────────────────────────────────────────────────────────
# Core inference: audio bytes → frame generator
# ──────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def infer_frames_from_audio(
    audio_path: str,
    session: AvatarSession,
    cfg: PipelineConfig,
) -> list[np.ndarray]:
    """
    Full inference pass: audio file → list of blended BGR frames.
    Optimised for minimum latency with batch_size=2.
    """
    ms = ModelState

    # 1. Extract Whisper features
    t_audio = time.perf_counter()
    whisper_features, librosa_length = ms.audio_processor.get_audio_feature(
        audio_path, weight_dtype=ms.weight_dtype
    )
    whisper_chunks = ms.audio_processor.get_whisper_chunk(
        whisper_features, ms.device, ms.weight_dtype, ms.whisper,
        librosa_length, fps=cfg.fps,
        audio_padding_length_left=cfg.audio_pad_left,
        audio_padding_length_right=cfg.audio_pad_right,
    )
    log.debug(f"Audio encoding: {(time.perf_counter()-t_audio)*1000:.1f}ms, {len(whisper_chunks)} chunks")

    # 2. Cycle latents to match whisper chunks
    n = len(whisper_chunks)
    latent_cycle = [
        session.input_latents[i % len(session.input_latents)]
        for i in range(n)
    ]

    # 3. Batch inference
    output_frames: list[np.ndarray] = []
    gen = datagen(whisper_chunks, latent_cycle, cfg.batch_size)

    t_infer = time.perf_counter()
    for whisper_batch, latent_batch in gen:
        audio_feat = ms.pe(whisper_batch.to(ms.device))
        latent_batch = latent_batch.to(device=ms.device, dtype=ms.unet.model.dtype)

        pred_latents = ms.unet.model(
            latent_batch, ms.timesteps,
            encoder_hidden_states=audio_feat
        ).sample
        pred_latents = pred_latents.to(device=ms.device, dtype=ms.vae.vae.dtype)
        recon = ms.vae.decode_latents(pred_latents)

        for res_frame in recon:
            output_frames.append(res_frame)

    log.debug(f"UNet+VAE inference: {(time.perf_counter()-t_infer)*1000:.1f}ms for {len(output_frames)} frames")

    # 4. Blend onto full frames
    blended: list[np.ndarray] = []
    for i, res_frame in enumerate(output_frames):
        frame_idx = (session.frame_count + i) % len(session.full_frames)
        full_frame = session.full_frames[frame_idx].copy()
        coords_i   = session.coords[frame_idx % len(session.coords)]
        mask_i     = session.mask_list[frame_idx % len(session.mask_list)]
        mask_coord = session.mask_coords[frame_idx % len(session.mask_coords)]

        x1, y1, x2, y2 = coords_i
        res_resized = cv2.resize(res_frame.astype(np.uint8), (x2 - x1, y2 - y1))
        full_frame[y1:y2, x1:x2] = res_resized

        # Soft blending with pre-computed mask
        blended_frame = get_image_blending(full_frame, res_resized, mask_i, mask_coord)
        blended.append(blended_frame)

    session.frame_count += len(output_frames)
    return blended


def frames_to_jpeg_b64(frames: list[np.ndarray], quality: int = 85) -> list[str]:
    """Encode BGR frames to base64 JPEG strings for WebSocket transport."""
    result = []
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    for frame in frames:
        _, buf = cv2.imencode(".jpg", frame, encode_param)
        result.append(base64.b64encode(buf.tobytes()).decode())
    return result


# ──────────────────────────────────────────────────────────────────────────────
# Streaming session: queued audio → queued frames (producer/consumer)
# ──────────────────────────────────────────────────────────────────────────────

class StreamSession:
    """
    Manages a bidirectional streaming session for one WebSocket client.
    Audio chunks arrive via push_audio(); blended frames are available via get_frames().
    """

    def __init__(self, session_id: str, avatar: AvatarSession, cfg: PipelineConfig):
        self.session_id  = session_id
        self.avatar      = avatar
        self.cfg         = cfg
        self._audio_q    = queue.Queue(maxsize=32)   # incoming raw PCM paths
        self._frame_q    = queue.Queue(maxsize=cfg.max_frame_queue * 4)
        self._stop       = threading.Event()
        self._worker     = threading.Thread(target=self._inference_loop, daemon=True)
        self._worker.start()
        self.stats = {"frames_sent": 0, "chunks_processed": 0, "avg_latency_ms": 0.0}

    def push_audio(self, audio_path: str):
        """Push a prepared audio file path for processing."""
        self._audio_q.put_nowait(audio_path)

    def get_frames_nowait(self) -> list[str]:
        """Non-blocking drain of ready JPEG-b64 frames."""
        frames = []
        while not self._frame_q.empty():
            try:
                frames.append(self._frame_q.get_nowait())
            except queue.Empty:
                break
        return frames

    def stop(self):
        self._stop.set()
        self._worker.join(timeout=2.0)

    def _inference_loop(self):
        while not self._stop.is_set():
            try:
                audio_path = self._audio_q.get(timeout=0.05)
            except queue.Empty:
                continue

            t0 = time.perf_counter()
            try:
                frames = infer_frames_from_audio(audio_path, self.avatar, self.cfg)
                b64_frames = frames_to_jpeg_b64(frames, self.cfg.jpeg_quality)
                for f in b64_frames:
                    self._frame_q.put(f, timeout=1.0)
                latency = (time.perf_counter() - t0) * 1000
                self.stats["chunks_processed"] += 1
                self.stats["frames_sent"] += len(b64_frames)
                # Running average
                n = self.stats["chunks_processed"]
                self.stats["avg_latency_ms"] = (
                    self.stats["avg_latency_ms"] * (n - 1) + latency
                ) / n
                log.debug(f"[{self.session_id}] chunk done in {latency:.0f}ms, {len(b64_frames)} frames queued")
            except Exception as e:
                log.error(f"[{self.session_id}] Inference error: {e}", exc_info=True)

# In-memory active stream sessions
_stream_sessions: dict[str, StreamSession] = {}

# ──────────────────────────────────────────────────────────────────────────────
# FastAPI app
# ──────────────────────────────────────────────────────────────────────────────

app = FastAPI(title="Live Avatar API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)


@app.on_event("startup")
async def startup_event():
    ModelState.load(CFG)


# ── REST endpoints ──────────────────────────────────────────────────────────

class LoadAvatarRequest(BaseModel):
    avatar_id: str
    avatar_path: str   # e.g. "./results/v15/avatars/my_avatar"


@app.post("/avatar/load")
def api_load_avatar(req: LoadAvatarRequest):
    """Pre-load an avatar into RAM (run once before streaming)."""
    with _avatar_lock:
        session = load_avatar(req.avatar_id, req.avatar_path)
        _avatars[req.avatar_id] = session
    return {"status": "loaded", "avatar_id": req.avatar_id,
            "frames": len(session.full_frames)}


@app.get("/avatar/list")
def api_list_avatars():
    return {"avatars": list(_avatars.keys())}


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_ready": ModelState.ready(),
        "device": str(ModelState.device),
        "dtype": str(ModelState.weight_dtype),
        "avatars_loaded": len(_avatars),
    }


# ── WebSocket streaming endpoint ────────────────────────────────────────────

@app.websocket("/stream/{avatar_id}")
async def ws_stream(ws: WebSocket, avatar_id: str):
    """
    WebSocket protocol:
      Client → Server:  JSON  { "type": "audio_chunk", "path": "/tmp/chunk_001.wav" }
                        JSON  { "type": "stop" }
      Server → Client:  JSON  { "type": "frame", "data": "<base64-jpeg>", "ts": 1234.5 }
                        JSON  { "type": "stats", ... }
                        JSON  { "type": "error", "message": "..." }
    """
    await ws.accept()

    if avatar_id not in _avatars:
        await ws.send_json({"type": "error", "message": f"Avatar '{avatar_id}' not loaded."})
        await ws.close()
        return

    session_id = f"{avatar_id}_{int(time.time()*1000)}"
    stream = StreamSession(session_id, _avatars[avatar_id], CFG)
    _stream_sessions[session_id] = stream
    log.info(f"Stream session started: {session_id}")

    # Background task: push ready frames to client
    async def frame_pusher():
        while True:
            frames = stream.get_frames_nowait()
            for f in frames:
                await ws.send_json({
                    "type": "frame",
                    "data": f,
                    "ts":   time.time(),
                    "session_id": session_id,
                })
            await asyncio.sleep(0.01)  # 10ms poll interval

    pusher_task = asyncio.create_task(frame_pusher())

    try:
        while True:
            raw = await ws.receive_text()
            msg = json.loads(raw)

            if msg.get("type") == "audio_chunk":
                audio_path = msg.get("path")
                if not audio_path or not os.path.exists(audio_path):
                    await ws.send_json({"type": "error", "message": f"Audio not found: {audio_path}"})
                    continue
                stream.push_audio(audio_path)

            elif msg.get("type") == "stop":
                break

            elif msg.get("type") == "stats":
                await ws.send_json({"type": "stats", **stream.stats})

    except WebSocketDisconnect:
        log.info(f"Client disconnected: {session_id}")
    finally:
        pusher_task.cancel()
        stream.stop()
        del _stream_sessions[session_id]
        log.info(f"Stream session closed: {session_id}")


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Live Avatar Server")
    parser.add_argument("--host",          type=str,  default="0.0.0.0")
    parser.add_argument("--port",          type=int,  default=8765)
    parser.add_argument("--batch_size",    type=int,  default=2)
    parser.add_argument("--jpeg_quality",  type=int,  default=85)
    parser.add_argument("--no_float16",    action="store_true")
    parser.add_argument("--unet_model_path", type=str, default="./models/musetalkV15/unet.pth")
    parser.add_argument("--unet_config",   type=str,  default="./models/musetalkV15/musetalk.json")
    parser.add_argument("--whisper_dir",   type=str,  default="./models/whisper")
    a = parser.parse_args()

    CFG.batch_size     = a.batch_size
    CFG.jpeg_quality   = a.jpeg_quality
    CFG.use_float16    = not a.no_float16
    CFG.unet_model_path = a.unet_model_path
    CFG.unet_config    = a.unet_config
    CFG.whisper_dir    = a.whisper_dir

    uvicorn.run(app, host=a.host, port=a.port, log_level="info")

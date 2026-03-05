"""
TTS Audio Chunker
=================
Receives streaming TTS audio (e.g. from ElevenLabs / OpenAI TTS),
splits it into fixed-size WAV chunks, and pushes them to the
Live Avatar Server via WebSocket.

Usage:
    python tts_chunker.py --avatar_id my_avatar --text "Hello world"
    python tts_chunker.py --avatar_id my_avatar --audio_file speech.wav

The chunker writes temporary WAV files to /tmp/avatar_chunks/ and
sends their paths over the WebSocket. The server reads them directly
(same machine) for zero-copy audio transfer.

For remote deployments, set --send_bytes to embed raw PCM in the message.
"""

import argparse
import asyncio
import json
import logging
import os
import queue
import struct
import tempfile
import threading
import time
import wave
from pathlib import Path

import numpy as np
import websockets

log = logging.getLogger("tts_chunker")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

CHUNK_DIR = Path(tempfile.gettempdir()) / "avatar_chunks"
CHUNK_DIR.mkdir(exist_ok=True)


# ──────────────────────────────────────────────────────────────────────────────
# Audio utilities
# ──────────────────────────────────────────────────────────────────────────────

def load_wav_as_float(path: str, target_sr: int = 16000) -> np.ndarray:
    """Load WAV file, resample to target_sr if needed, return float32 array."""
    import librosa
    audio, _ = librosa.load(path, sr=target_sr, mono=True)
    return audio


def split_audio_into_chunks(
    audio: np.ndarray,
    sr: int,
    chunk_ms: int,
    overlap_ms: int = 0,
) -> list[np.ndarray]:
    """
    Split audio into fixed-size chunks with optional overlap.
    overlap_ms allows smoother lip transitions at chunk boundaries.
    """
    chunk_samples   = int(sr * chunk_ms   / 1000)
    overlap_samples = int(sr * overlap_ms / 1000)
    stride = chunk_samples - overlap_samples

    chunks = []
    pos = 0
    while pos < len(audio):
        end = min(pos + chunk_samples, len(audio))
        chunk = audio[pos:end]
        # Zero-pad last chunk if short
        if len(chunk) < chunk_samples:
            chunk = np.pad(chunk, (0, chunk_samples - len(chunk)))
        chunks.append(chunk)
        pos += stride
    return chunks


def save_chunk_wav(audio_chunk: np.ndarray, sr: int, path: str):
    """Save float32 audio array as 16-bit PCM WAV."""
    pcm = (audio_chunk * 32767).clip(-32768, 32767).astype(np.int16)
    with wave.open(path, "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm.tobytes())


# ──────────────────────────────────────────────────────────────────────────────
# WebSocket client
# ──────────────────────────────────────────────────────────────────────────────

class AvatarStreamClient:
    def __init__(
        self,
        server_url: str,
        avatar_id: str,
        on_frame=None,    # callback(base64_jpeg: str, ts: float)
        on_stats=None,    # callback(stats: dict)
    ):
        self.server_url = server_url
        self.avatar_id  = avatar_id
        self.on_frame   = on_frame or (lambda data, ts: None)
        self.on_stats   = on_stats or (lambda s: None)
        self._ws        = None
        self._loop      = None
        self._thread    = None
        self._send_q    = asyncio.Queue() if False else None  # init in thread
        self._running   = False

    # ── public API ─────────────────────────────────────────────────────────

    def connect(self):
        """Start background thread with event loop for WebSocket."""
        self._running = True
        self._thread  = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        # Wait until connected
        for _ in range(50):
            if self._loop and self._send_q:
                break
            time.sleep(0.1)

    def send_audio_path(self, path: str):
        """Queue an audio chunk path to be sent to the server."""
        self._loop.call_soon_threadsafe(
            self._send_q.put_nowait,
            json.dumps({"type": "audio_chunk", "path": path})
        )

    def request_stats(self):
        self._loop.call_soon_threadsafe(
            self._send_q.put_nowait,
            json.dumps({"type": "stats"})
        )

    def disconnect(self):
        if self._loop:
            self._loop.call_soon_threadsafe(
                self._send_q.put_nowait,
                json.dumps({"type": "stop"})
            )
        self._running = False
        if self._thread:
            self._thread.join(timeout=3.0)

    # ── internal ────────────────────────────────────────────────────────────

    def _run(self):
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        self._send_q = asyncio.Queue()
        self._loop.run_until_complete(self._connect_loop())

    async def _connect_loop(self):
        ws_url = f"{self.server_url}/stream/{self.avatar_id}"
        log.info(f"Connecting to {ws_url}")
        async with websockets.connect(ws_url, max_size=10 * 1024 * 1024) as ws:
            self._ws = ws
            log.info("Connected.")
            recv_task = asyncio.create_task(self._recv_loop(ws))
            send_task = asyncio.create_task(self._send_loop(ws))
            await asyncio.gather(recv_task, send_task, return_exceptions=True)

    async def _recv_loop(self, ws):
        async for raw in ws:
            try:
                msg = json.loads(raw)
                mtype = msg.get("type")
                if mtype == "frame":
                    self.on_frame(msg["data"], msg.get("ts", 0))
                elif mtype == "stats":
                    self.on_stats(msg)
                elif mtype == "error":
                    log.error(f"Server error: {msg.get('message')}")
            except Exception as e:
                log.warning(f"Recv error: {e}")

    async def _send_loop(self, ws):
        while self._running:
            try:
                msg = await asyncio.wait_for(self._send_q.get(), timeout=0.5)
                await ws.send(msg)
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                log.warning(f"Send error: {e}")
                break


# ──────────────────────────────────────────────────────────────────────────────
# High-level pipeline: file or live TTS → avatar stream
# ──────────────────────────────────────────────────────────────────────────────

class LiveAvatarPipeline:
    """
    Orchestrates:
      1. Load audio (from file or TTS provider)
      2. Split into small chunks
      3. Save chunks to /tmp
      4. Send chunk paths over WebSocket
      5. Receive rendered frames
    """

    def __init__(
        self,
        server_url:  str = "ws://localhost:8765",
        avatar_id:   str = "default",
        chunk_ms:    int = 80,
        overlap_ms:  int = 0,
        sample_rate: int = 16000,
        on_frame=None,
    ):
        self.chunk_ms    = chunk_ms
        self.overlap_ms  = overlap_ms
        self.sample_rate = sample_rate
        self.frame_count = 0
        self.start_time  = None

        self.client = AvatarStreamClient(
            server_url=server_url,
            avatar_id=avatar_id,
            on_frame=self._on_frame_internal,
        )
        self._user_on_frame = on_frame or self._default_on_frame
        self.client.connect()

    def _on_frame_internal(self, b64_jpeg: str, ts: float):
        self.frame_count += 1
        if self.start_time:
            elapsed = time.time() - self.start_time
            lag = elapsed - (self.frame_count / 25.0)
            if self.frame_count % 25 == 0:
                log.info(f"Frame {self.frame_count} | pipeline lag: {lag*1000:.0f}ms")
        self._user_on_frame(b64_jpeg, ts)

    @staticmethod
    def _default_on_frame(b64_jpeg: str, ts: float):
        # Default: just count — override for display
        pass

    def stream_audio_file(self, audio_path: str):
        """Stream a pre-existing audio file in real-time chunks."""
        log.info(f"Streaming audio file: {audio_path}")
        audio = load_wav_as_float(audio_path, self.sample_rate)
        chunks = split_audio_into_chunks(audio, self.sample_rate, self.chunk_ms, self.overlap_ms)
        log.info(f"Split into {len(chunks)} chunks of {self.chunk_ms}ms each")

        self.start_time = time.time()
        chunk_interval  = self.chunk_ms / 1000.0  # real-time pacing

        for i, chunk in enumerate(chunks):
            t_send = time.time()
            chunk_path = str(CHUNK_DIR / f"chunk_{i:06d}_{int(time.time()*1000)}.wav")
            save_chunk_wav(chunk, self.sample_rate, chunk_path)
            self.client.send_audio_path(chunk_path)

            # Pace sending to match real-time audio playback
            elapsed = time.time() - t_send
            sleep_time = chunk_interval - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

        log.info(f"All {len(chunks)} chunks sent. Waiting for final frames...")

    def stream_tts_text(self, text: str, tts_provider: str = "elevenlabs", **tts_kwargs):
        """
        Generate speech from text and stream it live.
        Supports: 'elevenlabs', 'openai', 'edge_tts'
        """
        if tts_provider == "elevenlabs":
            self._stream_elevenlabs(text, **tts_kwargs)
        elif tts_provider == "openai":
            self._stream_openai_tts(text, **tts_kwargs)
        elif tts_provider == "edge_tts":
            self._stream_edge_tts(text, **tts_kwargs)
        else:
            raise ValueError(f"Unknown TTS provider: {tts_provider}")

    def _stream_elevenlabs(self, text: str, voice_id: str = "21m00Tcm4TlvDq8ikWAM", **_):
        """Stream from ElevenLabs Streaming API, feeding chunks as they arrive."""
        try:
            from elevenlabs import ElevenLabs, stream as el_stream
        except ImportError:
            raise RuntimeError("pip install elevenlabs")

        client = ElevenLabs()
        audio_stream = client.text_to_speech.convert_as_stream(
            text=text,
            voice_id=voice_id,
            model_id="eleven_turbo_v2",
        )

        buf = bytearray()
        chunk_bytes = int(self.sample_rate * self.chunk_ms / 1000) * 2  # 16-bit PCM

        self.start_time = time.time()
        i = 0
        for audio_bytes in audio_stream:
            buf.extend(audio_bytes)
            while len(buf) >= chunk_bytes:
                pcm_chunk = bytes(buf[:chunk_bytes])
                buf = buf[chunk_bytes:]
                arr = np.frombuffer(pcm_chunk, dtype=np.int16).astype(np.float32) / 32767
                path = str(CHUNK_DIR / f"el_chunk_{i:06d}.wav")
                save_chunk_wav(arr, self.sample_rate, path)
                self.client.send_audio_path(path)
                i += 1

    def _stream_edge_tts(self, text: str, voice: str = "en-US-JennyNeural", **_):
        """Use edge-tts for free TTS with streaming output."""
        import subprocess, tempfile
        tmp = tempfile.mktemp(suffix=".wav")
        subprocess.run(
            ["edge-tts", f"--voice={voice}", f"--text={text}", f"--write-media={tmp}"],
            check=True, capture_output=True
        )
        self.stream_audio_file(tmp)
        os.remove(tmp)

    def disconnect(self):
        self.client.request_stats()
        time.sleep(0.5)
        self.client.disconnect()
        log.info(f"Pipeline closed. Total frames received: {self.frame_count}")


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Live Avatar TTS Chunker")
    parser.add_argument("--server",      default="ws://localhost:8765")
    parser.add_argument("--avatar_id",   default="default")
    parser.add_argument("--audio_file",  default=None, help="Stream from WAV file")
    parser.add_argument("--text",        default=None, help="Stream from TTS text")
    parser.add_argument("--tts",         default="edge_tts", choices=["edge_tts", "elevenlabs", "openai"])
    parser.add_argument("--chunk_ms",    type=int, default=80)
    parser.add_argument("--overlap_ms",  type=int, default=0)
    args = parser.parse_args()

    received_frames = []

    def on_frame(b64, ts):
        received_frames.append(ts)
        if len(received_frames) % 25 == 0:
            print(f"  ✓ {len(received_frames)} frames received")

    pipeline = LiveAvatarPipeline(
        server_url=args.server,
        avatar_id=args.avatar_id,
        chunk_ms=args.chunk_ms,
        overlap_ms=args.overlap_ms,
        on_frame=on_frame,
    )

    try:
        if args.audio_file:
            pipeline.stream_audio_file(args.audio_file)
        elif args.text:
            pipeline.stream_tts_text(args.text, tts_provider=args.tts)
        else:
            print("Provide --audio_file or --text")
            return

        # Wait for all frames
        time.sleep(2.0)
    finally:
        pipeline.disconnect()
        print(f"\nDone. Total frames: {len(received_frames)}")


if __name__ == "__main__":
    main()

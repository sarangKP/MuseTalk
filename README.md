# Live Avatar Pipeline — Ultra Low-Latency MuseTalk

Real-time lip-sync avatar streaming with **target end-to-end latency < 150ms**.

---

## Architecture

```
TTS Provider (ElevenLabs / edge-tts / file)
        │
        ▼  80ms audio chunks
 tts_chunker.py  ──WebSocket──►  live_avatar_server.py
                                        │
                          ┌─────────────┤
                          │             │
                     Whisper tiny    Pre-loaded avatar
                     (~15ms)          (RAM)
                          │             │
                          └──► UNet FP16 forward
                               (~20ms, batch=2)
                                    │
                               VAE decode
                               (~10ms)
                                    │
                           Blend + JPEG encode
                               (~5ms)
                                    │
                          WebSocket frame push
                               (~2ms)
                                    │
                                    ▼
                             viewer.html / your app
```

**Total target latency: ~130–150ms per 2-frame chunk**

---

## Quick Start

### 1. Prepare Avatar (run once)

```bash
python pipeline/prepare_avatar.py \
    --avatar_id my_avatar \
    --video_path ./assets/speaker.mp4 \
    --version v15
```

### 2. Start the Server

```bash
python pipeline/live_avatar_server.py \
    --port 8765 \
    --batch_size 2 \
    --jpeg_quality 85
```

### 3. Load Avatar via REST

```bash
curl -X POST http://localhost:8765/avatar/load \
  -H "Content-Type: application/json" \
  -d '{"avatar_id": "my_avatar", "avatar_path": "./results/v15/avatars/my_avatar"}'
```

### 4. Stream Audio

```bash
# From a WAV file:
python pipeline/tts_chunker.py \
    --avatar_id my_avatar \
    --audio_file ./assets/speech.wav \
    --chunk_ms 80

# From text (edge-tts, free):
python pipeline/tts_chunker.py \
    --avatar_id my_avatar \
    --text "Hello, I am your live avatar." \
    --tts edge_tts
```

### 5. View in Browser

Open `pipeline/viewer.html` in a browser, connect to `ws://localhost:8765`, avatar ID `my_avatar`.

---

## Latency Tuning Knobs

| Parameter | Default | Effect |
|---|---|---|
| `batch_size` | 2 | Lower = less latency, higher = better GPU utilization |
| `chunk_ms` | 80 | Audio chunk size. 40ms = ultra-low latency, 160ms = smoother |
| `jpeg_quality` | 85 | Lower = faster encode, smaller payload |
| `use_float16` | True | ~2x faster inference on modern GPUs |
| `audio_pad_left/right` | 2 | Whisper context window for better sync |

### Latency vs Quality Trade-offs

```
chunk_ms=40,  batch=1 → ~70ms  latency  (choppy at low GPU speed)
chunk_ms=80,  batch=2 → ~130ms latency  ← recommended
chunk_ms=160, batch=4 → ~200ms latency  (smoother, higher quality)
chunk_ms=320, batch=8 → ~350ms latency  (best quality, noticeable delay)
```

---

## File Structure

```
pipeline/
├── live_avatar_server.py   # FastAPI + WebSocket inference server
├── tts_chunker.py          # TTS → audio chunks → WebSocket client
├── prepare_avatar.py       # One-time avatar pre-processing
├── viewer.html             # Browser monitor UI
└── README.md               # This file
```

---

## Dependencies

```bash
pip install fastapi uvicorn websockets
pip install edge-tts          # Free TTS
pip install elevenlabs        # Optional: ElevenLabs TTS
```

All MuseTalk dependencies are inherited from your existing `requirements.txt`.

---

## WebSocket Protocol

**Client → Server:**
```json
{ "type": "audio_chunk", "path": "/tmp/avatar_chunks/chunk_000001.wav" }
{ "type": "stats" }
{ "type": "stop" }
```

**Server → Client:**
```json
{ "type": "frame", "data": "<base64-jpeg>", "ts": 1234567890.123 }
{ "type": "stats", "avg_latency_ms": 128.4, "frames_sent": 750, "chunks_processed": 37 }
{ "type": "error", "message": "..." }
```

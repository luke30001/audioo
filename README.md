# Whisper Large V3 RunPod Worker

[![Deploy on RunPod](https://api.runpod.io/badge/luke30001/audioo)](https://console.runpod.io/hub/luke30001/audioo)

Serverless worker that transcribes audio with `openai/whisper-large-v3` on GPU.

## Build & run locally
```bash
docker build -t whisper-large-v3 -f runpod/Dockerfile .
docker run --gpus all -p 3000:3000 whisper-large-v3
```

## Expected inputs
- `audio_url` (string) – URL to download audio (https/http, any ffmpeg-supported format).
- `audio_base64` (string) – Base64-encoded audio bytes (alternative to `audio_url`).
- `audio_path` (string) – Local path already present in the container (for testing).
- `language` (string, optional) – ISO language hint (e.g., `"en"`, `"it"`).
- `timestamps` (bool|"word", optional) – `false` for no timestamps, `true` for chunk timestamps, `"word"` for per-word (default).
- `chunk_length_s` (number, optional) – Sliding window size; default `30`.
- `max_new_tokens` (number, optional) – Generation limit; default `448`.

Provide exactly one of `audio_url`, `audio_base64`, or `audio_path`.

### Sample job payload
```json
{
  "input": {
    "audio_url": "https://example.com/audio.mp3",
    "language": "en",
    "timestamps": "word",
    "chunk_length_s": 30,
    "max_new_tokens": 512
  }
}
```

The handler returns the Hugging Face ASR pipeline result, e.g.:
```json
{
  "text": "transcribed text ...",
  "chunks": [
    {"text": "transcribed", "timestamp": [0.0, 0.5]},
    {"text": "text", "timestamp": [0.5, 1.0]}
  ]
}
```

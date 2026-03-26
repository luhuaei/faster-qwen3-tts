# faster-qwen3-tts AI Pod Service

This directory is the AI Pod service payload for the Orin deployment of
`faster-qwen3-tts`.

The service exposes an OpenAI-compatible TTS server on `qwen3-tts-ai` and
waits for the startup warmup to finish before the container becomes healthy.

API 文档：

- [openai-api.md](/home/catdog/faster-qwen3-tts/ai-pod-service/openai-api.md)

Image tag:

- `lzcbox-90fc188b.lan:5001/x/faster-qwen3-tts:0.6b-custom-openai-orin-v4`

Key runtime defaults:

- model: `Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice` baked into the image
- mode: `custom`
- default voice: `vivian` when available, otherwise first supported speaker
- deterministic defaults: `do_sample=false`, `temperature=1.0`
- chunk size: `8`

The image can be rebuilt and pushed from this repo with:

```bash
python3 scripts/build_orin_aipod_image.py
```

If the target registry is temporarily unreachable from the Orin host, build only:

```bash
ORIN_AIPOD_PUSH=0 python3 scripts/build_orin_aipod_image.py
```

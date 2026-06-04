# faster-qwen3-tts AI Pod Service

This directory is the AI Pod service payload for the Orin deployment of
`faster-qwen3-tts`.

The service exposes an OpenAI-compatible TTS server on `qwen3tts-ai` and
waits for the startup warmup to finish before the container becomes healthy.

API 文档：

- [openai-api.md](/home/catdog/faster-qwen3-tts/ai-pod-service/openai-api.md)

Image tag:

- Built image tags are generated as `faster-qwen3-tts:<git-abbr>` by default.

Key runtime defaults:

- model: `Qwen/Qwen3-TTS-12Hz-0.6B-Base` baked into the image
- mode: `clone`
- builtin voices: `dylan` `eric` `ono_anna` `ryan` `serena` `sohee` `uncle_fu` `vivian`
- voices config: bundled at `/opt/build/faster-qwen3-tts/voices.json`
- default voice: `vivian`
- chunk size: `8`

The image can be rebuilt on a Jetson Docker daemon from this repo with:

```bash
uv run scripts/build_jetson_image.py \
  --target orin \
  --model-dir /path/to/Qwen3-TTS-12Hz-0.6B-Base \
  --tag-prefix 0.6b-base-clone-openai-orin
```

Use `--ssh nvidia@host` or `--docker-host ssh://nvidia@host` to override the
target preset. The image is built and left on the remote Docker daemon; this
script does not push to a registry.

```bash
docker --host ssh://nvidia@lzc-pod-juyIZt.lan version
```

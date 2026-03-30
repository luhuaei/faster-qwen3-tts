ARG BASE_IMAGE=127.0.0.1:5001/x/lzc-aipod-vllm:bffa39b-orin
ARG PREBUILT_APP_IMAGE=127.0.0.1:5001/x/faster-qwen3-tts:0.6b-custom-openai-orin-v1
ARG MODEL_NAME=Qwen3-TTS-12Hz-0.6B-CustomVoice
ARG MODEL_DIR=/opt/models/Qwen3-TTS-12Hz-0.6B-CustomVoice
ARG QWEN_TTS_MODEL=/opt/models/Qwen3-TTS-12Hz-0.6B-CustomVoice
ARG QWEN_TTS_MODE=custom
ARG QWEN_TTS_DEFAULT_VOICE=vivian

FROM ${PREBUILT_APP_IMAGE} AS prebuilt_app
FROM ${BASE_IMAGE}

ARG MODEL_NAME
ARG MODEL_DIR
ARG QWEN_TTS_MODEL
ARG QWEN_TTS_MODE
ARG QWEN_TTS_DEFAULT_VOICE

ENV DEBIAN_FRONTEND=noninteractive \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    APP_VENV=/opt/faster-qwen3-tts-venv \
    HF_HUB_DISABLE_SYMLINKS_WARNING=1 \
    PATH=/opt/faster-qwen3-tts-venv/bin:${PATH} \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/root/.cache/huggingface \
    XDG_CACHE_HOME=/root/.cache \
    QWEN_TTS_MODEL=${QWEN_TTS_MODEL} \
    QWEN_TTS_MODE=${QWEN_TTS_MODE} \
    QWEN_TTS_DEFAULT_VOICE=${QWEN_TTS_DEFAULT_VOICE} \
    QWEN_TTS_LANGUAGE=Auto \
    QWEN_TTS_CHUNK_SIZE=8 \
    QWEN_TTS_WARMUP_MAX_NEW_TOKENS=32

WORKDIR /opt/build

RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg libsndfile1 opus-tools sox \
    && rm -rf /var/lib/apt/lists/*

COPY --from=prebuilt_app /opt/faster-qwen3-tts-venv /opt/faster-qwen3-tts-venv
COPY MANIFEST.in README.md pyproject.toml /opt/build/faster-qwen3-tts/
COPY faster_qwen3_tts /opt/build/faster-qwen3-tts/faster_qwen3_tts
COPY examples/openai_server.py /opt/build/faster-qwen3-tts/examples/openai_server.py
COPY models/${MODEL_NAME} ${MODEL_DIR}

RUN uv pip install --python "${APP_VENV}/bin/python3" "qwen-tts>=0.1.1" \
    && uv pip install --python "${APP_VENV}/bin/python3" -e "/opt/build/faster-qwen3-tts[demo]" --no-build-isolation \
    && find /opt/models -mindepth 1 -maxdepth 1 ! -name "${MODEL_NAME}" -prune -exec rm -rf '{}' + \
    && find /opt/build -name '__pycache__' -type d -prune -exec rm -rf '{}' + \
    && find /root/.cache -type d -name '.locks' -prune -exec rm -rf '{}' + \
    && rm -rf /root/.cache/uv

ENV HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1

WORKDIR /workspace
EXPOSE 8000
ENTRYPOINT ["python3", "/opt/build/faster-qwen3-tts/examples/openai_server.py"]

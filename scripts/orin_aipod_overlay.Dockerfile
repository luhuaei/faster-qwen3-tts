ARG BASE_IMAGE=lzcbox-90fc188b.lan:5001/x/faster-qwen3-tts:0.6b-custom-openai-orin-v4
FROM ${BASE_IMAGE}

ENV QWEN_TTS_TEMPERATURE=1.0 \
    QWEN_TTS_DO_SAMPLE=0

COPY openai_server.py /opt/build/faster-qwen3-tts/examples/openai_server.py

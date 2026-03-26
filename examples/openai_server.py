#!/usr/bin/env python3
"""
OpenAI-compatible TTS API server for faster-qwen3-tts.

Exposes POST /v1/audio/speech compatible with OpenAI's TTS API, enabling
integration with OpenWebUI, llama-swap, and other OpenAI-compatible clients.

Usage:
    pip install "faster-qwen3-tts[demo]"

    # Single default voice:
    python examples/openai_server.py \\
        --ref-audio voice.wav --ref-text "Reference transcription" \\
        --language English

    # Multiple named voices from a JSON config:
    python examples/openai_server.py --voices voices.json

    # Custom model and port:
    python examples/openai_server.py \\
        --model Qwen/Qwen3-TTS-12Hz-0.6B-Base \\
        --ref-audio voice.wav --ref-text "transcript" \\
        --port 8000

Voices config (voices.json):
    {
        "alloy": {"ref_audio": "voice.wav", "ref_text": "...", "language": "English"},
        "echo":  {"ref_audio": "voice2.wav", "ref_text": "...", "language": "English"}
    }

API usage:
    curl -s http://localhost:8000/v1/audio/speech \\
        -H "Content-Type: application/json" \\
        -d '{"model": "tts-1", "input": "Hello!", "voice": "alloy", "response_format": "wav", "instruct": "Speak gently with a relaxed cadence."}' \\
        --output speech.wav
"""
import argparse
import asyncio
import io
import json
import logging
import os
import queue
import struct
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Any, AsyncGenerator, Optional

import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel, ValidationError

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Global state
# ---------------------------------------------------------------------------

app = FastAPI(title="faster-qwen3-tts OpenAI-compatible API")

tts_model = None
voices: dict = {}
default_voice: Optional[str] = None
generation_mode = "clone"
SAMPLE_RATE = 24000  # updated once the model loads
_model_lock = threading.Lock()  # prevent concurrent GPU inference
startup_warmup_enabled = True
startup_warmup_completed = False
startup_warmup_seconds: Optional[float] = None
MAX_UPLOAD_BYTES = 10 * 1024 * 1024


def _is_ready() -> bool:
    return tts_model is not None and (not startup_warmup_enabled or startup_warmup_completed)


def _parse_optional_bool(value: Optional[str]) -> Optional[bool]:
    if value is None:
        return None
    lowered = value.strip().lower()
    if not lowered:
        return None
    if lowered in {"1", "true", "yes", "on"}:
        return True
    if lowered in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"Invalid boolean value: {value!r}")


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------


class SpeechRequest(BaseModel):
    model: str = "tts-1"
    input: str
    voice: str = "alloy"
    response_format: str = "wav"  # wav | pcm | mp3
    speed: float = 1.0           # accepted but not yet applied
    instruct: Optional[str] = None
    language: Optional[str] = None
    temperature: Optional[float] = None
    do_sample: Optional[bool] = None
    repetition_penalty: Optional[float] = None


# ---------------------------------------------------------------------------
# Audio helpers
# ---------------------------------------------------------------------------


def _to_pcm16(pcm: np.ndarray) -> bytes:
    """Convert float32 numpy array to raw 16-bit little-endian PCM bytes."""
    return np.clip(pcm * 32768, -32768, 32767).astype(np.int16).tobytes()


def _wav_header(sample_rate: int, data_len: int = 0xFFFFFFFF) -> bytes:
    """Build a WAV header.  Use data_len=0xFFFFFFFF for streaming (unknown size)."""
    n_channels = 1
    bits = 16
    byte_rate = sample_rate * n_channels * bits // 8
    block_align = n_channels * bits // 8
    riff_size = 0xFFFFFFFF if data_len == 0xFFFFFFFF else 36 + data_len
    buf = io.BytesIO()
    buf.write(b"RIFF")
    buf.write(struct.pack("<I", riff_size))
    buf.write(b"WAVE")
    buf.write(b"fmt ")
    buf.write(struct.pack("<IHHIIHH", 16, 1, n_channels, sample_rate,
                          byte_rate, block_align, bits))
    buf.write(b"data")
    buf.write(struct.pack("<I", data_len))
    return buf.getvalue()


def _to_wav_bytes(pcm: np.ndarray, sample_rate: int) -> bytes:
    """Convert float32 numpy array to a complete WAV file in memory."""
    raw = _to_pcm16(pcm)
    return _wav_header(sample_rate, len(raw)) + raw


def _to_mp3_bytes(pcm: np.ndarray, sample_rate: int) -> bytes:
    """Convert float32 numpy array to MP3 bytes (requires pydub + ffmpeg)."""
    try:
        from pydub import AudioSegment
    except ImportError:
        raise HTTPException(
            status_code=400,
            detail="response_format='mp3' requires pydub: pip install pydub",
        )
    segment = AudioSegment(
        _to_pcm16(pcm),
        frame_rate=sample_rate,
        sample_width=2,
        channels=1,
    )
    buf = io.BytesIO()
    segment.export(buf, format="mp3")
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Voice resolution
# ---------------------------------------------------------------------------


def _casefold_voice_name(name: str) -> str:
    return name.casefold()


def _find_configured_voice_name(voice_name: str) -> Optional[str]:
    if voice_name in voices:
        return voice_name
    requested_name = _casefold_voice_name(voice_name)
    for configured_name in voices:
        if _casefold_voice_name(configured_name) == requested_name:
            return configured_name
    return None


def _resolve_preferred_default_voice(preferred_name: str) -> Optional[str]:
    stripped_name = preferred_name.strip()
    if not stripped_name:
        return None
    return _find_configured_voice_name(stripped_name)


def _validate_voice_name_uniqueness(configured_voices: dict) -> None:
    normalized_names: dict[str, str] = {}
    for voice_name in configured_voices:
        folded_name = _casefold_voice_name(voice_name)
        existing_name = normalized_names.get(folded_name)
        if existing_name is not None and existing_name != voice_name:
            raise ValueError(
                "Voice names must be unique when compared case-insensitively: "
                f"{existing_name!r} conflicts with {voice_name!r}"
            )
        normalized_names[folded_name] = voice_name


def resolve_voice(voice_name: str) -> dict:
    """Return voice config dict or fall back to default, else raise 400."""
    resolved_voice_name = _find_configured_voice_name(voice_name)
    if resolved_voice_name is not None:
        return voices[resolved_voice_name]
    if default_voice and default_voice in voices:
        logger.warning(
            "Voice %r not configured; falling back to default voice %r",
            voice_name,
            default_voice,
        )
        return voices[default_voice]
    raise HTTPException(
        status_code=400,
        detail=(
            f"Voice {voice_name!r} is not configured. "
            f"Available voices: {list(voices.keys())}"
        ),
    )


# ---------------------------------------------------------------------------
# Streaming helper: run sync generator in a background thread
# ---------------------------------------------------------------------------


def _resolve_instruct(voice_cfg: dict, request_instruct: Optional[str]) -> Optional[str]:
    voice_cfg = voice_cfg or {}
    if request_instruct is None:
        return voice_cfg.get("instruct")
    return request_instruct


def _resolve_language(voice_cfg: dict, request_language: Optional[str]) -> str:
    voice_cfg = voice_cfg or {}
    if request_language is None:
        return voice_cfg.get("language", "Auto")
    return request_language


def _resolve_optional_generation_setting(
    voice_cfg: Optional[dict],
    key: str,
    request_value,
):
    voice_cfg = voice_cfg or {}
    if request_value is not None:
        return request_value
    return voice_cfg.get(key)


def _build_generation_kwargs(
    voice_cfg: Optional[dict],
    request_instruct: Optional[str],
    request_language: Optional[str],
    request_temperature: Optional[float],
    request_do_sample: Optional[bool],
    request_repetition_penalty: Optional[float],
) -> dict:
    kwargs = {
        "language": _resolve_language(voice_cfg, request_language),
        "instruct": _resolve_instruct(voice_cfg, request_instruct),
    }
    temperature = _resolve_optional_generation_setting(
        voice_cfg,
        "temperature",
        request_temperature,
    )
    if temperature is not None:
        kwargs["temperature"] = temperature
    do_sample = _resolve_optional_generation_setting(
        voice_cfg,
        "do_sample",
        request_do_sample,
    )
    if do_sample is not None:
        kwargs["do_sample"] = do_sample
    repetition_penalty = _resolve_optional_generation_setting(
        voice_cfg,
        "repetition_penalty",
        request_repetition_penalty,
    )
    if repetition_penalty is not None:
        kwargs["repetition_penalty"] = repetition_penalty
    return kwargs


def _normalize_optional_form_value(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, str):
        value = value.strip()
        if not value:
            return None
    return value


async def _read_upload_bytes(upload: UploadFile, field_name: str) -> bytes:
    content = await upload.read()
    if not content:
        raise HTTPException(status_code=400, detail=f"{field_name!r} file is empty")
    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=400,
            detail=(
                f"{field_name!r} file is too large "
                f"({len(content) / 1024 / 1024:.1f} MiB > {MAX_UPLOAD_BYTES / 1024 / 1024:.1f} MiB)"
            ),
        )
    return content


def _temporary_upload_path(filename: Optional[str]) -> str:
    suffix = Path(filename or "upload.bin").suffix or ".bin"
    fd, path = tempfile.mkstemp(prefix="faster-qwen3-tts-", suffix=suffix)
    os.close(fd)
    return path


def _voice_clone_download_name(filename: Optional[str]) -> str:
    stem = Path(filename or "voice_clone").stem or "voice_clone"
    return f"{stem}.pt"


def _build_voice_clone_prompt_from_tensor(value: Any) -> dict:
    if not isinstance(value, torch.Tensor):
        raise HTTPException(
            status_code=400,
            detail="voice_clone_pt must contain a saved torch.Tensor speaker embedding",
        )
    device = getattr(tts_model, "device", "cpu")
    return {"ref_spk_embedding": [value.detach().to(device)]}


async def _load_voice_clone_prompt_upload(upload: UploadFile) -> dict:
    content = await _read_upload_bytes(upload, "voice_clone_pt")
    try:
        loaded = torch.load(io.BytesIO(content), map_location="cpu", weights_only=True)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Failed to load voice_clone_pt: {exc}") from exc
    return _build_voice_clone_prompt_from_tensor(loaded)


async def _extract_voice_clone_pt_bytes(ref_audio: UploadFile) -> bytes:
    if generation_mode != "clone":
        raise HTTPException(status_code=400, detail="voice clone pt export is only available in clone mode")

    audio_bytes = await _read_upload_bytes(ref_audio, "ref_audio")
    tmp_path = _temporary_upload_path(ref_audio.filename)
    try:
        with open(tmp_path, "wb") as f:
            f.write(audio_bytes)
        with _model_lock:
            prompt_items = tts_model.model.create_voice_clone_prompt(
                ref_audio=tmp_path,
                ref_text="",
                x_vector_only_mode=True,
            )
        if not prompt_items:
            raise HTTPException(status_code=500, detail="Voice clone prompt extraction returned no prompt items")
        speaker_embedding = getattr(prompt_items[0], "ref_spk_embedding", None)
        if not isinstance(speaker_embedding, torch.Tensor):
            raise HTTPException(status_code=500, detail="Voice clone prompt extraction returned no speaker embedding")
        buf = io.BytesIO()
        torch.save(speaker_embedding.detach().cpu(), buf)
        return buf.getvalue()
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


def _speech_request_from_form(form_data: Any) -> SpeechRequest:
    payload = {}
    for field_name in ("model", "input", "voice", "response_format", "speed"):
        value = _normalize_optional_form_value(form_data.get(field_name))
        if value is not None:
            payload[field_name] = value
    for field_name in ("instruct", "language", "temperature", "do_sample", "repetition_penalty"):
        value = _normalize_optional_form_value(form_data.get(field_name))
        if value is not None:
            payload[field_name] = value
    return SpeechRequest(**payload)


async def _stream_chunks(
    voice_cfg: Optional[dict],
    text: str,
    request_instruct: Optional[str],
    request_language: Optional[str],
    request_temperature: Optional[float],
    request_do_sample: Optional[bool],
    request_repetition_penalty: Optional[float],
    voice_clone_prompt: Optional[dict] = None,
) -> AsyncGenerator[bytes, None]:
    """
    Run generate_voice_clone_streaming in a background thread and yield
    raw PCM bytes for each chunk as they arrive.
    """
    q: queue.Queue = queue.Queue()
    _DONE = object()

    def producer():
        try:
            with _model_lock:
                if generation_mode == "clone":
                    generation_kwargs = _build_generation_kwargs(
                        voice_cfg,
                        request_instruct,
                        request_language,
                        request_temperature,
                        request_do_sample,
                        request_repetition_penalty,
                    )
                    clone_kwargs = {
                        "text": text,
                        "chunk_size": (voice_cfg or {}).get("chunk_size", 12),
                        "non_streaming_mode": False,
                        **generation_kwargs,
                    }
                    if voice_clone_prompt is not None:
                        clone_kwargs["voice_clone_prompt"] = voice_clone_prompt
                    else:
                        clone_kwargs["ref_audio"] = voice_cfg["ref_audio"]
                        clone_kwargs["ref_text"] = voice_cfg.get("ref_text", "")
                    generator = tts_model.generate_voice_clone_streaming(**clone_kwargs)
                elif generation_mode == "custom":
                    generation_kwargs = _build_generation_kwargs(
                        voice_cfg,
                        request_instruct,
                        request_language,
                        request_temperature,
                        request_do_sample,
                        request_repetition_penalty,
                    )
                    generator = tts_model.generate_custom_voice_streaming(
                        text=text,
                        speaker=voice_cfg.get("speaker", ""),
                        chunk_size=voice_cfg.get("chunk_size", 12),
                        **generation_kwargs,
                    )
                else:
                    raise RuntimeError(f"Unsupported generation mode: {generation_mode}")

                for chunk, _sr, _timing in generator:
                    q.put(chunk)
        except Exception as exc:
            q.put(exc)
        finally:
            q.put(_DONE)

    thread = threading.Thread(target=producer, daemon=True)
    thread.start()

    loop = asyncio.get_event_loop()
    while True:
        item = await loop.run_in_executor(None, q.get)
        if item is _DONE:
            break
        if isinstance(item, Exception):
            raise item
        yield _to_pcm16(item)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/health")
async def health():
    return {
        "status": "ok" if _is_ready() else "starting",
        "ready": _is_ready(),
        "model_loaded": tts_model is not None,
        "mode": generation_mode,
        "voices": list(voices.keys()),
        "startup_warmup_enabled": startup_warmup_enabled,
        "startup_warmup_completed": startup_warmup_completed,
        "startup_warmup_seconds": startup_warmup_seconds,
    }


@app.get("/v1/audio/voices")
async def list_voices():
    return {"voices": list(voices.keys()), "default_voice": default_voice, "mode": generation_mode}


async def create_speech(req: SpeechRequest, voice_clone_prompt: Optional[dict] = None):
    if tts_model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    if not req.input.strip():
        raise HTTPException(status_code=400, detail="'input' text is empty")
    if voice_clone_prompt is not None and generation_mode != "clone":
        raise HTTPException(status_code=400, detail="voice_clone_pt is only supported in clone mode")

    if generation_mode == "clone" and voice_clone_prompt is not None:
        voice_cfg = None
    else:
        voice_cfg = resolve_voice(req.voice)
    fmt = req.response_format.lower()

    _CONTENT_TYPES = {
        "wav": "audio/wav",
        "pcm": "audio/pcm",
        "mp3": "audio/mpeg",
    }
    if fmt not in _CONTENT_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"response_format {fmt!r} not supported. Use: wav, pcm, mp3",
        )
    content_type = _CONTENT_TYPES[fmt]

    # --- MP3: generate all audio, then encode (non-streaming) ---
    if fmt == "mp3":
        loop = asyncio.get_event_loop()

        def _generate():
            with _model_lock:
                if generation_mode == "clone":
                    generation_kwargs = _build_generation_kwargs(
                        voice_cfg,
                        req.instruct,
                        req.language,
                        req.temperature,
                        req.do_sample,
                        req.repetition_penalty,
                    )
                    clone_kwargs = {"text": req.input, **generation_kwargs}
                    if voice_clone_prompt is not None:
                        clone_kwargs["voice_clone_prompt"] = voice_clone_prompt
                    else:
                        clone_kwargs["ref_audio"] = voice_cfg["ref_audio"]
                        clone_kwargs["ref_text"] = voice_cfg.get("ref_text", "")
                    return tts_model.generate_voice_clone(**clone_kwargs)
                if generation_mode == "custom":
                    generation_kwargs = _build_generation_kwargs(
                        voice_cfg,
                        req.instruct,
                        req.language,
                        req.temperature,
                        req.do_sample,
                        req.repetition_penalty,
                    )
                    return tts_model.generate_custom_voice(
                        text=req.input,
                        speaker=voice_cfg.get("speaker", ""),
                        **generation_kwargs,
                    )
                raise RuntimeError(f"Unsupported generation mode: {generation_mode}")

        audio_arrays, sr = await loop.run_in_executor(None, _generate)
        audio = audio_arrays[0] if audio_arrays else np.zeros(1, dtype=np.float32)
        return Response(content=_to_mp3_bytes(audio, sr), media_type=content_type)

    # --- WAV / PCM: stream chunks as they are generated ---
    async def audio_stream():
        if fmt == "wav":
            yield _wav_header(SAMPLE_RATE)  # stream with unknown data length
        async for raw_chunk in _stream_chunks(
            voice_cfg,
            req.input,
            req.instruct,
            req.language,
            req.temperature,
            req.do_sample,
            req.repetition_penalty,
            voice_clone_prompt=voice_clone_prompt,
        ):
            yield raw_chunk

    return StreamingResponse(audio_stream(), media_type=content_type)


@app.post("/v1/audio/speech")
async def create_speech_http(request: Request):
    content_type = request.headers.get("content-type", "")
    if "application/json" in content_type:
        payload = await request.json()
        try:
            req = SpeechRequest(**payload)
        except ValidationError as exc:
            raise HTTPException(status_code=422, detail=exc.errors()) from exc
        return await create_speech(req)
    if "multipart/form-data" in content_type:
        form_data = await request.form()
        try:
            req = _speech_request_from_form(form_data)
        except ValidationError as exc:
            raise HTTPException(status_code=422, detail=exc.errors()) from exc
        voice_clone_prompt = None
        voice_clone_pt = form_data.get("voice_clone_pt")
        if hasattr(voice_clone_pt, "read") and getattr(voice_clone_pt, "filename", ""):
            voice_clone_prompt = await _load_voice_clone_prompt_upload(voice_clone_pt)
        return await create_speech(req, voice_clone_prompt=voice_clone_prompt)
    raise HTTPException(
        status_code=415,
        detail="Unsupported Content-Type. Use application/json or multipart/form-data",
    )


@app.post("/v1/audio/voice-clone/pt")
async def create_voice_clone_pt(
    ref_audio: UploadFile = File(...),
    model: str = Form("tts-1"),
    format: str = Form("pt"),
):
    del model
    if format.lower() != "pt":
        raise HTTPException(status_code=400, detail="Only format='pt' is supported")
    if tts_model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    pt_bytes = await _extract_voice_clone_pt_bytes(ref_audio)
    filename = _voice_clone_download_name(ref_audio.filename)
    return Response(
        content=pt_bytes,
        media_type="application/octet-stream",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def _parse_args():
    p = argparse.ArgumentParser(
        description="OpenAI-compatible TTS server for faster-qwen3-tts",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--model",
        default=os.environ.get("QWEN_TTS_MODEL", "Qwen/Qwen3-TTS-12Hz-1.7B-Base"),
        help="HuggingFace model ID or local path (default: Qwen/Qwen3-TTS-12Hz-1.7B-Base)",
    )
    p.add_argument(
        "--mode",
        default=os.environ.get("QWEN_TTS_MODE", "clone"),
        choices=["clone", "custom"],
        help="Generation mode (default: clone)",
    )
    p.add_argument(
        "--voices",
        default=os.environ.get("QWEN_TTS_VOICES"),
        metavar="FILE",
        help="JSON file mapping voice names to {ref_audio, ref_text, language}",
    )
    p.add_argument(
        "--ref-audio",
        default=os.environ.get("QWEN_TTS_REF_AUDIO"),
        metavar="FILE",
        help="Reference audio file when --voices is not used",
    )
    p.add_argument(
        "--ref-text",
        default=os.environ.get("QWEN_TTS_REF_TEXT", ""),
        help="Transcript of --ref-audio",
    )
    p.add_argument(
        "--language",
        default=os.environ.get("QWEN_TTS_LANGUAGE", "Auto"),
        help="Target language (English, French, Auto, …) when --voices is not used",
    )
    p.add_argument(
        "--speakers",
        default=os.environ.get("QWEN_TTS_SPEAKERS", ""),
        help="Comma-separated speaker list for --mode custom (default: all available speakers)",
    )
    p.add_argument(
        "--default-voice",
        default=os.environ.get("QWEN_TTS_DEFAULT_VOICE", ""),
        help="Default voice name returned when request voice is missing",
    )
    p.add_argument(
        "--instruct",
        default=os.environ.get("QWEN_TTS_INSTRUCT", ""),
        help="Optional instruct text passed to CustomVoice generation",
    )
    p.add_argument(
        "--temperature",
        type=float,
        default=(
            float(os.environ["QWEN_TTS_TEMPERATURE"])
            if os.environ.get("QWEN_TTS_TEMPERATURE", "").strip()
            else None
        ),
        help="Optional default temperature applied when request/voice config omits it",
    )
    p.add_argument(
        "--do-sample",
        type=_parse_optional_bool,
        default=_parse_optional_bool(os.environ.get("QWEN_TTS_DO_SAMPLE")),
        help="Optional default sampling flag applied when request/voice config omits it",
    )
    p.add_argument(
        "--repetition-penalty",
        type=float,
        default=(
            float(os.environ["QWEN_TTS_REPETITION_PENALTY"])
            if os.environ.get("QWEN_TTS_REPETITION_PENALTY", "").strip()
            else None
        ),
        help="Optional default repetition penalty applied when request/voice config omits it",
    )
    p.add_argument(
        "--chunk-size",
        type=int,
        default=int(os.environ.get("QWEN_TTS_CHUNK_SIZE", "12")),
        help="Streaming chunk size (default: 12)",
    )
    p.add_argument(
        "--warmup-text",
        default=os.environ.get(
            "QWEN_TTS_WARMUP_TEXT",
            "This startup warmup request captures CUDA graphs before the service becomes ready.",
        ),
        help="Warmup text used before the server is marked ready",
    )
    p.add_argument(
        "--warmup-max-new-tokens",
        type=int,
        default=int(os.environ.get("QWEN_TTS_WARMUP_MAX_NEW_TOKENS", "32")),
        help="Max new tokens for the startup warmup request (default: 32)",
    )
    p.add_argument(
        "--no-startup-warmup",
        action="store_true",
        help="Disable startup warmup before the server is marked ready",
    )
    p.add_argument("--host", default="0.0.0.0", help="Bind host (default: 0.0.0.0)")
    p.add_argument("--port", type=int, default=8000, help="Bind port (default: 8000)")
    p.add_argument("--device", default="cuda", help="Torch device (default: cuda)")
    return p.parse_args()


def _run_startup_warmup(warmup_text: str, warmup_max_new_tokens: int) -> None:
    global startup_warmup_completed, startup_warmup_seconds

    if not voices:
        logger.warning("Skipping startup warmup because no voices are configured")
        return

    voice_name = default_voice or next(iter(voices))
    voice_cfg = resolve_voice(voice_name)
    started = time.perf_counter()

    with _model_lock:
        if generation_mode == "clone":
            tts_model.generate_voice_clone(
                text=warmup_text,
                language=voice_cfg.get("language", "Auto"),
                ref_audio=voice_cfg["ref_audio"],
                ref_text=voice_cfg.get("ref_text", ""),
                max_new_tokens=warmup_max_new_tokens,
            )
        elif generation_mode == "custom":
            tts_model.generate_custom_voice(
                text=warmup_text,
                speaker=voice_cfg.get("speaker", ""),
                language=voice_cfg.get("language", "Auto"),
                instruct=voice_cfg.get("instruct", ""),
                max_new_tokens=warmup_max_new_tokens,
            )
        else:
            raise RuntimeError(f"Unsupported generation mode: {generation_mode}")

    startup_warmup_seconds = time.perf_counter() - started
    startup_warmup_completed = True
    logger.info(
        "Startup warmup completed in %.3fs using voice %s",
        startup_warmup_seconds,
        voice_name,
    )


def _configure_predictor_graph_defaults(
    model,
    *,
    temperature: Optional[float],
    do_sample: Optional[bool],
) -> None:
    predictor_graph = getattr(model, "predictor_graph", None)
    if predictor_graph is None:
        return
    if temperature is not None:
        predictor_graph.temperature = temperature
    if do_sample is not None:
        predictor_graph.do_sample = do_sample
    if do_sample is False:
        predictor_graph.top_k = 0
        predictor_graph.top_p = 1.0
    logger.info(
        "Predictor graph sampling defaults: do_sample=%s temperature=%s top_k=%s top_p=%s",
        predictor_graph.do_sample,
        predictor_graph.temperature,
        predictor_graph.top_k,
        predictor_graph.top_p,
    )


def main():
    global tts_model, voices, default_voice, SAMPLE_RATE, generation_mode
    global startup_warmup_enabled, startup_warmup_completed, startup_warmup_seconds

    args = _parse_args()
    startup_warmup_enabled = not args.no_startup_warmup
    startup_warmup_completed = False
    startup_warmup_seconds = None

    from faster_qwen3_tts import FasterQwen3TTS

    logger.info("Loading model %s on %s …", args.model, args.device)
    tts_model = FasterQwen3TTS.from_pretrained(
        args.model,
        device=args.device,
        dtype=torch.bfloat16,
    )
    _configure_predictor_graph_defaults(
        tts_model,
        temperature=args.temperature,
        do_sample=args.do_sample,
    )
    SAMPLE_RATE = tts_model.sample_rate
    generation_mode = args.mode

    if generation_mode == "clone":
        if args.voices:
            with open(args.voices) as f:
                voices = json.load(f)
            for voice_cfg in voices.values():
                voice_cfg.setdefault("chunk_size", args.chunk_size)
                if args.temperature is not None:
                    voice_cfg.setdefault("temperature", args.temperature)
                if args.do_sample is not None:
                    voice_cfg.setdefault("do_sample", args.do_sample)
                if args.repetition_penalty is not None:
                    voice_cfg.setdefault("repetition_penalty", args.repetition_penalty)
            _validate_voice_name_uniqueness(voices)
            default_voice = _resolve_preferred_default_voice(args.default_voice) or next(iter(voices))
            logger.info("Loaded %d voice(s) from %s", len(voices), args.voices)
        elif args.ref_audio:
            voices = {
                args.default_voice or "default": {
                    "ref_audio": args.ref_audio,
                    "ref_text": args.ref_text,
                    "language": args.language,
                    "chunk_size": args.chunk_size,
                    "temperature": args.temperature,
                    "do_sample": args.do_sample,
                    "repetition_penalty": args.repetition_penalty,
                }
            }
            default_voice = next(iter(voices))
            logger.info("Using single clone voice from --ref-audio: %s", args.ref_audio)
        else:
            print(
                "ERROR: clone mode requires --ref-audio <file> or --voices <config.json>",
                file=sys.stderr,
            )
            sys.exit(1)
    elif generation_mode == "custom":
        supported_speakers = tts_model.model.get_supported_speakers() or []
        if not supported_speakers:
            print("ERROR: CustomVoice mode found no supported speakers in the loaded model", file=sys.stderr)
            sys.exit(1)
        requested_speakers = [item.strip() for item in args.speakers.split(",") if item.strip()]
        selected_speakers = requested_speakers or supported_speakers
        supported_lookup = {speaker.lower(): speaker for speaker in supported_speakers}
        unresolved = [speaker for speaker in selected_speakers if speaker.lower() not in supported_lookup]
        if unresolved:
            print(
                f"ERROR: unsupported speakers for model {args.model}: {unresolved}. "
                f"Available: {supported_speakers}",
                file=sys.stderr,
            )
            sys.exit(1)
        voices = {
            speaker: {
                "speaker": supported_lookup[speaker.lower()],
                "language": args.language,
                "instruct": args.instruct,
                "chunk_size": args.chunk_size,
                "temperature": args.temperature,
                "do_sample": args.do_sample,
                "repetition_penalty": args.repetition_penalty,
            }
            for speaker in selected_speakers
        }
        preferred_default = args.default_voice.strip() if args.default_voice else ""
        if preferred_default:
            resolved_default = supported_lookup.get(preferred_default.lower())
            if resolved_default is None:
                logger.warning(
                    "Requested default voice %r is not available in %s; falling back",
                    preferred_default,
                    args.model,
                )
            else:
                default_voice = resolved_default
        if not default_voice and "vivian" in supported_lookup:
            default_voice = supported_lookup["vivian"]
        if not default_voice:
            default_voice = next(iter(voices))
        logger.info("Configured %d CustomVoice speaker(s): %s", len(voices), ", ".join(voices))
    else:
        print(f"ERROR: unsupported mode {generation_mode!r}", file=sys.stderr)
        sys.exit(1)

    if startup_warmup_enabled:
        logger.info("Running startup warmup before declaring the service ready")
        _run_startup_warmup(args.warmup_text, args.warmup_max_new_tokens)

    logger.info("Model ready. Sample rate: %d Hz", SAMPLE_RATE)
    logger.info("Server mode: %s", generation_mode)
    logger.info("Server listening on http://%s:%d", args.host, args.port)

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()

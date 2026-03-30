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
        "echo":  {"ref_audio": "voice2.wav", "ref_text": "...", "language": "English"},
        "cached": {"speaker_pt": "speaker.pt", "language": "English"}
    }

API usage:
    curl -s http://localhost:8000/v1/audio/speech \\
        -H "Content-Type: application/json" \\
        -d '{"model": "tts-1", "input": "Hello!", "voice": "alloy", "response_format": "wav", "instruct": "Speak gently with a relaxed cadence."}' \\
        --output speech.wav

    curl -s http://localhost:8000/v1/audio/speech \\
        -F "model=tts-1" \\
        -F "input=Hello from a request-scoped speaker embedding!" \\
        -F "voice=alloy" \\
        -F "response_format=wav" \\
        -F "voice_clone_pt=@speaker.pt;type=application/octet-stream" \\
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
from typing import AsyncGenerator, Optional

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
_voice_clone_pt_cache: dict[str, dict] = {}

def _is_ready() -> bool:
    return tts_model is not None and (not startup_warmup_enabled or startup_warmup_completed)


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------


class SpeechRequest(BaseModel):
    model: str = "tts-1"
    input: str
    voice: str = "alloy"
    response_format: str = "wav"  # wav | pcm | mp3 | opus
    speed: float = 1.0           # accepted but not yet applied
    instruct: Optional[str] = None
    language: Optional[str] = None
    seed: Optional[int] = None


def _serialize_speaker_embedding(speaker_embedding: torch.Tensor) -> bytes:
    buf = io.BytesIO()
    torch.save(speaker_embedding.detach().cpu(), buf)
    return buf.getvalue()


def _is_upload_file(value) -> bool:
    return hasattr(value, "read") and hasattr(value, "filename")


def _load_speaker_embedding_payload(payload: bytes) -> torch.Tensor:
    try:
        loaded = torch.load(io.BytesIO(payload), map_location="cpu", weights_only=True)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=f"Invalid voice clone pt file: {exc}") from exc

    if isinstance(loaded, torch.Tensor):
        speaker_embedding = loaded
    elif isinstance(loaded, dict):
        if "ref_spk_embedding" in loaded:
            ref_spk_embedding = loaded["ref_spk_embedding"]
            if not isinstance(ref_spk_embedding, list) or len(ref_spk_embedding) != 1:
                raise HTTPException(
                    status_code=400,
                    detail="Invalid voice clone pt file: ref_spk_embedding must be a single-item list",
                )
            speaker_embedding = ref_spk_embedding[0]
        elif "speaker_embedding" in loaded:
            speaker_embedding = loaded["speaker_embedding"]
        else:
            raise HTTPException(
                status_code=400,
                detail="Invalid voice clone pt file: expected a tensor or dict with ref_spk_embedding",
            )
    else:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid voice clone pt file: unsupported payload type {type(loaded).__name__}",
        )

    if not isinstance(speaker_embedding, torch.Tensor):
        raise HTTPException(status_code=400, detail="Invalid voice clone pt file: speaker embedding is not a tensor")
    return speaker_embedding


def _voice_clone_prompt_from_pt_bytes(payload: bytes) -> dict:
    if tts_model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    if not hasattr(tts_model, "build_voice_clone_prompt_from_embedding"):
        raise HTTPException(status_code=400, detail="Loaded model does not support voice clone pt files")
    speaker_embedding = _load_speaker_embedding_payload(payload)
    return tts_model.build_voice_clone_prompt_from_embedding(speaker_embedding)


def _load_voice_clone_prompt_from_path(path: str) -> dict:
    cached = _voice_clone_pt_cache.get(path)
    if cached is not None:
        return cached
    prompt = _voice_clone_prompt_from_pt_bytes(Path(path).read_bytes())
    _voice_clone_pt_cache[path] = prompt
    return prompt


def _resolve_clone_voice_prompt(voice_cfg: dict, request_voice_clone_prompt: Optional[dict]) -> Optional[dict]:
    if request_voice_clone_prompt is not None:
        return request_voice_clone_prompt

    speaker_pt = voice_cfg.get("speaker_pt") or voice_cfg.get("voice_clone_pt") or voice_cfg.get("pt")
    if not speaker_pt:
        return None
    return _load_voice_clone_prompt_from_path(speaker_pt)


def _voice_cfg_has_clone_source(voice_cfg: dict) -> bool:
    if voice_cfg.get("ref_audio"):
        return True
    if voice_cfg.get("speaker_pt") or voice_cfg.get("voice_clone_pt") or voice_cfg.get("pt"):
        return True
    return False


def _build_clone_generation_kwargs(
    voice_cfg: dict,
    text: str,
    request_instruct: Optional[str],
    request_language: Optional[str],
    request_seed: Optional[int],
    request_voice_clone_prompt: Optional[dict],
) -> dict:
    voice_clone_prompt = _resolve_clone_voice_prompt(voice_cfg, request_voice_clone_prompt)
    ref_audio = voice_cfg.get("ref_audio")
    if voice_clone_prompt is None and ref_audio is None:
        raise HTTPException(
            status_code=400,
            detail="Clone voice config requires ref_audio or speaker_pt/voice_clone_pt",
        )
    return dict(
        text=text,
        language=_resolve_language(voice_cfg, request_language),
        ref_audio=ref_audio,
        ref_text=voice_cfg.get("ref_text", ""),
        instruct=_resolve_instruct(voice_cfg, request_instruct),
        voice_clone_prompt=voice_clone_prompt,
        seed=request_seed,
    )


async def _parse_speech_http_request(request: Request) -> tuple[SpeechRequest, Optional[dict]]:
    content_type = request.headers.get("content-type", "")
    if content_type.startswith("application/json"):
        try:
            payload = await request.json()
            return SpeechRequest.model_validate(payload), None
        except ValidationError as exc:
            raise HTTPException(status_code=422, detail=exc.errors()) from exc

    if content_type.startswith("multipart/form-data") or content_type.startswith("application/x-www-form-urlencoded"):
        form = await request.form()
        payload = {}
        for field_name in SpeechRequest.model_fields:
            value = form.get(field_name)
            if value is None or _is_upload_file(value):
                continue
            payload[field_name] = value
        try:
            req = SpeechRequest.model_validate(payload)
        except ValidationError as exc:
            raise HTTPException(status_code=422, detail=exc.errors()) from exc

        pt_upload = form.get("voice_clone_pt") or form.get("speaker_pt") or form.get("pt")
        request_voice_clone_prompt = None
        if pt_upload is not None:
            if not _is_upload_file(pt_upload):
                raise HTTPException(status_code=400, detail="voice_clone_pt must be uploaded as a file")
            request_voice_clone_prompt = _voice_clone_prompt_from_pt_bytes(await pt_upload.read())
        return req, request_voice_clone_prompt

    raise HTTPException(
        status_code=415,
        detail="Unsupported Content-Type. Use application/json or multipart/form-data",
    )


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


def _to_opus_bytes(pcm: np.ndarray, sample_rate: int) -> bytes:
    """Convert float32 numpy array to Ogg Opus bytes (requires pydub + ffmpeg/libopus)."""
    try:
        from pydub import AudioSegment
    except ImportError:
        raise HTTPException(
            status_code=400,
            detail="response_format='opus' requires pydub: pip install pydub",
        )
    segment = AudioSegment(
        _to_pcm16(pcm),
        frame_rate=sample_rate,
        sample_width=2,
        channels=1,
    )
    buf = io.BytesIO()
    try:
        segment.export(buf, format="ogg", codec="libopus")
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(
            status_code=500,
            detail="response_format='opus' requires ffmpeg with libopus encoder support",
        ) from exc
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Voice resolution
# ---------------------------------------------------------------------------


def resolve_voice(voice_name: str) -> dict:
    """Return voice config dict or fall back to default, else raise 400."""
    if voice_name in voices:
        return voices[voice_name]
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
    if request_instruct is None:
        return voice_cfg.get("instruct")
    return request_instruct


def _resolve_language(voice_cfg: dict, request_language: Optional[str]) -> str:
    if request_language is None:
        return voice_cfg.get("language", "Auto")
    return request_language


async def _stream_chunks(
    voice_cfg: dict,
    text: str,
    request_instruct: Optional[str],
    request_language: Optional[str],
    request_seed: Optional[int],
    request_voice_clone_prompt: Optional[dict],
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
                    kwargs = _build_clone_generation_kwargs(
                        voice_cfg=voice_cfg,
                        text=text,
                        request_instruct=request_instruct,
                        request_language=request_language,
                        request_seed=request_seed,
                        request_voice_clone_prompt=request_voice_clone_prompt,
                    )
                    generator = tts_model.generate_voice_clone_streaming(
                        chunk_size=voice_cfg.get("chunk_size", 12),
                        non_streaming_mode=False,
                        **kwargs,
                    )
                elif generation_mode == "custom":
                    generator = tts_model.generate_custom_voice_streaming(
                        text=text,
                        speaker=voice_cfg.get("speaker", ""),
                        language=_resolve_language(voice_cfg, request_language),
                        instruct=_resolve_instruct(voice_cfg, request_instruct),
                        chunk_size=voice_cfg.get("chunk_size", 12),
                        seed=request_seed,
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


async def create_speech(req: SpeechRequest, request_voice_clone_prompt: Optional[dict] = None):
    if tts_model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    if not req.input.strip():
        raise HTTPException(status_code=400, detail="'input' text is empty")
    if request_voice_clone_prompt is not None and generation_mode != "clone":
        raise HTTPException(status_code=400, detail="voice_clone_pt is only supported when the server runs in clone mode")

    voice_cfg = resolve_voice(req.voice)
    fmt = req.response_format.lower()

    _CONTENT_TYPES = {
        "wav": "audio/wav",
        "pcm": "audio/pcm",
        "mp3": "audio/mpeg",
        "opus": "audio/ogg",
    }
    if fmt not in _CONTENT_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"response_format {fmt!r} not supported. Use: wav, pcm, mp3, opus",
        )
    content_type = _CONTENT_TYPES[fmt]

    # --- MP3 / Opus: generate all audio, then encode (non-streaming) ---
    if fmt in {"mp3", "opus"}:
        loop = asyncio.get_event_loop()

        def _generate():
            with _model_lock:
                if generation_mode == "clone":
                    kwargs = _build_clone_generation_kwargs(
                        voice_cfg=voice_cfg,
                        text=req.input,
                        request_instruct=req.instruct,
                        request_language=req.language,
                        request_seed=req.seed,
                        request_voice_clone_prompt=request_voice_clone_prompt,
                    )
                    return tts_model.generate_voice_clone(
                        **kwargs,
                    )
                if generation_mode == "custom":
                    return tts_model.generate_custom_voice(
                        text=req.input,
                        speaker=voice_cfg.get("speaker", ""),
                        language=_resolve_language(voice_cfg, req.language),
                        instruct=_resolve_instruct(voice_cfg, req.instruct),
                        seed=req.seed,
                    )
                raise RuntimeError(f"Unsupported generation mode: {generation_mode}")

        audio_arrays, sr = await loop.run_in_executor(None, _generate)
        audio = audio_arrays[0] if audio_arrays else np.zeros(1, dtype=np.float32)
        encoder = _to_mp3_bytes if fmt == "mp3" else _to_opus_bytes
        return Response(content=encoder(audio, sr), media_type=content_type)

    # --- WAV / PCM: stream chunks as they are generated ---
    async def audio_stream():
        if fmt == "wav":
            yield _wav_header(SAMPLE_RATE)  # stream with unknown data length
        async for raw_chunk in _stream_chunks(
            voice_cfg,
            req.input,
            req.instruct,
            req.language,
            req.seed,
            request_voice_clone_prompt,
        ):
            yield raw_chunk

    return StreamingResponse(audio_stream(), media_type=content_type)


@app.post("/v1/audio/speech")
async def create_speech_endpoint(request: Request):
    req, request_voice_clone_prompt = await _parse_speech_http_request(request)
    return await create_speech(req, request_voice_clone_prompt=request_voice_clone_prompt)


@app.post("/v1/audio/voice-clone/pt")
async def create_voice_clone_pt(
    ref_audio: UploadFile = File(...),
    filename: Optional[str] = Form(None),
):
    if tts_model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    if generation_mode != "clone":
        raise HTTPException(status_code=400, detail="voice clone pt extraction requires clone mode")
    if not hasattr(tts_model, "extract_speaker_embedding"):
        raise HTTPException(status_code=400, detail="Loaded model does not support speaker embedding extraction")

    audio_bytes = await ref_audio.read()
    if not audio_bytes:
        raise HTTPException(status_code=400, detail="Uploaded ref_audio file is empty")

    suffix = Path(ref_audio.filename or "reference.wav").suffix or ".wav"
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(prefix="voice-clone-", suffix=suffix, delete=False) as temp:
            temp.write(audio_bytes)
            temp_path = temp.name
        with _model_lock:
            speaker_embedding = tts_model.extract_speaker_embedding(temp_path)
    finally:
        if temp_path and os.path.exists(temp_path):
            os.unlink(temp_path)

    download_name = filename or "speaker.pt"
    headers = {"Content-Disposition": f'attachment; filename="{Path(download_name).name}"'}
    return Response(
        content=_serialize_speaker_embedding(speaker_embedding),
        media_type="application/octet-stream",
        headers=headers,
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
        help="JSON file mapping voice names to {ref_audio, ref_text, language} or {speaker_pt, language}",
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
        startup_warmup_seconds = 0.0
        startup_warmup_completed = True
        return

    voice_name = default_voice or next(iter(voices))
    voice_cfg = resolve_voice(voice_name)
    if generation_mode == "clone" and not _voice_cfg_has_clone_source(voice_cfg):
        logger.warning(
            "Skipping startup warmup because clone voice %s has no static ref_audio or speaker_pt",
            voice_name,
        )
        startup_warmup_seconds = 0.0
        startup_warmup_completed = True
        return
    started = time.perf_counter()

    with _model_lock:
        if generation_mode == "clone":
            kwargs = _build_clone_generation_kwargs(
                voice_cfg=voice_cfg,
                text=warmup_text,
                request_instruct=None,
                request_language=voice_cfg.get("language", "Auto"),
                request_seed=None,
                request_voice_clone_prompt=None,
            )
            tts_model.generate_voice_clone(
                max_new_tokens=warmup_max_new_tokens,
                **kwargs,
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
    SAMPLE_RATE = tts_model.sample_rate
    generation_mode = args.mode

    if generation_mode == "clone":
        if args.voices:
            with open(args.voices) as f:
                voices = json.load(f)
            default_voice = args.default_voice or next(iter(voices))
            logger.info("Loaded %d voice(s) from %s", len(voices), args.voices)
        elif args.ref_audio:
            voices = {
                args.default_voice or "default": {
                    "ref_audio": args.ref_audio,
                    "ref_text": args.ref_text,
                    "language": args.language,
                    "chunk_size": args.chunk_size,
                }
            }
            default_voice = next(iter(voices))
            logger.info("Using single clone voice from --ref-audio: %s", args.ref_audio)
        else:
            voices = {
                args.default_voice or "dynamic": {
                    "language": args.language,
                    "chunk_size": args.chunk_size,
                }
            }
            default_voice = next(iter(voices))
            logger.info(
                "Clone mode started without static ref_audio/voices; expecting request-level voice_clone_pt uploads"
            )
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

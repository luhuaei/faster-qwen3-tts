#!/usr/bin/env python3
from __future__ import annotations

import io
import json
import os
import posixpath
import shlex
import struct
import time
import urllib.request
import wave
from pathlib import Path

import paramiko
import torch
import torch.nn.functional as F


NO_PROXY_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))
HOST = os.environ.get("ORIN_AIPOD_HOST", "lzc-pod-juyIZt.lan")
USER = os.environ.get("ORIN_AIPOD_USER", "nvidia")
PASSWORD = os.environ.get("ORIN_AIPOD_PASSWORD", "nvidia")
REMOTE_ROOT = os.environ.get("ORIN_AIPOD_REMOTE_ROOT", "/home/nvidia/faster-qwen3-tts-aipod")
IMAGE_NAME = os.environ.get(
    "ORIN_AIPOD_IMAGE",
    "lzcbox-90fc188b.lan:5001/x/faster-qwen3-tts:0.6b-base-clone-openai-orin-v1",
)
CONTAINER_NAME = os.environ.get("ORIN_AIPOD_CONTAINER", "faster-qwen3-tts-orin-voice-clone-pt")
PORT = int(os.environ.get("ORIN_AIPOD_PORT", "18001"))
VOICE = os.environ.get("ORIN_AIPOD_VOICE", "dynamic")
SEED = int(os.environ.get("ORIN_AIPOD_SEED", "1234"))
SIMILARITY_THRESHOLD = float(os.environ.get("ORIN_VOICE_PT_SIM_THRESHOLD", "0.75"))
REFERENCE_AUDIO = Path(os.environ.get("ORIN_VOICE_CLONE_REF_AUDIO", "ref_audio.wav"))
TEXTS = [
    text.strip()
    for text in os.environ.get(
        "ORIN_VOICE_CLONE_TEXTS",
        "今天的风从江面慢慢吹过来，像有人把一句话轻轻放在耳边。||"
        "等夜色落下来以后，街角的小店才真正热闹起来，灯光也显得更暖。||"
        "如果你愿意慢一点说话，很多原本匆忙的事情都会忽然变得从容。",
    ).split("||")
    if text.strip()
]


class RemoteHost:
    def __init__(self) -> None:
        self.client = paramiko.SSHClient()
        self.client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        self.client.connect(
            hostname=HOST,
            username=USER,
            password=PASSWORD,
            look_for_keys=True,
            allow_agent=True,
            timeout=30,
        )

    def close(self) -> None:
        self.client.close()

    def run(self, command: str, *, check: bool = True, timeout: int | None = None) -> str:
        stdin, stdout, stderr = self.client.exec_command(command, timeout=timeout)
        channel = stdout.channel
        output_chunks: list[str] = []
        error_chunks: list[str] = []
        started = time.monotonic()
        while True:
            if channel.recv_ready():
                output_chunks.append(channel.recv(65536).decode("utf-8", errors="replace"))
            if channel.recv_stderr_ready():
                error_chunks.append(channel.recv_stderr(65536).decode("utf-8", errors="replace"))
            if channel.exit_status_ready():
                while channel.recv_ready():
                    output_chunks.append(channel.recv(65536).decode("utf-8", errors="replace"))
                while channel.recv_stderr_ready():
                    error_chunks.append(channel.recv_stderr(65536).decode("utf-8", errors="replace"))
                break
            if timeout is not None and (time.monotonic() - started) > timeout:
                channel.close()
                raise TimeoutError(f"remote command timed out after {timeout}s: {command}")
            time.sleep(0.1)
        exit_code = channel.recv_exit_status()
        combined = "".join(output_chunks) + "".join(error_chunks)
        if check and exit_code != 0:
            raise RuntimeError(f"remote command failed ({exit_code}): {command}\n{combined}")
        return combined


def wait_for_json(url: str, timeout_s: int) -> dict:
    deadline = time.time() + timeout_s
    last_error = "not_started"
    while time.time() < deadline:
        try:
            with NO_PROXY_OPENER.open(url, timeout=5) as response:
                return json.loads(response.read().decode("utf-8"))
        except Exception as exc:  # noqa: BLE001
            last_error = repr(exc)
            time.sleep(1)
    raise TimeoutError(f"timeout waiting for {url}: {last_error}")


def encode_multipart(
    *,
    fields: dict[str, str | int | float],
    files: list[tuple[str, str, bytes, str]],
) -> tuple[str, bytes]:
    boundary = f"----faster-qwen3-tts-{int(time.time() * 1000)}"
    chunks: list[bytes] = []
    for name, value in fields.items():
        chunks.append(f"--{boundary}\r\n".encode("utf-8"))
        chunks.append(f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode("utf-8"))
        chunks.append(str(value).encode("utf-8"))
        chunks.append(b"\r\n")
    for field_name, filename, payload, content_type in files:
        chunks.append(f"--{boundary}\r\n".encode("utf-8"))
        chunks.append(
            (
                f'Content-Disposition: form-data; name="{field_name}"; '
                f'filename="{filename}"\r\n'
            ).encode("utf-8")
        )
        chunks.append(f"Content-Type: {content_type}\r\n\r\n".encode("utf-8"))
        chunks.append(payload)
        chunks.append(b"\r\n")
    chunks.append(f"--{boundary}--\r\n".encode("utf-8"))
    return f"multipart/form-data; boundary={boundary}", b"".join(chunks)


def post_multipart(
    url: str,
    *,
    fields: dict[str, str | int | float],
    files: list[tuple[str, str, bytes, str]],
    accept: str,
    timeout: int = 600,
) -> tuple[int, dict[str, str], bytes]:
    content_type, payload = encode_multipart(fields=fields, files=files)
    request = urllib.request.Request(
        url,
        data=payload,
        headers={"Content-Type": content_type, "Accept": accept},
        method="POST",
    )
    with NO_PROXY_OPENER.open(request, timeout=timeout) as response:
        return response.status, dict(response.headers.items()), response.read()


def load_embedding(pt_bytes: bytes) -> torch.Tensor:
    return torch.load(io.BytesIO(pt_bytes), map_location="cpu", weights_only=True)


def cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.float().reshape(1, -1)
    b = b.float().reshape(1, -1)
    return float(F.cosine_similarity(a, b).item())


def wav_duration_seconds(payload: bytes) -> float:
    if len(payload) >= 44 and payload[:4] == b"RIFF" and payload[8:12] == b"WAVE":
        channels = struct.unpack("<H", payload[22:24])[0]
        sample_rate = struct.unpack("<I", payload[24:28])[0]
        bits_per_sample = struct.unpack("<H", payload[34:36])[0]
        data_bytes = max(len(payload) - 44, 0)
        bytes_per_frame = channels * bits_per_sample / 8.0
        if channels > 0 and sample_rate > 0 and bytes_per_frame > 0:
            return float(data_bytes / bytes_per_frame / sample_rate)
    with wave.open(io.BytesIO(payload), "rb") as wf:
        return float(wf.getnframes() / wf.getframerate())


def main() -> None:
    if not REFERENCE_AUDIO.exists():
        raise FileNotFoundError(f"Reference audio not found: {REFERENCE_AUDIO}")
    if not TEXTS:
        raise ValueError("No validation texts configured")

    output_dir = Path("benchmarks/results/orin_voice_clone_pt") / time.strftime("%Y%m%d-%H%M%S")
    output_dir.mkdir(parents=True, exist_ok=True)
    remote = RemoteHost()

    try:
        remote.run(f"mkdir -p {shlex.quote(REMOTE_ROOT)}")
        remote.run(f"docker rm -f {shlex.quote(CONTAINER_NAME)} >/dev/null 2>&1 || true", check=False)
        remote.run(
            " ".join(
                [
                    "docker",
                    "run",
                    "-d",
                    "--rm",
                    "--runtime",
                    "nvidia",
                    "--name",
                    shlex.quote(CONTAINER_NAME),
                    "-p",
                    f"{PORT}:8000",
                    "-e",
                    "HF_HUB_OFFLINE=1",
                    "-e",
                    "TRANSFORMERS_OFFLINE=1",
                    IMAGE_NAME,
                ]
            ),
            timeout=300,
        )

        base_url = f"http://{HOST}:{PORT}"
        health = wait_for_json(base_url + "/health", 900)
        voices = wait_for_json(base_url + "/v1/audio/voices", 60)
        voice_name = VOICE if VOICE in voices.get("voices", []) else voices.get("default_voice", VOICE)

        ref_audio_bytes = REFERENCE_AUDIO.read_bytes()
        source_pt_status_code, source_pt_headers, pt_bytes = post_multipart(
            base_url + "/v1/audio/voice-clone/pt",
            fields={"filename": "speaker.pt"},
            files=[("ref_audio", REFERENCE_AUDIO.name, ref_audio_bytes, "audio/wav")],
            accept="application/octet-stream",
        )
        (output_dir / "speaker.pt").write_bytes(pt_bytes)
        source_embedding = load_embedding(pt_bytes)

        synth_results = []
        generated_embeddings: list[torch.Tensor] = []
        for index, text in enumerate(TEXTS, start=1):
            status_code, headers, audio_bytes = post_multipart(
                base_url + "/v1/audio/speech",
                fields={
                    "model": "tts-1",
                    "input": text,
                    "voice": voice_name,
                    "response_format": "wav",
                    "seed": SEED,
                },
                files=[("voice_clone_pt", "speaker.pt", pt_bytes, "application/octet-stream")],
                accept="audio/wav",
            )
            wav_path = output_dir / f"sample_{index}.wav"
            wav_path.write_bytes(audio_bytes)
            synth_status, synth_pt_headers, synth_pt_bytes = post_multipart(
                base_url + "/v1/audio/voice-clone/pt",
                fields={"filename": f"sample_{index}.pt"},
                files=[("ref_audio", wav_path.name, audio_bytes, "audio/wav")],
                accept="application/octet-stream",
            )
            pt_path = output_dir / f"sample_{index}.pt"
            pt_path.write_bytes(synth_pt_bytes)
            embedding = load_embedding(synth_pt_bytes)
            generated_embeddings.append(embedding)
            synth_results.append(
                {
                    "index": index,
                    "text": text,
                    "speech_status_code": status_code,
                    "speech_content_type": headers.get("Content-Type", ""),
                    "audio_bytes": len(audio_bytes),
                    "audio_seconds": wav_duration_seconds(audio_bytes),
                    "embedding_status_code": synth_status,
                    "embedding_content_type": synth_pt_headers.get("Content-Type", ""),
                    "ref_similarity": cosine_similarity(source_embedding, embedding),
                }
            )

        pairwise = []
        for left in range(len(generated_embeddings)):
            for right in range(left + 1, len(generated_embeddings)):
                pairwise.append(
                    {
                        "left": left + 1,
                        "right": right + 1,
                        "similarity": cosine_similarity(generated_embeddings[left], generated_embeddings[right]),
                    }
                )

        all_similarities = [item["ref_similarity"] for item in synth_results] + [item["similarity"] for item in pairwise]
        result = {
            "health": health,
            "voices": voices,
            "voice": voice_name,
            "seed": SEED,
            "similarity_threshold": SIMILARITY_THRESHOLD,
            "source_pt_status_code": source_pt_status_code,
            "source_pt_content_type": source_pt_headers.get("Content-Type", ""),
            "reference_audio": str(REFERENCE_AUDIO),
            "samples": synth_results,
            "pairwise_similarities": pairwise,
            "min_similarity": min(all_similarities) if all_similarities else None,
            "passed": bool(all_similarities) and all(value >= SIMILARITY_THRESHOLD for value in all_similarities),
        }

        (output_dir / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(result, ensure_ascii=False, indent=2))

        if not result["passed"]:
            raise SystemExit(
                f"Voice consistency check failed: min_similarity={result['min_similarity']}, "
                f"threshold={SIMILARITY_THRESHOLD}"
            )
    finally:
        logs = remote.run(f"docker logs {shlex.quote(CONTAINER_NAME)}", check=False, timeout=60)
        (output_dir / "container.log").write_text(logs, encoding="utf-8")
        remote.run(f"docker rm -f {shlex.quote(CONTAINER_NAME)} >/dev/null 2>&1 || true", check=False)
        remote.close()


if __name__ == "__main__":
    main()

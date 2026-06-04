#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///
from __future__ import annotations

import argparse
import io
import json
import struct
import subprocess
import time
import urllib.error
import urllib.parse
import urllib.request
import wave
from pathlib import Path
from typing import Sequence


DEFAULT_TARGET_SSH = {
    "orin": "nvidia@lzc-pod-juyIZt.lan",
    "thor": "nvidia@tegra-ubuntu-t5000.lan",
    "t4000": "nvidia@tegra-ubuntu-t4000.lan",
}
DEFAULT_TEXT = "The evening air is calm, and every word should arrive clearly."
DEFAULT_OUTPUT_ROOT = Path("benchmarks/results/thor_openai_smoke")
RESPONSE_CONTENT_TYPES = {
    "wav": "audio/wav",
    "pcm": "audio/pcm",
    "mp3": "audio/mpeg",
    "opus": "audio/ogg",
}
NO_PROXY_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Smoke test the Jetson OpenAI-compatible TTS server through Docker's ssh:// transport.",
    )
    parser.add_argument("--target", choices=sorted(DEFAULT_TARGET_SSH), default="thor")
    parser.add_argument("--ssh", help="SSH target like nvidia@host. Uses the selected target default when omitted.")
    parser.add_argument("--docker-host", help="Full Docker host URL, for example ssh://nvidia@host.")
    parser.add_argument("--image", required=True, help="Image reference on the remote Docker daemon.")
    parser.add_argument("--port", type=int, default=18000, help="Remote host port to publish to container port 8000.")
    parser.add_argument("--container-name", default="faster-qwen3-tts-jetson-openai-smoke")
    parser.add_argument("--ref-audio", default="ref_audio.wav", help="Reference WAV used for voice_clone_pt extraction.")
    parser.add_argument("--voice", default="vivian")
    parser.add_argument("--text", default=DEFAULT_TEXT)
    parser.add_argument("--response-format", default="wav", choices=sorted(RESPONSE_CONTENT_TYPES))
    parser.add_argument("--startup-timeout", type=int, default=900)
    parser.add_argument("--request-timeout", type=int, default=600)
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    return parser.parse_args(argv)


def docker_host_from_args(args: argparse.Namespace) -> str:
    if args.docker_host:
        return args.docker_host
    ssh_target = args.ssh or DEFAULT_TARGET_SSH[args.target]
    if ssh_target.startswith("ssh://"):
        return ssh_target
    return f"ssh://{ssh_target}"


def hostname_from_ssh_target(ssh_target: str) -> str:
    target = ssh_target
    if target.startswith("ssh://"):
        parsed = urllib.parse.urlparse(target)
        if not parsed.hostname:
            raise ValueError(f"Could not parse hostname from {ssh_target!r}")
        return parsed.hostname
    return target.rsplit("@", 1)[-1].split(":", 1)[0]


def docker_command(docker_host: str, *args: str) -> list[str]:
    return ["docker", "--host", docker_host, *args]


def run_command(command: Sequence[str], *, check: bool = True, timeout: int | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        list(command),
        check=check,
        timeout=timeout,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )


def wait_for_json(url: str, timeout_s: int) -> dict:
    deadline = time.time() + timeout_s
    last_error = "not_started"
    while time.time() < deadline:
        try:
            with NO_PROXY_OPENER.open(url, timeout=5) as response:
                payload = json.loads(response.read().decode("utf-8"))
            if payload.get("ready", True):
                return payload
            last_error = json.dumps(payload, ensure_ascii=False)
        except Exception as exc:  # noqa: BLE001
            last_error = repr(exc)
        time.sleep(1)
    raise TimeoutError(f"timeout waiting for {url}: {last_error}")


def request_json(url: str, *, timeout: int) -> dict:
    with NO_PROXY_OPENER.open(url, timeout=timeout) as response:
        payload = response.read()
    return json.loads(payload.decode("utf-8"))


def post_json(url: str, payload: dict, *, accept: str, timeout: int) -> tuple[int, dict[str, str], bytes]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", "Accept": accept},
        method="POST",
    )
    try:
        with NO_PROXY_OPENER.open(request, timeout=timeout) as response:
            return response.status, dict(response.headers.items()), response.read()
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"POST {url} failed with HTTP {exc.code}: {body}") from exc


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
    timeout: int,
) -> tuple[int, dict[str, str], bytes]:
    content_type, payload = encode_multipart(fields=fields, files=files)
    request = urllib.request.Request(
        url,
        data=payload,
        headers={"Content-Type": content_type, "Accept": accept},
        method="POST",
    )
    try:
        with NO_PROXY_OPENER.open(request, timeout=timeout) as response:
            return response.status, dict(response.headers.items()), response.read()
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"POST {url} failed with HTTP {exc.code}: {body}") from exc


def content_type_matches(headers: dict[str, str], expected: str) -> bool:
    actual = header_value(headers, "Content-Type").split(";", 1)[0].strip().lower()
    return actual == expected


def header_value(headers: dict[str, str], name: str) -> str:
    expected = name.lower()
    for key, value in headers.items():
        if key.lower() == expected:
            return value
    return ""


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


def validate_audio(
    *,
    label: str,
    status_code: int,
    headers: dict[str, str],
    payload: bytes,
    response_format: str,
) -> dict:
    expected_type = RESPONSE_CONTENT_TYPES[response_format]
    if status_code != 200:
        raise RuntimeError(f"{label} returned HTTP {status_code}")
    if not content_type_matches(headers, expected_type):
        raise RuntimeError(f"{label} returned Content-Type {header_value(headers, 'Content-Type')!r}, expected {expected_type}")
    if len(payload) < 512:
        raise RuntimeError(f"{label} returned too few bytes: {len(payload)}")

    result = {
        "status_code": status_code,
        "content_type": header_value(headers, "Content-Type"),
        "audio_bytes": len(payload),
    }
    if response_format == "wav":
        if not (payload[:4] == b"RIFF" and payload[8:12] == b"WAVE"):
            raise RuntimeError(f"{label} did not return a WAV payload")
        duration = wav_duration_seconds(payload)
        if duration <= 0:
            raise RuntimeError(f"{label} returned invalid WAV duration: {duration}")
        result["audio_seconds"] = duration
    return result


def select_voice(voices_response: dict, requested_voice: str) -> str:
    available = voices_response.get("voices", [])
    if requested_voice in available:
        return requested_voice
    default_voice = voices_response.get("default_voice")
    if default_voice in available:
        return default_voice
    if available:
        return available[0]
    raise RuntimeError(f"No voices advertised by server: {voices_response}")


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    ref_audio = Path(args.ref_audio).expanduser().resolve()
    if not ref_audio.is_file():
        raise FileNotFoundError(f"Reference audio not found: {ref_audio}")

    docker_host = docker_host_from_args(args)
    host = hostname_from_ssh_target(docker_host)
    output_dir = Path(args.output_root) / time.strftime("%Y%m%d-%H%M%S")
    output_dir.mkdir(parents=True, exist_ok=True)
    base_url = f"http://{host}:{args.port}"
    log_path = output_dir / "container.log"

    run_command(docker_command(docker_host, "rm", "-f", args.container_name), check=False, timeout=120)
    try:
        run_command(
            docker_command(
                docker_host,
                "run",
                "-d",
                "--rm",
                "--runtime",
                "nvidia",
                "--name",
                args.container_name,
                "-p",
                f"{args.port}:8000",
                "-e",
                "HF_HUB_OFFLINE=1",
                "-e",
                "TRANSFORMERS_OFFLINE=1",
                args.image,
            ),
            timeout=300,
        )

        started = time.time()
        health = wait_for_json(base_url + "/health", args.startup_timeout)
        voices = request_json(base_url + "/v1/audio/voices", timeout=60)
        voice = select_voice(voices, args.voice)
        expected_type = RESPONSE_CONTENT_TYPES[args.response_format]

        speech_started = time.time()
        speech_status, speech_headers, speech_audio = post_json(
            base_url + "/v1/audio/speech",
            {
                "model": "tts-1",
                "input": args.text,
                "voice": voice,
                "response_format": args.response_format,
            },
            accept=expected_type,
            timeout=args.request_timeout,
        )
        speech_result = validate_audio(
            label="json speech",
            status_code=speech_status,
            headers=speech_headers,
            payload=speech_audio,
            response_format=args.response_format,
        )
        speech_result["request_seconds"] = time.time() - speech_started
        speech_output = output_dir / f"speech.{args.response_format}"
        speech_output.write_bytes(speech_audio)

        ref_audio_bytes = ref_audio.read_bytes()
        clone_status, clone_headers, pt_bytes = post_multipart(
            base_url + "/v1/audio/voice-clone/pt",
            fields={"filename": "speaker.pt"},
            files=[("ref_audio", ref_audio.name, ref_audio_bytes, "audio/wav")],
            accept="application/octet-stream",
            timeout=args.request_timeout,
        )
        if clone_status != 200:
            raise RuntimeError(f"voice_clone_pt extraction returned HTTP {clone_status}")
        if not content_type_matches(clone_headers, "application/octet-stream"):
            raise RuntimeError(
                "voice_clone_pt extraction returned Content-Type "
                f"{header_value(clone_headers, 'Content-Type')!r}, expected application/octet-stream"
            )
        if len(pt_bytes) < 512:
            raise RuntimeError(f"voice_clone_pt extraction returned too few bytes: {len(pt_bytes)}")
        (output_dir / "speaker.pt").write_bytes(pt_bytes)

        clone_speech_started = time.time()
        clone_speech_status, clone_speech_headers, clone_speech_audio = post_multipart(
            base_url + "/v1/audio/speech",
            fields={
                "model": "tts-1",
                "input": args.text,
                "voice": voice,
                "response_format": args.response_format,
            },
            files=[("voice_clone_pt", "speaker.pt", pt_bytes, "application/octet-stream")],
            accept=expected_type,
            timeout=args.request_timeout,
        )
        clone_speech_result = validate_audio(
            label="voice_clone_pt speech",
            status_code=clone_speech_status,
            headers=clone_speech_headers,
            payload=clone_speech_audio,
            response_format=args.response_format,
        )
        clone_speech_result["request_seconds"] = time.time() - clone_speech_started
        clone_speech_output = output_dir / f"speech_voice_clone_pt.{args.response_format}"
        clone_speech_output.write_bytes(clone_speech_audio)

        result = {
            "passed": True,
            "target": args.target,
            "docker_host": docker_host,
            "host": host,
            "image": args.image,
            "container_name": args.container_name,
            "base_url": base_url,
            "startup_seconds": time.time() - started,
            "health": health,
            "voices": voices,
            "voice": voice,
            "response_format": args.response_format,
            "speech": {**speech_result, "path": str(speech_output)},
            "voice_clone_pt": {
                "status_code": clone_status,
                "content_type": header_value(clone_headers, "Content-Type"),
                "bytes": len(pt_bytes),
                "path": str(output_dir / "speaker.pt"),
            },
            "voice_clone_pt_speech": {**clone_speech_result, "path": str(clone_speech_output)},
        }
        (output_dir / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    finally:
        logs = run_command(docker_command(docker_host, "logs", args.container_name), check=False, timeout=120)
        log_path.write_text(logs.stdout, encoding="utf-8")
        run_command(docker_command(docker_host, "rm", "-f", args.container_name), check=False, timeout=120)


if __name__ == "__main__":
    raise SystemExit(main())

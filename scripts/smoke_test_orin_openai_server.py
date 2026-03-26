#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import posixpath
import shlex
import time
import urllib.request
import hashlib
from pathlib import Path

import paramiko


LOCAL_TELEMETRY_SCRIPT = Path.home() / "lzc-aipod-pkgs" / "scripts" / "collect_jetson_metrics.py"
NO_PROXY_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))
HOST = "lzc-pod-juyIZt.lan"
USER = "nvidia"
PASSWORD = "nvidia"
REMOTE_ROOT = "/home/nvidia/faster-qwen3-tts-aipod"
IMAGE_NAME = os.environ.get(
    "ORIN_AIPOD_IMAGE",
    "lzcbox-90fc188b.lan:5001/x/faster-qwen3-tts:0.6b-custom-openai-orin-v4",
)
CONTAINER_NAME = "faster-qwen3-tts-aipod-smoke"
PORT = int(os.environ.get("ORIN_AIPOD_PORT", "18000"))
TEXT = os.environ.get("ORIN_AIPOD_TEXT", "今天的风比昨天轻一点，适合慢慢说话。")
TEXT_ALT = os.environ.get("ORIN_AIPOD_TEXT_ALT", "请把电梯口的蓝色箱子搬到三号会议室，动作快一点。")
VOICE = os.environ.get("ORIN_AIPOD_VOICE", "vivian")
VOICE_FILTER = [item.strip() for item in os.environ.get("ORIN_AIPOD_TEST_VOICES", "").split(",") if item.strip()]
REPEAT_REQUESTS = int(os.environ.get("ORIN_AIPOD_REPEAT_REQUESTS", "3"))
INSTRUCT_A = os.environ.get("ORIN_AIPOD_INSTRUCT_A", "请用温和、放松、娓娓道来的语气朗读。")
INSTRUCT_B = os.environ.get("ORIN_AIPOD_INSTRUCT_B", "请用干脆、明亮、带一点兴奋感的语气朗读。")
REPETITION_PENALTY = os.environ.get("ORIN_AIPOD_REPETITION_PENALTY", "").strip()


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
        self.sftp = self.client.open_sftp()

    def close(self) -> None:
        try:
            self.sftp.close()
        finally:
            self.client.close()

    def upload(self, local_path: Path, remote_path: str) -> None:
        self.run(f"mkdir -p {shlex.quote(posixpath.dirname(remote_path))}")
        self.sftp.put(str(local_path), remote_path)

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


def sanitize_filename(value: str) -> str:
    safe = []
    for ch in value:
        if ch.isalnum() or ch in {"-", "_"}:
            safe.append(ch)
        else:
            safe.append("_")
    return "".join(safe).strip("_") or "item"


def build_payload(*, voice: str, text: str, instruct: str | None = None) -> dict:
    payload = {
        "model": "tts-1",
        "input": text,
        "voice": voice,
        "response_format": "wav",
        "temperature": 1.0,
        "do_sample": False,
    }
    if instruct is not None:
        payload["instruct"] = instruct
    if REPETITION_PENALTY:
        payload["repetition_penalty"] = float(REPETITION_PENALTY)
    return payload


def fetch_audio(base_url: str, payload: dict) -> dict:
    payload = json.dumps(
        payload,
        ensure_ascii=False,
    ).encode("utf-8")
    request = urllib.request.Request(
        base_url + "/v1/audio/speech",
        data=payload,
        headers={"Content-Type": "application/json", "Accept": "audio/wav"},
        method="POST",
    )
    request_started = time.time()
    with NO_PROXY_OPENER.open(request, timeout=600) as response:
        status_code = response.status
        first_chunk = response.read(4096)
        first_chunk_seconds = time.time() - request_started
        chunks = [first_chunk]
        while True:
            chunk = response.read(65536)
            if not chunk:
                break
            chunks.append(chunk)
        audio = b"".join(chunks)

    return {
        "status_code": status_code,
        "first_chunk_seconds": first_chunk_seconds,
        "request_seconds": time.time() - request_started,
        "audio": audio,
        "audio_bytes": len(audio),
        "audio_sha256": hashlib.sha256(audio).hexdigest(),
    }


def write_audio(output_dir: Path, filename: str, audio: bytes) -> None:
    (output_dir / filename).write_bytes(audio)


def run_repeated_requests(
    *,
    base_url: str,
    voice: str,
    text: str,
    repeat_requests: int,
    output_dir: Path,
) -> dict:
    payload = build_payload(voice=voice, text=text)
    runs = []
    for index in range(repeat_requests):
        run = fetch_audio(base_url, payload)
        run["request_index"] = index
        runs.append(run)
        write_audio(output_dir, f"stable-{sanitize_filename(voice)}-{index}.wav", run["audio"])
    hashes = [run["audio_sha256"] for run in runs]
    consistent = len(set(hashes)) == 1
    if not consistent:
        raise RuntimeError(f"Inconsistent audio detected for voice={voice!r}: {hashes}")
    return {
        "voice": voice,
        "text": text,
        "repeat_requests": repeat_requests,
        "consistent_audio": consistent,
        "audio_sha256": hashes[0],
        "runs": [
            {
                "request_index": run["request_index"],
                "status_code": run["status_code"],
                "first_chunk_seconds": run["first_chunk_seconds"],
                "request_seconds": run["request_seconds"],
                "audio_bytes": run["audio_bytes"],
                "audio_sha256": run["audio_sha256"],
            }
            for run in runs
        ],
    }


def run_single_request(
    *,
    base_url: str,
    voice: str,
    text: str,
    output_dir: Path,
    instruct: str | None = None,
    filename_prefix: str,
) -> dict:
    payload = build_payload(voice=voice, text=text, instruct=instruct)
    run = fetch_audio(base_url, payload)
    write_audio(output_dir, f"{filename_prefix}.wav", run["audio"])
    return {
        "voice": voice,
        "text": text,
        "instruct": instruct,
        "status_code": run["status_code"],
        "first_chunk_seconds": run["first_chunk_seconds"],
        "request_seconds": run["request_seconds"],
        "audio_bytes": run["audio_bytes"],
        "audio_sha256": run["audio_sha256"],
    }


def main() -> None:
    remote = RemoteHost()
    output_dir = Path("benchmarks/results/orin_aipod_smoke") / time.strftime("%Y%m%d-%H%M%S")
    output_dir.mkdir(parents=True, exist_ok=True)
    telemetry_remote = posixpath.join(REMOTE_ROOT, "runs", f"{CONTAINER_NAME}.jsonl")
    telemetry_script_remote = posixpath.join(REMOTE_ROOT, "scripts", "collect_jetson_metrics.py")
    telemetry_pid = ""

    try:
        remote.upload(LOCAL_TELEMETRY_SCRIPT, telemetry_script_remote)
        remote.run(f"mkdir -p {shlex.quote(posixpath.dirname(telemetry_remote))}")
        remote.run(f"docker rm -f {shlex.quote(CONTAINER_NAME)} >/dev/null 2>&1 || true", check=False)
        telemetry_pid = remote.run(
            f"nohup python3 {shlex.quote(telemetry_script_remote)} --output {shlex.quote(telemetry_remote)} "
            f"--interval 1.0 --label smoke >/dev/null 2>&1 & echo $!",
            timeout=30,
        ).strip().splitlines()[-1]
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
        started = time.time()
        health = wait_for_json(base_url + "/health", 900)
        voices = wait_for_json(base_url + "/v1/audio/voices", 60)
        print(json.dumps({"health": health, "voices": voices}, ensure_ascii=False, indent=2), flush=True)

        available_voices = [str(item) for item in voices.get("voices", [])]
        if not available_voices:
            raise RuntimeError("No voices returned by /v1/audio/voices")
        if VOICE_FILTER:
            requested = {item.casefold(): item for item in VOICE_FILTER}
            selected_voices = [voice_name for voice_name in available_voices if voice_name.casefold() in requested]
            missing = [item for item in VOICE_FILTER if item.casefold() not in {voice.casefold() for voice in selected_voices}]
            if missing:
                raise RuntimeError(f"Requested voices not available: {missing}; available={available_voices}")
        else:
            selected_voices = available_voices

        if VOICE and VOICE.casefold() in {item.casefold() for item in selected_voices}:
            primary_voice = next(item for item in selected_voices if item.casefold() == VOICE.casefold())
        else:
            primary_voice = voices.get("default_voice") or selected_voices[0]

        stability_results = []
        for voice_name in selected_voices:
            stability_results.append(
                run_repeated_requests(
                    base_url=base_url,
                    voice=voice_name,
                    text=TEXT,
                    repeat_requests=REPEAT_REQUESTS,
                    output_dir=output_dir,
                )
            )

        cross_voice_hashes = {item["voice"]: item["audio_sha256"] for item in stability_results}
        duplicate_voice_hashes: dict[str, list[str]] = {}
        for voice_name, audio_hash in cross_voice_hashes.items():
            duplicate_voice_hashes.setdefault(audio_hash, []).append(voice_name)
        duplicate_voice_hashes = {
            audio_hash: names for audio_hash, names in duplicate_voice_hashes.items() if len(names) > 1
        }
        if duplicate_voice_hashes:
            raise RuntimeError(
                "Different voices produced identical audio for the same text: "
                + json.dumps(duplicate_voice_hashes, ensure_ascii=False)
            )

        text_variation_results = []
        for voice_name in selected_voices:
            alt_result = run_single_request(
                base_url=base_url,
                voice=voice_name,
                text=TEXT_ALT,
                output_dir=output_dir,
                filename_prefix=f"text-alt-{sanitize_filename(voice_name)}",
            )
            baseline_hash = cross_voice_hashes[voice_name]
            if alt_result["audio_sha256"] == baseline_hash:
                raise RuntimeError(
                    f"Different text produced identical audio for voice={voice_name!r}: "
                    f"{baseline_hash}"
                )
            text_variation_results.append(alt_result)

        instruct_none = run_single_request(
            base_url=base_url,
            voice=primary_voice,
            text=TEXT,
            output_dir=output_dir,
            instruct=None,
            filename_prefix=f"instruct-none-{sanitize_filename(primary_voice)}",
        )
        instruct_a = run_single_request(
            base_url=base_url,
            voice=primary_voice,
            text=TEXT,
            output_dir=output_dir,
            instruct=INSTRUCT_A,
            filename_prefix=f"instruct-a-{sanitize_filename(primary_voice)}",
        )
        instruct_b = run_single_request(
            base_url=base_url,
            voice=primary_voice,
            text=TEXT,
            output_dir=output_dir,
            instruct=INSTRUCT_B,
            filename_prefix=f"instruct-b-{sanitize_filename(primary_voice)}",
        )

        instruct_hashes = {
            "none": instruct_none["audio_sha256"],
            "instruct_a": instruct_a["audio_sha256"],
            "instruct_b": instruct_b["audio_sha256"],
        }
        if len(set(instruct_hashes.values())) != len(instruct_hashes):
            raise RuntimeError(
                "Different instruct values produced identical audio: "
                + json.dumps(instruct_hashes, ensure_ascii=False)
            )

        result = {
            "startup_seconds": time.time() - started,
            "health": health,
            "voices": voices,
            "selected_voices": selected_voices,
            "primary_voice": primary_voice,
            "repeat_requests": REPEAT_REQUESTS,
            "stability": stability_results,
            "cross_voice_hashes": cross_voice_hashes,
            "text_variation": {
                "baseline_text": TEXT,
                "alternate_text": TEXT_ALT,
                "per_voice": text_variation_results,
            },
            "instruct_variation": {
                "voice": primary_voice,
                "text": TEXT,
                "instruct_a": INSTRUCT_A,
                "instruct_b": INSTRUCT_B,
                "results": {
                    "none": instruct_none,
                    "instruct_a": instruct_a,
                    "instruct_b": instruct_b,
                },
            },
        }
        (output_dir / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(result, ensure_ascii=False, indent=2))
    finally:
        if telemetry_pid:
            remote.run(f"kill {shlex.quote(telemetry_pid)} >/dev/null 2>&1 || true", check=False)
        logs = remote.run(f"docker logs {shlex.quote(CONTAINER_NAME)}", check=False, timeout=60)
        (output_dir / "container.log").write_text(logs, encoding="utf-8")
        if remote.run(f"test -f {shlex.quote(telemetry_remote)} && echo yes || echo no", check=False).strip().endswith("yes"):
            local_telemetry = output_dir / "telemetry.jsonl"
            remote.sftp.get(telemetry_remote, str(local_telemetry))
        remote.run(f"docker rm -f {shlex.quote(CONTAINER_NAME)} >/dev/null 2>&1 || true", check=False)
        remote.close()


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import posixpath
import shlex
import time
import urllib.request
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
    "lzcbox-90fc188b.lan:5001/x/faster-qwen3-tts:0.6b-custom-openai-orin-v1",
)
CONTAINER_NAME = "faster-qwen3-tts-aipod-smoke"
PORT = int(os.environ.get("ORIN_AIPOD_PORT", "18000"))
TEXT = os.environ.get("ORIN_AIPOD_TEXT", "今天的风比昨天轻一点，适合慢慢说话。")
VOICE = os.environ.get("ORIN_AIPOD_VOICE", "vivian")


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

        payload = json.dumps(
            {
                "model": "tts-1",
                "input": TEXT,
                "voice": VOICE,
                "response_format": "wav",
            },
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

        result = {
            "startup_seconds": time.time() - started,
            "health": health,
            "voices": voices,
            "status_code": status_code,
            "first_chunk_seconds": first_chunk_seconds,
            "request_seconds": time.time() - request_started,
            "audio_bytes": len(audio),
        }
        (output_dir / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        (output_dir / "output.wav").write_bytes(audio)
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

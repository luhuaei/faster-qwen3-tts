#!/usr/bin/env python3
from __future__ import annotations

import os
import posixpath
import shlex
import tarfile
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

import paramiko


REPO_ROOT = Path(__file__).resolve().parents[1]
REMOTE_ROOT = "/home/nvidia/faster-qwen3-tts-aipod"
DEFAULT_IMAGE = "lzcbox-90fc188b.lan:5001/x/faster-qwen3-tts:0.6b-custom-openai-orin-v4"
MODEL_NAME = "Qwen3-TTS-12Hz-0.6B-CustomVoice"


@dataclass(frozen=True)
class RemoteTarget:
    host: str = "lzc-pod-juyIZt.lan"
    user: str = "nvidia"
    password: str = "nvidia"
    remote_root: str = REMOTE_ROOT
    base_image: str = "127.0.0.1:5001/x/lzc-aipod-vllm:bffa39b-orin"
    pip_index_url: str = "https://pypi.jetson-ai-lab.io/jp6/cu126/+simple"
    pip_extra_index_url: str = "https://pypi.org/simple"
    torchaudio_spec: str = "torchaudio==2.10.0"
    model_candidates: tuple[str, ...] = (
        f"/home/nvidia/faster-qwen3-tts-smoke/workspace/models/{MODEL_NAME}",
        f"/home/nvidia/faster-qwen3-tts-smoke/models/{MODEL_NAME}",
    )


class RemoteHost:
    def __init__(self, target: RemoteTarget):
        self.target = target
        self.client = paramiko.SSHClient()
        self.client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        self.client.connect(
            hostname=target.host,
            username=target.user,
            password=target.password,
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

    def run(self, command: str, timeout: int | None = None) -> str:
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
        if exit_code != 0:
            raise RuntimeError(f"remote command failed ({exit_code}): {command}\n{combined}")
        return combined

    def upload(self, local_path: Path, remote_path: str) -> None:
        self.run(f"mkdir -p {shlex.quote(posixpath.dirname(remote_path))}")
        self.sftp.put(str(local_path), remote_path)

    def exists(self, remote_path: str) -> bool:
        try:
            self.sftp.stat(remote_path)
            return True
        except FileNotFoundError:
            return False


def shell_join(parts: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in parts)


def build_context_tarball() -> Path:
    handle = tempfile.NamedTemporaryFile(prefix="faster-qwen3-tts-orin-", suffix=".tar", delete=False)
    handle.close()
    tar_path = Path(handle.name)
    with tarfile.open(tar_path, "w") as tar:
        for relative_path in [
            ".dockerignore",
            "faster-qwen3-tts-jetson.Dockerfile",
            "faster-qwen3-tts-requirements.txt",
            "MANIFEST.in",
            "README.md",
            "pyproject.toml",
            "examples/openai_server.py",
            "faster_qwen3_tts",
        ]:
            tar.add(REPO_ROOT / relative_path, arcname=relative_path)
    return tar_path


def main() -> None:
    target = RemoteTarget()
    image_name = os.environ.get("ORIN_AIPOD_IMAGE", DEFAULT_IMAGE).strip() or DEFAULT_IMAGE
    push_enabled = os.environ.get("ORIN_AIPOD_PUSH", "1").strip().lower() not in {"0", "false", "no"}
    model_source = os.environ.get("REMOTE_MODEL_SOURCE", "").strip()
    remote = RemoteHost(target)
    tar_path = build_context_tarball()
    remote_tar = posixpath.join(target.remote_root, "build", f"context-{int(time.time())}.tar")
    context_root = posixpath.join(target.remote_root, "build", "context")

    try:
        if not model_source:
            for candidate in target.model_candidates:
                if remote.exists(posixpath.join(candidate, "config.json")):
                    model_source = candidate
                    break
        if not model_source:
            raise FileNotFoundError(
                "Could not find a remote model directory for "
                f"{MODEL_NAME}. Set REMOTE_MODEL_SOURCE to an existing directory."
            )

        remote.upload(tar_path, remote_tar)
        remote.run(
            f"rm -rf {shlex.quote(context_root)} && mkdir -p {shlex.quote(context_root)} && "
            f"tar -xf {shlex.quote(remote_tar)} -C {shlex.quote(context_root)}",
            timeout=300,
        )
        remote.run(
            f"mkdir -p {shlex.quote(posixpath.join(context_root, 'models'))} && "
            f"cp -a {shlex.quote(model_source)} {shlex.quote(posixpath.join(context_root, 'models', MODEL_NAME))}",
            timeout=3600,
        )

        build_args = [
            "docker",
            "build",
            "--build-arg",
            f"BASE_IMAGE={target.base_image}",
            "--build-arg",
            f"PIP_INDEX_URL={target.pip_index_url}",
            "--build-arg",
            f"PIP_EXTRA_INDEX_URL={target.pip_extra_index_url}",
            "--build-arg",
            f"TORCHAUDIO_SPEC={target.torchaudio_spec}",
            "-f",
            "faster-qwen3-tts-jetson.Dockerfile",
            "-t",
            image_name,
            ".",
        ]

        proxy = os.environ.get("REMOTE_BUILD_PROXY", "").strip()
        if proxy:
            for key in ["HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"]:
                build_args.extend(["--build-arg", f"{key}={proxy}"])
        no_proxy = os.environ.get("REMOTE_BUILD_NO_PROXY", "").strip()
        if no_proxy:
            for key in ["NO_PROXY", "no_proxy"]:
                build_args.extend(["--build-arg", f"{key}={no_proxy}"])

        command = (
            f"cd {shlex.quote(context_root)} && "
            f"export DOCKER_BUILDKIT=1 && "
            f"{shell_join(build_args)}"
        )
        print(remote.run(command, timeout=14400))
        print(f"Built image on Orin: {image_name}")
        if push_enabled:
            print(remote.run(f"docker push {shlex.quote(image_name)}", timeout=14400))
            print(f"Pushed image: {image_name}")
        else:
            print("Skipped docker push because ORIN_AIPOD_PUSH=0")
    finally:
        remote.close()
        tar_path.unlink(missing_ok=True)


if __name__ == "__main__":
    main()

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
DEFAULT_IMAGE = "lzcbox-90fc188b.lan:5001/x/faster-qwen3-tts:0.6b-base-clone-openai-orin-v5"
DEFAULT_MODEL_NAME = "Qwen3-TTS-12Hz-0.6B-Base"
DEFAULT_MODEL_DOWNLOADER_IMAGE = "lzcbox-90fc188b.lan:5001/x/faster-qwen3-tts:0.6b-custom-openai-orin-v1"
DEFAULT_PREBUILT_APP_IMAGE = "127.0.0.1:5001/x/faster-qwen3-tts:0.6b-custom-openai-orin-v1"


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
    model_search_roots: tuple[str, ...] = (
        "/home/nvidia/faster-qwen3-tts-smoke/workspace/models",
        "/home/nvidia/faster-qwen3-tts-smoke/models",
        "/home/nvidia/faster-qwen3-tts-aipod/build/context/models",
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

    def run(self, command: str, timeout: int | None = None, *, stream: bool = False) -> str:
        stdin, stdout, stderr = self.client.exec_command(command, timeout=timeout)
        channel = stdout.channel
        output_chunks: list[str] = []
        error_chunks: list[str] = []
        started = time.monotonic()
        while True:
            if channel.recv_ready():
                chunk = channel.recv(65536).decode("utf-8", errors="replace")
                output_chunks.append(chunk)
                if stream:
                    print(chunk, end="", flush=True)
            if channel.recv_stderr_ready():
                chunk = channel.recv_stderr(65536).decode("utf-8", errors="replace")
                error_chunks.append(chunk)
                if stream:
                    print(chunk, end="", flush=True)
            if channel.exit_status_ready():
                while channel.recv_ready():
                    chunk = channel.recv(65536).decode("utf-8", errors="replace")
                    output_chunks.append(chunk)
                    if stream:
                        print(chunk, end="", flush=True)
                while channel.recv_stderr_ready():
                    chunk = channel.recv_stderr(65536).decode("utf-8", errors="replace")
                    error_chunks.append(chunk)
                    if stream:
                        print(chunk, end="", flush=True)
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


def build_model_candidates(target: RemoteTarget, model_name: str) -> tuple[str, ...]:
    return tuple(posixpath.join(root, model_name) for root in target.model_search_roots)


def normalize_remote_directory_permissions(
    remote: RemoteHost,
    *,
    image: str,
    remote_dir: str,
) -> None:
    parent_dir = posixpath.dirname(remote_dir)
    dir_name = posixpath.basename(remote_dir)
    command = shell_join(
        [
            "docker",
            "run",
            "--rm",
            "-v",
            f"{parent_dir}:/mnt",
            "--entrypoint",
            "sh",
            image,
            "-lc",
            (
                f"chown -R 1000:1000 /mnt/{dir_name} || true; "
                f"chmod -R a+rX /mnt/{dir_name}"
            ),
        ]
    )
    remote.run(command, timeout=3600)


def maybe_download_remote_model(
    remote: RemoteHost,
    target: RemoteTarget,
    *,
    model_name: str,
    model_source: str,
) -> str:
    """Resolve a remote path or model repo id into a remote local directory when needed."""
    config_path = posixpath.join(model_source, "config.json")
    if remote.exists(config_path):
        return model_source

    if "/" not in model_source:
        raise FileNotFoundError(
            f"Remote model source {model_source!r} does not exist and is not a valid HF repo id."
        )

    remote_model_root = posixpath.join(target.remote_root, "models")
    remote_model_dir = posixpath.join(remote_model_root, model_name)
    force_download = os.environ.get("REMOTE_MODEL_FORCE_DOWNLOAD", "0").strip().lower() in {"1", "true", "yes"}
    if remote.exists(posixpath.join(remote_model_dir, "config.json")) and not force_download:
        return remote_model_dir

    downloader_image = os.environ.get("REMOTE_MODEL_DOWNLOADER_IMAGE", DEFAULT_MODEL_DOWNLOADER_IMAGE).strip()
    model_provider = os.environ.get("REMOTE_MODEL_PROVIDER", "huggingface").strip().lower()
    hf_endpoint = os.environ.get("REMOTE_HF_ENDPOINT", "https://hf-mirror.com").strip() or "https://hf-mirror.com"
    remote.run(f"mkdir -p {shlex.quote(remote_model_root)}", timeout=60)
    download_parts = [
        "docker",
        "run",
        "--rm",
        "-v",
        f"{remote_model_root}:/models",
    ]
    if model_provider in {"modelscope", "ms"}:
        download_parts.extend(
            [
                "--entrypoint",
                "sh",
                downloader_image,
                "-lc",
                shlex.quote(
                    "export MODELSCOPE_CACHE=/models/.modelscope-cache && "
                    "rm -rf /tmp/modelscope-venv && "
                    "uv venv /tmp/modelscope-venv && "
                    "uv pip install --python /tmp/modelscope-venv/bin/python --no-cache-dir modelscope && "
                    "/tmp/modelscope-venv/bin/python -c "
                    + shlex.quote(
                        "from modelscope import snapshot_download; "
                        f"snapshot_download({model_source!r}, local_dir='/models/{model_name}')"
                    )
                ),
            ]
        )
    else:
        download_parts.extend(
            [
                "--entrypoint",
                "python3",
                "-e",
                "HF_HUB_OFFLINE=0",
                "-e",
                "TRANSFORMERS_OFFLINE=0",
                "-e",
                f"HF_ENDPOINT={hf_endpoint}",
                downloader_image,
                "-c",
                shlex.quote(
                    "from huggingface_hub import snapshot_download; "
                    f"snapshot_download(repo_id={model_source!r}, "
                    f"local_dir='/models/{model_name}', "
                    "local_dir_use_symlinks=False)"
                ),
            ]
        )
    download_command = " ".join(download_parts)
    remote.run(download_command, timeout=14400, stream=True)
    normalize_remote_directory_permissions(remote, image=downloader_image, remote_dir=remote_model_dir)
    if not remote.exists(posixpath.join(remote_model_dir, "config.json")):
        raise FileNotFoundError(
            f"Downloaded model {model_source!r}, but {remote_model_dir}/config.json was not found."
        )
    return remote_model_dir


def build_context_tarball() -> Path:
    handle = tempfile.NamedTemporaryFile(prefix="faster-qwen3-tts-orin-", suffix=".tar", delete=False)
    handle.close()
    tar_path = Path(handle.name)
    with tarfile.open(tar_path, "w") as tar:
        for relative_path in [
            ".dockerignore",
            "faster-qwen3-tts-jetson.Dockerfile",
            "faster-qwen3-tts-jetson-prebuilt.Dockerfile",
            "faster-qwen3-tts-requirements.txt",
            "MANIFEST.in",
            "README.md",
            "pyproject.toml",
            "voices.json",
            "voices",
            "examples/openai_server.py",
            "faster_qwen3_tts",
        ]:
            tar.add(REPO_ROOT / relative_path, arcname=relative_path)
    return tar_path


def main() -> None:
    target = RemoteTarget()
    base_image = os.environ.get("ORIN_AIPOD_BASE_IMAGE", target.base_image).strip() or target.base_image
    prebuilt_app_image = os.environ.get("ORIN_AIPOD_PREBUILT_APP_IMAGE", "").strip()
    image_name = os.environ.get("ORIN_AIPOD_IMAGE", DEFAULT_IMAGE).strip() or DEFAULT_IMAGE
    model_name = os.environ.get("ORIN_AIPOD_MODEL_NAME", DEFAULT_MODEL_NAME).strip() or DEFAULT_MODEL_NAME
    server_mode = os.environ.get("ORIN_AIPOD_MODE", "clone").strip() or "clone"
    default_voice = os.environ.get(
        "ORIN_AIPOD_DEFAULT_VOICE",
        "vivian",
    ).strip() or "vivian"
    push_enabled = os.environ.get("ORIN_AIPOD_PUSH", "1").strip().lower() not in {"0", "false", "no"}
    model_source = os.environ.get("REMOTE_MODEL_SOURCE", "").strip()
    remote = RemoteHost(target)
    tar_path = build_context_tarball()
    remote_tar = posixpath.join(target.remote_root, "build", f"context-{int(time.time())}.tar")
    context_root = posixpath.join(target.remote_root, "build", "context")

    try:
        if not model_source:
            for candidate in build_model_candidates(target, model_name):
                if remote.exists(posixpath.join(candidate, "config.json")):
                    model_source = candidate
                    break
        if not model_source:
            raise FileNotFoundError(
                "Could not find a remote model directory for "
                f"{model_name}. Set REMOTE_MODEL_SOURCE to an existing directory."
            )
        model_source = maybe_download_remote_model(
            remote,
            target,
            model_name=model_name,
            model_source=model_source,
        )
        normalize_remote_directory_permissions(
            remote,
            image=os.environ.get("REMOTE_MODEL_DOWNLOADER_IMAGE", DEFAULT_MODEL_DOWNLOADER_IMAGE).strip()
            or DEFAULT_MODEL_DOWNLOADER_IMAGE,
            remote_dir=model_source,
        )

        remote.upload(tar_path, remote_tar)
        remote.run(
            f"rm -rf {shlex.quote(context_root)} && mkdir -p {shlex.quote(context_root)} && "
            f"tar -xf {shlex.quote(remote_tar)} -C {shlex.quote(context_root)}",
            timeout=300,
        )
        remote.run(
            f"mkdir -p {shlex.quote(posixpath.join(context_root, 'models'))} && "
            f"cp -a {shlex.quote(model_source)} {shlex.quote(posixpath.join(context_root, 'models', model_name))}",
            timeout=3600,
        )

        build_args = [
            "docker",
            "build",
            "--build-arg",
            f"BASE_IMAGE={base_image}",
            "--build-arg",
            f"PIP_INDEX_URL={target.pip_index_url}",
            "--build-arg",
            f"PIP_EXTRA_INDEX_URL={target.pip_extra_index_url}",
            "--build-arg",
            f"TORCHAUDIO_SPEC={target.torchaudio_spec}",
            "--build-arg",
            f"MODEL_NAME={model_name}",
            "--build-arg",
            f"MODEL_DIR=/opt/models/{model_name}",
            "--build-arg",
            f"QWEN_TTS_MODEL=/opt/models/{model_name}",
            "--build-arg",
            f"QWEN_TTS_MODE={server_mode}",
            "--build-arg",
            f"QWEN_TTS_DEFAULT_VOICE={default_voice}",
            "-f",
            (
                "faster-qwen3-tts-jetson-prebuilt.Dockerfile"
                if prebuilt_app_image
                else "faster-qwen3-tts-jetson.Dockerfile"
            ),
            "-t",
            image_name,
            ".",
        ]
        if prebuilt_app_image:
            build_args[4:4] = [
                "--build-arg",
                f"PREBUILT_APP_IMAGE={prebuilt_app_image or DEFAULT_PREBUILT_APP_IMAGE}",
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
        remote.run(command, timeout=14400, stream=True)
        print(f"Built image on Orin: {image_name} (model={model_name}, mode={server_mode}, default_voice={default_voice})")
        if push_enabled:
            remote.run(f"docker push {shlex.quote(image_name)}", timeout=14400, stream=True)
            print(f"Pushed image: {image_name}")
        else:
            print("Skipped docker push because ORIN_AIPOD_PUSH=0")
    finally:
        remote.close()
        tar_path.unlink(missing_ok=True)


if __name__ == "__main__":
    main()

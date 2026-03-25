#!/usr/bin/env python3
from __future__ import annotations

import argparse
import io
import json
import os
import posixpath
import re
import shlex
import struct
import sys
import tarfile
import tempfile
import time
import urllib.request
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import paramiko

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.common import (  # noqa: E402
    build_text_verification,
    detect_system_info,
    ensure_dir,
    strip_asr_special_tokens,
    transcribe_audio_openai_compat,
    write_json,
)


DEFAULT_TEXT = (
    "临海的小城入秋总比别处慢半拍。清晨的风从骑楼缝隙里穿过去，带着一点潮湿，也带着刚出炉面包的甜味。"
    "她沿着旧街往前走，鞋跟轻轻敲在石板路上，像在替这座城市数着缓慢的心跳。街角书店的老板正把木门推开，"
    "门铃响了一下，惊醒了窗边打盹的猫。远处有人在摊位前挑选橘子，讨价还价的声音不高，却让整条街显得更有人情味。"
)
LANGUAGE_TO_ASR_CODE = {
    "chinese": "zh",
    "english": "en",
    "japanese": "ja",
    "korean": "ko",
    "german": "de",
    "french": "fr",
    "russian": "ru",
    "portuguese": "pt",
    "spanish": "es",
    "italian": "it",
}
REMOTE_PORT = 8000
SERVICE_NAME_PREFIX = "faster-qwen3-tts-smoke"
LOCAL_MODEL_DIR = ROOT / "models" / "Qwen3-TTS-12Hz-0.6B-CustomVoice"
FASTER_REPO = Path.home() / "faster-qwen3-tts"
COLLECT_JETSON_METRICS = Path.home() / "lzc-aipod-pkgs" / "scripts" / "collect_jetson_metrics.py"


@dataclass(frozen=True)
class TargetConfig:
    key: str
    machine_name: str
    host: str
    user: str
    password: str
    remote_root: str
    base_image: str
    pip_index_url: str
    pip_extra_index_url: str
    torchaudio_spec: str
    telemetry_backend: str
    existing_model_dir: str | None = None
    needs_maxn: bool = False


TARGETS: dict[str, TargetConfig] = {
    "t4000": TargetConfig(
        key="t4000",
        machine_name="T4000",
        host="tegra-ubuntu-t4000.lan",
        user="nvidia",
        password="nvidia",
        remote_root="/nvme_disk/thor-validation/faster-qwen3-tts-smoke",
        base_image="registry.lazycat.cloud/x/lzc-aipod-vllm:0.16.0-cu130-thor",
        pip_index_url="https://pypi.jetson-ai-lab.io/sbsa/cu130/+simple",
        pip_extra_index_url="https://pypi.org/simple",
        torchaudio_spec="torchaudio==2.9.0",
        telemetry_backend="nvidia-smi",
        existing_model_dir="/nvme_disk/thor-validation/qwen3-tts-bench/models/Qwen3-TTS-12Hz-0.6B-CustomVoice",
        needs_maxn=True,
    ),
    "orin": TargetConfig(
        key="orin",
        machine_name="lzc-pod-juyIZt",
        host="lzc-pod-juyIZt.lan",
        user="nvidia",
        password="nvidia",
        remote_root="/home/nvidia/faster-qwen3-tts-smoke",
        base_image="127.0.0.1:5001/x/lzc-aipod-vllm:bffa39b-orin",
        pip_index_url="https://pypi.jetson-ai-lab.io/jp6/cu126/+simple",
        pip_extra_index_url="https://pypi.org/simple",
        torchaudio_spec="torchaudio==2.10.0",
        telemetry_backend="jtop",
    ),
}


class RemoteHost:
    def __init__(self, target: TargetConfig):
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

    def run(self, command: str, *, check: bool = True, get_pty: bool = False, timeout: int | None = None) -> str:
        stdin, stdout, stderr = self.client.exec_command(command, get_pty=get_pty, timeout=timeout)
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
        exit_status = channel.recv_exit_status()
        combined = "".join(output_chunks) + "".join(error_chunks)
        if check and exit_status != 0:
            raise RuntimeError(f"remote command failed ({exit_status}): {command}\n{combined}")
        return combined

    def mkdir_p(self, remote_path: str) -> None:
        parts = []
        current = remote_path
        while current not in ("", "/"):
            parts.append(current)
            current = posixpath.dirname(current)
        for path in reversed(parts):
            try:
                self.sftp.stat(path)
            except FileNotFoundError:
                self.sftp.mkdir(path)

    def upload_file(self, local_path: Path, remote_path: str) -> None:
        self.mkdir_p(posixpath.dirname(remote_path))
        self.sftp.put(str(local_path), remote_path)

    def upload_bytes(self, payload: bytes, remote_path: str) -> None:
        self.mkdir_p(posixpath.dirname(remote_path))
        with self.sftp.file(remote_path, "wb") as fh:
            fh.write(payload)

    def remote_exists(self, remote_path: str) -> bool:
        try:
            self.sftp.stat(remote_path)
            return True
        except FileNotFoundError:
            return False

    def stat(self, remote_path: str):
        return self.sftp.stat(remote_path)

    def read_text(self, remote_path: str) -> str:
        with self.sftp.file(remote_path, "r") as fh:
            return fh.read().decode("utf-8", errors="replace")

    def download_file(self, remote_path: str, local_path: Path) -> None:
        ensure_dir(local_path.parent)
        self.sftp.get(remote_path, str(local_path))


def shell_join(parts: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in parts)


def remote_quote(path: str) -> str:
    return shlex.quote(path)


def infer_asr_language(language: str) -> str | None:
    return LANGUAGE_TO_ASR_CODE.get(language.strip().lower())


def wav_duration_seconds(path: Path) -> float:
    with path.open("rb") as fh:
        header = fh.read(44)
    if len(header) >= 44 and header[:4] == b"RIFF" and header[8:12] == b"WAVE":
        channels = struct.unpack("<H", header[22:24])[0]
        sample_rate = struct.unpack("<I", header[24:28])[0]
        bits_per_sample = struct.unpack("<H", header[34:36])[0]
        if channels > 0 and sample_rate > 0 and bits_per_sample > 0:
            bytes_per_frame = channels * bits_per_sample / 8.0
            if bytes_per_frame > 0:
                data_bytes = max(path.stat().st_size - 44, 0)
                return float(data_bytes / bytes_per_frame / sample_rate)
    with wave.open(str(path), "rb") as wf:
        return float(wf.getnframes() / wf.getframerate())


def build_context_tarball() -> Path:
    temp = tempfile.NamedTemporaryFile(prefix="faster-qwen3-tts-context-", suffix=".tar", delete=False)
    temp.close()
    tar_path = Path(temp.name)
    with tarfile.open(tar_path, "w") as tar:
        tar.add(ROOT / "docker" / "faster-qwen3-tts-jetson.Dockerfile", arcname="Dockerfile")
        tar.add(ROOT / "docker" / "faster-qwen3-tts-requirements.txt", arcname="faster-qwen3-tts-requirements.txt")
        tar.add(ROOT / "pyproject.toml", arcname="qwen3-tts/pyproject.toml")
        tar.add(ROOT / "README.md", arcname="qwen3-tts/README.md")
        tar.add(ROOT / "qwen_tts", arcname="qwen3-tts/qwen_tts")
        tar.add(FASTER_REPO / "pyproject.toml", arcname="faster-qwen3-tts/pyproject.toml")
        tar.add(FASTER_REPO / "README.md", arcname="faster-qwen3-tts/README.md")
        tar.add(FASTER_REPO / "faster_qwen3_tts", arcname="faster-qwen3-tts/faster_qwen3_tts")
        tar.add(FASTER_REPO / "examples" / "openai_server.py", arcname="faster-qwen3-tts/examples/openai_server.py")
        tar.add(FASTER_REPO / "demo", arcname="faster-qwen3-tts-demo")
    return tar_path


def upload_model_dir(remote: RemoteHost, local_dir: Path, remote_dir: str) -> None:
    for entry in local_dir.rglob("*"):
        relative = entry.relative_to(local_dir).as_posix()
        remote_path = posixpath.join(remote_dir, relative)
        if entry.is_dir():
            remote.mkdir_p(remote_path)
            continue
        remote.mkdir_p(posixpath.dirname(remote_path))
        needs_upload = True
        if remote.remote_exists(remote_path):
            local_stat = entry.stat()
            remote_stat = remote.stat(remote_path)
            needs_upload = local_stat.st_size != remote_stat.st_size
        if needs_upload:
            remote.upload_file(entry, remote_path)


def ensure_remote_context(remote: RemoteHost, target: TargetConfig, tar_path: Path) -> str:
    build_root = posixpath.join(target.remote_root, "build")
    context_root = posixpath.join(build_root, "context")
    tar_remote = posixpath.join(build_root, "context.tar")
    remote.run(f"mkdir -p {remote_quote(build_root)} && rm -rf {remote_quote(context_root)}")
    remote.upload_file(tar_path, tar_remote)
    remote.run(
        f"mkdir -p {remote_quote(context_root)} && "
        f"tar -xf {remote_quote(tar_remote)} -C {remote_quote(context_root)}"
    )
    return context_root


def ensure_remote_model(remote: RemoteHost, target: TargetConfig) -> str:
    workspace_model_dir = posixpath.join(target.remote_root, "workspace", "models", LOCAL_MODEL_DIR.name)
    if target.existing_model_dir and remote.remote_exists(posixpath.join(target.existing_model_dir, "config.json")):
        remote.run(f"mkdir -p {remote_quote(posixpath.dirname(workspace_model_dir))}")
        return target.existing_model_dir

    if not LOCAL_MODEL_DIR.exists():
        raise FileNotFoundError(f"local model not found: {LOCAL_MODEL_DIR}")

    upload_model_dir(remote, LOCAL_MODEL_DIR, workspace_model_dir)
    return workspace_model_dir


def configure_machine(remote: RemoteHost, target: TargetConfig) -> str:
    if not target.needs_maxn:
        return ""
    return remote.run(
        " && ".join(
            [
                f"echo {shlex.quote(target.password)} | sudo -S nvpmodel -m 0",
                f"echo {shlex.quote(target.password)} | sudo -S jetson_clocks",
                f"echo {shlex.quote(target.password)} | sudo -S nvpmodel -q --verbose",
                f"echo {shlex.quote(target.password)} | sudo -S jetson_clocks --show",
            ]
        ),
        get_pty=True,
    )


def build_remote_image(remote: RemoteHost, target: TargetConfig, context_root: str, image_name: str) -> str:
    build_args = [
        "docker", "build",
        "--build-arg", "BUILDKIT_INLINE_CACHE=1",
        "--build-arg", f"BASE_IMAGE={target.base_image}",
        "--build-arg", f"PIP_INDEX_URL={target.pip_index_url}",
        "--build-arg", f"PIP_EXTRA_INDEX_URL={target.pip_extra_index_url}",
        "--build-arg", f"TORCHAUDIO_SPEC={target.torchaudio_spec}",
        "-t", image_name,
        ".",
    ]
    build_proxy = os.environ.get("REMOTE_BUILD_PROXY", "").strip()
    if build_proxy:
        for proxy_var in ["HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"]:
            build_args.extend(["--build-arg", f"{proxy_var}={build_proxy}"])
    no_proxy = os.environ.get("REMOTE_BUILD_NO_PROXY", "").strip()
    if no_proxy:
        for proxy_var in ["NO_PROXY", "no_proxy"]:
            build_args.extend(["--build-arg", f"{proxy_var}={no_proxy}"])

    command = f"cd {remote_quote(context_root)} && export DOCKER_BUILDKIT=1 && {shell_join(build_args)}"
    return remote.run(command, timeout=7200)


def start_telemetry(remote: RemoteHost, target: TargetConfig, run_root: str) -> tuple[str, str]:
    telemetry_path = posixpath.join(run_root, f"{target.key}.telemetry.jsonl")
    if target.telemetry_backend == "jtop":
        remote_script = posixpath.join(target.remote_root, "scripts", "collect_jetson_metrics.py")
        if not remote.remote_exists(remote_script):
            remote.upload_file(COLLECT_JETSON_METRICS, remote_script)
        command = (
            f"mkdir -p {remote_quote(run_root)} && "
            f"nohup python3 {remote_quote(remote_script)} --output {remote_quote(telemetry_path)} "
            f"--interval 1.0 --label {remote_quote(target.key)} >/dev/null 2>&1 & echo $!"
        )
    else:
        csv_header = "timestamp,name,compute_cap,temperature.gpu,utilization.gpu,utilization.memory,power.draw,memory.used,memory.total"
        query = (
            "timestamp,name,compute_cap,temperature.gpu,utilization.gpu,"
            "utilization.memory,power.draw,memory.used,memory.total"
        )
        command = (
            f"mkdir -p {remote_quote(run_root)} && "
            f"printf '%s\\n' {remote_quote(csv_header)} > {remote_quote(telemetry_path)} && "
            f"nohup bash -lc "
            f"{remote_quote(f'while true; do nvidia-smi --query-gpu={query} --format=csv,noheader >> {telemetry_path}; sleep 1; done')} "
            f">/dev/null 2>&1 & echo $!"
        )
    pid = remote.run(command).strip().splitlines()[-1].strip()
    return pid, telemetry_path


def stop_telemetry(remote: RemoteHost, pid: str) -> None:
    if not pid:
        return
    remote.run(f"kill {shlex.quote(pid)} >/dev/null 2>&1 || true", check=False)


def start_container(
    remote: RemoteHost,
    target: TargetConfig,
    image_name: str,
    container_name: str,
    model_source_dir: str,
) -> None:
    workspace_root = posixpath.join(target.remote_root, "workspace")
    run_args = [
        "docker", "run", "-d", "--runtime", "nvidia", "--network", "host",
        "--name", container_name,
        "-e", "HF_HUB_OFFLINE=1",
        "-e", "TRANSFORMERS_OFFLINE=1",
        "-v", f"{workspace_root}:/workspace",
    ]
    container_model_dir = f"/workspace/models/{LOCAL_MODEL_DIR.name}"
    if model_source_dir != posixpath.join(workspace_root, "models", LOCAL_MODEL_DIR.name):
        run_args.extend(["-v", f"{model_source_dir}:{container_model_dir}:ro"])
    run_args.extend(
        [
            image_name,
            "--mode", "custom",
            "--model", container_model_dir,
            "--language", "Chinese",
            "--default-voice", "vivian",
            "--speakers", "vivian",
            "--host", "0.0.0.0",
            "--port", str(REMOTE_PORT),
            "--chunk-size", "8",
        ]
    )
    remote.run(f"docker rm -f {remote_quote(container_name)} >/dev/null 2>&1 || true", check=False)
    remote.run(shell_join(run_args), timeout=300)


def wait_for_health(base_url: str, timeout_s: int) -> tuple[float, dict[str, Any]]:
    started = time.perf_counter()
    deadline = started + timeout_s
    last_error = "service_not_started"
    while time.perf_counter() < deadline:
        try:
            with urllib.request.urlopen(base_url.rstrip("/") + "/health", timeout=5) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
            return time.perf_counter() - started, payload
        except Exception as exc:  # noqa: BLE001
            last_error = repr(exc)
            time.sleep(1)
    raise TimeoutError(f"service failed to become ready within {timeout_s}s: {last_error}")


def fetch_voices(base_url: str) -> dict[str, Any]:
    with urllib.request.urlopen(base_url.rstrip("/") + "/v1/audio/voices", timeout=30) as resp:
        return json.loads(resp.read().decode("utf-8"))


def request_speech(base_url: str, text: str, voice: str) -> tuple[dict[str, Any], bytes]:
    payload = json.dumps(
        {
            "model": "tts-1",
            "input": text,
            "voice": voice,
            "response_format": "wav",
        },
        ensure_ascii=False,
    ).encode("utf-8")
    req = urllib.request.Request(
        base_url.rstrip("/") + "/v1/audio/speech",
        data=payload,
        headers={"Content-Type": "application/json", "Accept": "audio/wav"},
        method="POST",
    )
    started = time.perf_counter()
    with urllib.request.urlopen(req, timeout=600) as resp:
        status = resp.status
        headers = {str(k): str(v) for k, v in resp.headers.items()}
        first_chunk = resp.read(4096)
        first_chunk_seconds = time.perf_counter() - started
        chunks = [first_chunk]
        while True:
            chunk = resp.read(65536)
            if not chunk:
                break
            chunks.append(chunk)
    duration_seconds = time.perf_counter() - started
    return {
        "status_code": status,
        "headers": headers,
        "first_chunk_seconds": first_chunk_seconds,
        "duration_seconds": duration_seconds,
    }, b"".join(chunks)


def summarize_hot_requests(hot_requests: list[dict[str, Any]]) -> dict[str, Any]:
    if not hot_requests:
        raise ValueError("hot_requests must not be empty")
    count = len(hot_requests)
    return {
        "runs": count,
        "status_code": hot_requests[-1]["status_code"],
        "headers": hot_requests[-1]["headers"],
        "first_chunk_seconds": sum(item["first_chunk_seconds"] for item in hot_requests) / count,
        "duration_seconds": sum(item["duration_seconds"] for item in hot_requests) / count,
        "first_chunk_seconds_min": min(item["first_chunk_seconds"] for item in hot_requests),
        "duration_seconds_min": min(item["duration_seconds"] for item in hot_requests),
        "first_chunk_seconds_last": hot_requests[-1]["first_chunk_seconds"],
        "duration_seconds_last": hot_requests[-1]["duration_seconds"],
    }


def verify_audio(wav_path: Path, expected_text: str, language: str, args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any]]:
    asr_payload = transcribe_audio_openai_compat(
        wav_path,
        base_url=args.asr_base_url,
        model=args.asr_model,
        timeout=args.asr_timeout,
        language=infer_asr_language(language),
        response_format="json",
    )
    cleaned_text = strip_asr_special_tokens(asr_payload.get("text") or asr_payload.get("result") or "")
    if "text" in asr_payload:
        asr_payload["text"] = cleaned_text
    if "result" in asr_payload:
        asr_payload["result"] = cleaned_text
    verification = build_text_verification(
        expected_text,
        cleaned_text,
        min_length_ratio=args.verify_min_length_ratio,
        min_matching_ratio=args.verify_min_coverage_ratio,
    )
    return asr_payload, verification


def parse_t4000_telemetry(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    rows = []
    for raw_line in path.read_text(encoding="utf-8", errors="replace").splitlines()[1:]:
        line = raw_line.strip()
        if not line:
            continue
        parts = [item.strip() for item in line.split(",")]
        if len(parts) < 9:
            continue
        rows.append(
            {
                "temperature_g": _to_float(parts[3]),
                "gpu_util": _to_float(parts[4]),
                "mem_util": _to_float(parts[5]),
                "power_draw_w": _to_float(parts[6]),
                "memory_used_mib": _to_float(parts[7]),
                "memory_total_mib": _to_float(parts[8]),
            }
        )
    return {
        "samples": len(rows),
        "max_temperature_c": _max_key(rows, "temperature_g"),
        "max_power_w": _max_key(rows, "power_draw_w"),
        "max_gpu_util_pct": _max_key(rows, "gpu_util"),
        "max_memory_used_mib": _max_key(rows, "memory_used_mib"),
    }


def _to_float(value: str) -> float | None:
    cleaned = value.strip()
    if not cleaned or cleaned in {"[N/A]", "N/A"}:
        return None
    match = re.search(r"-?\d+(?:\.\d+)?", cleaned)
    if match:
        cleaned = match.group(0)
    try:
        return float(cleaned)
    except Exception:  # noqa: BLE001
        return None


def _max_key(rows: list[dict[str, float | None]], key: str) -> float | None:
    values = [row[key] for row in rows if row.get(key) is not None]
    return max(values) if values else None


def parse_orin_telemetry_summary(path: Path) -> dict[str, Any]:
    summary_path = path.with_suffix(".summary.json")
    if not summary_path.exists():
        samples = []
        if not path.exists():
            return {}
        for raw_line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            line = raw_line.strip()
            if not line:
                continue
            try:
                samples.append(json.loads(line))
            except json.JSONDecodeError:
                continue

        def max_derived(key: str) -> float | None:
            values = []
            for sample in samples:
                value = (sample.get("derived") or {}).get(key)
                if value is not None:
                    values.append(value)
            return max(values) if values else None

        return {
            "samples": len(samples),
            "max_temp_c": max(filter(None, [max_derived("gpu_temp_c"), max_derived("cpu_temp_c")]), default=None),
            "max_power_mw": max_derived("power_mw"),
            "max_fan_pwm": max_derived("fan_pwm"),
            "max_gpu_util_pct": max_derived("gpu_util_pct"),
        }
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    return payload.get("summary", {})


def collect_remote_artifacts(remote: RemoteHost, target: TargetConfig, run_root: str, local_run_dir: Path, container_name: str, telemetry_path: str) -> dict[str, Any]:
    ensure_dir(local_run_dir)
    service_log = remote.run(f"docker logs {remote_quote(container_name)}", check=False)
    (local_run_dir / "service.log").write_text(service_log, encoding="utf-8")
    remote.run(f"docker rm -f {remote_quote(container_name)} >/dev/null 2>&1 || true", check=False)

    local_telemetry = local_run_dir / Path(telemetry_path).name
    if remote.remote_exists(telemetry_path):
        remote.download_file(telemetry_path, local_telemetry)
    if target.telemetry_backend == "jtop":
        remote_summary = posixpath.splitext(telemetry_path)[0] + ".summary.json"
        if remote.remote_exists(remote_summary):
            remote.download_file(remote_summary, local_telemetry.with_suffix(".summary.json"))
        telemetry_summary = parse_orin_telemetry_summary(local_telemetry)
    else:
        telemetry_summary = parse_t4000_telemetry(local_telemetry)

    return telemetry_summary


def remote_environment_snapshot(remote: RemoteHost, target: TargetConfig) -> dict[str, Any]:
    if target.telemetry_backend == "jtop":
        raw = remote.run(
            "python3 - <<'PY'\n"
            "import json\n"
            "from jtop import jtop\n"
            "def safe(value):\n"
            "    if isinstance(value, dict):\n"
            "        return {str(k): safe(v) for k, v in value.items()}\n"
            "    if isinstance(value, (list, tuple)):\n"
            "        return [safe(item) for item in value]\n"
            "    if value is None or isinstance(value, (str, int, float, bool)):\n"
            "        return value\n"
            "    return str(value)\n"
            "with jtop(interval=0.1) as jetson:\n"
            "    if jetson.ok():\n"
            "        print(json.dumps({'stats': safe(jetson.stats), 'board': safe(jetson.board)}, ensure_ascii=False))\n"
            "PY",
            timeout=60,
        )
        return json.loads(raw.strip().splitlines()[-1])
    raw = remote.run(
        "nvidia-smi --query-gpu=name,compute_cap,driver_version,temperature.gpu,power.draw,memory.total "
        "--format=csv,noheader,nounits",
    )
    return {"nvidia_smi": raw.strip()}


def write_report(run_dir: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Faster Qwen3-TTS Smoke Test",
        "",
        f"- `machine_name`: {payload['machine_name']}",
        f"- `host`: {payload['host']}",
        f"- `base_image`: {payload['base_image']}",
        f"- `startup_seconds`: {payload['startup_seconds']:.3f}",
        f"- `startup_warmup_completed`: {payload['health'].get('startup_warmup_completed')}",
        f"- `startup_warmup_seconds`: {payload['health'].get('startup_warmup_seconds')}",
        f"- `hot_request_runs`: {payload['request'].get('runs', 1)}",
        f"- `request_seconds`: {payload['request']['duration_seconds']:.3f}",
        f"- `first_chunk_seconds`: {payload['request']['first_chunk_seconds']:.3f}",
        f"- `request_seconds_min`: {payload['request'].get('duration_seconds_min', payload['request']['duration_seconds']):.3f}",
        f"- `first_chunk_seconds_min`: {payload['request'].get('first_chunk_seconds_min', payload['request']['first_chunk_seconds']):.3f}",
        f"- `audio_seconds`: {payload['audio_seconds']:.3f}",
        f"- `x_realtime`: {payload['x_realtime']:.3f}",
        f"- `verification_passed`: {payload['verification']['passed']}",
        f"- `verification_reason`: {payload['verification']['reason']}",
        f"- `voices`: {', '.join(payload['voices'].get('voices', []))}",
    ]
    telemetry = payload.get("telemetry_summary") or {}
    if telemetry:
        lines.extend(
            [
                "",
                "## Telemetry",
                "",
                *(f"- `{key}`: {value}" for key, value in telemetry.items()),
            ]
        )
    (run_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_comparison(output_dir: Path, results: list[dict[str, Any]]) -> None:
    ensure_dir(output_dir)
    ordered = sorted(results, key=lambda item: item["machine_name"])
    comparison = {
        "results": ordered,
        "winner_startup": min(ordered, key=lambda item: item["startup_seconds"])["machine_name"],
        "winner_request": min(ordered, key=lambda item: item["request"]["duration_seconds"])["machine_name"],
        "winner_x_realtime": max(ordered, key=lambda item: item["x_realtime"])["machine_name"],
    }
    write_json(output_dir / "comparison.json", comparison)
    lines = [
        "# Faster Qwen3-TTS Smoke Comparison",
        "",
        "| machine | startup_s | request_s | first_chunk_s | audio_s | x_realtime | passed |",
        "| --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for item in ordered:
        lines.append(
            "| {machine_name} | {startup_seconds:.3f} | {request_s:.3f} | {first_chunk_s:.3f} | {audio_seconds:.3f} | {x_realtime:.3f} | {passed} |".format(
                machine_name=item["machine_name"],
                startup_seconds=item["startup_seconds"],
                request_s=item["request"]["duration_seconds"],
                first_chunk_s=item["request"]["first_chunk_seconds"],
                audio_seconds=item["audio_seconds"],
                x_realtime=item["x_realtime"],
                passed=item["verification"]["passed"],
            )
        )
    (output_dir / "comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_target(args: argparse.Namespace, target: TargetConfig, tar_path: Path) -> dict[str, Any]:
    run_dir = ensure_dir(ROOT / "benchmarks" / "results" / target.machine_name / "faster_qwen3_tts_smoke" / args.run_tag)
    local_client_info = detect_system_info()
    image_name = f"{SERVICE_NAME_PREFIX}:{args.run_tag}-{target.key}"
    container_name = f"{SERVICE_NAME_PREFIX}-{args.run_tag}-{target.key}".replace("_", "-")
    remote = RemoteHost(target)
    telemetry_pid = ""
    telemetry_path = ""
    run_root = posixpath.join(target.remote_root, "runs", args.run_tag)
    result: dict[str, Any] | None = None
    try:
        ensure_remote_context(remote, target, tar_path)
        model_source_dir = ensure_remote_model(remote, target)
        performance_log = configure_machine(remote, target)
        if performance_log:
            (run_dir / "performance_mode.log").write_text(performance_log, encoding="utf-8")

        context_root = posixpath.join(target.remote_root, "build", "context")
        build_log = build_remote_image(remote, target, context_root, image_name)
        (run_dir / "build.log").write_text(build_log, encoding="utf-8")

        environment = remote_environment_snapshot(remote, target)
        write_json(run_dir / "environment.json", environment)

        start_container(remote, target, image_name, container_name, model_source_dir)
        base_url = f"http://{target.host}:{REMOTE_PORT}"
        startup_seconds, health_payload = wait_for_health(base_url, args.startup_timeout)
        write_json(run_dir / "health.json", health_payload)

        telemetry_pid, telemetry_path = start_telemetry(remote, target, run_root)
        voices_payload = fetch_voices(base_url)
        write_json(run_dir / "voices_result.json", voices_payload)
        hot_request_runs: list[dict[str, Any]] = []
        hot_request_outputs: list[dict[str, Any]] = []
        for request_idx in range(args.hot_request_runs):
            request_meta, audio_bytes = request_speech(base_url, args.text, args.voice)
            output_wav = run_dir / f"output_hot_{request_idx + 1}.wav"
            output_wav.write_bytes(audio_bytes)
            audio_seconds = wav_duration_seconds(output_wav)
            hot_request_runs.append(request_meta)
            hot_request_outputs.append(
                {
                    "request_index": request_idx + 1,
                    "request": request_meta,
                    "audio_seconds": audio_seconds,
                    "output_wav": output_wav.name,
                }
            )
        request_meta = summarize_hot_requests(hot_request_runs)
        selected_output = hot_request_outputs[-1]
        selected_output_wav = run_dir / selected_output["output_wav"]
        output_wav = run_dir / "output.wav"
        output_wav.write_bytes(selected_output_wav.read_bytes())
        audio_seconds = selected_output["audio_seconds"]
        asr_payload, verification = verify_audio(output_wav, args.text, "Chinese", args)
        write_json(run_dir / "candidate_result.json", {
            "request": request_meta,
            "audio_seconds": audio_seconds,
            "hot_requests": hot_request_outputs,
            "selected_request_index": selected_output["request_index"],
            "asr": asr_payload,
            "verification": verification,
            "voices": voices_payload,
        })
        result = {
            "machine_name": target.machine_name,
            "host": target.host,
            "base_image": target.base_image,
            "image_name": image_name,
            "startup_seconds": startup_seconds,
            "request": request_meta,
            "audio_seconds": audio_seconds,
            "x_realtime": audio_seconds / request_meta["duration_seconds"] if request_meta["duration_seconds"] > 0 else 0.0,
            "hot_requests": hot_request_outputs,
            "selected_request_index": selected_output["request_index"],
            "voices": voices_payload,
            "health": health_payload,
            "verification": verification,
            "asr": asr_payload,
            "local_client_info": local_client_info,
        }
        write_json(run_dir / "result.json", result)
        write_report(run_dir, result)
    finally:
        if telemetry_pid:
            stop_telemetry(remote, telemetry_pid)
        telemetry_summary = collect_remote_artifacts(
            remote,
            target,
            run_root,
            run_dir,
            container_name,
            telemetry_path or posixpath.join(run_root, f"{target.key}.telemetry.jsonl"),
        )
        result_path = run_dir / "result.json"
        if result_path.exists():
            payload = json.loads(result_path.read_text(encoding="utf-8"))
            payload["telemetry_summary"] = telemetry_summary
            write_json(result_path, payload)
            write_report(run_dir, payload)
            if result is not None:
                result["telemetry_summary"] = telemetry_summary
        remote.close()
    if result is None:
        raise RuntimeError(f"target run did not produce a result for {target.machine_name}")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build and smoke-test faster-qwen3-tts on remote Jetson machines.")
    parser.add_argument("--targets", default="t4000,orin", help="Comma-separated targets: t4000,orin")
    parser.add_argument("--run-tag", default=time.strftime("smoke-%Y%m%d-%H%M%S"))
    parser.add_argument("--text", default=DEFAULT_TEXT)
    parser.add_argument("--voice", default="vivian")
    parser.add_argument("--asr-base-url", default=os.environ.get("ASR_BASE_URL", "http://192.168.1.230:10001"))
    parser.add_argument("--asr-model", default=os.environ.get("ASR_MODEL", "sensevoice-small"))
    parser.add_argument("--asr-timeout", type=int, default=int(os.environ.get("ASR_TIMEOUT", "300")))
    parser.add_argument("--verify-min-length-ratio", type=float, default=float(os.environ.get("VERIFY_MIN_LENGTH_RATIO", "0.75")))
    parser.add_argument("--verify-min-coverage-ratio", type=float, default=float(os.environ.get("VERIFY_MIN_COVERAGE_RATIO", "0.65")))
    parser.add_argument("--startup-timeout", type=int, default=600)
    parser.add_argument("--hot-request-runs", type=int, default=int(os.environ.get("HOT_REQUEST_RUNS", "2")))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    target_keys = [item.strip().lower() for item in args.targets.split(",") if item.strip()]
    if not target_keys:
        raise SystemExit("no targets selected")
    missing = [item for item in target_keys if item not in TARGETS]
    if missing:
        raise SystemExit(f"unknown targets: {missing}")

    tar_path = build_context_tarball()
    results: list[dict[str, Any]] = []
    try:
        for key in target_keys:
            result = run_target(args, TARGETS[key], tar_path)
            if not result["verification"]["passed"]:
                raise SystemExit(
                    f"{result['machine_name']} ASR verification failed: {result['verification']['reason']}"
                )
            results.append(result)
    finally:
        tar_path.unlink(missing_ok=True)

    if len(results) > 1:
        comparison_dir = ROOT / "benchmarks" / "results" / "comparisons" / "faster_qwen3_tts_smoke" / args.run_tag
        write_comparison(comparison_dir, results)


if __name__ == "__main__":
    main()

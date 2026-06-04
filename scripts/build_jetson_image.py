#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///
from __future__ import annotations

import argparse
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_NAME = "Qwen3-TTS-12Hz-0.6B-Base"
DEFAULT_IMAGE = "faster-qwen3-tts"
DEFAULT_DOCKERFILE = "faster-qwen3-tts-jetson.Dockerfile"
THOR_DOCKERFILE = "faster-qwen3-tts-jetson-thor.Dockerfile"

CONTEXT_PATHS = (
    ".dockerignore",
    "faster-qwen3-tts-requirements.txt",
    "MANIFEST.in",
    "README.md",
    "pyproject.toml",
    "voices.json",
    "voices",
    "examples/openai_server.py",
    "faster_qwen3_tts",
)


@dataclass(frozen=True)
class TargetPreset:
    key: str
    ssh: str
    dockerfile: str
    base_image: str
    pip_index_url: str
    pip_extra_index_url: str
    torchaudio_spec: str


TARGET_PRESETS: dict[str, TargetPreset] = {
    "orin": TargetPreset(
        key="orin",
        ssh="nvidia@lzc-pod-juyIZt.lan",
        dockerfile=DEFAULT_DOCKERFILE,
        base_image="127.0.0.1:5001/x/lzc-aipod-vllm:bffa39b-orin",
        pip_index_url="https://pypi.jetson-ai-lab.io/jp6/cu126/+simple",
        pip_extra_index_url="https://pypi.org/simple",
        torchaudio_spec="torchaudio==2.10.0",
    ),
    "thor": TargetPreset(
        key="thor",
        ssh="nvidia@tegra-ubuntu-t5000.lan",
        dockerfile=THOR_DOCKERFILE,
        base_image="nvcr.io/nvidia/pytorch:25.08-py3",
        pip_index_url="https://pypi.jetson-ai-lab.io/sbsa/cu130/+simple",
        pip_extra_index_url="https://pypi.org/simple",
        torchaudio_spec="torchaudio==2.8.0",
    ),
    "t4000": TargetPreset(
        key="t4000",
        ssh="nvidia@tegra-ubuntu-t4000.lan",
        dockerfile=THOR_DOCKERFILE,
        base_image="registry.lazycat.cloud/x/lzc-aipod-vllm:0.16.0-cu130-thor",
        pip_index_url="https://pypi.jetson-ai-lab.io/sbsa/cu130/+simple",
        pip_extra_index_url="https://pypi.org/simple",
        torchaudio_spec="torchaudio==2.10.0",
    ),
}


@dataclass(frozen=True)
class BuildConfig:
    target: str
    docker_host: str
    dockerfile_path: Path
    dockerfile_context_path: str
    local_model_dir: Path
    model_name: str
    image: str
    tag: str
    image_ref: str
    base_image: str
    pip_index_url: str
    pip_extra_index_url: str
    torchaudio_spec: str
    qwen_tts_mode: str
    default_voice: str
    extra_build_args: tuple[str, ...]
    dry_run: bool


@dataclass(frozen=True)
class BuildContext:
    path: Path
    dockerfile_arg: str
    summary_entries: tuple[str, ...]


Runner = Callable[..., subprocess.CompletedProcess[str]]


def compute_tag(commit_abbr: str, tag_prefix: str | None, tag: str | None = None) -> str:
    if tag is not None:
        tag = tag.strip()
        if not tag:
            raise ValueError("--tag must not be empty")
        return tag
    prefix = (tag_prefix or "").strip().strip("-")
    if not prefix:
        return commit_abbr
    return f"{prefix}-{commit_abbr}"


def git_commit_abbr(repo_root: Path) -> str:
    return (
        subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=repo_root,
            text=True,
        )
        .strip()
    )


def git_is_dirty(repo_root: Path) -> bool:
    status = subprocess.check_output(
        ["git", "status", "--porcelain"],
        cwd=repo_root,
        text=True,
    )
    return bool(status.strip())


def docker_host_from_args(*, docker_host: str | None, ssh: str | None, preset: TargetPreset) -> str:
    if docker_host:
        return docker_host
    ssh_target = ssh or preset.ssh
    if ssh_target.startswith("ssh://"):
        return ssh_target
    return f"ssh://{ssh_target}"


def resolve_existing_path(path_value: str, repo_root: Path, *, kind: str) -> Path:
    path = Path(path_value).expanduser()
    if not path.is_absolute():
        path = repo_root / path
    path = path.resolve()
    if not path.exists():
        raise FileNotFoundError(f"{kind} does not exist: {path}")
    return path


def dockerfile_context_path(dockerfile_path: Path, repo_root: Path) -> str:
    try:
        return dockerfile_path.relative_to(repo_root).as_posix()
    except ValueError:
        return dockerfile_path.name


def parse_build_arg(value: str) -> str:
    if "=" not in value or not value.split("=", 1)[0]:
        raise argparse.ArgumentTypeError("--build-arg must be in KEY=VALUE form")
    return value


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a Jetson Docker image through Docker's ssh:// remote host support.",
    )
    parser.add_argument("--target", choices=sorted(TARGET_PRESETS), default="orin")
    parser.add_argument("--ssh", help="SSH target like nvidia@host. Uses the selected target default when omitted.")
    parser.add_argument("--docker-host", help="Full Docker host URL, for example ssh://nvidia@host.")
    parser.add_argument("--dockerfile", help="Dockerfile to use. Defaults to the selected target preset.")
    parser.add_argument("--model-dir", required=True, help="Local model directory to copy into the build context.")
    parser.add_argument("--model-name", help="Model directory name inside context/models. Defaults to --model-dir basename.")
    parser.add_argument("--image", default=DEFAULT_IMAGE, help="Output image repository/name without tag.")
    parser.add_argument("--tag", help="Exact output image tag. Overrides --tag-prefix and git commit suffixing.")
    parser.add_argument("--tag-prefix", help="Optional prefix; final tag is <prefix>-<git-abbr>.")
    parser.add_argument("--base-image", help="Override the selected target base image.")
    parser.add_argument("--pip-index-url", help="Override the selected target pip index URL.")
    parser.add_argument("--pip-extra-index-url", help="Override the selected target pip extra index URL.")
    parser.add_argument("--torchaudio-spec", help="Override the selected target torchaudio requirement.")
    parser.add_argument("--mode", default="clone", help="QWEN_TTS_MODE build arg.")
    parser.add_argument("--default-voice", default="vivian", help="QWEN_TTS_DEFAULT_VOICE build arg.")
    parser.add_argument("--build-arg", action="append", default=[], type=parse_build_arg, help="Extra Docker build arg.")
    parser.add_argument("--dry-run", action="store_true", help="Print the build command and context summary only.")
    return parser.parse_args(argv)


def build_config_from_args(args: argparse.Namespace, repo_root: Path, commit_abbr: str) -> BuildConfig:
    preset = TARGET_PRESETS[args.target]
    dockerfile_path = resolve_existing_path(args.dockerfile or preset.dockerfile, repo_root, kind="Dockerfile")
    local_model_dir = resolve_existing_path(args.model_dir, repo_root, kind="Model directory")
    if not local_model_dir.is_dir():
        raise NotADirectoryError(f"Model directory is not a directory: {local_model_dir}")

    model_name = args.model_name or local_model_dir.name
    tag = compute_tag(commit_abbr, args.tag_prefix, args.tag)
    image_ref = f"{args.image}:{tag}"
    return BuildConfig(
        target=args.target,
        docker_host=docker_host_from_args(docker_host=args.docker_host, ssh=args.ssh, preset=preset),
        dockerfile_path=dockerfile_path,
        dockerfile_context_path=dockerfile_context_path(dockerfile_path, repo_root),
        local_model_dir=local_model_dir,
        model_name=model_name,
        image=args.image,
        tag=tag,
        image_ref=image_ref,
        base_image=args.base_image or preset.base_image,
        pip_index_url=args.pip_index_url or preset.pip_index_url,
        pip_extra_index_url=args.pip_extra_index_url if args.pip_extra_index_url is not None else preset.pip_extra_index_url,
        torchaudio_spec=args.torchaudio_spec or preset.torchaudio_spec,
        qwen_tts_mode=args.mode,
        default_voice=args.default_voice,
        extra_build_args=tuple(args.build_arg),
        dry_run=args.dry_run,
    )


def copy_into_context(src: Path, dst: Path) -> None:
    if src.is_dir():
        shutil.copytree(src, dst, copy_function=shutil.copy2)
    else:
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def create_build_context(config: BuildConfig, repo_root: Path, *, temp_parent: Path | None = None) -> BuildContext:
    context_dir = Path(tempfile.mkdtemp(prefix="faster-qwen3-tts-jetson-", dir=temp_parent))
    summary: list[str] = []

    for relative in CONTEXT_PATHS:
        src = repo_root / relative
        if not src.exists():
            continue
        copy_into_context(src, context_dir / relative)
        summary.append(relative)

    dockerfile_dst = context_dir / config.dockerfile_context_path
    copy_into_context(config.dockerfile_path, dockerfile_dst)
    summary.append(config.dockerfile_context_path)

    model_dst = context_dir / "models" / config.model_name
    copy_into_context(config.local_model_dir, model_dst)
    summary.append(f"models/{config.model_name}")

    return BuildContext(
        path=context_dir,
        dockerfile_arg=config.dockerfile_context_path,
        summary_entries=tuple(sorted(set(summary))),
    )


def docker_build_args(config: BuildConfig) -> list[str]:
    build_args = [
        f"BASE_IMAGE={config.base_image}",
        f"PIP_INDEX_URL={config.pip_index_url}",
        f"PIP_EXTRA_INDEX_URL={config.pip_extra_index_url}",
        f"TORCHAUDIO_SPEC={config.torchaudio_spec}",
        f"MODEL_NAME={config.model_name}",
        f"MODEL_DIR=/opt/models/{config.model_name}",
        f"QWEN_TTS_MODEL=/opt/models/{config.model_name}",
        f"QWEN_TTS_MODE={config.qwen_tts_mode}",
        f"QWEN_TTS_DEFAULT_VOICE={config.default_voice}",
    ]
    build_args.extend(config.extra_build_args)
    return build_args


def docker_build_command(config: BuildConfig, context: BuildContext) -> list[str]:
    command = [
        "docker",
        "--host",
        config.docker_host,
        "build",
    ]
    for build_arg in docker_build_args(config):
        command.extend(["--build-arg", build_arg])
    command.extend(["-f", context.dockerfile_arg, "-t", config.image_ref, str(context.path)])
    return command


def print_dry_run(config: BuildConfig, context: BuildContext, command: Sequence[str]) -> None:
    print(f"Target: {config.target}")
    print(f"Docker host: {config.docker_host}")
    print(f"Image: {config.image_ref}")
    print(f"Context: {context.path}")
    print("Docker command:")
    print("  " + shlex.join(command))
    print("Context entries:")
    for entry in context.summary_entries:
        print(f"  {entry}")


def run_docker_build(command: Sequence[str], *, runner: Runner) -> None:
    env = os.environ.copy()
    env.setdefault("DOCKER_BUILDKIT", "0")
    runner(command, check=True, env=env, text=True)


def main(
    argv: Sequence[str] | None = None,
    *,
    repo_root: Path = REPO_ROOT,
    runner: Runner = subprocess.run,
) -> int:
    args = parse_args(argv)
    commit_abbr = git_commit_abbr(repo_root)
    if git_is_dirty(repo_root):
        print(
            "warning: git worktree is dirty; image tag still uses current commit "
            f"{commit_abbr}",
            file=sys.stderr,
        )

    config = build_config_from_args(args, repo_root, commit_abbr)
    context: BuildContext | None = None
    try:
        context = create_build_context(config, repo_root)
        command = docker_build_command(config, context)
        if config.dry_run:
            print_dry_run(config, context, command)
            return 0
        run_docker_build(command, runner=runner)
        print(f"Built image on {config.docker_host}: {config.image_ref}")
        return 0
    finally:
        if context is not None:
            shutil.rmtree(context.path, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import importlib.util
import shutil
import sys
from argparse import Namespace
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "build_jetson_image.py"


def load_module():
    spec = importlib.util.spec_from_file_location("build_jetson_image", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def make_model_dir(tmp_path: Path, name: str = "Qwen3-TTS-12Hz-0.6B-Base") -> Path:
    model_dir = tmp_path / name
    model_dir.mkdir()
    (model_dir / "config.json").write_text("{}", encoding="utf-8")
    (model_dir / "weights.bin").write_text("weights", encoding="utf-8")
    return model_dir


def test_compute_tag_with_and_without_prefix():
    module = load_module()

    assert module.compute_tag("abc1234", None) == "abc1234"
    assert module.compute_tag("abc1234", "0.6b-base-clone-openai-orin") == "0.6b-base-clone-openai-orin-abc1234"
    assert module.compute_tag("abc1234", "ignored-prefix", "0.6b-base-clone-openai-thor") == (
        "0.6b-base-clone-openai-thor"
    )


def test_target_presets_and_cli_override_priority(tmp_path):
    module = load_module()
    model_dir = make_model_dir(tmp_path)

    args = module.parse_args(
        [
            "--target",
            "thor",
            "--ssh",
            "nvidia@custom-host",
            "--base-image",
            "example/base:latest",
            "--model-dir",
            str(model_dir),
        ]
    )
    config = module.build_config_from_args(args, REPO_ROOT, "abc1234")

    assert config.docker_host == "ssh://nvidia@custom-host"
    assert config.base_image == "example/base:latest"
    assert config.dockerfile_path == (REPO_ROOT / "faster-qwen3-tts-jetson-thor.Dockerfile").resolve()
    assert config.pip_index_url == "https://pypi.jetson-ai-lab.io/sbsa/cu130/+simple"
    assert config.torchaudio_spec == "torchaudio==2.8.0"


def test_target_defaults_keep_orin_and_thor_separate(tmp_path):
    module = load_module()
    model_dir = make_model_dir(tmp_path)

    orin_args = module.parse_args(["--target", "orin", "--model-dir", str(model_dir)])
    thor_args = module.parse_args(["--target", "thor", "--model-dir", str(model_dir)])
    orin = module.build_config_from_args(orin_args, REPO_ROOT, "abc1234")
    thor = module.build_config_from_args(thor_args, REPO_ROOT, "abc1234")

    assert orin.docker_host == "ssh://nvidia@lzc-pod-juyIZt.lan"
    assert orin.dockerfile_context_path == "faster-qwen3-tts-jetson.Dockerfile"
    assert orin.base_image == "127.0.0.1:5001/x/lzc-aipod-vllm:bffa39b-orin"
    assert orin.pip_index_url == "https://pypi.jetson-ai-lab.io/jp6/cu126/+simple"

    assert thor.docker_host == "ssh://nvidia@tegra-ubuntu-t5000.lan"
    assert thor.dockerfile_context_path == "faster-qwen3-tts-jetson-thor.Dockerfile"
    assert thor.base_image == "nvcr.io/nvidia/pytorch:25.08-py3"
    assert thor.pip_index_url == "https://pypi.jetson-ai-lab.io/sbsa/cu130/+simple"
    assert thor.torchaudio_spec == "torchaudio==2.8.0"


def test_t4000_keeps_explicit_vllm_base(tmp_path):
    module = load_module()
    model_dir = make_model_dir(tmp_path)

    args = module.parse_args(["--target", "t4000", "--model-dir", str(model_dir)])
    config = module.build_config_from_args(args, REPO_ROOT, "abc1234")

    assert config.docker_host == "ssh://nvidia@tegra-ubuntu-t4000.lan"
    assert config.dockerfile_context_path == "faster-qwen3-tts-jetson-thor.Dockerfile"
    assert config.base_image == "registry.lazycat.cloud/x/lzc-aipod-vllm:0.16.0-cu130-thor"
    assert config.torchaudio_spec == "torchaudio==2.10.0"


def test_docker_host_overrides_ssh(tmp_path):
    module = load_module()
    model_dir = make_model_dir(tmp_path)

    args = module.parse_args(
        [
            "--target",
            "orin",
            "--ssh",
            "nvidia@ignored",
            "--docker-host",
            "ssh://nvidia@docker-host",
            "--model-dir",
            str(model_dir),
        ]
    )
    config = module.build_config_from_args(args, REPO_ROOT, "abc1234")

    assert config.docker_host == "ssh://nvidia@docker-host"


def test_dockerfile_path_is_resolved(tmp_path):
    module = load_module()
    model_dir = make_model_dir(tmp_path)

    args = module.parse_args(
        [
            "--dockerfile",
            "faster-qwen3-tts-jetson.Dockerfile",
            "--model-dir",
            str(model_dir),
        ]
    )
    config = module.build_config_from_args(args, REPO_ROOT, "abc1234")

    assert config.dockerfile_path == (REPO_ROOT / "faster-qwen3-tts-jetson.Dockerfile").resolve()
    assert config.dockerfile_context_path == "faster-qwen3-tts-jetson.Dockerfile"


def test_context_places_model_under_models_model_name(tmp_path):
    module = load_module()
    model_dir = make_model_dir(tmp_path, "source-model")
    dockerfile = tmp_path / "Custom.Dockerfile"
    dockerfile.write_text("FROM scratch\n", encoding="utf-8")
    args = Namespace(
        target="orin",
        ssh=None,
        docker_host=None,
        dockerfile=str(dockerfile),
        model_dir=str(model_dir),
        model_name="Qwen3-TTS-12Hz-0.6B-Base",
        image="faster-qwen3-tts",
        tag_prefix=None,
        tag=None,
        base_image=None,
        pip_index_url=None,
        pip_extra_index_url=None,
        torchaudio_spec=None,
        mode="clone",
        default_voice="vivian",
        build_arg=[],
        dry_run=True,
    )
    config = module.build_config_from_args(args, REPO_ROOT, "abc1234")
    context = module.create_build_context(config, REPO_ROOT, temp_parent=tmp_path)

    try:
        assert (context.path / "models" / "Qwen3-TTS-12Hz-0.6B-Base" / "config.json").is_file()
        assert (context.path / "Custom.Dockerfile").is_file()
        assert "models/Qwen3-TTS-12Hz-0.6B-Base" in context.summary_entries
    finally:
        shutil.rmtree(context.path, ignore_errors=True)


def test_dry_run_does_not_call_docker(tmp_path, capsys):
    module = load_module()
    model_dir = make_model_dir(tmp_path)
    calls = []

    def fake_runner(*args, **kwargs):
        calls.append((args, kwargs))

    result = module.main(
        [
            "--target",
            "orin",
            "--model-dir",
            str(model_dir),
            "--tag-prefix",
            "0.6b-base-clone-openai-orin",
            "--dry-run",
        ],
        repo_root=REPO_ROOT,
        runner=fake_runner,
    )

    output = capsys.readouterr().out
    assert result == 0
    assert calls == []
    assert "docker --host ssh://nvidia@lzc-pod-juyIZt.lan build" in output
    assert "faster-qwen3-tts:0.6b-base-clone-openai-orin-" in output
    assert "models/Qwen3-TTS-12Hz-0.6B-Base" in output


def test_exact_tag_overrides_tag_prefix(tmp_path, capsys):
    module = load_module()
    model_dir = make_model_dir(tmp_path)
    calls = []

    def fake_runner(*args, **kwargs):
        calls.append((args, kwargs))

    result = module.main(
        [
            "--target",
            "thor",
            "--model-dir",
            str(model_dir),
            "--tag-prefix",
            "ignored",
            "--tag",
            "0.6b-base-clone-openai-thor",
            "--dry-run",
        ],
        repo_root=REPO_ROOT,
        runner=fake_runner,
    )

    output = capsys.readouterr().out
    assert result == 0
    assert calls == []
    assert "docker --host ssh://nvidia@tegra-ubuntu-t5000.lan build" in output
    assert "-f faster-qwen3-tts-jetson-thor.Dockerfile" in output
    assert "faster-qwen3-tts:0.6b-base-clone-openai-thor" in output
    assert "faster-qwen3-tts:ignored-" not in output


def test_thor_dry_run_uses_pytorch_base_not_vllm(tmp_path, capsys):
    module = load_module()
    model_dir = make_model_dir(tmp_path)
    calls = []

    def fake_runner(*args, **kwargs):
        calls.append((args, kwargs))

    result = module.main(
        [
            "--target",
            "thor",
            "--model-dir",
            str(model_dir),
            "--tag",
            "0.6b-base-clone-openai-pytorch-thor",
            "--dry-run",
        ],
        repo_root=REPO_ROOT,
        runner=fake_runner,
    )

    output = capsys.readouterr().out
    assert result == 0
    assert calls == []
    assert "BASE_IMAGE=nvcr.io/nvidia/pytorch:25.08-py3" in output
    assert "TORCHAUDIO_SPEC=torchaudio==2.8.0" in output
    assert "faster-qwen3-tts:0.6b-base-clone-openai-pytorch-thor" in output
    assert "ghcr.io/nvidia-ai-iot/vllm:latest-jetson-thor" not in output


def test_thor_dockerfile_installs_torch_sensitive_packages_without_deps():
    dockerfile = (REPO_ROOT / "faster-qwen3-tts-jetson-thor.Dockerfile").read_text(encoding="utf-8")

    assert "ARG BASE_IMAGE=nvcr.io/nvidia/pytorch:25.08-py3" in dockerfile
    assert "uv venv \"${APP_VENV}\" --python /usr/bin/python3 --system-site-packages --seed" in dockerfile
    assert "pip install --no-cache-dir --no-deps --force-reinstall \"${TORCHAUDIO_SPEC}\"" in dockerfile
    assert "pip install --no-cache-dir --no-deps \"qwen-tts>=0.1.1\"" in dockerfile
    assert "pip install --no-cache-dir --no-deps --no-build-isolation -e" in dockerfile
    assert "nv25.08" in dockerfile

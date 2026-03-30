#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


DEFAULT_SOURCE = Path("/home/catdog/lazycat-reader/voices")
DEFAULT_OUTPUT = Path(__file__).resolve().parents[1] / "voices.json"
DEFAULT_TARGET_ROOT = "/opt/build/faster-qwen3-tts/voices"


def build_builtin_voices(source_dir: Path, target_root: str) -> dict[str, dict[str, str]]:
    manifest_path = source_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    voices: dict[str, dict[str, str]] = {}

    for item in manifest:
        voice_id = item["voice"]
        voices[voice_id] = {
            "speaker_pt": f"{target_root.rstrip('/')}/{voice_id}.pt",
            "language": "Chinese",
            "instruct": item["instruct"],
            "style": item["style"],
            "nickname": item["nickname"],
            "text": item["text"],
        }

    return voices


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build builtin voices.json from lazycat-reader voices/*.pt",
    )
    parser.add_argument(
        "--source",
        default=str(DEFAULT_SOURCE),
        help=f"Source voices directory (default: {DEFAULT_SOURCE})",
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT),
        help=f"Output voices.json path (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--target-root",
        default=DEFAULT_TARGET_ROOT,
        help=f"speaker_pt path prefix written into voices.json (default: {DEFAULT_TARGET_ROOT})",
    )
    args = parser.parse_args()

    source_dir = Path(args.source).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    voices = build_builtin_voices(source_dir, args.target_root)
    output_path.write_text(
        json.dumps(voices, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()

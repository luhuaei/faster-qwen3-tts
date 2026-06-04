#!/usr/bin/env python3
from __future__ import annotations

import sys

from build_jetson_image import main as build_jetson_main


def main() -> int:
    print(
        "scripts/build_orin_aipod_image.py is deprecated. "
        "Use `uv run scripts/build_jetson_image.py --target orin --model-dir <local-model-dir>`.",
        file=sys.stderr,
    )
    return build_jetson_main(sys.argv[1:])


if __name__ == "__main__":
    raise SystemExit(main())

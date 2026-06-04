#!/usr/bin/env python3
from __future__ import annotations

import sys

from build_jetson_image import main as build_jetson_main


def has_target(argv: list[str]) -> bool:
    return any(arg == "--target" or arg.startswith("--target=") for arg in argv)


def main() -> int:
    argv = sys.argv[1:]
    if not has_target(argv):
        argv = ["--target", "orin", *argv]
    return build_jetson_main(argv)


if __name__ == "__main__":
    raise SystemExit(main())

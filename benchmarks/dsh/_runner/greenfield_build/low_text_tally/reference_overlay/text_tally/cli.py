from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

from .core import tally_utf8


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    args = parser.parse_args(argv)

    try:
        data = args.input.read_bytes()
    except OSError:
        sys.stderr.buffer.write(b"input read failed\n")
        return 1

    try:
        output = tally_utf8(data)
    except UnicodeDecodeError:
        sys.stderr.buffer.write(b"invalid UTF-8 input\n")
        return 2

    sys.stdout.buffer.write(output)
    return 0


__all__ = ["main"]

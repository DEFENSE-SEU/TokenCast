from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

from .analyzer import EvidenceError, analyze, write_artifacts


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Reconcile segmented order evidence")
    parser.add_argument("evidence_directory")
    parser.add_argument("output_directory")
    args = parser.parse_args(argv)
    try:
        analysis = analyze(args.evidence_directory)
        write_artifacts(analysis, args.output_directory)
    except (EvidenceError, OSError) as exc:
        print(str(exc), file=sys.stderr)
        return 2
    return 0

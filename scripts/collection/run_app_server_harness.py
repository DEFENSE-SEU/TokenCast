"""Compatibility entry point for the installed Codex collector."""

from token_prediction.collection.codex_runner import HARNESS_CONFIG, main, run

__all__ = ["HARNESS_CONFIG", "main", "run"]

if __name__ == "__main__":
    raise SystemExit(main())

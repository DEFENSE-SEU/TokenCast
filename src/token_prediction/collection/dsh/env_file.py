from __future__ import annotations

import json
import re
from collections.abc import MutableMapping
from pathlib import Path


_ENVIRONMENT_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\Z")


class EnvironmentFileError(ValueError):
    pass


def parse_environment_file(text: str) -> dict[str, str]:


    values: dict[str, str] = {}
    for line_number, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        name, separator, raw_value = line.partition("=")
        name = name.strip()
        if not separator or _ENVIRONMENT_NAME.fullmatch(name) is None:
            raise EnvironmentFileError(
                f"environment file line {line_number} must be NAME=VALUE"
            )
        if name in values:
            raise EnvironmentFileError(
                f"environment file defines {name!r} more than once"
            )
        value = _parse_value(raw_value.strip(), line_number=line_number)
        if "\x00" in value:
            raise EnvironmentFileError(
                f"environment file line {line_number} contains a NUL byte"
            )
        values[name] = value
    return values


def _parse_value(raw_value: str, *, line_number: int) -> str:
    if not raw_value:
        return ""
    if raw_value.startswith('"'):
        try:
            value = json.loads(raw_value)
        except json.JSONDecodeError as exc:
            raise EnvironmentFileError(
                f"environment file line {line_number} has an invalid double-quoted value"
            ) from exc
        if not isinstance(value, str):
            raise EnvironmentFileError(
                f"environment file line {line_number} must contain a string value"
            )
        return value
    if raw_value.startswith("'"):
        if len(raw_value) < 2 or not raw_value.endswith("'"):
            raise EnvironmentFileError(
                f"environment file line {line_number} has an invalid single-quoted value"
            )
        return raw_value[1:-1]
    if raw_value.endswith(("'", '"')):
        raise EnvironmentFileError(
            f"environment file line {line_number} has an unmatched quote"
        )
    return raw_value


def load_environment_file(
    path: str | Path,
    *,
    environment: MutableMapping[str, str],
) -> tuple[str, ...]:


    source = Path(path)
    try:
        text = source.read_text(encoding="utf-8-sig")
    except OSError as exc:
        raise EnvironmentFileError(f"cannot read environment file {source}") from exc
    values = parse_environment_file(text)
    loaded: list[str] = []
    for name, value in values.items():
        if name in environment:
            continue
        environment[name] = value
        loaded.append(name)
    return tuple(loaded)


__all__ = [
    "EnvironmentFileError",
    "load_environment_file",
    "parse_environment_file",
]

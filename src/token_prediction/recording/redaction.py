from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any

_SECRET_KEYS = {
    "access_token",
    "refresh_token",
    "authorization",
    "api_key",
    "apikey",
    "client_secret",
    "cookie",
    "set-cookie",
}
_BEARER = re.compile(r"(?i)\bBearer\s+[A-Za-z0-9._~+/=-]+")


def redact_text(value: str) -> str:
    return _BEARER.sub("Bearer [REDACTED]", str(value or ""))


def redact_secrets(value: Any) -> Any:
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key, item in value.items():
            key_text = str(key)
            if key_text.lower() in _SECRET_KEYS:
                result[key_text] = "[REDACTED]"
            else:
                result[key_text] = redact_secrets(item)
        return result
    if isinstance(value, list):
        return [redact_secrets(item) for item in value]
    if isinstance(value, tuple):
        return tuple(redact_secrets(item) for item in value)
    if isinstance(value, str):
        return redact_text(value)
    return value

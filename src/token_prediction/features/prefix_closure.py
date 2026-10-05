from __future__ import annotations

import re

from token_prediction.features.text_embedding import hashed_text_embedding

PREFIX_CLOSURE_VERSION_ID = "prefix-closure-v1"
LAST_LINE_EMBEDDING_DIM = 64
LAST_LINE_CHARS = 2000

TOOL_ARGUMENT_KEYS: tuple[str, ...] = (
    "command",
    "description",
    "workdir",
    "timeoutMs",
    "file_path",
    "old_string",
    "new_string",
    "content",
    "status",
    "todos",
)
SHELL_WORDS: tuple[str, ...] = (
    "git",
    "python",
    "pytest",
    "cat",
    "sed",
    "grep",
    "ls",
    "cd",
    "echo",
    "rg",
    "find",
    "pip",
    "Get-Content",
    "Select-String",
    "Get-ChildItem",
    "Set-Content",
    "Remove-Item",
)
CONTINUATION_ENDINGS: tuple[str, ...] = (";", "\\", "|", "&&", "{", "(", ",", "[")

_SHELL_COMMAND = re.compile(
    r"(?:^|[\n;|&(]\s*|\$ErrorActionPreference\s*=\s*'Stop';\s*)("
    + "|".join(re.escape(word) for word in SHELL_WORDS)
    + r")\b",
    re.M,
)
_HEREDOC_OPEN = re.compile(r"<<-?\s*['\"]?(\w+)['\"]?")
_POWERSHELL_HERE_OPEN = re.compile(r"@['\"]\s*\n")
_POWERSHELL_HERE_CLOSE = re.compile(r"\n['\"]@")
_EOF_WORD = re.compile(r"\bEOF\b")
_JSON_ESCAPES = {"n": "\n", "t": "\t", "r": "\r", '"': '"', "\\": "\\", "/": "/"}

PREFIX_CLOSURE_FEATURE_NAMES: tuple[str, ...] = (
    "closure_is_json",
    "closure_json_depth",
    "closure_in_string",
    "closure_top_level_closed",
    "closure_current_value_chars",
    "closure_unmatched_paren",
    "closure_unmatched_bracket",
    "closure_unmatched_brace",
    "closure_double_quote_parity",
    "closure_single_quote_parity",
    "closure_in_fence",
    "closure_fence_count",
    "closure_in_heredoc",
    "closure_eof_count",
    "closure_line_count",
    "closure_shell_line_count",
    "closure_distinct_shell_count",
    "closure_last_line_chars",
    "closure_ends_newline",
    "closure_last_line_continuation",
    "closure_ends_close",
    "closure_ends_double_quote",
    *(f"closure_key_{key}" for key in TOOL_ARGUMENT_KEYS),
    "closure_key_other",
)
PREFIX_LAST_LINE_EMBEDDING = "generation_prefix_last_line_embedding"


def _scan_json(text: str) -> tuple[str, str, int, int, int, str | None, int, int]:


    stripped = text.lstrip()
    if not (stripped.startswith("{") or stripped.startswith("[")):
        return text, text, 0, 0, 0, None, 0, 0
    decoded: list[str] = []
    contents: list[str] = []
    depth = 0
    in_string = False
    top_level_closed = 0
    current_key: str | None = None
    current_value_chars = 0
    buffer: list[str] = []
    index, length = 0, len(text)
    while index < length:
        char = text[index]
        if in_string:
            if char == "\\" and index + 1 < length:
                piece = _JSON_ESCAPES.get(text[index + 1], text[index + 1])
                index += 2
            elif char == '"':
                in_string = False
                index += 1
                after = index
                while after < length and text[after] in " \t\r\n":
                    after += 1
                value = "".join(buffer)
                if after < length and text[after] == ":":
                    current_key = value
                else:
                    contents.append(value)
                    contents.append("\n")
                buffer = []
                decoded.append('"')
                continue
            else:
                piece = char
                index += 1
            buffer.append(piece)
            decoded.append(piece)
            continue
        if char == '"':
            in_string = True
            buffer = []
        elif char in "{[":
            depth += 1
        elif char in "}]":
            depth -= 1
            if depth == 0:
                top_level_closed += 1
        decoded.append(char)
        index += 1
    if in_string:
        value = "".join(buffer)
        contents.append(value)
        current_value_chars = len(value)
    return (
        "".join(decoded),
        "".join(contents),
        depth,
        int(in_string),
        top_level_closed,
        current_key,
        current_value_chars,
        1,
    )


def prefix_closure_features(text: str) -> dict[str, object]:


    decoded, contents, depth, in_string, top_closed, key, value_chars, is_json = _scan_json(text)
    fences = contents.count("```")
    heredoc_tags = _HEREDOC_OPEN.findall(contents)
    content_lines = contents.split("\n")
    closed_heredocs = sum(
        1 for tag in heredoc_tags if any(line.strip() == tag for line in content_lines)
    )
    powershell_open = len(_POWERSHELL_HERE_OPEN.findall(contents))
    powershell_close = len(_POWERSHELL_HERE_CLOSE.findall(contents))
    shell_hits = _SHELL_COMMAND.findall(contents)
    last_line = decoded.rsplit("\n", 1)[-1]
    non_empty = [line for line in decoded.split("\n") if line.strip()]
    last_non_empty = non_empty[-1].rstrip() if non_empty else ""
    features: dict[str, object] = {
        "closure_is_json": is_json,
        "closure_json_depth": depth,
        "closure_in_string": in_string,
        "closure_top_level_closed": top_closed,
        "closure_current_value_chars": value_chars,
        "closure_unmatched_paren": contents.count("(") - contents.count(")"),
        "closure_unmatched_bracket": contents.count("[") - contents.count("]"),
        "closure_unmatched_brace": contents.count("{") - contents.count("}"),
        "closure_double_quote_parity": contents.count('"') % 2,
        "closure_single_quote_parity": contents.count("'") % 2,
        "closure_in_fence": fences % 2,
        "closure_fence_count": fences,
        "closure_in_heredoc": int(
            len(heredoc_tags) > closed_heredocs or powershell_open > powershell_close
        ),
        "closure_eof_count": len(_EOF_WORD.findall(contents)),
        "closure_line_count": decoded.count("\n") + 1,
        "closure_shell_line_count": len(shell_hits),
        "closure_distinct_shell_count": len(set(shell_hits)),
        "closure_last_line_chars": len(last_line),
        "closure_ends_newline": int(decoded.endswith("\n")),
        "closure_last_line_continuation": int(last_non_empty.endswith(CONTINUATION_ENDINGS)),
        "closure_ends_close": int(decoded.endswith(("}", "]"))),
        "closure_ends_double_quote": int(decoded.endswith('"')),
    }
    for argument in TOOL_ARGUMENT_KEYS:
        features[f"closure_key_{argument}"] = int(key == argument)
    features["closure_key_other"] = int(key is not None and key not in TOOL_ARGUMENT_KEYS)
    features[PREFIX_LAST_LINE_EMBEDDING] = hashed_text_embedding(
        last_line[-LAST_LINE_CHARS:], dim=LAST_LINE_EMBEDDING_DIM
    )
    return features


__all__ = [
    "LAST_LINE_EMBEDDING_DIM",
    "PREFIX_CLOSURE_FEATURE_NAMES",
    "PREFIX_CLOSURE_VERSION_ID",
    "PREFIX_LAST_LINE_EMBEDDING",
    "TOOL_ARGUMENT_KEYS",
    "prefix_closure_features",
]

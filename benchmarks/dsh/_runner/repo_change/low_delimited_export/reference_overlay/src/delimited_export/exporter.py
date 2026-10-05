from __future__ import annotations

import csv
import io
from collections.abc import Iterable, Mapping, Sequence


_DELIMITERS = {
    "comma": ",",
    "tab": "\t",
    "pipe": "|",
}


def _resolve_delimiter(value: object) -> str:
    if not isinstance(value, str) or value not in _DELIMITERS:
        raise ValueError(f"unknown delimiter: {value!r}")
    return _DELIMITERS[value]


def export_rows(
    fieldnames: Sequence[str],
    rows: Iterable[Mapping[str, str]],
    *,
    delimiter: str = "comma",
) -> str:


    delimiter_character = _resolve_delimiter(delimiter)
    output = io.StringIO(newline="")
    writer = csv.DictWriter(
        output,
        fieldnames=list(fieldnames),
        delimiter=delimiter_character,
        quotechar='"',
        lineterminator="\n",
        quoting=csv.QUOTE_MINIMAL,
        doublequote=True,
    )
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue()

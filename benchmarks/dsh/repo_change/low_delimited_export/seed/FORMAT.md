# Delimited export format v1

This file freezes the library contract used by the task.

## Input

`fieldnames` is a non-empty sequence of distinct Python `str` values. `rows` is
an iterable of mappings. Each row has exactly the declared field names, and its
values are Python `str` values. The sequence order defines column order. The
iteration order of `rows` defines record order.

The keyword-only `delimiter` argument accepts these exact names:

| Name | CSV delimiter |
| --- | --- |
| `comma` | `,` |
| `tab` | U+0009 TAB |
| `pipe` | `|` |

Every other value raises `ValueError`. Delimiter validation completes before
the first item is requested from `rows`.

## Output

`export_rows` returns one Python `str`. The first record is the header in
`fieldnames` order. The remaining records follow `rows` iteration order. Even
an export with zero data rows contains its header and a final LF.

Encoding follows Python 3.12 `csv.DictWriter` behavior with the selected
single-character delimiter, quote character `"`, `doublequote=True`, and
`quoting=csv.QUOTE_MINIMAL`. A quote character inside a quoted field is written
twice. Unicode characters remain unchanged in the returned string.

Every record terminator is exactly one U+000A LF. This rule is independent of
the host platform. Newline characters inside field values are preserved inside
quoted fields.

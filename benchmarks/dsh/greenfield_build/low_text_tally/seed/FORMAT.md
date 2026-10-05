# Text tally format v1

## Tokenization and counting

Input files contain strict UTF-8 bytes. Decode the complete input before
tokenization. Invalid UTF-8 is invalid input.

Scan the decoded string from left to right. A token is a maximal non-empty run
of original characters whose individual `str.isalnum()` result is true under
Python 3.12. Every other character is a separator. Apply `str.casefold()` to a
completed token, then increment that canonical term. Unicode normalization is
not applied. The returned `count_terms` dictionary has canonical terms inserted
in ascending Python string order.

## Canonical report

The report object has this exact schema:

```json
{"counts":{"term":2},"total_terms":2,"unique_terms":1,"version":1}
```

`counts` maps each canonical term to its positive integer count.
`total_terms` is the sum of all counts, and `unique_terms` is the number of
members in `counts`.

Serialize the complete object with Python 3.12 `json.dumps` using
`ensure_ascii=False`, `sort_keys=True`, and separators `(",", ":")`. Encode it
as UTF-8 and append exactly one U+000A LF byte. `tally_utf8` returns these bytes.

The CLI reads `INPUT.txt` as bytes and writes the `tally_utf8` result directly
to stdout. Success returns 0 with empty stderr. Invalid UTF-8 returns 2 with
empty stdout. Input file I/O failure returns 1 with empty stdout.

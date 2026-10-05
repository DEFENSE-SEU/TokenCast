# Build a Unicode-aware text tally library and CLI

Build the `text_tally` Python package described in `FORMAT.md`. It must expose
these reusable library functions:

```python
count_terms(text: str) -> dict[str, int]
tally_utf8(data: bytes) -> bytes
```

It must also provide this command:

```text
python -m text_tally INPUT.txt
```

Implement the three frozen requirements below. Their identifiers are stable
evaluation labels.

- **[R1] UTF-8 and tokenization:** decode input bytes as strict UTF-8. A token
  is a maximal non-empty run of characters for which Python 3.12
  `str.isalnum()` returns true. Punctuation, symbols, whitespace, combining
  marks, and every other character end a token. Find token boundaries in the
  original text before canonicalizing each token. Invalid UTF-8 must raise
  `UnicodeDecodeError` through `tally_utf8`; the CLI must return 2 with empty
  stdout.
- **[R2] Unicode casefold counting:** canonicalize every complete token with
  `str.casefold()` and count equal canonical strings together. Do not apply a
  Unicode normalization transform. `count_terms` must return its keys in
  Python string order.
- **[R3] canonical library and CLI output:** `tally_utf8` must return the exact
  UTF-8 JSON bytes specified by `FORMAT.md`. Successful CLI stdout must equal
  those library bytes, including one final LF on Windows. Equivalent input
  bytes must produce identical output bytes.

Keep `count_terms` and `tally_utf8` importable from `text_tally`. Use Python
3.12 standard-library code only. Network access is unavailable.

Run the actor-visible tests with:

```text
python -m unittest discover -s tests -v
```

Submit the package source. Test caches and generated output are outside the
deliverable.

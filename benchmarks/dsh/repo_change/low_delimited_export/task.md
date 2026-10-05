# Add selectable delimiters to an exporter

The repository contains a small `delimited_export` package. Its public
`export_rows(fieldnames, rows)` function currently writes comma-delimited CSV.
Extend that API to this exact signature:

```python
export_rows(fieldnames, rows, *, delimiter="comma") -> str
```

Implement the three frozen requirements below. Their identifiers are stable
evaluation labels.

- **[R1] Named delimiters:** accept the exact delimiter names `"comma"`,
  `"tab"`, and `"pipe"`, mapped to `,`, TAB, and `|`. Every other value must
  raise `ValueError`. Validate `delimiter` before requesting any item from
  `rows`.
- **[R2] Default compatibility:** an omitted `delimiter` must retain the
  existing comma-delimited result. Explicit `delimiter="comma"` must produce
  the same result.
- **[R3] CSV behavior:** preserve the caller's field order and row order. Use
  minimal CSV quoting, double embedded quote characters, preserve Unicode text,
  and terminate every record with one LF (`\n`) on every platform. The selected
  delimiter controls which fields require delimiter quoting.

`FORMAT.md` freezes the input domain and exact serialization rules. Keep
`delimited_export.export_rows` as the public import. Use Python 3.12
standard-library code only. Network access is unavailable.

Run the actor-visible tests with:

```text
python -m unittest discover -s tests -v
```

Submit the changed Python source. Generated environments, caches, and test
output files are outside the deliverable.

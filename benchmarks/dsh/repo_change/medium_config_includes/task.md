# Add recursive JSON configuration includes

The repository contains a `config_stack` package whose `load_config` function
loads one JSON object. Extend it to compose configurations through recursive
includes while retaining this public call:

```python
load_config(path, *, root=None)
```

`path` identifies the entry document. When `root` is supplied, it identifies
the configuration-tree boundary. Both are resolved through the local
filesystem. When `root` is omitted, the resolved parent of the entry document
is the boundary. The entry document and every included document must resolve
inside that boundary.

Every configuration document is UTF-8 JSON with an object at the top level.
The optional top-level `$include` member is a list of strings. Each string is a
relative filesystem path interpreted from the directory containing the
document that declares it. Absolute include strings are invalid. The
`$include` control member is removed from the returned configuration; members
with that name inside nested objects remain ordinary data. Invalid top-level
types or include-list shapes raise `ConfigFormatError`.

Implement these five frozen requirements:

- **[R1] relative resolution:** resolve every include from the including
  document's directory, including includes declared by other included files.
- **[R2] include precedence:** process includes from left to right. Merge two
  object values recursively. For every other type pairing, the later value
  replaces the earlier value as a whole.
- **[R3] local precedence:** after all includes for one document have been
  composed, merge that document's remaining local members last. Falsy values
  such as `false`, `null`, zero, empty strings, empty lists, and empty objects
  participate according to the same merge rules.
- **[R4] cycle reporting:** detect cycles through the active recursion stack.
  Raise `IncludeCycleError` with a `chain` attribute containing a tuple of
  root-relative POSIX paths. The tuple starts at the repeated active document,
  follows every include edge in the cycle, and repeats the first path at the
  end. A shared document reached by separate completed branches is a valid
  diamond and may be loaded again.
- **[R5] root containment:** raise `IncludeOutsideRootError` before reading an
  entry or include that resolves outside the boundary. This includes `..`
  traversal, sibling paths that only share a textual prefix, symbolic links to
  outside targets, and every absolute include string. Resolved path containment
  determines the boundary decision.

Keep the public `load_config`, `ConfigFormatError`, `IncludeCycleError`, and
`IncludeOutsideRootError` names exported by `config_stack`. Filesystem and JSON
decoding errors outside these declared cases may retain their standard-library
exception types.

Run the actor-visible tests with:

```text
python -m unittest discover -s tests -v
```

Use Python 3.12 standard-library code. Network access is unavailable. Submit
the changed source code. Test caches and temporary files are outside the
deliverable.

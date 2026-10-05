# Configuration include format v1

## Documents

Each file is UTF-8 JSON whose top-level value is an object. A top-level
`$include` member, when present, is a JSON array containing only strings. It is
composition metadata and does not appear in the loaded result. A nested member
named `$include` is ordinary configuration data.

## Paths and root

`load_config(path, *, root=None)` resolves `path` and `root` with filesystem
semantics. The default root is the resolved entry document's parent directory.
The resolved entry and all resolved includes stay inside the resolved root.

Include strings are relative paths. Their base directory is the parent of the
file containing that include string. An absolute include string raises
`IncludeOutsideRootError`, including one that names a file below the root.
Containment follows resolved filesystem targets, so a symbolic link below the
root that targets an outside file is outside.

## Composition

For each document, includes are loaded in listed order. Their results are
merged from left to right, followed by the document's local members. When the
old and new values for one member are both objects, composition applies the
same rule recursively. Every other new value replaces the old value in full.
JSON values retain their ordinary types, including all falsy values.

## Cycles

Cycle detection tracks resolved documents in the active recursive call stack.
`IncludeCycleError.chain` is a tuple of root-relative paths written with `/`.
It starts with the repeated active document and repeats that path after the
last include edge. A path leaves the active stack when its recursive load
finishes, which permits diamond-shaped include graphs.

Malformed top-level values and malformed `$include` members raise
`ConfigFormatError`. The public exception classes derive from `ValueError`.

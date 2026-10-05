# Repair resumable batch cursor semantics

The `cursor_batch` package reads an input JSONL file in source order and writes
the canonical output rows defined in `FORMAT.md`. Its command is:

```text
python -m cursor_batch INPUT.jsonl OUTPUT.jsonl CHECKPOINT.json
```

Input IDs contain Unicode scalar values, and input values use the signed
64-bit integer range frozen in `FORMAT.md`.

Repair these five behaviors. The identifiers are stable evaluation labels.

1. **[R1] Durable progress:** advance a checkpoint only after its corresponding
   output row has been appended, flushed, and passed through `os.fsync`.
2. **[R2] Duplicate-free resume:** validate the durable output prefix and resume
   after its final complete row. A complete valid output prefix may be ahead of
   the checkpoint after an interruption. Discard an incomplete trailing output
   fragment before continuing. Persist a catch-up checkpoint before appending
   the next output row.
3. **[R3] Deterministic order:** preserve source order in fresh, interrupted,
   and resumed executions. A completed rerun must leave the output unchanged.
4. **[R4] Invalid state:** return exit code 2 before adding output or changing a
   checkpoint when input, checkpoint, or complete output-prefix state violates
   `FORMAT.md`. Other I/O failures return exit code 1.
5. **[R5] Atomic checkpoint:** write and `os.fsync` a temporary file in the
   checkpoint directory, then commit it with `os.replace`. A failed replacement
   must preserve the previous checkpoint.

Keep the command and module surfaces used by the existing tests. In particular,
`cursor_batch.output.append_durable(path, row)` returns only after `row` has
been written, flushed, synchronized, and the output descriptor has been closed.
`cursor_batch.checkpoint.store_checkpoint(path, next_index)` performs its write,
flush, synchronization, close, and replacement through a same-directory
temporary file. Each operation through close uses the descriptor opened for the
corresponding output or temporary file. Any import style and internal call graph
are acceptable when these observable contracts hold.

Run the public tests with:

```text
python -m unittest discover -s tests -v
```

Use Python 3.12 standard-library code. Network access is unavailable.
Submit the repaired source code. Output and checkpoint files created while
running tests are transient execution state and are outside the submission.

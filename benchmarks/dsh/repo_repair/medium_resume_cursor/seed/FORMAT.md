# Cursor batch format v1

This file freezes the data and recovery contract used by the task.

## Input

The input is UTF-8 JSONL. Physical records end with LF or CRLF, including the
final record. Every record is a JSON object with exactly these fields:

- `id`: a non-empty Unicode-scalar string with no isolated UTF-16 surrogate
  code points, unique within the file;
- `value`: a JSON integer from `-9223372036854775808` through
  `9223372036854775807`; booleans are invalid integers.

Blank records, an unterminated final record, invalid UTF-8 or JSON, and schema
violations are invalid input state.

## Output

Input row `i` produces this object:

```json
{"id":"row-id","source_index":0,"square":9}
```

`source_index` is zero-based and `square` is `value * value`. Each object is
encoded with UTF-8, `ensure_ascii=False`, sorted keys, compact separators, and
one final LF. A valid complete output is a contiguous canonical prefix of the
input transformations.

Bytes after the final LF form an incomplete trailing fragment. Recovery may
truncate that fragment. Every LF-terminated row must match its canonical input
transformation exactly. Extra complete rows and mismatched complete rows make
the resume state invalid.

## Checkpoint

An absent checkpoint represents `next_index == 0`. A present checkpoint is a
UTF-8 JSON object with exactly these fields:

```json
{"next_index":1,"version":1}
```

`version` is the integer `1`. `next_index` is an integer in the inclusive range
from zero to the number of input rows. Booleans are invalid. Reading accepts
any UTF-8 JSON object with this exact schema. Writing uses sorted keys, compact
separators, and one final LF.

At startup, the checkpoint may equal or trail the number of valid complete
output rows. A trailing checkpoint is brought forward atomically before the
next output row is processed. A checkpoint ahead of complete output makes the
state invalid.

## Interruption boundaries

The observable boundaries for one item are:

1. append the complete canonical output row;
2. flush and `os.fsync` the output file;
3. write and `os.fsync` a same-directory checkpoint temporary file;
4. commit the checkpoint with `os.replace`.

The append, flush, synchronization, and close in steps 1-2 use the descriptor
opened for that output append, in that order. The write, flush, synchronization,
and close in step 3 use the descriptor opened for the checkpoint temporary file,
in that order. Step 4 follows the successful temporary-file close. Synchronizing
an unrelated descriptor does not make either file durable.

An interruption before a complete LF leaves a removable trailing fragment. An
interruption after output synchronization may leave output ahead of the prior
checkpoint. An interruption before `os.replace` leaves the prior checkpoint
authoritative. Re-running from any of these states produces each item once.

## Exit behavior

The command returns 0 after successful completion. It returns 2 for malformed
input or checkpoint data, an out-of-range cursor, a checkpoint ahead of output,
an extra complete output row, or a mismatched complete output row. Invalid state
is detected before the command adds output or changes the checkpoint. Other I/O
failures return 1.

# DSH task packs

The tracked packs in this directory are the authored P0 protocol-development
cohort described below. Collection scripts also materialize SWE-bench,
MMLU-Pro, and LongBench-v2 packs under `repo_repair/` for local use; those
downloaded packs are excluded from Git.

This directory contains authored Python 3.12 tasks for the protocol-development
P0 cohort. Each task has an exact statement, a seed workspace, public tests,
and a compact machine-readable receipt.

The runner copies only a task's `seed/` directory into a fresh actor workspace.
It sends the exact UTF-8 contents of `task.md` as the user request. The
`_runner/` tree remains outside that workspace and supplies the hidden oracle
plus a reference overlay after the actor session reaches a terminal state.

Each `pack.json` retains a compact asset receipt with task-statement size, seed
file count and size, Python line count, and test data size. The shared verifier
recomputes these values when the task pack is checked. Tracked task-pack path
components use printable ASCII, exact casing, and the portable Windows filename
subset. Task text and file contents retain full UTF-8 and Unicode support.

The shared verifier checks the frozen difficulty evidence, actor-visible file
inventory, expected seed failure, reference solvability, requirement coverage,
and deterministic oracle output. It launches no actor model and makes no
provider request. Oracle time budgets cover a complete fresh-copy evaluation;
the high event-ledger pack uses 30 seconds because its boundary suite launches
many short Python subprocesses on Windows.

P0 supports integration and data-contract diagnosis. Its twelve tasks remain
twelve experimental task units across the four actor conditions. Difficulty is
a preregistered engineering stratum in this cohort. P1 will add independent
families and repeated runs for statistical comparisons.

All twelve packs are verified across the repository-repair, repository-change,
greenfield-build, and investigation-artifact families. The P0 runner now
freezes all 48 run ledgers before workspace creation, materializes fresh actor
copies under short indexed run directories, records terminal states and capture
paths, evaluates isolated workspace copies, and emits the structural
sample-flow report. The preset-aware headless observer preflight passes with a
deterministic offline adapter. The production CLI can execute or resume the 12
local-Qwen rows through the installed headless profile while retaining the full
48-run ledger. The same CLI exposes hosted execution commands for the 36
managed-model rows and requires explicit commercial authorization.

The local workspace split is a cooperative evaluation boundary. A release
benchmark with confidential tests should execute actors on workers where the
`_runner/` tree is absent, then transfer completed terminal workspaces to an
evaluator.

The local verifier copies an oracle into `.dsh_oracle/` inside a temporary
evaluation copy after the actor phase. Candidate code can read that directory
during oracle execution. P0 uses this path for protocol development and makes
no adversarial test-secrecy claim.

## License scope

- Seed code, public tests, oracle code, and reference overlays use the MIT
  License in `LICENSES/MIT.txt`.
- Authored task statements, test data, and expected data artifacts use
  CC0 1.0 Universal under `LICENSES/CC0-1.0-NOTICE.txt` and the linked legal
  code.

These benchmark-scoped terms do not set a license for other repository files.

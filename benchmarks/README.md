# Benchmark layout

`dsh/` and `openhands/` name the agent harnesses used to run tasks. The task
benchmarks used by the research include SWE-bench, Search-R1, MMLU-Pro, and
LongBench-v2. A task's benchmark and its execution harness are separate fields
in a collection record.

`dsh/` contains twelve tracked, authored P0 task packs and their verifier
assets. Collection scripts can also materialize external task packs there from
separately obtained benchmark data; those generated packs are ignored by Git.

`openhands/` contains the OpenHands CodeAct collection entry point and its
execution notes. Raw OpenHands trajectories and downloaded task workspaces live
under ignored `workspace/` directories and are not shipped in this repository.

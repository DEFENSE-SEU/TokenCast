from __future__ import annotations

import os
import subprocess
import sys
from dataclasses import dataclass


WORKER_MARKERS = ("qwen-worker.py",)


OWNER_MARKERS = ("run_dsh", "dsh_runner", "collect_call_update_checkpoints")


class StaleWorkerError(RuntimeError):
    pass


@dataclass(frozen=True)
class ProcessEntry:
    pid: int
    command: str


def _list_processes() -> list[ProcessEntry]:


    if sys.platform != "win32":
        try:
            output = subprocess.run(
                ["ps", "-eo", "pid=,args="],
                capture_output=True,
                text=True,
                errors="replace",
                timeout=60,
            ).stdout
        except (OSError, subprocess.SubprocessError) as exc:
            raise StaleWorkerError(f"cannot read the process table: {exc}") from exc
        entries = []
        for line in output.splitlines():
            head, _, rest = line.strip().partition(" ")
            if head.isdigit():
                entries.append(ProcessEntry(int(head), rest))
        return entries

    script = (
        "Get-CimInstance Win32_Process | "
        "ForEach-Object { \"$($_.ProcessId)`t$($_.CommandLine)\" }"
    )
    try:
        output = subprocess.run(
            ["powershell", "-NoProfile", "-Command", script],
            capture_output=True,
            text=True,
            errors="replace",
            timeout=120,
        ).stdout
    except (OSError, subprocess.SubprocessError) as exc:
        raise StaleWorkerError(f"cannot read the process table: {exc}") from exc
    entries = []
    for line in output.splitlines():
        pid_text, _, command = line.partition("\t")
        pid_text = pid_text.strip()
        if pid_text.isdigit():
            entries.append(ProcessEntry(int(pid_text), command.strip()))
    return entries


def _matches(command: str, markers: tuple[str, ...]) -> bool:
    lowered = command.lower()
    return any(marker.lower() in lowered for marker in markers)


def find_stale_workers(
    entries: list[ProcessEntry] | None = None,
    *,
    worker_markers: tuple[str, ...] = WORKER_MARKERS,
    owner_markers: tuple[str, ...] = OWNER_MARKERS,
) -> tuple[ProcessEntry, ...]:


    table = _list_processes() if entries is None else entries
    workers = [entry for entry in table if _matches(entry.command, worker_markers)]
    if not workers:
        return ()
    owners = [
        entry
        for entry in table
        if _matches(entry.command, owner_markers)
        and not _matches(entry.command, worker_markers)
    ]
    if owners:

        return ()
    return tuple(workers)


def reclaim_stale_workers(
    entries: list[ProcessEntry] | None = None,
    *,
    dry_run: bool = False,
) -> dict[str, object]:


    stale = find_stale_workers(entries)
    reclaimed: list[int] = []
    failed: list[int] = []
    if not dry_run:
        for entry in stale:
            try:
                if sys.platform == "win32":
                    subprocess.run(
                        ["taskkill", "/PID", str(entry.pid), "/F"],
                        capture_output=True,
                        timeout=60,
                    )
                else:
                    os.kill(entry.pid, 9)
                reclaimed.append(entry.pid)
            except (OSError, subprocess.SubprocessError):
                failed.append(entry.pid)
    return {
        "found": [entry.pid for entry in stale],
        "reclaimed": reclaimed,
        "failed": failed,
        "dry_run": dry_run,
    }


__all__ = [
    "OWNER_MARKERS",
    "ProcessEntry",
    "StaleWorkerError",
    "WORKER_MARKERS",
    "find_stale_workers",
    "reclaim_stale_workers",
]

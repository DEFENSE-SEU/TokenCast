"""Process-local, two-step execution of supported Codex task packs."""

from __future__ import annotations

import argparse
import copy
import threading
import time
from dataclasses import dataclass, field
from uuid import uuid4

from token_prediction.codex_run import run_task
from token_prediction.project_run import run_project

CONFIRM_SECONDS = 1800
_ACTIVE = {"preparing", "awaiting_confirmation", "running", "cancelling"}


@dataclass
class _Task:
    args: argparse.Namespace
    id: str = field(default_factory=lambda: uuid4().hex)
    state: str = "preparing"
    prediction: dict | None = None
    collection: dict | None = None
    confirmation_id: str = field(default_factory=lambda: uuid4().hex)
    result: dict | None = None
    expires_at: float | None = None
    deadline: float | None = None
    decision: threading.Event = field(default_factory=threading.Event)
    cancel: threading.Event = field(default_factory=threading.Event)
    worker: threading.Thread | None = None


class TaskRuns:
    def __init__(self, service):
        self.service = service
        self._tasks: dict[str, _Task] = {}
        self._lock = threading.RLock()
        self._closed = False

    def prepare(self, args: argparse.Namespace) -> dict:
        args = copy.copy(args)
        if getattr(args, "project", None) is not None:
            args.project = args.project.resolve()
        else:
            args.pack = args.pack.resolve()
        args.output = args.output.resolve()
        with self._lock:
            if self._closed:
                return {"status": "unavailable", "error": "server is closing"}
            for task in self._tasks.values():
                if task.args.output == args.output:
                    if vars(task.args) != vars(args):
                        return {"status": "conflict", "error": "output belongs to another request"}
                    return self.status(task.id)
            if sum(task.state in _ACTIVE for task in self._tasks.values()) >= 4:
                return {"status": "busy", "error": "finish or cancel an active task first"}
            task = _Task(args)
            self._tasks[task.id] = task
            task.worker = threading.Thread(target=self._run, args=(task,), daemon=True)
            task.worker.start()
            return self.status(task.id)

    def _run(self, task: _Task) -> None:
        def preview(prediction: dict) -> None:
            with self._lock:
                if task.cancel.is_set():
                    raise RuntimeError("task cancelled before execution")
                if prediction.get("mode") == "collect_only":
                    task.collection = copy.deepcopy(prediction)
                else:
                    task.prediction = copy.deepcopy(prediction)
                task.state = "awaiting_confirmation"
                task.expires_at = time.time() + CONFIRM_SECONDS
                task.deadline = time.monotonic() + CONFIRM_SECONDS
            while not task.decision.wait(max(0, task.deadline - time.monotonic())):
                if time.monotonic() >= task.deadline:
                    break
            with self._lock:
                self._expire(task)
                if task.state != "running":
                    raise RuntimeError(f"task {task.state}; no turn started")

        try:
            runner = run_project if getattr(task.args, "project", None) is not None else run_task
            result = runner(self.service, task.args, before_execute=preview, cancel=task.cancel)
        except Exception as exc:
            result = {"status": "run_failed", "error": str(exc)}
        with self._lock:
            task.result = result
            if task.state not in {"cancelled", "expired"}:
                if task.cancel.is_set():
                    task.state = "cancelled"
                else:
                    task.state = "completed" if result["status"] in {"ok", "pending"} else "failed"

    def _expire(self, task: _Task) -> None:
        if (task.state == "awaiting_confirmation"
                and time.monotonic() >= task.deadline):
            task.state = "expired"
            task.cancel.set()
            task.decision.set()

    def status(self, task_id: str) -> dict:
        with self._lock:
            task = self._tasks.get(task_id)
            if task is None:
                return {"status": "unknown_task", "error": "task is unknown or its server restarted"}
            self._expire(task)
            return copy.deepcopy({
                "status": "ok", "task_id": task.id, "state": task.state,
                "output": str(task.args.output), "expires_at": task.expires_at,
                "prediction": task.prediction, "result": task.result,
                "collection": task.collection,
                "confirmation_id": task.confirmation_id if task.collection is not None else None,
            })

    def start(self, task_id: str, prediction_id: str | None = None,
              confirmation_id: str | None = None) -> dict:
        with self._lock:
            value = self.status(task_id)
            if value["status"] != "ok":
                return value
            task = self._tasks[task_id]
            if task.collection is not None:
                if prediction_id is not None or confirmation_id != task.confirmation_id:
                    return {"status": "conflict", "error": "collection confirmation does not match"}
            elif (task.prediction is None or task.prediction["prediction_id"] != prediction_id
                  or confirmation_id is not None):
                return {"status": "conflict", "error": "prediction does not match this task"}
            if task.state == "awaiting_confirmation":
                task.state = "running"
                task.decision.set()
            elif task.state not in {"running", "completed"}:
                return {"status": "invalid_state", "state": task.state,
                        "error": "prepare a new task and show its prediction before execution"}
            return self.status(task_id)

    def cancel(self, task_id: str) -> dict:
        with self._lock:
            value = self.status(task_id)
            if value["status"] != "ok":
                return value
            task = self._tasks[task_id]
            if task.state in _ACTIVE:
                task.state = "cancelling"
                task.cancel.set()
                task.decision.set()
            return self.status(task_id)

    def close(self) -> None:
        with self._lock:
            self._closed = True
            for task in self._tasks.values():
                self.cancel(task.id)
        # All pending gates are now closed. Workers release their own app-server.
        deadline = time.monotonic() + 15
        for task in self._tasks.values():
            task.worker.join(timeout=max(0, deadline - time.monotonic()))

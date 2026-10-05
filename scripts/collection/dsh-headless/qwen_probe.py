from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, TextIO

import torch


PROBE_SCHEMA_VERSION = 2
PROBE_CONTRACT_ID = "dsh-generation-probe-v2"
PROBE_SOURCE_ID = "dsh_session_probe_v2"
PROJECTION_POLICY_ID = "rademacher-last-layer-v2"
PROJECTION_POLICY_VERSION = 2
PROJECTION_DIMENSION = 64
PROJECTION_SEED = 729
CAUSAL_ALIGNMENT = "post_source_output_token_prefix_before_next_sample"
_CHECKPOINT_KEYS = {
    "schema_version",
    "record_type",
    "session_id",
    "attempt_id",
    "checkpoint_ordinal",
    "observer_committed_tokens",
    "source_output_tokens",
    "hidden_state_projection",
}


def projection_policy(hidden_size: int) -> dict[str, object]:
    return {
        "policy_id": PROJECTION_POLICY_ID,
        "policy_version": PROJECTION_POLICY_VERSION,
        "source_layer": "last",
        "dimension": PROJECTION_DIMENSION,
        "seed": PROJECTION_SEED,
        "hidden_size": hidden_size,
        "distribution": "rademacher",
        "scaling": "inverse_sqrt_hidden_size",
        "dtype": "float32",
        "rng_algorithm": "torch-cpu-generator-v1",
        "fit_scope": "fixed_without_dataset_fit",
    }


def build_projection_matrix(hidden_size: int, device: Any) -> torch.Tensor:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(PROJECTION_SEED)
    matrix = torch.randint(
        0,
        2,
        (hidden_size, PROJECTION_DIMENSION),
        generator=generator,
        dtype=torch.int8,
    )
    matrix = matrix.to(dtype=torch.float32).mul_(2).sub_(1)
    matrix.div_(math.sqrt(hidden_size))
    return matrix.to(device)


def project_hidden_state(hidden_state: torch.Tensor, matrix: torch.Tensor) -> tuple[float, ...]:
    projected = hidden_state.float() @ matrix
    return tuple(float(value) for value in projected.detach().cpu().tolist())


class GenerationProbeWriter:
    def __init__(self, *, model_revision: str, projection: Mapping[str, object]) -> None:
        self.model_revision = model_revision
        self.projection = dict(projection)
        self.path: Path | None = None
        self.header: dict[str, object] | None = None
        self.stream: TextIO | None = None
        self.active = False

    def begin(self, config: Mapping[str, Any] | None) -> None:
        if config is None:
            self.active = False
            return
        path = Path(str(config["path"]))
        header: dict[str, object] = {
            "schema_version": PROBE_SCHEMA_VERSION,
            "record_type": "generation_probe/session",
            "contract_id": PROBE_CONTRACT_ID,
            "capture_revision": str(config["capture_revision"]),
            "source_id": PROBE_SOURCE_ID,
            "session_id": str(config["session_id"]),
            "model_id": str(config["model_id"]),
            "model_revision": self.model_revision,
            "projection": self.projection,
            "causal_alignment": CAUSAL_ALIGNMENT,
        }
        if self.path == path:
            if self.header != header:
                raise RuntimeError("Generation Probe session metadata changed within one file")
            self.active = True
            return
        self.close()
        path.parent.mkdir(parents=True, exist_ok=True)
        existing = path.exists()
        if existing:
            self._validate_existing(path, header)
        self.stream = path.open("a" if existing else "x", encoding="utf-8", newline="\n")
        self.path = path
        self.header = header
        self.active = True
        if not existing:
            self._write(header)

    def record(
        self,
        acknowledgement: Mapping[str, Any],
        hidden_state_projection: Sequence[float],
        *,
        source_output_tokens: int,
    ) -> None:
        if not self.active:
            return
        if type(source_output_tokens) is not int or source_output_tokens < 1:
            raise RuntimeError("Generation Probe source_output_tokens must be a positive integer")
        projection = [float(value) for value in hidden_state_projection]
        for checkpoint in acknowledgement["generation_probe_checkpoints"]:
            self._write(
                {
                    "schema_version": PROBE_SCHEMA_VERSION,
                    "record_type": "generation_probe/checkpoint",
                    "session_id": checkpoint["session_id"],
                    "attempt_id": checkpoint["attempt_id"],
                    "checkpoint_ordinal": checkpoint["checkpoint_ordinal"],
                    "observer_committed_tokens": checkpoint["observer_committed_tokens"],
                    "source_output_tokens": source_output_tokens,
                    "hidden_state_projection": projection,
                }
            )

    def _write(self, record: Mapping[str, object]) -> None:
        if self.stream is None:
            raise RuntimeError("Generation Probe writer has no open session")
        self.stream.write(
            json.dumps(record, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n"
        )
        self.stream.flush()

    @staticmethod
    def _validate_existing(path: Path, header: Mapping[str, object]) -> None:
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeError) as error:
            raise RuntimeError(f"cannot resume Generation Probe sidecar: {error}") from error
        if not text.endswith("\n"):
            raise RuntimeError("cannot resume an incomplete Generation Probe sidecar")
        lines = text[:-1].split("\n")
        if not lines or any(not line for line in lines):
            raise RuntimeError("cannot resume a malformed Generation Probe sidecar")
        try:
            rows = [json.loads(line) for line in lines]
        except json.JSONDecodeError as error:
            raise RuntimeError(f"cannot resume Generation Probe sidecar: {error}") from error
        expected_header = json.dumps(
            header,
            ensure_ascii=False,
            separators=(",", ":"),
            allow_nan=False,
        )
        if lines[0] != expected_header:
            raise RuntimeError("Generation Probe session metadata changed within one file")
        seen: set[tuple[str, str, int, int]] = set()
        previous_checkpoint_by_attempt: dict[str, tuple[int, int, int]] = {}
        for line, row in zip(lines[1:], rows[1:], strict=True):
            if not isinstance(row, dict) or set(row) != _CHECKPOINT_KEYS:
                raise RuntimeError("cannot resume a malformed Generation Probe checkpoint")
            try:
                canonical = json.dumps(
                    row,
                    ensure_ascii=False,
                    separators=(",", ":"),
                    allow_nan=False,
                )
            except (TypeError, ValueError) as error:
                raise RuntimeError("cannot resume a malformed Generation Probe checkpoint") from error
            if line != canonical:
                raise RuntimeError("cannot resume a malformed Generation Probe checkpoint")
            projection = row["hidden_state_projection"]
            key = (
                row["session_id"],
                row["attempt_id"],
                row["checkpoint_ordinal"],
                row["observer_committed_tokens"],
            )
            if (
                type(row["schema_version"]) is not int
                or row["schema_version"] != PROBE_SCHEMA_VERSION
                or row["record_type"] != "generation_probe/checkpoint"
                or row["session_id"] != header["session_id"]
                or not isinstance(row["attempt_id"], str)
                or not row["attempt_id"]
                or type(row["checkpoint_ordinal"]) is not int
                or row["checkpoint_ordinal"] < 1
                or type(row["observer_committed_tokens"]) is not int
                or row["observer_committed_tokens"] < 1
                or type(row["source_output_tokens"]) is not int
                or row["source_output_tokens"] < 1
                or not isinstance(projection, list)
                or len(projection) != PROJECTION_DIMENSION
                or any(
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not math.isfinite(value)
                    for value in projection
                )
                or key in seen
            ):
                raise RuntimeError("cannot resume a malformed Generation Probe checkpoint")
            previous = previous_checkpoint_by_attempt.get(row["attempt_id"])
            current = (
                row["checkpoint_ordinal"],
                row["observer_committed_tokens"],
                row["source_output_tokens"],
            )
            expected_ordinal = 1 if previous is None else previous[0] + 1
            if current[0] != expected_ordinal or (
                previous is not None
                and (current[1] <= previous[1] or current[2] < previous[2])
            ):
                raise RuntimeError("cannot resume a malformed Generation Probe checkpoint")
            seen.add(key)
            previous_checkpoint_by_attempt[row["attempt_id"]] = current

    def close(self) -> None:
        if self.stream is not None:
            self.stream.close()
        self.path = None
        self.header = None
        self.stream = None
        self.active = False


__all__ = [
    "CAUSAL_ALIGNMENT",
    "GenerationProbeWriter",
    "PROBE_SCHEMA_VERSION",
    "PROBE_CONTRACT_ID",
    "PROBE_SOURCE_ID",
    "PROJECTION_DIMENSION",
    "PROJECTION_POLICY_ID",
    "PROJECTION_POLICY_VERSION",
    "PROJECTION_SEED",
    "build_projection_matrix",
    "project_hidden_state",
    "projection_policy",
]

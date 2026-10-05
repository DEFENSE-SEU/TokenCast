from __future__ import annotations

import math
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from token_prediction.collection.readers.dsh_session import (
    DshSessionMetadata,
    DshSessionReader,
    validate_dsh_observer_sidecar,
)
from token_prediction.collection.readers.json_stream import JsonStreamError, strict_json_loads
from token_prediction.contracts import CanonicalEvent, EventType, Observable, SourceCapabilities
from token_prediction.trajectory import Trajectory


class DshGenerationProbeSchemaError(ValueError):
    pass


@dataclass(frozen=True)
class GenerationProbeMetadata:


    dsh: DshSessionMetadata = field(default_factory=DshSessionMetadata)
    sidecar_name: str = "generation-probe.jsonl"
    schema_version: int = 2
    contract_id: str = "dsh-generation-probe-v2"
    source_id: str = "dsh_session_probe_v2"
    projection_policy_id: str = "rademacher-last-layer-v2"
    projection_policy_version: int = 2
    projection_source_layer: str = "last"
    projection_dimension: int = 64
    projection_seed: int = 729
    projection_distribution: str = "rademacher"
    projection_scaling: str = "inverse_sqrt_hidden_size"
    projection_dtype: str = "float32"
    projection_rng_algorithm: str = "torch-cpu-generator-v1"
    projection_fit_scope: str = "fixed_without_dataset_fit"
    causal_alignment: str = "post_source_output_token_prefix_before_next_sample"
    stop_signal_policy_id: str | None = None
    stop_signal_window_tokens: int | None = None
    stop_signal_aggregation: str | None = None

    @property
    def carries_stop_signals(self) -> bool:
        return self.stop_signal_policy_id is not None


STOP_PROBABILITY_FEATURE = "stop_prob_mean_16"
STOP_ENTROPY_FEATURE = "next_token_entropy_mean_16"
STOP_SIGNAL_FEATURES = (STOP_PROBABILITY_FEATURE, STOP_ENTROPY_FEATURE)


def generation_probe_v3_metadata(
    dsh: DshSessionMetadata | None = None,
) -> GenerationProbeMetadata:


    return GenerationProbeMetadata(
        dsh=dsh if dsh is not None else DshSessionMetadata(),
        schema_version=3,
        contract_id="dsh-generation-probe-v3",
        source_id="dsh_session_probe_v3",
        stop_signal_policy_id="eos-entropy-window-16-v1",
        stop_signal_window_tokens=16,
        stop_signal_aggregation="mean_over_window",
    )


_SESSION_KEYS = {
    "schema_version",
    "record_type",
    "contract_id",
    "capture_revision",
    "source_id",
    "session_id",
    "model_id",
    "model_revision",
    "projection",
    "causal_alignment",
}
_PROJECTION_KEYS = {
    "policy_id",
    "policy_version",
    "source_layer",
    "dimension",
    "seed",
    "hidden_size",
    "distribution",
    "scaling",
    "dtype",
    "rng_algorithm",
    "fit_scope",
}
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
_STOP_SIGNAL_KEYS = {
    "policy_id",
    "window_tokens",
    "aggregation",
    "stop_token",
}
_FORBIDDEN_FIELDS = {
    "label",
    "target",
    "final_output_tokens",
    "remaining_tokens",
    "task_terminal",
}


ProbeJoinKey = tuple[str, str, int, int]
StopSignals = tuple[float, float]


def _session_keys(metadata: GenerationProbeMetadata) -> set[str]:
    if metadata.carries_stop_signals:
        return _SESSION_KEYS | {"stop_signal"}
    return _SESSION_KEYS


def _checkpoint_keys(metadata: GenerationProbeMetadata) -> set[str]:
    if metadata.carries_stop_signals:
        return _CHECKPOINT_KEYS | set(STOP_SIGNAL_FEATURES)
    return _CHECKPOINT_KEYS


def validate_dsh_generation_probe_sidecar(
    path: str | Path,
    observer_path: str | Path,
    session_id: str,
    model_id: str,
    model_revision: str,
    metadata: GenerationProbeMetadata | None = None,
) -> None:


    names = metadata or GenerationProbeMetadata()
    _validate_metadata(names)
    projections, source_output_tokens, _stop_signals, provenance = _read_probe_sidecar(
        Path(path),
        session_id,
        model_id,
        names,
    )
    expected_revision = _text(model_revision, "expected Generation Probe model_revision")
    if provenance["model_revision"] != expected_revision:
        raise DshGenerationProbeSchemaError("Generation Probe and expected model revisions differ")

    observer_facts = validate_dsh_observer_sidecar(
        observer_path,
        session_id,
        require_terminal=True,
    )
    _validate_checkpoint_join(set(projections), set(observer_facts.checkpoint_keys))
    _validate_source_output_token_values(
        source_output_tokens,
        dict(observer_facts.terminal_output_tokens),
    )


class DshGenerationProbeReader:


    source_id = "dsh_session_probe_v2"
    capabilities = SourceCapabilities(
        source_id=source_id,
        observables=(DshSessionReader.capabilities.observables | {Observable.HIDDEN_STATE}),
        source="declared",
    )

    def default_metadata(self) -> GenerationProbeMetadata:
        return GenerationProbeMetadata()

    def read(
        self,
        location: str | Path,
        metadata: GenerationProbeMetadata | None = None,
        *,
        base_trajectory: Trajectory | None = None,
    ) -> Trajectory:
        root = Path(location).resolve()
        names = metadata if metadata is not None else self.default_metadata()
        _validate_metadata(names)
        trajectory = (
            DshSessionReader().read(root, names.dsh) if base_trajectory is None else base_trajectory
        )
        capture_id, session_id = _capture_identity(root / names.dsh.manifest_name)
        if trajectory.trajectory_id != f"dsh:{capture_id}:{session_id}":
            raise DshGenerationProbeSchemaError(
                "base trajectory identity differs from the Generation Probe capture"
            )
        model_id = _trajectory_model_id(trajectory)
        projections, source_output_tokens, stop_signals, provenance = _read_probe_sidecar(
            root / names.sidecar_name,
            session_id,
            model_id,
            names,
        )
        _validate_launch_model_revision(trajectory, str(provenance["model_revision"]))
        condition_id = _text(
            trajectory.events[0].payload.get("condition_id"),
            "DSH launch condition_id",
        )
        provenance["condition_id"] = condition_id
        canonical = _canonical_checkpoint_keys(trajectory, session_id)

        # Require an exact checkpoint join before attaching sidecar features to the prefix.
        _validate_checkpoint_join(set(projections), set(canonical))
        _validate_source_output_tokens(trajectory, source_output_tokens)

        events: list[CanonicalEvent] = []
        for event in trajectory.events:
            if event.event_type == EventType.TASK_STARTED:
                payload = event.payload
                if "generation_probe_provenance" in payload:
                    raise DshGenerationProbeSchemaError(
                        "DSH launch already contains Generation Probe provenance"
                    )
                payload["generation_probe_provenance"] = provenance
                events.append(event.with_payload(payload))
                continue
            if event.event_type != EventType.GENERATION_CHECKPOINT:
                events.append(event)
                continue
            key = _canonical_join_key(event, session_id)
            payload = event.payload
            if "hidden_state_projection" in payload:
                raise DshGenerationProbeSchemaError(
                    "canonical checkpoint already contains hidden_state_projection"
                )
            if "source_output_tokens" in payload:
                raise DshGenerationProbeSchemaError(
                    "canonical checkpoint already contains source_output_tokens"
                )
            payload["hidden_state_projection"] = list(projections[key])
            payload["source_output_tokens"] = source_output_tokens[key]
            if names.carries_stop_signals:
                for name in STOP_SIGNAL_FEATURES:
                    if name in payload:
                        raise DshGenerationProbeSchemaError(
                            f"canonical checkpoint already contains {name}"
                        )
                stop_probability, stop_entropy = stop_signals[key]
                payload[STOP_PROBABILITY_FEATURE] = stop_probability
                payload[STOP_ENTROPY_FEATURE] = stop_entropy
            events.append(event.with_payload(payload))
        return Trajectory.from_events(events)


class DshGenerationProbeV3Reader(DshGenerationProbeReader):


    source_id = "dsh_session_probe_v3"
    capabilities = SourceCapabilities(
        source_id=source_id,
        observables=(
            DshSessionReader.capabilities.observables
            | {Observable.HIDDEN_STATE, Observable.LOGPROBS}
        ),
        source="declared",
    )

    def default_metadata(self) -> GenerationProbeMetadata:
        return generation_probe_v3_metadata()


def _validate_metadata(metadata: GenerationProbeMetadata) -> None:
    known = (
        GenerationProbeMetadata(dsh=metadata.dsh),
        generation_probe_v3_metadata(metadata.dsh),
    )
    if metadata not in known:
        raise DshGenerationProbeSchemaError("Generation Probe metadata contract is fixed")


def _capture_identity(path: Path) -> tuple[str, str]:
    try:
        text = path.read_text(encoding="utf-8")
        value = strict_json_loads(text, context="DSH capture manifest")
    except (OSError, UnicodeError, JsonStreamError) as exc:
        raise DshGenerationProbeSchemaError(f"cannot read DSH capture manifest: {exc}") from exc
    manifest = _object(value, "DSH capture manifest")
    return (
        _text(manifest.get("capture_id"), "DSH capture capture_id"),
        _text(manifest.get("session_id"), "DSH capture session_id"),
    )


def _iter_jsonl(path: Path) -> Iterator[tuple[int, Mapping[str, Any]]]:
    try:
        with path.open("r", encoding="utf-8", newline="") as stream:
            for line_number, line in enumerate(stream, start=1):
                if not line.endswith("\n"):
                    raise DshGenerationProbeSchemaError(
                        f"{path.name} has an incomplete final record"
                    )
                try:
                    value = strict_json_loads(
                        line,
                        context=f"{path.name} line {line_number}",
                    )
                except JsonStreamError as exc:
                    raise DshGenerationProbeSchemaError(str(exc)) from exc
                yield line_number, _object(value, f"{path.name} line {line_number}")
    except DshGenerationProbeSchemaError:
        raise
    except (OSError, UnicodeError) as exc:
        raise DshGenerationProbeSchemaError(f"cannot read {path.name}: {exc}") from exc


def _read_probe_sidecar(
    path: Path,
    session_id: str,
    model_id: str,
    metadata: GenerationProbeMetadata,
) -> tuple[
    dict[ProbeJoinKey, tuple[float, ...]],
    dict[ProbeJoinKey, int],
    dict[ProbeJoinKey, StopSignals],
    dict[str, object],
]:
    rows = _iter_jsonl(path)
    try:
        line_number, header = next(rows)
    except StopIteration as exc:
        raise DshGenerationProbeSchemaError("Generation Probe sidecar is empty") from exc
    _reject_forbidden_fields(header, f"Generation Probe line {line_number}")
    _exact_keys(header, _session_keys(metadata), "Generation Probe session header")
    if header["schema_version"] != metadata.schema_version or isinstance(
        header["schema_version"], bool
    ):
        raise DshGenerationProbeSchemaError(
            f"Generation Probe schema_version must be {metadata.schema_version}"
        )
    if header["record_type"] != "generation_probe/session":
        raise DshGenerationProbeSchemaError("Generation Probe session header must appear first")
    if header["contract_id"] != metadata.contract_id:
        raise DshGenerationProbeSchemaError("Generation Probe contract_id is unsupported")
    capture_revision = _text(
        header["capture_revision"],
        "Generation Probe capture_revision",
    )
    if header["source_id"] != metadata.source_id:
        raise DshGenerationProbeSchemaError("Generation Probe source_id is unsupported")
    if header["session_id"] != session_id:
        raise DshGenerationProbeSchemaError("Generation Probe and DSH session ids differ")
    sidecar_model_id = _text(header["model_id"], "Generation Probe model_id")
    if sidecar_model_id != model_id:
        raise DshGenerationProbeSchemaError("Generation Probe and DSH model ids differ")
    model_revision = _text(header["model_revision"], "Generation Probe model_revision")
    causal_alignment = _text(
        header["causal_alignment"],
        "Generation Probe causal_alignment",
    )
    if causal_alignment != metadata.causal_alignment:
        raise DshGenerationProbeSchemaError("Generation Probe causal_alignment is unsupported")
    projection_policy = _validate_projection_policy(header["projection"], metadata)
    provenance: dict[str, object] = {
        "capture_revision": capture_revision,
        "model_id": sidecar_model_id,
        "model_revision": model_revision,
        "projection_policy": projection_policy,
        "causal_alignment": causal_alignment,
    }
    if metadata.carries_stop_signals:
        provenance["stop_signal_policy"] = _validate_stop_signal_policy(
            header["stop_signal"], metadata
        )

    projections: dict[ProbeJoinKey, tuple[float, ...]] = {}
    source_output_tokens: dict[ProbeJoinKey, int] = {}
    stop_signals: dict[ProbeJoinKey, StopSignals] = {}
    for line_number, row in rows:
        label = f"Generation Probe checkpoint at line {line_number}"
        _reject_forbidden_fields(row, label)
        if row.get("record_type") != "generation_probe/checkpoint":
            raise DshGenerationProbeSchemaError(
                "only Generation Probe checkpoint records may follow the session header"
            )
        _exact_keys(row, _checkpoint_keys(metadata), label)
        if row["schema_version"] != metadata.schema_version or isinstance(
            row["schema_version"], bool
        ):
            raise DshGenerationProbeSchemaError(
                "Generation Probe checkpoint schema_version must be "
                f"{metadata.schema_version}"
            )
        key = _sidecar_join_key(row, session_id)
        if key in projections:
            raise DshGenerationProbeSchemaError("duplicate Generation Probe checkpoint join key")
        projections[key] = _projection_vector(
            row["hidden_state_projection"],
            metadata.projection_dimension,
        )
        source_output_tokens[key] = _positive_int(
            row["source_output_tokens"],
            "Generation Probe source_output_tokens",
        )
        if metadata.carries_stop_signals:
            stop_signals[key] = (
                _probability(
                    row[STOP_PROBABILITY_FEATURE],
                    f"Generation Probe {STOP_PROBABILITY_FEATURE}",
                ),
                _non_negative_float(
                    row[STOP_ENTROPY_FEATURE],
                    f"Generation Probe {STOP_ENTROPY_FEATURE}",
                ),
            )
    return projections, source_output_tokens, stop_signals, provenance


def _validate_stop_signal_policy(
    value: object,
    metadata: GenerationProbeMetadata,
) -> dict[str, object]:


    policy = _object(value, "Generation Probe stop_signal")
    _exact_keys(policy, _STOP_SIGNAL_KEYS, "Generation Probe stop_signal")
    policy_id = _text(policy["policy_id"], "Generation Probe stop_signal policy_id")
    if policy_id != metadata.stop_signal_policy_id:
        raise DshGenerationProbeSchemaError("Generation Probe stop_signal policy is unsupported")
    window = _positive_int(
        policy["window_tokens"],
        "Generation Probe stop_signal window_tokens",
    )
    if window != metadata.stop_signal_window_tokens:
        raise DshGenerationProbeSchemaError("Generation Probe stop_signal window is unsupported")
    aggregation = _text(
        policy["aggregation"],
        "Generation Probe stop_signal aggregation",
    )
    if aggregation != metadata.stop_signal_aggregation:
        raise DshGenerationProbeSchemaError(
            "Generation Probe stop_signal aggregation is unsupported"
        )
    stop_token = _text(policy["stop_token"], "Generation Probe stop_signal stop_token")
    return {
        "policy_id": policy_id,
        "window_tokens": window,
        "aggregation": aggregation,
        "stop_token": stop_token,
    }


def _validate_source_output_tokens(
    trajectory: Trajectory,
    source_output_tokens: Mapping[ProbeJoinKey, int],
) -> None:
    terminal_output_tokens: dict[str, int] = {}
    for event in trajectory.events:
        if event.event_type not in {EventType.API_COMPLETED, EventType.API_FAILED}:
            continue
        if event.attempt_id is None:
            continue
        usage = event.payload.get("usage")
        if not isinstance(usage, Mapping):
            continue
        output_tokens = usage.get("output_tokens")
        if (
            isinstance(output_tokens, int)
            and not isinstance(output_tokens, bool)
            and output_tokens >= 0
        ):
            terminal_output_tokens[event.attempt_id] = output_tokens

    _validate_source_output_token_values(source_output_tokens, terminal_output_tokens)


def _validate_source_output_token_values(
    source_output_tokens: Mapping[ProbeJoinKey, int],
    terminal_output_tokens: Mapping[str, int],
) -> None:
    by_attempt: dict[str, list[tuple[int, int]]] = {}
    for (
        _session_id,
        attempt_id,
        checkpoint_ordinal,
        _committed,
    ), value in source_output_tokens.items():
        by_attempt.setdefault(attempt_id, []).append((checkpoint_ordinal, value))

    for attempt_id, checkpoints in by_attempt.items():
        ordered = sorted(checkpoints)
        if any(current < previous for (_, previous), (_, current) in zip(ordered, ordered[1:])):
            raise DshGenerationProbeSchemaError(
                "Generation Probe source_output_tokens decreased within one Attempt"
            )
        if attempt_id not in terminal_output_tokens:
            raise DshGenerationProbeSchemaError(
                "Generation Probe checkpoint has no terminal provider output_tokens bound"
            )
        terminal = terminal_output_tokens[attempt_id]
        if any(value > terminal for _, value in ordered):
            raise DshGenerationProbeSchemaError(
                "Generation Probe source_output_tokens exceeds terminal provider output_tokens"
            )


def _validate_checkpoint_join(
    observed: set[ProbeJoinKey],
    expected: set[ProbeJoinKey],
) -> None:
    missing = expected - observed
    extra = observed - expected
    if missing or extra:
        raise DshGenerationProbeSchemaError(
            "Generation Probe checkpoints do not form a one-to-one join "
            f"with expected checkpoints (missing={len(missing)}, extra={len(extra)})"
        )


def _trajectory_model_id(trajectory: Trajectory) -> str:
    model_ids = {
        _text(event.payload.get("model"), "DSH Request model")
        for event in trajectory.events
        if event.event_type == EventType.REQUEST_BUILT
    }
    if len(model_ids) != 1:
        raise DshGenerationProbeSchemaError(
            "Generation Probe requires one resolved model per DSH session"
        )
    return next(iter(model_ids))


def _validate_projection_policy(
    value: object,
    metadata: GenerationProbeMetadata,
) -> dict[str, object]:
    projection = _object(value, "Generation Probe projection policy")
    _reject_forbidden_fields(projection, "Generation Probe projection policy")
    _exact_keys(projection, _PROJECTION_KEYS, "Generation Probe projection policy")
    resolved: dict[str, object] = {
        "policy_id": _text(projection["policy_id"], "Generation Probe projection policy_id"),
        "policy_version": _positive_int(
            projection["policy_version"],
            "Generation Probe projection policy_version",
        ),
        "source_layer": _text(
            projection["source_layer"],
            "Generation Probe projection source_layer",
        ),
        "dimension": _positive_int(
            projection["dimension"],
            "Generation Probe projection dimension",
        ),
        "seed": _non_negative_int(
            projection["seed"],
            "Generation Probe projection seed",
        ),
        "hidden_size": _positive_int(
            projection["hidden_size"],
            "Generation Probe projection hidden_size",
        ),
        "distribution": _text(
            projection["distribution"],
            "Generation Probe projection distribution",
        ),
        "scaling": _text(
            projection["scaling"],
            "Generation Probe projection scaling",
        ),
        "dtype": _text(projection["dtype"], "Generation Probe projection dtype"),
        "rng_algorithm": _text(
            projection["rng_algorithm"],
            "Generation Probe projection rng_algorithm",
        ),
        "fit_scope": _text(
            projection["fit_scope"],
            "Generation Probe projection fit_scope",
        ),
    }
    expected = {
        "policy_id": metadata.projection_policy_id,
        "policy_version": metadata.projection_policy_version,
        "source_layer": metadata.projection_source_layer,
        "dimension": metadata.projection_dimension,
        "seed": metadata.projection_seed,
        "distribution": metadata.projection_distribution,
        "scaling": metadata.projection_scaling,
        "dtype": metadata.projection_dtype,
        "rng_algorithm": metadata.projection_rng_algorithm,
        "fit_scope": metadata.projection_fit_scope,
    }
    if any(resolved[name] != expected[name] for name in expected):
        raise DshGenerationProbeSchemaError("Generation Probe projection policy drifted")
    return resolved


def _validate_launch_model_revision(trajectory: Trajectory, model_revision: str) -> None:
    launch_revision = trajectory.events[0].payload.get("model_revision")
    if launch_revision is None:
        raise DshGenerationProbeSchemaError("Generation Probe requires DSH launch model_revision")
    if _text(launch_revision, "DSH launch model_revision") != model_revision:
        raise DshGenerationProbeSchemaError(
            "Generation Probe and DSH launch model revisions differ"
        )


def _canonical_checkpoint_keys(
    trajectory: Trajectory,
    session_id: str,
) -> dict[ProbeJoinKey, CanonicalEvent]:
    checkpoints: dict[ProbeJoinKey, CanonicalEvent] = {}
    for event in trajectory.events:
        if event.event_type != EventType.GENERATION_CHECKPOINT:
            continue
        key = _canonical_join_key(event, session_id)
        if key in checkpoints:
            raise DshGenerationProbeSchemaError("duplicate canonical checkpoint join key")
        checkpoints[key] = event
    return checkpoints


def _canonical_join_key(event: CanonicalEvent, session_id: str) -> ProbeJoinKey:
    if event.attempt_id is None:
        raise DshGenerationProbeSchemaError("canonical checkpoint is missing attempt_id")
    payload = event.payload
    return (
        session_id,
        event.attempt_id,
        _positive_int(payload.get("checkpoint_ordinal"), "canonical checkpoint ordinal"),
        _non_negative_int(
            payload.get("observer_committed_tokens"),
            "canonical observer committed tokens",
        ),
    )


def _sidecar_join_key(row: Mapping[str, Any], session_id: str) -> ProbeJoinKey:
    observed_session = _text(row.get("session_id"), "Generation Probe session_id")
    if observed_session != session_id:
        raise DshGenerationProbeSchemaError("Generation Probe and DSH session ids differ")
    return (
        observed_session,
        _text(row.get("attempt_id"), "Generation Probe attempt_id"),
        _positive_int(row.get("checkpoint_ordinal"), "Generation Probe checkpoint ordinal"),
        _non_negative_int(
            row.get("observer_committed_tokens"),
            "Generation Probe observer committed tokens",
        ),
    )


def _projection_vector(value: object, dimension: int) -> tuple[float, ...]:
    if not isinstance(value, list) or len(value) != dimension:
        raise DshGenerationProbeSchemaError(
            f"hidden_state_projection must have dimension {dimension}"
        )
    if any(isinstance(item, bool) or not isinstance(item, (int, float)) for item in value):
        raise DshGenerationProbeSchemaError("hidden_state_projection must contain numbers")
    projection = tuple(float(item) for item in value)
    if any(not math.isfinite(item) for item in projection):
        raise DshGenerationProbeSchemaError("hidden_state_projection must be finite")
    return projection


def _reject_forbidden_fields(value: object, label: str) -> None:
    if isinstance(value, Mapping):
        forbidden = _FORBIDDEN_FIELDS & set(value)
        if forbidden:
            raise DshGenerationProbeSchemaError(
                f"{label} contains forbidden future or label fields: {', '.join(sorted(forbidden))}"
            )
        for nested in value.values():
            _reject_forbidden_fields(nested, label)
    elif isinstance(value, list):
        for nested in value:
            _reject_forbidden_fields(nested, label)


def _exact_keys(value: Mapping[str, Any], expected: set[str], label: str) -> None:
    if set(value) != expected:
        raise DshGenerationProbeSchemaError(f"{label} has missing or unknown fields")


def _object(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise DshGenerationProbeSchemaError(f"{label} must be an object")
    return value


def _text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise DshGenerationProbeSchemaError(f"{label} must be a non-empty string")
    return value


def _non_negative_int(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise DshGenerationProbeSchemaError(f"{label} must be a non-negative integer")
    return value


def _probability(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise DshGenerationProbeSchemaError(f"{label} must be a number")
    resolved = float(value)
    if not math.isfinite(resolved) or not 0.0 <= resolved <= 1.0:
        raise DshGenerationProbeSchemaError(f"{label} must be a probability")
    return resolved


def _non_negative_float(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise DshGenerationProbeSchemaError(f"{label} must be a number")
    resolved = float(value)
    if not math.isfinite(resolved) or resolved < 0.0:
        raise DshGenerationProbeSchemaError(f"{label} must be finite and non-negative")
    return resolved


def _positive_int(value: object, label: str) -> int:
    resolved = _non_negative_int(value, label)
    if resolved == 0:
        raise DshGenerationProbeSchemaError(f"{label} must be positive")
    return resolved


__all__ = [
    "DshGenerationProbeReader",
    "DshGenerationProbeSchemaError",
    "GenerationProbeMetadata",
    "validate_dsh_generation_probe_sidecar",
]

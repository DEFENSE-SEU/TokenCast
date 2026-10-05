from __future__ import annotations

import json
from dataclasses import replace
from typing import Any, Iterable, Mapping

from token_prediction.contracts import (
    CanonicalEvent,
    EventType,
    Observable,
    SourceCapabilities,
)
from token_prediction.identifiers import identifier
from token_prediction.trajectory import Trajectory

from .points import prediction_input_contract_id_from_capability
from .schema import (
    CAPABILITY_DATASET_SCHEMA_VERSION,
    DatasetRow,
    SupervisedDataset,
)
from token_prediction.dataset.lifecycle import _required_id


REQUEST_SHAPE_PROJECTION_ID = "request_boundary_shape_v1"
REQUEST_SHAPE_FEATURES = ("request_message_count", "request_content_chars")
AGGREGATE_TASK_SHAPE_PROJECTION_ID = "aggregate_task_usage_shape_v1"
AGGREGATE_TASK_SHAPE_INPUT_POLICY_ID = "aggregate_task_usage_launch_input_contract_v1"
AGGREGATE_TASK_SHAPE_FEATURES = (
    "agent_id",
    "llm_self_estimated_total_tokens",
    "model_id",
    "repo_id",
    "task_char_count",
    "task_code_fence_count",
    "task_line_count",
    "task_word_count",
)
_REQUEST_SHAPE_CONTRACT = {
    "projection_id": REQUEST_SHAPE_PROJECTION_ID,
    "visibility": "same_request_built_boundary",
    "source_payload_fields": list(REQUEST_SHAPE_FEATURES),
    "output_features": [
        {"name": "request_message_count", "dtype": "optional_non_negative_integer"},
        {"name": "request_content_chars", "dtype": "optional_non_negative_integer"},
    ],
    "forbidden_inputs": ["labels", "post_response_usage", "future_events"],
}


def request_shape_input_contract_id(base_input_contract_id: str) -> str:


    return identifier(
        "prediction-input-contract",
        "request-shape-v1",
        _required_id(base_input_contract_id, name="base_input_contract_id"),
        REQUEST_SHAPE_PROJECTION_ID,
    )


def aggregate_task_shape_input_contract_id(
    capability_profile_id: str,
) -> str:


    return identifier(
        "prediction-input-contract",
        "aggregate-task-shape-v1",
        _required_id(capability_profile_id, name="capability_profile_id"),
        AGGREGATE_TASK_SHAPE_INPUT_POLICY_ID,
        "+".join(AGGREGATE_TASK_SHAPE_FEATURES),
    )


def supported_input_contract_ids_from_capability(
    capability_profile_id: str,
    *,
    capabilities: SourceCapabilities | None = None,
) -> frozenset[str]:


    resolved_profile_id = _required_id(
        capability_profile_id,
        name="capability_profile_id",
    )
    base = prediction_input_contract_id_from_capability(
        capability_profile_id=resolved_profile_id,
    )
    supported = {base, request_shape_input_contract_id(base)}
    if capabilities is not None:
        if capabilities.capability_profile_id != resolved_profile_id:
            raise ValueError("capabilities do not match the declared profile id")
        if Observable.TASK_AGGREGATE_USAGE in capabilities.observables:
            supported.add(aggregate_task_shape_input_contract_id(resolved_profile_id))
    return frozenset(supported)


def request_shape_projection_document() -> Mapping[str, Any]:


    return json.loads(json.dumps(_REQUEST_SHAPE_CONTRACT, sort_keys=True))


def _optional_non_negative_integer(value: object, *, name: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer or missing")
    return value


def augment_request_shape_features(
    dataset: SupervisedDataset,
    trajectories: Iterable[Trajectory],
) -> SupervisedDataset:


    if dataset.schema_version != CAPABILITY_DATASET_SCHEMA_VERSION:
        raise ValueError("request-shape projection requires a capability dataset")
    if dataset.input_contract_id is None:
        raise ValueError("source dataset is missing its input contract id")
    resolved_trajectories = tuple(trajectories)
    if not resolved_trajectories:
        raise ValueError("request-shape projection requires trajectories")
    by_trajectory: dict[str, Trajectory] = {}
    events: dict[tuple[str, str], CanonicalEvent] = {}
    prediction_boundaries = {
        EventType.TASK_STARTED,
        EventType.REQUEST_BUILT,
        EventType.GENERATION_CHECKPOINT,
    }
    for trajectory in resolved_trajectories:
        if trajectory.trajectory_id in by_trajectory:
            raise ValueError("request-shape projection repeats trajectory_id")
        by_trajectory[trajectory.trajectory_id] = trajectory
        for event in trajectory.events:
            if event.event_type not in prediction_boundaries:
                continue
            key = (trajectory.trajectory_id, event.event_id)
            if key in events:
                raise ValueError("request-shape projection repeats an event identity")
            events[key] = event
    dataset_trajectories = {row.point.trajectory_id for row in dataset.rows}
    if dataset_trajectories != set(by_trajectory):
        raise ValueError("request-shape trajectories do not exactly cover the dataset")

    rows: list[DatasetRow] = []
    for row in dataset.rows:
        point = row.point
        trajectory = by_trajectory[point.trajectory_id]
        if point.task_id != trajectory.task_id or point.run_id != trajectory.run_id:
            raise ValueError("request-shape point and trajectory identity differ")
        try:
            event = events[(point.trajectory_id, point.source_event_id)]
        except KeyError as exc:
            raise ValueError("request-shape point boundary is missing from trajectories") from exc
        if event.event_seq != point.cutoff_event_seq:
            raise ValueError("request-shape point cutoff differs from its boundary")
        if any(name in point.features for name in REQUEST_SHAPE_FEATURES):
            raise ValueError("request-shape features are already present")

        features = dict(point.features)
        if event.event_type == EventType.REQUEST_BUILT:
            for name in REQUEST_SHAPE_FEATURES:
                features[name] = _optional_non_negative_integer(
                    event.payload.get(name),
                    name=name,
                )
        rows.append(replace(row, point=point.with_features(features)))

    input_contract_id = request_shape_input_contract_id(dataset.input_contract_id)
    derived_point_set_id = identifier(
        "prediction-point-set",
        "request-shape-v1",
        dataset.point_set_id or dataset.dataset_id,
        REQUEST_SHAPE_PROJECTION_ID,
    )
    return SupervisedDataset(
        dataset_id=identifier(
            "derived-dataset",
            "v1",
            dataset.dataset_id,
            REQUEST_SHAPE_PROJECTION_ID,
        ),
        rows=tuple(rows),
        schema_version=dataset.schema_version,
        source_descriptor_id=dataset.source_descriptor_id,
        capability_profile_id=dataset.capability_profile_id,
        input_contract_id=input_contract_id,
        point_set_id=derived_point_set_id,
    )


__all__ = [
    "AGGREGATE_TASK_SHAPE_FEATURES",
    "AGGREGATE_TASK_SHAPE_INPUT_POLICY_ID",
    "AGGREGATE_TASK_SHAPE_PROJECTION_ID",
    "REQUEST_SHAPE_FEATURES",
    "REQUEST_SHAPE_PROJECTION_ID",
    "aggregate_task_shape_input_contract_id",
    "augment_request_shape_features",
    "request_shape_input_contract_id",
    "request_shape_projection_document",
    "supported_input_contract_ids_from_capability",
]

from __future__ import annotations

import unittest
from dataclasses import replace
from unittest.mock import patch

from token_prediction.contracts import Observable, SourceCapabilities, SourceDescriptor
from token_prediction.dataset import (
    INTERNAL_PREDICTION_POSITIONS,
    PUBLIC_PREDICTION_POSITIONS,
    PredictionPosition,
    PredictionTarget,
    build_capability_supervised_dataset,
    build_prediction_points,
    prediction_input_contract_id,
)

from tests.helpers import _change_second_call_usage, make_two_call_trajectory


def _descriptor(*, local_count: bool = False) -> SourceDescriptor:
    observables = {
        Observable.ATTEMPT_USAGE,
        Observable.REQUEST_BOUNDARIES,
        Observable.TASK_TERMINATION,
        Observable.TASK_USAGE,
    }
    if local_count:
        observables.add(Observable.REQUEST_LOCAL_COUNT)
    return SourceDescriptor(
        source_id="point-fixture",
        revision="revision-1",
        manifest_path="workspace/manifests/point-fixture.json",
        capabilities=SourceCapabilities(
            source_id="point-fixture",
            observables=frozenset(observables),
        ),
    )


class PredictionPointTests(unittest.TestCase):
    def test_prediction_positions_have_exact_public_and_internal_sets(self) -> None:
        self.assertEqual(
            PUBLIC_PREDICTION_POSITIONS,
            frozenset(
                {
                    PredictionPosition.TASK_LAUNCH,
                    PredictionPosition.CALL_PRE,
                    PredictionPosition.CALL_UPDATE,
                    PredictionPosition.TASK_UPDATE,
                }
            ),
        )
        self.assertEqual(
            INTERNAL_PREDICTION_POSITIONS,
            frozenset(
                {
                    PredictionPosition.TASK_SEED,
                    PredictionPosition.ATTEMPT_INIT,
                }
            ),
        )
        self.assertFalse(PUBLIC_PREDICTION_POSITIONS & INTERNAL_PREDICTION_POSITIONS)
        self.assertEqual(
            PUBLIC_PREDICTION_POSITIONS | INTERNAL_PREDICTION_POSITIONS,
            frozenset(PredictionPosition),
        )

    def test_prefix_builder_never_calls_label_builders(self) -> None:
        trajectory = make_two_call_trajectory(0)
        descriptor = _descriptor()
        with (
            patch(
                "token_prediction.dataset.labels.build_prediction_labels",
                side_effect=AssertionError("labels are forbidden"),
            ),
            patch(
                "token_prediction.dataset.labels.build_task_aggregate_label",
                side_effect=AssertionError("labels are forbidden"),
            ),
            patch(
                "token_prediction.dataset.labels.build_generation_labels",
                side_effect=AssertionError("labels are forbidden"),
            ),
        ):
            result = build_prediction_points((trajectory,), descriptor)
        self.assertEqual(len(result.points), 9)
        self.assertEqual(
            result.input_contract_id,
            prediction_input_contract_id(descriptor),
        )

    def test_schema_v2_point_rows_and_dataset_ids_expose_natural_keys(self) -> None:
        trajectory = make_two_call_trajectory(0)
        dataset_ids = set()
        for local_count in (False, True):
            descriptor = _descriptor(local_count=local_count)
            point_set = build_prediction_points((trajectory,), descriptor)
            dataset = build_capability_supervised_dataset((trajectory,), descriptor)
            self.assertEqual(
                point_set.points,
                tuple(row.point for row in dataset.rows),
            )
            self.assertTrue(dataset.dataset_id.startswith("capability-supervised-dataset:"))
            self.assertIn("point-fixture", dataset.dataset_id)
            dataset_ids.add(dataset.dataset_id)
        self.assertEqual(len(dataset_ids), 2)

    def test_suffix_observation_cannot_change_prefix_point_or_input_contract(self) -> None:
        original = make_two_call_trajectory(0)
        changed = _change_second_call_usage(original)
        descriptor = _descriptor()
        first = build_prediction_points((original,), descriptor)
        second = build_prediction_points((changed,), descriptor)

        def task_seed(result: object):
            return next(
                point
                for point in result.points
                if point.position == PredictionPosition.TASK_SEED
                and point.target == PredictionTarget.TASK_PROVIDER_ACCOUNTED_REMAINING_TOKENS
            )

        self.assertEqual(task_seed(first), task_seed(second))
        self.assertEqual(first.input_contract_id, second.input_contract_id)
        self.assertEqual(first.point_set_id, second.point_set_id)

    def test_input_contract_excludes_manifest_revision_provenance(self) -> None:
        first = _descriptor()
        changed = replace(
            first,
            revision="revision-2",
        )
        self.assertNotEqual(first.descriptor_id, changed.descriptor_id)
        self.assertEqual(
            prediction_input_contract_id(first),
            prediction_input_contract_id(changed),
        )

    def test_proxy_local_counts_are_removed_but_missing_features_remain_explicit(self) -> None:
        point_set = build_prediction_points(
            (make_two_call_trajectory(0),),
            _descriptor(),
        )
        task_seed = next(
            point
            for point in point_set.points
            if point.position == PredictionPosition.TASK_SEED
            and point.target == PredictionTarget.TASK_PROVIDER_ACCOUNTED_REMAINING_TOKENS
        )
        self.assertNotIn("current_request_tokens_local", task_seed.features)
        self.assertNotIn("request_delta_tokens", task_seed.features)
        self.assertIn("last_call_output_tokens", task_seed.features)
        self.assertIsNone(task_seed.features["last_call_output_tokens"])


if __name__ == "__main__":
    unittest.main()

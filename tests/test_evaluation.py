from __future__ import annotations

import unittest

from token_prediction.contracts import EventType
from token_prediction.dataset import (
    PredictionPosition,
    PredictionTarget,
    build_supervised_dataset,
)
from token_prediction.estimators import TokenForecast
from token_prediction.evaluation import (
    CalibrationExample,
    FittedExpansionCalibrator,
    IdentityCalibrator,
    ScoredForecast,
    TaskMaxConformalCalibrator,
    build_cohort_metadata_authority,
    evaluate_cohort_forecasts,
    evaluate_forecasts,
    evaluate_task_forecasts,
)
from token_prediction.trajectory import Trajectory
from token_prediction.experiments.candidates import prediction_cohort
from tests.helpers import make_two_call_trajectory
from tests.target_helpers import four_position_target_trajectory


class EvaluationTests(unittest.TestCase):
    def test_identity_calibrator_rejects_nonzero_expansion(self) -> None:
        with self.assertRaisesRegex(ValueError, "exactly zero"):
            FittedExpansionCalibrator("none", 0.1, 1.0)

    def test_fitted_calibrator_round_trip_is_strict_and_replays(self) -> None:
        target = PredictionTarget.CALL_BILLABLE_OUTPUT_TOKENS
        fitted = FittedExpansionCalibrator("task_max_conformal", 0.1, 3.5)
        loaded = FittedExpansionCalibrator.from_dict(fitted.to_dict())
        forecast = TokenForecast("p", target, 5, 10, 12)
        self.assertEqual(loaded, fitted)
        self.assertEqual(loaded.transform(forecast), fitted.transform(forecast))
        for mutated in (
            {**fitted.to_dict(), "extra": True},
            {**fitted.to_dict(), "expansion": -1.0},
            {**fitted.to_dict(), "interval_alpha": float("nan")},
            {**fitted.to_dict(), "calibrator_id": "unknown"},
        ):
            with self.assertRaises((TypeError, ValueError)):
                FittedExpansionCalibrator.from_dict(mutated)

    def test_metric_suite_matches_hand_calculation(self) -> None:
        target = PredictionTarget.CALL_BILLABLE_OUTPUT_TOKENS
        rows = (
            ScoredForecast("t0", "r0", TokenForecast("p0", target, 7, 8, 11), 10, 1),
            ScoredForecast("t1", "r1", TokenForecast("p1", target, 22, 25, 28), 20, 1),
        )
        metrics = evaluate_forecasts(rows, alpha=0.1)
        self.assertAlmostEqual(float(metrics["mae"]), 3.5)
        self.assertAlmostEqual(float(metrics["mdape"]), 0.2)
        self.assertAlmostEqual(float(metrics["wape"]), 7 / 30)
        self.assertAlmostEqual(float(metrics["lower_pinball_loss"]), 1.025)
        self.assertAlmostEqual(float(metrics["upper_pinball_loss"]), 0.225)
        self.assertAlmostEqual(float(metrics["mean_pinball_loss"]), 0.625)
        self.assertAlmostEqual(float(metrics["nominal_coverage"]), 0.9)
        self.assertAlmostEqual(float(metrics["coverage"]), 0.5)
        self.assertAlmostEqual(float(metrics["coverage_calibration_error"]), 0.4)
        self.assertAlmostEqual(float(metrics["task_simultaneous_coverage"]), 0.5)
        self.assertAlmostEqual(float(metrics["mean_bias"]), 1.5)
        self.assertAlmostEqual(float(metrics["underestimation_rate"]), 0.5)
        self.assertAlmostEqual(float(metrics["interval_score"]), 25.0)
        self.assertAlmostEqual(float(metrics["normalized_interval_width"]), 0.35)

    def test_interval_tail_and_reserved_capacity_metrics_are_weighted(self) -> None:
        target = PredictionTarget.CALL_BILLABLE_OUTPUT_TOKENS
        rows = (
            ScoredForecast(
                "below",
                "run-below",
                TokenForecast(
                    "point-below",
                    target,
                    0,
                    5,
                    8,
                    raw_lower=0,
                    raw_point=5,
                    raw_upper=8,
                ),
                10,
                1,
            ),
            ScoredForecast(
                "above",
                "run-above",
                TokenForecast(
                    "point-above",
                    target,
                    7,
                    8,
                    10,
                    raw_lower=7,
                    raw_point=8,
                    raw_upper=10,
                ),
                5,
                2,
            ),
            ScoredForecast(
                "covered",
                "run-covered",
                TokenForecast(
                    "point-covered",
                    target,
                    4,
                    6,
                    9,
                    raw_lower=4,
                    raw_point=6,
                    raw_upper=9,
                ),
                6,
                3,
            ),
        )
        metrics = evaluate_forecasts(rows)
        self.assertAlmostEqual(metrics["interval_below_truth_rate"], 1 / 6)
        self.assertAlmostEqual(metrics["interval_above_truth_rate"], 2 / 6)
        self.assertAlmostEqual(metrics["target_exceeds_upper_rate"], 1 / 6)
        self.assertAlmostEqual(metrics["mean_extra_reserved_tokens"], 19 / 6)
        self.assertAlmostEqual(
            metrics["coverage"]
            + metrics["interval_below_truth_rate"]
            + metrics["interval_above_truth_rate"],
            1.0,
        )
        self.assertEqual(
            metrics["raw_interval_below_truth_rate"],
            metrics["interval_below_truth_rate"],
        )
        self.assertEqual(
            metrics["raw_interval_above_truth_rate"],
            metrics["interval_above_truth_rate"],
        )
        self.assertEqual(
            metrics["raw_target_exceeds_upper_rate"],
            metrics["target_exceeds_upper_rate"],
        )
        self.assertEqual(
            metrics["raw_mean_extra_reserved_tokens"],
            metrics["mean_extra_reserved_tokens"],
        )

    def test_interval_tail_boundaries_are_inclusive_coverage(self) -> None:
        target = PredictionTarget.CALL_BILLABLE_OUTPUT_TOKENS
        rows = (
            ScoredForecast(
                "at-upper",
                "run-upper",
                TokenForecast("point-upper", target, 0, 5, 10),
                10,
                1,
            ),
            ScoredForecast(
                "at-lower",
                "run-lower",
                TokenForecast("point-lower", target, 10, 12, 15),
                10,
                1,
            ),
        )
        metrics = evaluate_forecasts(rows)
        self.assertEqual(metrics["coverage"], 1.0)
        self.assertEqual(metrics["interval_below_truth_rate"], 0.0)
        self.assertEqual(metrics["interval_above_truth_rate"], 0.0)
        self.assertEqual(metrics["target_exceeds_upper_rate"], 0.0)

    def test_empty_or_nonfinite_metric_inputs_fail_closed(self) -> None:
        target = PredictionTarget.CALL_BILLABLE_OUTPUT_TOKENS
        with self.assertRaisesRegex(ValueError, "empty"):
            evaluate_forecasts(())
        with self.assertRaisesRegex(ValueError, "target values must be finite"):
            evaluate_forecasts(
                (
                    ScoredForecast(
                        "task",
                        "run",
                        TokenForecast("point", target, 0, 1, 2),
                        float("nan"),
                        1,
                    ),
                )
            )

    def test_zero_target_does_not_divide_by_zero(self) -> None:
        target = PredictionTarget.CALL_BILLABLE_OUTPUT_TOKENS
        metrics = evaluate_forecasts(
            (ScoredForecast("t", "r", TokenForecast("p", target, 0, 0, 1), 0, 1),)
        )
        self.assertEqual(metrics["wape"], 0.0)
        self.assertEqual(metrics["mdape"], 0.0)
        self.assertEqual(metrics["normalized_interval_width"], 1.0)

    def test_task_metrics_are_label_free_weighted_aggregates(self) -> None:
        target = PredictionTarget.CALL_BILLABLE_OUTPUT_TOKENS
        rows = (
            ScoredForecast("t0", "r0", TokenForecast("p0", target, 5, 8, 12), 10, 1),
            ScoredForecast("t0", "r1", TokenForecast("p1", target, 15, 18, 25), 20, 3),
            ScoredForecast("t1", "r2", TokenForecast("p2", target, 0, 2, 4), 3, 2),
        )
        metrics = evaluate_task_forecasts(rows, alpha=0.1)
        self.assertEqual(tuple(metrics), ("t0", "t1"))
        self.assertEqual(metrics["t0"].n_points, 2)
        self.assertEqual(metrics["t0"].n_trajectories, 2)
        self.assertEqual(metrics["t0"].weight_sum, 4)
        self.assertEqual(metrics["t0"].weighted_mae, 2)
        self.assertEqual(metrics["t0"].weighted_coverage, 1)
        self.assertNotIn("target_value", metrics["t0"].to_dict())

    def test_cohort_metrics_reduce_points_to_equal_weight_tasks(self) -> None:
        target = PredictionTarget.CALL_BILLABLE_OUTPUT_TOKENS
        rows = (
            ScoredForecast(
                "many-points",
                "run-0",
                TokenForecast("p0", target, 10, 10, 10),
                10,
                1,
                task_type="repo_repair",
                actor_model_id="model-a",
            ),
            ScoredForecast(
                "many-points",
                "run-1",
                TokenForecast("p1", target, 10, 10, 10),
                10,
                1,
                task_type="repo_repair",
                actor_model_id="model-a",
            ),
            ScoredForecast(
                "one-point",
                "run-2",
                TokenForecast("p2", target, 20, 20, 20),
                10,
                1,
                task_type="repo_repair",
                actor_model_id=None,
            ),
        )

        report = evaluate_cohort_forecasts(rows)
        task_type = report["dimensions"]["task_type"][0]
        self.assertEqual(task_type["n_tasks"], 2)
        self.assertEqual(task_type["n_points"], 3)
        self.assertEqual(task_type["task_macro_mae"], 5.0)
        actor_rows = report["dimensions"]["actor_model_id"]
        self.assertEqual([item["value"] for item in actor_rows], ["model-a", None])
        self.assertEqual(report["missing_value_policy"], "null_cohort")

    def test_trajectory_authority_keeps_terminal_metadata_out_of_model_features(self) -> None:
        source = make_two_call_trajectory(0)
        events = tuple(
            event.with_payload(
                {
                    **event.payload,
                    "task_family_id": "family-a",
                    "split_group_id": "repository-a",
                    "task_type": "repo_repair",
                    "actor_model_id": "actor-a",
                    "model_family_id": "family-model-a",
                }
            )
            if event.event_type == EventType.TASK_STARTED
            else event.with_payload({**event.payload, "outcome": "failed"})
            if event.event_type == EventType.TASK_FINISHED
            else event
            for event in source.events
        )
        trajectory = Trajectory.from_events(events)
        dataset = build_supervised_dataset((trajectory,))
        authority = build_cohort_metadata_authority((trajectory,))
        point = dataset.rows[0].point

        metadata = authority.for_point(point)

        self.assertEqual(metadata.task_family_id, "family-a")
        self.assertEqual(metadata.split_group_id, "repository-a")
        self.assertEqual(metadata.task_type, "repo_repair")
        self.assertEqual(metadata.actor_model_id, "actor-a")
        self.assertEqual(metadata.model_family_id, "family-model-a")
        self.assertEqual(metadata.termination_state, "failed")
        self.assertEqual(metadata.trajectory_length_bucket, "short_1_2_calls")
        self.assertNotIn("termination_state", point.features)
        self.assertNotIn("trajectory_length_bucket", point.features)
        self.assertNotIn("model_family_id", point.features)
        original_dataset = build_supervised_dataset((source,))
        original_point = next(
            row.point for row in original_dataset.rows if row.point.point_id == point.point_id
        )
        self.assertEqual(original_point.features, point.features)
        self.assertEqual(
            build_cohort_metadata_authority((source,)).for_point(original_point).termination_state,
            "success",
        )

        report = evaluate_cohort_forecasts(
            (
                ScoredForecast(
                    point.task_id,
                    point.trajectory_id,
                    TokenForecast(point.point_id, point.target, 0, 1, 2),
                    1,
                    1,
                    **metadata.to_dict(),
                ),
            )
        )
        self.assertEqual(
            report["dimensions"]["termination_state"][0]["value"],
            "failed",
        )
        self.assertEqual(
            report["dimensions"]["trajectory_length_bucket"][0]["value"],
            "short_1_2_calls",
        )
        self.assertEqual(
            report["trajectory_length_policy_id"],
            "logical_call_count_buckets_v1",
        )

        retry_trajectory = four_position_target_trajectory()
        retry_dataset = build_supervised_dataset((retry_trajectory,))
        retry_point = next(
            row.point for row in retry_dataset.rows if row.point.attempt_id == "attempt-1"
        )
        retry_authority = build_cohort_metadata_authority((retry_trajectory,))
        self.assertEqual(retry_authority.for_point(retry_point).retry_state, "retry")

        call_points = [
            row.point
            for row in retry_dataset.rows
            if row.point.position == PredictionPosition.CALL_PRE
            and row.point.target == PredictionTarget.CALL_BILLABLE_OUTPUT_TOKENS
        ]
        first_call = next(
            point for point in call_points if point.logical_call_id.endswith("call-0")
        )
        second_call = next(
            point for point in call_points if point.logical_call_id.endswith("call-1")
        )
        self.assertIsNone(retry_authority.for_point(first_call).retry_state)
        self.assertEqual(
            retry_authority.for_scoring_point(first_call).retry_state,
            "retried_call",
        )
        self.assertEqual(
            retry_authority.for_scoring_point(second_call).retry_state,
            "single_attempt_call",
        )
        self.assertIsNone(
            prediction_cohort(first_call, None, retry_authority)["retry_state"]
        )
        self.assertEqual(
            prediction_cohort(
                first_call,
                None,
                retry_authority,
                scoring_only=True,
            )["retry_state"],
            "retried_call",
        )

    def test_task_max_calibration_uses_one_conservative_score_per_task(self) -> None:
        target = PredictionTarget.CALL_BILLABLE_OUTPUT_TOKENS
        examples = (
            CalibrationExample("t0", TokenForecast("p0", target, 8, 10, 12), 20),
            CalibrationExample("t0", TokenForecast("p1", target, 8, 10, 12), 14),
            CalibrationExample("t1", TokenForecast("p2", target, 8, 10, 12), 13),
        )
        fitted = TaskMaxConformalCalibrator(alpha=0.1).fit(examples)
        transformed = fitted.transform(TokenForecast("test", target, 8, 10, 12))
        self.assertEqual((transformed.lower, transformed.upper), (0.0, 20.0))
        self.assertEqual(
            (transformed.raw_lower, transformed.raw_point, transformed.raw_upper),
            (8, 10, 12),
        )

    def test_calibration_preserves_estimator_native_quantiles(self) -> None:
        target = PredictionTarget.CALL_BILLABLE_OUTPUT_TOKENS
        forecast = TokenForecast(
            "point",
            target,
            0,
            9,
            9,
            raw_lower=-2,
            raw_point=9,
            raw_upper=8,
        )
        transformed = IdentityCalibrator().fit(()).transform(forecast)
        self.assertEqual(
            (transformed.raw_lower, transformed.raw_point, transformed.raw_upper),
            (-2, 9, 8),
        )

    def test_raw_interval_metrics_use_pre_calibration_quantiles(self) -> None:
        target = PredictionTarget.CALL_BILLABLE_OUTPUT_TOKENS
        rows = (
            ScoredForecast(
                "t0",
                "r0",
                TokenForecast(
                    "p0",
                    target,
                    5,
                    8,
                    15,
                    raw_lower=7,
                    raw_point=8,
                    raw_upper=11,
                ),
                10,
                1,
            ),
            ScoredForecast(
                "t1",
                "r1",
                TokenForecast(
                    "p1",
                    target,
                    10,
                    25,
                    30,
                    raw_lower=22,
                    raw_point=25,
                    raw_upper=28,
                ),
                20,
                1,
            ),
        )
        metrics = evaluate_forecasts(rows, alpha=0.1)
        self.assertEqual(metrics["raw_coverage"], 0.5)
        self.assertAlmostEqual(metrics["raw_coverage_calibration_error"], 0.4)
        self.assertEqual(metrics["raw_task_simultaneous_coverage"], 0.5)
        self.assertEqual(metrics["raw_interval_score"], 25.0)
        self.assertAlmostEqual(metrics["raw_lower_pinball_loss"], 1.025)
        self.assertAlmostEqual(metrics["raw_upper_pinball_loss"], 0.225)
        self.assertAlmostEqual(metrics["raw_mean_pinball_loss"], 0.625)
        self.assertEqual(metrics["raw_median_interval_width"], 4.0)
        self.assertAlmostEqual(metrics["raw_normalized_interval_width"], 0.35)
        self.assertEqual(metrics["quantile_crossing_rate"], 0.0)

    def test_raw_crossing_is_counted_before_validity_repair(self) -> None:
        target = PredictionTarget.CALL_BILLABLE_OUTPUT_TOKENS
        metrics = evaluate_forecasts(
            (
                ScoredForecast(
                    "task",
                    "trajectory",
                    TokenForecast(
                        "point",
                        target,
                        0,
                        10,
                        20,
                        raw_lower=20,
                        raw_point=10,
                        raw_upper=5,
                    ),
                    10,
                    1,
                ),
            )
        )
        self.assertEqual(metrics["quantile_crossing_rate"], 1.0)
        self.assertGreaterEqual(float(metrics["raw_interval_score"]), 0.0)

    def test_task_simultaneous_coverage_requires_every_point(self) -> None:
        target = PredictionTarget.CALL_BILLABLE_OUTPUT_TOKENS
        rows = (
            ScoredForecast(
                "t0",
                "r0",
                TokenForecast(
                    "p0",
                    target,
                    0,
                    5,
                    10,
                    raw_lower=0,
                    raw_point=5,
                    raw_upper=10,
                ),
                5,
                1,
            ),
            ScoredForecast(
                "t0",
                "r0",
                TokenForecast(
                    "p1",
                    target,
                    0,
                    5,
                    10,
                    raw_lower=0,
                    raw_point=15,
                    raw_upper=30,
                ),
                20,
                1,
            ),
            ScoredForecast(
                "t1",
                "r1",
                TokenForecast(
                    "p2",
                    target,
                    0,
                    5,
                    10,
                    raw_lower=10,
                    raw_point=15,
                    raw_upper=20,
                ),
                5,
                1,
            ),
        )
        metrics = evaluate_forecasts(rows)
        self.assertAlmostEqual(float(metrics["coverage"]), 2 / 3)
        self.assertEqual(metrics["task_simultaneous_coverage"], 0.5)
        self.assertAlmostEqual(float(metrics["raw_coverage"]), 2 / 3)
        self.assertEqual(metrics["raw_task_simultaneous_coverage"], 0.5)

    def test_partial_or_mixed_raw_forecasts_fail_closed(self) -> None:
        target = PredictionTarget.CALL_BILLABLE_OUTPUT_TOKENS
        with self.assertRaisesRegex(ValueError, "provided together"):
            TokenForecast("partial", target, 0, 1, 2, raw_lower=0)
        with self.assertRaisesRegex(ValueError, "all rows or no rows"):
            evaluate_forecasts(
                (
                    ScoredForecast(
                        "t0",
                        "r0",
                        TokenForecast(
                            "raw",
                            target,
                            0,
                            1,
                            2,
                            raw_lower=0,
                            raw_point=1,
                            raw_upper=2,
                        ),
                        1,
                        1,
                    ),
                    ScoredForecast("t1", "r1", TokenForecast("plain", target, 0, 1, 2), 1, 1),
                )
            )


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from scripts.experiments.audit_prediction_export import audit_prediction_export
from token_prediction.config import load_config
from token_prediction.contracts import EventType
from token_prediction.dataset import PredictionPoint, PredictionPosition, PredictionTarget
from token_prediction.experiments.candidates import (
    AblationAxis,
    AblationSpec,
    CandidateRole,
)
from token_prediction.experiments.lineage import ArtifactVerificationError
from token_prediction.experiments.prediction_export import (
    LEGACY_PREDICTION_EXPORT_SCHEMA_ID,
    load_prediction_export,
)
from token_prediction.pipeline import (
    dataset_from_trajectories,
    inspect_replay,
    load_jsonl_events,
    load_trajectory,
    _progress_known_amount,
    run_configured_experiments,
    task_family_registry_from_trajectories,
)
from token_prediction.trajectory import Trajectory
from tests.helpers import make_two_call_trajectory, write_trajectory
from tests.target_helpers import (
    four_position_target_trajectory,
    target_source_descriptor,
)


ROOT = Path(__file__).resolve().parents[1]


def _with_task_family(
    trajectory: Trajectory,
    family_id: str,
    *,
    split_group_id: str | None = None,
) -> Trajectory:
    attempt_usages = tuple(
        event.payload["usage"]
        for event in trajectory.events
        if event.event_type in {EventType.API_COMPLETED, EventType.API_FAILED}
    )
    task_input_tokens = sum(int(usage["input_tokens"]) for usage in attempt_usages)
    task_output_tokens = sum(int(usage["output_tokens"]) for usage in attempt_usages)
    task_usage = {
        "input_tokens": task_input_tokens,
        "output_tokens": task_output_tokens,
        "total_tokens": task_input_tokens + task_output_tokens,
    }
    return Trajectory.from_events(
        tuple(
            event.with_payload(
                {
                    **event.payload,
                    "task_family_id": family_id,
                    **(
                        {"split_group_id": split_group_id}
                        if split_group_id is not None
                        else {}
                    ),
                    "task_type": "repo_repair",
                    "model_family_id": "fixture-model-family",
                }
            )
            if event.event_type == EventType.TASK_STARTED
            else (
                event.with_payload({**event.payload, "usage": task_usage})
                if event.event_type in {EventType.TASK_FINISHED, EventType.TASK_ABORTED}
                else event
            )
            for event in trajectory.events
        )
    )


class PredictionPipelineTests(unittest.TestCase):
    def test_provider_remaining_progress_uses_causal_accounted_tokens(self) -> None:
        point = PredictionPoint(
            point_id="point",
            source_event_id="event",
            task_id="task",
            trajectory_id="trajectory",
            run_id="run",
            prediction_context_id="context",
            condition_id="condition",
            logical_call_id="call",
            attempt_id=None,
            cutoff_event_seq=3,
            position=PredictionPosition.TASK_UPDATE,
            target=PredictionTarget.TASK_PROVIDER_ACCOUNTED_REMAINING_TOKENS,
            features={
                "cumulative_provider_input_tokens": 120,
                "cumulative_provider_output_tokens": 30,
            },
            known_offset_tokens=0,
            token_unit_contract_id="provider:test-v1",
        )

        self.assertEqual(
            _progress_known_amount(point),
            (150, "cumulative_provider_input_plus_output"),
        )

    def test_replay_loads_canonical_events_and_labels(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "events.jsonl"
            write_trajectory(path, make_two_call_trajectory(0))
            events = load_jsonl_events(path)
            trajectory = load_trajectory(path)
            report = inspect_replay(path)

        self.assertEqual(len(events), len(trajectory.events))
        self.assertEqual(report["task_id"], trajectory.task_id)
        self.assertEqual(report["trajectory_id"], trajectory.trajectory_id)
        self.assertGreater(report["prediction_points"], 0)
        self.assertTrue(report["labels"])

    def test_invalid_jsonl_reports_the_line_number(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "events.jsonl"
            path.write_text("{}\nnot-json\n", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "line 1|line 2"):
                load_jsonl_events(path)

    def test_grouping_ids_remain_outside_estimator_features(self) -> None:
        trajectory = _with_task_family(
            make_two_call_trajectory(0),
            "fixture-family",
            split_group_id="fixture-split-group",
        )
        config = load_config(ROOT / "configs" / "experiments" / "mvp.toml")

        dataset = dataset_from_trajectories((trajectory,), config)

        self.assertTrue(dataset.rows)
        self.assertTrue(
            all(
                "task_family_id" not in row.point.features
                and "split_group_id" not in row.point.features
                and "model_family_id" not in row.point.features
                and "termination_state" not in row.point.features
                for row in dataset.rows
            )
        )

    def test_hierarchical_mode_builds_the_four_position_target_dataset(self) -> None:
        descriptor = target_source_descriptor(
            task_aggregate_usage=True,
            attempt_boundaries=True,
            output_deltas=True,
            observer_token_accounting=True,
        )
        config = replace(
            load_config(ROOT / "configs" / "experiments" / "mvp.toml"),
            dataset_mode="hierarchical",
            source_descriptor=descriptor,
        )

        dataset = dataset_from_trajectories(
            (four_position_target_trajectory(),),
            config,
        )

        positions = {row.point.position for row in dataset.rows}
        targets = {row.point.target for row in dataset.rows}
        self.assertTrue(
            {
                PredictionPosition.TASK_LAUNCH,
                PredictionPosition.CALL_PRE,
                PredictionPosition.CALL_UPDATE,
                PredictionPosition.TASK_UPDATE,
            }
            <= positions
        )
        self.assertIn(
            PredictionTarget.ATTEMPT_PROVIDER_FINAL_OUTPUT_TOKENS,
            targets,
        )
        self.assertIn(
            PredictionTarget.ATTEMPT_OBSERVER_REMAINING_VISIBLE_OUTPUT_TOKENS,
            targets,
        )
        self.assertEqual(dataset.source_descriptor_id, descriptor.descriptor_id)

    def test_pipeline_requires_task_family_on_every_trajectory(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root / "missing-family.jsonl"
            write_trajectory(path, make_two_call_trajectory(0))
            config = replace(
                load_config(ROOT / "configs" / "experiments" / "mvp.toml"),
                workspace=root / "workspace",
            )

            with self.assertRaisesRegex(
                ValueError,
                "TASK_STARTED payload requires task_family_id",
            ):
                run_configured_experiments(config, [path], output_dir=root / "result")

    def test_pipeline_rejects_inconsistent_family_for_the_same_task(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths: list[Path] = []
            for run_index, family_id in enumerate(("family-a", "family-b")):
                path = root / f"run-{run_index}.jsonl"
                write_trajectory(
                    path,
                    _with_task_family(
                        make_two_call_trajectory(0, run_index),
                        family_id,
                    ),
                )
                paths.append(path)
            config = replace(
                load_config(ROOT / "configs" / "experiments" / "mvp.toml"),
                workspace=root / "workspace",
            )

            with self.assertRaisesRegex(
                ValueError,
                "task task-0 has inconsistent TASK_STARTED split grouping values",
            ):
                run_configured_experiments(config, paths, output_dir=root / "result")

    def test_pipeline_split_registry_prefers_broader_group_id(self) -> None:
        trajectories = (
            _with_task_family(
                make_two_call_trajectory(0),
                "fine-family-a",
                split_group_id="shared-group",
            ),
            _with_task_family(
                make_two_call_trajectory(1),
                "fine-family-b",
                split_group_id="shared-group",
            ),
        )

        registry = task_family_registry_from_trajectories(trajectories)

        self.assertEqual(
            registry.task_to_split_group,
            {"task-0": "shared-group", "task-1": "shared-group"},
        )

    def test_configured_baselines_run_and_publish_readable_results(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            event_paths: list[Path] = []
            for task_index in range(32):
                path = root / "events" / f"task-{task_index:03d}.jsonl"
                path.parent.mkdir(parents=True, exist_ok=True)
                write_trajectory(
                    path,
                    _with_task_family(
                        make_two_call_trajectory(task_index, 0),
                        f"fine-family-{task_index:03d}",
                        split_group_id=f"split-group-{task_index // 2:03d}",
                    ),
                )
                event_paths.append(path)

            config = replace(
                load_config(ROOT / "configs" / "experiments" / "mvp.toml"),
                workspace=root / "workspace",
            )
            config = replace(
                config,
                candidates=tuple(
                    candidate
                    if candidate.candidate_id == "empirical_quantile"
                    else replace(
                        candidate,
                        role=CandidateRole.ABLATION,
                        ablation=AblationSpec(
                            reference_candidate_id="empirical_quantile",
                            axis=AblationAxis.METHOD,
                            allowed_config_paths=frozenset(
                                {
                                    "estimator_id",
                                    "feature_set",
                                    "graph.updater_estimator_id",
                                    "params.feature_name",
                                }
                            ),
                        ),
                    )
                    for candidate in config.candidates
                ),
            )
            summary = run_configured_experiments(
                config,
                event_paths,
            )
            output = summary.output_dir

            run_document = json.loads((output / "run.json").read_text(encoding="utf-8"))
            model_cell_documents = {
                cell_name: json.loads(
                    (output / "models" / cell_name / "cell.json").read_text(encoding="utf-8")
                )
                for cell_name in run_document["model_cells"]
            }
            metrics = json.loads((output / "metrics.json").read_text(encoding="utf-8"))
            comparisons = json.loads(
                (output / "comparisons.json").read_text(encoding="utf-8")
            )
            prediction_lines = list(load_prediction_export(output))
            audit_summary = audit_prediction_export(output)

            prediction_path = output / "predictions.jsonl"
            original_predictions = prediction_path.read_text(encoding="utf-8")
            raw_first_prediction = json.loads(original_predictions.splitlines()[0])
            for defaulted_field in (
                "dataset_id",
                "split_plan_id",
                "eligibility_policy_id",
                "target",
                "model_revision",
                "fitted_artifact_id",
            ):
                self.assertNotIn(defaulted_field, raw_first_prediction)
            package_manifests = [
                json.loads(path.read_text(encoding="utf-8"))
                for path in (output / "models").rglob("manifest.json")
                if '"stage_name": "fitted-fold-package"'
                in path.read_text(encoding="utf-8")
            ]
            first_prediction = prediction_lines[0]
            wrong_package = next(
                manifest
                for manifest in package_manifests
                if manifest["metadata"]["experiment_id"]
                == first_prediction["experiment_id"]
                and manifest["metadata"]["candidate_id"]
                == first_prediction["candidate_id"]
                and manifest["metadata"]["fold"] != first_prediction["fold"]
            )
            altered_predictions = [dict(item) for item in prediction_lines]
            altered_predictions[0]["fitted_artifact_id"] = wrong_package["artifact_id"]
            prediction_path.write_text(
                "".join(
                    json.dumps(item, sort_keys=True) + "\n"
                    for item in altered_predictions
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(
                ValueError,
                "overrides default fitted_artifact_id",
            ):
                audit_prediction_export(output)
            prediction_path.write_text(original_predictions, encoding="utf-8")

            run_path = output / "run.json"
            original_run = run_path.read_text(encoding="utf-8")
            altered_run = json.loads(original_run)
            candidate_defaults = altered_run["prediction_record_defaults"][
                "candidates"
            ][first_prediction["experiment_id"]][first_prediction["candidate_id"]]
            candidate_defaults["fitted_artifact_ids"][str(first_prediction["fold"])] = (
                wrong_package["artifact_id"]
            )
            run_path.write_text(json.dumps(altered_run), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "metadata fold disagrees"):
                audit_prediction_export(output)
            run_path.write_text(original_run, encoding="utf-8")

            fold_path = next(
                path
                for path in (output / "models").rglob("fold.json")
                if json.loads(path.read_text(encoding="utf-8"))["fold"] == 0
            )
            original_fold = fold_path.read_bytes()
            altered_fold = original_fold.replace(b'"fold": 0', b'"fold": 9', 1)
            self.assertEqual(len(altered_fold), len(original_fold))
            fold_path.write_bytes(altered_fold)
            with self.assertRaisesRegex(
                ArtifactVerificationError,
                "SHA-256 content differs",
            ):
                audit_prediction_export(output)
            fold_path.write_bytes(original_fold)

            legacy_run = dict(run_document)
            legacy_run["prediction_export_schema_id"] = (
                LEGACY_PREDICTION_EXPORT_SCHEMA_ID
            )
            run_path.write_text(json.dumps(legacy_run), encoding="utf-8")
            legacy_predictions = []
            for item in prediction_lines:
                legacy = dict(item)
                legacy["prediction_export_schema_id"] = (
                    LEGACY_PREDICTION_EXPORT_SCHEMA_ID
                )
                del legacy["fitted_artifact_id"]
                del legacy["reasoning_setting"]
                del legacy["repository"]
                del legacy["progress_known_amount"]
                del legacy["progress_known_amount_source"]
                legacy_predictions.append(legacy)
            prediction_path.write_text(
                "".join(
                    json.dumps(item, sort_keys=True) + "\n"
                    for item in legacy_predictions
                ),
                encoding="utf-8",
            )
            legacy_audit = audit_prediction_export(output)
            self.assertEqual(legacy_audit["fitted_package_count"], 0)
            run_path.write_text(original_run, encoding="utf-8")
            prediction_path.write_text(original_predictions, encoding="utf-8")

            altered_metrics = json.loads(
                (output / "metrics.json").read_text(encoding="utf-8")
            )
            altered_experiment = next(iter(altered_metrics))
            altered_candidate = next(iter(altered_metrics[altered_experiment]))
            altered_metrics[altered_experiment][altered_candidate]["metrics"]["mae"] += 1
            (output / "metrics.json").write_text(
                json.dumps(altered_metrics),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "metrics.*mae differs"):
                audit_prediction_export(output)

        self.assertEqual(output.parent, (config.workspace / "experiments").resolve())
        self.assertTrue(output.name.startswith("run-"))
        self.assertLessEqual(len(output.name), 32)
        self.assertEqual(summary.experiment_count, 3)
        self.assertEqual(summary.candidate_run_count, 6)
        self.assertEqual(run_document["run_id"], summary.run_id)
        self.assertEqual(run_document["dataset_id"], summary.dataset_id)
        self.assertIn("task_family_registry_id", run_document)
        self.assertEqual(
            run_document["split_group_registry_id"],
            run_document["task_family_registry_id"],
        )
        self.assertEqual(run_document["experiment_template_count"], 3)
        self.assertEqual(run_document["expanded_experiment_cell_count"], 3)
        self.assertEqual(run_document["paired_feature_comparison_count"], 3)
        self.assertEqual(sum(len(arms) for arms in comparisons.values()), 3)
        self.assertEqual(audit_summary["status"], "passed")
        self.assertEqual(audit_summary["prediction_record_count"], len(prediction_lines))
        self.assertEqual(audit_summary["fitted_package_count"], 30)
        self.assertEqual(audit_summary["paired_comparison_count"], 3)
        cell_authority = run_document["experiment_cell_authority"]
        self.assertEqual(cell_authority["dataset_id"], summary.dataset_id)
        self.assertEqual(cell_authority["policy_id"], "eligible-development-cells-v1")
        cells = cell_authority["cells"]
        template_ids = {
            "task_launch_total",
            "task_update_unknown_remaining",
            "call_pre_unknown_billable",
        }
        self.assertEqual(
            {cell["template_experiment_id"] for cell in cells},
            template_ids,
        )
        self.assertEqual(set(metrics), {cell["experiment_id"] for cell in cells})
        expected_model_cells = {
            f"cell-{index:03d}-{cell['template_experiment_id']}": cell["experiment_id"]
            for index, cell in enumerate(cells, 1)
        }
        self.assertEqual(run_document["model_cells"], expected_model_cells)
        for cell_name, experiment_id in run_document["model_cells"].items():
            self.assertEqual(
                model_cell_documents[cell_name],
                {"experiment_id": experiment_id},
            )
        self.assertTrue(
            all(
                "--condition=" in cell["experiment_id"]
                and "--unit=" in cell["experiment_id"]
                and cell["eligible_row_count"] > 0
                for cell in cells
            )
        )
        self.assertTrue(
            any(
                item["task_family_id"] != item["split_group_id"]
                for item in prediction_lines
            )
        )
        self.assertTrue(prediction_lines)
        self.assertTrue(
            all(
                "model_revision" in item
                and "fitted_artifact_id" in item
                and "eligibility_policy_id" in item
                for item in prediction_lines
            )
        )
        self.assertTrue(
            all(
                item["prediction_export_schema_id"]
                == run_document["prediction_export_schema_id"]
                and item["experiment_run_id"] == run_document["run_id"]
                and item["dataset_id"] == run_document["dataset_id"]
                and item["split_plan_id"] == run_document["split_plan_id"]
                and item["source_id"] == "canonical_jsonl"
                and item["source_event_id"]
                and item["run_id"]
                and item["prediction_context_id"]
                and item["position"]
                and "provider" in item
                and "model_id" in item
                and "reasoning_setting" in item
                and "repository" in item
                and "progress_known_amount" in item
                and item["progress_known_amount_source"]
                and "checkpoint_ordinal" in item
                and item["label_status"] == "observed"
                and isinstance(item["target_value"], int)
                and item["bound_direction"] is None
                and item["bound_value"] is None
                for item in prediction_lines
            )
        )
        self.assertTrue(
            all(
                item["task_family_id"] is not None
                and item["split_group_id"] is not None
                and item["actor_model_id"] == "fixture-model"
                and item["task_type"] == "repo_repair"
                and item["model_family_id"] == "fixture-model-family"
                and item["termination_state"] == "success"
                for item in prediction_lines
            )
        )
        self.assertTrue(
            all(
                item["retry_state"]
                == (
                    None
                    if item["target"]
                    == PredictionTarget.TASK_TOTAL_ACCOUNTED_TOKENS.value
                    else "single_attempt_call"
                )
                for item in prediction_lines
            )
        )
        launch_id = next(
            cell["experiment_id"]
            for cell in cells
            if cell["template_experiment_id"] == "task_launch_total"
        )
        first_candidate = metrics[launch_id]["empirical_quantile"]
        cohorts = first_candidate["cohort_metrics"]
        self.assertEqual(cohorts["aggregation_unit"], "task")
        self.assertEqual(cohorts["task_weighting"], "equal")
        self.assertTrue(cohorts["dimensions"]["task_family_id"])
        self.assertTrue(cohorts["dimensions"]["split_group_id"])
        self.assertEqual(
            [item["value"] for item in cohorts["dimensions"]["termination_state"]],
            ["success"],
        )


if __name__ == "__main__":
    unittest.main()

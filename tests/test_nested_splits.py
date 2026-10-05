from __future__ import annotations

import unittest
from dataclasses import replace

from token_prediction.dataset import (
    INNER_FOLDS,
    assign_inner_task_folds,
    assign_permanent_task_holdout,
)


class NestedSplitTests(unittest.TestCase):
    def test_permanent_holdout_uses_task_identity_only(self) -> None:
        tasks = [f"task-{index:03d}" for index in range(100)]
        first = assign_permanent_task_holdout(tasks)
        duplicated_runs_and_families = [
            task for task in reversed(tasks) for _run_or_family in range(4)
        ]
        second = assign_permanent_task_holdout(duplicated_runs_and_families)
        self.assertEqual(first, second)
        self.assertTrue(first.development_tasks)
        self.assertTrue(first.final_holdout_tasks)
        self.assertFalse(first.development_tasks & first.final_holdout_tasks)
        self.assertEqual(
            first.development_tasks | first.final_holdout_tasks,
            frozenset(tasks),
        )

    def test_permanent_assignment_binds_dataset_without_changing_task_identity(self) -> None:
        assignment = assign_permanent_task_holdout((f"task-{index:03d}" for index in range(100)))
        first = assignment.bind("dataset-a")
        second = assignment.bind("dataset-b")
        self.assertEqual(first.assignment_id, second.assignment_id)
        self.assertNotEqual(first.holdout_plan_id, second.holdout_plan_id)
        self.assertEqual(first.development_tasks, second.development_tasks)
        self.assertEqual(first.final_holdout_tasks, second.final_holdout_tasks)
        with self.assertRaisesRegex(ValueError, "plan id"):
            replace(first, holdout_plan_id="wrong-plan")

        mutated = list(assignment.assignments)
        task, partition = mutated[0]
        mutated[0] = (
            task,
            "final_holdout" if partition == "development" else "development",
        )
        with self.assertRaisesRegex(ValueError, "seeded order"):
            replace(assignment, assignments=tuple(mutated))

    def test_permanent_holdout_fails_closed_for_small_or_empty_partition(self) -> None:
        with self.assertRaisesRegex(ValueError, "too few tasks"):
            assign_permanent_task_holdout((f"task-{index}" for index in range(INNER_FOLDS)))
        with self.assertRaisesRegex(ValueError, "between zero and one"):
            assign_permanent_task_holdout(
                (f"task-{index}" for index in range(6)),
                final_holdout_fraction=0,
            )

    def test_inner_oof_partition_is_holdout_then_next_validation_then_fit(self) -> None:
        tasks = [f"task-{index:02d}" for index in range(15)]
        assignment = assign_inner_task_folds(tasks, seed=20260719)
        self.assertEqual(assignment.folds, 5)
        self.assertEqual(assignment.task_ids, frozenset(tasks))
        for holdout_fold in range(5):
            partition = assignment.partition(holdout_fold)
            expected_holdout = frozenset(
                task for task, fold in assignment.assignments if fold == holdout_fold
            )
            expected_validation = frozenset(
                task for task, fold in assignment.assignments if fold == (holdout_fold + 1) % 5
            )
            self.assertEqual(partition.holdout_tasks, expected_holdout)
            self.assertEqual(partition.validation_tasks, expected_validation)
            self.assertEqual(len(partition.initializer_fit_tasks), 9)
            self.assertEqual(
                partition.initializer_fit_tasks
                | partition.validation_tasks
                | partition.holdout_tasks,
                frozenset(tasks),
            )
            self.assertFalse(partition.holdout_tasks & partition.initializer_fit_tasks)

    def test_each_task_receives_only_its_inner_holdout_model(self) -> None:
        assignment = assign_inner_task_folds(
            (f"task-{index:02d}" for index in range(10)),
            seed=20260720,
        )
        for task, fold in assignment.assignments:
            partition = assignment.partition(fold)
            self.assertIn(task, partition.holdout_tasks)
            self.assertNotIn(task, partition.validation_tasks)
            self.assertNotIn(task, partition.initializer_fit_tasks)

    def test_inner_assignment_is_reproducible_and_empty_folds_fail_closed(self) -> None:
        tasks = [f"task-{index:02d}" for index in range(10)]
        first = assign_inner_task_folds(tasks, seed=20260721)
        second = assign_inner_task_folds(reversed(tasks), seed=20260721)
        self.assertEqual(first, second)
        self.assertNotEqual(
            first.assignment_id,
            assign_inner_task_folds(tasks, seed=20260722).assignment_id,
        )
        with self.assertRaisesRegex(ValueError, "at least five"):
            assign_inner_task_folds(tasks[:4], seed=1)
        with self.assertRaisesRegex(ValueError, "exactly five"):
            assign_inner_task_folds(tasks, seed=1, folds=4)
        mutated = list(first.assignments)
        first_task, first_fold = mutated[0]
        second_index = next(
            index for index, (_task, fold) in enumerate(mutated) if fold != first_fold
        )
        second_task, second_fold = mutated[second_index]
        mutated[0] = (first_task, second_fold)
        mutated[second_index] = (second_task, first_fold)
        with self.assertRaisesRegex(ValueError, "seeded balance policy"):
            replace(first, assignments=tuple(mutated))

    def test_inner_assignment_balances_each_declared_task_cohort(self) -> None:
        tasks = [f"task-{index:02d}" for index in range(20)]
        groups = {
            "scarce-task-seed-labels": tasks[:7],
            "overlapping-update-cohort": tasks[3:16],
        }
        first = assign_inner_task_folds(
            tasks,
            seed=20260719,
            balance_groups=groups,
        )
        second = assign_inner_task_folds(
            reversed(tasks),
            seed=20260719,
            balance_groups={
                "overlapping-update-cohort": reversed(tasks[3:16]),
                "scarce-task-seed-labels": reversed(tasks[:7]),
            },
        )
        self.assertEqual(first, second)
        mapping = first.task_to_fold
        for members in groups.values():
            self.assertEqual({mapping[task] for task in members}, set(range(5)))
        with self.assertRaisesRegex(ValueError, "cannot cover all 5 folds"):
            assign_inner_task_folds(
                tasks,
                seed=20260719,
                balance_groups={"too-small": tasks[:4]},
            )


if __name__ == "__main__":
    unittest.main()

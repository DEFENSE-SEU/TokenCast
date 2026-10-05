from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from token_prediction.api import TokenCast
from token_prediction.collection.readers.codex_task import (
    INPUT_CONTRACT_ID,
    CodexTaskReader,
    CodexTaskSchemaError,
    build_launch_event,
    launch_point,
)
from token_prediction.contracts import EventType, Observable
from token_prediction.dataset.builder import build_supervised_dataset
from token_prediction.dataset.labels import build_task_aggregate_label
from token_prediction.dataset.schema import LabelStatus, PredictionPosition, PredictionTarget
from token_prediction.estimators.lightgbm import FittedLightGBMQuantiles
from token_prediction.estimators.tabular_encoder import FoldTabularEncoder


class CodexTaskTests(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.prompt = "Fix the parser bug.\nKeep the interface.\n"
        self.capture = {
            "schema_version": 1,
            "task_id": "parser",
            "run_id": "run-uuid",
            "thread_id": "thread-1",
            "turn_id": "turn-1",
            "model": "requested",
            "resolved_model": "resolved",
            "provider": "openai-codex",
            "agent_id": "codex",
            "agent_version": "codex-cli 0.157.0",
            "reasoning_effort": "low",
            "started_at": 1_700_000_000,
            "ended_at": 1_700_000_009,
            "turn_status": "completed",
            "interrupt_reason": None,
        }
        usage = {
            "inputTokens": 100,
            "outputTokens": 20,
            "totalTokens": 120,
            "cachedInputTokens": 60,
            "cacheWriteInputTokens": 0,
            "reasoningOutputTokens": 5,
        }
        self.rows = [
            self.row("out", {"id": 2, "method": "thread/start", "params": {"model": "requested"}}),
            self.row(
                "in",
                {
                    "id": 2,
                    "result": {
                        "thread": {"id": "thread-1", "sessionId": "thread-1"},
                        "model": "resolved",
                        "modelProvider": "openai",
                    },
                },
            ),
            self.row(
                "out",
                {
                    "id": 3,
                    "method": "turn/start",
                    "params": {
                        "threadId": "thread-1",
                        "effort": "low",
                        "input": [{"type": "text", "text": self.prompt}],
                    },
                },
            ),
            self.notification("turn/started", {"turn": {"id": "turn-1", "status": "inProgress"}}),
            self.row("in", {"id": 3, "result": {"turn": {"id": "turn-1", "status": "inProgress"}}}),
            self.notification(
                "thread/tokenUsage/updated",
                {"turnId": "turn-1", "tokenUsage": {"total": usage, "last": usage}},
            ),
            self.notification("turn/completed", {"turn": {"id": "turn-1", "status": "completed"}}),
        ]

    def row(self, direction: str, message: dict) -> dict:
        return {"t": 1_700_000_005, "direction": direction, "message": message}

    def notification(self, method: str, params: dict) -> dict:
        return self.row("in", {"method": method, "params": {"threadId": "thread-1"} | params})

    def read(self, rows: list[dict] | None = None):
        (self.root / "capture.json").write_text(json.dumps(self.capture), encoding="utf-8")
        (self.root / "prompt.md").write_bytes(self.prompt.encode())
        (self.root / "events.jsonl").write_text(
            "\n".join(json.dumps(row) for row in (self.rows if rows is None else rows)),
            encoding="utf-8",
        )
        return CodexTaskReader().read(self.root)

    def test_task_scope_and_duplicate_snapshot(self) -> None:
        rows = copy.deepcopy(self.rows)
        rows.insert(6, copy.deepcopy(rows[5]))
        foreign = copy.deepcopy(rows[5])
        foreign["message"]["params"]["threadId"] = "another-thread"
        foreign["message"]["params"]["tokenUsage"]["total"]["totalTokens"] = 99999
        rows.insert(6, foreign)
        foreign_turn = copy.deepcopy(self.rows[5])
        foreign_turn["message"]["params"]["turnId"] = "another-turn"
        rows.insert(6, foreign_turn)
        trajectory = self.read(rows)
        self.assertEqual(
            [event.event_type for event in trajectory.events],
            [EventType.TASK_STARTED, EventType.TASK_FINISHED],
        )
        self.assertTrue(
            all(
                event.logical_call_id is None and event.attempt_id is None
                for event in trajectory.events
            )
        )
        self.assertEqual(
            build_task_aggregate_label(trajectory.events).total_accounted_tokens.value, 120
        )
        dataset = build_supervised_dataset([trajectory])
        self.assertEqual(len(dataset.rows), 1)
        self.assertEqual(dataset.rows[0].point.position, PredictionPosition.TASK_LAUNCH)
        self.assertEqual(dataset.rows[0].label, 120)
        self.assertEqual(
            CodexTaskReader.capabilities.observables,
            frozenset(
                {
                    Observable.TASK_TEXT,
                    Observable.TASK_USAGE,
                    Observable.TASK_AGGREGATE_USAGE,
                    Observable.TASK_TERMINATION,
                }
            ),
        )
        self.assertIn(
            "codex-app-server-reported-input-output-v1",
            trajectory.events[0].payload["token_unit_contract_id"],
        )

    def test_launch_is_available_before_turn_and_stable_after_completion(self) -> None:
        pre_turn = {
            key: value
            for key, value in self.capture.items()
            if key not in {"turn_id", "turn_status", "ended_at", "interrupt_reason"}
        }
        launch = build_launch_event(pre_turn, self.prompt)
        self.assertEqual(launch, self.read().events[0])
        self.assertNotIn("turn_id", launch.payload)
        self.assertEqual(launch.payload["requested_model"], "requested")
        self.assertEqual(launch.payload["configured_model"], "resolved")
        self.assertEqual(self.read().events[-1].payload["turn_id"], "turn-1")
        point = launch_point(launch)
        self.assertEqual(point, launch_point(self.read().events[0]))
        self.assertEqual(point.features["task_char_count"], len(self.prompt.strip()))
        self.assertEqual(len(point.features["task_embedding"]), 256)
        self.assertEqual(point.cutoff_event_seq, 0)
        self.assertEqual(point.known_offset_tokens, 0)
        self.assertIsNone(point.logical_call_id)
        self.assertIsNone(point.attempt_id)

    def test_delayed_usage_uses_last_matching_total_once(self) -> None:
        rows = copy.deepcopy(self.rows)
        delayed = copy.deepcopy(rows[5])
        delayed["t"] += 1
        delayed["message"]["params"]["tokenUsage"]["total"].update(outputTokens=21, totalTokens=121)
        delayed["message"]["params"]["tokenUsage"]["last"] = {
            "inputTokens": 1,
            "outputTokens": 1,
            "totalTokens": 2,
        }
        rows.extend((delayed, copy.deepcopy(delayed)))
        foreign = copy.deepcopy(delayed)
        foreign["message"]["params"]["turnId"] = "another-turn"
        foreign["message"]["params"]["tokenUsage"]["total"]["totalTokens"] = 999
        rows.append(foreign)
        trajectory = self.read(rows)
        self.assertEqual(
            build_task_aggregate_label(trajectory.events).total_accounted_tokens.value, 121
        )
        self.assertEqual(trajectory.events[-1].payload["turn_id"], "turn-1")

    def native_runtime(self):
        self.capture.update(
            sandbox="workspace-write",
            task_type="repair",
            approval_policy="never",
            harness_config=["features.multi_agent=false"],
        )
        self.rows[0]["message"]["params"].update(
            sandbox=self.capture["sandbox"], approvalPolicy=self.capture["approval_policy"]
        )
        observation = {
            "mode": "codex",
            "launch": {
                key: value
                for key, value in self.capture.items()
                if key not in {"turn_id", "turn_status", "ended_at", "interrupt_reason"}
            },
            "prompt": self.prompt,
        }
        point = launch_point(build_launch_event(observation["launch"], self.prompt))
        encoder = FoldTabularEncoder.fit((point,))
        quantiles = (0.05, 0.5, 0.95)
        model = FittedLightGBMQuantiles(
            estimator_id="native-test-model",
            target=PredictionTarget.TASK_TOTAL_ACCOUNTED_TOKENS,
            position=PredictionPosition.TASK_LAUNCH,
            dataset_id="native-test-dataset",
            input_contract_id=INPUT_CONTRACT_ID,
            token_unit_contract_id=point.token_unit_contract_id,
            allowed_condition_ids=(point.condition_id,),
            encoder=encoder,
            boosters={
                q: SimpleNamespace(predict=lambda matrix, value=value, **kwargs: [value])
                for q, value in zip(quantiles, (100, 110, 130))
            },
            best_iterations={q: 1 for q in quantiles},
            quantiles=quantiles,
            fit_report=SimpleNamespace(feature_set_id=encoder.schema.schema_id),
        )
        loader = patch(
            "token_prediction.estimators.lightgbm_bundle.load_lightgbm_bundle",
            return_value=model,
        )
        load_mock = loader.start()
        self.addCleanup(loader.stop)
        bundle = self.root / "model"
        bundle.mkdir()
        manifest = {
            "bundle_schema_version": 3,
            "bundle_format": "lightgbm-text",
            "position": "task_launch",
            "target": "task_total_accounted_tokens",
            "encoder": {"path": "encoder.json"},
            "models": {"q50": "model.txt"},
            "model_id": "native-test-model",
            "model_revision": "native-test-revision",
            "deployment_scope": {
                "client": "codex",
                "validation_only": True,
                "approval_policy": "never",
                "harness_config": ["features.multi_agent=false"],
            },
        }
        (bundle / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        (bundle / "encoder.json").write_text("{}", encoding="utf-8")
        (bundle / "model.txt").write_text("fake booster", encoding="utf-8")
        return TokenCast(bundle, self.root / "usage.sqlite3"), observation, load_mock

    def test_native_api_pre_turn_prediction_and_final_usage_pairing(self) -> None:
        service, observation, loader = self.native_runtime()
        self.assertEqual(service.model_status()["supported_modes"], ["codex"])
        replay = service._codex_observation(observation)
        replay["mode"] = "replay"
        replay["context"]["runtime_mode"] = "offline"
        rejected = service.predict_tokens(replay)
        self.assertEqual(rejected["status"], "unsupported_scope", rejected)
        self.assertNotIn("forecast", rejected)
        self.assertFalse(service.data_path.exists())
        prediction = service.predict_tokens(observation)
        self.assertEqual(prediction["status"], "ok", prediction)
        self.assertEqual(prediction["forecast"], {"lower": 100, "point": 110, "upper": 130})
        self.assertNotIn("turn_id", prediction["scope"])
        self.assertEqual(service.predict_tokens(observation), {**prediction, "status": "duplicate"})
        changed = {**observation, "prompt": "A different task"}
        self.assertEqual(service.predict_tokens(changed)["status"], "conflict")
        other = copy.deepcopy(observation)
        other["launch"].update(thread_id="thread-2", run_id="run-2")
        second = service.predict_tokens(other)
        self.assertEqual(second["status"], "ok", second)
        self.assertNotEqual(second["prediction_id"], prediction["prediction_id"])
        trajectory = self.read()
        self.assertEqual(prediction["scope"]["source_event_id"], trajectory.events[0].event_id)
        actual = {
            "event_id": trajectory.events[-1].event_id,
            "scope": prediction["scope"],
            "label_status": "observed",
            "value": build_task_aggregate_label(trajectory.events).total_accounted_tokens.value,
            "source": "codex_task_v1",
        }
        pending = {**actual, "label_status": "missing", "value": None}
        self.assertEqual(
            service.record_usage(prediction["prediction_id"], pending)["status"], "pending"
        )
        wrong = {**actual, "scope": second["scope"]}
        self.assertEqual(
            service.record_usage(prediction["prediction_id"], wrong)["status"], "scope_mismatch"
        )
        for value in (120.5, True, -1, 2**63):
            with self.subTest(actual=value):
                self.assertEqual(
                    service.record_usage(prediction["prediction_id"], {**actual, "value": value})[
                        "status"
                    ],
                    "invalid_input",
                )
        store = service._store()
        with patch.object(service, "_store", return_value=store), patch.object(
            store, "get_prediction", return_value={**prediction, "mode": "replay"}
        ):
            self.assertEqual(
                service.record_usage(prediction["prediction_id"], {**actual, "value": 120.5})[
                    "status"
                ],
                "invalid_input",
            )
        matched = service.record_usage(prediction["prediction_id"], actual)
        self.assertEqual(matched["status"], "matched", matched)
        self.assertEqual(matched["actual"], 120)
        recovered = TokenCast(service.model_path, service.data_path)
        self.assertEqual(recovered.predict_tokens(observation)["status"], "duplicate")
        self.assertEqual(
            recovered.record_usage(prediction["prediction_id"], actual)["status"], "duplicate"
        )
        self.assertEqual(loader.call_count, 2)

    def test_native_api_rejects_future_metadata_and_incompatible_configuration(self) -> None:
        service, original, _ = self.native_runtime()
        mutations = [
            ("turn_id", "future-turn", "invalid_input"),
            ("ended_at", 1_700_000_009, "invalid_input"),
            ("schema_version", True, "invalid_input"),
            ("approval_policy", "on-request", "invalid_input"),
            ("harness_config", [], "invalid_input"),
            ("resolved_model", "other-model", "unsupported_scope"),
            ("agent_version", "codex-cli 0.158.0", "unsupported_scope"),
            ("reasoning_effort", "high", "unsupported_scope"),
            ("sandbox", "read-only", "unsupported_scope"),
            ("sandbox", "", "invalid_input"),
            ("task_type", None, "invalid_input"),
        ]
        for key, value, expected in mutations:
            observation = copy.deepcopy(original)
            observation["launch"][key] = value
            with self.subTest(key=key):
                result = service.predict_tokens(observation)
                self.assertEqual(result["status"], expected, result)
        result = service.predict_tokens({**original, "prompt": ""})
        self.assertEqual(result["status"], "invalid_input", result)
        self.assertFalse(service.data_path.exists())

    def test_scope_proof_and_multi_agent_rejections(self) -> None:
        cases = []
        wrong_thread = copy.deepcopy(self.rows)
        wrong_thread[1]["message"]["result"]["thread"]["id"] = "wrong"
        cases.append(wrong_thread)
        wrong_turn = copy.deepcopy(self.rows)
        wrong_turn[4]["message"]["result"]["turn"]["id"] = "wrong"
        cases.append(wrong_turn)
        resumed = copy.deepcopy(self.rows)
        resumed[0]["message"]["method"] = "thread/resume"
        cases.append(resumed)
        multiple = copy.deepcopy(self.rows)
        multiple.insert(5, copy.deepcopy(multiple[2]))
        cases.append(multiple)
        other_started = copy.deepcopy(self.rows)
        other_started.insert(5, self.notification("turn/started", {"turn": {"id": "second"}}))
        cases.append(other_started)
        collab = copy.deepcopy(self.rows)
        collab.insert(
            5,
            self.notification(
                "item/started", {"turnId": "turn-1", "item": {"type": "collabAgentToolCall"}}
            ),
        )
        cases.append(collab)
        child = copy.deepcopy(self.rows)
        child.insert(
            5,
            self.row(
                "in",
                {
                    "method": "thread/started",
                    "params": {"thread": {"id": "child", "parentThreadId": "thread-1"}},
                },
            ),
        )
        cases.append(child)
        wrong_prompt = copy.deepcopy(self.rows)
        wrong_prompt[2]["message"]["params"]["input"][0]["text"] = "A different task"
        cases.append(wrong_prompt)
        for index, field, value in (
            (0, "config", {"model_provider": "other"}),
            (2, "model", "other-model"),
            (2, "sandboxPolicy", {"type": "dangerFullAccess"}),
        ):
            overridden = copy.deepcopy(self.rows)
            overridden[index]["message"]["params"][field] = value
            cases.append(overridden)
        for rows in cases:
            with self.subTest(message=rows[0]["message"]), self.assertRaises(CodexTaskSchemaError):
                self.read(rows)

    def test_capture_window_and_thread_configuration_are_enforced(self) -> None:
        self.capture.update(sandbox="workspace-write", approval_policy="never")
        self.rows[0]["message"]["params"].update(sandbox="workspace-write", approvalPolicy="never")
        self.assertEqual(self.read().events[-1].event_type, EventType.TASK_FINISHED)
        for field, value in (("sandbox", "read-only"), ("approvalPolicy", "on-request")):
            rows = copy.deepcopy(self.rows)
            rows[0]["message"]["params"][field] = value
            with (
                self.subTest(field=field),
                self.assertRaisesRegex(CodexTaskSchemaError, "disagrees with thread/start"),
            ):
                self.read(rows)
        for timestamp in (self.capture["started_at"] - 1, self.capture["ended_at"] + 1):
            rows = copy.deepcopy(self.rows)
            rows[5]["t"] = timestamp
            with (
                self.subTest(timestamp=timestamp),
                self.assertRaisesRegex(CodexTaskSchemaError, "outside the capture window"),
            ):
                self.read(rows)

    def test_usage_validation_and_decreasing_snapshot(self) -> None:
        mutations = [
            ("inputTokens", True),
            ("inputTokens", 100.0),
            ("inputTokens", -1),
            ("totalTokens", 121),
            ("cachedInputTokens", 101),
            ("reasoningOutputTokens", 21),
            ("cacheWriteInputTokens", None),
            ("inputTokens", float("nan")),
        ]
        for key, value in mutations:
            rows = copy.deepcopy(self.rows)
            rows[5]["message"]["params"]["tokenUsage"]["total"][key] = value
            with self.subTest(key=key, value=value), self.assertRaises(CodexTaskSchemaError):
                self.read(rows)
        rows = copy.deepcopy(self.rows)
        lower = copy.deepcopy(rows[5])
        lower["message"]["params"]["tokenUsage"]["total"] |= {"inputTokens": 99, "totalTokens": 119}
        rows.insert(6, lower)
        with self.assertRaisesRegex(CodexTaskSchemaError, "decreased"):
            self.read(rows)

    def test_missing_usage_and_interruption_do_not_become_complete_labels(self) -> None:
        missing = self.read(self.rows[:5] + self.rows[6:])
        self.assertEqual(
            build_task_aggregate_label(missing.events).total_accounted_tokens.status,
            LabelStatus.MISSING,
        )
        self.capture["turn_status"] = "interrupted"
        self.capture["interrupt_reason"] = "timeout"
        self.rows[-1]["message"]["params"]["turn"]["status"] = "interrupted"
        aborted = self.read()
        self.assertEqual(aborted.events[-1].event_type, EventType.TASK_ABORTED)
        self.assertIsNone(aborted.events[-1].payload["usage"])
        self.assertEqual(aborted.events[-1].payload["partial_usage"]["total_tokens"], 120)
        self.assertEqual(
            build_task_aggregate_label(aborted.events).total_accounted_tokens.status,
            LabelStatus.CENSORED,
        )


if __name__ == "__main__":
    unittest.main()

from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import multiprocessing

import pytest

from token_prediction.usage_store import StoreConflict, UsageStore


def _record_from_process(path):
    store = UsageStore(path)
    prediction = store.save_prediction("p", {"tokens": 42})
    actual = store.record_actual("p", "event", {"input": 10, "output": 20})
    return prediction["status"], actual["status"]


def test_restart_idempotence_and_immutable_records(tmp_path):
    path = tmp_path / "nested" / "usage.sqlite3"
    store = UsageStore(path)
    record = {"tokens": 42, "raw": {"text": "真实输入", "optional": None}}
    assert store.save_prediction("p", record)["status"] == "recorded"
    record["tokens"] = 99
    assert UsageStore(path).get_prediction("p")["tokens"] == 42
    original = store.get_prediction("p")
    assert UsageStore(path).save_prediction("p", original)["status"] == "duplicate"
    with pytest.raises(StoreConflict):
        store.save_prediction("p", record)
    with pytest.raises(ValueError, match="unknown prediction"):
        store.record_actual("unknown", "event", {})
    assert store.record_actual("p", "event", {"input": 10})["status"] == "recorded"
    assert UsageStore(path).record_actual("p", "event", {"input": 10})["status"] == "duplicate"
    for event_id, actual in [("event", {"input": 11}), ("other", {"input": 10})]:
        with pytest.raises(StoreConflict):
            store.record_actual("p", event_id, actual)
    store.save_prediction("other", {})
    with pytest.raises(StoreConflict):
        store.record_actual("other", "event", {"input": 10})
    assert store.get_actual("other") is None
    assert store.get_actual("p") == {"event_id": "event", "record": {"input": 10}}
    assert store.get_prediction("missing") is None
    assert store.get_actual("missing") is None


def test_two_connections_race_to_pair_same_event(tmp_path):
    path = tmp_path / "usage.sqlite3"
    stores = [UsageStore(path), UsageStore(path)]
    for prediction_id in ("first", "second"):
        stores[0].save_prediction(prediction_id, {})

    def pair(item):
        store, prediction_id = item
        try:
            return store.record_actual(prediction_id, "shared-event", {})["status"]
        except StoreConflict:
            return "conflict"

    with ThreadPoolExecutor(max_workers=2) as executor:
        statuses = list(executor.map(pair, zip(stores, ("first", "second"))))
    assert sorted(statuses) == ["conflict", "recorded"]
    assert sum(stores[0].get_actual(key) is not None for key in ("first", "second")) == 1


def test_multiple_processes_record_each_snapshot_once(tmp_path):
    path = tmp_path / "usage.sqlite3"
    with ProcessPoolExecutor(
        max_workers=2, mp_context=multiprocessing.get_context("spawn")
    ) as pool:
        results = list(pool.map(_record_from_process, [path] * 4))
    assert [status for status, _ in results].count("recorded") == 1
    assert [status for _, status in results].count("recorded") == 1
    assert UsageStore(path).get_actual("p")["record"] == {"input": 10, "output": 20}


@pytest.mark.parametrize(
    "record",
    [[], {1: "key"}, {"x": (1, 2)}, {"x": float("nan")}, {"x": float("inf")}, {"x": object()}],
)
def test_rejects_non_json_records(tmp_path, record):
    store = UsageStore(tmp_path / "usage.sqlite3")
    with pytest.raises(ValueError):
        store.save_prediction("p", record)
    store.save_prediction("p", {})
    with pytest.raises(ValueError):
        store.record_actual("p", "e", record)
    assert store.get_actual("p") is None


def test_rejects_invalid_ids_and_cycles(tmp_path):
    store = UsageStore(tmp_path / "usage.sqlite3")
    for invalid in (None, 1, "", "   "):
        with pytest.raises(ValueError):
            store.save_prediction(invalid, {})
        with pytest.raises(ValueError):
            store.get_prediction(invalid)
        with pytest.raises(ValueError):
            store.get_actual(invalid)
        with pytest.raises(ValueError):
            store.record_actual("p", invalid, {})
    cyclic = {}
    cyclic["self"] = cyclic
    with pytest.raises(ValueError):
        store.save_prediction("p", cyclic)

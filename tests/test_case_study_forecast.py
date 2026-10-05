import pytest

from token_prediction.case_study.forecast import calibrate, summarize


def rows(n):
    return [dict(task_id=str(i), lower=10, upper=20, actual=30+i) for i in range(n)]


def test_small_calibration_does_not_claim_finite_90_percent_interval():
    small = calibrate("call_start", rows(8), training_task_ids=[])
    assert small.expansion is None
    full = calibrate("call_start", rows(9), training_task_ids=[])
    assert full.expansion == 18


def test_task_grouping_and_disjointness():
    fitted = calibrate("call_start", rows(9) + rows(9), training_task_ids=[])
    assert len(fitted.task_ids) == 9
    with pytest.raises(ValueError):
        calibrate("call_start", rows(9), training_task_ids=["0"])
    with pytest.raises(ValueError):
        summarize([10, 20], point="call_start", task_id="0", calibration=fitted)


def test_confirmed_floor_and_remaining_total():
    result = summarize([0, 10, 20], point="in_call_update", task_id="x",
                       confirmed_in_scope=50)
    assert result["raw_interval"] == [50, 50]
    assert result["interval"] is None
    terminal = summarize([], point="task_update", task_id="x", spent=123,
                         terminal_actual=0)
    assert terminal["updated_task_total"] == 123
    assert terminal["interval"] == [0, 0]


def test_invalid_samples_and_wrong_point():
    with pytest.raises(ValueError):
        summarize([float("nan")], point="task_start", task_id="x")
    fitted = calibrate("task_start", rows(9), training_task_ids=[])
    with pytest.raises(ValueError):
        summarize([1], point="call_start", task_id="x", calibration=fitted)

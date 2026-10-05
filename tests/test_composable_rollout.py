import numpy as np
import torch
from token_prediction.estimators.composable_segment import ComposableSegmentModel, SegmentConfig


def setup(monkeypatch, *, segments=8, known=False, stop=True, mixed=False):
    model = ComposableSegmentModel(
        SegmentConfig(
            2,
            2,
            1,
            max_horizon=4,
            max_rollout_segments=segments,
            use_call_count=True,
            use_task_size=True,
        )
    )
    model.statistics = {"ready": torch.tensor(1.0)}
    query = dict(
        task=np.array([[2.0, 3.0]]),
        state=np.array([[1.0, 0.0]]),
        observation=np.ones((1, 1)),
        horizon=np.array([4]),
        input_length=np.array([10.0]),
        input_known=np.array([1.0]),
        spent=np.array([7.0]),
        allowance=np.array([4.0]),
        call_limit_known=np.array([known]),
        completed_calls=np.array([3.0]),
        task_text_bytes=np.array([123.0]),
        request_group=np.array([2]),
        request_surface_complete=np.array([True]),
    )
    seen = []

    def segment(n, g, cost, initial, end, terminal):
        return dict(
            n=n,
            g=g,
            b=cost - n * initial,
            cost=cost,
            initial_length=initial,
            outgoing_length=initial + g,
            end_state=end,
            terminal=terminal,
        )

    def initial(q, left, right, samples, *, differentiable):
        shape = (len(q["horizon"]), samples)
        ones = torch.ones(shape)
        end = torch.zeros((*shape, 2))
        end[..., 0] = 5
        terminal = ones.clone() if known else torch.zeros(shape)
        if mixed:
            terminal[:, 0] = 1
        total = segment(ones * 4, ones * 8, ones * 70, ones * 10, end, terminal)
        first = segment(ones, ones * 2, ones * 13, ones * 10, end, torch.zeros(shape))
        return total, first

    def draw(q, samples, *, differentiable):
        seen.append({k: v.clone() for k, v in q.items()})
        shape = (len(q["horizon"]), samples)
        ones = torch.ones(shape)
        initial = q["input_length"][:, None].expand(shape)
        n = ones * (2 if stop else 4)
        g = ones * (-2 if stop else 1)
        cost = ones * 50 if stop else n * initial + 5
        end = q["state"][:, None, :].expand(*shape, 2).clone()
        end[..., 0] += 1
        terminal = ones if stop else torch.zeros(shape)
        return segment(n, g, cost, initial, end, terminal)

    monkeypatch.setattr(model, "_compose_draw", initial)
    monkeypatch.setattr(model, "_draw", draw)
    return model, query, seen


def test_unknown_limit_particle_continues_with_exact_segment_accounting(monkeypatch):
    model, q, seen = setup(monkeypatch)
    result = model.forecast(q, samples=3)
    assert len(seen) == 1
    np.testing.assert_allclose(result["n"], 6)
    np.testing.assert_allclose(result["g"], 6)
    np.testing.assert_allclose(result["b"], 60)
    np.testing.assert_allclose(result["cost"], 120)
    np.testing.assert_allclose(result["cost"], result["n"] * result["initial_length"] + result["b"])
    np.testing.assert_allclose(result["task_total_samples"], 127)
    np.testing.assert_allclose(result["call_samples"], 13)
    assert not result["horizon_exhausted"].any()
    nextq = seen[0]
    assert torch.all(nextq["input_known"] == 1) and torch.all(nextq["input_length"] == 18)
    assert torch.all(nextq["spent"] == 77) and torch.all(nextq["completed_calls"] == 7)
    assert torch.all(nextq["horizon"] == 4) and torch.all(nextq["allowance"] == 4)
    assert torch.all(nextq["call_limit_known"] == 0) and torch.all(nextq["observation"] == 0)
    assert not nextq["request_surface_complete"].any()
    assert torch.all(nextq["task_text_bytes"] == 123) and torch.all(nextq["request_group"] == 2)
    torch.testing.assert_close(nextq["task"], torch.tensor([[2.0, 3.0]]).expand_as(nextq["task"]))


def test_observed_call_limit_does_not_trigger_continuation(monkeypatch):
    model, q, seen = setup(monkeypatch, known=True)
    result = model.forecast(q, samples=3)
    assert not seen
    np.testing.assert_allclose(result["cost"], 70)
    assert not result["horizon_exhausted"].any()


def test_legacy_single_segment_guard_keeps_unresolved_samples_visible(monkeypatch):
    assert SegmentConfig(2, 2, 1).max_rollout_segments == 1
    model, q, seen = setup(monkeypatch, segments=1)
    result = model.forecast(q, samples=3)
    assert not seen
    np.testing.assert_allclose(result["cost"], 70)
    assert result["horizon_exhausted"].all()


def test_eight_chunk_computation_limit_retains_unresolved_marker(monkeypatch):
    model, q, seen = setup(monkeypatch, segments=8, stop=False)
    result = model.forecast(q, samples=3)
    assert len(seen) == 7
    np.testing.assert_allclose(result["n"], 32)
    assert result["horizon_exhausted"].all()
    np.testing.assert_allclose(result["call_samples"], 13)
    np.testing.assert_allclose(result["cost"], result["n"] * result["initial_length"] + result["b"])


def test_continuation_preserves_already_absorbed_particles(monkeypatch):
    model, q, seen = setup(monkeypatch, mixed=True)
    result = model.forecast(q, samples=3)
    assert len(seen) == 1
    assert result["cost"][0, 0] == 70
    np.testing.assert_allclose(result["cost"][0, 1:], 120)
    np.testing.assert_allclose(result["call_samples"], 13)

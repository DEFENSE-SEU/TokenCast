import math

import numpy as np
import pytest

torch = pytest.importorskip("torch")
scipy_special = pytest.importorskip("scipy.special")

from token_prediction.estimators.composable_segment import (
    ComposableSegmentModel,
    SegmentConfig,
)


def data():
    return {"task": np.zeros((6, 2)), "state": np.zeros((6, 1)),
        "observation": np.zeros((6, 1)), "horizon": np.array([1., 2., 4., 4., 4., 1.]),
        "allowance": np.full(6, 4.), "input_length": np.full(6, 100.),
        "input_known": np.ones(6), "spent": np.zeros(6),
        "n": np.array([1., 2., 2., 4., 4., 1.]),
        "terminal": np.array([0., 0., 1., 1., 0., 1.]),
        "g": np.zeros(6), "b": np.full(6, 100.),
        "end_state": np.zeros((6, 1)), "call_limit_known": np.zeros(6)}


def model():
    result = ComposableSegmentModel(SegmentConfig(2, 1, 1, max_horizon=4,
        hidden_dim=12, count_distribution="discretized_lognormal"))
    result._fit_statistics(result._data(data(), labels=True))
    return result


def fixed_parameters(model, location=math.log(5), scale=.8):
    with torch.no_grad():
        model.network["count"].weight.zero_()
        model.network["count"].bias.copy_(torch.tensor([location, math.log(math.expm1(scale - .05))]))


def test_log_count_probabilities_match_independent_gaussian_intervals_and_gradients():

    n, mu, sigma = np.meshgrid([1., 2., 4., 50., 500.], [-20., -2., 0., 3., 20.], [.05, .3, 1., 5.], indexing="ij")
    n, mu, sigma = [x.ravel() for x in (n, mu, sigma)]
    upper = (np.log(n) - mu) / sigma
    lower = (np.log(np.maximum(n - 1, 1)) - mu) / sigma
    large = np.where(lower > 0, scipy_special.log_ndtr(-lower), scipy_special.log_ndtr(upper))
    small = np.where(lower > 0, scipy_special.log_ndtr(-upper), scipy_special.log_ndtr(lower))
    difference = np.where(n > 1, small - large, -1.)
    expected = np.where(n > 1, large + np.log(-np.expm1(difference)), scipy_special.log_ndtr(upper))
    loc = torch.tensor(mu, requires_grad=True)
    scale = torch.tensor(sigma, requires_grad=True)
    actual = ComposableSegmentModel._count_log_pmf(torch.tensor(n), loc, scale)
    np.testing.assert_allclose(actual.detach().numpy(), expected, rtol=1e-9, atol=1e-8)
    actual.sum().backward()
    assert torch.isfinite(loc.grad).all() and torch.isfinite(scale.grad).all()
    boundary = torch.tensor([0., 1., 2., 500.], dtype=torch.float64)
    survival = ComposableSegmentModel._count_log_survival(boundary, torch.tensor(2.), torch.tensor(.7))
    expected_sf = scipy_special.log_ndtr((2 - np.log([1., 1., 2., 500.])) / float(torch.tensor(.7)))
    expected_sf[0] = 0
    np.testing.assert_allclose(survival.numpy(), expected_sf, atol=1e-10)
    assert torch.autograd.gradcheck(lambda a, b: ComposableSegmentModel._count_log_pmf(
        torch.tensor([1., 2., 500.]), a, b).sum(),
        (torch.tensor([.2, 3., 5.], dtype=torch.float64, requires_grad=True),
         torch.tensor([.6, .8, .3], dtype=torch.float64, requires_grad=True)))


def test_exact_count_and_termination_likelihood_includes_execution_cap_tail():
    m = model()
    fixed_parameters(m)
    q = m._data(data(), labels=True)
    location, scale = m._count_parameters(q)
    nll, stop = m._count_stop_nll(q, m._encode(q))
    point = m._count_log_pmf(q["n"], location, scale)
    tail = m._count_log_survival(q["horizon"], location, scale)
    torch.testing.assert_close(nll, -torch.where(q["terminal"] > .5, point, tail))
    assert (stop == 0).all()
    capped = {key: value.clone() for key, value in q.items()}
    capped["call_limit_known"][:] = 1
    capped["terminal"][4] = 1
    capped["allowance"][5] = 1
    nll, _ = m._count_stop_nll(capped, m._encode(capped))
    expected = -m._count_log_survival(torch.tensor(3.), location[3:5], scale[3:5])
    torch.testing.assert_close(nll[3:5], expected)
    assert float(nll[5].detach()) == 0
    assert torch.isfinite(m._nll(capped)).all()


def test_count_parameters_ignore_computation_horizon_and_old_config_stays_categorical(tmp_path):
    m = model()
    q = m._data(data(), labels=False)
    a = m._count_parameters(q)
    altered = {**q, "horizon": torch.full_like(q["horizon"], 4)}
    b = m._count_parameters(altered)
    for first, second in zip(a, b):
        torch.testing.assert_close(first, second, atol=0, rtol=0)
    assert m.network["count"].out_features == 2
    assert not list(m.network["stop"].parameters())
    m.save(tmp_path / "new.pt")
    restored = ComposableSegmentModel.load(tmp_path / "new.pt")
    for key, value in m.sample(data(), samples=64, seed=5).items():
        np.testing.assert_array_equal(value, restored.sample(data(), samples=64, seed=5)[key])
    old = ComposableSegmentModel(SegmentConfig(2, 1, 1, max_horizon=4))
    old._fit_statistics(old._data(data(), labels=True))
    old.save(tmp_path / "old.pt")
    saved = torch.load(tmp_path / "old.pt", weights_only=True)
    saved["config"].pop("count_distribution")
    torch.save(saved, tmp_path / "old.pt")
    loaded = ComposableSegmentModel.load(tmp_path / "old.pt")
    assert loaded.config.count_distribution == "categorical"
    np.testing.assert_array_equal(old.sample(data(), samples=16, seed=9)["cost"],
                                  loaded.sample(data(), samples=16, seed=9)["cost"])


def test_joint_sampling_frequencies_single_call_tail_and_straight_through():
    m = model()
    fixed_parameters(m)
    q = {key: value[:1].clone() for key, value in m._data(data(), labels=False).items()}
    q["horizon"][:] = 3
    loc, scale = m._count_parameters(q)
    count, terminal, _ = m._parametric_count_draw(q, 30000, False)
    for value in (1, 2, 3):
        observed = ((count == value) & (terminal == 1)).float().mean()
        expected = m._count_log_pmf(torch.tensor(float(value)), loc, scale).exp()[0]
        assert float(observed) == pytest.approx(float(expected.detach()), abs=.012)
    assert float((terminal == 0).float().mean()) == pytest.approx(
        float(m._count_log_survival(torch.tensor(3.), loc, scale).exp()[0].detach()), abs=.012)
    q["horizon"][:] = 1
    q["allowance"][:] = 1
    q["call_limit_known"][:] = 1
    sampled = m._draw(q, 128, differentiable=True)
    assert (sampled["n"] == 1).all() and (sampled["terminal"] == 1).all()
    torch.testing.assert_close(sampled["g"], torch.zeros_like(sampled["g"]), atol=1e-5, rtol=0)

    count, terminal, single = m._parametric_count_draw(q, 128, True)
    torch.testing.assert_close(single, torch.ones_like(single), atol=1e-6, rtol=0)
    q["horizon"][:] = 3
    q["allowance"][:] = 4
    q["call_limit_known"][:] = 0
    count, terminal, _ = m._parametric_count_draw(q, 256, True)
    (count.mean() + terminal.mean()).backward()
    assert torch.isfinite(m.network["count"].bias.grad).all()
    assert m.network["count"].bias.grad.abs().sum() > 0
    composed, first = m._compose_draw(q, torch.ones(1), torch.full((1,), 2), 64, differentiable=True)
    torch.testing.assert_close(composed["cost"], composed["n"] * composed["initial_length"] + composed["b"], atol=.01, rtol=1e-5)
    assert torch.isfinite(first["cost"]).all()


def test_empty_horizon_and_absorbing_state_contract():
    m = model()
    query = {key: value[:1].copy() for key, value in data().items()}
    query["horizon"][:] = 0
    empty = m.sample(query, samples=16)
    assert (empty["n"] == 0).all() and (empty["cost"] == 0).all() and (empty["terminal"] == 0).all()
    query["is_terminal"] = np.ones(1)
    absorbed = m.sample(query, samples=16)
    assert (absorbed["n"] == 0).all() and (absorbed["terminal"] == 1).all()

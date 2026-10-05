import numpy as np
import pytest

torch = pytest.importorskip("torch")

from token_prediction.estimators.composable_segment import (
    ComposableSegmentModel,
    SegmentConfig,
    compose_segments,
    energy_distance,
)


def _data(rows=12):
    n = np.tile([1, 2, 3], rows // 3).astype(float)
    length = np.linspace(100, 200, rows)

    cost = length + 40
    g = -length * 0.2
    terminal = n < 3
    g[n == 1] = 0
    cost += (length + g) * terminal * (n > 1)
    return {"task": np.zeros((rows, 2)), "state": np.ones((rows, 2)),
            "observation": np.zeros((rows, 1)), "horizon": np.full(rows, 3),
            "input_length": length, "input_known": np.arange(rows) % 2,
            "spent": np.arange(rows) * 100, "allowance": np.full(rows, 4),
            "n": n, "g": g, "b": cost - n * length,
            "end_state": np.ones((rows, 2)) * 2, "terminal": terminal,
            "weight": np.ones(rows)}


def test_composition_associativity_and_signed_cost():
    a, b, c = ({"n": 2, "g": -100, "b": -40},
               {"n": 1, "g": 30, "b": 70}, {"n": 3, "g": -20, "b": 12})
    left = compose_segments(compose_segments(a, b), c)
    right = compose_segments(a, compose_segments(b, c))
    assert left == right
    assert left["n"] * 500 + left["b"] == (
        a["n"] * 500 + a["b"] + b["n"] * 400 + b["b"] + c["n"] * 430 + c["b"])


@pytest.mark.parametrize("energy_coordinates", ["physical", "asinh"])
def test_pilot_joint_training_accounting_and_roundtrip(tmp_path, energy_coordinates):
    train, validation = _data(), _data()
    validation["task"][:] = 1000
    config = SegmentConfig(2, 2, 1, max_horizon=4, hidden_dim=12,
                           max_epochs=2, batch_size=12, consistency_samples=3,
                           energy_coordinates=energy_coordinates)
    model = ComposableSegmentModel(config).fit(train, validation)
    assert torch.equal(model.statistics["task_mean"], torch.zeros(2))
    assert np.isfinite(model.fit_report["history"][0]["train_energy"])
    q = {key: value for key, value in train.items() if key in model.INPUT_KEYS}
    samples = model.sample(q, samples=10)
    np.testing.assert_allclose(samples["cost"], samples["n"] * samples["initial_length"]
                               + samples["b"], rtol=1e-5)
    assert (samples["cost"] >= samples["initial_length"]).all()
    assert (samples["outgoing_length"] >= 0).all()
    assert (samples["terminal"][samples["n"] < 3] == 1).all()
    q["confirmed_call"] = np.full(12, 500)
    forecast = model.forecast(q, samples=10, seed=4)
    assert (forecast["call_samples"] >= 500).all()
    np.testing.assert_allclose(forecast["task_total_samples"],
        q["spent"][:, None] + forecast["task_remaining_samples"])
    path = tmp_path / "pilot.pt"
    model.save(path)
    loaded = ComposableSegmentModel.load(path)
    np.testing.assert_array_equal(forecast["task_total_samples"],
                                  loaded.forecast(q, samples=10, seed=4)["task_total_samples"])


def test_unknown_input_label_is_masked_and_terminal_is_absorbing():
    data = _data()
    model = ComposableSegmentModel(SegmentConfig(2, 2, 1, max_horizon=4, max_epochs=1))
    q = model._data(data, labels=True)
    model._fit_statistics(q)
    q["input_known"][:] = 0
    hidden = model._encode(q).detach()
    q["input_length"] += 10000
    torch.testing.assert_close(hidden, model._encode(q))
    q["is_terminal"][:] = 1
    result = model._draw(q, 8, differentiable=False)
    assert (result["n"] == 0).all() and (result["cost"] == 0).all()
    assert (result["g"] == 0).all() and (result["b"] == 0).all()
    torch.testing.assert_close(result["end_state"], q["state"][:, None].expand(-1, 8, -1))


@pytest.mark.parametrize("first_probability", [0.0, 0.5, 1.0])
def test_energy_and_conditional_composition_have_gradients(first_probability):
    x = torch.randn(2, 4, 3, requires_grad=True)
    zeros = torch.zeros_like(x)
    torch.testing.assert_close(energy_distance(zeros, zeros), torch.zeros(2, dtype=torch.float64),
                               atol=1e-6, rtol=0)
    assert (energy_distance(zeros, zeros + 3) > 0).all()
    model = ComposableSegmentModel(SegmentConfig(2, 2, 1, max_horizon=4,
        consistency_first_call_probability=first_probability))
    q = model._data(_data(), labels=True)
    model._fit_statistics(q)
    loss = model._consistency(q)
    loss.backward()
    for name in ("count", "stop", "continuous", "encoder"):
        grads = [p.grad for p in model.network[name].parameters()]
        assert all(g is not None and torch.isfinite(g).all() for g in grads)
        assert sum(float(g.abs().sum()) for g in grads) > 0


def test_first_call_consistency_uses_forecast_split(monkeypatch):
    model = ComposableSegmentModel(SegmentConfig(2, 2, 1, max_horizon=4,
        consistency_first_call_probability=1.0))
    q = model._data(_data(), labels=True)
    model._fit_statistics(q)
    captured = []
    original = model._compose_draw

    def compose(query, left, right, samples, *, differentiable):
        captured.append((left.detach().clone(), right.detach().clone()))
        return original(query, left, right, samples, differentiable=differentiable)

    monkeypatch.setattr(model, "_compose_draw", compose)
    model._consistency(q)
    assert len(captured) == 1
    torch.testing.assert_close(captured[0][0], torch.ones_like(q["horizon"]))
    torch.testing.assert_close(captured[0][1], q["horizon"] - 1)


def test_asinh_energy_preserves_signed_growth_and_finite_gradients():
    model = ComposableSegmentModel(SegmentConfig(2, 2, 1, energy_coordinates="asinh"))
    values = torch.tensor([-1e12, -1., 0., 1., 1e12], requires_grad=True)
    transformed = model._energy_coordinates(values)
    assert torch.all(transformed[1:] > transformed[:-1])
    torch.testing.assert_close(transformed.sinh(), values, rtol=2e-6, atol=1e-6)
    transformed.sum().backward()
    assert torch.isfinite(values.grad).all() and (values.grad > 0).all()
    assert values.grad[2] == 1


def test_future_conditioning_uses_predicted_state_and_confirmed_floor(monkeypatch):
    model = ComposableSegmentModel(SegmentConfig(2, 2, 1, max_horizon=4))
    q = model._data(_data(), labels=True)
    model._fit_statistics(q)
    q["confirmed_call"][:] = 1e6
    original_draw = model._draw
    captured = []

    def draw(query, samples, *, differentiable):
        captured.append({key: value.detach().clone() for key, value in query.items()})
        return original_draw(query, samples, differentiable=differentiable)

    monkeypatch.setattr(model, "_draw", draw)
    combined, first = model._compose_draw(q, torch.ones(12), torch.full((12,), 2),
                                          3, differentiable=False)
    future = captured[1]
    torch.testing.assert_close(future["state"], first["end_state"].reshape(36, 2))
    torch.testing.assert_close(future["input_length"], first["outgoing_length"].reshape(-1))
    torch.testing.assert_close(future["spent"],
                               (q["spent"][:, None] + first["cost"]).reshape(-1))
    assert (future["observation"] == 0).all()
    assert (first["cost"] >= 1e6).all()
    torch.testing.assert_close(combined["cost"], combined["n"] * combined["initial_length"]
                               + combined["b"], atol=0.5, rtol=1e-5)


def test_zero_mass_known_input_mask_and_allowance_termination():
    data = _data()
    base = data["input_length"] + (data["input_length"] + data["g"]) * data["terminal"] * (data["n"] > 1)
    data["b"] = base - data["input_length"] * data["n"]
    model = ComposableSegmentModel(SegmentConfig(2, 2, 1, max_horizon=4))
    q = model._data(data, labels=True)
    model._fit_statistics(q)
    assert torch.isfinite(model._nll(q)).all()
    query = {key: value.copy() for key, value in data.items() if key in model.INPUT_KEYS}
    query["input_known"][:] = 0
    first = model.sample(query, samples=5, seed=31)
    query["input_length"][:] = 999999
    second = model.sample(query, samples=5, seed=31)
    np.testing.assert_array_equal(first["cost"], second["cost"])
    query["horizon"][:] = 1
    query["allowance"][:] = 1
    final = model.sample(query, samples=5)
    assert (final["terminal"] == 1).all()
    query["horizon"][:] = 0
    empty = model.sample(query, samples=5)
    assert (empty["terminal"] == 0).all()
    assert (empty["cost"] == 0).all() and (empty["g"] == 0).all()
    query["is_terminal"] = np.ones(12)
    query["confirmed_call"] = np.ones(12) * 100
    absorbed = model.forecast(query, samples=5)
    assert (absorbed["task_remaining_samples"] == 0).all()
    query["is_terminal"][:] = 0
    query["allowance"][:] = 0
    exhausted = model.sample(query, samples=5)
    assert (exhausted["terminal"] == 1).all()


def test_terminal_support_and_energy_unbiased_for_identical_distributions():
    data = _data()
    model = ComposableSegmentModel(SegmentConfig(2, 2, 1, max_horizon=4))
    broken = {key: value.copy() for key, value in data.items()}
    broken["g"][0] = 3
    with pytest.raises(ValueError, match="terminal single call"):
        model._data(broken, labels=True)
    q = model._data(data, labels=True)
    model._fit_statistics(q)
    sample = model._draw(q, 128, differentiable=False)
    required = sample["initial_length"] + sample["outgoing_length"] * sample["terminal"] * (sample["n"] > 1)
    assert (sample["cost"] >= required - 1e-4).all()
    with torch.random.fork_rng():
        torch.manual_seed(77)
        x, y = torch.randn(6000, 4, 1), torch.randn(6000, 4, 1)
    assert abs(float(energy_distance(x, y).mean())) < 0.04


def test_small_outgoing_boundary_survives_large_initial_context(monkeypatch):
    model = ComposableSegmentModel(SegmentConfig(2, 2, 1, max_horizon=4))
    q = model._data(_data(), labels=True)
    model._fit_statistics(q)
    q["input_length"][:] = 1e8
    q["input_known"][:] = 1
    q["horizon"][:] = q["allowance"][:] = 2

    def fixed_count(hidden, horizon):
        logits = torch.full((*hidden.shape[:-1], 4), -torch.inf)
        logits[..., 1] = 0
        return logits

    def fixed_lengths(hidden, count, terminal):
        mean = torch.zeros((*hidden.shape[:-1], 5))
        st = model.statistics
        mean[..., 1] = (torch.tensor(.25).log() - st["length_mean"][1]) / st["length_scale"][1]
        return mean, torch.full_like(mean, -100), torch.full((*hidden.shape[:-1], 3), -100.)

    monkeypatch.setattr(model, "_count_logits", fixed_count)
    monkeypatch.setattr(model, "_continuous", fixed_lengths)
    sampled = model._draw(q, 4, differentiable=False)
    torch.testing.assert_close(sampled["outgoing_length"], torch.full_like(sampled["outgoing_length"], .25))
    q["is_terminal"][:] = 1
    absorbed = model._draw(q, 4, differentiable=False)
    torch.testing.assert_close(absorbed["outgoing_length"], q["input_length"][:, None].expand(-1, 4))


def test_termination_gradient_controls_following_segment_cost():
    model = ComposableSegmentModel(SegmentConfig(2, 2, 1, max_horizon=4))
    q = model._data(_data(), labels=True)
    model._fit_statistics(q)


    with torch.no_grad():
        for parameter in model.network.parameters():
            parameter.zero_()
    result, _ = model._compose_draw(q, torch.ones(12), torch.ones(12),
                                    128, differentiable=True)
    result["cost"].mean().backward()
    gradient = model.network["stop"][-1].bias.grad
    assert gradient is not None and torch.isfinite(gradient).all()
    assert gradient.item() < 0


def test_unused_coordinate_overflow_and_physical_log_scale_bound(monkeypatch):
    model = ComposableSegmentModel(SegmentConfig(2, 2, 1, max_horizon=4))
    q = model._data(_data(), labels=True)
    model._fit_statistics(q)
    model.statistics["length_scale"][:] = 3
    hidden = model._encode(q)
    _, log_scale, _ = model._continuous(hidden, q["n"], q["terminal"])
    assert (log_scale[:, :3].exp() * model.statistics["length_scale"] < 2).all()
    original = model._continuous

    def exploding(hidden, count, terminal):
        mean, log_scale, zeros = original(hidden, count, terminal)
        mean = mean.clone()
        mean[..., :2] = 1000
        return mean, log_scale, zeros

    monkeypatch.setattr(model, "_continuous", exploding)
    q["input_known"][:] = 1
    q["horizon"][:] = 1
    q["allowance"][:] = 1
    result = model._draw(q, 3, differentiable=False)
    assert torch.isfinite(result["cost"]).all()
    q["is_terminal"][:] = 1
    assert (model._draw(q, 3, differentiable=False)["cost"] == 0).all()
    q["is_terminal"][:] = 0
    q["input_known"][:] = 0
    with pytest.raises(ValueError, match="overflow"):
        model._draw(q, 3, differentiable=False)
    large = torch.tensor([[[1e26], [0.0]]], requires_grad=True)
    loss = energy_distance(large, torch.zeros_like(large)).sum()
    assert torch.isfinite(loss)
    loss.backward()
    assert torch.isfinite(large.grad).all()


def test_bernoulli_state_support_and_exact_binary_likelihood():
    data = _data()
    data["end_state"][:] = np.array([0, 1])
    model = ComposableSegmentModel(SegmentConfig(2, 2, 1, max_horizon=4,
        max_epochs=1, ending_state_distribution="bernoulli"))
    model.fit(data, data)
    sample = model.sample(data, samples=64)
    assert np.isin(sample["end_state"], [0, 1]).all()
    q = model._data(data, labels=True)
    assert torch.isfinite(model._nll(q)).all()
    hidden = model._encode(q)
    mean, _, _ = model._continuous(hidden, q["n"], q["terminal"])
    changed = {**q, "end_state": 1 - q["end_state"]}
    expected = (torch.nn.functional.binary_cross_entropy_with_logits(
        mean[:, 3:], changed["end_state"], reduction="none") -
        torch.nn.functional.binary_cross_entropy_with_logits(
        mean[:, 3:], q["end_state"], reduction="none")).sum(-1)
    torch.testing.assert_close(model._nll(changed) - model._nll(q), expected,
                               atol=5e-6, rtol=1e-4)
    data["end_state"][0, 0] = .5
    with pytest.raises(ValueError, match="binary"):
        model._data(data, labels=True)


def test_unknown_call_limit_is_censored_without_forced_termination():
    model = ComposableSegmentModel(SegmentConfig(2, 2, 1, max_horizon=4))
    q = model._data(_data(), labels=True)
    model._fit_statistics(q)
    with torch.no_grad():
        for parameter in model.network["stop"].parameters():
            parameter.zero_()
        model.network["stop"][-1].bias.fill_(-100)
    query = {key: value.copy() for key, value in _data().items() if key in model.INPUT_KEYS}
    query["horizon"][:] = 1
    query["allowance"][:] = 1
    known = model.forecast(query, samples=16)
    assert (known["terminal"] == 1).all()
    assert not known["horizon_exhausted"].any()
    query["call_limit_known"] = np.zeros(12, dtype=bool)
    unknown = model.forecast(query, samples=16)
    assert (unknown["terminal"] == 0).all()
    assert unknown["horizon_exhausted"].all()
    query["horizon"][:] = 0
    query["allowance"][:] = 0
    empty = model.forecast(query, samples=16)
    assert (empty["terminal"] == 0).all() and empty["horizon_exhausted"].all()


def test_schema_one_checkpoint_preserves_original_condition_scaling(tmp_path):
    model = ComposableSegmentModel(SegmentConfig(2, 2, 1, max_horizon=4,
                                                 condition_scaling="linear_cap"))
    model._fit_statistics(model._data(_data(), labels=True))
    expected = model.forecast(_data(), samples=16, seed=99)
    path = tmp_path / "legacy.pt"
    model.save(path)
    saved = torch.load(path, weights_only=True)
    saved["schema_version"] = 1
    saved["config"].pop("condition_scaling")
    saved["config"].pop("ending_state_distribution")
    torch.save(saved, path)
    loaded = ComposableSegmentModel.load(path)
    assert loaded.config.condition_scaling == "linear_cap"
    assert loaded.config.ending_state_distribution == "gaussian"
    np.testing.assert_array_equal(loaded.forecast(_data(), samples=16, seed=99)["cost"],
                                  expected["cost"])
    modern = ComposableSegmentModel(SegmentConfig(2, 2, 1, max_horizon=500))
    torch.testing.assert_close(modern._count_feature(torch.tensor([1., 2.])),
                               torch.tensor([1., 2.]).log1p())


def test_context_surplus_scaling_preserves_density_units_and_physical_cost():
    model = ComposableSegmentModel(SegmentConfig(2, 2, 1, max_horizon=4,
        surplus_parameterization="context_scaled", min_log_length_std=.2))
    q = model._data(_data(), labels=True)
    model._fit_statistics(q)
    torch.testing.assert_close(model.statistics["single_call_surplus_scale"], torch.tensor(40.))
    count = torch.tensor([1., 2., 2., 3.])
    scale = model._surplus_scale(count, torch.full((4,), 100.), torch.full((4,), 200.),
                                 torch.tensor([0., 0., 1., 1.]))
    torch.testing.assert_close(scale, torch.tensor([40., 230., 80., 270.]))
    one = {key: value[q["n"] == 1] for key, value in q.items()}
    one["horizon"] = torch.ones_like(one["horizon"])
    old_nll = model._nll(one).detach()
    with torch.random.fork_rng():
        torch.manual_seed(1)
        old_samples = model._draw(one, 32, differentiable=False)

    model.statistics["single_call_surplus_scale"] *= 10
    model.statistics["length_mean"][2] -= np.log(10)
    torch.testing.assert_close(model._nll(one), old_nll, atol=1e-5, rtol=1e-5)
    with torch.random.fork_rng():
        torch.manual_seed(1)
        changed = model._draw(one, 32, differentiable=False)
    torch.testing.assert_close(changed["cost"], old_samples["cost"], atol=1e-4, rtol=1e-5)
    one["input_known"][:] = 0
    with torch.random.fork_rng():
        torch.manual_seed(3)
        unknown = model._draw(one, 16, differentiable=False)
        one["input_length"] += 10000
        torch.manual_seed(3)
        hidden_label_changed = model._draw(one, 16, differentiable=False)
    torch.testing.assert_close(unknown["cost"], hidden_label_changed["cost"], atol=0, rtol=0)
    mean, log_scale, _ = model._continuous(model._encode(q), q["n"], q["terminal"])
    actual_std = log_scale[:, :3].exp() * model.statistics["length_scale"]
    assert (actual_std >= .2).all() and (actual_std <= 2).all()
    assert torch.isfinite(mean).all()


def test_gradient_norm_overflow_fallback_keeps_nonfinite_guard():
    model = ComposableSegmentModel(SegmentConfig(2, 2, 1, max_horizon=4))
    parameter = next(model.network.parameters())
    parameter.grad = torch.full_like(parameter, 1e30)
    original_norm = model._clip_gradients()
    assert torch.isfinite(original_norm) and original_norm > 1e30
    assert model._wide_norm_fallbacks == 1
    assert float(torch.linalg.vector_norm(parameter.grad.double())) == pytest.approx(10, rel=1e-6)
    parameter.grad.fill_(float("nan"))
    with pytest.raises(ValueError, match="nonfinite"):
        model._clip_gradients()


def test_training_mean_budget_imputation_preserves_known_predictions_and_no_limit():
    model = ComposableSegmentModel(SegmentConfig(2, 2, 1, max_horizon=4))
    data = _data()
    model._fit_statistics(model._data(data, labels=True))
    weights = {key: value.clone() for key, value in model.network.state_dict().items()}
    original = model.forecast(data, samples=16, seed=42)
    model.adapt_unknown_budget(data)
    assert model.config.unknown_allowance_encoding == "train_mean"
    torch.testing.assert_close(model.statistics["allowance_feature_mean"], torch.tensor(5.).log())
    for key, value in weights.items():
        torch.testing.assert_close(model.network.state_dict()[key], value, atol=0, rtol=0)
    adapted = model.forecast(data, samples=16, seed=42)
    for key in original:
        np.testing.assert_array_equal(adapted[key], original[key])
    q = model._data(data, labels=False)
    known_hidden = model._encode(q)
    q["call_limit_known"][:] = 0
    torch.testing.assert_close(model._encode(q), known_hidden, atol=0, rtol=0)
    with torch.no_grad():
        for parameter in model.network["stop"].parameters():
            parameter.zero_()
        model.network["stop"][-1].bias.fill_(-100)
    query = {key: value.copy() for key, value in data.items()}
    query["allowance"][:] = 1
    query["horizon"][:] = 1
    query["call_limit_known"] = np.zeros(12)
    unknown = model.forecast(query, samples=16)
    assert (unknown["terminal"] == 0).all() and unknown["horizon_exhausted"].all()


def test_request_scaled_initial_density_sampling_and_training_only_statistics(tmp_path):
    data = _data()
    data.update(n=np.ones(12), horizon=np.ones(12), terminal=np.ones(12),
                g=np.zeros(12), b=np.full(12, 40.), input_known=np.zeros(12),
                observation=np.zeros((12, 8)))
    data["observation"][:, 1] = 1
    data["observation"][:, 4] = np.log1p(data["input_length"] * 4) / 16
    model = ComposableSegmentModel(SegmentConfig(2, 2, 8, max_horizon=4,
        request_scaled_initial_length=True))
    q = model._data(data, labels=True)
    model._fit_statistics(q)
    assert float(model.statistics["request_log_token_ratio"].exp()) == pytest.approx(.25, rel=1e-6)
    assert int(model.statistics["request_ratio_rows"]) == 12
    assert abs(float(model.statistics["length_mean"][0])) < 1e-6
    augmented = {key: torch.cat((value, value.clone())) for key, value in q.items()}
    augmented["input_known"][12:] = 1
    augmented["input_length"][12:] = 1e9
    model._fit_statistics(augmented)
    assert abs(float(model.statistics["length_mean"][0])) < 1e-6
    assert float(model.statistics["request_log_token_ratio"].exp()) == pytest.approx(.25, rel=1e-6)

    with torch.no_grad():
        for parameter in model.network.parameters():
            parameter.zero_()
    original = model.sample(data, samples=32, seed=9)
    scaled = {key: value.copy() for key, value in data.items()}
    scaled["input_length"] *= 3
    scaled["observation"][:, 4] = np.log1p(scaled["input_length"] * 4) / 16
    result = model.sample(scaled, samples=32, seed=9)
    np.testing.assert_allclose(result["initial_length"], original["initial_length"] * 3, rtol=2e-6)
    torch.testing.assert_close(model._nll(model._data(scaled, labels=True)) - model._nll(q),
                               torch.full((12,), np.log(3)), atol=3e-6, rtol=1e-5)

    scaled["input_length"][:] = 0
    hidden = model.sample(scaled, samples=32, seed=9)
    np.testing.assert_array_equal(hidden["initial_length"], result["initial_length"])
    conditions = {key: value.clone() for key, value in q.items()}
    conditions["input_known"][0] = 1
    conditions["observation"][1, 1] = 0
    conditions["observation"][1, 0] = 1
    conditions["observation"][2, 4] = 0
    assert (model._initial_anchor_log(conditions)[:3] == 0).all()
    path = tmp_path / "request.pt"
    model.save(path)
    restored = ComposableSegmentModel.load(path)
    np.testing.assert_array_equal(restored.sample(data, samples=32, seed=9)["initial_length"],
                                  original["initial_length"])


def test_termination_state_is_deterministic_density_and_composition_support():
    data = _data()
    data["state"][:, 1] = 0
    data["end_state"][:] = 0
    data["end_state"][:, 1] = data["terminal"]
    model = ComposableSegmentModel(SegmentConfig(2, 2, 1, max_horizon=4,
        termination_state_index=1, ending_state_distribution="bernoulli"))
    q = model._data(data, labels=True)
    model._fit_statistics(q)
    with torch.random.fork_rng():
        torch.manual_seed(17)
        sampled = model._draw(q, 32, differentiable=True)
    torch.testing.assert_close(sampled["end_state"][..., 1], sampled["terminal"], atol=0, rtol=0)

    sampled["end_state"][q["n"] == 3, :, 1].sum().backward()
    assert model.network["stop"][-1].bias.grad.abs().sum() > 0
    assert model._energy_vector(sampled).shape[-1] == 5
    forecast = model.forecast(data, samples=64, seed=1)
    np.testing.assert_array_equal(forecast["end_state"][..., 1], forecast["terminal"])
    loss = model._nll(q).detach()

    with torch.no_grad():
        model.network["continuous"][-1].bias[4] += 100
    torch.testing.assert_close(model._nll(q), loss, atol=0, rtol=0)
    bad = {key: value.copy() for key, value in data.items()}
    bad["end_state"][0, 1] = 1 - bad["terminal"][0]
    with pytest.raises(ValueError, match="ending termination state"):
        model._data(bad, labels=True)
    bad = {key: value.copy() for key, value in data.items()}
    bad["is_terminal"] = np.zeros(12)
    bad["state"][0, 1] = 1
    with pytest.raises(ValueError, match="observed termination state"):
        model._data(bad, labels=False)
    bad.pop("is_terminal")
    absorbed = model.forecast(bad, samples=8, seed=2)
    assert (absorbed["cost"][0] == 0).all()
    assert (absorbed["end_state"][0, :, 1] == 1).all()


def test_hierarchical_request_prior_is_causal_and_matches_joint_density(tmp_path):
    data = _data()
    data.update(n=np.ones(12), horizon=np.ones(12), terminal=np.ones(12),
                g=np.zeros(12), b=np.full(12, 40.), input_known=np.zeros(12),
                observation=np.zeros((12, 8)))
    data["observation"][:, 1] = 1
    data["request_surface_complete"] = np.ones(12, dtype=bool)
    data["observation"][:, 4] = np.log1p(data["input_length"] * 4) / 16
    model = ComposableSegmentModel(SegmentConfig(2, 2, 8, max_horizon=4,
                                                 request_input_prior="lognormal"))
    q = model._data(data, labels=True)
    model._fit_statistics(q)
    assert float(model.statistics["request_prior_mean"].exp()) == pytest.approx(.25, rel=1e-6)
    conditioned = {**q, "input_known": torch.ones(12)}
    sigma = model.statistics["request_prior_std"]
    expected_prior_nll = sigma.log() + .5 * np.log(2 * np.pi) + q["input_length"].log()
    expected_prior_nll -= torch.log1p(-model.statistics["request_prior_zero"])
    torch.testing.assert_close(model._nll(q), model._conditional_nll(conditioned) + expected_prior_nll)
    original = model.sample(data, samples=64, seed=19)
    altered = {key: value.copy() for key, value in data.items()}
    altered["input_length"][:] = 1e9
    hidden = model.sample(altered, samples=64, seed=19)
    for key in original:
        np.testing.assert_array_equal(original[key], hidden[key])
    assert (data["input_known"] == 0).all()
    altered["observation"][:, 4] = np.log1p(data["input_length"] * 12) / 16
    scaled = model.sample(altered, samples=64, seed=19)
    np.testing.assert_allclose(scaled["initial_length"], original["initial_length"] * 3, rtol=2e-6)

    altered["input_known"][:] = 1
    altered["input_length"][:] = 72
    known = model.sample(altered, samples=64, seed=19)
    assert (known["initial_length"] == 72).all()
    model.save(tmp_path / "prior.pt")
    loaded = ComposableSegmentModel.load(tmp_path / "prior.pt")
    np.testing.assert_array_equal(loaded.sample(data, samples=64, seed=19)["cost"], original["cost"])


def test_call_progress_updates_across_composition_and_joint_energy_identifies_input(monkeypatch):
    data = _data()
    data["completed_calls"] = np.arange(12)
    model = ComposableSegmentModel(SegmentConfig(2, 2, 1, max_horizon=4,
        use_call_count=True, energy_include_initial_length=True))
    q = model._data(data, labels=True)
    model._fit_statistics(q)
    observed = []
    original = model._draw
    def record(query, samples, *, differentiable):
        observed.append(query["completed_calls"].clone())
        return original(query, samples, differentiable=differentiable)
    monkeypatch.setattr(model, "_draw", record)
    _, first = model._compose_draw(q, torch.ones(12), torch.full((12,), 3), 8, differentiable=False)
    torch.testing.assert_close(observed[1].reshape(12, 8), q["completed_calls"][:, None] + first["n"])
    shifted = {**first, "initial_length": first["initial_length"] + 100}
    assert not torch.equal(model._energy_vector(first), model._energy_vector(shifted))
    missing = {k: v for k, v in data.items() if k != "completed_calls"}
    with pytest.raises(ValueError, match="require causal completed_calls"):
        model.forecast(missing)

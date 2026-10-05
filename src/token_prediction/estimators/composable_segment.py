from __future__ import annotations

import math
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Mapping

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class SegmentConfig:
    task_dim: int
    state_dim: int
    observation_dim: int
    max_horizon: int = 64
    hidden_dim: int = 64
    learning_rate: float = 0.002
    max_epochs: int = 100
    patience: int = 12
    batch_size: int = 128
    consistency_weight: float = 0.1
    consistency_samples: int = 4
    consistency_first_call_probability: float = 0.0
    max_log_length_std: float = 2.0
    min_log_length_std: float = 0.02
    condition_scaling: str = "log1p"
    count_distribution: str = "categorical"
    ending_state_distribution: str = "gaussian"
    surplus_parameterization: str = "absolute"
    unknown_allowance_encoding: str = "zero"
    request_scaled_initial_length: bool = False
    request_input_prior: str = "none"
    use_call_count: bool = False
    energy_include_initial_length: bool = False
    energy_coordinates: str = "physical"
    task_dropout: float = 0.0
    request_prior_groups: bool = False
    use_task_size: bool = False
    max_rollout_segments: int = 1
    termination_state_index: int | None = None
    seed: int = 20260919

    def __post_init__(self):
        if min(self.task_dim, self.state_dim, self.observation_dim) < 0:
            raise ValueError("feature dimensions must be nonnegative")
        if min(self.max_horizon, self.hidden_dim, self.max_epochs, self.patience,
               self.batch_size, self.consistency_samples, self.max_rollout_segments) < 1:
            raise ValueError("sizes and training limits must be positive")
        if self.consistency_samples < 2:
            raise ValueError("energy-distance estimation needs at least two samples")
        if self.learning_rate <= 0 or self.consistency_weight < 0:
            raise ValueError("invalid learning rate or consistency weight")
        if not 0 <= self.task_dropout < 1:
            raise ValueError("task dropout must lie in [0,1)")
        if not 0 <= self.consistency_first_call_probability <= 1:
            raise ValueError("first-call split probability must lie in [0,1]")
        if self.energy_coordinates not in {"physical", "asinh"}:
            raise ValueError("unsupported energy coordinates")
        if not 0 < self.min_log_length_std < self.max_log_length_std:
            raise ValueError("log-length deviation bounds must satisfy 0 < minimum < maximum")
        if self.condition_scaling not in {"linear_cap", "log1p"}:
            raise ValueError("unsupported count/horizon condition scaling")
        if self.count_distribution not in {"categorical", "discretized_lognormal"}:
            raise ValueError("unsupported count distribution")
        if self.ending_state_distribution not in {"gaussian", "bernoulli"}:
            raise ValueError("unsupported ending-state distribution")
        if self.surplus_parameterization not in {"absolute", "context_scaled"}:
            raise ValueError("unsupported surplus parameterization")
        if self.unknown_allowance_encoding not in {"zero", "train_mean"}:
            raise ValueError("unsupported unknown-allowance encoding")
        if self.request_scaled_initial_length and self.observation_dim < 5:
            raise ValueError("request scaling needs call-point flags and request bytes in observation[:5]")
        if self.request_input_prior not in {"none", "lognormal"}:
            raise ValueError("unsupported request input prior")
        if self.request_input_prior != "none" and (self.observation_dim < 5 or self.request_scaled_initial_length):
            raise ValueError("request input prior requires byte features and absolute initial coordinates")
        if self.termination_state_index is not None and not 0 <= self.termination_state_index < self.state_dim:
            raise ValueError("termination state index is outside the state vector")


def compose_segments(a: Mapping, b: Mapping) -> dict:

    # Every call in the right segment inherits the left segment's context growth.
    return {"n": a["n"] + b["n"], "g": a["g"] + b["g"],
            "b": a["b"] + b["b"] + b["n"] * a["g"]}


def energy_distance(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:

    m, n = x.shape[-2], y.shape[-2]
    if min(m, n) < 2:
        raise ValueError("energy distance requires two samples from each distribution")


    x, y = x.double(), y.double()
    xx, yy = torch.cdist(x, x), torch.cdist(y, y)
    within_x = (xx.sum((-1, -2)) - xx.diagonal(dim1=-2, dim2=-1).sum(-1)) / (m * (m - 1))
    within_y = (yy.sum((-1, -2)) - yy.diagonal(dim1=-2, dim2=-1).sum(-1)) / (n * (n - 1))
    return 2 * torch.cdist(x, y).mean((-1, -2)) - within_x - within_y


class ComposableSegmentModel:


    SCHEMA = 2
    INPUT_KEYS = ("task", "state", "observation", "horizon", "input_length",
                  "input_known", "spent", "allowance")

    def __init__(self, config: SegmentConfig):
        self.config = config
        torch.set_num_threads(1)
        with torch.random.fork_rng():
            torch.manual_seed(config.seed)
            width = config.task_dim + config.state_dim + config.observation_dim + 5 + int(config.use_call_count) + int(config.use_task_size)
            hidden = config.hidden_dim
            self.network = nn.ModuleDict({
                "encoder": nn.Sequential(nn.Linear(width, hidden), nn.SiLU(),
                                          nn.Linear(hidden, hidden), nn.SiLU()),
                "count": nn.Linear(hidden, config.max_horizon if config.count_distribution == "categorical" else 2),
                "stop": nn.Sequential(nn.Linear(hidden + 1, hidden), nn.SiLU(),
                                       nn.Linear(hidden, 1)) if config.count_distribution == "categorical" else nn.Identity(),
                "continuous": nn.Sequential(nn.Linear(hidden + 2, hidden), nn.SiLU(),
                                             nn.Linear(hidden, 2 * (3 + config.state_dim) + 3)),
            })
        self.statistics: dict[str, torch.Tensor] = {}
        self.fit_report: dict = {}

    def _data(self, data: Mapping, *, labels: bool) -> dict[str, torch.Tensor]:
        q = {key: torch.as_tensor(np.asarray(data[key]), dtype=torch.float32)
             for key in self.INPUT_KEYS}
        size = len(q["horizon"])
        if self.config.use_task_size and "task_text_bytes" not in data:
            raise ValueError("task-size-conditioned models require visible task_text_bytes")
        q["task_text_bytes"] = torch.as_tensor(data.get("task_text_bytes", np.zeros(size)), dtype=torch.float32)
        if (q["task_text_bytes"].shape != (size,) or not torch.isfinite(q["task_text_bytes"]).all()
                or (q["task_text_bytes"] < 0).any()):
            raise ValueError("task_text_bytes must contain nonnegative finite sizes")
        if self.config.use_call_count and "completed_calls" not in data:
            raise ValueError("call-count-conditioned models require causal completed_calls")
        q["completed_calls"] = torch.as_tensor(data.get("completed_calls", np.zeros(size)), dtype=torch.float32)
        q["request_group"] = torch.as_tensor(data.get("request_group", np.zeros(size)), dtype=torch.int64)
        q["request_surface_complete"] = torch.as_tensor(data.get("request_surface_complete", np.zeros(size)), dtype=torch.bool)
        if q["request_surface_complete"].shape != (size,):
            raise ValueError("request_surface_complete must be a vector")
        if q["request_group"].shape != (size,) or (q["request_group"] < 0).any():
            raise ValueError("request_group must contain nonnegative IDs")
        if (q["completed_calls"].shape != (size,) or not torch.isfinite(q["completed_calls"]).all()
                or (q["completed_calls"] < 0).any() or (q["completed_calls"] != q["completed_calls"].round()).any()):
            raise ValueError("completed_calls must contain nonnegative integers")
        for key, width in (("task", self.config.task_dim), ("state", self.config.state_dim),
                           ("observation", self.config.observation_dim)):
            if q[key].shape != (size, width):
                raise ValueError(f"{key} must have shape ({size}, {width})")
        for key in self.INPUT_KEYS[3:]:
            if q[key].shape != (size,):
                raise ValueError(f"{key} must be a vector")
        if not size or any(not torch.isfinite(v).all() for v in q.values()):
            raise ValueError("inputs must be nonempty and finite")
        for key in ("horizon", "allowance"):
            if ((q[key] < 0) | (q[key] > self.config.max_horizon)
                    | (q[key] != q[key].round())).any():
                raise ValueError(f"{key} must be integer in [0,max_horizon]")
        if (q["horizon"] > q["allowance"]).any():
            raise ValueError("segment horizon exceeds remaining call allowance")
        if (q["input_length"] < 0).any() or (q["spent"] < 0).any():
            raise ValueError("input length and confirmed spend must be nonnegative")
        if ((q["input_known"] != 0) & (q["input_known"] != 1)).any():
            raise ValueError("input_known must contain booleans")
        termination_index = self.config.termination_state_index
        default_terminal = (q["state"][:, termination_index] if termination_index is not None
                            else np.zeros(size))
        q["is_terminal"] = torch.as_tensor(data.get("is_terminal", default_terminal),
                                            dtype=torch.float32)
        q["confirmed_call"] = torch.as_tensor(data.get("confirmed_call", np.zeros(size)),
                                               dtype=torch.float32)
        q["call_limit_known"] = torch.as_tensor(data.get("call_limit_known", np.ones(size)),
                                                 dtype=torch.float32)
        for key in ("is_terminal", "confirmed_call", "call_limit_known"):
            if q[key].shape != (size,) or not torch.isfinite(q[key]).all():
                raise ValueError(f"invalid {key}")
        if ((q["is_terminal"] != 0) & (q["is_terminal"] != 1)).any():
            raise ValueError("is_terminal must contain booleans")
        if ((q["call_limit_known"] != 0) & (q["call_limit_known"] != 1)).any():
            raise ValueError("call_limit_known must contain booleans")
        if (q["confirmed_call"] < 0).any():
            raise ValueError("confirmed call consumption must be nonnegative")
        if termination_index is not None:
            if (q["state"][:, termination_index] != q["is_terminal"]).any():
                raise ValueError("observed termination state disagrees with is_terminal")
        q["is_terminal"] = torch.maximum(q["is_terminal"],
            (q["allowance"] == 0).float() * q["call_limit_known"])
        if termination_index is not None:
            q["state"] = q["state"].clone()
            q["state"][:, termination_index] = q["is_terminal"]
        if labels:
            for key in ("n", "g", "b", "end_state", "terminal"):
                q[key] = torch.as_tensor(np.asarray(data[key]), dtype=torch.float32)
                expected = (size, self.config.state_dim) if key == "end_state" else (size,)
                if q[key].shape != expected or not torch.isfinite(q[key]).all():
                    raise ValueError(f"invalid label {key}")
            q["weight"] = torch.as_tensor(data.get("weight", np.ones(size)),
                                           dtype=torch.float32)
            if (q["weight"].shape != (size,) or not torch.isfinite(q["weight"]).all()
                    or (q["weight"] < 0).any() or q["weight"].sum() <= 0):
                raise ValueError("invalid training weights")
            if ((q["n"] < 1) | (q["n"] > q["horizon"])
                    | (q["n"] != q["n"].round())).any():
                raise ValueError("active segment count must be integer in [1,horizon]")
            if ((q["terminal"] != 0) & (q["terminal"] != 1)).any():
                raise ValueError("termination labels must be boolean")
            if termination_index is not None and (q["end_state"][:, termination_index] != q["terminal"]).any():
                raise ValueError("ending termination state disagrees with terminal label")
            if ((q["n"] < q["horizon"]) & (q["terminal"] != 1)).any():
                raise ValueError("a segment ending before its horizon must terminate")
            if ((q["n"] == q["allowance"]) & (q["terminal"] != 1)
                    & (q["call_limit_known"] == 1)).any():
                raise ValueError("exhausting the call allowance terminates the task")
            if self.config.ending_state_distribution == "bernoulli":
                if ((q["end_state"] != 0) & (q["end_state"] != 1)).any():
                    raise ValueError("Bernoulli ending-state labels must be binary")
            if (q["is_terminal"] != 0).any():
                raise ValueError("absorbing states do not need learned training examples")
            cost = q["n"] * q["input_length"] + q["b"]
            outgoing = q["input_length"] + q["g"]
            minimum_cost = q["input_length"] + outgoing * q["terminal"] * (q["n"] > 1)
            if ((cost < minimum_cost - 0.1) | (outgoing < -0.1)).any():
                raise ValueError("segment has physically invalid consumption or outgoing length")
            if (((q["n"] == 1) & (q["terminal"] == 1)) & (q["g"].abs() > 1e-6)).any():
                raise ValueError("a terminal single call must have zero boundary growth")
        return q

    @staticmethod
    def _location(x: torch.Tensor, weight: torch.Tensor):
        weight = weight[:, None] / weight.sum()
        mean = (x * weight).sum(0)
        scale = ((x - mean).square() * weight).sum(0).sqrt().clamp_min(1.0)
        return mean, scale

    @staticmethod
    def _length_coordinates(q):
        length = q["input_length"]
        outgoing = (length + q["g"]).clamp_min(0)
        minimum_cost = length + outgoing * q["terminal"] * (q["n"] > 1)
        extra = (q["n"] * length + q["b"] - minimum_cost).clamp_min(0)
        return torch.stack((length, outgoing, extra), -1)

    @staticmethod
    def _positive_logs(values):
        return torch.where(values > 0, values.clamp_min(1e-30).log(), torch.zeros_like(values))

    def _surplus_scale(self, count, initial, outgoing, terminal, last_input_charged=None):

        if self.config.surplus_parameterization == "absolute":
            return torch.ones_like(count)
        if last_input_charged is None:
            last_input_charged = (count > 1).float()
        internal_count = (count - 1 - terminal * last_input_charged).clamp_min(0)
        return (count * self.statistics["single_call_surplus_scale"]
                + internal_count * (initial + outgoing) / 2).clamp_min(1e-12)

    def _length_target_logs(self, q, physical):
        logs = self._positive_logs(physical)
        if self.config.request_scaled_initial_length:
            logs = logs.clone()
            logs[:, 0] -= self._initial_anchor_log(q)
        if self.config.surplus_parameterization == "context_scaled":
            scale = self._surplus_scale(q["n"], physical[:, 0], physical[:, 1], q["terminal"])
            logs = logs.clone()
            logs[:, 2] -= scale.log()
        return logs

    def _request_log_bytes(self, q):


        obs = q["observation"]
        eligible = ((obs[:, 1] > .5) | (obs[:, 2] > .5)) & (obs[:, 4] > 0)
        eligible &= q["input_known"] < .5
        if self.config.request_input_prior == "lognormal":
            eligible &= q["request_surface_complete"]
        x = (obs[:, 4] * 16).clamp_min(1e-12)
        log_bytes = x + (-torch.expm1(-x)).log()
        return eligible, log_bytes

    def _initial_anchor_log(self, q):
        if not self.config.request_scaled_initial_length:
            return torch.zeros_like(q["horizon"])
        eligible, log_bytes = self._request_log_bytes(q)
        return torch.where(eligible, log_bytes + self.statistics["request_log_token_ratio"],
                           torch.zeros_like(log_bytes))

    def _fit_request_ratio(self, q):
        eligible, log_bytes = self._request_log_bytes(q)


        selected = eligible & (q["horizon"] == 1) & (q["input_length"] > 0) & (q["weight"] > 0)
        if not selected.any():
            raise ValueError("request scaling needs positive train-only h=1 call inputs with visible bytes")
        weights = q["weight"][selected]
        log_ratio = q["input_length"][selected].log() - log_bytes[selected]
        self.statistics["request_log_token_ratio"] = (log_ratio * weights).sum() / weights.sum()
        self.statistics["request_ratio_rows"] = selected.sum()

    def _fit_statistics(self, q):
        for key in ("task", "state", "observation"):
            mean, scale = self._location(q[key], q["weight"])
            self.statistics[key + "_mean"] = mean
            self.statistics[key + "_scale"] = scale
        self._fit_allowance_mean(q, q["weight"])
        if self.config.request_scaled_initial_length:
            self._fit_request_ratio(q)
        if self.config.request_input_prior == "lognormal":
            for key in ("group_ids", "group_means", "group_stds", "group_zeros"):
                self.statistics.pop("request_prior_" + key, None)
            eligible, log_bytes = self._request_log_bytes(q)
            selected = eligible & (q["observation"][:, 1] > .5) & (q["horizon"] == 1) & (q["weight"] > 0)
            positive = selected & (q["input_length"] > 0)
            if not positive.any():
                raise ValueError("request input prior requires positive training Call Start rows")
            w = q["weight"][positive]
            r = q["input_length"][positive].log() - log_bytes[positive]
            mean = (r * w).sum() / w.sum()
            variance = ((r - mean).square() * w).sum() / w.sum()
            self.statistics["request_prior_mean"] = mean
            self.statistics["request_prior_std"] = variance.sqrt().clamp_min(1e-3)
            self.statistics["request_prior_zero"] = (
                (q["weight"][selected] * (q["input_length"][selected] == 0)).sum()
                / q["weight"][selected].sum()).clamp(1e-6, 1 - 1e-6)
            self.statistics["request_prior_rows"] = selected.sum()
            if self.config.request_prior_groups:
                groups, means, stds, zeros = [], [], [], []
                for group in torch.unique(q["request_group"][selected]):
                    chosen = selected & (q["request_group"] == group)
                    pos = chosen & (q["input_length"] > 0)
                    if not group or pos.sum() < 2:
                        continue
                    weight = q["weight"][pos]
                    ratio = q["input_length"][pos].log() - log_bytes[pos]
                    mu = (ratio * weight).sum() / weight.sum()
                    groups.append(group)
                    means.append(mu)
                    stds.append((((ratio - mu).square() * weight).sum() / weight.sum()).sqrt().clamp_min(1e-3))
                    zeros.append(((q["weight"][chosen] * (q["input_length"][chosen] == 0)).sum()
                                  / q["weight"][chosen].sum()).clamp(1e-6, 1-1e-6))
                if groups:
                    for name, values in (("group_ids", groups), ("group_means", means), ("group_stds", stds), ("group_zeros", zeros)):
                        self.statistics["request_prior_" + name] = torch.stack(values)
        values = self._length_coordinates(q)
        if self.config.surplus_parameterization == "context_scaled":
            single = (q["n"] == 1) & (values[:, 2] > 0) & (q["weight"] > 0)
            if single.any():
                weights = q["weight"][single]
                prior = ((values[single, 2].log() * weights).sum() / weights.sum()).exp()
            else:

                prior = torch.tensor(1.0)
            self.statistics["single_call_surplus_scale"] = prior
        logs = self._length_target_logs(q, values)
        means, scales = [], []
        for index in range(3):
            observed = values[:, index] > 0
            if index == 0 and self.config.request_scaled_initial_length:
                observed &= q["input_known"] < .5
            if not observed.any() or q["weight"][observed].sum() <= 0:
                means.append(torch.tensor(0.0))
                scales.append(torch.tensor(1.0))
            else:
                mean, scale = self._location(logs[observed, index:index + 1], q["weight"][observed])
                means.append(mean[0])
                scales.append(scale[0])
        self.statistics["length_mean"] = torch.stack(means)
        self.statistics["length_scale"] = torch.stack(scales)
        for key, x in (("input", torch.log1p(q["input_length"][:, None])),
                       ("spent", torch.log1p(q["spent"][:, None]))):
            mean, scale = self._location(x, q["weight"])
            self.statistics[key + "_mean"] = mean
            self.statistics[key + "_scale"] = scale
        physical = torch.stack((q["n"], q["g"],
                                q["n"] * q["input_length"] + q["b"]), -1)
        physical = self._energy_coordinates(physical)
        _, scale = self._location(physical, q["weight"])
        self.statistics["energy_scale"] = scale
        if self.config.energy_include_initial_length:
            _, initial_scale = self._location(self._energy_coordinates(q["input_length"][:, None]), q["weight"])
            self.statistics["energy_initial_scale"] = initial_scale[0]

    def _fit_allowance_mean(self, q, weights):
        known = (q["call_limit_known"] > .5) & (weights > 0)
        if known.any():
            value = (self._count_feature(q["allowance"][known]) * weights[known]).sum() / weights[known].sum()
        else:
            value = torch.tensor(0.0)
        self.statistics["allowance_feature_mean"] = value

    def adapt_unknown_budget(self, training_inputs: Mapping):

        if not self.statistics:
            raise ValueError("fit or load a model before adapting missing-budget inputs")
        q = self._data(training_inputs, labels=False)
        weights = torch.as_tensor(training_inputs.get("weight", np.ones(len(q["horizon"]))),
                                    dtype=torch.float32)
        if (weights.shape != q["horizon"].shape or not torch.isfinite(weights).all()
                or (weights < 0).any() or weights.sum() <= 0):
            raise ValueError("invalid training-only adaptation weights")
        self._fit_allowance_mean(q, weights)
        self.config = replace(self.config, unknown_allowance_encoding="train_mean")
        self.fit_report = {**self.fit_report, "unknown_allowance_encoding": "train_mean",
                           "unknown_budget_adaptation": {
            "method": "training_mean_feature_imputation", "weights_unchanged": True,
            "statistics_source": "training_partition_only", "rows": len(q["horizon"]),
            "observed_budget_rows": int((q["call_limit_known"] > .5).sum()),
            "allowance_feature_mean": float(self.statistics["allowance_feature_mean"])}}
        return self

    def _encode(self, q):
        st = self.statistics
        blocks = [(q[key] - st[key + "_mean"]) / st[key + "_scale"]
                  for key in ("task", "state", "observation")]

        known_length = ((torch.log1p(q["input_length"]) - st["input_mean"][0])
                        / st["input_scale"][0]) * q["input_known"]
        spent = ((torch.log1p(q["spent"]) - st["spent_mean"][0])
                 / st["spent_scale"][0])
        allowance_feature = self._count_feature(q["allowance"])
        if self.config.unknown_allowance_encoding == "train_mean":
            if "allowance_feature_mean" not in st:
                raise ValueError("missing training-only allowance mean; fit or adapt the model")
            allowance_feature = torch.where(q["call_limit_known"] > .5, allowance_feature,
                                             st["allowance_feature_mean"])
        else:
            allowance_feature = allowance_feature * q["call_limit_known"]
        blocks.append(torch.stack((known_length, q["input_known"], spent,
                                   self._count_feature(q["horizon"]),
                                   allowance_feature), -1))
        if self.config.use_call_count:
            blocks.append(q["completed_calls"].log1p()[:, None])
        if self.config.use_task_size:
            blocks.append(q["task_text_bytes"].log1p()[:, None] / 16)
        return self.network["encoder"](torch.cat(blocks, -1).clamp(-30, 30))

    def _count_feature(self, count):
        if self.config.condition_scaling == "log1p":
            return count.log1p()
        return count / self.config.max_horizon

    def _count_logits(self, hidden, horizon):
        if self.config.count_distribution != "categorical":
            raise ValueError("parametric counts require the joint count/termination distribution")
        logits = self.network["count"](hidden)
        counts = torch.arange(1, self.config.max_horizon + 1, device=logits.device)
        return logits.masked_fill(counts > horizon[..., None].clamp_min(1), -1e9)

    def _count_parameters(self, q):

        fixed = {**q, "horizon": torch.full_like(q["horizon"], self.config.max_horizon)}
        values = self.network["count"](self._encode(fixed))
        return values[..., 0].double(), F.softplus(values[..., 1].double()) + .05

    @staticmethod
    def _count_log_survival(boundary, location, scale):

        boundary = boundary.double()
        z = (location - boundary.clamp_min(1).log()) / scale
        return torch.where(boundary > 0, torch.special.log_ndtr(z), torch.zeros_like(z))

    @staticmethod
    def _count_log_pmf(count, location, scale):

        count = count.double()
        upper = (count.log() - location) / scale
        lower = ((count - 1).clamp_min(1).log() - location) / scale
        right_tail = lower > 0
        large = torch.special.log_ndtr(torch.where(right_tail, -lower, upper))
        small = torch.special.log_ndtr(torch.where(right_tail, -upper, lower))


        difference = torch.where(count > 1, small - large, torch.full_like(large, -1))
        interval = large + (-torch.expm1(difference)).log()
        return torch.where(count > 1, interval, torch.special.log_ndtr(upper))

    def _count_stop_nll(self, q, hidden):
        if self.config.count_distribution == "categorical":
            count = F.cross_entropy(self._count_logits(hidden, q["horizon"]),
                                    q["n"].long() - 1, reduction="none")
            stop = F.binary_cross_entropy_with_logits(
                self._stop_logits(hidden, q["n"]), q["terminal"], reduction="none")
            stop = stop * ((q["n"] == q["horizon"])
                & ((q["n"] < q["allowance"]) | (q["call_limit_known"] == 0)))
            return count, stop
        location, scale = self._count_parameters(q)
        event = self._count_log_pmf(q["n"], location, scale)
        event = torch.where(q["terminal"] < .5,
            self._count_log_survival(q["horizon"], location, scale), event)

        cap = ((q["n"] == q["allowance"]) & (q["call_limit_known"] > .5))
        event = torch.where(cap,
            self._count_log_survival(q["allowance"] - 1, location, scale), event)
        return -event, torch.zeros_like(event)

    def _parametric_count_draw(self, q, samples, differentiable):
        location, scale = self._count_parameters(q)
        horizon = q["horizon"].clamp_min(1)
        values = torch.arange(1, self.config.max_horizon + 1, dtype=torch.float64)
        point = self._count_log_pmf(values[None, :], location[:, None], scale[:, None])
        point = point.masked_fill(values[None, :] > horizon[:, None], -torch.inf)
        tail = self._count_log_survival(horizon, location, scale)
        logits = torch.cat((point, tail[:, None]), -1).float()[:, None, :].expand(-1, samples, -1)
        h = horizon[:, None].expand(-1, samples)
        if differentiable:
            joint = F.gumbel_softmax(logits, tau=.75, hard=True, dim=-1)
            count = (joint[..., :-1] * values.float()).sum(-1) + joint[..., -1] * h
            terminal = 1 - joint[..., -1]
            single = joint[..., 0] + joint[..., -1] * (h == 1)
        else:
            category = torch.distributions.Categorical(logits=logits).sample()
            tail_sample = category == self.config.max_horizon
            count = torch.where(tail_sample, h, category.float() + 1)
            terminal = (~tail_sample).float()
            single = (count == 1).float()
        cap = (q["call_limit_known"] > .5) & (q["horizon"] >= q["allowance"])
        terminal = torch.where(cap[:, None], torch.ones_like(terminal), terminal)
        return count, terminal, single

    def _continuous(self, hidden, count, terminal):
        values = self.network["continuous"](torch.cat((hidden,
            self._count_feature(count)[..., None], terminal[..., None]), -1))
        mean, log_scale = values[..., :-3].chunk(2, -1)


        length_std = self.config.min_log_length_std + (
            self.config.max_log_length_std - self.config.min_log_length_std) * log_scale[..., :3].sigmoid()
        length_log_scale = length_std.log() - self.statistics["length_scale"].log()
        log_scale = torch.cat((length_log_scale, log_scale[..., 3:].clamp(-4, 2)), -1)
        return mean, log_scale, values[..., -3:]

    def _stop_logits(self, hidden, count):
        return self.network["stop"](torch.cat((hidden,
            self._count_feature(count)[..., None]), -1)).squeeze(-1)

    def _request_prior_parameters(self, q):
        st = self.statistics
        mean = torch.full_like(q["horizon"], st["request_prior_mean"])
        std = torch.full_like(mean, st["request_prior_std"])
        zero = torch.full_like(mean, st["request_prior_zero"])
        if self.config.request_prior_groups and "request_prior_group_ids" in st:
            for index, group in enumerate(st["request_prior_group_ids"]):
                selected = q["request_group"] == group
                mean = torch.where(selected, st["request_prior_group_means"][index], mean)
                std = torch.where(selected, st["request_prior_group_stds"][index], std)
                zero = torch.where(selected, st["request_prior_group_zeros"][index], zero)
        return mean, std, zero

    def _nll(self, q):
        if self.config.request_input_prior != "lognormal":
            return self._conditional_nll(q)
        eligible, log_bytes = self._request_log_bytes(q)


        conditioned = {**q, "input_known": torch.where(eligible, torch.ones_like(q["input_known"]), q["input_known"])}
        prior_mean, prior_std, prior_zero = self._request_prior_parameters(q)
        log_length = q["input_length"].clamp_min(1e-30).log()
        standardized = (log_length - log_bytes - prior_mean) / prior_std
        positive_nll = .5 * standardized.square() + prior_std.log() + .5 * math.log(2 * math.pi) + log_length
        prior_nll = torch.where(q["input_length"] > 0,
                               positive_nll - torch.log1p(-prior_zero), -prior_zero.log())
        return self._conditional_nll(conditioned) + torch.where(eligible, prior_nll, torch.zeros_like(prior_nll))

    def _conditional_nll(self, q):
        hidden = self._encode(q)
        count_loss, stop_loss = self._count_stop_nll(q, hidden)
        mean, log_scale, zero_logits = self._continuous(hidden, q["n"], q["terminal"])
        physical = self._length_coordinates(q)
        lengths = self._length_target_logs(q, physical)
        st = self.statistics
        target = torch.cat(((lengths - st["length_mean"]) / st["length_scale"],
                            (q["end_state"] - st["state_mean"]) / st["state_scale"]), -1)
        normal = 0.5 * ((target - mean) / log_scale.exp()).square() + log_scale
        normal = normal + 0.5 * math.log(2 * math.pi)


        jacobian = torch.cat((self._positive_logs(physical) + st["length_scale"].log(),
                               st["state_scale"].log().expand(len(lengths), -1)), -1)
        mask = torch.ones_like(target)
        mask[:, :3] = (physical > 0).float()
        observed = torch.ones_like(physical)
        observed[:, 0] = 1 - q["input_known"]

        observed[:, 1] = 1 - ((q["n"] == 1) & (q["terminal"] == 1)).float()
        mask[:, :3] *= observed
        state_loss = torch.zeros_like(count_loss)
        state_observed = torch.ones(self.config.state_dim)
        if self.config.termination_state_index is not None:
            state_observed[self.config.termination_state_index] = 0
            mask[:, 3 + self.config.termination_state_index] = 0
        if self.config.ending_state_distribution == "bernoulli":
            mask[:, 3:] = 0
            state_loss = F.binary_cross_entropy_with_logits(
                mean[:, 3:], q["end_state"], reduction="none")
            state_loss = (state_loss * state_observed).sum(-1)
        zero_loss = F.binary_cross_entropy_with_logits(zero_logits, (physical == 0).float(),
                                                       reduction="none")
        return (count_loss + stop_loss + ((normal + jacobian) * mask).sum(-1)
                + (zero_loss * observed).sum(-1) + state_loss)

    def _draw(self, q, samples: int, *, differentiable: bool):
        if self.config.request_input_prior != "lognormal":
            return self._conditional_draw(q, samples, differentiable=differentiable)
        eligible, log_bytes = self._request_log_bytes(q)
        if not eligible.any():
            return self._conditional_draw(q, samples, differentiable=differentiable)
        batch = len(q["horizon"])
        prior_mean, prior_std, prior_zero = self._request_prior_parameters(q)
        shape = (batch, samples)
        prior = (log_bytes[:, None] + prior_mean[:, None] + prior_std[:, None] * torch.randn(shape)).exp()
        prior = prior * (torch.rand(shape) >= prior_zero[:, None])
        def repeat(value):
            return value[:, None].expand(-1, samples, *value.shape[1:]).reshape(batch * samples, *value.shape[1:])

        conditioned = {key: repeat(q[key]) for key in (*self.INPUT_KEYS, "is_terminal", "confirmed_call", "call_limit_known", "completed_calls", "request_group", "request_surface_complete", "task_text_bytes")}
        mask = repeat(eligible)
        conditioned["input_length"] = torch.where(mask, prior.reshape(-1), conditioned["input_length"])
        conditioned["input_known"] = torch.where(mask, torch.ones_like(conditioned["input_known"]), conditioned["input_known"])
        result = self._conditional_draw(conditioned, 1, differentiable=differentiable)
        return {key: value.reshape(batch, samples, *value.shape[2:]) for key, value in result.items()}

    def _conditional_draw(self, q, samples: int, *, differentiable: bool):
        hidden = self._encode(q)[:, None, :].expand(-1, samples, -1)
        horizon = q["horizon"][:, None].expand(-1, samples)
        if self.config.count_distribution == "discretized_lognormal":
            count, terminal, single_count = self._parametric_count_draw(q, samples, differentiable)
        else:
            logits = self._count_logits(hidden, horizon)
            counts = torch.arange(1, self.config.max_horizon + 1, dtype=torch.float32)
            if differentiable:
                one_hot = F.gumbel_softmax(logits, tau=0.75, hard=True, dim=-1)
                count = (one_hot * counts).sum(-1)
                single_count = one_hot[..., 0]
            else:
                count = torch.distributions.Categorical(logits=logits).sample().float() + 1
                single_count = (count == 1).float()
            probability = self._stop_logits(hidden, count).sigmoid()
            terminal = (torch.rand_like(probability) < probability).float()
            if differentiable:
                # Pass gradients through the probability of the sampled stop decision.
                terminal = terminal + probability - probability.detach()
            forced_stop = (count < horizon) | ((count >= q["allowance"][:, None])
                                               & (q["call_limit_known"][:, None] > 0.5))
            terminal = torch.where(forced_stop, torch.ones_like(terminal), terminal)
        mean, log_scale, zero_logits = self._continuous(hidden, count, terminal)
        latent = mean + log_scale.exp() * torch.randn_like(mean)
        st = self.statistics

        log_lengths = latent[..., :3] * st["length_scale"] + st["length_mean"]
        if self.config.request_scaled_initial_length:
            log_lengths = log_lengths.clone()
            log_lengths[..., 0] += self._initial_anchor_log(q)[:, None]
        zero_probability = zero_logits.sigmoid()
        zeros = (torch.rand_like(zero_probability) < zero_probability).float()
        if differentiable:
            zeros = zeros + zero_probability - zero_probability.detach()
        known = q["input_known"][:, None].bool().expand_as(count)
        active = (1 - q["is_terminal"][:, None]) * (horizon > 0).float()
        required = torch.stack((~known, ~((count == 1) & (terminal > 0.5)),
                                torch.ones_like(known)), -1) & (active > 0.5)[..., None]


        safe = required | (log_lengths < math.log(torch.finfo(log_lengths.dtype).max) - 1)
        log_lengths = torch.where(safe, log_lengths, torch.zeros_like(log_lengths))
        lengths = log_lengths.exp() * (1 - zeros)
        if not torch.isfinite(lengths).all():
            raise ValueError("segment distribution overflow; inspect fitted scales and conditioning inputs")
        initial = torch.where(known, q["input_length"][:, None], lengths[..., 0])
        outgoing = lengths[..., 1]
        if differentiable:
            single_terminal = single_count * terminal
            outgoing = outgoing * (1 - single_terminal) + initial * single_terminal
        else:
            outgoing = torch.where((count == 1) & (terminal > 0.5), initial, outgoing)
        last_input_charged = 1 - single_count
        surplus_scale = self._surplus_scale(count, initial, outgoing, terminal, last_input_charged)
        cost = initial + outgoing * terminal * last_input_charged + lengths[..., 2] * surplus_scale
        if not torch.isfinite(cost).all():
            raise ValueError("segment cost overflow; inspect fitted scales and conditioning inputs")
        end = latent[..., 3:] * st["state_scale"] + st["state_mean"]
        if self.config.ending_state_distribution == "bernoulli":
            state_probability = mean[..., 3:].sigmoid()
            end = (torch.rand_like(state_probability) < state_probability).float()
            if differentiable:
                end = end + state_probability - state_probability.detach()


        count = count * active
        cost = cost * active
        growth = (outgoing - initial) * active

        outgoing = outgoing * active + initial * (1 - active)
        start_state = q["state"][:, None, :]
        end = start_state + (end - start_state) * active[..., None]
        terminal = q["is_terminal"][:, None] + terminal * active
        if self.config.termination_state_index is not None:
            end = end.clone()
            end[..., self.config.termination_state_index] = terminal
        return {"n": count, "g": growth, "b": cost - count * initial,
                "cost": cost, "initial_length": initial, "outgoing_length": outgoing,
                "end_state": end, "terminal": terminal}

    def _compose_draw(self, q, left_horizon, right_horizon, samples, *, differentiable):
        first_query = {**q, "horizon": left_horizon}
        a = self._draw(first_query, samples, differentiable=differentiable)
        active = ((q["is_terminal"] < 0.5) & (left_horizon > 0))[:, None]
        adjusted = torch.where(active, torch.maximum(a["cost"], q["confirmed_call"][:, None]),
                               a["cost"])
        a["b"] = a["b"] + adjusted - a["cost"]
        a["cost"] = adjusted
        batch = len(q["horizon"])

        def repeat(value):
            return value[:, None].expand(-1, samples, *value.shape[1:]).reshape(
                batch * samples, *value.shape[1:])

        # Future observations are unavailable when continuing from this sampled boundary.
        b_query = {"task": repeat(q["task"]), "state": a["end_state"].reshape(
            batch * samples, self.config.state_dim),
            "observation": torch.zeros(batch * samples, self.config.observation_dim),
            "horizon": repeat(right_horizon),
            "input_length": a["outgoing_length"].reshape(-1),
            "input_known": torch.ones(batch * samples),
            "spent": (q["spent"][:, None] + a["cost"]).reshape(-1),
            "allowance": (q["allowance"][:, None] - a["n"]).clamp_min(0).reshape(-1),
            "call_limit_known": repeat(q["call_limit_known"]),
            "completed_calls": (q["completed_calls"][:, None] + a["n"]).reshape(-1),
            "request_group": repeat(q["request_group"]),
            "request_surface_complete": torch.zeros(batch * samples, dtype=torch.bool),
            "task_text_bytes": repeat(q["task_text_bytes"]),
            "is_terminal": a["terminal"].reshape(-1),
            "confirmed_call": torch.zeros(batch * samples)}
        b_flat = self._draw(b_query, 1, differentiable=differentiable)
        b = {key: value.reshape(batch, samples, *value.shape[2:])
             for key, value in b_flat.items()}
        combined = compose_segments(a, b)

        combined.update(cost=a["cost"] + b["cost"], initial_length=a["initial_length"],
                        outgoing_length=b["outgoing_length"], end_state=b["end_state"],
                        terminal=b["terminal"])
        return combined, a

    def _energy_coordinates(self, values):
        if self.config.energy_coordinates == "asinh":


            return values.asinh()
        return values

    def _energy_vector(self, samples):
        physical = self._energy_coordinates(torch.stack((samples["n"], samples["g"], samples["cost"]), -1))
        state = ((samples["end_state"] - self.statistics["state_mean"])
                 / self.statistics["state_scale"])
        if self.config.termination_state_index is not None:
            index = self.config.termination_state_index
            state = torch.cat((state[..., :index], state[..., index + 1:]), -1)
        blocks = [physical / self.statistics["energy_scale"], state, samples["terminal"][..., None]]
        if self.config.energy_include_initial_length:
            blocks.append(self._energy_coordinates(samples["initial_length"][..., None]) / self.statistics["energy_initial_scale"])
        return torch.cat(blocks, -1)

    def _consistency(self, q):
        selected = q["horizon"] >= 2
        if not selected.any():
            return next(self.network.parameters()).sum() * 0
        q = {key: value[selected] for key, value in q.items()}
        left = (torch.rand_like(q["horizon"]) * (q["horizon"] - 1)).floor() + 1
        if self.config.consistency_first_call_probability:


            first = torch.rand_like(left) < self.config.consistency_first_call_probability
            left = torch.where(first, torch.ones_like(left), left)
        direct = self._draw(q, self.config.consistency_samples, differentiable=True)
        composed, _ = self._compose_draw(q, left, q["horizon"] - left,
                                        self.config.consistency_samples, differentiable=True)
        distance = energy_distance(self._energy_vector(direct), self._energy_vector(composed))
        return (distance * q["weight"]).sum() / q["weight"].sum().clamp_min(1e-12)

    def _clip_gradients(self):
        parameters = list(self.network.parameters())
        try:
            return torch.nn.utils.clip_grad_norm_(parameters, 10, error_if_nonfinite=True)
        except RuntimeError as exc:


            gradients = [p.grad for p in parameters if p.grad is not None]
            norm = torch.linalg.vector_norm(torch.stack([
                torch.linalg.vector_norm(g.double()) for g in gradients]))
            if not torch.isfinite(norm):
                raise ValueError("nonfinite compositional model gradients") from exc
            factor = min(1.0, float(10 / (norm + 1e-6)))
            with torch.no_grad():
                for gradient in gradients:
                    gradient.mul_(factor)
            self._wide_norm_fallbacks = getattr(self, "_wide_norm_fallbacks", 0) + 1
            return norm

    def fit(self, train: Mapping, validation: Mapping):
        tr, va = self._data(train, labels=True), self._data(validation, labels=True)
        self._fit_statistics(tr)
        optimizer = torch.optim.AdamW(self.network.parameters(), lr=self.config.learning_rate,
                                      weight_decay=1e-4)
        best, best_epoch, stale, best_state = math.inf, 0, 0, None
        self._wide_norm_fallbacks = 0
        history = []
        with torch.random.fork_rng():
            torch.manual_seed(self.config.seed)
            for epoch in range(1, self.config.max_epochs + 1):
                self.network.train()
                order = torch.randperm(len(tr["horizon"]))
                loss_sum, energy_sum, steps = 0.0, 0.0, 0
                for indices in order.split(self.config.batch_size):
                    batch = {key: value[indices] for key, value in tr.items()}
                    if batch["weight"].sum() <= 0:
                        continue
                    if self.config.task_dropout:


                        drop = torch.rand(len(indices)) < self.config.task_dropout
                        batch["task"] = torch.where(drop[:, None], self.statistics["task_mean"], batch["task"])
                    nll = (self._nll(batch) * batch["weight"]).sum() / batch["weight"].sum()
                    energy = self._consistency(batch) if self.config.consistency_weight else nll * 0
                    loss = nll + self.config.consistency_weight * energy
                    if not torch.isfinite(loss):
                        raise ValueError("nonfinite compositional training objective")
                    optimizer.zero_grad()
                    loss.backward()
                    self._clip_gradients()
                    optimizer.step()
                    loss_sum += float(nll.detach())
                    energy_sum += float(energy.detach())
                    steps += 1
                self.network.eval()
                with torch.no_grad():
                    score = float((self._nll(va) * va["weight"]).sum() / va["weight"].sum())
                history.append({"epoch": epoch, "train_nll": loss_sum / max(1, steps),
                                "train_energy": energy_sum / max(1, steps),
                                "validation_nll": score})
                if score < best - 1e-7:
                    best, best_epoch, stale = score, epoch, 0
                    best_state = {key: value.detach().clone()
                                  for key, value in self.network.state_dict().items()}
                else:
                    stale += 1
                    if stale >= self.config.patience:
                        break
        if best_state is None:
            raise ValueError("training produced no finite validation checkpoint")
        self.network.load_state_dict(best_state)
        self.network.eval()
        self.fit_report = {"status": "unvalidated_method_pilot", "best_epoch": best_epoch,
                           "validation_nll": best, "epochs_run": len(history),
                           "train_rows": len(tr["n"]), "validation_rows": len(va["n"]),
                           "normalization_source": "training_partition_only",
                           "float64_gradient_norm_fallbacks": self._wide_norm_fallbacks,
                           "distribution": "count_stop_conditional_zero_inflated_lognormal_and_state_"
                               + self.config.ending_state_distribution,
                           "condition_scaling": self.config.condition_scaling,
                           "count_distribution": self.config.count_distribution,
                           "surplus_parameterization": self.config.surplus_parameterization,
                           "unknown_allowance_encoding": self.config.unknown_allowance_encoding,
                           "request_scaled_initial_length": self.config.request_scaled_initial_length,
                           "request_input_prior": self.config.request_input_prior,
                           "use_call_count": self.config.use_call_count,
                           "energy_include_initial_length": self.config.energy_include_initial_length,
                           "energy_coordinates": self.config.energy_coordinates,
                           "task_dropout": self.config.task_dropout,
                           "request_prior_groups": self.config.request_prior_groups,
                           "use_task_size": self.config.use_task_size,
                           "termination_state_index": self.config.termination_state_index,
                           "request_anchor_protocol": "weighted geometric L/bytes from train unknown-L h=1 call cutoffs"
                               if self.config.request_scaled_initial_length else None,
                           "energy_estimator": "unbiased_u_statistic_straight_through_discrete_samples",
                           "consistency_first_call_probability": self.config.consistency_first_call_probability,
                           "discrete_gradient": "biased_straight_through_surrogate",
                           "log_length_std_range": [self.config.min_log_length_std,
                                                    self.config.max_log_length_std],
                           "history": history}
        return self

    def _query(self, query):
        if not self.statistics:
            raise ValueError("model must be fitted or loaded")
        return self._data(query, labels=False)

    def sample(self, query: Mapping, *, samples: int = 128, seed: int = 0) -> dict:
        if samples < 1:
            raise ValueError("sample count must be positive")
        q = self._query(query)
        self.network.eval()
        with torch.random.fork_rng(), torch.no_grad():
            torch.manual_seed(seed)
            result = self._draw(q, samples, differentiable=False)
        return {key: value.numpy() for key, value in result.items()}

    def forecast(self, query: Mapping, *, samples: int = 128, seed: int = 0) -> dict:
        if samples < 1:
            raise ValueError("sample count must be positive")
        q = self._query(query)
        self.network.eval()
        with torch.random.fork_rng(), torch.no_grad():
            torch.manual_seed(seed)
            allowance = q["allowance"]
            combined, first = self._compose_draw(q, allowance.clamp(max=1),
                (allowance - 1).clamp_min(0), samples, differentiable=False)
            self._extend_unknown_horizon(q, combined, samples)
            combined["call_samples"] = first["cost"]
            combined["task_remaining_samples"] = combined["cost"].clamp_min(0)
            combined["task_total_samples"] = q["spent"][:, None] + combined["task_remaining_samples"]
            combined["horizon_exhausted"] = ((combined["terminal"] < 0.5)
                & (combined["n"] >= allowance[:, None]))
        return {key: value.numpy() for key, value in combined.items()}

    def _extend_unknown_horizon(self, q, combined, samples):

        for _ in range(1, self.config.max_rollout_segments):
            unresolved = ((combined["terminal"] < .5)
                          & (q["call_limit_known"][:, None] < .5)
                          & (combined["n"] >= q["allowance"][:, None]))
            if not unresolved.any():
                break
            row, sample = torch.where(unresolved)
            count = len(row)
            span = torch.full((count,), float(self.config.max_horizon))
            query = {"task": q["task"][row], "state": combined["end_state"][row, sample],
                     "observation": torch.zeros(count, self.config.observation_dim),
                     "horizon": span, "allowance": span,
                     "input_length": combined["outgoing_length"][row, sample],
                     "input_known": torch.ones(count),
                     "spent": q["spent"][row] + combined["cost"][row, sample],
                     "completed_calls": q["completed_calls"][row] + combined["n"][row, sample],
                     "call_limit_known": torch.zeros(count), "is_terminal": torch.zeros(count),
                     "confirmed_call": torch.zeros(count), "request_group": q["request_group"][row],
                     "request_surface_complete": torch.zeros(count, dtype=torch.bool),
                     "task_text_bytes": q["task_text_bytes"][row]}
            continuation = {key: value[:, 0] for key, value in self._draw(query, 1, differentiable=False).items()}
            prior = {key: combined[key][row, sample] for key in ("n", "g", "b")}
            joined = compose_segments(prior, continuation)
            for key in ("n", "g", "b"):
                combined[key][row, sample] = joined[key]
            combined["cost"][row, sample] += continuation["cost"]
            for key in ("outgoing_length", "end_state", "terminal"):
                combined[key][row, sample] = continuation[key]

    def save(self, path: str | Path):
        if not self.statistics:
            raise ValueError("cannot save an unfitted model")
        torch.save({"schema_version": self.SCHEMA, "config": asdict(self.config),
                    "state_dict": self.network.state_dict(), "statistics": self.statistics,
                    "fit_report": self.fit_report}, path)

    @classmethod
    def load(cls, path: str | Path):
        saved = torch.load(path, map_location="cpu", weights_only=True)
        if saved.get("schema_version") not in {1, cls.SCHEMA}:
            raise ValueError("unsupported segment model schema")
        config = dict(saved["config"])


        config.setdefault("condition_scaling", "linear_cap")
        config.setdefault("ending_state_distribution", "gaussian")
        model = cls(SegmentConfig(**config))
        model.network.load_state_dict(saved["state_dict"])
        model.statistics = saved["statistics"]
        model.fit_report = saved["fit_report"]
        model.network.eval()
        return model

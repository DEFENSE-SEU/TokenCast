from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from token_prediction.dataset import PredictionPoint, PredictionPosition, PredictionTarget

from .base import (
    FitContext,
    ObservedTransition,
    RunContext,
    TokenForecast,
    TrainingExample,
    TrainingView,
    validate_fit_token_unit,
    validate_run_token_unit,
)
from .conditional_hazard import _is_remaining_target
from .mlp import _load_neural_dependencies, _view_condition_ids
from .neural_encoder import NeuralFeatureEncoder
from .neural_runtime import configure_deterministic_training, normalize_training_device

RATIO_GRID_ESTIMATOR_VERSION = 1
MASS_FLOOR = 1e-6
INITIALIZERS = frozenset({"prior", "lightgbm_quantile"})
SCORERS = frozenset({"mlp", "tilt"})
TILT_CURVATURES = frozenset({"free", "nonpositive"})
SUMMARY_WIDTH = 9
INTERVAL_WIDTH = 4
_CALL_SCOPED_POSITIONS = frozenset(
    {PredictionPosition.CALL_UPDATE, PredictionPosition.CALL_PRE, PredictionPosition.ATTEMPT_INIT}
)


def _derived_seed(seed: int, fold: int) -> int:
    return random.Random(f"ratio-grid-v{RATIO_GRID_ESTIMATOR_VERSION}:{seed}:{fold}").randrange(1, 2**31 - 1)


@dataclass(frozen=True)
class GridBatch:


    floors: Any
    lowers: Any
    uppers: Any
    valid: Any

    @property
    def ceiling(self) -> int:
        return int(self.uppers[0, -1])

    def rows(self, index: Any) -> "GridBatch":
        return GridBatch(self.floors[index], self.lowers[index], self.uppers[index], self.valid[index])


def build_grids(np: Any, floors: Any, ceiling: int, n_intervals: int) -> GridBatch:


    floors = np.asarray(floors, dtype=np.int64).reshape(-1)
    ceiling = int(ceiling)
    span = np.maximum(ceiling - floors, 0)
    ratios = np.arange(1, n_intervals, dtype=np.float64) / n_intervals
    cuts = np.rint(np.expm1(np.log1p(span.astype(np.float64))[:, None] * ratios[None, :])).astype(np.int64)
    lowers = np.concatenate([np.zeros((len(floors), 1), dtype=np.int64), cuts], axis=1) + floors[:, None]
    uppers = np.concatenate([lowers[:, 1:] - 1, np.maximum(ceiling, floors)[:, None]], axis=1)
    return GridBatch(floors=floors, lowers=lowers, uppers=uppers, valid=uppers >= lowers)


def grid_cdf(np: Any, grids: GridBatch, masses: Any, x: Any) -> Any:


    width = np.where(grids.valid, grids.uppers - grids.lowers + 1, 1).astype(np.float64)
    fraction = np.clip(
        (np.asarray(x, dtype=np.float64)[:, :, None] - grids.lowers[:, None, :].astype(np.float64) + 1.0) / width[:, None, :],
        0.0,
        1.0,
    )
    return np.einsum("nkm,nm->nk", fraction, np.where(grids.valid, masses, 0.0))


def transfer_reference(np: Any, previous: GridBatch, masses: Any, grids: GridBatch) -> Any:


    n, m = grids.lowers.shape
    probes = np.concatenate([grids.uppers, grids.lowers - 1, (grids.floors - 1)[:, None]], axis=1)
    cdf = grid_cdf(np, previous, masses, probes)
    tail = 1.0 - cdf[:, -1]
    q = np.where(grids.valid, cdf[:, :m] - cdf[:, m : 2 * m], 0.0)
    bad = (tail <= 1e-9) | ~np.isfinite(q).all(axis=1) | (q.sum(axis=1) <= 1e-12)
    q = q / np.where(bad, 1.0, tail)[:, None]
    uniform = grids.valid / grids.valid.sum(axis=1, keepdims=True)
    q = np.where(bad[:, None], uniform, q)
    q = np.where(grids.valid, np.maximum(q, MASS_FLOOR), 0.0)
    return q / q.sum(axis=1, keepdims=True)


def grid_quantile(np: Any, grids: GridBatch, masses: Any, tau: float) -> Any:


    cumulative = np.cumsum(masses, axis=1)
    index = np.minimum((cumulative < tau * (1.0 - 1e-12)).sum(axis=1), masses.shape[1] - 1)
    rows = np.arange(masses.shape[0])
    mass = masses[rows, index]
    before = cumulative[rows, index] - mass
    fraction = np.clip(np.where(mass > 0, (tau - before) / np.where(mass > 0, mass, 1.0), 0.5), 0.0, 1.0)
    lower = grids.lowers[rows, index].astype(np.float64)
    upper = grids.uppers[rows, index].astype(np.float64)
    return lower - 0.5 + fraction * (upper - lower + 1.0)


def label_interval(np: Any, grids: GridBatch, totals: Any) -> Any:


    y = np.clip(np.asarray(totals, dtype=np.float64), grids.floors, grids.uppers[:, -1])
    hit = grids.valid & (grids.uppers >= y[:, None])
    return np.where(hit.any(axis=1), np.argmax(hit, axis=1), grids.lowers.shape[1] - 1)


def posterior(np: Any, log_q: Any, g: Any, valid: Any) -> Any:


    logits = np.where(valid, log_q.astype(np.float64) + g.astype(np.float64), -np.inf)
    logits = logits - logits.max(axis=1, keepdims=True)
    p = np.exp(logits)
    p = np.where(valid, np.maximum(p / p.sum(axis=1, keepdims=True), MASS_FLOOR), 0.0)
    return p / p.sum(axis=1, keepdims=True)


def reference_encoding(
    np: Any,
    grids: GridBatch,
    q: Any,
    log_ceiling: float,
    init_quantiles: Any | None = None,
) -> tuple[Any, Any, Any]:


    floors = grids.floors.astype(np.float64)
    span_log = np.maximum(np.log1p(grids.uppers[:, -1] - floors), 1e-6)
    size = grids.valid.sum(axis=1)
    log_q = np.where(grids.valid, np.log(np.maximum(q, MASS_FLOOR)), math.log(MASS_FLOOR))
    entropy = np.where(size > 1, -(q * log_q * grids.valid).sum(axis=1) / np.maximum(np.log(size), 1e-6), 0.0)
    if init_quantiles is None:
        fresh = [np.zeros(len(floors)) for _ in range(3)]
    else:
        fresh = [np.log1p(np.maximum(np.asarray(init_quantiles, dtype=np.float64)[:, k] - floors, 0.0)) / span_log for k in range(3)]
    summary = np.stack(
        [
            np.log1p(floors) / log_ceiling,
            span_log / log_ceiling,
            np.log1p(np.maximum(grid_quantile(np, grids, q, 0.5) - floors, 0.0)) / span_log,
            np.log1p(np.maximum(grid_quantile(np, grids, q, 0.05) - floors, 0.0)) / span_log,
            np.log1p(np.maximum(grid_quantile(np, grids, q, 0.95) - floors, 0.0)) / span_log,
            entropy,
            *fresh,
        ],
        axis=1,
    ).astype(np.float32)
    rank = np.cumsum(grids.valid, axis=1) - 1
    interval = np.stack(
        [
            np.log1p(np.maximum(grids.lowers - grids.floors[:, None], 0)) / span_log[:, None],
            np.log1p(np.maximum(grids.uppers - grids.floors[:, None] + 1, 0)) / span_log[:, None],
            log_q + np.log(np.maximum(size, 1))[:, None],
            (rank + 0.5) / np.maximum(size, 1)[:, None],
        ],
        axis=2,
    )
    interval = np.where(grids.valid[:, :, None], interval, 0.0).astype(np.float32)
    return summary, interval, log_q.astype(np.float32)


def _build_scorer(
    torch: Any,
    feature_dim: int,
    hidden_dim: int,
    scorer: str = "mlp",
    tilt_curvature: str = "free",
) -> Any:


    class Scorer(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.event = torch.nn.Sequential(torch.nn.Linear(feature_dim + SUMMARY_WIDTH, hidden_dim), torch.nn.SiLU())
            if scorer == "tilt":
                self.score = torch.nn.Linear(hidden_dim, 3)
                last = self.score
            else:
                self.score = torch.nn.Sequential(
                    torch.nn.Linear(hidden_dim + INTERVAL_WIDTH, hidden_dim),
                    torch.nn.SiLU(),
                    torch.nn.Linear(hidden_dim, 1),
                )
                last = self.score[-1]

            torch.nn.init.zeros_(last.weight)
            torch.nn.init.zeros_(last.bias)

        def forward(self, x: Any, summary: Any, interval: Any, mask: Any) -> Any:
            hidden = self.event(torch.cat([x, summary], dim=-1))
            if scorer == "tilt":
                theta = self.score(hidden)
                u = 0.5 * (interval[..., 0] + interval[..., 1])
                first = ((torch.cumsum(mask.to(torch.int64), dim=1) == 1) & mask).to(u.dtype)
                curvature = theta[:, 1:2]
                if tilt_curvature == "nonpositive":
                    curvature = -torch.nn.functional.silu(curvature)
                g = theta[:, 0:1] * u + curvature * u * u + theta[:, 2:3] * first
            else:
                expanded = hidden.unsqueeze(1).expand(-1, interval.shape[1], -1)
                g = self.score(torch.cat([expanded, interval], dim=-1)).squeeze(-1)
            return g.masked_fill(~mask, 0.0)

    return Scorer()


def _ratio_loss(torch: Any, g: Any, log_q: Any, z: Any, mask: Any, weights: Any) -> Any:


    positive = -torch.nn.functional.logsigmoid(g.gather(1, z.unsqueeze(1)).squeeze(1))
    negative = -(log_q.exp() * torch.nn.functional.logsigmoid(-g) * mask.to(g.dtype)).sum(dim=1)
    return torch.sum((positive + negative) * weights) / torch.sum(weights)


@dataclass
class _Records:


    x: Any
    offsets: Any
    totals: Any
    weights: Any
    labels: Any
    task_ids: tuple[str, ...]
    chains: tuple[tuple[int, ...], ...]
    initial: Any = None
    init_quantiles: Any = None


@dataclass
class _ReplayBatch:
    x: Any
    summary: Any
    interval: Any
    log_q: Any
    mask: Any
    z: Any
    weights: Any
    medians: Any
    labels: Any


def _recursion_key(point: PredictionPoint) -> tuple[str, ...]:
    if point.position in _CALL_SCOPED_POSITIONS:
        return (point.trajectory_id, str(point.logical_call_id), str(point.attempt_id))
    return (point.trajectory_id,)


def _chains(examples: Sequence[TrainingExample]) -> tuple[tuple[int, ...], ...]:
    grouped: dict[tuple[str, ...], list[tuple[int, str, int]]] = {}
    for index, example in enumerate(examples):
        grouped.setdefault(_recursion_key(example.point), []).append(
            (example.point.cutoff_event_seq, example.point.point_id, index)
        )
    return tuple(tuple(index for _, _, index in sorted(items)) for _, items in sorted(grouped.items()))


def _point_offset(point: PredictionPoint, remaining_target: bool, offset_features: tuple[str, ...] = ()) -> int:


    if offset_features:
        total = 0.0
        for name in offset_features:
            value = point.features.get(name)
            if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value > 0:
                total += float(value)
        return int(total)
    if not remaining_target or point.known_offset_tokens is None:
        return 0
    return int(point.known_offset_tokens)


def _records(
    np: Any,
    view: TrainingView,
    encoder: NeuralFeatureEncoder,
    remaining_target: bool,
    offset_features: tuple[str, ...] = (),
) -> _Records:
    examples = view.observed_examples
    matrix = encoder.transform(tuple(example.point for example in examples)).matrix.astype(np.float32)
    offsets = np.array(
        [_point_offset(example.point, remaining_target, offset_features) for example in examples], dtype=np.int64
    )
    labels = np.array([float(example.target_value) for example in examples], dtype=np.float64)
    return _Records(
        x=matrix,
        offsets=offsets,
        totals=labels + offsets,
        weights=np.array([float(example.sample_weight) for example in examples], dtype=np.float32),
        labels=labels,
        task_ids=tuple(example.point.task_id for example in examples),
        chains=_chains(examples),
    )


def _forecast_quantiles(np: Any, fitted: Any, examples: Sequence[TrainingExample], view: TrainingView) -> Any:


    grouped: dict[str, list[int]] = {}
    for index, example in enumerate(examples):
        grouped.setdefault(example.point.trajectory_id, []).append(index)
    out = np.zeros((len(examples), 3), dtype=np.float64)
    for trajectory_id, indices in sorted(grouped.items()):
        ordered = sorted(indices, key=lambda i: (examples[i].point.cutoff_event_seq, examples[i].point.point_id))
        first = examples[ordered[0]].point
        session = fitted.start(
            RunContext(
                first.task_id,
                trajectory_id,
                first.run_id,
                dataset_id=view.dataset_id,
                condition_id=first.condition_id,
                target=first.target,
                input_contract_id=view.input_contract_id,
                target_unit_contract_id=first.token_unit_contract_id,
                observation_unit_contract_id=first.token_unit_contract_id,
            )
        )
        for index in ordered:
            forecast = session.predict(examples[index].point)
            out[index] = (forecast.lower, forecast.point, forecast.upper)
    return out


class _GridSpace:


    def __init__(
        self,
        np: Any,
        *,
        ceiling: int,
        n_intervals: int,
        prior_masses: Any,
        initializer_mix: float = 0.0,
    ) -> None:
        self.np = np
        self.ceiling = ceiling
        self.n_intervals = n_intervals
        self.prior_grid = build_grids(np, [0], ceiling, n_intervals)
        self.prior_masses = prior_masses.reshape(1, -1)
        self.initializer_mix = float(initializer_mix)
        self.log_ceiling = max(math.log1p(ceiling), 1e-6)

    def initializer_masses(self, quantiles: Any, floors: Any) -> Any:


        np = self.np
        ceiling = float(self.ceiling)
        floors = np.clip(np.asarray(floors, dtype=np.float64).reshape(-1), 0.0, ceiling)
        grids = build_grids(np, floors.astype(np.int64), self.ceiling, self.n_intervals)
        span = np.maximum(ceiling - floors, 0.0)
        rel = np.clip(np.asarray(quantiles, dtype=np.float64).reshape(-1, 3) - floors[:, None], 0.0, span[:, None])
        rel = np.maximum.accumulate(rel, axis=1)
        knots_x = np.log1p(np.column_stack([np.zeros(len(rel)), rel, span])) + np.arange(5) * 1e-9
        knots_y = np.array([0.0, 0.05, 0.5, 0.95, 1.0])
        lowers = np.log1p(np.maximum(grids.lowers - floors[:, None], 0.0))
        uppers = np.log1p(np.maximum(grids.uppers + 1.0 - floors[:, None], 0.0))
        masses = np.stack(
            [np.interp(hi, kx, knots_y) - np.interp(lo, kx, knots_y) for hi, lo, kx in zip(uppers, lowers, knots_x)]
        )
        masses = np.where(grids.valid, np.maximum(masses, 0.0), 0.0)
        masses = masses / np.maximum(masses.sum(axis=1, keepdims=True), 1e-12)
        if self.initializer_mix > 0.0:
            prior = transfer_reference(np, self._repeat_prior(len(floors))[0], self._repeat_prior(len(floors))[1], grids)
            masses = (1.0 - self.initializer_mix) * masses + self.initializer_mix * prior
        masses = np.where(grids.valid, np.maximum(masses, MASS_FLOOR), 0.0)
        return masses / masses.sum(axis=1, keepdims=True)

    def _repeat_prior(self, n: int) -> tuple[GridBatch, Any]:
        np = self.np
        return (
            GridBatch(
                np.repeat(self.prior_grid.floors, n),
                np.repeat(self.prior_grid.lowers, n, axis=0),
                np.repeat(self.prior_grid.uppers, n, axis=0),
                np.repeat(self.prior_grid.valid, n, axis=0),
            ),
            np.repeat(self.prior_masses, n, axis=0),
        )

    def step(
        self,
        offsets: Any,
        previous: tuple[GridBatch, Any] | None,
        initial: Any | None = None,
        init_quantiles: Any | None = None,
    ) -> tuple[GridBatch, Any, Any, Any, Any]:


        np = self.np
        grids = build_grids(np, offsets, self.ceiling, self.n_intervals)
        if previous is None and initial is not None:
            q = np.where(grids.valid, np.maximum(initial, MASS_FLOOR), 0.0)
            q = q / q.sum(axis=1, keepdims=True)
        else:
            source, masses = previous if previous is not None else self._repeat_prior(len(offsets))
            q = transfer_reference(np, source, masses, grids)
        summary, interval, log_q = reference_encoding(np, grids, q, self.log_ceiling, init_quantiles)
        return grids, q, summary, interval, log_q


def _replay(
    torch: Any,
    space: _GridSpace,
    model: Any,
    records: _Records,
    chains: Sequence[Sequence[int]],
    *,
    recursive: bool,
    device: str,
) -> _ReplayBatch:


    np = space.np
    rows = np.array(sorted(index for chain in chains for index in chain), dtype=np.int64)
    position = {int(index): row for row, index in enumerate(rows)}
    n, m = len(rows), space.n_intervals
    summary = np.zeros((n, SUMMARY_WIDTH), dtype=np.float32)
    interval = np.zeros((n, m, INTERVAL_WIDTH), dtype=np.float32)
    log_q = np.full((n, m), math.log(MASS_FLOOR), dtype=np.float32)
    mask = np.zeros((n, m), dtype=bool)
    z = np.zeros(n, dtype=np.int64)
    medians = np.zeros(n, dtype=np.float64)
    state_grid: list[GridBatch | None] = [None] * len(chains)
    state_mass: list[Any] = [None] * len(chains)
    model.eval()
    for step in range(max(len(chain) for chain in chains)):
        active = [chain_id for chain_id, chain in enumerate(chains) if len(chain) > step]
        indices = np.array([chains[chain_id][step] for chain_id in active], dtype=np.int64)
        local = np.array([position[int(index)] for index in indices], dtype=np.int64)
        previous = None
        if recursive and all(state_grid[chain_id] is not None for chain_id in active) and step > 0:
            previous = (
                GridBatch(
                    np.concatenate([state_grid[c].floors for c in active]),
                    np.concatenate([state_grid[c].lowers for c in active]),
                    np.concatenate([state_grid[c].uppers for c in active]),
                    np.concatenate([state_grid[c].valid for c in active]),
                ),
                np.concatenate([state_mass[c] for c in active]),
            )
        initial = records.initial[indices] if (records.initial is not None and previous is None) else None
        fresh = records.init_quantiles[indices] if records.init_quantiles is not None else None
        grids, _, summary[local], interval[local], log_q[local] = space.step(records.offsets[indices], previous, initial, fresh)
        mask[local] = grids.valid
        z[local] = label_interval(np, grids, records.totals[indices])
        with torch.inference_mode():
            g = model(
                torch.from_numpy(records.x[indices]).to(device),
                torch.from_numpy(summary[local]).to(device),
                torch.from_numpy(interval[local]).to(device),
                torch.from_numpy(mask[local]).to(device),
            ).cpu().numpy()
        p = posterior(np, log_q[local], g, grids.valid)
        medians[local] = grid_quantile(np, grids, p, 0.5) - records.offsets[indices]
        for k, chain_id in enumerate(active):
            state_grid[chain_id] = grids.rows(slice(k, k + 1))
            state_mass[chain_id] = p[k : k + 1]
    return _ReplayBatch(
        x=torch.from_numpy(records.x[rows]).to(device),
        summary=torch.from_numpy(summary).to(device),
        interval=torch.from_numpy(interval).to(device),
        log_q=torch.from_numpy(log_q).to(device),
        mask=torch.from_numpy(mask).to(device),
        z=torch.from_numpy(z).to(device),
        weights=torch.from_numpy(records.weights[rows]).to(device),
        medians=medians,
        labels=records.labels[rows],
    )


def _train_epoch(torch: Any, model: Any, optimizer: Any, batch: _ReplayBatch, *, batch_size: int, generator: Any) -> float:
    model.train()
    n = int(batch.x.shape[0])
    order = torch.randperm(n, generator=generator)
    total = 0.0
    for start in range(0, n, batch_size):
        idx = order[start : start + batch_size]
        optimizer.zero_grad()
        g = model(batch.x[idx], batch.summary[idx], batch.interval[idx], batch.mask[idx])
        loss = _ratio_loss(torch, g, batch.log_q[idx], batch.z[idx], batch.mask[idx], batch.weights[idx])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total += float(loss.detach()) * len(idx)
    return total / max(n, 1)


def _weighted_mae(np: Any, batch: _ReplayBatch) -> float:
    weights = batch.weights.cpu().numpy().astype(np.float64)
    return float(np.sum(weights * np.abs(batch.medians - batch.labels)) / np.sum(weights))


def _concat_batches(torch: Any, np: Any, batches: Sequence[_ReplayBatch]) -> _ReplayBatch:
    return _ReplayBatch(
        x=torch.cat([b.x for b in batches]),
        summary=torch.cat([b.summary for b in batches]),
        interval=torch.cat([b.interval for b in batches]),
        log_q=torch.cat([b.log_q for b in batches]),
        mask=torch.cat([b.mask for b in batches]),
        z=torch.cat([b.z for b in batches]),
        weights=torch.cat([b.weights for b in batches]),
        medians=np.concatenate([b.medians for b in batches]),
        labels=np.concatenate([b.labels for b in batches]),
    )


@dataclass
class RatioGridSession:
    context: RunContext
    target: PredictionTarget
    position: PredictionPosition
    allowed_condition_ids: tuple[str, ...]
    encoder: NeuralFeatureEncoder
    model: Any
    space: _GridSpace
    remaining_target: bool
    recursive: bool
    quantiles: tuple[float, float, float]
    offset_features: tuple[str, ...] = ()
    initializer_session: Any = None
    state: tuple[GridBatch, Any] | None = None
    state_key: tuple[str, ...] | None = None

    def predict(self, point: PredictionPoint) -> TokenForecast:
        if point.trajectory_id != self.context.trajectory_id:
            raise ValueError("prediction point trajectory_id does not match the session")
        if point.token_unit_contract_id != self.context.target_unit_contract_id:
            raise ValueError("prediction point token unit does not match the session")
        if point.target != self.target or point.position != self.position:
            raise ValueError("prediction point target/position differ from the fitted cell")
        if point.condition_id not in self.allowed_condition_ids:
            raise ValueError(f"condition_id {point.condition_id!r} is outside the bundle scope")
        np, torch = _load_neural_dependencies()
        offset = _point_offset(point, self.remaining_target, self.offset_features)
        key = _recursion_key(point)
        previous = self.state if (self.recursive and self.state is not None and self.state_key == key) else None
        initial = fresh = None
        if self.initializer_session is not None:
            seed = self.initializer_session.predict(point)
            fresh = np.array([[seed.lower + offset, seed.point + offset, seed.upper + offset]], dtype=np.float64)
            if previous is None:
                initial = self.space.initializer_masses(fresh, np.array([offset], dtype=np.float64))
        grids, _, summary, interval, log_q = self.space.step(np.array([offset], dtype=np.int64), previous, initial, fresh)
        x = self.encoder.transform((point,)).matrix.astype(np.float32)
        self.model.eval()
        with torch.inference_mode():
            g = self.model(
                torch.from_numpy(x),
                torch.from_numpy(summary),
                torch.from_numpy(interval),
                torch.from_numpy(grids.valid),
            ).numpy()
        p = posterior(np, log_q, g, grids.valid)
        self.state, self.state_key = (grids, p), key
        raw = tuple(float(grid_quantile(np, grids, p, tau)[0]) - offset for tau in self.quantiles)
        point_value = max(0.0, raw[1])
        lower = min(max(0.0, raw[0]), point_value)
        upper = max(max(0.0, raw[2]), point_value)
        return TokenForecast(
            point_id=point.point_id,
            target=self.target,
            lower=lower,
            point=point_value,
            upper=upper,
            raw_lower=raw[0],
            raw_point=raw[1],
            raw_upper=raw[2],
        )

    def observe(self, transition: ObservedTransition) -> None:
        del transition


@dataclass(frozen=True)
class FittedRatioGrid:
    estimator_id: str
    target: PredictionTarget
    position: PredictionPosition
    dataset_id: str
    input_contract_id: str | None
    token_unit_contract_id: str | None
    allowed_condition_ids: tuple[str, ...]
    encoder: NeuralFeatureEncoder
    model: Any
    space: Any
    remaining_target: bool
    recursive: bool
    quantiles: tuple[float, float, float]
    fit_report: Mapping[str, Any]
    offset_features: tuple[str, ...] = ()
    initializer: Any = None

    def start(self, context: RunContext) -> RatioGridSession:
        validate_run_token_unit(self.token_unit_contract_id, context)
        if context.target is not None and context.target != self.target:
            raise ValueError("RunContext target does not match the fitted target")
        return RatioGridSession(
            initializer_session=self.initializer.start(context) if self.initializer is not None else None,
            context=context,
            target=self.target,
            position=self.position,
            allowed_condition_ids=self.allowed_condition_ids,
            encoder=self.encoder,
            model=self.model,
            space=self.space,
            remaining_target=self.remaining_target,
            recursive=self.recursive,
            quantiles=self.quantiles,
            offset_features=self.offset_features,
        )


class RatioGridEstimator:
    estimator_id = "ratio_grid"

    def __init__(
        self,
        *,
        n_intervals: int = 48,
        hidden_dim: int = 64,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        batch_size: int = 256,
        max_rounds: int = 60,
        epochs_per_round: int = 1,
        patience: int = 10,
        replay_folds: int = 3,
        recursive: bool = True,
        upper_bound: int | None = None,
        upper_bound_margin: float = 2.0,
        offset_features: Sequence[str] | None = None,
        initializer: str = "prior",
        initializer_params: Mapping[str, Any] | None = None,
        initializer_mix: float = 0.01,
        scorer: str = "mlp",
        tilt_curvature: str = "free",
        training_device: str = "cpu",
    ) -> None:
        for name, value, low, high in (
            ("n_intervals", n_intervals, 1, 1024),
            ("hidden_dim", hidden_dim, 1, 1024),
            ("batch_size", batch_size, 1, 65536),
            ("max_rounds", max_rounds, 1, 1000),
            ("epochs_per_round", epochs_per_round, 1, 100),
            ("replay_folds", replay_folds, 1, 20),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or not low <= value <= high:
                raise ValueError(f"{name} must be an integer in [{low}, {high}]")
        if isinstance(patience, bool) or not isinstance(patience, int) or not 1 <= patience <= max_rounds:
            raise ValueError("patience must be an integer in [1, max_rounds]")
        for name, value in (("learning_rate", learning_rate), ("upper_bound_margin", upper_bound_margin)):
            if not (isinstance(value, (int, float)) and math.isfinite(value) and value > 0):
                raise ValueError(f"{name} must be finite and positive")
        if not (isinstance(weight_decay, (int, float)) and math.isfinite(weight_decay) and weight_decay >= 0):
            raise ValueError("weight_decay must be finite and non-negative")
        if upper_bound is not None and (isinstance(upper_bound, bool) or not isinstance(upper_bound, int) or upper_bound < 1):
            raise ValueError("upper_bound must be a positive integer or None")
        if not isinstance(recursive, bool):
            raise ValueError("recursive must be a boolean")
        names = tuple(offset_features or ())
        if isinstance(offset_features, str) or not all(isinstance(name, str) and name.strip() for name in names):
            raise ValueError("offset_features must be a list of feature names")
        self.offset_features = names
        if initializer not in INITIALIZERS:
            raise ValueError(f"initializer must be one of {sorted(INITIALIZERS)}")
        if initializer_params is not None and not isinstance(initializer_params, Mapping):
            raise ValueError("initializer_params must be a mapping")
        if not (isinstance(initializer_mix, (int, float)) and math.isfinite(initializer_mix) and 0 <= initializer_mix < 1):
            raise ValueError("initializer_mix must be in [0, 1)")
        self.initializer = initializer
        self.initializer_params = dict(initializer_params or {})
        self.initializer_mix = float(initializer_mix)
        if scorer not in SCORERS:
            raise ValueError(f"scorer must be one of {sorted(SCORERS)}")
        if tilt_curvature not in TILT_CURVATURES:
            raise ValueError(f"tilt_curvature must be one of {sorted(TILT_CURVATURES)}")
        self.scorer = scorer
        self.tilt_curvature = tilt_curvature
        self.n_intervals = n_intervals
        self.hidden_dim = hidden_dim
        self.learning_rate = float(learning_rate)
        self.weight_decay = float(weight_decay)
        self.batch_size = batch_size
        self.max_rounds = max_rounds
        self.epochs_per_round = epochs_per_round
        self.patience = patience
        self.replay_folds = replay_folds
        self.recursive = recursive
        self.upper_bound = upper_bound
        self.upper_bound_margin = float(upper_bound_margin)
        self.training_device = normalize_training_device(training_device)

    def fit(self, train: TrainingView, validation: TrainingView, context: FitContext) -> FittedRatioGrid:
        token_unit_contract_id = validate_fit_token_unit(train, validation)
        if train.position != validation.position or train.target != validation.target:
            raise ValueError("train and validation views must share position and target")
        conditions = _view_condition_ids(train, description="train")
        if conditions != _view_condition_ids(validation, description="validation"):
            raise ValueError("train and validation views must share condition scope")
        if not train.observed_examples or not validation.observed_examples:
            raise ValueError("ratio grid training requires observed labels in train and validation")
        quantiles = (context.interval_alpha / 2, 0.5, 1 - context.interval_alpha / 2)
        np, torch = _load_neural_dependencies()
        seed = _derived_seed(context.seed, context.fold)
        device = configure_deterministic_training(torch, seed=seed, device=self.training_device)
        remaining_target = _is_remaining_target(train.target)
        encoder = NeuralFeatureEncoder.fit(tuple(example.point for example in train.observed_examples))
        if encoder.schema.output_width <= 0:
            raise ValueError("ratio grid requires at least one usable train-fold feature")
        train_records = _records(np, train, encoder, remaining_target, self.offset_features)
        validation_records = _records(np, validation, encoder, remaining_target, self.offset_features)

        ceiling = self.upper_bound
        if ceiling is None:
            ceiling = int(math.ceil(float(train_records.totals.max()) * self.upper_bound_margin)) + 1
        prior_grid = build_grids(np, [0], ceiling, self.n_intervals)
        bins = label_interval(
            np,
            GridBatch(
                np.repeat(prior_grid.floors, len(train_records.totals)),
                np.repeat(prior_grid.lowers, len(train_records.totals), axis=0),
                np.repeat(prior_grid.uppers, len(train_records.totals), axis=0),
                np.repeat(prior_grid.valid, len(train_records.totals), axis=0),
            ),
            train_records.totals,
        )
        prior = np.bincount(bins, weights=train_records.weights.astype(np.float64), minlength=self.n_intervals)
        prior = np.where(prior_grid.valid[0], np.maximum(prior / prior.sum(), MASS_FLOOR), 0.0)
        prior /= prior.sum()
        space = _GridSpace(
            np, ceiling=ceiling, n_intervals=self.n_intervals, prior_masses=prior, initializer_mix=self.initializer_mix
        )


        tasks = sorted(set(train_records.task_ids))
        random.Random(seed).shuffle(tasks)
        folds = max(1, min(self.replay_folds, len(tasks)))
        fold_of_task = {task: index % folds for index, task in enumerate(tasks)}
        chain_fold = [fold_of_task[train_records.task_ids[chain[0]]] for chain in train_records.chains]


        initializer_fitted = None
        initializer_validation_mae = None
        if self.initializer == "lightgbm_quantile":
            from .lightgbm import LightGBMQuantileEstimator

            train_examples = train.observed_examples
            validation_examples = validation.observed_examples

            def subset_view(indices: Sequence[int]) -> TrainingView:
                return TrainingView(
                    dataset_id=train.dataset_id,
                    position=train.position,
                    target=train.target,
                    examples=tuple(train_examples[index] for index in indices),
                    input_contract_id=train.input_contract_id,
                    token_unit_contract_id=train.token_unit_contract_id,
                )

            oof = np.zeros((len(train_examples), 3), dtype=np.float64)
            if folds > 1:
                for fold in range(folds):
                    fit_indices = [i for i, ex in enumerate(train_examples) if fold_of_task[ex.point.task_id] != fold]
                    held = [i for i, ex in enumerate(train_examples) if fold_of_task[ex.point.task_id] == fold]
                    component = LightGBMQuantileEstimator(**self.initializer_params).fit(subset_view(fit_indices), validation, context)
                    oof[held] = _forecast_quantiles(np, component, [train_examples[i] for i in held], train)
            initializer_fitted = LightGBMQuantileEstimator(**self.initializer_params).fit(train, validation, context)
            if folds == 1:
                oof = _forecast_quantiles(np, initializer_fitted, train_examples, train)
            validation_quantiles = _forecast_quantiles(np, initializer_fitted, validation_examples, validation)
            train_records.init_quantiles = oof + train_records.offsets[:, None]
            validation_records.init_quantiles = validation_quantiles + validation_records.offsets[:, None]
            train_records.initial = space.initializer_masses(train_records.init_quantiles, train_records.offsets)
            validation_records.initial = space.initializer_masses(validation_records.init_quantiles, validation_records.offsets)
            weights = validation_records.weights.astype(np.float64)
            initializer_validation_mae = float(
                np.sum(weights * np.abs(validation_quantiles[:, 1] - validation_records.labels)) / np.sum(weights)
            )

        def make_model() -> Any:
            return _build_scorer(
                torch, encoder.schema.output_width, self.hidden_dim, self.scorer, self.tilt_curvature
            ).to(device)

        def make_optimizer(model: Any) -> Any:
            return torch.optim.AdamW(model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)

        deploy = make_model()
        deploy_optimizer = make_optimizer(deploy)
        references = [make_model() for _ in range(folds)] if folds > 1 else []
        reference_optimizers = [make_optimizer(model) for model in references]
        generator = torch.Generator().manual_seed(seed)

        def replay(model: Any, chain_ids: Sequence[int]) -> _ReplayBatch:
            return _replay(
                torch, space, model, train_records, [train_records.chains[chain_id] for chain_id in chain_ids],
                recursive=self.recursive, device=device,
            )

        history: list[dict[str, float]] = []
        best_mae, best_round, best_state = math.inf, 0, None
        for round_index in range(self.max_rounds):

            if folds > 1:
                held_out = [replay(references[fold], [cid for cid, f in enumerate(chain_fold) if f == fold]) for fold in range(folds)]
                own = [replay(references[fold], [cid for cid, f in enumerate(chain_fold) if f != fold]) for fold in range(folds)]
                deploy_batch = _concat_batches(torch, np, held_out)
            else:
                deploy_batch = replay(deploy, list(range(len(train_records.chains))))
                own = []
            train_loss = 0.0
            for _ in range(self.epochs_per_round):
                for fold, batch in enumerate(own):
                    _train_epoch(torch, references[fold], reference_optimizers[fold], batch, batch_size=self.batch_size, generator=generator)
                train_loss = _train_epoch(torch, deploy, deploy_optimizer, deploy_batch, batch_size=self.batch_size, generator=generator)
            validation_batch = _replay(
                torch, space, deploy, validation_records, validation_records.chains, recursive=self.recursive, device=device
            )
            with torch.inference_mode():
                deploy.eval()
                validation_loss = float(
                    _ratio_loss(
                        torch,
                        deploy(validation_batch.x, validation_batch.summary, validation_batch.interval, validation_batch.mask),
                        validation_batch.log_q, validation_batch.z, validation_batch.mask, validation_batch.weights,
                    )
                )
            mae = _weighted_mae(np, validation_batch)
            history.append({"round": round_index, "train_loss": train_loss, "validation_loss": validation_loss, "validation_mae": mae})
            if mae < best_mae:
                best_mae, best_round = mae, round_index
                best_state = {k: v.detach().clone() for k, v in deploy.state_dict().items()}
            elif round_index - best_round >= self.patience:
                break
        deploy.load_state_dict(best_state)
        deploy.to("cpu").eval()
        return FittedRatioGrid(
            estimator_id=self.estimator_id,
            target=train.target,
            position=train.position,
            dataset_id=train.dataset_id,
            input_contract_id=train.input_contract_id,
            token_unit_contract_id=token_unit_contract_id,
            allowed_condition_ids=conditions,
            encoder=encoder,
            model=deploy,
            space=space,
            remaining_target=remaining_target,
            recursive=self.recursive,
            quantiles=quantiles,
            offset_features=self.offset_features,
            initializer=initializer_fitted,
            fit_report={
                "estimator_id": self.estimator_id,
                "estimator_version": RATIO_GRID_ESTIMATOR_VERSION,
                "offset_features": list(self.offset_features),
                "initializer": self.initializer,
                "initializer_mix": self.initializer_mix,
                "initializer_validation_mae": initializer_validation_mae,
                "scorer": self.scorer,
                "tilt_curvature": self.tilt_curvature,
                "n_intervals": self.n_intervals,
                "grid_intervals": int(prior_grid.valid.sum()),
                "ceiling": ceiling,
                "hidden_dim": self.hidden_dim,
                "input_dim": encoder.schema.output_width,
                "replay_folds": folds,
                "recursive": self.recursive,
                "remaining_target": remaining_target,
                "train_points": int(len(train_records.totals)),
                "train_chains": len(train_records.chains),
                "validation_points": int(len(validation_records.totals)),
                "rounds_run": len(history),
                "best_round": best_round,
                "best_validation_mae": best_mae,
                "prior_median": float(grid_quantile(np, prior_grid, prior.reshape(1, -1), 0.5)[0]),
                "history": history,
                "seed": seed,
                "quantiles": list(quantiles),
            },
        )


__all__ = [
    "FittedRatioGrid",
    "GridBatch",
    "RatioGridEstimator",
    "RatioGridSession",
    "build_grids",
    "grid_cdf",
    "grid_quantile",
    "label_interval",
    "posterior",
    "reference_encoding",
    "transfer_reference",
]

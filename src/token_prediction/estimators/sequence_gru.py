from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Any, Mapping

from token_prediction.dataset import PredictionPoint, PredictionPosition, PredictionTarget

from .base import (
    FitContext,
    ObservedTransition,
    RunContext,
    TokenForecast,
    TrainingView,
    validate_fit_token_unit,
    validate_run_token_unit,
)
from .mlp import _load_neural_dependencies, _view_condition_ids, _weighted_quantile_loss
from .neural_encoder import NeuralFeatureEncoder
from .neural_runtime import configure_deterministic_training, normalize_training_device

SEQUENCE_GRU_ESTIMATOR_VERSION = 1


def _derived_seed(seed: int, fold: int) -> int:
    return random.Random(f"sequence-gru-v{SEQUENCE_GRU_ESTIMATOR_VERSION}:{seed}:{fold}").randrange(1, 2**31 - 1)


def _weighted_median(view: TrainingView) -> float:
    ordered = sorted((float(ex.target_value), float(ex.sample_weight)) for ex in view.examples)
    threshold = sum(w for _, w in ordered) / 2.0
    cumulative = 0.0
    for value, weight in ordered:
        cumulative += weight
        if cumulative >= threshold:
            return max(1.0, value)
    return 1.0


def _repair(raw: tuple[float, float, float]) -> tuple[float, float, float]:
    point = max(0.0, raw[1])
    return min(max(0.0, raw[0]), point), point, max(max(0.0, raw[2]), point)


@dataclass
class SequenceGRUSession:
    context: RunContext
    target: PredictionTarget
    position: PredictionPosition
    allowed_condition_ids: tuple[str, ...]
    encoder: NeuralFeatureEncoder
    model: Any
    target_scale: float = 1.0
    hidden: Any = None

    def predict(self, point: PredictionPoint) -> TokenForecast:
        if point.trajectory_id != self.context.trajectory_id:
            raise ValueError("prediction point trajectory_id does not match the session")
        if point.token_unit_contract_id != self.context.target_unit_contract_id:
            raise ValueError("prediction point token unit does not match the session")
        if point.target != self.target or point.position != self.position:
            raise ValueError("prediction point target/position differ from the fitted cell")
        if point.condition_id not in self.allowed_condition_ids:
            raise ValueError(f"condition_id {point.condition_id!r} is outside the bundle scope")
        _, torch = _load_neural_dependencies()
        encoded = self.encoder.transform((point,))
        x = torch.from_numpy(encoded.matrix).to(dtype=torch.float32).unsqueeze(0)
        self.model.eval()
        with torch.inference_mode():
            out, self.hidden = self.model.step(x, self.hidden)
            # Restore the fitted target scale before returning forecast values.
            raw = tuple(float(v) * self.target_scale for v in out[0, 0].tolist())
        lower, center, upper = _repair(raw)
        return TokenForecast(
            point_id=point.point_id,
            target=self.target,
            lower=lower,
            point=center,
            upper=upper,
            raw_lower=raw[0],
            raw_point=raw[1],
            raw_upper=raw[2],
        )

    def observe(self, transition: ObservedTransition) -> None:
        del transition


@dataclass(frozen=True)
class FittedSequenceGRU:
    estimator_id: str
    target: PredictionTarget
    position: PredictionPosition
    dataset_id: str
    input_contract_id: str | None
    token_unit_contract_id: str | None
    allowed_condition_ids: tuple[str, ...]
    encoder: NeuralFeatureEncoder
    model: Any
    fit_report: Mapping[str, Any]
    target_scale: float = 1.0

    def start(self, context: RunContext) -> SequenceGRUSession:
        validate_run_token_unit(self.token_unit_contract_id, context)
        if context.target is not None and context.target != self.target:
            raise ValueError("RunContext target does not match the fitted target")
        return SequenceGRUSession(
            context=context,
            target=self.target,
            position=self.position,
            allowed_condition_ids=self.allowed_condition_ids,
            encoder=self.encoder,
            model=self.model,
            target_scale=self.target_scale,
        )


def _build_network(torch: Any, input_dim: int, hidden_dim: int) -> Any:
    class Network(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.gru = torch.nn.GRU(input_dim, hidden_dim, batch_first=True)
            self.head = torch.nn.Linear(hidden_dim, 3)

        def forward(self, x: Any) -> Any:
            out, _ = self.gru(x)
            return self.head(out)

        def step(self, x: Any, hidden: Any) -> tuple[Any, Any]:
            out, hidden = self.gru(x, hidden)
            return self.head(out), hidden

    return Network()


class SequenceGRUEstimator:
    estimator_id = "sequence_gru"

    def __init__(
        self,
        *,
        hidden_dim: int = 64,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        max_epochs: int = 300,
        patience: int = 20,
        training_device: str = "cpu",
    ) -> None:
        if isinstance(hidden_dim, bool) or not isinstance(hidden_dim, int) or not 1 <= hidden_dim <= 1024:
            raise ValueError("hidden_dim must be an integer in [1, 1024]")
        if not (isinstance(learning_rate, (int, float)) and math.isfinite(learning_rate) and learning_rate > 0):
            raise ValueError("learning_rate must be finite and positive")
        if not (isinstance(weight_decay, (int, float)) and math.isfinite(weight_decay) and weight_decay >= 0):
            raise ValueError("weight_decay must be finite and non-negative")
        if isinstance(max_epochs, bool) or not isinstance(max_epochs, int) or not 1 <= max_epochs <= 500:
            raise ValueError("max_epochs must be an integer in [1, 500]")
        if isinstance(patience, bool) or not isinstance(patience, int) or not 1 <= patience <= max_epochs:
            raise ValueError("patience must be an integer in [1, max_epochs]")
        self.hidden_dim = hidden_dim
        self.learning_rate = float(learning_rate)
        self.weight_decay = float(weight_decay)
        self.max_epochs = max_epochs
        self.patience = patience
        self.training_device = normalize_training_device(training_device)

    def _tensors(self, torch: Any, np: Any, view: TrainingView, encoder: NeuralFeatureEncoder, device: str):

        sequences = view.sequences()
        width = encoder.schema.output_width
        longest = max(len(seq) for seq in sequences)
        x = np.zeros((len(sequences), longest, width), dtype=np.float32)
        y = np.zeros((len(sequences), longest), dtype=np.float32)
        w = np.zeros((len(sequences), longest), dtype=np.float32)
        for i, seq in enumerate(sequences):
            x[i, : len(seq)] = encoder.transform(tuple(ex.point for ex in seq)).matrix
            y[i, : len(seq)] = [ex.target_value for ex in seq]
            w[i, : len(seq)] = [ex.sample_weight for ex in seq]
        return (torch.from_numpy(x).to(device), torch.from_numpy(y).to(device), torch.from_numpy(w).to(device))

    def fit(self, train: TrainingView, validation: TrainingView, context: FitContext) -> FittedSequenceGRU:
        token_unit_contract_id = validate_fit_token_unit(train, validation)
        if train.position != validation.position or train.target != validation.target:
            raise ValueError("train and validation views must share position and target")
        conditions = _view_condition_ids(train, description="train")
        if conditions != _view_condition_ids(validation, description="validation"):
            raise ValueError("train and validation views must share condition scope")
        quantiles = (context.interval_alpha / 2, 0.5, 1 - context.interval_alpha / 2)
        np, torch = _load_neural_dependencies()
        seed = _derived_seed(context.seed, context.fold)
        device = configure_deterministic_training(torch, seed=seed, device=self.training_device)
        encoder = NeuralFeatureEncoder.fit(tuple(example.point for example in train.examples))
        if encoder.schema.output_width <= 0:
            raise ValueError("sequence GRU requires at least one usable train-fold feature")
        train_x, train_y, train_w = self._tensors(torch, np, train, encoder, device)
        val_x, val_y, val_w = self._tensors(torch, np, validation, encoder, device)


        target_scale = _weighted_median(train)
        train_y, val_y = train_y / target_scale, val_y / target_scale
        model = _build_network(torch, encoder.schema.output_width, self.hidden_dim).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)

        def loss_of(x: Any, y: Any, w: Any) -> Any:
            out = model(x)
            # Exclude padded timesteps from the loss and its gradients.
            mask = w > 0
            return _weighted_quantile_loss(torch, out[mask], y[mask], w[mask], quantiles, q50_huber_delta=None)

        best_loss, best_state, best_epoch, history = math.inf, None, 0, []
        for epoch in range(1, self.max_epochs + 1):
            model.train()
            optimizer.zero_grad()
            loss = loss_of(train_x, train_y, train_w)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            model.eval()
            with torch.inference_mode():
                val_loss = float(loss_of(val_x, val_y, val_w))
            history.append(val_loss)
            if val_loss < best_loss:
                best_loss, best_epoch = val_loss, epoch
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            elif epoch - best_epoch >= self.patience:
                break
        model.load_state_dict(best_state)
        model.to("cpu").eval()
        return FittedSequenceGRU(
            estimator_id=self.estimator_id,
            target=train.target,
            position=train.position,
            dataset_id=train.dataset_id,
            input_contract_id=train.input_contract_id,
            token_unit_contract_id=token_unit_contract_id,
            allowed_condition_ids=conditions,
            encoder=encoder,
            model=model,
            target_scale=target_scale,
            fit_report={
                "estimator_id": self.estimator_id,
                "estimator_version": SEQUENCE_GRU_ESTIMATOR_VERSION,
                "hidden_dim": self.hidden_dim,
                "input_dim": encoder.schema.output_width,
                "train_trajectories": int(train_x.shape[0]),
                "train_points": len(train.examples),
                "validation_points": len(validation.examples),
                "best_epoch": best_epoch,
                "best_validation_loss": best_loss,
                "epochs_run": len(history),
                "seed": seed,
                "target_scale": target_scale,
                "quantiles": list(quantiles),
            },
        )


__all__ = ["FittedSequenceGRU", "SequenceGRUEstimator", "SequenceGRUSession"]

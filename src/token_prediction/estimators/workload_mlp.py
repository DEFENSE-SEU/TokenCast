from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

AUXILIARY_NAMES = ("attempts", "reuse", "h1", "h2", "h4", "h8", "h16")
ARM_AUXILIARIES = {
    "direct": (),
    "direct_n": ("attempts",),
    "direct_m": ("reuse",),
    "direct_nm": ("attempts", "reuse"),
    "direct_horizons": ("h1", "h2", "h4", "h8", "h16"),
}


def weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    order = np.argsort(values, kind="stable")
    return float(values[order][np.searchsorted(np.cumsum(weights[order]), weights.sum() / 2)])


@dataclass(frozen=True)
class WorkloadFitConfig:
    hidden_dims: tuple[int, int] = (128, 64)
    learning_rate: float = 0.001
    weight_decay: float = 0.0001
    auxiliary_weight: float = 1.0
    max_epochs: int = 150
    patience: int = 15
    seed: int = 20260915


class WorkloadMLP:


    def __init__(self, input_dim: int, config: WorkloadFitConfig):
        import torch

        if input_dim < 1 or min(config.hidden_dims) < 1:
            raise ValueError("network dimensions must be positive")
        if config.max_epochs < 1 or config.patience < 1:
            raise ValueError("training epoch limits must be positive")
        if config.learning_rate <= 0 or config.weight_decay < 0 or config.auxiliary_weight < 0:
            raise ValueError("invalid optimization settings")
        torch.set_num_threads(1)
        torch.manual_seed(config.seed)
        torch.use_deterministic_algorithms(True)
        self.config = config
        self.input_dim = input_dim
        self.network = torch.nn.Sequential(
            torch.nn.Linear(input_dim, config.hidden_dims[0]), torch.nn.SiLU(),
            torch.nn.Linear(config.hidden_dims[0], config.hidden_dims[1]), torch.nn.SiLU(),
            torch.nn.Linear(config.hidden_dims[1], 1 + len(AUXILIARY_NAMES)),
        )
        self.target_scale = 1.0
        self.auxiliary_mean = np.zeros(len(AUXILIARY_NAMES))
        self.auxiliary_scale = np.ones(len(AUXILIARY_NAMES))
        self.fit_report: dict[str, Any] = {}

    def fit(self, x: np.ndarray, y: np.ndarray, weights: np.ndarray,
            auxiliary: np.ndarray, *, arm: str,
            validation_x: np.ndarray, validation_y: np.ndarray,
            validation_weights: np.ndarray) -> "WorkloadMLP":
        import torch

        if arm not in ARM_AUXILIARIES:
            raise ValueError(f"unknown workload arm: {arm}")
        x, y, weights = self._validate(x, y, weights)
        vx, vy, vw = self._validate(validation_x, validation_y, validation_weights)
        auxiliary = np.asarray(auxiliary, dtype=float)
        if auxiliary.shape != (len(y), len(AUXILIARY_NAMES)):
            raise ValueError("auxiliary label matrix has the wrong shape")
        if np.isinf(auxiliary).any() or (auxiliary[np.isfinite(auxiliary)] < 0).any():
            raise ValueError("auxiliary labels must be nonnegative or missing (NaN)")
        active = [AUXILIARY_NAMES.index(name) for name in ARM_AUXILIARIES[arm]]
        self.target_scale = max(1.0, weighted_median(y, weights))
        transformed = np.zeros_like(auxiliary)
        mask = np.isfinite(auxiliary)
        for index in active:
            observed = mask[:, index] & (weights > 0)
            if not observed.any():
                raise ValueError(f"no observed training labels for {AUXILIARY_NAMES[index]}")
            values = np.log1p(auxiliary[observed, index])
            ws = weights[observed]
            mean = float(np.average(values, weights=ws))
            scale = max(1e-6, float(np.sqrt(np.average((values - mean) ** 2, weights=ws))))
            self.auxiliary_mean[index], self.auxiliary_scale[index] = mean, scale
            transformed[observed, index] = (values - mean) / scale
        def tensor(array):
            return torch.as_tensor(array, dtype=torch.float32)
        tx, ty, tw = tensor(x), tensor(y / self.target_scale), tensor(weights)
        tax, tm = tensor(transformed), tensor(mask)
        tvx, tvy, tvw = tensor(vx), tensor(vy / self.target_scale), tensor(vw)
        optimizer = torch.optim.AdamW(self.network.parameters(),
                                     lr=self.config.learning_rate,
                                     weight_decay=self.config.weight_decay)
        best, best_epoch, stale = float("inf"), 0, 0
        best_state = None
        for epoch in range(1, self.config.max_epochs + 1):
            self.network.train()
            output = self.network(tx)
            main = torch.nn.functional.softplus(output[:, 0])
            loss = ((main - ty).abs() * tw).sum() / tw.sum()
            auxiliary_losses = []
            for index in active:
                ws = tw * tm[:, index]
                errors = torch.nn.functional.smooth_l1_loss(
                    output[:, index + 1], tax[:, index], reduction="none")
                auxiliary_losses.append((errors * ws).sum() / ws.sum())
            if auxiliary_losses:
                loss = loss + self.config.auxiliary_weight * torch.stack(auxiliary_losses).mean()
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 10.0)
            optimizer.step()
            self.network.eval()
            with torch.no_grad():
                predicted = torch.nn.functional.softplus(self.network(tvx)[:, 0])
                score = float(((predicted - tvy).abs() * tvw).sum() / tvw.sum())
            if not np.isfinite(score):
                raise ValueError("non-finite validation error")
            if score < best - 1e-8:
                best, best_epoch, stale = score, epoch, 0
                best_state = {key: value.detach().clone()
                              for key, value in self.network.state_dict().items()}
            else:
                stale += 1
                if stale >= self.config.patience:
                    break
        self.network.load_state_dict(best_state)
        self.fit_report = {
            "arm": arm, "best_epoch": best_epoch, "epochs_run": epoch,
            "validation_mae": best * self.target_scale,
            "target_scale": self.target_scale,
            "parameter_count": sum(p.numel() for p in self.network.parameters()),
            "active_auxiliaries": list(ARM_AUXILIARIES[arm]),
            "auxiliary_observed": {name: int(mask[:, i].sum())
                                   for i, name in enumerate(AUXILIARY_NAMES)},
            "auxiliary_mean": self.auxiliary_mean.tolist(),
            "auxiliary_scale": self.auxiliary_scale.tolist(),
        }
        return self

    def _validate(self, x, y, weights):
        x, y, weights = (np.asarray(a, dtype=float) for a in (x, y, weights))
        if x.ndim != 2 or x.shape != (len(y), self.input_dim) or y.ndim != 1:
            raise ValueError("feature/target shape mismatch")
        if not len(y) or weights.shape != y.shape:
            raise ValueError("empty or misaligned training data")
        if not all(np.isfinite(a).all() for a in (x, y, weights)):
            raise ValueError("features, main labels, and weights must be finite")
        if (y < 0).any() or (weights < 0).any() or weights.sum() <= 0:
            raise ValueError("invalid main labels or sample weights")
        return x, y, weights

    def predict(self, causal_features: np.ndarray) -> np.ndarray:
        import torch

        x = np.asarray(causal_features, dtype=np.float32)
        if x.ndim != 2 or x.shape[1] != self.input_dim or not np.isfinite(x).all():
            raise ValueError("invalid causal prediction features")
        self.network.eval()
        with torch.no_grad():
            raw = self.network(torch.as_tensor(x))[:, 0]
            return torch.nn.functional.softplus(raw).numpy() * self.target_scale

    def save(self, path: Path) -> None:
        import torch
        from dataclasses import asdict

        torch.save({"schema_version": 1, "input_dim": self.input_dim,
                    "config": asdict(self.config), "target_scale": self.target_scale,
                    "state_dict": self.network.state_dict(), "fit_report": self.fit_report}, path)

    @classmethod
    def load(cls, path: Path) -> "WorkloadMLP":
        import torch

        state = torch.load(path, map_location="cpu", weights_only=True)
        if state["schema_version"] != 1:
            raise ValueError("unsupported workload model version")
        config = dict(state["config"])
        config["hidden_dims"] = tuple(config["hidden_dims"])
        model = cls(state["input_dim"], WorkloadFitConfig(**config))
        model.target_scale = state["target_scale"]
        model.network.load_state_dict(state["state_dict"])
        model.fit_report = state["fit_report"]
        model.auxiliary_mean = np.asarray(model.fit_report["auxiliary_mean"])
        model.auxiliary_scale = np.asarray(model.fit_report["auxiliary_scale"])
        return model

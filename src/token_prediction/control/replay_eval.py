from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Sequence

from token_prediction.estimators import TokenForecast


DECISION_INPUTS = ("point", "lower", "upper", "oracle")


@dataclass(frozen=True)
class ReplayCheckpoint:


    cutoff_event_seq: int
    spent_tokens: int
    forecast: TokenForecast | None

    def __post_init__(self) -> None:
        if self.cutoff_event_seq < 0:
            raise ValueError("checkpoint cutoff must be non-negative")
        if isinstance(self.spent_tokens, bool) or not isinstance(self.spent_tokens, int):
            raise ValueError("checkpoint spent tokens must be an integer")
        if self.spent_tokens < 0:
            raise ValueError("checkpoint spent tokens must be non-negative")


@dataclass(frozen=True)
class TrajectoryReplayRecord:


    task_id: str
    trajectory_id: str
    succeeded: bool
    total_tokens: int
    checkpoints: tuple[ReplayCheckpoint, ...] = ()
    launch_forecast: TokenForecast | None = None

    def __post_init__(self) -> None:
        for name in ("task_id", "trajectory_id"):
            if not str(getattr(self, name)).strip():
                raise ValueError(f"{name} is required")
        if isinstance(self.total_tokens, bool) or not isinstance(self.total_tokens, int):
            raise ValueError("total tokens must be an integer")
        if self.total_tokens < 0:
            raise ValueError("total tokens must be non-negative")
        cutoffs = [checkpoint.cutoff_event_seq for checkpoint in self.checkpoints]
        if cutoffs != sorted(cutoffs) or len(cutoffs) != len(set(cutoffs)):
            raise ValueError("replay checkpoints must be strictly ordered")
        spents = [checkpoint.spent_tokens for checkpoint in self.checkpoints]
        if any(late < early for early, late in zip(spents, spents[1:])):
            raise ValueError("replay spent tokens must be non-decreasing")
        if any(spent > self.total_tokens for spent in spents):
            raise ValueError("replay spent tokens cannot exceed the run total")


def _decision_estimate(
    forecast: TokenForecast | None,
    decision_input: str,
    *,
    oracle_remaining: int,
) -> float | None:
    if decision_input == "oracle":
        return float(oracle_remaining)
    if forecast is None:
        return None
    if decision_input == "point":
        return forecast.point
    if decision_input == "lower":
        return forecast.lower
    if decision_input == "upper":
        return forecast.upper
    raise ValueError(f"unknown decision input {decision_input!r}")


@dataclass(frozen=True)
class PolicyOutcome:


    policy_id: str
    budget: int
    decision_input: str
    n_runs: int
    executed_runs: int
    aborted_runs: int
    refused_runs: int
    spent_tokens: int
    successes: int
    wrongly_stopped_successes: int

    @property
    def success_rate(self) -> float:
        return self.successes / self.n_runs if self.n_runs else 0.0

    @property
    def wrongly_stopped_rate(self) -> float:
        return self.wrongly_stopped_successes / self.n_runs if self.n_runs else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "policy_id": self.policy_id,
            "budget": self.budget,
            "decision_input": self.decision_input,
            "n_runs": self.n_runs,
            "executed_runs": self.executed_runs,
            "aborted_runs": self.aborted_runs,
            "refused_runs": self.refused_runs,
            "spent_tokens": self.spent_tokens,
            "successes": self.successes,
            "success_rate": self.success_rate,
            "wrongly_stopped_successes": self.wrongly_stopped_successes,
            "wrongly_stopped_rate": self.wrongly_stopped_rate,
        }


def _validate_budget(budget: int) -> None:
    if isinstance(budget, bool) or not isinstance(budget, int) or budget <= 0:
        raise ValueError("budget must be a positive integer")


def evaluate_no_control(
    records: Sequence[TrajectoryReplayRecord],
    *,
    budget: int,
) -> PolicyOutcome:


    _validate_budget(budget)
    return PolicyOutcome(
        policy_id="no_control",
        budget=budget,
        decision_input="none",
        n_runs=len(records),
        executed_runs=len(records),
        aborted_runs=0,
        refused_runs=0,
        spent_tokens=sum(record.total_tokens for record in records),
        successes=sum(record.succeeded for record in records),
        wrongly_stopped_successes=0,
    )


def evaluate_fixed_cap(
    records: Sequence[TrajectoryReplayRecord],
    *,
    budget: int,
) -> PolicyOutcome:


    _validate_budget(budget)
    spent = 0
    successes = 0
    aborted = 0
    wrongly_stopped = 0
    for record in records:
        if record.total_tokens <= budget:
            spent += record.total_tokens
            successes += int(record.succeeded)
        else:
            spent += budget
            aborted += 1


            wrongly_stopped += int(
                record.succeeded and record.total_tokens <= budget
            )
    return PolicyOutcome(
        policy_id="fixed_cap",
        budget=budget,
        decision_input="none",
        n_runs=len(records),
        executed_runs=len(records) - aborted,
        aborted_runs=aborted,
        refused_runs=0,
        spent_tokens=spent,
        successes=successes,
        wrongly_stopped_successes=wrongly_stopped,
    )


def evaluate_oracle(
    records: Sequence[TrajectoryReplayRecord],
    *,
    budget: int,
) -> PolicyOutcome:


    _validate_budget(budget)
    executed = [
        record
        for record in records
        if record.succeeded and record.total_tokens <= budget
    ]
    return PolicyOutcome(
        policy_id="oracle",
        budget=budget,
        decision_input="oracle",
        n_runs=len(records),
        executed_runs=len(executed),
        aborted_runs=0,
        refused_runs=len(records) - len(executed),
        spent_tokens=sum(record.total_tokens for record in executed),
        successes=len(executed),
        wrongly_stopped_successes=0,
    )


def evaluate_admission_policy(
    records: Sequence[TrajectoryReplayRecord],
    *,
    budget: int,
    decision_input: str = "upper",
) -> PolicyOutcome:


    _validate_budget(budget)
    if decision_input not in DECISION_INPUTS:
        raise ValueError(f"unknown decision input {decision_input!r}")
    spent = 0
    successes = 0
    refused = 0
    wrongly_stopped = 0
    for record in records:
        estimate = _decision_estimate(
            record.launch_forecast,
            decision_input,
            oracle_remaining=record.total_tokens,
        )
        if estimate is None:
            raise ValueError(
                f"admission requires a launch forecast for {record.trajectory_id}"
            )
        if estimate > budget:
            refused += 1
            wrongly_stopped += int(
                record.succeeded and record.total_tokens <= budget
            )
            continue
        spent += record.total_tokens
        successes += int(record.succeeded)
    return PolicyOutcome(
        policy_id="admission",
        budget=budget,
        decision_input=decision_input,
        n_runs=len(records),
        executed_runs=len(records) - refused,
        aborted_runs=0,
        refused_runs=refused,
        spent_tokens=spent,
        successes=successes,
        wrongly_stopped_successes=wrongly_stopped,
    )


def evaluate_abort_policy(
    records: Sequence[TrajectoryReplayRecord],
    *,
    budget: int,
    decision_input: str = "lower",
) -> PolicyOutcome:


    _validate_budget(budget)
    if decision_input not in DECISION_INPUTS:
        raise ValueError(f"unknown decision input {decision_input!r}")
    spent = 0
    successes = 0
    aborted = 0
    wrongly_stopped = 0
    for record in records:
        stop_at: int | None = None
        for checkpoint in record.checkpoints:
            estimate = _decision_estimate(
                checkpoint.forecast,
                decision_input,
                oracle_remaining=record.total_tokens - checkpoint.spent_tokens,
            )
            if estimate is None:
                continue
            if not math.isfinite(estimate):
                raise ValueError("abort estimates must be finite")
            if checkpoint.spent_tokens + estimate > budget:
                stop_at = checkpoint.spent_tokens
                break
        if stop_at is None:
            spent += record.total_tokens
            successes += int(record.succeeded)
        else:
            # A replayed stop spends the observed prefix and receives no completion credit.
            spent += stop_at
            aborted += 1
            wrongly_stopped += int(
                record.succeeded and record.total_tokens <= budget
            )
    return PolicyOutcome(
        policy_id="abort",
        budget=budget,
        decision_input=decision_input,
        n_runs=len(records),
        executed_runs=len(records) - aborted,
        aborted_runs=aborted,
        refused_runs=0,
        spent_tokens=spent,
        successes=successes,
        wrongly_stopped_successes=wrongly_stopped,
    )


def pareto_frontier(
    records: Sequence[TrajectoryReplayRecord],
    *,
    budgets: Sequence[int],
    decision_input: str = "upper",
) -> dict[str, Any]:


    if not records:
        raise ValueError("pareto evaluation requires replay records")
    resolved = tuple(budgets)
    if (
        not resolved
        or any(
            isinstance(value, bool) or not isinstance(value, int) or value <= 0
            for value in resolved
        )
        or list(resolved) != sorted(set(resolved))
    ):
        raise ValueError("budgets must be unique increasing positive integers")
    has_launch = all(record.launch_forecast is not None for record in records)
    has_checkpoints = any(record.checkpoints for record in records)
    rows: list[dict[str, Any]] = []
    for budget in resolved:
        rows.append(evaluate_no_control(records, budget=budget).to_dict())
        rows.append(evaluate_fixed_cap(records, budget=budget).to_dict())
        rows.append(evaluate_oracle(records, budget=budget).to_dict())
        if has_launch:
            rows.append(
                evaluate_admission_policy(
                    records, budget=budget, decision_input=decision_input
                ).to_dict()
            )
        if has_checkpoints:
            rows.append(
                evaluate_abort_policy(
                    records, budget=budget, decision_input=decision_input
                ).to_dict()
            )
    return {
        "policy_suite_id": "prefix-preserving-controls-v1",
        "decision_input": decision_input,
        "budgets": list(resolved),
        "n_runs": len(records),
        "rows": rows,
    }


__all__ = [
    "DECISION_INPUTS",
    "PolicyOutcome",
    "ReplayCheckpoint",
    "TrajectoryReplayRecord",
    "evaluate_abort_policy",
    "evaluate_admission_policy",
    "evaluate_fixed_cap",
    "evaluate_no_control",
    "evaluate_oracle",
    "pareto_frontier",
]

from __future__ import annotations

from decimal import Decimal

from .reversals import BalanceOperation


def calculate_balances(operations: tuple[BalanceOperation, ...]) -> dict[str, Decimal]:
    balances: dict[str, float] = {}
    for operation in operations:
        balances[operation.sku] = balances.get(operation.sku, 0.0) + float(operation.delta)
    return {sku: Decimal(str(quantity)) for sku, quantity in balances.items()}

from __future__ import annotations

from decimal import Decimal, localcontext

from .reversals import BalanceOperation


def calculate_balances(operations: tuple[BalanceOperation, ...]) -> dict[str, Decimal]:
    values = [operation.delta for operation in operations]
    lowest_exponent = min((value.as_tuple().exponent for value in values), default=0)
    greatest_adjusted = max((value.adjusted() for value in values if value), default=0)
    integer_digits = max(1, greatest_adjusted + 1)
    fractional_digits = max(0, -lowest_exponent)
    carry_digits = len(str(max(1, len(values))))
    precision = max(28, integer_digits + fractional_digits + carry_digits + 2)
    with localcontext() as context:
        context.prec = precision
        balances: dict[str, Decimal] = {}
        for operation in operations:
            balances[operation.sku] = balances.get(operation.sku, Decimal(0)) + operation.delta
    return balances

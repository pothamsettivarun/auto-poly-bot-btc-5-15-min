from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List


@dataclass
class RiskConfig:
    max_position_per_market: float = 400.0
    max_gross_exposure: float = 1500.0
    max_daily_drawdown: float = -500.0


def gross_exposure(positions: List[Dict]) -> float:
    total = 0.0
    for p in positions or []:
        if p.get("status") == "active":
            total += float(p.get("current_value", 0.0) or 0.0)
    return total


def market_exposure(positions: List[Dict], market_id: str) -> float:
    for p in positions or []:
        if p.get("status") == "active" and p.get("market_id") == market_id:
            return float(p.get("current_value", 0.0) or 0.0)
    return 0.0


def daily_drawdown(equity: float, day_start_equity: float) -> float:
    return float(equity) - float(day_start_equity)


def risk_check(
    *,
    config: RiskConfig,
    positions: List[Dict],
    market_id: str,
    current_equity: float,
    day_start_equity: float,
    new_order_notional: float,
) -> tuple[bool, str]:
    dd = daily_drawdown(current_equity, day_start_equity)
    if dd <= config.max_daily_drawdown:
        return False, f"kill_switch_drawdown:{dd:.2f}"

    g = gross_exposure(positions)
    if (g + float(new_order_notional)) > config.max_gross_exposure:
        return False, f"gross_exposure_limit:{g:.2f}+{new_order_notional:.2f}"

    m = market_exposure(positions, market_id)
    if (m + float(new_order_notional)) > config.max_position_per_market:
        return False, f"market_exposure_limit:{m:.2f}+{new_order_notional:.2f}"

    return True, "ok"
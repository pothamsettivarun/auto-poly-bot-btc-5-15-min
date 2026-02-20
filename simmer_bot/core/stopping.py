from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass
class ExecuteDecision:
    execute: bool
    reason: str
    alpha_ok: bool
    guarantee_ok: bool
    risk_ok: bool
    d: float
    g: float
    guarantee: float
    net_threshold: float
    alpha: float


def best_iter_by_guarantee(candidates: List[Dict[str, float]]) -> Optional[Dict[str, float]]:
    """
    Forced interruption fallback: choose iterate maximizing (D-g).
    Each candidate should contain keys: t, D, g (or divergence, gap).
    """
    if not candidates:
        return None

    best = None
    best_val = float("-inf")
    for c in candidates:
        d = float(c.get("D", c.get("divergence", 0.0)))
        g = float(c.get("g", c.get("gap", 0.0)))
        val = d - g
        if val > best_val:
            best_val = val
            best = dict(c)
            best["guarantee"] = val
    return best


def should_execute(
    *,
    D_t: float,
    g_t: float,
    fees_est: float,
    slippage_est: float,
    risk_buffer: float,
    risk_ok: bool,
    alpha: float = 0.9,
    min_net_threshold: float = 0.0,
) -> ExecuteDecision:
    """
    Proposition-4.1 driven decision gate.

    Rules:
      1) alpha extraction: g_t <= (1-alpha) D_t
      2) guarantee net of execution costs >= min threshold
      3) risk checks pass

    guarantee_raw = D_t - g_t
    guarantee_net = guarantee_raw - (fees + slippage + risk_buffer)
    """
    D_t = float(D_t)
    g_t = float(g_t)
    fees_est = float(fees_est)
    slippage_est = float(slippage_est)
    risk_buffer = float(risk_buffer)

    guarantee_raw = D_t - g_t
    guarantee_net = guarantee_raw - (fees_est + slippage_est + risk_buffer)

    alpha_ok = (D_t > 0.0) and (g_t <= (1.0 - float(alpha)) * D_t)
    guarantee_ok = guarantee_net >= float(min_net_threshold)

    execute = bool(alpha_ok and guarantee_ok and risk_ok)

    if not risk_ok:
        reason = "risk_blocked"
    elif D_t <= 0:
        reason = "non_positive_D"
    elif not alpha_ok:
        reason = "alpha_not_met"
    elif not guarantee_ok:
        reason = "net_guarantee_below_threshold"
    else:
        reason = "execute"

    return ExecuteDecision(
        execute=execute,
        reason=reason,
        alpha_ok=alpha_ok,
        guarantee_ok=guarantee_ok,
        risk_ok=bool(risk_ok),
        d=D_t,
        g=g_t,
        guarantee=guarantee_net,
        net_threshold=float(min_net_threshold),
        alpha=float(alpha),
    )
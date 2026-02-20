from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, Optional

from core.constraints import ConstraintModel


@dataclass
class OracleResult:
    feasible: bool
    z: Optional[Dict[int, int]]
    status: str
    latency_ms: int
    retries: int = 0


class IPOOracle:
    """
    Thin feasibility/LMO oracle with retry/timeout semantics.

    Phase-C default implementation uses ConstraintModel methods directly.
    Replace _solve_once / _lmo_once with real MIP backend when wiring Gurobi.
    """

    def __init__(self, constraints: ConstraintModel, timeout_ms: int = 300, max_retries: int = 1):
        self.constraints = constraints
        self.timeout_ms = int(timeout_ms)
        self.max_retries = int(max_retries)

    def _solve_once(self, i: int, value: int, sigma_hat: Dict[int, int], warm_start: Optional[Dict[int, int]] = None) -> OracleResult:
        t0 = time.time()

        if self.timeout_ms <= 0:
            return OracleResult(False, None, "timeout", 0)

        try:
            ok = self.constraints.is_feasible_assignment(i, value, sigma_hat)
        except Exception:
            latency_ms = int((time.time() - t0) * 1000)
            return OracleResult(False, None, "error", latency_ms)

        latency_ms = int((time.time() - t0) * 1000)
        if latency_ms > self.timeout_ms:
            return OracleResult(False, None, "timeout", latency_ms)

        if ok:
            try:
                z = self.constraints.build_vertex(sigma_hat, explicit={i: value})
            except Exception:
                return OracleResult(False, None, "error", latency_ms)
            return OracleResult(True, z, "feasible", latency_ms)

        return OracleResult(False, None, "infeasible", latency_ms)

    def solve_feasible(self, i: int, value: int, sigma_hat: Dict[int, int], warm_start: Optional[Dict[int, int]] = None) -> OracleResult:
        retries = 0
        best: Optional[OracleResult] = None

        for _ in range(self.max_retries + 1):
            r = self._solve_once(i, value, sigma_hat, warm_start=warm_start)
            best = r
            if r.status in ("feasible", "infeasible"):
                r.retries = retries
                return r
            retries += 1

        assert best is not None
        best.retries = retries
        return best

    def solve_linear_objective(self, gradient: Dict[int, float], sigma_hat: Dict[int, int]) -> OracleResult:
        t0 = time.time()
        if self.timeout_ms <= 0:
            return OracleResult(False, None, "timeout", 0)
        try:
            z = self.constraints.build_descent_vertex(sigma_hat=sigma_hat, gradient=gradient)
            latency_ms = int((time.time() - t0) * 1000)
            return OracleResult(True, z, "feasible", latency_ms)
        except Exception:
            latency_ms = int((time.time() - t0) * 1000)
            return OracleResult(False, None, "error", latency_ms)
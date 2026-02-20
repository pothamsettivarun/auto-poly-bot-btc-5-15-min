from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, List, Tuple

from core.constraints import ConstraintModel
from core.ip_oracle import IPOOracle


@dataclass
class InitFWCert:
    i: int
    feasible0: bool
    feasible1: bool
    status0: str
    status1: str
    latency0_ms: int
    latency1_ms: int


def _vertex_key(z: Dict[int, int]) -> Tuple[Tuple[int, int], ...]:
    return tuple(sorted((int(k), int(v)) for k, v in z.items()))


def _avg_vertex(Z0: List[Dict[int, int]]) -> Dict[int, float]:
    if not Z0:
        return {}
    keys = sorted(Z0[0].keys())
    n = float(len(Z0))
    out: Dict[int, float] = {}
    for k in keys:
        out[k] = sum(float(z.get(k, 0.0)) for z in Z0) / n
    return out


def init_fw(
    sigma: Dict[int, int],
    constraints: ConstraintModel,
    oracle: IPOOracle,
    *,
    fail_on_empty_z0: bool = True,
    enforce_strict_interior: bool = True,
) -> tuple[Dict[int, int], List[Dict[int, int]], Dict[int, float], List[InitFWCert], Dict[str, int]]:
    """
    Algorithm 3-style InitFW.

    Returns:
      sigma_hat, Z0, u, certs, metrics

    Strict production-like guards:
      - fail_on_empty_z0=True: raise if no feasible seed vertices collected
      - enforce_strict_interior=True: raise if unresolved u_i not in (0,1)
    """
    t0 = time.time()
    sigma_hat: Dict[int, int] = dict(sigma)
    Z0_map: Dict[Tuple[Tuple[int, int], ...], Dict[int, int]] = {}
    certs: List[InitFWCert] = []

    unresolved = constraints.unresolved_indices(sigma_hat)

    for i in unresolved:
        r1 = oracle.solve_feasible(i, 1, sigma_hat)
        r0 = oracle.solve_feasible(i, 0, sigma_hat)

        certs.append(
            InitFWCert(
                i=i,
                feasible0=r0.feasible,
                feasible1=r1.feasible,
                status0=r0.status,
                status1=r1.status,
                latency0_ms=r0.latency_ms,
                latency1_ms=r1.latency_ms,
            )
        )

        if r1.feasible and r1.z is not None:
            Z0_map[_vertex_key(r1.z)] = r1.z
        if r0.feasible and r0.z is not None:
            Z0_map[_vertex_key(r0.z)] = r0.z

        # forced settlement extension
        if r1.feasible and not r0.feasible:
            sigma_hat[i] = 1
        elif r0.feasible and not r1.feasible:
            sigma_hat[i] = 0

    Z0 = list(Z0_map.values())

    if not Z0:
        if fail_on_empty_z0:
            raise RuntimeError("InitFW failed: Z0 is empty")
        # fallback mode for research/debug only
        z = constraints.build_vertex(sigma_hat)
        Z0 = [z]

    u = _avg_vertex(Z0)

    unresolved_after = constraints.unresolved_indices(sigma_hat)
    interior_violations = 0
    for i in unresolved_after:
        ui = float(u.get(i, 0.0))
        if not (0.0 < ui < 1.0):
            interior_violations += 1

    if enforce_strict_interior and interior_violations > 0:
        raise RuntimeError(f"InitFW failed interior check: unresolved coords with u_i not in (0,1): {interior_violations}")

    forced_count = sum(1 for k in sigma_hat if k not in sigma)
    metrics = {
        "runtime_ms": int((time.time() - t0) * 1000),
        "probed_count": len(unresolved),
        "z0_size": len(Z0),
        "forced_settlements": forced_count,
        "interior_violations": interior_violations,
    }

    return sigma_hat, Z0, u, certs, metrics
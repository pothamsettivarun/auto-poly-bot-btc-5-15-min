from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple
import math

from core.ip_oracle import IPOOracle


@dataclass
class FWIter:
    t: int
    epsilon: float
    gap: float
    gap_u: float
    divergence: float
    guarantee: float
    gamma: float


def _clip_prob(x: float, floor: float) -> float:
    return min(1.0 - floor, max(floor, float(x)))


def _dot(a: List[float], b: List[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def _v_add(a: List[float], b: List[float], wa: float = 1.0, wb: float = 1.0) -> List[float]:
    return [wa * x + wb * y for x, y in zip(a, b)]


def _to_vec(keys: List[int], d: Dict[int, float | int]) -> List[float]:
    return [float(d.get(k, 0.0)) for k in keys]


def _to_dict(keys: List[int], v: List[float]) -> Dict[int, float]:
    return {k: float(x) for k, x in zip(keys, v)}


def kl_div(mu: List[float], q: List[float], floor: float) -> float:
    s = 0.0
    for m, qq in zip(mu, q):
        m = _clip_prob(m, floor)
        qq = _clip_prob(qq, floor)
        s += m * math.log(m / qq)
    return s


def grad_kl(mu: List[float], q: List[float], floor: float) -> List[float]:
    g = []
    for m, qq in zip(mu, q):
        m = _clip_prob(m, floor)
        qq = _clip_prob(qq, floor)
        g.append(math.log(m / qq) + 1.0)
    return g


def run_barrier_fw(
    target_q: Dict[int, float],
    sigma_hat: Dict[int, int],
    Z0: List[Dict[int, int]],
    u: Dict[int, float],
    oracle: IPOOracle,
    *,
    alpha: float = 0.9,
    max_iters: int = 100,
    epsilon0: float = 0.1,
    epsilon_floor: float = 1e-4,
    prob_floor: float = 1e-9,
) -> Tuple[Dict[str, object], List[FWIter]]:
    """
    Barrier Frank-Wolfe skeleton with adaptive contraction.

    Objective proxy: minimize KL(mu || q) over contracted polytope.
    Tracks best iterate by max (D-g) where D=KL(mu||q), g=FW gap.
    """
    if not Z0:
        raise RuntimeError("run_barrier_fw: Z0 is empty")

    keys = sorted(target_q.keys())
    q = _to_vec(keys, target_q)
    u_vec = _to_vec(keys, u)

    # initialize at interior point
    mu = list(u_vec)
    eps = float(epsilon0)

    best = {
        "iter": 0,
        "mu": _to_dict(keys, mu),
        "divergence": kl_div(mu, q, prob_floor),
        "gap": float("inf"),
        "guarantee": float("-inf"),
        "epsilon": eps,
    }

    history: List[FWIter] = []

    for t in range(1, max_iters + 1):
        # hard guard against boundary blowups
        mu = [_clip_prob(x, prob_floor) for x in mu]
        if any((not math.isfinite(x)) for x in mu):
            raise RuntimeError("BarrierFW abort: non-finite mu")

        grad = grad_kl(mu, q, prob_floor)
        if any((not math.isfinite(x)) for x in grad):
            raise RuntimeError("BarrierFW abort: non-finite gradient")

        grad_dict = _to_dict(keys, grad)
        r = oracle.solve_linear_objective(gradient=grad_dict, sigma_hat=sigma_hat)
        if not r.feasible or r.z is None:
            raise RuntimeError("BarrierFW abort: LMO failed")

        s_raw = _to_vec(keys, r.z)
        # contracted vertex
        s = _v_add(s_raw, u_vec, wa=(1.0 - eps), wb=eps)

        # FW gap and interior gap
        gap = _dot(_v_add(mu, s, wa=1.0, wb=-1.0), grad)
        gap_u = _dot(_v_add(u_vec, mu, wa=1.0, wb=-1.0), grad)

        # adaptive epsilon update (paper-style)
        if gap_u < 0:
            ratio = gap / (-4.0 * gap_u)
            if ratio < eps:
                eps = max(epsilon_floor, min(ratio, eps / 2.0))

        D = kl_div(mu, q, prob_floor)
        guarantee = D - gap

        # best iterate by max(D-g)
        if guarantee > float(best.get("guarantee", float("-inf"))):
            best = {
                "iter": t,
                "mu": _to_dict(keys, mu),
                "divergence": D,
                "gap": gap,
                "guarantee": guarantee,
                "epsilon": eps,
            }

        # stopping rule alpha-extraction
        if D > 0 and gap <= (1.0 - alpha) * D:
            history.append(FWIter(t=t, epsilon=eps, gap=gap, gap_u=gap_u, divergence=D, guarantee=guarantee, gamma=0.0))
            break

        # step size (simple diminishing)
        gamma = 2.0 / (t + 2.0)
        mu = _v_add(mu, s, wa=(1.0 - gamma), wb=gamma)

        history.append(FWIter(t=t, epsilon=eps, gap=gap, gap_u=gap_u, divergence=D, guarantee=guarantee, gamma=gamma))

    return best, history
from __future__ import annotations

import time
import uuid
from dataclasses import dataclass
from typing import Callable, Dict, Optional


@dataclass
class ExecutionResult:
    status: str  # submitted|filled|partial|rejected|timeout|error
    client_order_id: str
    provider_order_id: Optional[str]
    requested_amount: float
    filled_amount: float
    remaining_amount: float
    avg_price: Optional[float]
    raw: Dict


class IdempotencyStore:
    """Simple in-memory idempotency map for loop-runtime safety."""

    def __init__(self):
        self._store: Dict[str, ExecutionResult] = {}

    def get(self, key: str) -> Optional[ExecutionResult]:
        return self._store.get(key)

    def put(self, key: str, value: ExecutionResult) -> None:
        self._store[key] = value


class ExecutionAdapter:
    """
    Minimal strict execution wrapper.

    submit_fn: callable(headers, body, timeout_sec) -> provider response dict
    cancel_fn: callable(headers, provider_order_id) -> dict (optional)
    replace_fn: callable(headers, provider_order_id, new_body, timeout_sec) -> dict (optional)
    """

    def __init__(
        self,
        submit_fn: Callable[[Dict, Dict, int], Dict],
        cancel_fn: Optional[Callable[[Dict, str], Dict]] = None,
        replace_fn: Optional[Callable[[Dict, str, Dict, int], Dict]] = None,
        store: Optional[IdempotencyStore] = None,
    ):
        self.submit_fn = submit_fn
        self.cancel_fn = cancel_fn
        self.replace_fn = replace_fn
        self.store = store or IdempotencyStore()

    def submit_idempotent(
        self,
        *,
        idem_key: str,
        headers: Dict,
        body: Dict,
        timeout_sec: int = 8,
    ) -> ExecutionResult:
        cached = self.store.get(idem_key)
        if cached is not None:
            return cached

        client_order_id = body.get("client_order_id") or f"oc-{uuid.uuid4().hex[:16]}"
        body = dict(body)
        body["client_order_id"] = client_order_id

        requested = float(body.get("amount", 0.0) or 0.0)

        t0 = time.time()
        try:
            raw = self.submit_fn(headers, body, timeout_sec)
        except TimeoutError:
            res = ExecutionResult(
                status="timeout",
                client_order_id=client_order_id,
                provider_order_id=None,
                requested_amount=requested,
                filled_amount=0.0,
                remaining_amount=requested,
                avg_price=None,
                raw={"error": "timeout"},
            )
            self.store.put(idem_key, res)
            return res
        except Exception as e:
            res = ExecutionResult(
                status="error",
                client_order_id=client_order_id,
                provider_order_id=None,
                requested_amount=requested,
                filled_amount=0.0,
                remaining_amount=requested,
                avg_price=None,
                raw={"error": str(e)},
            )
            self.store.put(idem_key, res)
            return res

        _lat_ms = int((time.time() - t0) * 1000)
        success = bool(raw.get("success"))
        provider_order_id = raw.get("order_id") or raw.get("trade_id")

        if not success:
            res = ExecutionResult(
                status="rejected",
                client_order_id=client_order_id,
                provider_order_id=provider_order_id,
                requested_amount=requested,
                filled_amount=0.0,
                remaining_amount=requested,
                avg_price=None,
                raw=raw,
            )
            self.store.put(idem_key, res)
            return res

        # Prefer explicit provider fill fields. Avoid guessing from `cost` because semantics vary.
        filled = None
        for k in ("filled_amount", "amount_filled", "filled_notional", "executed_amount"):
            if raw.get(k) is not None:
                try:
                    filled = float(raw.get(k) or 0.0)
                    break
                except Exception:
                    pass

        # If provider gives no explicit fill, treat as submitted/acknowledged (not assumed filled).
        if filled is None:
            status = "submitted"
            filled = 0.0
            remaining = requested
        else:
            filled = max(0.0, min(requested, filled))
            remaining = max(0.0, requested - filled)
            if remaining > 1e-9 and filled > 0:
                status = "partial"
            elif filled > 0:
                status = "filled"
            else:
                status = "submitted"

        res = ExecutionResult(
            status=status,
            client_order_id=client_order_id,
            provider_order_id=provider_order_id,
            requested_amount=requested,
            filled_amount=filled,
            remaining_amount=remaining,
            avg_price=float(raw.get("price") or raw.get("new_price") or 0.0) or None,
            raw=raw,
        )
        self.store.put(idem_key, res)
        return res


def reconcile_fill(position_before: float, side: str, result: ExecutionResult) -> float:
    """
    Conservative reconciliation policy:
      - only filled_amount changes inventory
      - partial leaves residual intent unfilled (no implicit retry)
    """
    delta = float(result.filled_amount)
    if side.lower() == "yes":
        return position_before + delta
    if side.lower() == "no":
        return position_before - delta
    return position_before